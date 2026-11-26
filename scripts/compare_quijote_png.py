"""Compare miniflims PNG simulations with one Quijote-PNG pair.

Quijote-PNG (Coulton et al. 2023, arXiv:2206.01619): 1 Gpc/h, 512^3
particles, 2LPT ICs at z = 127, Gadget-3, f_NL = +-100 per template; the "p"
and "m" boxes of realization i share the same seed, so their difference
isolates the primordial signal. Data are on Globus only (see README).

    # 1. our side (Quijote cosmology and box, paired +-f_NL, several seeds)
    .venv/bin/python scripts/compare_quijote_png.py generate [--n 128] [--seeds 4]

    # 2. once one Quijote pair is downloaded (density cubes or snapshots)
    .venv/bin/python scripts/compare_quijote_png.py compare --set LC \
        --plus  quijote/LC_p_0_df_m_256_CIC_z=0.npy \
        --minus quijote/LC_m_0_df_m_256_CIC_z=0.npy

    # without Quijote files, `compare` plots our side alone
    .venv/bin/python scripts/compare_quijote_png.py compare

A Quijote input may be a density cube (.npy; delta, 1 + delta or counts:
normalised to 1 + delta) or a snapshot (an HDF5 file, or a snapdir with
snap_*.hdf5 parts; needs h5py and hdf5plugin), which is painted with CIC.
All statistics use the CIC-compensated FFT estimators of jaxpng.measure.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from miniflims import load_config  # noqa: E402
from miniflims.data import build_response  # noqa: E402
from miniflims.model import FieldModel  # noqa: E402

BOX = 1000.0  # Mpc/h
QUIJOTE_COSMOLOGY = {"h": 0.6711, "omega_m": 0.3175, "omega_b": 0.049, "n_s": 0.9624, "sigma8": 0.834}
F_NL = 100.0
SETS = {"LC": "local", "EQ": "equilateral"}  # Quijote set prefix -> jaxpng template
GRAVITY = {"zeldovich": {"model": "zeldovich"},
           "tcola": {"model": "tcola", "cola": {"n_steps": 10, "z_start": 19.0, "force_sampling": 2}}}
K_EQUIL = np.arange(0.02, 0.301, 0.02)  # h/Mpc
K1, K2 = 0.1, 0.2  # h/Mpc, for the angle dependence
THETAS = np.linspace(0.1, 0.95, 10) * np.pi
OUT = Path("comparisons/quijote_png")


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def _jaxpng():
    from miniflims.png import import_jaxpng

    jp = import_jaxpng()
    jax.config.update("jax_enable_x64", True)
    return jp


def statistics(delta: np.ndarray, box: float = BOX) -> dict:
    """P(k), equilateral B(k,k,k) and B(K1, K2, theta) of a CIC density contrast."""
    jp = _jaxpng()
    d = jnp.asarray(delta, jnp.float64)
    k, pk = jp.measure.power_spectrum(d, box, MAS="CIC")
    b_eq = [float(jp.measure.bispectrum(d, box, kk, kk, jp.measure.EQUILATERAL_THETA, MAS="CIC")["B"][0])
            for kk in K_EQUIL]
    b_th = jp.measure.bispectrum(d, box, K1, K2, jnp.asarray(THETAS), MAS="CIC")
    return {"k": np.asarray(k), "pk": np.asarray(pk), "b_eq": np.asarray(b_eq),
            "b_theta": np.asarray(b_th["B"]), "k3_theta": np.asarray(b_th["k"][2:])}


# --------------------------------------------------------------------------- #
# our side
# --------------------------------------------------------------------------- #
def _model(n, template, gravity):
    over = {"grid": {"n_grid": n, "box_size": BOX}, "cosmology": QUIJOTE_COSMOLOGY,
            "forward": {**GRAVITY[gravity], "redshift": 0.0},
            "png": {"enabled": True, "template": template}}
    cfg = load_config(overrides=over)
    return FieldModel(cfg, build_response(cfg))


def generate(n: int, seeds: int, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for seed in range(seeds):
        x = jnp.asarray(np.random.default_rng(1000 + seed).standard_normal((n,) * 3))
        for qset, template in SETS.items():
            ics = None
            for gravity in GRAVITY:
                path = out / f"miniflims_{qset}_{gravity}_n{n}_seed{seed}.npz"
                if path.exists():
                    continue
                t0 = time.time()
                model = _model(n, template, gravity)
                if ics is None:  # the same ICs for both gravity models
                    dg = model.gaussian_density_k(x)
                    ng = jax.jit(lambda d: model.png.ng_density_k(d, jnp.asarray([1.0])))(dg)
                    ics = {s: jnp.fft.irfftn(dg + s * F_NL * ng, s=x.shape) for s in (+1, -1)}
                fwd = jax.jit(model.forward)
                stats = {}
                for sign, label in ((+1, "p"), (-1, "m")):
                    rho = np.asarray(fwd(ics[sign]))
                    for key, val in statistics(rho - 1.0).items():
                        stats[f"{label}/{key}"] = val
                np.savez(path, **stats, meta=np.array(json.dumps(
                    {"set": qset, "template": template, "gravity": gravity, "n": n, "seed": seed,
                     "f_NL": F_NL, "box": BOX, "cosmology": QUIJOTE_COSMOLOGY})))
                print(f"{path.name}: {time.time() - t0:.0f}s", flush=True)


# --------------------------------------------------------------------------- #
# Quijote side
# --------------------------------------------------------------------------- #
def load_quijote(path: str, n_paint: int = 256) -> np.ndarray:
    """1 + delta from a density cube (native grid) or a snapshot (painted with
    CIC on n_paint^3). Statistics are CIC-compensated on whatever grid the
    field has, and their k shells (multiples of k_F) line up with ours."""
    p = Path(path)
    if p.suffix == ".npy":
        cube = np.load(p).astype(np.float64)
        mean = cube.mean()
        return cube + 1.0 if abs(mean) < 1e-3 else cube / mean  # delta, or 1 + delta / counts
    return _paint_snapshot(p, n_paint)


def _paint_snapshot(p: Path, n: int) -> np.ndarray:
    import h5py
    import hdf5plugin  # noqa: F401  (Blosc filter used by Quijote)

    files = sorted(p.glob("*.hdf5")) if p.is_dir() else [p]
    if not files:
        raise FileNotFoundError(f"no HDF5 snapshot files in {p}")
    rho = np.zeros(n**3)
    total = 0
    for f in files:
        with h5py.File(f, "r") as h:
            box = float(h["Header"].attrs["BoxSize"])
            box = box / 1e3 if box > 1e4 else box  # kpc/h -> Mpc/h
            if abs(box - BOX) > 1e-6:
                raise ValueError(f"{f}: box {box} Mpc/h, expected {BOX}")
            pos = h["PartType1/Coordinates"]
            scale = n / (float(h["Header"].attrs["BoxSize"]))
            for start in range(0, pos.shape[0], 2**22):
                q = np.asarray(pos[start:start + 2**22], np.float64) * scale
                total += q.shape[0]
                i0 = np.floor(q).astype(np.int64)
                fr = q - i0
                for dx in (0, 1):
                    wx = fr[:, 0] if dx else 1 - fr[:, 0]
                    for dy in (0, 1):
                        wy = fr[:, 1] if dy else 1 - fr[:, 1]
                        for dz in (0, 1):
                            wz = fr[:, 2] if dz else 1 - fr[:, 2]
                            idx = ((i0[:, 0] + dx) % n * n + (i0[:, 1] + dy) % n) * n + (i0[:, 2] + dz) % n
                            rho += np.bincount(idx, weights=wx * wy * wz, minlength=n**3)
    return (rho * n**3 / total).reshape(n, n, n)


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def compare(qset: str, plus, minus, n: int, out: Path):
    from miniflims.cosmology import Cosmology
    from miniflims.plotting import AXIS, GRID, INK, INK2, SERIES, _mpl, logx
    from miniflims.png import poisson_kernel
    from miniflims.power import linear_power

    plt, _, _ = _mpl()
    dash = (0, (4, 3))
    runs = {}
    for f in sorted(out.glob(f"miniflims_{qset}_*_n{n}_seed*.npz")):
        with np.load(f) as z:
            meta = json.loads(str(z["meta"]))
            runs.setdefault(meta["gravity"], []).append({k: z[k] for k in z.files if k != "meta"})
    if not runs:
        raise SystemExit(f"no miniflims runs for {qset} at n={n} in {out}: run `generate` first")
    quijote = None
    if plus and minus:
        quijote = {}
        for label, path in (("p", plus), ("m", minus)):
            for key, val in statistics(load_quijote(path) - 1.0).items():
                quijote[f"{label}/{key}"] = val
        nk = len(next(iter(runs.values()))[0]["p/k"])
        for key in [k_ for k_ in quijote if k_.endswith("/pk") or k_.endswith("/k")]:
            quijote[key] = quijote[key][:nk]  # same k_F shells; ours stop at our Nyquist

    cosmo = Cosmology.from_config(load_config(overrides={"cosmology": QUIJOTE_COSMOLOGY}).cosmology)

    def tree(k1, k2, k3):
        """Primordial tree-level B_delta = M1 M2 M3 B_Phi per unit f_NL (z = 0, linear)."""
        from miniflims.png import import_jaxpng  # noqa: F401
        ks = [jnp.asarray(k) for k in (k1, k2, k3)]
        P = [linear_power(k, cosmo) for k in ks]
        M = [poisson_kernel(k, cosmo) for k in ks]
        Pphi = [p / m**2 for p, m in zip(P, M)]
        jp = _jaxpng()
        b_phi = jp.bispectra.template_bispectrum(SETS[qset], *Pphi)
        return np.asarray(M[0] * M[1] * M[2] * b_phi)

    def mean_sd(arrs):
        a = np.asarray(arrs)
        return a.mean(0), (a.std(0, ddof=1) if len(a) > 1 else np.zeros_like(a[0]))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.6))
    k = next(iter(runs.values()))[0]["p/k"]
    sel = (k > 0.01) & (k < 0.35)
    colors = {"zeldovich": SERIES[0], "tcola": SERIES[1], "quijote": INK}
    names = {"zeldovich": "miniflims Zel'dovich", "tcola": "miniflims tCOLA"}

    ax = axes[0, 0]
    for g, rs in runs.items():
        m, sd = mean_sd([0.5 * (r["p/pk"] + r["m/pk"]) for r in rs])
        ax.plot(k[sel], m[sel], color=colors[g], label=f"{names[g]} ({len(rs)} pairs)")
    if quijote is not None:
        ax.plot(k[sel], 0.5 * (quijote["p/pk"] + quijote["m/pk"])[sel], color=INK, lw=1.4, ls=dash,
                label="Quijote-PNG (1 pair)")
    ax.plot(k[sel], np.asarray(linear_power(jnp.asarray(k[sel]), cosmo)), color=AXIS, lw=1.2, label="linear (EH)")
    logx(ax)
    ax.set(yscale="log", xlabel="k  [h/Mpc]", ylabel="P(k)  [(Mpc/h)³]", title="Matter power spectrum, z = 0")
    ax.legend(loc="lower left")

    ax = axes[0, 1]
    for g, rs in runs.items():
        m, sd = mean_sd([(r["p/pk"] - r["m/pk"]) / (r["p/pk"] + r["m/pk"]) / F_NL for r in rs])
        ax.fill_between(k[sel], (m - sd)[sel], (m + sd)[sel], color=colors[g], alpha=0.2, lw=0)
        ax.plot(k[sel], m[sel], color=colors[g], label=names[g])
    if quijote is not None:
        q = (quijote["p/pk"] - quijote["m/pk"]) / (quijote["p/pk"] + quijote["m/pk"]) / F_NL
        ax.plot(k[sel], q[sel], color=INK, lw=1.4, ls=dash, label="Quijote-PNG")
    ax.axhline(0, color=AXIS, lw=1.0)
    logx(ax)
    ax.set(xlabel="k  [h/Mpc]", ylabel="(P₊ − P₋) / (P₊ + P₋) / f_NL",
           title=f"Power-spectrum response to {SETS[qset]} f_NL (paired)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.legend(loc="best")

    ax = axes[1, 0]
    for g, rs in runs.items():
        m, sd = mean_sd([(r["p/b_eq"] - r["m/b_eq"]) / (2 * F_NL) for r in rs])
        ax.fill_between(K_EQUIL, m - sd, m + sd, color=colors[g], alpha=0.2, lw=0)
        ax.plot(K_EQUIL, m, "o-", ms=5, color=colors[g], label=names[g])
    if quijote is not None:
        ax.plot(K_EQUIL, (quijote["p/b_eq"] - quijote["m/b_eq"]) / (2 * F_NL), "s", ms=6, color=INK,
                label="Quijote-PNG")
    ax.plot(K_EQUIL, tree(K_EQUIL, K_EQUIL, K_EQUIL), color=INK2, lw=1.2, ls=dash, label="primordial tree level")
    ax.set(yscale="symlog", xlabel="k  [h/Mpc]", ylabel="ΔB(k,k,k) per unit f_NL  [(Mpc/h)⁶]",
           title="Equilateral bispectrum response, (B₊ − B₋) / 2 f_NL")
    ax.set_yscale("symlog", linthresh=10.0)
    ax.legend(loc="best")

    ax = axes[1, 1]
    r0 = next(iter(runs.values()))[0]
    k3 = r0["p/k3_theta"]
    for g, rs in runs.items():
        m, sd = mean_sd([(r["p/b_theta"] - r["m/b_theta"]) / (2 * F_NL) for r in rs])
        ax.fill_between(THETAS / np.pi, m - sd, m + sd, color=colors[g], alpha=0.2, lw=0)
        ax.plot(THETAS / np.pi, m, "o-", ms=5, color=colors[g], label=names[g])
    if quijote is not None:
        ax.plot(THETAS / np.pi, (quijote["p/b_theta"] - quijote["m/b_theta"]) / (2 * F_NL), "s", ms=6,
                color=INK, label="Quijote-PNG")
    ax.plot(THETAS / np.pi, tree(np.full_like(k3, K1), np.full_like(k3, K2), k3), color=INK2, lw=1.2, ls=dash,
            label="primordial tree level")
    ax.set(xlabel="θ₁₂ / π   (θ → 1: squeezed, k₃ → k₂ − k₁)", ylabel="ΔB per unit f_NL  [(Mpc/h)⁶]",
           title=f"Bispectrum response vs angle, k₁ = {K1}, k₂ = {K2} h/Mpc")
    ax.legend(loc="best")

    fig.suptitle(f"{SETS[qset]} PNG, f_NL = ±{F_NL:g}, L = {BOX:g} Mpc/h, {n}³, z = 0"
                 + ("" if quijote is not None else "   —   Quijote-PNG not loaded: miniflims only"),
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = out / f"quijote_png_{qset}{'' if quijote is not None else '_miniflims_only'}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"figure: {path}")


def compare_single(path: str, n: int, out: Path, label: str = "Quijote-PNG (1 box)"):
    """One Quijote box (any PNG set) against the miniflims ensembles, pooled over
    all sets and signs: at f_NL = +-100 the PNG imprint is far below the cosmic
    variance of one box, so this tests cosmology and gravity, not PNG."""
    from miniflims.cosmology import Cosmology
    from miniflims.plotting import AXIS, INK, INK2, SERIES, _mpl, logx
    from miniflims.power import linear_power

    plt, _, _ = _mpl()
    runs = {}
    for f in sorted(out.glob(f"miniflims_*_n{n}_seed*.npz")):
        with np.load(f) as z:
            g = json.loads(str(z["meta"]))["gravity"]
            for sign in ("p", "m"):
                runs.setdefault(g, []).append({k: z[f"{sign}/{k}"] for k in ("k", "pk", "b_eq", "b_theta", "k3_theta")})
    if not runs:
        raise SystemExit(f"no miniflims runs at n={n} in {out}: run `generate` first")
    q = statistics(load_quijote(path) - 1.0)
    k = next(iter(runs.values()))[0]["k"]
    nk = len(k)
    qk, qpk = q["k"][:nk], q["pk"][:nk]
    sel = (k > 0.008) & (k < np.pi * n / BOX * 0.9)
    cosmo = Cosmology.from_config(load_config(overrides={"cosmology": QUIJOTE_COSMOLOGY}).cosmology)
    colors = {"zeldovich": SERIES[0], "tcola": SERIES[1]}
    names = {"zeldovich": "miniflims Zel'dovich", "tcola": "miniflims tCOLA"}
    dash = (0, (4, 3))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.6))
    ax = axes[0, 0]
    for g, rs in runs.items():
        a = np.array([r["pk"] for r in rs])
        m, sd = a.mean(0), a.std(0, ddof=1)
        ax.fill_between(k[sel], (m - sd)[sel], (m + sd)[sel], color=colors[g], alpha=0.2, lw=0)
        ax.plot(k[sel], m[sel], color=colors[g], label=f"{names[g]} ({len(rs)} boxes, ±1σ)")
    ax.plot(qk[sel], qpk[sel], color=INK, lw=1.6, label=label)
    ax.plot(k[sel], np.asarray(linear_power(jnp.asarray(k[sel]), cosmo)), color=AXIS, lw=1.2, ls=dash, label="linear (EH)")
    logx(ax)
    ax.set(yscale="log", xlabel="k  [h/Mpc]", ylabel="P(k)  [(Mpc/h)³]  (CIC compensated)", title="Matter power spectrum, z = 0")
    ax.legend(loc="lower left")

    ax = axes[0, 1]
    for g, rs in runs.items():
        a = np.array([r["pk"] for r in rs])
        m, sd = a.mean(0), a.std(0, ddof=1)
        ax.fill_between(k[sel], ((m - sd) / m)[sel], ((m + sd) / m)[sel], color=colors[g], alpha=0.15, lw=0)
        ax.plot(k[sel], (qpk / m)[sel], color=colors[g], label=f"Quijote / {names[g]}")
    ax.axhline(1, color=INK, lw=1.0, ls=dash)
    logx(ax)
    ax.set(xlabel="k  [h/Mpc]", ylabel="P_Quijote / ⟨P_miniflims⟩",
           title="Ratio (bands: ±1σ of one miniflims box)")
    ax.legend(loc="upper left")

    ax = axes[1, 0]
    for g, rs in runs.items():
        a = np.array([r["b_eq"] for r in rs])
        m, sd = a.mean(0), a.std(0, ddof=1)
        ax.fill_between(K_EQUIL, m - sd, m + sd, color=colors[g], alpha=0.2, lw=0)
        ax.plot(K_EQUIL, m, "o-", ms=4, color=colors[g], label=names[g])
    ax.plot(K_EQUIL, q["b_eq"], "s", ms=6, color=INK, label=label)
    ax.set(yscale="log", xlabel="k  [h/Mpc]", ylabel="B(k,k,k)  [(Mpc/h)⁶]", title="Equilateral bispectrum, z = 0")
    ax.legend(loc="upper right")

    ax = axes[1, 1]
    for g, rs in runs.items():
        a = np.array([r["b_theta"] for r in rs])
        m, sd = a.mean(0), a.std(0, ddof=1)
        ax.fill_between(THETAS / np.pi, m - sd, m + sd, color=colors[g], alpha=0.2, lw=0)
        ax.plot(THETAS / np.pi, m, "o-", ms=4, color=colors[g], label=names[g])
    ax.plot(THETAS / np.pi, q["b_theta"], "s", ms=6, color=INK, label=label)
    ax.set(yscale="log", xlabel="θ₁₂ / π   (θ → 1: squeezed)", ylabel="B  [(Mpc/h)⁶]",
           title=f"Bispectrum vs angle, k₁ = {K1}, k₂ = {K2} h/Mpc")
    ax.legend(loc="upper left")
    fig.suptitle(f"One Quijote-PNG box vs miniflims ({n}³ particles, Quijote cosmology, f_NL = ±{F_NL:g} pooled), "
                 f"L = {BOX:g} Mpc/h, z = 0", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path_out = out / "quijote_png_single_box.png"
    fig.savefig(path_out)
    plt.close(fig)
    print(f"figure: {path_out}")
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--n", type=int, default=128)
    g.add_argument("--seeds", type=int, default=4)
    g.add_argument("--out", default=str(OUT))
    c = sub.add_parser("compare")
    c.add_argument("--set", default="LC", choices=sorted(SETS))
    c.add_argument("--plus")
    c.add_argument("--minus")
    c.add_argument("--single", help="one Quijote box of any set: compare cosmology and gravity only")
    c.add_argument("--n", type=int, default=128)
    c.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.cmd == "generate":
        generate(a.n, a.seeds, Path(a.out))
    elif a.single:
        compare_single(a.single, a.n, Path(a.out))
    else:
        compare(a.set, a.plus, a.minus, a.n, Path(a.out))


if __name__ == "__main__":
    main()
