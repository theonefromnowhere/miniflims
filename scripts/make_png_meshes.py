"""Dark-matter density meshes with primordial non-Gaussianity.

    .venv/bin/python scripts/make_png_meshes.py [--n 64] [--box 256] [--seed 42] [--out meshes/png64_L256]

Every mesh uses the same white noise, so differences between meshes come
from the PNG parameters and the gravity model only. The initial conditions
are Phi = Phi_G + f_NL Phi_NG from jaxpng (see src/miniflims/png.py), evolved
to z = 0 with Zel'dovich ("lpt") or tCOLA, and painted with CIC on an n^3
mesh from n^3 particles.

Output, per mesh <name>:
    <name>.npz      rho (1 + delta, z = 0), delta_lin (linear density incl. PNG,
                    z = 0), and the settings as a JSON string ``meta``
    <name>.yaml     the full miniflims configuration used
plus manifest.json (all meshes), white_noise.npy and png_meshes.png (figure).
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

from miniflims import diagnostics as diag  # noqa: E402
from miniflims import load_config  # noqa: E402
from miniflims.config import dump_yaml  # noqa: E402
from miniflims.data import build_response  # noqa: E402
from miniflims.model import FieldModel  # noqa: E402

# name, gravity, template (None = Gaussian), parameters
MESHES = [
    ("lpt_gaussian", "lpt", None, {}),
    ("lpt_local_p100", "lpt", "local", {"f_NL": 100.0}),
    ("lpt_local_m100", "lpt", "local", {"f_NL": -100.0}),
    ("lpt_equil_p500", "lpt", "equilateral", {"f_NL": 500.0}),
    ("lpt_feature_p500_w20", "lpt", "feature", {"f_NL": 500.0, "omega": 20.0, "phase": 0.0}),
    ("cola_gaussian", "cola", None, {}),
    ("cola_local_p100", "cola", "local", {"f_NL": 100.0}),
    ("cola_equil_p500", "cola", "equilateral", {"f_NL": 500.0}),
]

COLA = {"n_steps": 10, "z_start": 19.0, "force_sampling": 2}


def build(n, box, gravity, template, params):
    over = {
        "grid": {"n_grid": n, "box_size": box},
        "forward": {"model": "zeldovich" if gravity == "lpt" else "tcola", "redshift": 0.0},
    }
    if gravity == "cola":
        over["forward"]["cola"] = COLA
    if template is not None:
        over["png"] = {
            "enabled": True,
            "template": template,
            "params": {k: {"init": v} for k, v in params.items()},
        }
    cfg = load_config(overrides=over)
    model = FieldModel(cfg, build_response(cfg))
    png = jnp.asarray(model.png.from_dict(params)) if model.png is not None else None
    return cfg, model, png


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--box", type=float, default=256.0, help="Mpc/h")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="meshes/png64_L256")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    x = np.random.default_rng(args.seed).standard_normal((args.n,) * 3)
    np.save(out / "white_noise.npy", x)
    xj = jnp.asarray(x)
    manifest = []
    for name, gravity, template, params in MESHES:
        t0 = time.time()
        cfg, model, png = build(args.n, args.box, gravity, template, params)
        delta_lin = np.asarray(model.linear_density(xj, png))
        rho = np.asarray(jax.jit(model.forward)(jnp.asarray(delta_lin)))
        meta = {
            "name": name, "gravity": gravity, "template": template or "gaussian", "params": params,
            "n_grid": args.n, "box_size": args.box, "seed": args.seed, "redshift": 0.0,
            "cola": COLA if gravity == "cola" else None,
            "lattice_offset_cells": float(model.forward.lattice_offset),
            "rho_mean": float(rho.mean()), "rho_std": float(rho.std()), "rho_min": float(rho.min()),
            "delta_lin_std": float(delta_lin.std()), "seconds": round(time.time() - t0, 1),
        }
        np.savez_compressed(out / f"{name}.npz", rho=rho.astype(np.float32),
                            delta_lin=delta_lin.astype(np.float32), meta=np.array(json.dumps(meta)))
        (out / f"{name}.yaml").write_text(dump_yaml(cfg))
        manifest.append(meta)
        print(f"{name:24s} std(rho) {meta['rho_std']:.3f}  min {meta['rho_min']:.3f}  {meta['seconds']:5.1f}s")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    figure(out, manifest, args.box)


def figure(out: Path, manifest, box):
    from miniflims.plotting import AXIS, GRID, INK, INK2, SERIES, _mpl, image, logx

    plt, seq, div = _mpl()
    load = {m["name"]: np.load(out / f"{m['name']}.npz") for m in manifest}
    rho = {k: v["rho"].astype(float) for k, v in load.items()}
    lin = {k: v["delta_lin"].astype(float) for k, v in load.items()}
    n = next(iter(rho.values())).shape[0]
    sl = n // 2
    ext = (0, box, 0, box)
    dash = (0, (4, 3))

    fig, axes = plt.subplots(3, 4, figsize=(17.5, 12.6))
    # row 1: Gaussian LPT, Gaussian COLA, and the gravity difference
    lg = {k: np.log10(np.maximum(rho[k][:, :, sl], 1e-2)) for k in ("lpt_gaussian", "cola_gaussian")}
    vmin, vmax = min(a.min() for a in lg.values()), max(a.max() for a in lg.values())
    for ax, k, title in ((axes[0, 0], "lpt_gaussian", "Zel'dovich, Gaussian ICs"),
                         (axes[0, 1], "cola_gaussian", "tCOLA (10 steps), Gaussian ICs")):
        image(ax, lg[k], seq, ext, "log₁₀ ρ", vmin, vmax)
        ax.set(title=title, xlabel="x  [Mpc/h]", ylabel="y  [Mpc/h]")

    # PNG imprint on the density: rho(f_NL) - rho(Gaussian), same white noise
    diffs = [("lpt_local_p100", "lpt_gaussian", "Zel'dovich: local f_NL = +100"),
             ("lpt_equil_p500", "lpt_gaussian", "Zel'dovich: equilateral f_NL = +500")]
    for ax, (a, b, title) in zip((axes[0, 2], axes[0, 3]), diffs):
        d = (rho[a] - rho[b])[:, :, sl]
        lim = np.percentile(np.abs(d), 99.5)
        image(ax, d, div, ext, "ρ_PNG − ρ_Gauss", -lim, lim)
        ax.set(title=title + "\n(minus the Gaussian mesh)", xlabel="x  [Mpc/h]", ylabel="y  [Mpc/h]")

    # row 2: power spectra relative to the Gaussian mesh of the same gravity
    nb = 24
    groups = (("lpt", "Zel'dovich"), ("cola", "tCOLA"))
    for ax, (g, label) in zip((axes[1, 0], axes[1, 1]), groups):
        k, p0, _ = diag.power_spectrum(rho[f"{g}_gaussian"] - 1, box, nb)
        for col, m in zip(SERIES[1:], [m for m in manifest if m["gravity"] == g and m["template"] != "gaussian"]):
            pk = diag.power_spectrum(rho[m["name"]] - 1, box, nb)[1]
            ax.plot(k, pk / p0 - 1, color=col, label=_label(m))
        ax.axhline(0, color=INK, lw=1.0, ls=dash)
        logx(ax)
        ax.set(xlabel="k  [h/Mpc]", ylabel="P / P_Gauss − 1", title=f"{label}: power spectrum change")
        ax.legend(loc="best")

    ax = axes[1, 2]
    k, p_lpt, _ = diag.power_spectrum(rho["lpt_gaussian"] - 1, box, nb)
    p_cola = diag.power_spectrum(rho["cola_gaussian"] - 1, box, nb)[1]
    p_lin = diag.power_spectrum(lin["lpt_gaussian"], box, nb)[1]
    ax.plot(k, p_lin, color=INK, lw=1.2, ls=dash, label="linear")
    ax.plot(k, p_lpt, color=SERIES[0], label="Zel'dovich")
    ax.plot(k, p_cola, color=SERIES[1], label="tCOLA")
    logx(ax)
    ax.set(yscale="log", xlabel="k  [h/Mpc]", ylabel="P(k)  [(Mpc/h)³]", title="Gaussian meshes, z = 0")
    ax.legend(loc="lower left")

    ax = axes[1, 3]
    bins = np.linspace(-1, 1.6, 90)
    for col, k_ in zip(SERIES, ("lpt_gaussian", "cola_gaussian")):
        ax.hist(np.log10(np.maximum(rho[k_].ravel(), 1e-3)), bins, histtype="step", lw=2, color=col,
                density=True, label="Zel'dovich" if k_.startswith("lpt") else "tCOLA")
    ax.set(yscale="log", xlabel="log₁₀ ρ", ylabel="PDF", title="One-point density PDF (Gaussian ICs)")
    ax.legend(loc="upper right")

    # row 3: the primordial signal itself - equilateral bispectrum of the linear field
    from miniflims.png import import_jaxpng

    jp = import_jaxpng()
    jax.config.update("jax_enable_x64", True)
    kf = 2 * np.pi / box
    kk = kf * np.arange(3, n // 3)
    theory = _theory_equilateral(box, n)
    for ax, (g, label) in zip((axes[2, 0], axes[2, 1]), groups):
        b0 = _bk_equil(jp, lin[f"{g}_gaussian"], box, kk)
        for col, m in zip(SERIES[1:], [m for m in manifest if m["gravity"] == g and m["template"] != "gaussian"]):
            ax.plot(kk, _bk_equil(jp, lin[m["name"]], box, kk) - b0, color=col, label=_label(m))
            ax.plot(kk, theory(kk, m), color=col, lw=1.2, ls=dash)
        ax.axhline(0, color=AXIS, lw=1.0)
        ax.set(xlabel="k  [h/Mpc]", ylabel="B_lin(k,k,k) − B_Gauss  [(Mpc/h)⁶]",
               title=f"Linear ICs ({label} runs): equilateral ΔB")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        ax.set_xlim(kk[0], 0.3)
        handles, labels = ax.get_legend_handles_labels()
        from matplotlib.lines import Line2D
        handles.append(Line2D([], [], color=INK2, lw=1.2, ls=dash))
        labels.append("tree level 6 f_NL P_lin² / M")
        ax.legend(handles, labels, loc="best")

    ax = axes[2, 2]
    b0 = _bk_equil(jp, rho["lpt_gaussian"] - 1, box, kk)
    b1 = _bk_equil(jp, rho["cola_gaussian"] - 1, box, kk)
    ax.plot(kk, b0, color=SERIES[0], label="Zel'dovich")
    ax.plot(kk, b1, color=SERIES[1], label="tCOLA")
    ax.set(yscale="log", xlabel="k  [h/Mpc]", ylabel="B(k,k,k)  [(Mpc/h)⁶]",
           title="Gravitational bispectrum, Gaussian ICs (z = 0)")
    ax.legend(loc="best")

    ax = axes[2, 3]
    ax.axis("off")
    rows = [f"{m['name']:22s} {m['rho_std']:.3f}  {m['rho_min']:.3f}" for m in manifest]
    ax.text(0.0, 1.0, f"{n}³ particles and mesh, L = {box:g} Mpc/h, seed {manifest[0]['seed']}\n"
            f"tCOLA: {COLA['n_steps']} steps from z = {COLA['z_start']:g}, "
            f"force mesh {COLA['force_sampling']}×\n\nmesh                    std(ρ)  min(ρ)\n" + "\n".join(rows),
            family="monospace", fontsize=8.5, va="top", color=INK2, transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(out / "png_meshes.png")
    plt.close(fig)
    print(f"figure: {out / 'png_meshes.png'}")


def _theory_equilateral(box, n):
    """Tree-level B_delta(k,k,k) = M^3 B_Phi = 6 f_NL P_lin(k)^2 / M(k) for every
    template at the equilateral configuration (x sin(3 omega k + phase) for the
    feature template)."""
    from miniflims.cosmology import Cosmology
    from miniflims.png import poisson_kernel
    from miniflims.power import linear_power

    cosmo = Cosmology.from_config(load_config().cosmology)

    def theory(k, m):
        k = jnp.asarray(k)
        b = 6.0 * m["params"]["f_NL"] * linear_power(k, cosmo) ** 2 / poisson_kernel(k, cosmo)
        if "omega" in m["params"]:
            b = b * jnp.sin(3 * m["params"]["omega"] * k + m["params"].get("phase", 0.0))
        return np.asarray(b)

    return theory


def _binned_theory_equilateral(box, n):
    """Tree-level B_delta averaged over exactly the triangles the FFT estimator
    uses for B(k,k,k) (shells [k - k_F, k + k_F), jaxpng.measure conventions):
    sum_tri B / N_tri, computed with FFTs for the separable template. At low k
    this differs from B at (k,k,k): the shells contain non-equilateral triangles."""
    from miniflims.cosmology import Cosmology
    from miniflims.png import TEMPLATES, import_jaxpng, poisson_kernel
    from miniflims.power import linear_power

    jp = import_jaxpng()
    jax.config.update("jax_enable_x64", True)
    cosmo = Cosmology.from_config(load_config().cosmology)
    kf = 2 * np.pi / box
    _, _, _, kint = jp.measure._integer_k(n)
    kphys = jnp.asarray(kint) * kf
    M = poisson_kernel(kphys, cosmo)
    k0 = jnp.asarray(0.01)
    amp = float(linear_power(k0, cosmo) / poisson_kernel(k0, cosmo) ** 2 / jp.grids.P_kernel(k0, cosmo.n_s))
    irf = lambda a: jnp.fft.irfftn(a, s=(n,) * 3)  # noqa: E731

    def theory(kk, m):
        p = m["params"]
        terms = TEMPLATES[m["template"]].build(jp, cosmo.n_s, p.get("omega", 0.0), p.get("phase", 0.0))
        out = []
        for k in np.atleast_1d(kk):
            mask = (kint >= (k - kf) / kf) & (kint < (k + kf) / kf)
            tri = float(jnp.sum(irf(jnp.where(mask, 1.0 + 0j, 0)) ** 3))
            tot = 0.0
            for gs in terms:
                F = [irf(jnp.where(mask, M * amp ** (2 / 3) * jnp.broadcast_to(jnp.asarray(g(kphys), jnp.float64), kphys.shape), 0.0)
                         .astype(complex)) for g in gs]
                tot += float(jnp.sum(F[0] * F[1] * F[2]))
            out.append(p["f_NL"] * tot / tri)
        return np.asarray(out)

    return theory


def _bk_equil(jp, field, box, kk):
    return np.array([float(jp.measure.bispectrum(jnp.asarray(field), box, k, k, jp.measure.EQUILATERAL_THETA)["B"][0])
                     for k in kk])


def _label(m):
    p = m["params"]
    s = f"{m['template']} f_NL = {p['f_NL']:+g}"
    if "omega" in p:
        s += f", ω = {p['omega']:g}"
    return s


if __name__ == "__main__":
    main()
