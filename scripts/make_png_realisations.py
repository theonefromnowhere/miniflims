"""Several realizations of tCOLA (or Zel'dovich) PNG meshes.

    .venv/bin/python scripts/make_png_realisations.py [--n 64] [--box 1024] [--realisations 10]
                                                      [--gravity cola] [--out meshes/cola64_L1024_x10]

For each realization r (white-noise seed = --seed + r) every PNG setting in
SETTINGS is run from the same white noise, so differences within a
realization come from f_NL alone and differences between realizations are
cosmic variance.

Output: r<rr>/<setting>.npz (rho, delta_lin as float32, meta), r<rr>/white_noise.npy,
manifest.json, and realisations.png (ensemble sanity checks).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_png_meshes import COLA, _binned_theory_equilateral, _theory_equilateral, build  # noqa: E402

from miniflims import diagnostics as diag  # noqa: E402

PRESETS = {
    # the PNG settings of the tCOLA meshes in make_png_meshes.py
    "templates": [
        ("gaussian", None, {}),
        ("local_p100", "local", {"f_NL": 100.0}),
        ("equil_p500", "equilateral", {"f_NL": 500.0}),
    ],
    # oscillatory feature, fixed f_NL, two frequencies
    "feature_omega": [
        ("gaussian", None, {}),
        ("feature_p500_w5", "feature", {"f_NL": 500.0, "omega": 5.0, "phase": 0.0}),
        ("feature_p500_w20", "feature", {"f_NL": 500.0, "omega": 20.0, "phase": 0.0}),
    ],
    "feature_w100": [
        ("gaussian", None, {}),
        ("feature_p500_w100", "feature", {"f_NL": 500.0, "omega": 100.0, "phase": 0.0}),
    ],
}
SETTINGS = PRESETS["templates"]


def label_of(name):
    for n, template, p in SETTINGS:
        if n == name:
            if template is None:
                return "Gaussian"
            s = f"{template} f_NL = {p['f_NL']:+g}"
            return s + (f", ω = {p['omega']:g}" if "omega" in p else "")
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--box", type=float, default=1024.0, help="Mpc/h")
    ap.add_argument("--realisations", type=int, default=10)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--gravity", choices=["cola", "lpt"], default="cola")
    ap.add_argument("--force-sampling", type=int, default=4,
                    help="tCOLA force mesh / particle grid (4 recommended for >10 Mpc/h cells)")
    ap.add_argument("--n-steps", type=int, default=10)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="templates")
    ap.add_argument("--plot-only", action="store_true", help="only redo the figure of an existing --out")
    ap.add_argument("--seeds-from", help="reuse the seeds of an earlier run (its manifest.json)")
    ap.add_argument("--random-seeds", action="store_true",
                    help="draw the white-noise seeds at random (recorded in manifest.json)")
    ap.add_argument("--out", default="meshes/cola64_L1024_x10")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    COLA.update(force_sampling=args.force_sampling, n_steps=args.n_steps)  # used by build()
    SETTINGS[:] = PRESETS[args.preset]
    if args.plot_only:
        figure(out, args)
        return
    if args.seeds_from:
        old = json.loads(Path(args.seeds_from).read_text())
        seeds = [m["seed"] for m in sorted(old, key=lambda m: m["realisation"]) if m["setting"] == "gaussian"]
        args.realisations = len(seeds)
    elif args.random_seeds:
        seeds = [int(s) for s in np.random.SeedSequence().generate_state(args.realisations, dtype=np.uint32)]
    else:
        seeds = [args.seed + r for r in range(args.realisations)]

    models = {}  # built once per setting, reused for all realizations
    for name, template, params in SETTINGS:
        cfg, model, png = build(args.n, args.box, args.gravity, template, params)
        models[name] = (cfg, model, png, jax.jit(model.forward))

    manifest = []
    for r in range(args.realisations):
        seed = seeds[r]
        rdir = out / f"r{r:02d}"
        rdir.mkdir(exist_ok=True)
        x = np.random.default_rng(seed).standard_normal((args.n,) * 3)
        np.save(rdir / "white_noise.npy", x)
        xj = jnp.asarray(x)
        for name, template, params in SETTINGS:
            t0 = time.time()
            cfg, model, png, fwd = models[name]
            delta_lin = np.asarray(model.linear_density(xj, png))
            rho = np.asarray(fwd(jnp.asarray(delta_lin)))
            meta = {
                "realisation": r, "seed": seed, "setting": name, "template": template or "gaussian",
                "params": params, "gravity": args.gravity, "n_grid": args.n, "box_size": args.box,
                "redshift": 0.0, "cola": COLA if args.gravity == "cola" else None,
                "rho_mean": float(rho.mean()), "rho_std": float(rho.std()),
                "seconds": round(time.time() - t0, 1),
            }
            np.savez_compressed(rdir / f"{name}.npz", rho=rho.astype(np.float32),
                                delta_lin=delta_lin.astype(np.float32), meta=np.array(json.dumps(meta)))
            manifest.append(meta)
        print(f"realisation {r:2d} (seed {seed}): " + ", ".join(
            f"{m['setting']} std(rho) {m['rho_std']:.3f}" for m in manifest[-len(SETTINGS):]), flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    figure(out, args)


def figure(out: Path, args):
    from miniflims.plotting import AXIS, INK, INK2, SERIES, _mpl, image, logx
    from miniflims.png import import_jaxpng

    plt, seq, div = _mpl()
    jp = import_jaxpng()
    jax.config.update("jax_enable_x64", True)
    dash = (0, (4, 3))
    box, n, R = args.box, args.n, args.realisations
    names = [s[0] for s in SETTINGS]
    load = {(r, s): np.load(out / f"r{r:02d}" / f"{s}.npz") for r in range(R) for s in names}
    rho = {key: v["rho"].astype(float) for key, v in load.items()}
    lin = {key: v["delta_lin"].astype(float) for key, v in load.items()}

    # CIC-compensated estimators (jaxpng.measure, Pylians conventions), k < k_Nyquist
    def pk_of(f, mas):
        k_, p_ = jp.measure.power_spectrum(jnp.asarray(f), box, MAS=mas)
        return np.asarray(k_), np.asarray(p_)

    kny = np.pi * n / box
    k_all, _ = pk_of(rho[0, "gaussian"] - 1, "CIC")
    ksel = k_all < kny
    k = k_all[ksel]
    pk = {s: np.array([pk_of(rho[r, s] - 1, "CIC")[1][ksel] for r in range(R)]) for s in names}
    kf = 2 * np.pi / box
    kk = kf * np.arange(2, n // 3)

    def bq(f, mas):
        out_b, out_q = [], []
        for kv in kk:
            res = jp.measure.bispectrum(jnp.asarray(f), box, kv, kv, jp.measure.EQUILATERAL_THETA, MAS=mas)
            out_b.append(float(res["B"][0]))
            out_q.append(float(res["Q"][0]))
        return out_b, out_q

    b_lin = {s: np.array([bq(lin[r, s], "None")[0] for r in range(R)]) for s in names}
    bq_nl = {s: [bq(rho[r, s] - 1, "CIC") for r in range(R)] for s in names}
    b_nl = {s: np.array([b for b, _ in bq_nl[s]]) for s in names}
    q_nl = {s: np.array([q for _, q in bq_nl[s]]) for s in names}
    theory = _binned_theory_equilateral(box, n)  # averaged over the estimator's triangles

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    label = {s: label_of(s) for s in names}

    ax = axes[0, 0]
    for col, s in zip(SERIES, names):
        m, sd = pk[s].mean(0), pk[s].std(0, ddof=1)
        ax.fill_between(k, m - sd, m + sd, color=col, alpha=0.2, lw=0)
        ax.plot(k, m, color=col, label=label[s])
    from miniflims.cosmology import Cosmology
    from miniflims.power import linear_power
    from miniflims import load_config

    cosmo = Cosmology.from_config(load_config().cosmology)
    ax.plot(k, np.asarray(linear_power(jnp.asarray(k), cosmo)), color=INK, lw=1.2, ls=dash, label="linear")
    logx(ax)
    ax.set(yscale="log", xlabel="k  [h/Mpc]", ylabel="P(k)  [(Mpc/h)³]  (CIC compensated)",
           title=f"Mean ± scatter of {R} realizations (tCOLA, z = 0)")
    ax.legend(loc="lower left")

    ax = axes[0, 1]
    for col, s in zip(SERIES[1:], names[1:]):
        ratio = pk[s] / pk["gaussian"] - 1  # paired: same white noise
        m, se = ratio.mean(0), ratio.std(0, ddof=1) / np.sqrt(R)
        ax.fill_between(k, m - se, m + se, color=col, alpha=0.25, lw=0)
        ax.plot(k, m, color=col, label=label[s])
    ax.axhline(0, color=INK, lw=1.0, ls=dash)
    logx(ax)
    ax.set(xlabel="k  [h/Mpc]", ylabel="P / P_Gauss − 1  (paired, mean ± s.e.)",
           title="Power-spectrum response to PNG")
    ax.legend(loc="upper left")

    ax = axes[0, 2]
    oscillating = any("omega" in p for _, _, p in SETTINGS)
    theory_cache = {}
    for col, s in zip(SERIES[1:], names[1:]):
        meta = json.loads(str(load[0, s]["meta"]))
        d = b_lin[s] - b_lin["gaussian"]
        t = theory_cache[s] = theory(kk, meta)
        if oscillating:  # a ratio blows up at the zero crossings: show Delta B and theory
            m, se = (d * kk**4).mean(0), (d * kk**4).std(0, ddof=1) / np.sqrt(R)
            ax.plot(kk, t * kk**4, color=col, lw=1.2, ls=dash)
        else:
            m, se = (d / t).mean(0), (d / t).std(0, ddof=1) / np.sqrt(R)
        ax.fill_between(kk, m - se, m + se, color=col, alpha=0.25, lw=0)
        ax.plot(kk, m, "o-", ms=4, color=col, label=label[s])
    if oscillating:
        ax.axhline(0, color=AXIS, lw=1.0)
        ax.set(xlabel="k  [h/Mpc]", ylabel="k⁴ ΔB_lin(k,k,k)  [(Mpc/h)²]  (mean ± s.e.)",
               title="Primordial bispectrum of the ICs\n(dashed: tree level averaged over the estimator's triangles)")
    else:
        ax.axhline(1, color=INK, lw=1.0, ls=dash)
        ax.set(ylim=(0, 2), xlabel="k  [h/Mpc]", ylabel="ΔB_lin(k,k,k) / bin-averaged tree level  (mean ± s.e.)",
               title="Primordial bispectrum of the ICs vs theory")
    ax.legend(loc="best")

    ax = axes[1, 0]
    for col, s in zip(SERIES, names):
        m, se = q_nl[s].mean(0), q_nl[s].std(0, ddof=1) / np.sqrt(R)
        ax.fill_between(kk, m - se, m + se, color=col, alpha=0.25, lw=0)
        ax.plot(kk, m, "o-", ms=4, color=col, label=label[s])
    ax.set(xlabel="k  [h/Mpc]", ylabel="Q(k,k,k)  (mean ± s.e.)",
           title="Reduced equilateral bispectrum of the evolved density")
    ax.legend(loc="upper right")

    ax = axes[1, 1]
    for col, s in zip(SERIES[1:], names[1:]):
        d = b_nl[s] - b_nl["gaussian"]
        m, se = d.mean(0), d.std(0, ddof=1) / np.sqrt(R)
        ax.fill_between(kk, m - se, m + se, color=col, alpha=0.25, lw=0)
        ax.plot(kk, m, "o-", ms=4, color=col, label=label[s] + " (evolved)")
        ax.plot(kk, theory_cache[s], color=col, lw=1.2, ls=dash)
    if oscillating:
        ax.set_yscale("symlog", linthresh=1e5)
        ax.axhline(0, color=AXIS, lw=1.0)
    else:
        ax.set_yscale("log")
    ax.set(xlabel="k  [h/Mpc]", ylabel="B − B_Gauss  [(Mpc/h)⁶]  (paired, CIC compensated)",
           title="PNG imprint on the evolved bispectrum\n(dashed: primordial tree level)")
    ax.legend(loc="upper right")

    ax = axes[1, 2]
    sl = n // 2
    d = (rho[0, names[1]] - rho[0, "gaussian"])[:, :, sl]
    lim = np.percentile(np.abs(d), 99.5)
    image(ax, d, div, (0, box, 0, box), "ρ_PNG − ρ_Gauss", -lim, lim)
    ax.set(title=f"Realization 0: {label[names[1]]} minus Gaussian", xlabel="x  [Mpc/h]", ylabel="y  [Mpc/h]")

    fig.suptitle(f"{R} realizations × {len(SETTINGS)} PNG settings, {n}³ particles and mesh, L = {box:g} Mpc/h, "
                 f"tCOLA {COLA['n_steps']} steps from z = {COLA['z_start']:g} (force mesh {COLA['force_sampling']}×)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "realisations.png")
    plt.close(fig)
    print(f"figure: {out / 'realisations.png'}")


if __name__ == "__main__":
    main()
