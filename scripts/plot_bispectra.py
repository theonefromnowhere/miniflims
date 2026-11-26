"""Binned bispectra of a run's fields against the mock truth, and their differences.

    JAX_PLATFORMS=cpu .venv-run/bin/python -u scripts/plot_bispectra.py runs/png_feature_w20_10k \
        [--n-bins 8] [--n-mc 200] [--from-sweep 1000]

Fields: the white noise x, the primordial linear density delta_L (with each
state's own PNG parameters) and the evolved density rho - 1. For each, the
binned bispectrum B(k1, k2, k3) over every closed triangle of |k| bins
(k1 <= k2 <= k3 bins, the estimator of the white-noise bispectrum prior),
shown in units of its scatter for Gaussian initial conditions:

    z = (B - <B>_G) / sigma_G,   from --n-mc Gaussian realizations

so |z| ~ 1 is cosmic variance (for rho - 1, <B>_G is the gravitational
bispectrum). delta_L is also split into its white-noise part (delta_L at
f_NL = 0) and its PNG part (the rest), to show which part carries the signal.

Posterior: the field samples in <run>/samples from --from-sweep on (both
chains, with the PNG parameters of the same sweep) and the latest checkpoint.

Writes <run>/bispectra/: bispectra.png (by triangle: truth, posterior,
posterior - truth), bispectra_ksum.png (delta_L parts against k1 + k2 + k3),
bispectra.npz and summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from miniflims.checkpoint import iter_samples, load_checkpoint, sample_sweeps
from miniflims.config import load_config
from miniflims.data import load_dataset
from miniflims.model import FieldModel
from miniflims.pipeline import set_precision
from miniflims.plotting import INK, MUTED, SERIES, _mpl

FIELDS = ("x", "lin", "lin_wn", "lin_png", "fin")
TITLES = {
    "x": "White noise x",
    "lin": "Primordial linear density δ_L (own PNG)",
    "lin_wn": "δ_L, white-noise part (f_NL = 0)",
    "lin_png": "δ_L, PNG part",
    "fin": "Evolved density ρ − 1",
}


def make_estimator(grid, n_bins: int, k_max=None):
    """B(f) for every closed triangle of |k| bins: sum_x d_i d_j d_l over the
    number of triangles (d_b: f filtered to shell b)."""
    n = grid.n
    kmag = np.asarray(grid.kmag())
    kf = grid.k_fundamental
    edges = np.linspace(0.5 * kf, k_max or grid.k_nyquist, n_bins + 1)
    masks = jnp.asarray(np.stack([(kmag >= edges[b]) & (kmag < edges[b + 1]) for b in range(n_bins)]), float)
    ones = jnp.fft.irfftn(masks.astype(complex), s=(n,) * 3, axes=(1, 2, 3))
    trip = [(i, j, l) for i in range(n_bins) for j in range(i, n_bins) for l in range(j, n_bins)]
    counts = np.array([float(jnp.sum(ones[i] * ones[j] * ones[l])) for i, j, l in trip])
    keep = counts > 1e-9 * n**-3
    trip, counts = np.array(trip)[keep], counts[keep]
    idx, cnt = jnp.asarray(trip, jnp.int32), jnp.asarray(counts)

    @jax.jit
    def bispectrum(f):
        d = jnp.fft.irfftn(jnp.fft.rfftn(f)[None] * masks, s=f.shape, axes=(1, 2, 3))
        tri = jax.lax.map(lambda t: jnp.sum(d[t[0]] * d[t[1]] * d[t[2]]), idx)
        return tri / cnt

    centres = 0.5 * (edges[1:] + edges[:-1])
    return bispectrum, trip, centres, edges


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--n-bins", type=int, default=8)
    ap.add_argument("--k-max", type=float, default=None)
    ap.add_argument("--n-mc", type=int, default=200)
    ap.add_argument("--from-sweep", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    run = Path(args.run_dir)
    out = run / "bispectra"
    out.mkdir(exist_ok=True)
    cfg = load_config(run / "config.yaml")
    set_precision(cfg.precision)
    data = load_dataset(cfg)
    model = FieldModel(cfg, data.response, data.counts)
    if model.mesh.n != model.grid.n:
        raise SystemExit("needs n_mesh == n_grid")
    ckpt, meta = load_checkpoint(run)

    bis, trip, centres, edges = make_estimator(model.grid, args.n_bins, args.k_max)
    n_tri = len(trip)
    shape = (model.grid.n,) * 3

    @jax.jit
    def gaussian_fields(x):
        lin = jnp.fft.irfftn(model.gaussian_density_k(x), s=shape)
        return bis(x), bis(lin), bis(model.forward(lin) - 1.0)

    @jax.jit
    def state_fields(x, png):
        lin_wn = jnp.fft.irfftn(model.gaussian_density_k(x), s=shape)
        lin = model.linear_density(x, png)
        b_lin = bis(lin)
        b_wn = bis(lin_wn)
        return dict(x=bis(x), lin=b_lin, lin_wn=b_wn, lin_png=b_lin - b_wn, fin=bis(model.forward(lin) - 1.0))

    # Gaussian calibration
    print(f"{n_tri} triangle bins; Gaussian calibration with {args.n_mc} realizations", flush=True)
    draws = {"x": [], "lin": [], "fin": []}
    for k in jax.random.split(jax.random.PRNGKey(args.seed), args.n_mc):
        bx, bl, bf = gaussian_fields(jax.random.normal(k, shape))
        draws["x"].append(np.asarray(bx)), draws["lin"].append(np.asarray(bl)), draws["fin"].append(np.asarray(bf))
    mc = {k: np.array(v) for k, v in draws.items()}
    mean = {k: mc[k].mean(0) for k in mc}
    sig = {k: mc[k].std(0, ddof=1) for k in mc}
    # the delta_L parts are measured in the units of delta_L; the PNG part has zero Gaussian mean
    for k in ("lin_wn", "lin_png"):
        mean[k], sig[k] = (mean["lin"] if k == "lin_wn" else 0.0 * mean["lin"]), sig["lin"]

    def z(b):
        return {k: (np.asarray(b[k]) - mean[k]) / sig[k] for k in FIELDS}

    truth = z(state_fields(jnp.asarray(data.truth["white_noise"]), jnp.asarray(data.truth["png"])))

    # posterior: field samples (both chains) + latest checkpoint
    png_trace = ckpt.trace["png"]
    sweeps = [s for s in sample_sweeps(run, "field", upto=ckpt.iteration) if s >= args.from_sweep]
    post, labels = [], []
    for s, w in iter_samples(run, "field", sweeps):
        for c in range(w.shape[0]):
            post.append(z(state_fields(jnp.asarray(w[c], jnp.float64), jnp.asarray(png_trace[c, s]))))
            labels.append((s, c))
    latest = [z(state_fields(jnp.asarray(ckpt.state["field"][c]), jnp.asarray(ckpt.state["png"][c, 0])))
              for c in range(ckpt.state["field"].shape[0])]
    print(f"posterior: {len(post)} samples (sweeps {sweeps[0]}-{sweeps[-1]}) + checkpoint at sweep {ckpt.iteration}",
          flush=True)
    P = {k: np.array([p[k] for p in post]) for k in FIELDS}
    L = {k: np.array([p[k] for p in latest]) for k in FIELDS}

    # chi^2 per field: truth, posterior mean over samples, and of (sample - truth)
    summ = {"run": str(run), "checkpoint_sweep": ckpt.iteration, "n_triangles": n_tri, "n_mc": args.n_mc,
            "n_samples": len(post), "sample_sweeps": [int(s) for s in sweeps], "edges": edges.tolist(),
            "chi2_gaussian_expectation": n_tri, "fields": {}}
    for k in FIELDS:
        summ["fields"][k] = {
            "chi2_truth": float(np.sum(truth[k] ** 2)),
            "chi2_samples_median": float(np.median(np.sum(P[k] ** 2, axis=1))),
            "chi2_latest": [float(np.sum(v**2)) for v in L[k]],
            "chi2_sample_minus_truth_median": float(np.median(np.sum((P[k] - truth[k]) ** 2, axis=1))),
            "corr_posterior_mean_truth": float(np.corrcoef(P[k].mean(0), truth[k])[0, 1]),
        }
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    np.savez(out / "bispectra.npz", triangles=trip, centres=centres, edges=edges, sweeps=np.array(labels),
             **{f"truth/{k}": truth[k] for k in FIELDS}, **{f"samples/{k}": P[k] for k in FIELDS},
             **{f"latest/{k}": L[k] for k in FIELDS}, **{f"mc_mean/{k}": mean[k] for k in mc},
             **{f"mc_std/{k}": sig[k] for k in mc})

    plt, _, _ = _mpl()
    tri_x = np.arange(n_tri)
    k1_starts = [i for i in range(n_tri) if i == 0 or trip[i, 0] != trip[i - 1, 0]]

    def by_triangle(ax, k, diff):
        ref = truth[k] if diff else 0.0
        q16, q50, q84 = np.percentile(P[k] - ref, [16, 50, 84], axis=0)
        ax.axhspan(-1, 1, color=MUTED, alpha=0.15, lw=0, label="Gaussian ICs ±1σ" if not diff else "±1σ_G")
        ax.axhline(0, color=INK, lw=0.7)
        ax.fill_between(tri_x, q16, q84, step="mid", color=SERIES[0], alpha=0.25, lw=0)
        ax.step(tri_x, q50, where="mid", color=SERIES[0], lw=1.3,
                label=f"posterior{' − truth' if diff else ''} (median, 68%)")
        for c, v in enumerate(L[k]):
            ax.plot(tri_x, v - ref, "o", ms=2.5, color=SERIES[1 + c], alpha=0.8,
                    label=f"chain {c + 1}, sweep {ckpt.iteration}{' − truth' if diff else ''}")
        if not diff:
            ax.step(tri_x, truth[k], where="mid", color=INK, lw=1.1, ls=(0, (3, 2)), label="truth")
        for s in k1_starts[1:]:
            ax.axvline(s - 0.5, color=MUTED, lw=0.5, alpha=0.6)
        ax.set_xlim(-0.5, n_tri - 0.5)
        f = summ["fields"][k]
        txt = (f"χ²: truth {f['chi2_truth']:.0f}, samples {f['chi2_samples_median']:.0f} (Gaussian ≈ {n_tri})"
               if not diff else f"χ² of (sample − truth): {f['chi2_sample_minus_truth_median']:.0f};  "
               f"corr(posterior mean, truth) = {f['corr_posterior_mean_truth']:.2f}")
        ax.text(0.01, 0.97, txt, transform=ax.transAxes, va="top", fontsize=8, color=INK)
        ax.set(xlabel="triangle bin (k₁ ≤ k₂ ≤ k₃; grey lines: next k₁ bin)", ylabel="(B − ⟨B⟩_G) / σ_G")
        ax.set_title(TITLES[k] + (": posterior − truth" if diff else ": truth and posterior"))

    rows = ("x", "lin", "fin")
    fig, axes = plt.subplots(len(rows), 2, figsize=(17, 4.2 * len(rows)))
    for r, k in enumerate(rows):
        by_triangle(axes[r, 0], k, False)
        by_triangle(axes[r, 1], k, True)
    axes[0, 0].legend(loc="lower right", fontsize=7.5, ncol=2, frameon=False)
    axes[0, 1].legend(loc="lower right", fontsize=7.5, ncol=2, frameon=False)
    fig.suptitle(f"{run.name}: binned bispectra, {args.n_bins} |k| bins up to "
                 f"{edges[-1]:.3f} h/Mpc ({n_tri} triangles); posterior from {len(post)} samples, "
                 f"sweeps {sweeps[0]}–{sweeps[-1]}", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "bispectra.png", dpi=130)
    plt.close(fig)

    # against k1 + k2 + k3: the feature template oscillates as sin(omega * sum k)
    ksum = centres[trip].sum(1)
    nb = 12
    bins = np.linspace(ksum.min() - 1e-9, ksum.max() + 1e-9, nb + 1)
    which = np.digitize(ksum, bins) - 1
    kc = 0.5 * (bins[1:] + bins[:-1])

    def binned(v):
        return np.array([v[..., which == b].mean(-1) if np.any(which == b) else np.nan for b in range(nb)]).T

    has_omega = "omega" in model.png.names
    jf = model.png.names.index("f_NL")
    chain_of = np.array([c for _, c in labels])
    n_ch = png_trace.shape[0]
    if has_omega:
        jo = model.png.names.index("omega")
        om_t = float(data.truth["png"][jo])
        lab_t = f"truth (ω = {om_t:g}, f_NL = {float(data.truth['png'][jf]):g})"
        note = f"Template ∝ f_NL sin(ω Σk): period 2π/ω = {2 * np.pi / om_t:.2f} h/Mpc (truth)"
    else:
        lab_t = f"truth (f_NL = {float(data.truth['png'][jf]):g})"
        note = f"{cfg.png.template} template"

    def lab_chain(c):
        f = float(np.median(png_trace[c, args.from_sweep:, jf]))
        if has_omega:
            return f"chain {c + 1} (median ω = {float(np.median(png_trace[c, args.from_sweep:, jo])):.1f}, f_NL = {f:.0f})"
        return f"chain {c + 1} (median f_NL = {f:.0f})"

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), sharey=True)
    for ax, k in zip(axes, ("lin", "lin_wn", "lin_png")):
        ax.axhspan(-1, 1, color=MUTED, alpha=0.15, lw=0)
        ax.axhline(0, color=INK, lw=0.7)
        ax.plot(ksum, truth[k], "o", ms=2.5, color=INK, alpha=0.35)
        ax.plot(kc, binned(truth[k]), color=INK, lw=1.4, ls=(0, (3, 2)), label=lab_t)
        for c in range(n_ch):  # each chain on its own: chains can sit in different modes
            sel = chain_of == c
            if not np.any(sel):
                continue
            pb = binned(P[k][sel])  # (samples, nb)
            q16, q50, q84 = np.nanpercentile(pb, [16, 50, 84], axis=0)
            col = SERIES[(c + 1) % len(SERIES)] if n_ch > 1 else SERIES[0]
            ax.fill_between(kc, q16, q84, color=col, alpha=0.2, lw=0)
            ax.plot(kc, q50, color=col, lw=1.4, label=lab_chain(c))
        ax.set(title=TITLES[k], xlabel="k₁ + k₂ + k₃  [h/Mpc]  (bin centres)")
    axes[0].set_ylabel("(B − ⟨B⟩_G) / σ_G")
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle(f"{run.name}: δ_L bispectrum against k₁ + k₂ + k₃ (dots: truth per triangle bin; lines: mean in "
                 f"{nb} k-sum bins, per chain with 68% posterior band). {note}",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "bispectra_ksum.png", dpi=130)
    plt.close(fig)

    for k in FIELDS:
        f = summ["fields"][k]
        print(f"{k:8s} chi2 truth {f['chi2_truth']:7.1f}  samples {f['chi2_samples_median']:7.1f}  "
              f"latest {', '.join(f'{v:.1f}' for v in f['chi2_latest'])}  (sample-truth) "
              f"{f['chi2_sample_minus_truth_median']:7.1f}  corr {f['corr_posterior_mean_truth']:+.2f}")
    print(out / "bispectra.png")
    print(out / "bispectra_ksum.png")


if __name__ == "__main__":
    main()
