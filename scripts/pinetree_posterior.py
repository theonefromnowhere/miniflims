"""Posterior of the PineTree parameters of a run (galaxies.population: pinetree)
against the mock truth: raw parameters and what they predict. Works on a
running run (latest checkpoint) or a finished one.

Parameters: posterior mean and std of each sampled raw parameter (whitened
runs are mapped back, theta = theta_ref + L z), truth, pull, R-hat.
Predictions, evaluated on the TRUE density field for posterior draws, so they
isolate the population model from the field reconstruction:
  - total expected count sum(lambda) vs the truth and the observed N
  - lambda in bins of the true density, ratio to the truth
  - the halo mass function above threshold, ratio to the truth

    .venv-run/bin/python scripts/pinetree_posterior.py runs/pinetree_sc_theta [--burn N] [--draws 100]
"""
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from miniflims import diagnostics as diag
from miniflims.checkpoint import load_checkpoint
from miniflims.config import load_config
from miniflims.data import load_dataset
from miniflims.model import FieldModel
from miniflims.pinetree import NAMES
from miniflims.pipeline import set_precision

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--burn", type=int, default=None, help="sweeps to drop [sampler.n_warmup]")
    ap.add_argument("--draws", type=int, default=100, help="posterior draws for the predictions")
    ap.add_argument("--out", default=None, help="figure [RUN/pinetree_posterior.png]")
    a = ap.parse_args()

    run = Path(a.run)
    cfg = load_config(run / "config.yaml")
    set_precision(cfg.precision)
    ckpt, meta = load_checkpoint(run)
    burn = cfg.sampler.n_warmup if a.burn is None else a.burn
    z = np.asarray(ckpt.trace["nuisance"])[:, :, 0, :]  # (chains, sweeps, n_specs)
    if z.shape[1] <= burn + 10:
        raise SystemExit(f"only {z.shape[1]} sweeps so far, burn-in {burn}: pass --burn")
    data = load_dataset(cfg)
    model = FieldModel(cfg, data.response, data.counts, halos=data.halos)
    pt = model.pinetree
    th = pt.theta_np(z[:, burn:])  # (chains, n, 19) raw
    truth = pt.theta_np(np.asarray(data.truth["nuisance"])[0])
    sampled = np.flatnonzero([n not in ("amp_w0", "amp_b0") for n in NAMES]) if pt.L is None else pt.sampled_raw
    print(f"{run}: sweeps {z.shape[1]} (burn {burn}), chains {z.shape[0]}, {int(data.counts.sum())} haloes")
    print(f"{'param':>8s} {'truth':>10s} {'mean':>10s} {'std':>9s} {'pull':>6s} {'R-hat':>6s}")
    for j in sampled:
        x = th[:, :, j]
        sd = x.std()
        print(f"{NAMES[j]:>8s} {truth[j]:10.4f} {x.mean():10.4f} {sd:9.3g} {(x.mean() - truth[j]) / sd:+6.2f} "
              f"{diag.gelman_rubin(x):6.3f}")

    # predictions on the true density field
    rho = jnp.asarray(data.truth["rho"])
    flat = th.reshape(-1, th.shape[-1])
    pick = np.random.default_rng(0).choice(len(flat), min(a.draws, len(flat)), replace=False)
    rate = jax.jit(lambda t: pt.rate(rho, t))
    lam_true = np.asarray(rate(jnp.asarray(truth)))
    lam = np.array([np.asarray(rate(jnp.asarray(flat[i]))) for i in pick])
    n_obs = int(data.counts.sum())
    tot = lam.sum(1)
    print(f"\nsum(lambda): truth {lam_true.sum():.1f}, observed N {n_obs}, posterior {tot.mean():.1f} ± {tot.std():.1f}")
    r = np.asarray(rho).ravel()
    q = np.quantile(r, [0, 0.5, 0.8, 0.95, 0.99, 0.999, 1.0])
    idx = np.clip(np.digitize(r, q[1:-1]), 0, len(q) - 2)
    rb_true = np.bincount(idx, lam_true) / np.bincount(idx)
    rb = np.array([np.bincount(idx, l) / np.bincount(idx) for l in lam])
    print("lambda per voxel in bins of the true rho (quantiles 0-50-80-95-99-99.9-100 %): posterior / truth")
    for b in range(len(q) - 1):
        print(f"  rho {q[b]:7.3g}-{q[b + 1]:<7.3g} truth {rb_true[b]:.4g}  ratio {np.mean(rb[:, b] / rb_true[b]):.3f}"
              f" ± {np.std(rb[:, b] / rb_true[b]):.3f}")
    m_edges = np.exp(np.linspace(pt.ln_threshold, np.log(3e15), 11))
    rib = jax.jit(lambda t, lo, hi: jnp.sum(pt.rate_in_bin(rho, t, lo, hi)))
    hmf_true = np.array([float(rib(jnp.asarray(truth), lo, hi)) for lo, hi in zip(m_edges[:-1], m_edges[1:])])
    hmf = np.array([[float(rib(jnp.asarray(flat[i]), lo, hi)) for lo, hi in zip(m_edges[:-1], m_edges[1:])]
                    for i in pick[:30]])
    hmf_obs = np.histogram(np.asarray(data.halos["ln_mass"]), np.log(m_edges))[0]
    print("HMF above threshold: posterior / truth (30 draws), observed / truth")
    for b in range(len(hmf_true)):
        print(f"  {m_edges[b]:.2e}-{m_edges[b + 1]:.2e}: {np.mean(hmf[:, b] / hmf_true[b]):.3f} ± "
              f"{np.std(hmf[:, b] / hmf_true[b]):.3f}   obs {hmf_obs[b] / hmf_true[b]:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": GRID, "figure.facecolor": SURF, "axes.facecolor": SURF,
                         "legend.frameon": False})
    n = len(sampled)
    ncol = 6
    nrow = int(np.ceil(n / ncol)) + 1
    fig = plt.figure(figsize=(3.0 * ncol, 2.3 * nrow))
    all_th = pt.theta_np(z)
    for k, j in enumerate(sampled):
        ax = fig.add_subplot(nrow, ncol, k + 1)
        for c in range(all_th.shape[0]):
            ax.plot(all_th[c, :, j], lw=0.6, color=SERIES[c % 3])
        ax.axhline(truth[j], color=INK, lw=1, ls=(0, (4, 3)))
        ax.axvline(burn, color=MUTED, lw=0.8)
        ax.set_title(NAMES[j], fontsize=9, loc="left")
    ax = fig.add_subplot(nrow, 3, 3 * nrow - 2)
    ax.hist(tot, 30, color=SERIES[0], alpha=0.7)
    ax.axvline(lam_true.sum(), color=INK, ls="--", lw=1, label="truth Σλ")
    ax.axvline(n_obs, color=SERIES[1], lw=1.2, label="observed N")
    ax.set_title("Σλ on the true field", loc="left"); ax.legend(fontsize=8)
    ax = fig.add_subplot(nrow, 3, 3 * nrow - 1)
    mid = np.sqrt(np.maximum(q[:-1], 1e-3) * q[1:])
    ratio = rb / rb_true
    ax.fill_between(mid, *np.percentile(ratio, [16, 84], axis=0), color=SERIES[0], alpha=0.3, lw=0)
    ax.plot(mid, ratio.mean(0), color=SERIES[0]); ax.axhline(1, color=INK2, lw=1)
    ax.set_xscale("log"); ax.set_xlabel("true ρ (bin)"); ax.set_title("λ(ρ): posterior / truth", loc="left")
    ax = fig.add_subplot(nrow, 3, 3 * nrow)
    mc = np.sqrt(m_edges[1:] * m_edges[:-1])
    ratio = hmf / hmf_true
    ax.fill_between(mc, *np.percentile(ratio, [16, 84], axis=0), color=SERIES[0], alpha=0.3, lw=0)
    ax.plot(mc, ratio.mean(0), color=SERIES[0], label="posterior")
    ax.plot(mc, hmf_obs / hmf_true, "o", ms=3, color=SERIES[1], label="observed")
    ax.axhline(1, color=INK2, lw=1); ax.set_xscale("log"); ax.set_xlabel("M [M_sun/h]")
    ax.set_title("HMF above threshold / truth", loc="left"); ax.legend(fontsize=8)
    fig.suptitle(f"{run.name}: PineTree parameters (traces: chains; dashed: truth; grey: end of warm-up)",
                 x=0.01, ha="left")
    fig.tight_layout()
    out = Path(a.out or run / "pinetree_posterior.png")
    fig.savefig(out, dpi=110)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
