"""Figures of the latest state of a (running or stopped) run, read from its
checkpoint, against the mock truth.

    JAX_PLATFORMS=cpu python scripts/plot_last_sample.py runs/png_feature_w20 [--plot-only]

Writes to <run_dir>/last_sample/:
    fields.png          slices: white noise, f_NL M Phi_NG, evolved density, counts
    spectra.png         white-noise spectrum, P(k), cross-correlation and transfer
                        with the truth, the non-Gaussian density part
    png_likelihood.png  log-likelihood of the PNG parameters on each field:
                        conditional slices in f_NL and omega, profile over f_NL
                        as a function of omega, best-fit f_NL(omega)
    gaussianity.png     white-noise bispectrum chi^2 vs Gaussian draws, one-point
                        PDF, galaxy counts vs evolved density
    summary.json        the numbers behind the figures
    profiles.npz        cached likelihood scans (--plot-only re-plots from it)

The mock is regenerated from the run's configuration; the script checks that
it reproduces the traced log-likelihood of every chain.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from miniflims import diagnostics as diag
from miniflims.checkpoint import load_checkpoint
from miniflims.config import from_dict
from miniflims.data import make_mock
from miniflims.model import FieldModel
from miniflims.pipeline import set_precision
from miniflims.plotting import INK, INK2, MUTED, SERIES, _mpl, image, logx

DASH = (0, (4, 3))


def states(ckpt, data):
    """[(label, white noise, nuisance (cats, params), png vector)], truth first."""
    out = [("truth", data.truth["white_noise"], data.truth["nuisance"], data.truth["png"])]
    for c in range(ckpt.state["field"].shape[0]):
        out.append((f"chain {c + 1}", ckpt.state["field"][c], ckpt.state["nuisance"][c], ckpt.state["png"][c, 0]))
    return out


def colour(i):
    return INK if i == 0 else SERIES[(i - 1) % len(SERIES)]


def style(i):
    return dict(color=colour(i), ls=DASH if i == 0 else "-", lw=1.4 if i == 0 else 1.8)


# --------------------------------------------------------------------------- #
# likelihood scans of the PNG parameters on a fixed field
# --------------------------------------------------------------------------- #
def scan(m, sts, omegas, f_grid, log=print):
    pm = m.png
    shape = (m.grid.n,) * 3
    i_om = pm.names.index("omega")
    ng_fn = jax.jit(lambda dk, p, w: pm.ng_density_k(dk, p.at[i_om].set(w)))
    # lax.map in batches of 8: a vmap over hundreds of f_NL values would hold
    # hundreds of forward models in memory at once
    ll_fn = jax.jit(lambda dk, ng, th, fs: jax.lax.map(
        lambda f: m.log_likelihood_rho(m.forward(jnp.fft.irfftn(dk + f * ng, s=shape)), th), fs, batch_size=8))
    lo, hi = float(pm.lower[0]), float(pm.upper[0])

    def fhat(dk, ng, th):
        """argmax over f_NL: ln L is close to quadratic in f_NL (the density is
        linear in it), so three parabola steps converge."""
        c = 0.0
        for h in (1000.0, 100.0, 10.0):
            fs = jnp.asarray([c - h, c, c + h])
            l0, l1, l2 = np.asarray(ll_fn(dk, ng, th, fs))
            curv = l2 - 2 * l1 + l0
            step = -h * (l2 - l0) / (2 * curv) if curv < 0 else (h if l2 > l0 else -h) * 3
            c = float(np.clip(c + np.clip(step, -5 * h, 5 * h), lo, hi))
        return c, float(ll_fn(dk, ng, th, jnp.asarray([c]))[0])

    res = {}
    for j, (lab, x, th, p) in enumerate(sts):
        t0 = time.time()
        dk = m.gaussian_density_k(jnp.asarray(x))
        th, p = jnp.asarray(th), jnp.asarray(p)
        # conditional in f_NL at this state's omega
        ng = ng_fn(dk, p, p[i_om])
        cond_f = np.asarray(ll_fn(dk, ng, th, jnp.asarray(f_grid)))
        # omega: conditional at this state's f_NL, and profile over f_NL
        cond_w, prof_w, best_f = [], [], []
        for w in omegas:
            ng = ng_fn(dk, p, w)
            cond_w.append(float(ll_fn(dk, ng, th, p[:1])[0]))
            f, l = fhat(dk, ng, th)
            best_f.append(f)
            prof_w.append(l)
        # this field at the other states' PNG parameters
        at = {}
        for lab2, _, _, p2 in sts:
            ng = ng_fn(dk, p, float(p2[i_om]))
            at[lab2] = float(ll_fn(dk, ng, th, jnp.asarray([float(p2[0])]))[0])
        res[lab] = dict(cond_f=cond_f, cond_w=np.array(cond_w), prof_w=np.array(prof_w),
                        best_f=np.array(best_f), at=at)
        log(f"  scanned {lab:8s} in {time.time() - t0:.0f} s")
    return res


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out", help="output directory (default <run_dir>/last_sample)")
    ap.add_argument("--slab", type=int, default=4, help="cells averaged in z for density and count slices")
    ap.add_argument("--omega-step", type=float, default=0.05)
    ap.add_argument("--n-gauss", type=int, default=200, help="Gaussian white-noise draws for the chi^2 reference")
    ap.add_argument("--plot-only", action="store_true", help="re-plot from the cached profiles.npz")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "last_sample"
    out.mkdir(parents=True, exist_ok=True)
    ckpt, meta = load_checkpoint(run_dir)
    cfg = from_dict(meta["config"])
    set_precision(cfg.precision)
    data = make_mock(cfg)
    m = FieldModel(cfg, data.response, data.counts)
    sts = states(ckpt, data)
    it = ckpt.iteration
    summary = {"run_dir": str(run_dir), "sweep": it, "total": meta["total"], "png_names": m.png.names,
               "nuisance_names": m.param_names, "states": {}}
    print(f"{run_dir}: sweep {it}/{meta['total']}")

    # ---------------------------------------------------------------- fields
    fields = {}
    for j, (lab, x, th, p) in enumerate(sts):
        xj, thj, pj = jnp.asarray(x), jnp.asarray(th), jnp.asarray(p)
        dk = m.gaussian_density_k(xj)
        ng = np.asarray(jnp.fft.irfftn(pj[0] * m.png.ng_density_k(dk, pj), s=x.shape))
        rho = np.asarray(m.final_density(xj, pj))
        lam = np.asarray(m.expected_counts(thj[0], jnp.asarray(rho), m.response[0]))
        ll = float(m.log_likelihood_rho(jnp.asarray(rho), thj))
        fields[lab] = dict(x=np.asarray(x), ng=ng, rho=rho, lam=lam)
        s = dict(png=dict(zip(m.png.names, map(float, p))), nuisance=dict(zip(m.param_names, map(float, th[0]))),
                 log_likelihood=ll, white_noise_moments=diag.gaussianity_moments(np.asarray(x)))
        if j > 0:
            traced = float(ckpt.trace["log_likelihood"][j - 1, it - 1])
            s["log_likelihood_traced"] = traced
            if abs(ll - traced) > 1e-6 * abs(traced):
                raise RuntimeError(f"{lab}: regenerated data give logL {ll}, the run traced {traced}")
        summary["states"][lab] = s
    print("  data reproduce the traced log-likelihoods")

    # ----------------------------------------------- white-noise Gaussianity
    m._setup_bs_prior(dataclasses.replace(cfg.prior.bispectrum, lambda_bs=1.0))  # calibrate the statistic
    chi2 = jax.jit(m.white_noise_bispectrum_chi2)
    key = jax.random.PRNGKey(2024)
    gauss_chi2 = np.array([float(chi2(jax.random.normal(k, (m.grid.n,) * 3)))
                           for k in jax.random.split(key, args.n_gauss)])
    n_tri = int(m._bs["idx"].shape[0])
    for lab, *_ in sts:
        summary["states"][lab]["bispectrum_chi2"] = float(chi2(jnp.asarray(fields[lab]["x"])))
    summary["bispectrum_chi2_gaussian"] = dict(n_bins=n_tri, mean=float(gauss_chi2.mean()), std=float(gauss_chi2.std()))

    # ------------------------------------------------------ likelihood scans
    spec = {s.name: s for s in m.png.specs}
    omegas = np.round(np.arange(spec["omega"].lower, min(spec["omega"].upper, 40.0) + 1e-9, args.omega_step), 6)
    f_grid = np.arange(-2500.0, 2500.0 + 1e-9, 10.0)
    cache = out / "profiles.npz"
    if args.plot_only and cache.exists():
        with np.load(cache, allow_pickle=True) as f:
            prof = f["prof"].item()
            omegas, f_grid = f["omegas"], f["f_grid"]
    else:
        print(f"  likelihood scans: {len(omegas)} omegas x {len(sts)} fields")
        prof = scan(m, sts, omegas, f_grid)
        np.savez(cache, prof=np.array(prof, dtype=object), omegas=omegas, f_grid=f_grid)
    for lab, r in prof.items():
        k = int(np.argmax(r["prof_w"]))
        summary["states"][lab]["profile_best"] = dict(omega=float(omegas[k]), f_NL=float(r["best_f"][k]),
                                                      log_likelihood=float(r["prof_w"][k]))
        summary["states"][lab]["log_likelihood_at_png_of"] = r["at"]

    # ----------------------------------------------------------------- plots
    plt, seq, div = _mpl()
    L, n = cfg.grid.box_size, cfg.grid.n_grid
    ext = (0, L, 0, L)
    iz = n // 2
    sl = slice(iz - args.slab // 2, iz - args.slab // 2 + args.slab)
    labels = [lab for lab, *_ in sts]

    def title(lab):
        s = summary["states"][lab]["png"]
        return f"{lab}: f_NL = {s['f_NL']:.0f}, ω = {s['omega']:.2f}"

    # fields.png
    fig, axes = plt.subplots(4, len(sts), figsize=(4.3 * len(sts), 14.5))
    ngmax = max(np.percentile(np.abs(fields[l]["ng"][:, :, iz]), 99.5) for l in labels)
    for j, lab in enumerate(labels):
        f = fields[lab]
        image(axes[0, j], f["x"][:, :, iz], div, ext, "x", -3.5, 3.5)
        axes[0, j].set_title(f"White noise · {lab}")
        image(axes[1, j], f["ng"][:, :, iz], div, ext, "δ_NG", -ngmax, ngmax)
        axes[1, j].set_title(f"f_NL M Φ_NG · {title(lab)}", fontsize=9)
        image(axes[2, j], np.log10(np.maximum(f["rho"][:, :, sl].mean(2), 1e-2)), div, ext, "log₁₀(1 + δ)", -1.0, 1.0)
        axes[2, j].set_title(f"Evolved density · {lab}")
    N = np.asarray(data.counts[0])[:, :, sl].sum(2)
    lam1 = fields[labels[1]]["lam"][:, :, sl].sum(2)
    vmax = np.percentile(N, 99.5)
    image(axes[3, 0], N, seq, ext, "counts", 0, vmax)
    axes[3, 0].set_title("Data: galaxy counts")
    image(axes[3, 1], lam1, seq, ext, "λ", 0, vmax)
    axes[3, 1].set_title(f"Expected counts λ · {labels[1]}")
    if len(sts) > 2:
        image(axes[3, 2], (N - lam1) / np.sqrt(np.maximum(lam1, 1e-9)), div, ext, "(N − λ)/√λ", -4, 4)
        axes[3, 2].set_title(f"Pearson residual · {labels[1]}")
    for ax in axes[:, 1:].ravel():
        ax.set_yticklabels([])
    for ax in axes[:, 0]:
        ax.set_ylabel("y  [Mpc/h]")
    for ax in axes[-1]:
        ax.set_xlabel("x  [Mpc/h]")
    fig.suptitle(f"{run_dir.name}, sweep {it}: slice z = {iz * L / n:.0f} Mpc/h (density and counts: "
                 f"{args.slab}-cell slab, {args.slab * L / n:.0f} Mpc/h)", x=0.01, ha="left", fontsize=11,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out / "fields.png")
    plt.close(fig)

    # spectra.png
    nb = 16
    cellv = (L / n) ** 3
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))
    truth = fields["truth"]
    for j, lab in enumerate(labels):
        f = fields[lab]
        k, pk, cnt = diag.power_spectrum(f["x"], L, nb)
        axes[0, 0].plot(k, pk / cellv, label=lab, **style(j))
        k, pk, _ = diag.power_spectrum(f["rho"] - 1, L, nb)
        axes[0, 1].plot(k, pk, label=lab, **style(j))
        k, pk, _ = diag.power_spectrum(f["ng"], L, nb)
        axes[1, 1].plot(k, pk, label=title(lab), **style(j))
        if j == 0:
            continue
        k, r = diag.cross_correlation(f["x"], truth["x"], L, nb)
        axes[0, 2].plot(k, r, color=colour(j), lw=1.8, label=f"{lab}: white noise")
        k, r = diag.cross_correlation(f["rho"], truth["rho"], L, nb)
        axes[0, 2].plot(k, r, color=colour(j), lw=1.8, ls=":", label=f"{lab}: evolved density")
        k, pa, _ = diag.power_spectrum(f["rho"] - 1, L, nb)
        k, pb, _ = diag.power_spectrum(truth["rho"] - 1, L, nb)
        axes[1, 0].plot(k, np.sqrt(pa / pb), color=colour(j), lw=1.8, label=lab)
        k, r = diag.cross_correlation(f["ng"], truth["ng"], L, nb)
        axes[1, 2].plot(k, r, color=colour(j), lw=1.8, label=lab)
    band = np.sqrt(2.0 / np.maximum(cnt, 1))
    axes[0, 0].fill_between(k, 1 - band, 1 + band, color=MUTED, alpha=0.18, lw=0, label="Gaussian ±1σ")
    axes[0, 0].axhline(1, color=MUTED, lw=0.8)
    axes[0, 0].set(title="White-noise power (1 for unit white noise)", ylabel="P_x / cell volume")
    axes[0, 1].set(title="Evolved density power", ylabel="P(k)  [(Mpc/h)³]", yscale="log")
    axes[0, 2].axhline(1, color=MUTED, lw=0.8)
    axes[0, 2].set(title="Cross-correlation with the truth", ylabel="r(k)", ylim=(-0.1, 1.05))
    axes[1, 0].axhline(1, color=MUTED, lw=0.8)
    axes[1, 0].set(title="Transfer function √(P_sample / P_truth)", ylabel="T(k)")
    axes[1, 1].set(title="Power of the PNG density f_NL M Φ_NG", ylabel="P(k)  [(Mpc/h)³]", yscale="log")
    axes[1, 2].axhline(0, color=MUTED, lw=0.8)
    axes[1, 2].set(title="PNG density: correlation with the truth", ylabel="r(k)", ylim=(-1.05, 1.05))
    for ax in axes.ravel():
        logx(ax)
        ax.set_xlabel("k  [h/Mpc]")
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "spectra.png")
    plt.close(fig)

    # png_likelihood.png
    truth_png = summary["states"]["truth"]["png"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))
    for j, lab in enumerate(labels):
        r = prof[lab]
        s = summary["states"][lab]["png"]
        y = r["cond_f"] - r["cond_f"].max()
        axes[0, 0].plot(f_grid, y, label=f"{lab} field, ω = {s['omega']:.2f}", **style(j))
        y = r["cond_w"] - r["cond_w"].max()
        axes[0, 1].plot(omegas, y, label=f"{lab} field, f_NL = {s['f_NL']:.0f}", **style(j))
        y = r["prof_w"] - r["prof_w"].max()
        axes[1, 0].plot(omegas, y, label=f"{lab} field", **style(j))
        axes[1, 1].plot(omegas, r["best_f"], label=f"{lab} field", **{**style(j), "lw": 1.2})
        if j > 0:  # where the chain currently is
            for ax, xv in ((axes[0, 0], s["f_NL"]), (axes[0, 1], s["omega"]), (axes[1, 0], s["omega"])):
                ax.axvline(xv, color=colour(j), lw=0.8, alpha=0.6)
            axes[1, 1].plot([s["omega"]], [s["f_NL"]], "o", ms=8, color=colour(j), mec="white", mew=1.5)
    axes[1, 1].plot([truth_png["omega"]], [truth_png["f_NL"]], "*", ms=13, color=INK, mec="white", mew=1.2,
                    label="truth")
    for ax, xv in ((axes[0, 0], truth_png["f_NL"]), (axes[0, 1], truth_png["omega"]), (axes[1, 0], truth_png["omega"])):
        ax.axvline(xv, color=INK, lw=0.8, ls=DASH)
    axes[0, 0].set(title="Conditional log-likelihood in f_NL (ω at each state's value)", xlabel="f_NL",
                   ylabel="Δ ln L", ylim=(-400, 15))
    axes[0, 1].set(title="Conditional log-likelihood in ω (f_NL at each state's value)", xlabel="ω  [Mpc/h]",
                   ylabel="Δ ln L", ylim=(-400, 15))
    axes[1, 0].set(title="Profile log-likelihood in ω (maximised over f_NL)", xlabel="ω  [Mpc/h]",
                   ylabel="Δ ln L", ylim=(-400, 15))
    axes[1, 1].set(title="Best-fit f_NL at each ω (dot: chain now, star: truth)", xlabel="ω  [Mpc/h]",
                   ylabel="f̂_NL(ω)")
    axes[1, 1].axhline(0, color=MUTED, lw=0.8)
    for ax in axes.ravel():
        ax.legend(loc="lower right" if ax is not axes[1, 1] else "best")
    fig.suptitle(f"{run_dir.name}, sweep {it}: PNG likelihood on each field (nuisance at each state's values; "
                 "vertical lines: chain now (solid), truth (dashed))", x=0.01, ha="left", fontsize=10.5,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "png_likelihood.png")
    plt.close(fig)

    # gaussianity.png
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    ax = axes[0]
    ax.hist(gauss_chi2, bins=30, color=MUTED, alpha=0.45, lw=0, label=f"{args.n_gauss} Gaussian white-noise draws")
    for j, lab in enumerate(labels):
        v = summary["states"][lab]["bispectrum_chi2"]
        ax.axvline(v, label=f"{lab}: {v:.0f}", **{**style(j), "lw": 2.0})
    ax.set(title=f"White-noise bispectrum χ² ({n_tri} triangle bins)", xlabel="χ²", ylabel="draws")
    ax.legend(loc="upper right")
    ax = axes[1]
    edges = np.linspace(-5, 5, 81)
    c = 0.5 * (edges[1:] + edges[:-1])
    ax.plot(c, np.exp(-c**2 / 2) / np.sqrt(2 * np.pi), color=MUTED, lw=3, alpha=0.5, label="N(0, 1)")
    for j, lab in enumerate(labels):
        mo = summary["states"][lab]["white_noise_moments"]
        h, _ = np.histogram(fields[lab]["x"], edges, density=True)
        ax.plot(c, np.where(h > 0, h, np.nan), label=f"{lab}: var {mo['variance']:.3f}, "
                f"skew {mo['skewness']:+.3f}, kurt {mo['kurtosis']:+.3f}", **{**style(j), "lw": 1.3})
    ax.set(yscale="log", ylim=(1e-6, 1), title="White-noise one-point PDF", xlabel="x", ylabel="density")
    ax.legend(loc="lower center", fontsize=7)
    ax = axes[2]
    lb = np.linspace(-1.2, 1.0, 23)
    lc = 0.5 * (lb[1:] + lb[:-1])
    cnt = np.asarray(data.counts[0])
    for j, lab in enumerate(labels):
        f = fields[lab]
        lr = np.log10(np.maximum(f["rho"], 1e-3)).ravel()
        idx = np.digitize(lr, lb) - 1
        ok = (idx >= 0) & (idx < len(lc))
        num = np.bincount(idx[ok], minlength=len(lc))
        mean = np.bincount(idx[ok], weights=cnt.ravel()[ok], minlength=len(lc)) / np.maximum(num, 1)
        lam_b = np.bincount(idx[ok], weights=f["lam"].ravel()[ok], minlength=len(lc)) / np.maximum(num, 1)
        good = num > 20
        ax.plot(10 ** lc[good], mean[good], "o", ms=4, color=colour(j), alpha=0.8)
        th = summary["states"][lab]["nuisance"]
        ax.plot(10 ** lc[good], lam_b[good], label=f"{lab}: n̄ = {th['nbar']:.3f}, α = {th['alpha']:.3f}", **style(j))
    ax.set(xscale="log", yscale="log", title="Counts vs evolved density (dots: data, lines: λ)",
           xlabel="1 + δ (each state's own field)", ylabel="⟨N⟩ per cell")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "gaussianity.png")
    plt.close(fig)

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for name in ("fields.png", "spectra.png", "png_likelihood.png", "gaussianity.png", "summary.json"):
        print(out / name)


if __name__ == "__main__":
    main()
