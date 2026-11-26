"""Figures: shared style and the diagnostics of a finished run.

    miniflims plot runs/default  ->  runs/default/{traces,field,diagnostics}.png

Needs matplotlib (``pip install -e ".[plot]"``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Palette: fixed categorical order, one-hue sequential ramp, two-hue diverging
# ramp with a neutral midpoint. Reference lines (truth, theory) use ink, not a
# series colour.
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
_SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
_DIV = ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f0a3a2", "#e34948", "#8a1f1f"]


def truth_log_likelihood(model, z) -> float | None:
    """Log-likelihood of the data at the mock truth (white noise, nuisance and
    PNG parameters), the quantity traced as ``log_likelihood``; None without
    a stored truth. ``z``: arrays of chains.npz (``truth/*``)."""
    if "truth/white_noise" not in z or "truth/nuisance" not in z:
        return None
    import jax.numpy as jnp

    png = jnp.asarray(z["truth/png"]) if "truth/png" in z else None
    if model.png is not None and png is None:
        return None
    return float(model.log_likelihood(jnp.asarray(z["truth/white_noise"]), jnp.asarray(z["truth/nuisance"]), png))


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.fontsize": 8, "text.color": INK,
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    })
    seq = LinearSegmentedColormap.from_list("miniflims_seq", _SEQ)
    div = LinearSegmentedColormap.from_list("miniflims_div", _DIV)
    return plt, seq, div


def logx(ax):
    """Log x axis without minor tick labels (they collide on narrow ranges)."""
    from matplotlib.ticker import NullFormatter

    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(NullFormatter())


def image(ax, data, cmap, extent, label, vmin=None, vmax=None):
    """Slice image with its own colour bar; no grid over the pixels."""
    im = ax.imshow(data.T, origin="lower", cmap=cmap, extent=extent, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.grid(False)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(label, color=INK2)
    cb.outline.set_edgecolor(AXIS)
    return im


def _chunks(seq, n):
    return [seq[i : i + n] for i in range(0, len(seq), n)]


# --------------------------------------------------------------------------- #
def plot_run(run_dir, max_field_samples: int = 40) -> list[Path]:
    """Trace and field figures for a directory written by ``pipeline.save``."""
    import jax
    import jax.numpy as jnp

    from . import diagnostics as diag
    from .config import load_config
    from .model import FieldModel
    from .pipeline import set_precision

    plt, seq, div = _mpl()
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    set_precision(cfg.precision)
    with np.load(run_dir / "chains.npz") as f:
        z = {k: f[k] for k in f.files}
    model = FieldModel(cfg, z["response"], z["counts"])
    nw, box, nb = cfg.sampler.n_warmup, cfg.grid.box_size, cfg.output.n_k_bins
    names = list(model.param_names)
    has_truth = "truth/nuisance" in z
    out = []

    # ------------------------------------------------------------- traces
    theta = z["trace/nuisance"]  # (chains, sweeps, cats, params)
    n_chains, n_iter, n_cat, _ = theta.shape
    panels = [("loglike", None, None)] + [
        (names[j], c, j) for c in range(n_cat) for j in range(len(names)) if model.sampled[j]
    ]
    if model.png is not None and "trace/png" in z:
        panels += [(name, "png", j) for j, name in enumerate(model.png.names) if model.png.sampled[j]]
    ncol = min(3, len(panels))
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.5 * nrow), squeeze=False)
    it = np.arange(n_iter)
    for ax, (name, c, j) in zip(axes.ravel(), panels):
        if j is None:
            series = z["trace/log_likelihood"]
        elif c == "png":
            series = z["trace/png"][:, :, j]
        else:
            series = theta[:, :, c, j]
        for ch in range(min(n_chains, len(SERIES))):
            ax.plot(it, series[ch], color=SERIES[ch], lw=1.2, label=f"chain {ch}")
        ax.axvspan(0, nw, color=GRID, alpha=0.6, lw=0)
        if j is None:
            ax.set_title("log-likelihood")
            ll_true = truth_log_likelihood(model, z)
            top = series.max() if ll_true is None else max(series.max(), ll_true)
            lo = np.percentile(series[:, nw // 2 :], 0.5)
            pad = 0.05 * abs(top - lo)
            ax.set_ylim(lo - pad, top + pad)
            if ll_true is not None:
                ax.axhline(ll_true, color=INK, lw=1.0, ls=(0, (4, 3)))
                post = series[:, nw:] if n_iter > nw else series[:, n_iter // 2 :]
                ax.text(0.98, 0.04, f"truth {ll_true:,.0f};  chains − truth: "
                        + ", ".join(f"{float(np.mean(post[ch]) - ll_true):+,.0f}" for ch in range(n_chains))
                        + " (mean after warm-up)", transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=7.5, color=INK2)
        else:
            if c == "png":
                ax.set_title(f"{name}  (primordial)")
                if "truth/png" in z:
                    ax.axhline(z["truth/png"][j], color=INK, lw=1.0, ls=(0, (4, 3)))
            else:
                ax.set_title(f"{name}  (catalogue {c})" if n_cat > 1 else name)
                if has_truth:
                    ax.axhline(z["truth/nuisance"][c, j], color=INK, lw=1.0, ls=(0, (4, 3)))
        ax.set_xlabel("sweep")
        ax.set_xlim(0, n_iter)
    for ax in axes.ravel()[len(panels) :]:
        ax.set_visible(False)
    # one legend for the whole figure, above the panels
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [Line2D([], [], color=SERIES[ch], lw=2, label=f"chain {ch}") for ch in range(min(n_chains, len(SERIES)))]
    handles.append(Patch(color=GRID, alpha=0.6, label="warm-up (not used)"))
    if has_truth:
        handles.append(Line2D([], [], color=INK, lw=1.0, ls=(0, (4, 3)), label="truth"))
    top = 1 - 0.45 / (2.5 * nrow)
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, top))
    out.append(run_dir / "traces.png")
    fig.savefig(out[-1])
    plt.close(fig)

    # -------------------------------------------------------------- field
    # white-noise samples: streamed to samples/ (current) or inside chains.npz
    if "fields" in z:
        fields = z["fields"]  # (chains, kept, n, n, n)
        sweeps = nw + cfg.sampler.thin * np.arange(fields.shape[1])
    elif (run_dir / "samples").is_dir():
        from .checkpoint import load_samples

        per_sweep = max(1, max_field_samples // n_chains)
        sweeps, fields = load_samples(run_dir, "field", max_samples=per_sweep, upto=n_iter)
    else:
        return out
    flat = fields.reshape(-1, *fields.shape[2:])
    sel_idx = np.linspace(0, len(flat) - 1, min(max_field_samples, len(flat))).astype(int)
    sel = flat[sel_idx]
    # sweep at which each selected field was collected, to pair it with the
    # nuisance parameters of the same state
    kept = fields.shape[1]
    sel_chain, sel_sweep = sel_idx // kept, np.asarray(sweeps)[sel_idx % kept]
    if model.png is not None:  # each sample with the PNG parameters of its sweep
        png_sel = jnp.asarray(z["trace/png"][sel_chain, sel_sweep])
        dens = jax.jit(jax.vmap(model.final_density))
    else:
        png_sel = jnp.zeros((len(sel), 1))
        dens = jax.jit(jax.vmap(lambda w, p: model.final_density(w)))
    rho = np.concatenate([
        np.asarray(dens(jnp.asarray(c, jnp.result_type(float)), p))
        for c, p in zip(_chunks(sel, 8), _chunks(png_sel, 8))
    ])
    rho_mean, rho_std = rho.mean(axis=0), rho.std(axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4))
    ax = axes[0, 0]
    pk = np.array([diag.power_spectrum(r - 1, box, nb)[1] for r in rho])
    k = diag.power_spectrum(rho[0] - 1, box, nb)[0]
    lo, mid, hi = np.percentile(pk, [16, 50, 84], axis=0)
    ax.fill_between(k, lo, hi, color=SERIES[0], alpha=0.25, lw=0)
    ax.plot(k, mid, color=SERIES[0], label="posterior samples (median, 68%)")
    ax.plot(k, diag.power_spectrum(rho_mean - 1, box, nb)[1], color=SERIES[1], label="posterior-mean field")
    if has_truth:
        ax.plot(k, diag.power_spectrum(z["truth/rho"] - 1, box, nb)[1], color=INK, lw=1.2,
                ls=(0, (4, 3)), label="truth")
    ax.set(yscale="log", xlabel="k  [h/Mpc]", ylabel="P(k)  [(Mpc/h)³]",
           title="Evolved density power spectrum")
    logx(ax)
    ax.legend(loc="lower left")

    ax = axes[0, 1]
    if has_truth:
        pairs = [("evolved density", SERIES[1], rho - 1, z["truth/rho"] - 1)]
        if "truth/white_noise" in z:  # not for data drawn from a simulation
            pairs.insert(0, ("initial conditions", SERIES[0], sel, z["truth/white_noise"]))
        for lab, col, samples, truth in pairs:
            r = np.array([diag.cross_correlation(s, truth, box, nb)[1] for s in samples])
            ax.fill_between(k, *np.percentile(r, [16, 84], axis=0), color=col, alpha=0.25, lw=0)
            ax.plot(k, np.median(r, axis=0), color=col, label=lab)
        ax.axhline(1.0, color=AXIS, lw=1.0)
        ax.set(ylim=(0, 1.05), xlabel="k  [h/Mpc]", ylabel="r(k)",
               title="Cross-correlation of samples with the truth")
        logx(ax)
        ax.legend(loc="lower left")
    else:
        ax.set_visible(False)

    ax = axes[0, 2]
    ratios = np.array([np.asarray(model.white_noise_power_ratio(jnp.asarray(s, jnp.result_type(float)))) for s in sel])
    good = np.asarray(model._ps["good"])
    idx = np.arange(good.sum())
    sigma = np.sqrt(np.asarray(model._ps["var"]))[good]
    ax.fill_between(idx, 1 - sigma, 1 + sigma, color=GRID, lw=0, label="prior ±1σ")
    ax.errorbar(idx, ratios.mean(axis=0)[good], ratios.std(axis=0)[good], fmt="o", ms=5,
                color=SERIES[0], ecolor=SERIES[0], elinewidth=1.2, label="posterior samples")
    ax.axhline(1.0, color=INK, lw=1.0, ls=(0, (4, 3)))
    ax.set(xlabel="|k| bin", ylabel="P_i / P̄", title="White-noise power in k-bins (should be flat)")
    ax.legend(loc="upper right")

    mid_z = rho.shape[-1] // 2
    ext = (0, box, 0, box)
    lg = lambda a: np.log10(np.maximum(a[:, :, mid_z], 1e-2))
    panels = [("Posterior mean  log₁₀ ρ", lg(rho_mean)), ("Posterior std of ρ", rho_std[:, :, mid_z])]
    if has_truth:
        panels.insert(0, ("Truth  log₁₀ ρ", lg(z["truth/rho"])))
    else:
        panels.insert(0, ("Galaxy counts (catalogue 0)", z["counts"][0][:, :, mid_z]))
    vmin = min(p[1].min() for p in panels[:2]) if has_truth else None
    vmax = max(p[1].max() for p in panels[:2]) if has_truth else None
    for i, (ax, (title, data)) in enumerate(zip(axes[1], panels)):
        shared = has_truth and i < 2
        image(ax, data, seq, ext, title.split("  ")[-1] if "ρ" in title else "N",
              vmin if shared else None, vmax if shared else None)
        ax.set(title=title, xlabel="x  [Mpc/h]", ylabel="y  [Mpc/h]")
    fig.tight_layout()
    out.append(run_dir / "field.png")
    fig.savefig(out[-1])
    plt.close(fig)

    out.append(run_dir / "diagnostics.png")
    _diagnostics_figure(plt, z, model, cfg, sel, rho, theta[sel_chain, sel_sweep], out[-1], png_sel)
    return out


def _warmup(ax, nw):
    ax.axvspan(0, nw, color=GRID, alpha=0.6, lw=0)


def _diagnostics_figure(plt, z, model, cfg, white, rho, theta_sel, path, png_sel=None):
    """Sampler health, mixing, Gaussianity of the inferred initial conditions
    and a posterior predictive check of the counts. ``white``, ``rho`` and
    ``theta_sel`` are matched posterior samples (field, density, nuisance)."""
    import jax.numpy as jnp
    from jax.scipy.special import ndtri

    from . import diagnostics as diag

    nw = cfg.sampler.n_warmup
    names = list(model.param_names)
    theta = z["trace/nuisance"]
    n_chains, n_iter, n_cat, _ = theta.shape
    has_truth = "truth/nuisance" in z
    has_hmc = "trace/field/accept_prob" in z
    it = np.arange(n_iter)
    chains = range(min(n_chains, len(SERIES)))
    dash = (0, (4, 3))

    fig, axes = plt.subplots(3, 3, figsize=(14, 11.2))

    # ---------------------------------------------------------- HMC health
    ax = axes[0, 0]
    if has_hmc:
        acc = z["trace/field/accept_prob"]
        win = max(5, n_iter // 40)
        kern = np.ones(win) / win
        for ch in chains:
            ax.plot(it[win - 1 :], np.convolve(acc[ch], kern, mode="valid"), color=SERIES[ch], lw=1.4,
                    label=f"chain {ch}")
        ax.axhline(cfg.sampler.hmc.target_accept, color=INK, lw=1.0, ls=dash, label="target")
        _warmup(ax, nw)
        ax.set(xlim=(0, n_iter), ylim=(0, 1.02), xlabel="sweep", ylabel="acceptance probability",
               title=f"HMC acceptance ({win}-sweep running mean)")
        ax.legend(loc="lower right", ncol=min(n_chains, 4) + 1)

        ax = axes[0, 1]
        eps = z["trace/field/step_size"]
        for ch in chains:
            ax.plot(it, eps[ch], color=SERIES[ch], lw=1.0)
        _warmup(ax, nw)
        ax.set(yscale="log", xlim=(0, n_iter), xlabel="sweep", ylabel="leapfrog step size",
               title="Step size: dual averaging, then frozen")

        ax = axes[0, 2]
        dh = z["trace/field/delta_energy"][:, nw:].ravel()
        dh = dh[np.isfinite(dh)]
        lo, hi = np.percentile(dh, [0.5, 99.5])
        ax.hist(np.clip(dh, lo, hi), 60, color=SERIES[0], alpha=0.85, lw=0)
        ax.axvline(0, color=INK, lw=1.0, ls=dash)
        ax.text(0.98, 0.95, f"E[exp(−ΔH)] = {np.mean(np.exp(-dh)):.3f}\n(1 in expectation for correct HMC)",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2.0), zorder=6)
        ax.set(xlabel="ΔH  (energy error of a trajectory)", ylabel="trajectories",
               title="Hamiltonian error after warm-up")
    else:
        for a in axes[0]:
            a.set_visible(False)

    # -------------------------------------------------------------- mixing
    post = {"log-likelihood": z["trace/log_likelihood"][:, nw:]}
    for c in range(n_cat):
        for j, name in enumerate(names):
            if model.sampled[j]:
                post[f"{name} (cat {c})" if n_cat > 1 else name] = theta[:, nw:, c, j]

    ax = axes[1, 0]
    n_post = post["log-likelihood"].shape[1]
    max_lag = max(10, min(200, n_post // 3))
    for (label, x), col in zip(list(post.items())[:4], SERIES):
        acf = np.mean([diag.autocorrelation(c) for c in x], axis=0)[:max_lag]
        ax.plot(np.arange(acf.size), acf, color=col, lw=1.6,  # shorter than max_lag for very short runs
                label=f"{label}   τ = {diag.integrated_autocorr_time(x):.0f}")
    ax.axhline(0, color=AXIS, lw=1.0)
    ax.set(xlim=(0, max_lag - 1), ylim=(-0.25, 1.02), xlabel="lag  [sweeps]", ylabel="autocorrelation",
           title="Autocorrelation after warm-up")
    ax.legend(loc="upper right")

    ax = axes[1, 1]
    labels = list(post)[::-1]
    ess = [diag.effective_sample_size(post[k]) for k in labels]
    rhat = [diag.gelman_rubin(post[k]) for k in labels]
    ypos = np.arange(len(labels))
    ax.barh(ypos, ess, height=0.62, color=SERIES[0])
    for y, e, r in zip(ypos, ess, rhat):
        ax.text(e, y, f"  R̂ {r:.2f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(ypos, labels, fontsize=7.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(ess) * 1.3)
    ax.set(xlabel=f"effective sample size  (of {n_post * n_chains} draws)", title="Effective sample size and split R̂")

    ax = axes[1, 2]
    sampled = [j for j in range(len(names)) if model.sampled[j]]
    if len(sampled) >= 2:
        ja, jb = sampled[:2]
        step = max(1, n_post // 400)
        for ch in chains:
            ax.scatter(theta[ch, nw::step, 0, ja], theta[ch, nw::step, 0, jb], s=9, color=SERIES[ch], alpha=0.6,
                       edgecolor="none", label=f"chain {ch}")
        if has_truth:
            t = z["truth/nuisance"][0]
            ax.scatter([t[ja]], [t[jb]], marker="+", s=160, color=INK, linewidth=2.0, label="truth", zorder=5)
        ax.set(xlabel=names[ja], ylabel=names[jb],
               title="Joint posterior of two nuisance parameters" + (" (cat 0)" if n_cat > 1 else ""))
        ax.legend(loc="best")
    else:
        ax.set_visible(False)

    # ----------------------------------------------- Gaussianity of the ICs
    ax = axes[2, 0]
    x = np.asarray(white[-min(10, len(white)) :], float).ravel()
    grid = np.linspace(-4.5, 4.5, 300)
    ax.hist(x, np.linspace(-4.5, 4.5, 61), density=True, color=SERIES[0], alpha=0.85, lw=0,
            label="posterior samples")
    ax.plot(grid, np.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi), color=INK, lw=1.2, ls=dash, label="N(0, 1) prior")
    m = diag.gaussianity_moments(x)
    ax.text(0.02, 0.95, f"skewness {m['skewness']:+.3f}\nexcess kurtosis {m['kurtosis']:+.3f}",
            transform=ax.transAxes, va="top", fontsize=8, color=INK2)
    ax.set(xlabel="white-noise amplitude", ylabel="density", title="One-point PDF of the inferred initial conditions")
    ax.legend(loc="upper right")

    ax = axes[2, 1]
    one = np.sort(np.asarray(white[-1], float).ravel())
    q = np.asarray(ndtri((np.arange(one.size) + 0.5) / one.size))
    thin = max(1, one.size // 2000)
    ax.plot([-4.5, 4.5], [-4.5, 4.5], color=INK, lw=1.0, ls=dash)
    ax.scatter(q[::thin], one[::thin], s=8, color=SERIES[0], edgecolor="none")
    ax.set(xlim=(-4.5, 4.5), ylim=(-4.5, 4.5), aspect="equal", xlabel="N(0, 1) quantile",
           ylabel="sample quantile", title="Q–Q plot, last posterior sample")

    # ------------------------------------------------ posterior predictive
    ax = axes[2, 2]
    rng = np.random.default_rng(0)
    resp = np.asarray(model.response[0])
    mask = resp > 0
    data = np.asarray(model.counts[0])[mask]
    draws = []
    for i, (r, th) in enumerate(zip(rho, theta_sel)):
        row = th[0]
        ops = None
        if model.nonlocal_bias is not None:  # non-local bias operators of this sample
            p_i = None if model.png is None else jnp.asarray(png_sel[i])
            ops = model.bias_operators(jnp.asarray(r), model.bias_potential(jnp.asarray(white[i])), p_i)
        lam = np.asarray(model.expected_counts(jnp.asarray(row), jnp.asarray(r), model.response[0], ops))
        p = {name: float(row[i]) for i, name in enumerate(names)}
        draws.append(model.likelihood.sample(rng, lam, r, p, model.floor)[mask])
    draws = np.asarray(draws)
    top = int(np.ceil(max(np.percentile(data, 99.9), np.percentile(draws, 99.9)))) + 1
    edges = np.arange(-0.5, top + 1.5) if top <= 60 else np.linspace(-0.5, top + 0.5, 61)
    centres = 0.5 * (edges[1:] + edges[:-1])
    h = np.array([np.histogram(np.minimum(d, edges[-1] - 0.5), edges)[0] for d in draws])
    lo, mid, hi = np.percentile(h, [2.5, 50, 97.5], axis=0)
    ax.fill_between(centres, np.maximum(lo, 0.5), np.maximum(hi, 0.5), step="mid", color=SERIES[0], alpha=0.3,
                    lw=0, label="posterior predictive (95%)")
    ax.step(centres, np.maximum(mid, 0.5), where="mid", color=SERIES[0], lw=1.4)
    hd = np.histogram(np.minimum(data, edges[-1] - 0.5), edges)[0]
    ax.scatter(centres[hd > 0], hd[hd > 0], s=14, color=INK, zorder=5, label="observed counts")
    mu, sd = draws.mean(axis=0), draws.std(axis=0)
    ok = sd > 0
    chi2 = np.mean(((data - mu)[ok] / sd[ok]) ** 2)
    ax.text(0.98, 0.72, f"mean squared Pearson residual {chi2:.2f}\n(below 1 is expected: the field\nis fitted to these same counts)", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=INK2,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2.0), zorder=6)
    ax.set(yscale="log", xlabel="galaxies per voxel" + ("  (catalogue 0)" if n_cat > 1 else ""),
           ylabel="voxels", title="Posterior predictive check of the counts")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
