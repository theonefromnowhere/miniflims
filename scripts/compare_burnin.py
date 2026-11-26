"""Burn-in of several runs side by side: traces of f_NL, omega, alpha, the
log-likelihood and the white-noise variance (from progress.log), and f_NL
against var(x).

    python scripts/compare_burnin.py runs/png_null_w20 runs/png_feature_w20 -o runs/png_null_w20/burnin_vs_w20.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from miniflims.checkpoint import load_checkpoint
from miniflims.plotting import INK, MUTED, SERIES, _mpl


def truth_loglike(run_dir: Path):
    """log-likelihood at the mock truth (regenerated from the run's config)."""
    from miniflims.config import load_config
    from miniflims.data import load_dataset
    from miniflims.model import FieldModel
    from miniflims.pipeline import set_precision
    from miniflims.plotting import truth_log_likelihood

    cfg = load_config(run_dir / "config.yaml")
    if cfg.data.source != "mock":
        return None
    set_precision(cfg.precision)
    data = load_dataset(cfg)
    model = FieldModel(cfg, data.response, data.counts)
    return truth_log_likelihood(model, {f"truth/{k}": v for k, v in data.truth.items()})


def load(run_dir: Path):
    """traces (chains, sweeps) up to the last sweep done, var(x) from progress.log, truth."""
    # the checkpoint is newer than chains.npz while a finished run is extended
    loaded = load_checkpoint(run_dir)
    tr, n = (loaded[0].trace, loaded[0].iteration) if loaded is not None else (None, -1)
    if (run_dir / "chains.npz").exists():
        with np.load(run_dir / "chains.npz") as z:
            if z["trace/log_likelihood"].shape[1] > n:
                tr = {k[6:]: z[k] for k in z.files if k.startswith("trace/")}
                n = tr["log_likelihood"].shape[1]
    import yaml

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    truth = cfg["data"]["mock"]["truth"]
    sweeps, var = [], []
    for line in (run_dir / "progress.log").read_text().splitlines():
        m = re.search(r"sweep +(\d+)/.*var\(x\) ([\d./]+)", line)
        if m:
            sweeps.append(int(m.group(1)))
            var.append([float(v) for v in m.group(2).split("/")])
    return dict(n=n, ll_true=truth_loglike(run_dir), f_nl=tr["png"][:, :n, 0], omega=tr["png"][:, :n, 1] if tr["png"].shape[-1] > 1 else np.full_like(tr["png"][:, :n, 0], np.nan), alpha=tr["nuisance"][:, :n, 0, 1],
                logl=tr["log_likelihood"][:, :n], var_sweeps=np.array(sweeps), var=np.array(var), truth=truth,
                n_warmup=cfg["sampler"]["n_warmup"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    runs = [(Path(r).name, load(Path(r))) for r in args.runs]
    plt, _, _ = _mpl()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    panels = [("f_NL", "f_nl", "f_NL"), ("ω  [Mpc/h]", "omega", "omega"), ("bias α", "alpha", "alpha"),
              ("log-likelihood", "logl", None)]
    for i, (name, d) in enumerate(runs):
        col = SERIES[i]
        it = np.arange(d["n"])
        for ax, (title, key, tkey) in zip(axes.ravel(), panels):
            for c in range(d[key].shape[0]):
                ax.plot(it, d[key][c], color=col, lw=1.1, ls="-" if c == 0 else (0, (1, 1.5)),
                        label=f"{name}, chain {c + 1}")
            if tkey and tkey in d["truth"]:
                ax.axhline(d["truth"][tkey], color=col, lw=1.0, ls=(0, (4, 3)), alpha=0.9)
            if key == "logl" and d["ll_true"] is not None:
                ax.axhline(d["ll_true"], color=col, lw=1.0, ls=(0, (4, 3)), alpha=0.9)
            ax.set(title=title, xlabel="sweep")
        ax = axes[1, 1]
        for c in range(d["var"].shape[1] if d["var"].size else 0):
            ax.plot(d["var_sweeps"], d["var"][:, c], marker="o", ms=3.5, color=col, lw=1.1,
                    ls="-" if c == 0 else (0, (1, 1.5)), label=f"{name}, chain {c + 1}")
        ax = axes[1, 2]
        if d["var"].size:
            ok = (d["var_sweeps"] >= 1) & (d["var_sweeps"] <= d["n"])
            s = d["var_sweeps"][ok]
            for c in range(d["var"].shape[1]):
                ax.plot(d["var"][ok, c], d["f_nl"][c, s - 1], "o", ms=4, color=col, alpha=0.75 if c == 0 else 0.45,
                        label=f"{name}, chain {c + 1}")
            if "f_NL" in d["truth"]:
                ax.axhline(d["truth"]["f_NL"], color=col, lw=1.0, ls=(0, (4, 3)))
    ll = np.concatenate([d["logl"][:, d["n"] // 3:].ravel() for _, d in runs])
    tops = [d["ll_true"] for _, d in runs if d["ll_true"] is not None] + [ll.max()]
    axes[1, 0].set_ylim(np.percentile(ll, 0.5) - 500, max(tops) + 200)
    axes[1, 1].axhline(1.0, color=INK, lw=0.9, ls=(0, (4, 3)))
    axes[1, 1].set(title="White-noise variance var(x) (1 in the typical set)", xlabel="sweep")
    axes[1, 2].set(title="f_NL against var(x) (every 20 sweeps)", xlabel="var(x)", ylabel="f_NL")
    for ax in axes.ravel():
        nw = runs[0][1]["n_warmup"]
        if ax is not axes[1, 2]:
            ax.axvspan(0, nw, color=MUTED, alpha=0.08, lw=0)
    axes[0, 0].legend(loc="lower right", fontsize=7.5)
    axes[1, 1].legend(loc="lower right", fontsize=7.5)
    fig.suptitle("Burn-in: " + " vs ".join(n for n, _ in runs) + "  (dashed: each run's truth, same colour; "
                 "shaded: warm-up)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
