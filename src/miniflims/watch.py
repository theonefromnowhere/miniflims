"""Follow a running (or finished) miniflims run through its checkpoints.

    miniflims watch runs/mine [--interval 30] [--once]

For every new checkpoint it appends one line to <run_dir>/progress.log and
redraws <run_dir>/progress.png (traces of the log-likelihood, HMC acceptance
and step size, and every sampled parameter, with the mock truth). It only
reads the run directory, so it can be started, stopped and restarted at any
time, also on a login node for a Slurm job.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from .checkpoint import load_checkpoint, read_meta


def _sampled(meta: dict) -> list[tuple[str, str, int, int]]:
    """(label, trace key, row, column) of every sampled parameter."""
    from .bias import BIAS_MODELS
    from .likelihood import LIKELIHOODS
    from .nonlocal_bias import term_specs
    from .png import PARAM_DEFAULTS, TEMPLATES

    cfg = meta["config"]
    gal = cfg["galaxies"]
    names = (["nbar"] + [s.name for s in BIAS_MODELS.get(gal["bias"]["model"]).param_specs]
             + [s.name for s in term_specs(gal["bias"].get("non_local", {}).get("terms", []))]
             + [s.name for s in LIKELIHOODS.get(gal["likelihood"]["model"]).param_specs])
    over = {"nbar": gal["nbar"], **gal["bias"]["params"], **gal["likelihood"]["params"]}
    out = []
    for c in range(gal["n_catalogues"]):
        for j, n in enumerate(names):
            if (over.get(n) or {}).get("sample", True) is not False:
                out.append((n if gal["n_catalogues"] == 1 else f"{n}[{c}]", "nuisance", c, j))
    png = cfg.get("png", {})
    if png.get("enabled"):
        pnames = ["f_NL"] + list(TEMPLATES[png["template"]].shape_params)
        for j, n in enumerate(pnames):
            sample = (png["params"].get(n) or {}).get("sample")
            if sample is None:
                sample = PARAM_DEFAULTS[n].sample
            if sample:
                out.append((n, "png", -1, j))
    return out


def _values(trace: dict, key: str, row: int, col: int) -> np.ndarray:
    """(chains, sweeps) array of one parameter."""
    t = trace[key]
    return t[:, :, col] if row < 0 else t[:, :, row, col]


def _line(meta: dict, trace: dict, params, last: int, block: int, field=None) -> str:
    it, total = meta["iteration"], meta["total"]
    nw = meta["config"]["sampler"]["n_warmup"]
    wall = meta.get("wall_time_s", 0.0)
    rate = it / wall * 3600 if wall > 0 else float("nan")
    eta = (total - it) / rate if rate == rate and rate > 0 else float("nan")
    phase = "done" if it >= total else ("warm-up" if it < nw else "sampling")
    sl = slice(max(0, it - block), it)
    parts = [f"{meta['written'][:19]}", f"sweep {it:5d}/{total}", f"{phase:8s}",
             f"{rate:5.0f} sw/h", f"ETA {eta:5.2f} h" if eta == eta else "ETA    - "]
    if "field/accept_prob" in trace:
        parts.append(f"acc {trace['field/accept_prob'][:, sl].mean():.2f}")
        parts.append(f"eps {trace['field/step_size'][:, sl].mean():.4f}")
    if "log_likelihood" in trace:
        ll = trace["log_likelihood"][:, it - 1]
        parts.append("logL " + "/".join(f"{v:.0f}" for v in ll))
    if field is not None:  # white-noise variance: 1 in the typical set, < 1 while burning in from a small start
        parts.append("var(x) " + "/".join(f"{v:.3f}" for v in field.reshape(field.shape[0], -1).var(axis=1)))
    for label, key, row, col in params:
        v = _values(trace, key, row, col)[:, it - 1]
        parts.append(f"{label} " + "/".join(f"{x:.4g}" for x in v))
    return "  ".join(parts)


def _truths(meta: dict) -> dict:
    truth = meta["config"]["data"]["mock"]["truth"] if meta["config"]["data"]["source"] == "mock" else {}
    return {k: (v[0] if isinstance(v, list) else v) for k, v in truth.items()}


def _plot(run_dir: Path, meta: dict, trace: dict, params):
    from .plotting import GRID, INK, SERIES, _mpl

    plt, _, _ = _mpl()
    nw = meta["config"]["sampler"]["n_warmup"]
    truth = _truths(meta)
    panels = [("log-likelihood", trace["log_likelihood"], None)]
    if "field/accept_prob" in trace:
        panels.append(("HMC acceptance", trace["field/accept_prob"], meta["config"]["sampler"]["hmc"]["target_accept"]))
        panels.append(("HMC step size", trace["field/step_size"], None))
    for label, key, row, col in params:
        panels.append((label, _values(trace, key, row, col), truth.get(label.split("[")[0])))
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 2.6 * nrow + 0.5), squeeze=False)
    it = meta["iteration"]
    x = np.arange(it)
    for ax, (title, y, ref) in zip(axes.ravel(), panels):
        for c in range(y.shape[0]):
            ax.plot(x, y[c, :it], color=SERIES[c % len(SERIES)], lw=1.1)
        ax.axvspan(0, min(nw, it), color=GRID, alpha=0.6, lw=0)
        if ref is not None:
            ax.axhline(ref, color=INK, lw=1.0, ls=(0, (4, 3)))
        if title == "log-likelihood" and it > 10:
            lo = np.percentile(y[:, it // 2:it], 1)
            ax.set_ylim(lo - 0.1 * abs(y[:, :it].max() - lo), None)
        ax.set(title=title, xlabel="sweep", xlim=(0, max(it, 1)))
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(f"{run_dir.name}: sweep {it}/{meta['total']}  (shaded: warm-up; dashed: truth / target)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    tmp = run_dir / ".progress.png.tmp.png"
    fig.savefig(tmp)
    plt.close(fig)
    tmp.replace(run_dir / "progress.png")


def watch(run_dir, interval: float = 30.0, once: bool = False, block: Optional[int] = None,
          plot: bool = True, echo=print) -> int:
    run_dir = Path(run_dir)
    log = run_dir / "progress.log"
    seen = None
    if log.exists():  # resume without repeating the last line
        lines = [l for l in log.read_text().splitlines() if l and not l.startswith("#")]
        if lines and "sweep" in lines[-1]:
            seen = int(lines[-1].split("sweep")[1].split("/")[0])
    while True:
        meta = read_meta(run_dir)
        if meta is not None and meta["iteration"] != seen:
            ckpt, meta = load_checkpoint(run_dir)
            params = _sampled(meta)
            blk = block or meta["config"]["sampler"]["checkpoint"]["every"] or 20
            if not log.exists():
                with open(log, "w") as f:
                    f.write(f"# {run_dir}: per-checkpoint progress (chain values separated by '/'; "
                            f"acc and eps averaged over the last {blk} sweeps)\n")
                    f.write("# truth: " + ", ".join(f"{k} = {v}" for k, v in _truths(meta).items()) + "\n")
            line = _line(meta, ckpt.trace, params, ckpt.iteration, blk, ckpt.state.get("field"))
            with open(log, "a") as f:
                f.write(line + "\n")
            echo(line)
            if plot:
                _plot(run_dir, meta, ckpt.trace, params)
            seen = meta["iteration"]
            if meta["iteration"] >= meta["total"]:
                return 0
        if once:
            return 0 if meta is not None else 1
        time.sleep(interval)
