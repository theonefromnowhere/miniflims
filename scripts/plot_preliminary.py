"""Preliminary results of a running (or stopped) run, from its latest
checkpoint: what `miniflims run` and `miniflims plot` would write at the end,
computed over the sweeps done so far.

    .venv-run/bin/python -u scripts/plot_preliminary.py runs/png_feature_w20_10k [--max-field-samples 40]

Writes <run_dir>/prelim_<sweep>/: summary.json, chains.npz, config.yaml,
traces.png, field.png, diagnostics.png. Field statistics use whatever samples
are in <run_dir>/samples up to that sweep (a subset is enough, e.g. every
100th sweep copied from the cluster); the folder links to them.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

from miniflims.checkpoint import iter_samples, load_checkpoint, sample_sweeps
from miniflims.config import load_config
from miniflims.data import load_dataset
from miniflims.model import FieldModel
from miniflims.pipeline import RunResult, field_statistics, save, set_precision, summarize
from miniflims.plotting import plot_run
from miniflims.samplers.gibbs import GibbsResult


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--max-field-samples", type=int, default=40)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ckpt, meta = load_checkpoint(run_dir)
    if ckpt is None:
        raise SystemExit(f"{run_dir}: no checkpoint")
    it = ckpt.iteration
    out = run_dir / f"prelim_{it:05d}"
    out.mkdir(exist_ok=True)

    cfg = load_config(run_dir / "config.yaml")
    cfg = dataclasses.replace(cfg, output=dataclasses.replace(cfg.output, dir=str(out)))
    set_precision(cfg.precision)
    data = load_dataset(cfg)
    model = FieldModel(cfg, data.response, data.counts)
    nw = cfg.sampler.n_warmup
    if it <= nw:
        raise SystemExit(f"{run_dir}: still in warm-up (sweep {it}/{nw})")

    res = GibbsResult(ckpt.state, ckpt.block_states, ckpt.trace, None, nw,
                      float(meta.get("wall_time_s", 0.0)), it, completed=False)
    sweeps = sample_sweeps(run_dir, "field", upto=it)
    stats = field_statistics(model, iter_samples(run_dir, "field", sweeps), ckpt.trace.get("png"))
    summary = summarize(cfg, model, data, res, stats, res.wall_time)
    summary["preliminary"] = {"sweep": it, "total": meta["total"], "checkpoint_written": meta["written"],
                              "field_sample_sweeps": [int(s) for s in sweeps]}
    save(RunResult(cfg, model, data, res, summary, None,
                   stats.mean_density if stats else None, completed=False, run_dir=out))

    link = out / "samples"
    if not link.exists():
        os.symlink(os.path.relpath(run_dir / "samples", out), link)
    for p in plot_run(out, max_field_samples=args.max_field_samples):
        print(p)
    print(out / "summary.json")


if __name__ == "__main__":
    main()
