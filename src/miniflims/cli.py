"""Command line.

    miniflims run -c config.yaml -o sampler.n_samples=500 [--fresh]
    miniflims status runs/mine
    miniflims watch runs/mine          (progress.log + progress.png, one entry per checkpoint)
    miniflims plot runs/mine

Exit codes of ``run``: 0 finished, 2 configuration error, 3 stopped by a
signal after writing a checkpoint (run the same command again to resume).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ConfigError, dump_yaml, load_config


def _components() -> dict:
    from .bias import BIAS_MODELS
    from .blocks import BLOCKS
    from .data import SELECTIONS
    from .forward import FORWARD_MODELS
    from .likelihood import LIKELIHOODS
    from .nonlocal_bias import NONLOCAL_TERMS
    from .samplers import INTEGRATORS
    from .transfer import TRANSFER_FUNCTIONS

    regs = (TRANSFER_FUNCTIONS, FORWARD_MODELS, BIAS_MODELS, NONLOCAL_TERMS, LIKELIHOODS, SELECTIONS,
            INTEGRATORS, BLOCKS)
    return {r.kind: r.names() for r in regs}


def _status(run_dir: str) -> int:
    from .checkpoint import read_meta, sample_sweeps

    path = Path(run_dir)
    meta = read_meta(path)
    if meta is None:
        done = (path / "summary.json").exists()
        print(f"{run_dir}: {'finished (no checkpoint kept)' if done else 'no checkpoint yet'}")
        return 0 if done else 1
    it, total = meta["iteration"], meta["total"]
    nw = meta["config"]["sampler"]["n_warmup"]
    wall = meta.get("wall_time_s", 0.0)
    rate = it / wall if wall > 0 else float("nan")
    eta = (total - it) / rate if rate == rate and rate > 0 else float("nan")
    phase = "finished" if it >= total else ("warm-up" if it < nw else "sampling")
    print(f"{run_dir}: sweep {it}/{total} ({100 * it / total:.1f}%), {phase}")
    print(f"  last checkpoint  {meta['written']}")
    print(f"  compute time     {wall / 3600:.2f} h  ({rate * 3600:.0f} sweeps/h)")
    if it < total and eta == eta:
        print(f"  remaining        ~{eta / 3600:.2f} h of compute")
    if (path / "samples").is_dir():
        print(f"  field samples    {len(sample_sweeps(path))}")
    print(f"  summary.json     {'yes' if (path / 'summary.json').exists() else 'not yet'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="miniflims", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in (
        ("run", "run or resume the inference pipeline"),
        ("show-config", "print the fully resolved configuration"),
    ):
        p = sub.add_parser(name, help=help_)
        p.add_argument("-c", "--config", help="YAML config file")
        p.add_argument(
            "-o", "--override", action="append", default=[], metavar="KEY=VALUE",
            help="dotted override, e.g. sampler.hmc.n_steps=30 (repeatable)",
        )
        if name == "run":
            p.add_argument("--fresh", action="store_true",
                           help="discard the checkpoint and samples in output.dir and start over")
    sub.add_parser("components", help="list registered components")
    pp = sub.add_parser("plot", help="write traces.png, field.png and diagnostics.png for a finished run")
    pp.add_argument("run_dir")
    ps = sub.add_parser("status", help="progress of a (running or stopped) run from its checkpoint")
    ps.add_argument("run_dir")
    pw = sub.add_parser("watch", help="follow a run: progress.log and progress.png in the run directory")
    pw.add_argument("run_dir")
    pw.add_argument("--interval", type=float, default=30.0, help="seconds between checks")
    pw.add_argument("--once", action="store_true", help="write the current state and exit")
    pw.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "plot":
        from .plotting import plot_run

        for path in plot_run(args.run_dir):
            print(path)
        return 0

    if args.cmd == "status":
        return _status(args.run_dir)

    if args.cmd == "watch":
        from .watch import watch

        return watch(args.run_dir, args.interval, args.once, plot=not args.no_plot)

    if args.cmd == "components":
        for kind, names in _components().items():
            print(f"{kind}: {', '.join(names)}")
        return 0

    try:
        cfg = load_config(args.config, args.override)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    if args.cmd == "show-config":
        print(dump_yaml(cfg), end="")
        return 0

    from .pipeline import EXIT_INTERRUPTED, run

    try:
        result = run(cfg, fresh=args.fresh)
    except ConfigError as e:  # e.g. resuming with an incompatible configuration
        print(f"config error: {e}", file=sys.stderr)
        return 2
    if not result.completed:
        print(f"stopped at sweep {result.gibbs.n_done}; checkpoint in {cfg.output.dir}. "
              "Run the same command again to resume.")
        return EXIT_INTERRUPTED
    print(json.dumps(result.summary, indent=2))
    if cfg.output.dir:
        print(f"written to {cfg.output.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
