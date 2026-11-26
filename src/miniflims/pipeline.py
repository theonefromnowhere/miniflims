"""End-to-end run: config -> data -> model -> Gibbs sampling -> diagnostics.

With ``output.dir`` set, a run writes a restart file every
``sampler.checkpoint.every`` sweeps and resumes from it automatically.
SIGTERM, SIGUSR1 and SIGINT stop the run after the current sweep with a
checkpoint (a second SIGINT aborts immediately); ``run`` then returns with
``completed=False`` and the CLI exits with code 3.
"""
from __future__ import annotations

import json
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from . import diagnostics as diag
from .blocks import BLOCKS
from .checkpoint import (
    check_resumable,
    clear_run,
    iter_samples,
    load_checkpoint,
    sample_sweeps,
    sample_writer,
    save_checkpoint,
)
from .config import Config, ConfigError, dump_yaml
from .data import Dataset, load_dataset
from .model import HYPER_DEFAULTS, FieldModel
from .samplers import GibbsResult, gated_block, make_hyper_fn, run_gibbs

EXIT_INTERRUPTED = 3  # CLI exit code: stopped early, checkpoint written


@dataclass
class RunResult:
    config: Config
    model: FieldModel
    data: Dataset
    gibbs: GibbsResult
    summary: Optional[dict]  # None if the run was interrupted
    fields: Optional[np.ndarray]  # in-memory white-noise samples (chains, kept, n, n, n)
    mean_density: Optional[np.ndarray]  # posterior mean of rho on the mesh
    completed: bool = True
    run_dir: Optional[Path] = None  # output.dir; streamed samples are in run_dir/samples


def set_precision(precision: str) -> None:
    if precision not in ("float32", "float64"):
        raise ValueError("precision must be float32 or float64")
    jax.config.update("jax_enable_x64", precision == "float64")


def initial_state(model: FieldModel, cfg: Config, key) -> dict:
    """Over-dispersed starting points: low-amplitude random white noise (as
    BORG does) and, optionally, scattered nuisance parameters."""
    s = cfg.sampler
    k_f, k_n, k_p = jax.random.split(key, 3)
    shape = (s.n_chains,) + (model.grid.n,) * 3
    field = s.init.field_scale * jax.random.normal(k_f, shape)
    nuisance = jax.vmap(lambda k: model.init_nuisance(k, s.init.nuisance_jitter))(
        jax.random.split(k_n, s.n_chains)
    )
    state = {"field": field, "nuisance": nuisance}
    if model.png is not None:  # one row per chain, in the slice-block layout
        state["png"] = jax.vmap(lambda k: model.png.init(k, s.init.nuisance_jitter)[None])(
            jax.random.split(k_p, s.n_chains)
        )
    return state


def _log_line(i, info) -> str:
    parts = []
    if "field/accept_prob" in info:
        parts.append(f"acc={info['field/accept_prob'].mean():.2f}")
        parts.append(f"eps={info['field/step_size'].mean():.4f}")
    if "log_likelihood" in info:
        parts.append(f"logL={info['log_likelihood'].mean():.1f}")
    if "png" in info:  # chain means of the PNG parameters
        parts.append("png=" + ",".join(f"{v:.4g}" for v in info["png"].mean(axis=0)))
    return "  ".join(parts)


def _tee(logger: Callable, path: Path) -> Callable:
    """Also append every log line, time-stamped, to ``path``."""
    def log(msg: str) -> None:
        logger(msg)
        with open(path, "a") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S  ") + msg + "\n")
    return log


# --------------------------------------------------------------------------- #
# Posterior summaries of the field, one sample at a time
# --------------------------------------------------------------------------- #
@dataclass
class FieldStats:
    mean_white: np.ndarray  # posterior mean of the white noise
    last_white: np.ndarray  # last kept sample of every chain, (chains, n, n, n)
    mean_density: np.ndarray  # posterior mean of rho
    n_kept: int  # kept sweeps (each with all chains)


def field_statistics(model: FieldModel, samples: Iterable[tuple[int, np.ndarray]],
                     png_trace: Optional[np.ndarray] = None) -> Optional[FieldStats]:
    """``samples`` yields (sweep, white noise of shape (chains, n, n, n)).
    With PNG, ``png_trace`` (chains, sweeps, n_png) gives the PNG parameters
    of the same sweep, which the density needs."""
    dtype = jnp.result_type(float)
    if model.png is not None:
        if png_trace is None:
            raise ValueError("png.enabled: field_statistics needs the PNG trace")
        dens = jax.jit(jax.vmap(model.final_density))
    else:
        dens = jax.jit(jax.vmap(lambda w, p: model.final_density(w)))
    sum_w = sum_rho = last = None
    n = 0
    for sweep, w in samples:
        w = np.asarray(w)
        p = jnp.asarray(png_trace[:, sweep]) if png_trace is not None else jnp.zeros((w.shape[0], 1))
        rho = np.asarray(dens(jnp.asarray(w, dtype), p))
        sum_w = w.sum(axis=0, dtype=np.float64) if sum_w is None else sum_w + w.sum(axis=0, dtype=np.float64)
        sum_rho = rho.sum(axis=0) if sum_rho is None else sum_rho + rho.sum(axis=0)
        last, n = w, n + w.shape[0]
    if n == 0:
        return None
    return FieldStats(sum_w / n, last, sum_rho / n, n // last.shape[0])


def apply_gates(blocks, gates: dict, logger: Callable = print) -> list:
    """Wrap the blocks named in ``sampler.gates`` so that each acts only once
    its condition holds (per chain, checked every sweep)."""
    names = [b.name for b in blocks]
    unknown = sorted(set(gates) - set(names))
    if unknown:
        raise ConfigError(f"sampler.gates: {unknown} not in sampler.blocks {names}")
    out = []
    for b in blocks:
        g = gates.get(b.name)
        if g is None or g.min_field_variance is None:
            out.append(b)
            continue
        if b.name == "field":
            raise ConfigError("sampler.gates.field: the field block cannot wait for the field variance")
        thr = float(g.min_field_variance)
        logger(f"block '{b.name}' acts once the white-noise variance <x^2> >= {thr:g} (per chain)")
        out.append(gated_block(b, lambda st, thr=thr: jnp.mean(st["field"] ** 2) >= thr))
    return out


def summarize(cfg: Config, model: FieldModel, data: Dataset, res: GibbsResult,
              stats: Optional[FieldStats], wall_time: float) -> dict:
    nw = res.n_warmup
    tr = {k: v[:, nw:] for k, v in res.trace.items()}
    out: dict = {"wall_time_s": wall_time, "n_chains": cfg.sampler.n_chains, "n_sweeps": res.n_done}

    if "field/accept_prob" in tr:
        out["hmc"] = {
            "accept_rate": float(tr["field/accepted"].mean()),
            "mean_accept_prob": float(tr["field/accept_prob"].mean()),
            "step_size": float(tr["field/step_size"].mean()),
        }
    if "log_likelihood" in tr:
        ll = tr["log_likelihood"]
        out["log_likelihood"] = {
            "mean": float(ll.mean()),
            "rhat": diag.gelman_rubin(ll),
            "ess": diag.effective_sample_size(ll),
        }
    if "nuisance" in tr:
        th = tr["nuisance"]  # (chains, iters, cats, params)
        truth = data.truth.get("nuisance") if data.truth else None
        nuis = {}
        for c in range(th.shape[2]):
            for j, name in enumerate(model.param_names):
                if not model.sampled[j]:
                    continue
                x = th[:, :, c, j]
                entry = {
                    "mean": float(x.mean()),
                    "std": float(x.std()),
                    "rhat": diag.gelman_rubin(x),
                    "ess": diag.effective_sample_size(x),
                }
                if truth is not None:
                    entry["truth"] = float(truth[c, j])
                nuis[f"cat{c}/{name}"] = entry
        out["nuisance"] = nuis
    if "png" in tr:
        pp = tr["png"]  # (chains, iters, n_png)
        truth = data.truth.get("png") if data.truth else None
        png = {}
        for j, name in enumerate(model.png.names):
            x = pp[:, :, j]
            entry = {"mean": float(x.mean()), "std": float(x.std()), "sampled": bool(model.png.sampled[j])}
            if model.png.sampled[j] and "png/n_evals" in res.trace:
                entry.update(rhat=diag.gelman_rubin(x), ess=diag.effective_sample_size(x))
            if truth is not None:
                entry["truth"] = float(truth[j])
            png[name] = entry
        out["png"] = png

    nb = cfg.output.n_k_bins
    if stats is not None:
        out["n_field_samples"] = stats.n_kept
        out["white_noise_moments"] = diag.gaussianity_moments(stats.last_white)
        if data.truth is not None and "white_noise" in data.truth:
            k, r = diag.cross_correlation(stats.mean_white, data.truth["white_noise"], model.grid.box, nb)
            out["ic_cross_correlation"] = {"k": k.tolist(), "r": r.tolist()}
        if data.truth is not None and "rho" in data.truth:
            k, r = diag.cross_correlation(
                stats.mean_density - 1.0, data.truth["rho"] - 1.0, model.mesh.box, nb
            )
            out["density_cross_correlation"] = {"k": k.tolist(), "r": r.tolist()}
    return out


# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #
class _StopOnSignal:
    """Turns SIGTERM / SIGUSR1 / SIGINT into a request to stop after the
    current sweep. A second SIGINT raises KeyboardInterrupt."""

    SIGNALS = ("SIGTERM", "SIGUSR1", "SIGINT")

    def __init__(self, logger: Callable):
        self.requested = False
        self.logger = logger
        self._previous = {}

    def _handler(self, signum, frame):
        if self.requested and signum == signal.SIGINT:
            raise KeyboardInterrupt
        self.requested = True
        self.logger(f"received {signal.Signals(signum).name}: checkpointing after the current sweep")

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            for name in self.SIGNALS:
                sig = getattr(signal, name, None)
                if sig is not None:
                    self._previous[sig] = signal.signal(sig, self._handler)
        return self

    def __exit__(self, *exc):
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)
        return False

    def __call__(self) -> bool:
        return self.requested


# --------------------------------------------------------------------------- #
def run(cfg: Config, logger: Callable = print, fresh: bool = False) -> RunResult:
    """Run (or resume) the inference described by ``cfg``.

    fresh   discard any checkpoint and streamed samples in ``output.dir``
            (same as ``sampler.checkpoint.resume: false``).
    """
    set_precision(cfg.precision)
    s = cfg.sampler
    total = s.n_warmup + s.n_samples
    run_dir = Path(cfg.output.dir) if cfg.output.dir is not None else None

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        logger = _tee(logger, run_dir / "run.log")
    data = load_dataset(cfg)
    model = FieldModel(cfg, data.response, data.counts)
    if cfg.output.log_every:
        logger(
            f"grid {model.grid.n}^3, mesh {model.mesh.n}^3, box {model.grid.box:g} Mpc/h, "
            f"{model.n_catalogues} catalogue(s), {int(data.counts.sum())} galaxies, "
            f"nuisance: {model.param_names}"
        )

    resume, previous_wall = None, 0.0
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        if fresh or not s.checkpoint.resume:
            clear_run(run_dir)
        else:
            loaded = load_checkpoint(run_dir)
            if loaded is not None:
                resume, meta = loaded
                check_resumable(meta, cfg)
                previous_wall = float(meta.get("wall_time_s", 0.0))
                note = " (latest was unreadable: used checkpoint.prev.npz)" if meta.get("recovered_from_previous") else ""
                logger(f"resuming from sweep {resume.iteration}/{total}, checkpoint of {meta['written']}{note}")
        (run_dir / "config.yaml").write_text(dump_yaml(cfg))

    key = jax.random.PRNGKey(s.seed)
    k_init, k_run = jax.random.split(key)
    state = initial_state(model, cfg, k_init)
    blocks = apply_gates([BLOCKS.get(name)(model, cfg) for name in s.blocks], s.gates, logger)
    if model.png is not None and "png" not in s.blocks:
        logger(f"png.enabled without a 'png' Gibbs block: PNG parameters fixed at {model.png.default().tolist()}")
    hyper_fn = make_hyper_fn(s.schedules, model.hyper_defaults, s.n_warmup)

    stream = run_dir is not None and cfg.output.save_fields
    t_session = time.perf_counter()

    def on_checkpoint(ckpt):
        save_checkpoint(
            run_dir, ckpt, cfg,
            extra={"wall_time_s": previous_wall + time.perf_counter() - t_session},
            keep_previous=s.checkpoint.keep_previous,
        )

    with _StopOnSignal(logger) as stop:
        res = run_gibbs(
            k_run,
            blocks,
            state,
            n_warmup=s.n_warmup,
            n_samples=s.n_samples,
            hyper_fn=hyper_fn,
            monitor=model.monitor,
            collect=lambda st: {"field": st["field"].astype(jnp.float32)},
            thin=s.thin,
            log_every=cfg.output.log_every,
            log_fn=_log_line,
            logger=logger,
            resume=resume,
            checkpoint_every=s.checkpoint.every if run_dir is not None else 0,
            on_checkpoint=on_checkpoint if run_dir is not None else None,
            sample_sink=sample_writer(run_dir) if stream else None,
            should_stop=stop if run_dir is not None else None,
            progress=cfg.output.progress,
            progress_interval=cfg.output.progress_interval,
        )
    wall = previous_wall + time.perf_counter() - t_session

    if not res.completed:
        return RunResult(cfg, model, data, res, None, None, None, completed=False, run_dir=run_dir)

    if stream:
        samples = iter_samples(run_dir, "field", sample_sweeps(run_dir, "field", upto=total))
        fields = None
    else:
        fields = res.samples["field"] if res.samples else None
        samples = (
            ((s.n_warmup + s.thin * j, fields[:, j]) for j in range(fields.shape[1])) if fields is not None else ()
        )
    stats = field_statistics(model, samples, res.trace.get("png"))

    summary = summarize(cfg, model, data, res, stats, wall)
    result = RunResult(cfg, model, data, res, summary, fields,
                       stats.mean_density if stats else None, completed=True, run_dir=run_dir)
    if run_dir is not None:
        save(result)
    return result


def save(result: RunResult) -> Path:
    cfg = result.config
    out = Path(cfg.output.dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(dump_yaml(cfg))
    (out / "summary.json").write_text(json.dumps(result.summary, indent=2))

    arrays = {f"trace/{k}": v for k, v in result.gibbs.trace.items()}
    arrays["param_names"] = np.array(result.model.param_names)
    arrays["counts"] = result.data.counts
    arrays["response"] = result.data.response
    if result.data.truth is not None:
        arrays.update({f"truth/{k}": v for k, v in result.data.truth.items()})
    if result.mean_density is not None:
        arrays["mean_density"] = result.mean_density
    if cfg.output.save_fields and result.fields is not None:
        arrays["fields"] = result.fields  # only when not streamed to samples/
    np.savez_compressed(out / "chains.npz", **arrays)
    return out
