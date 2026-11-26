"""Run directory layout, restart files and streamed samples.

    <output.dir>/
        checkpoint.npz        restart file (sampler state, PRNG key, traces)
        checkpoint.prev.npz   the one before (fallback if the latest is damaged)
        samples/field_<sweep>.npy   kept white-noise samples, (chains, n, n, n) float32
        config.yaml  summary.json  chains.npz  *.png    final outputs

A checkpoint is written to a temporary file and then renamed, so a job
killed while writing leaves the previous checkpoint intact.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .config import Config, ConfigError, from_dict, to_dict
from .samplers import Checkpoint

CHECKPOINT = "checkpoint.npz"
PREVIOUS = "checkpoint.prev.npz"
SAMPLES = "samples"
FORMAT_VERSION = 1

# Settings that may differ between the run that wrote a checkpoint and the
# run that resumes from it. Everything else must be identical.
RESUMABLE_CHANGES = ("output.", "sampler.checkpoint.", "sampler.n_samples")
# ... except these, which decide where the kept samples go
NOT_RESUMABLE = ("output.save_fields",)


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def config_differences(old: Config, new: Config) -> list[str]:
    """Settings that differ between two configs and forbid resuming."""
    a, b = _flatten(to_dict(old)), _flatten(to_dict(new))
    keys = sorted(set(a) | set(b))
    return [
        f"{k}: {a.get(k)!r} -> {b.get(k)!r}"
        for k in keys
        if a.get(k) != b.get(k)
        and (k in NOT_RESUMABLE or not any(k.startswith(p) for p in RESUMABLE_CHANGES))
    ]


# --------------------------------------------------------------------------- #
# Restart files
# --------------------------------------------------------------------------- #
def save_checkpoint(directory, ckpt: Checkpoint, cfg: Config, extra: Optional[dict] = None,
                    keep_previous: bool = True) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    total = cfg.sampler.n_warmup + cfg.sampler.n_samples
    meta = {
        "format": FORMAT_VERSION,
        "iteration": ckpt.iteration,
        "total": total,
        "completed": ckpt.iteration >= total,
        "written": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "config": to_dict(cfg),
        **(extra or {}),
    }
    arrays = {"meta": np.array(json.dumps(meta)), "key": np.asarray(ckpt.key)}
    arrays.update({f"state/{k}": v for k, v in ckpt.state.items()})
    for name, leaves in ckpt.block_states.items():
        arrays[f"blocks/{name}/count"] = np.array(len(leaves))
        arrays.update({f"blocks/{name}/{i}": leaf for i, leaf in enumerate(leaves)})
    arrays.update({f"trace/{k}": v for k, v in ckpt.trace.items()})
    if ckpt.samples:
        arrays.update({f"samples/{k}": v for k, v in ckpt.samples.items()})

    path, tmp = directory / CHECKPOINT, directory / (CHECKPOINT + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    if keep_previous and path.exists():
        os.replace(path, directory / PREVIOUS)
    os.replace(tmp, path)
    return path


def read_meta(directory) -> Optional[dict]:
    path = Path(directory) / CHECKPOINT
    if not path.exists():
        return None
    with np.load(path) as f:
        return json.loads(str(f["meta"]))


def load_checkpoint(directory) -> Optional[tuple[Checkpoint, dict]]:
    """(checkpoint, meta) from ``directory``, or None if there is none.
    Falls back to the previous checkpoint if the latest cannot be read."""
    directory = Path(directory)
    for name in (CHECKPOINT, PREVIOUS):
        path = directory / name
        if not path.exists():
            continue
        try:
            with np.load(path) as f:
                meta = json.loads(str(f["meta"]))
                if meta.get("format") != FORMAT_VERSION:
                    raise ValueError(f"unsupported checkpoint format {meta.get('format')}")
                state = {k[6:]: f[k] for k in f.files if k.startswith("state/")}
                trace = {k[6:]: f[k] for k in f.files if k.startswith("trace/")}
                samples = {k[8:]: f[k] for k in f.files if k.startswith("samples/")} or None
                blocks = {}
                for k in f.files:
                    if k.startswith("blocks/") and k.endswith("/count"):
                        block = k[len("blocks/") : -len("/count")]
                        blocks[block] = [f[f"blocks/{block}/{i}"] for i in range(int(f[k]))]
                ckpt = Checkpoint(int(meta["iteration"]), f["key"], state, blocks, trace, samples)
            if name == PREVIOUS:
                meta["recovered_from_previous"] = True
            return ckpt, meta
        except Exception as e:  # damaged file: try the previous one
            last_error = e
            continue
    if (directory / CHECKPOINT).exists() or (directory / PREVIOUS).exists():
        raise RuntimeError(f"no readable checkpoint in {directory}: {last_error}")
    return None


def check_resumable(meta: dict, cfg: Config) -> None:
    old = from_dict(meta["config"])
    diffs = config_differences(old, cfg)
    if diffs:
        raise ConfigError(
            "the checkpoint in output.dir was written with a different configuration:\n  "
            + "\n  ".join(diffs)
            + "\nuse a new output.dir, or start over with `miniflims run --fresh`"
        )
    total = cfg.sampler.n_warmup + cfg.sampler.n_samples
    if meta["iteration"] > total:
        raise ConfigError(
            f"the checkpoint is at sweep {meta['iteration']}, more than the {total} now requested "
            "(sampler.n_samples can only be increased)"
        )


def clear_run(directory) -> None:
    """Remove restart files and streamed samples (``--fresh``)."""
    directory = Path(directory)
    for name in (CHECKPOINT, PREVIOUS, CHECKPOINT + ".tmp"):
        (directory / name).unlink(missing_ok=True)
    samples = directory / SAMPLES
    if samples.is_dir():
        for p in samples.glob("*.npy"):
            p.unlink()


# --------------------------------------------------------------------------- #
# Streamed samples
# --------------------------------------------------------------------------- #
def sample_writer(directory):
    """Sink for ``run_gibbs``: one .npy per kept sweep and variable."""
    folder = Path(directory) / SAMPLES
    folder.mkdir(parents=True, exist_ok=True)

    def write(sweep: int, arrays: dict) -> None:
        for name, value in arrays.items():
            path = folder / f"{name}_{sweep:07d}.npy"
            tmp = folder / f".{path.name}.tmp"
            with open(tmp, "wb") as f:
                np.save(f, value)
            os.replace(tmp, path)

    return write


def sample_sweeps(directory, name: str = "field", upto: Optional[int] = None) -> list[int]:
    """Sweeps for which a sample file exists (only those < ``upto``)."""
    folder = Path(directory) / SAMPLES
    sweeps = sorted(int(p.stem.rsplit("_", 1)[1]) for p in folder.glob(f"{name}_*.npy"))
    return [s for s in sweeps if upto is None or s < upto]


def iter_samples(directory, name: str = "field", sweeps=None) -> Iterator[tuple[int, np.ndarray]]:
    folder = Path(directory) / SAMPLES
    for s in sweeps if sweeps is not None else sample_sweeps(directory, name):
        yield s, np.load(folder / f"{name}_{s:07d}.npy")


def load_samples(directory, name: str = "field", max_samples: Optional[int] = None,
                 upto: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """(sweeps, samples) with samples of shape (chains, kept, ...). With
    ``max_samples``, an evenly spaced subset of at most that many sweeps."""
    sweeps = sample_sweeps(directory, name, upto)
    if not sweeps:
        raise FileNotFoundError(f"no '{name}' samples in {Path(directory) / SAMPLES}")
    if max_samples is not None and len(sweeps) > max_samples:
        idx = np.linspace(0, len(sweeps) - 1, max_samples).astype(int)
        sweeps = [sweeps[i] for i in idx]
    arrays = [a for _, a in iter_samples(directory, name, sweeps)]
    return np.asarray(sweeps), np.stack(arrays, axis=1)
