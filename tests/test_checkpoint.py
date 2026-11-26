"""Checkpoint / resume, signal handling, streamed samples and progress."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from miniflims import load_config, run, update
from miniflims import pipeline
from miniflims.checkpoint import (
    CHECKPOINT,
    PREVIOUS,
    config_differences,
    load_checkpoint,
    load_samples,
    read_meta,
    sample_sweeps,
)
from miniflims.cli import main
from miniflims.config import ConfigError, HMCConfig, SliceConfig
from miniflims.samplers import hmc_block, run_gibbs, slice_block

ROOT = Path(__file__).resolve().parents[1]


def small(tmp_path, name="run", **over):
    cfg = {
        "grid": {"n_grid": 8, "box_size": 200.0},
        "data": {"mock": {"truth": {"alpha": 1.4}}},
        "sampler": {"n_chains": 2, "n_warmup": 12, "n_samples": 18, "thin": 3,
                    "hmc": {"n_steps": 6, "n_steps_min": 3}, "checkpoint": {"every": 5}},
        "output": {"dir": str(tmp_path / name), "log_every": 0, "progress": False},
    }
    return update(load_config(), _merge(cfg, over))


def _merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return out


@pytest.fixture
def stop_after(monkeypatch):
    """Make the pipeline's signal watcher request a stop after n sweeps."""

    def arm(n):
        calls = {"n": 0}

        def fake(self):
            calls["n"] += 1
            return self.requested or calls["n"] >= n

        monkeypatch.setattr(pipeline._StopOnSignal, "__call__", fake)

    return arm


def assert_same_run(a: Path, b: Path):
    with np.load(a / "chains.npz") as fa, np.load(b / "chains.npz") as fb:
        assert sorted(fa.files) == sorted(fb.files)
        for k in fa.files:
            np.testing.assert_array_equal(fa[k], fb[k], err_msg=k)
    sa, xa = load_samples(a)
    sb, xb = load_samples(b)
    np.testing.assert_array_equal(sa, sb)
    np.testing.assert_array_equal(xa, xb)
    ja, jb = (json.loads((d / "summary.json").read_text()) for d in (a, b))
    ja.pop("wall_time_s"), jb.pop("wall_time_s")
    assert ja == jb


# --------------------------------------------------------------------- pipeline
def test_interrupted_and_resumed_run_is_identical_to_an_uninterrupted_one(tmp_path, stop_after):
    full = run(small(tmp_path, "full"))
    assert full.completed

    for stop_at in (4, 13):  # once in warm-up, once while sampling
        cfg = small(tmp_path, f"split{stop_at}")
        stop_after(stop_at)
        part = run(cfg)
        assert not part.completed and part.summary is None
        assert part.gibbs.n_done == stop_at
        meta = read_meta(cfg.output.dir)
        assert meta["iteration"] == stop_at and not meta["completed"]
        assert not (Path(cfg.output.dir) / "summary.json").exists()

        stop_after(10**9)
        rest = run(cfg)
        assert rest.completed and rest.gibbs.n_done == 30
        assert read_meta(cfg.output.dir)["completed"]
        assert_same_run(Path(full.run_dir), Path(cfg.output.dir))


def test_checkpoint_cadence_and_contents(tmp_path):
    cfg = small(tmp_path)
    seen = []
    orig = pipeline.save_checkpoint

    def spy(directory, ckpt, *a, **k):
        seen.append(ckpt.iteration)
        return orig(directory, ckpt, *a, **k)

    pipeline.save_checkpoint, saved = spy, pipeline.save_checkpoint
    try:
        res = run(cfg)
    finally:
        pipeline.save_checkpoint = saved
    assert seen == [5, 10, 15, 20, 25, 30]
    d = Path(cfg.output.dir)
    assert (d / CHECKPOINT).exists() and (d / PREVIOUS).exists()
    assert not list(d.glob("*.tmp"))
    ckpt, meta = load_checkpoint(d)
    assert ckpt.iteration == 30 and meta["total"] == 30 and meta["wall_time_s"] > 0
    assert set(ckpt.state) == {"field", "nuisance"} and set(ckpt.block_states) == {"field", "nuisance"}
    assert ckpt.trace["log_likelihood"].shape == (2, 30)
    assert ckpt.samples is None  # streamed to samples/, not stored in the checkpoint
    np.testing.assert_array_equal(ckpt.state["field"], np.asarray(res.gibbs.state["field"]))
    assert sample_sweeps(d) == [12, 15, 18, 21, 24, 27]


def test_checkpointing_can_be_limited_to_the_end(tmp_path):
    cfg = update(small(tmp_path), ["sampler.checkpoint.every=0", "sampler.checkpoint.keep_previous=false"])
    run(cfg)
    d = Path(cfg.output.dir)
    assert read_meta(d)["iteration"] == 30 and not (d / PREVIOUS).exists()


def test_real_signal_checkpoints_and_restores_handlers(tmp_path):
    cfg = update(small(tmp_path), ["output.log_every=1"])
    before = signal.getsignal(signal.SIGUSR1)
    lines = []

    def logger(msg):
        lines.append(msg)
        if msg.startswith("[warmup     7/"):
            os.kill(os.getpid(), signal.SIGUSR1)

    res = run(cfg, logger=logger)
    assert not res.completed and res.gibbs.n_done == 7
    assert any("received SIGUSR1" in l for l in lines)
    assert read_meta(cfg.output.dir)["iteration"] == 7
    assert signal.getsignal(signal.SIGUSR1) is before
    res = run(cfg, logger=lines.append)
    assert res.completed
    assert any(l.startswith("resuming from sweep 7/30") for l in lines)


def test_in_memory_samples_are_checkpointed_too(tmp_path, stop_after):
    full = run(update(small(tmp_path, "a"), ["output.save_fields=false"]))
    cfg = update(small(tmp_path, "b"), ["output.save_fields=false"])
    stop_after(20)
    run(cfg)
    ckpt, _ = load_checkpoint(cfg.output.dir)
    assert ckpt.samples["field"].shape == (2, 3, 8, 8, 8)  # sweeps 12, 15, 18
    stop_after(10**9)
    res = run(cfg)
    np.testing.assert_array_equal(res.fields, full.fields)
    assert not (Path(cfg.output.dir) / "samples").exists()


def test_extending_a_finished_run(tmp_path):
    cfg = small(tmp_path)
    run(cfg)
    longer = update(cfg, ["sampler.n_samples=27"])
    res = run(longer)
    assert res.completed and res.gibbs.n_done == 39
    assert res.gibbs.trace["log_likelihood"].shape == (2, 39)
    assert sample_sweeps(cfg.output.dir) == [12, 15, 18, 21, 24, 27, 30, 33, 36]
    # identical to having asked for the longer run from the start
    direct = run(update(small(tmp_path, "direct"), ["sampler.n_samples=27"]))
    assert_same_run(Path(direct.run_dir), Path(cfg.output.dir))


@pytest.mark.parametrize(
    "change,ok",
    [
        (["output.log_every=7", "output.progress=true", "sampler.checkpoint.every=3"], True),
        (["sampler.n_samples=30"], True),
        (["sampler.hmc.n_steps=9"], False),
        (["sampler.seed=1"], False),
        (["galaxies.bias.model=truncated_power_law"], False),
        (["output.save_fields=false"], False),
        (["sampler.n_samples=3"], False),  # fewer than already done
    ],
)
def test_resume_compatibility(tmp_path, stop_after, change, ok):
    cfg = small(tmp_path)
    stop_after(20)
    run(cfg)
    stop_after(10**9)
    new = update(cfg, change)
    if ok:
        assert run(new).completed
    else:
        with pytest.raises(ConfigError):
            run(new)


def test_config_differences_lists_offending_keys():
    a = load_config()
    b = update(a, ["sampler.hmc.step_size=0.5", "output.log_every=3", "cosmology.h=0.7"])
    diffs = config_differences(a, b)
    assert len(diffs) == 2 and diffs[0].startswith("cosmology.h") and "sampler.hmc.step_size" in diffs[1]


def test_fresh_and_resume_false_start_over(tmp_path, stop_after):
    cfg = small(tmp_path)
    stop_after(20)
    run(cfg)
    stop_after(10**9)
    lines = []
    res = run(cfg, logger=lines.append, fresh=True)
    assert res.completed and not any("resuming" in l for l in lines)
    # an incompatible config is fine once the old run is discarded
    res = run(update(cfg, ["sampler.seed=5", "sampler.checkpoint.resume=false"]))
    assert res.completed


def test_damaged_checkpoint_falls_back_to_previous(tmp_path, stop_after):
    cfg = small(tmp_path)
    stop_after(12)
    run(cfg)
    d = Path(cfg.output.dir)
    (d / CHECKPOINT).write_bytes(b"truncated")
    ckpt, meta = load_checkpoint(d)
    assert ckpt.iteration == 10 and meta["recovered_from_previous"]
    stop_after(10**9)
    lines = []
    assert run(cfg, logger=lines.append).completed
    assert any("checkpoint.prev.npz" in l for l in lines)
    (d / CHECKPOINT).write_bytes(b"x")
    (d / PREVIOUS).write_bytes(b"y")
    with pytest.raises(RuntimeError):
        load_checkpoint(d)


# ------------------------------------------------------------------ run_gibbs
def _toy_blocks():
    def logp(state, hyper):
        return -0.5 * jnp.sum(state["x"] ** 2) - 0.5 * jnp.sum((state["s"] - 1.0) ** 2)

    def cond(state, hyper):
        return (lambda row, aux: logp({**state, "s": row[None]}, hyper)), jnp.zeros(1)

    return [
        hmc_block("x", "x", logp, HMCConfig(step_size=0.2, n_steps=5)),
        slice_block("s", "s", cond, [-5.0], [5.0], [1.0], SliceConfig()),
    ]


def test_run_gibbs_resume_matches_and_progress_bar(capsys):
    state = {"x": jnp.ones((3, 4)), "s": jnp.zeros((3, 1, 1))}
    kw = dict(n_warmup=6, n_samples=9, collect=lambda st: {"x": st["x"]}, thin=2,
              monitor=lambda st, h: {"x": st["x"]})
    full = run_gibbs(jax.random.PRNGKey(0), _toy_blocks(), state, **kw)

    ckpts, sweeps = [], iter(range(1, 100))
    part = run_gibbs(jax.random.PRNGKey(0), _toy_blocks(), state, checkpoint_every=4,
                     on_checkpoint=ckpts.append, should_stop=lambda: next(sweeps) >= 8, **kw)
    assert not part.completed and part.n_done == 8 and [c.iteration for c in ckpts] == [4, 8]
    rest = run_gibbs(jax.random.PRNGKey(123), _toy_blocks(), {k: v * 0 for k, v in state.items()},
                     resume=ckpts[-1], progress=True, progress_interval=0.0, **kw)
    assert rest.completed and rest.n_done == 15
    for k in full.trace:
        np.testing.assert_array_equal(full.trace[k], rest.trace[k], err_msg=k)
    np.testing.assert_array_equal(full.samples["x"], rest.samples["x"])
    err = capsys.readouterr().err
    assert "15/15" in err and "sampling" in err


def test_run_gibbs_rejects_mismatched_checkpoints():
    state = {"x": jnp.ones((2, 4)), "s": jnp.zeros((2, 1, 1))}
    ckpts = []
    run_gibbs(jax.random.PRNGKey(0), _toy_blocks(), state, n_warmup=2, n_samples=2,
              checkpoint_every=2, on_checkpoint=ckpts.append)
    good = ckpts[-1]
    with pytest.raises(ValueError):  # beyond the requested length
        run_gibbs(jax.random.PRNGKey(0), _toy_blocks(), state, n_warmup=1, n_samples=1, resume=good)
    with pytest.raises(ValueError):  # different blocks
        run_gibbs(jax.random.PRNGKey(0), _toy_blocks()[:1], state, n_warmup=2, n_samples=2, resume=good)
    with pytest.raises(ValueError):  # different state variables
        run_gibbs(jax.random.PRNGKey(0), _toy_blocks(), {"x": state["x"]}, n_warmup=2, n_samples=2, resume=good)


# ------------------------------------------------------------------------ CLI
def test_cli_status_and_exit_codes(tmp_path, stop_after, capsys):
    cfg = small(tmp_path)
    d = cfg.output.dir
    assert main(["status", d]) == 1
    stop_after(20)
    args = ["run", "-o", "grid.n_grid=8", "-o", "grid.box_size=200", "-o", "sampler.n_warmup=12",
            "-o", "sampler.n_samples=18", "-o", "sampler.thin=3", "-o", "sampler.n_chains=2",
            "-o", "sampler.hmc.n_steps=6", "-o", "sampler.hmc.n_steps_min=3",
            "-o", "data.mock.truth={alpha: 1.4}", "-o", "sampler.checkpoint.every=5",
            "-o", f"output.dir={d}", "-o", "output.log_every=0", "-o", "output.progress=false"]
    assert main(args) == pipeline.EXIT_INTERRUPTED
    capsys.readouterr()
    assert main(["status", d]) == 0
    out = capsys.readouterr().out
    assert "sweep 20/30 (66.7%), sampling" in out and "field samples    3" in out
    assert main(args + ["-o", "sampler.seed=9"]) == 2  # incompatible resume
    assert "sampler.seed" in capsys.readouterr().err
    stop_after(10**9)
    assert main(args) == 0
    capsys.readouterr()
    assert main(["status", d]) == 0
    assert "sweep 30/30 (100.0%), finished" in capsys.readouterr().out
    assert main(["plot", d]) == 0


@pytest.mark.slow
def test_sigterm_to_a_real_process_then_resume(tmp_path):
    d = tmp_path / "proc"
    cmd = [sys.executable, "-m", "miniflims", "run", "-c", str(ROOT / "configs" / "default.yaml"),
           "-o", "grid.n_grid=16", "-o", "grid.box_size=400", "-o", "sampler.n_warmup=150",
           "-o", "sampler.n_samples=150", "-o", "sampler.checkpoint.every=10",
           "-o", f"output.dir={d}", "-o", "output.log_every=0", "-o", "output.progress_interval=0"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.time() + 240
    while read_meta(d) is None or read_meta(d)["iteration"] < 20:
        assert proc.poll() is None and time.time() < deadline, proc.stderr.read() if proc.poll() else "timeout"
        time.sleep(0.2)
    proc.send_signal(signal.SIGTERM)
    out, err = proc.communicate(timeout=120)
    assert proc.returncode == 3, err
    assert "received SIGTERM" in out + err and "Run the same command again to resume" in out
    assert "sweep" in err  # tqdm progress on stderr
    stopped_at = read_meta(d)["iteration"]
    assert 20 <= stopped_at < 300
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr
    assert f"resuming from sweep {stopped_at}/300" in done.stdout
    assert read_meta(d)["completed"] and (d / "summary.json").exists()
    assert sample_sweeps(d) == list(range(150, 300, 10))  # default.yaml keeps every 10th sweep
