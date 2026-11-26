"""Generic block-Gibbs driver with checkpoint / resume and a progress bar.

The sampler state is a dict of arrays. A ``Block`` updates part of it with
any kernel that leaves the corresponding conditional invariant; one sweep
applies the blocks in order. All chains are advanced together (vmap) by one
jit-compiled sweep; the outer loop is plain Python so that schedules, logging,
checkpoints and sample collection stay simple.

Restarting: everything the loop carries from one sweep to the next (state,
block adaptation states, PRNG key, traces and in-memory samples) is exposed as
a ``Checkpoint``. Resuming from one continues the exact same computation, so
an interrupted run gives the same chains as an uninterrupted one (bit for bit
on CPU; GPU scatter-adds are not deterministic).
"""
from __future__ import annotations

import sys
import time
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np


class Block(NamedTuple):
    name: str
    # init(state) -> block state (adaptation variables), for ONE chain
    init: Callable
    # step(key, state, block_state, hyper, warmup: bool)
    #   -> (state, block_state, info dict of arrays), for ONE chain
    step: Callable


def gated_block(block: Block, is_open: Callable) -> Block:
    """``block`` acting only on sweeps where ``is_open(state)`` (a scalar
    bool, for one chain) holds; otherwise the state and the block's
    adaptation are left as they were. The block is evaluated either way (as
    under vmap over chains anyway). Adds ``active`` to the block's info."""

    def step(key, state, bstate, hyper, warmup: bool):
        on = is_open(state)
        new_state, new_bs, info = block.step(key, state, bstate, hyper, warmup)

        def pick(a, b):
            return jnp.where(on, a, b)

        return (jax.tree_util.tree_map(pick, new_state, state),
                jax.tree_util.tree_map(pick, new_bs, bstate),
                {**info, "active": on})

    return Block(block.name, block.init, step)


class Checkpoint(NamedTuple):
    """Everything needed to continue a run after ``iteration`` sweeps."""

    iteration: int  # sweeps completed
    key: np.ndarray  # PRNG key for the next sweep
    state: dict  # name -> array, leading chain axis
    block_states: dict  # block name -> list of pytree leaves (leading chain axis)
    trace: dict  # name -> (chains, iteration, ...)
    samples: Optional[dict]  # name -> (chains, kept, ...) when kept in memory


class GibbsResult(NamedTuple):
    state: dict  # final state, leading axis = chain
    block_states: dict
    trace: dict  # "block/key" or monitor key -> (n_chains, n_done, ...)
    samples: Optional[dict]  # collected post-warm-up, (n_chains, n_kept, ...), unless streamed
    n_warmup: int
    wall_time: float  # seconds spent in this call
    n_done: int = 0  # sweeps completed (including any resumed ones)
    completed: bool = True  # False if stopped early by ``should_stop``


def make_sweep(blocks, monitor=None, warmup: bool = False):
    """One Gibbs sweep for a single chain (pure JAX)."""

    def sweep(key, state, block_states, hyper):
        keys = jax.random.split(key, len(blocks))
        out = {}
        new_bs = dict(block_states)
        for blk, k in zip(blocks, keys):
            state, new_bs[blk.name], info = blk.step(
                k, state, new_bs[blk.name], hyper, warmup
            )
            out.update({f"{blk.name}/{name}": v for name, v in info.items()})
        if monitor is not None:
            out.update(monitor(state, hyper))
        return state, new_bs, out

    return sweep


def _stack(lists: dict) -> dict:
    return {k: np.stack(v, axis=1) for k, v in lists.items()}


def _unstack(arrays: dict) -> dict:
    return {k: list(np.moveaxis(np.asarray(v), 1, 0)) for k, v in arrays.items()}


def run_gibbs(
    key,
    blocks,
    state: dict,
    *,
    n_warmup: int,
    n_samples: int,
    hyper_fn: Optional[Callable] = None,
    monitor: Optional[Callable] = None,
    collect: Optional[Callable] = None,
    thin: int = 1,
    log_every: int = 0,
    log_fn: Optional[Callable] = None,
    logger: Optional[Callable] = None,
    resume: Optional[Checkpoint] = None,
    checkpoint_every: int = 0,
    on_checkpoint: Optional[Callable[[Checkpoint], None]] = None,
    sample_sink: Optional[Callable[[int, dict], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    progress: bool = False,
    progress_interval: Optional[float] = None,
) -> GibbsResult:
    """Run sweeps up to ``n_warmup + n_samples`` in total.

    state      dict of arrays with a leading chain axis (ignored when resuming).
    hyper_fn   iteration -> dict of scalar hyper-parameters passed to the blocks
               (traced, so schedules do not trigger recompilation).
    monitor    (state, hyper) -> dict of arrays traced every sweep.
    collect    state -> dict of arrays kept every ``thin`` sweeps after
               warm-up (per chain; use it for the large variables).
    log_fn     (iteration, info dict of numpy arrays) -> str, used for the
               periodic log lines and the progress-bar postfix.
    resume     continue from this checkpoint instead of ``key`` / ``state``.
    on_checkpoint  called with a ``Checkpoint`` every ``checkpoint_every``
               sweeps, at the end, and when stopping early.
    sample_sink    (sweep, {name: array}) -> None; collected samples go here
               instead of being kept in memory (and in checkpoints).
    should_stop    () -> bool, checked after every sweep; True ends the run
               early (after a checkpoint) with ``completed=False``.
    progress   show a tqdm bar on stderr, refreshed at most every
               ``progress_interval`` s (default 0.5 on a terminal, 30 otherwise).
    """
    names = [b.name for b in blocks]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate block names: {names}")
    if thin < 1:
        raise ValueError("thin must be at least 1")
    total = n_warmup + n_samples
    hyper_fn = hyper_fn or (lambda i: {})

    if resume is not None:
        if not 0 <= resume.iteration <= total:
            raise ValueError(
                f"checkpoint is at sweep {resume.iteration}, beyond the requested {total}"
            )
        if set(resume.state) != set(state):
            raise ValueError(f"checkpoint state {sorted(resume.state)} does not match {sorted(state)}")
        if set(resume.block_states) != set(names):
            raise ValueError(
                f"checkpoint blocks {sorted(resume.block_states)} do not match {names}"
            )
        state = {k: jax.numpy.asarray(v) for k, v in resume.state.items()}
    block_states = {b.name: jax.vmap(b.init)(state) for b in blocks}
    if resume is not None:
        for name, leaves in resume.block_states.items():
            treedef = jax.tree_util.tree_structure(block_states[name])
            if treedef.num_leaves != len(leaves):
                raise ValueError(f"checkpoint state of block '{name}' does not match its structure")
            block_states[name] = jax.tree_util.tree_unflatten(
                treedef, [jax.numpy.asarray(v) for v in leaves]
            )
        key = jax.numpy.asarray(resume.key)
    start = resume.iteration if resume is not None else 0

    sweeps = {
        w: jax.jit(jax.vmap(make_sweep(blocks, monitor, w), in_axes=(0, 0, 0, None)))
        for w in (True, False)
    }
    collect_fn = jax.jit(jax.vmap(collect)) if collect is not None else None
    n_chains = next(iter(state.values())).shape[0]

    trace: dict[str, list] = _unstack(resume.trace) if resume is not None else {}
    samples: dict[str, list] = (
        _unstack(resume.samples) if resume is not None and resume.samples else {}
    )

    bar = None
    if progress:
        from tqdm import tqdm

        interactive = sys.stderr.isatty()
        interval = progress_interval if progress_interval is not None else (0.5 if interactive else 30.0)
        bar = tqdm(
            total=total, initial=start, unit="sweep", file=sys.stderr, mininterval=interval,
            maxinterval=max(interval, 10.0), dynamic_ncols=interactive, ncols=None if interactive else 110,
            ascii=not interactive, smoothing=0.05,
            desc="warm-up" if start < n_warmup else "sampling",
        )

    def emit(msg: str) -> None:
        if bar is not None and (logger is None or logger is print):
            bar.write(msg, file=sys.stdout)  # keeps the bar intact
        elif logger is not None:
            logger(msg)
        else:
            print(msg)

    def checkpoint(n_done: int) -> Checkpoint:
        return Checkpoint(
            iteration=n_done,
            key=np.asarray(key),
            state={k: np.asarray(v) for k, v in state.items()},
            block_states={
                k: [np.asarray(x) for x in jax.tree_util.tree_leaves(v)] for k, v in block_states.items()
            },
            trace=_stack(trace) if trace else {},
            samples=_stack(samples) if samples else None,
        )

    t0 = time.perf_counter()
    done, stopped = start, False
    try:
        for i in range(start, total):
            warm = i < n_warmup
            if bar is not None and i == n_warmup:
                bar.set_description("sampling", refresh=False)
            key, sub = jax.random.split(key)
            state, block_states, info = sweeps[warm](
                jax.random.split(sub, n_chains), state, block_states, hyper_fn(i)
            )
            info = {k: np.asarray(v) for k, v in info.items()}
            for k, v in info.items():
                trace.setdefault(k, []).append(v)
            if collect_fn is not None and not warm and (i - n_warmup) % thin == 0:
                kept = {k: np.asarray(v) for k, v in collect_fn(state).items()}
                if sample_sink is not None:
                    sample_sink(i, kept)
                else:
                    for k, v in kept.items():
                        samples.setdefault(k, []).append(v)
            done = i + 1

            msg = log_fn(i, info) if log_fn else ""
            if bar is not None:
                bar.set_postfix_str(msg, refresh=False)
                bar.update(1)
            if log_every and (done % log_every == 0 or i == start):
                phase = "warmup" if warm else "sample"
                emit(f"[{phase} {done:>5d}/{total}] {time.perf_counter() - t0:7.1f}s  {msg}")

            stopped = bool(should_stop()) if should_stop is not None else False
            due = checkpoint_every > 0 and done % checkpoint_every == 0
            if on_checkpoint is not None and (due or stopped or done == total):
                on_checkpoint(checkpoint(done))
            if stopped:
                emit(f"stopped after sweep {done}/{total}; checkpoint written" if on_checkpoint
                     else f"stopped after sweep {done}/{total}")
                break
        else:
            if on_checkpoint is not None and start == total:
                on_checkpoint(checkpoint(done))  # nothing to do, but keep the files current
    finally:
        if bar is not None:
            bar.close()

    return GibbsResult(
        state=state,
        block_states=block_states,
        trace=_stack(trace) if trace else {},
        samples=_stack(samples) if samples else None,
        n_warmup=n_warmup,
        wall_time=time.perf_counter() - t0,
        n_done=done,
        completed=done == total,
    )
