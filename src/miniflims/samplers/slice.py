"""Univariate slice sampling (Neal 2003) with stepping-out and shrinkage, and
a coordinate-wise Gibbs block built on it.

This is how BORG / Manticore update the bias and noise parameters: one at a
time, conditioned on the current density field, with no tuning-sensitive
proposal.
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from .gibbs import Block


class SliceInfo(NamedTuple):
    n_evals: jnp.ndarray  # log-density evaluations used
    width: jnp.ndarray  # final (shrunk) bracket width
    moved: jnp.ndarray  # False only if max_shrink was exhausted


def slice_sample_1d(
    key,
    logp: Callable,
    x0,
    width,
    lower=-jnp.inf,
    upper=jnp.inf,
    max_stepout: int = 8,
    max_shrink: int = 64,
):
    """One slice-sampling update of a scalar with support [lower, upper].

    Leaves the density exp(logp) restricted to [lower, upper] invariant.
    The bracket is expanded at most ``max_stepout`` times in total (Neal's
    limit m, split randomly between the two sides).
    """
    k_y, k_u, k_j, k_s = jax.random.split(key, 4)
    dtype = jnp.result_type(x0, float)
    x0 = jnp.asarray(x0, dtype)

    def f(v):  # NaN -> -inf; never evaluated outside the support
        lp = logp(jnp.clip(v, lower, upper))
        return jnp.where(jnp.isnan(lp), -jnp.inf, lp)

    log_y = f(x0) + jnp.log(jax.random.uniform(k_y, dtype=dtype))

    left = x0 - width * jax.random.uniform(k_u, dtype=dtype)
    right = left + width
    j = jnp.floor(max_stepout * jax.random.uniform(k_j, dtype=dtype)).astype(jnp.int32)
    k = (max_stepout - 1) - j

    def expand(edge, budget, sign, bound_ok):
        # move ``edge`` outward by ``width`` while it is inside the slice
        def go(e, b):
            return (b > 0) & bound_ok(e) & (f(e) > log_y)

        def body(c):
            e, b, _, n = c
            e, b = e + sign * width, b - 1
            return e, b, go(e, b), n + 1

        e, _, _, n = jax.lax.while_loop(
            lambda c: c[2], body, (edge, budget, go(edge, budget), jnp.int32(1))
        )
        return e, n

    left, n_l = expand(left, j, -1.0, lambda e: e > lower)
    right, n_r = expand(right, k, 1.0, lambda e: e < upper)
    left = jnp.maximum(left, lower)
    right = jnp.minimum(right, upper)

    def shrink_body(c):
        lo, hi, _, _, n, kk = c
        kk, sub = jax.random.split(kk)
        x1 = lo + (hi - lo) * jax.random.uniform(sub, dtype=dtype)
        ok = f(x1) >= log_y
        lo = jnp.where(ok | (x1 >= x0), lo, x1)
        hi = jnp.where(ok | (x1 < x0), hi, x1)
        return lo, hi, x1, ok, n + 1, kk

    lo, hi, x1, ok, n_s, _ = jax.lax.while_loop(
        lambda c: (~c[3]) & (c[4] < max_shrink),
        shrink_body,
        (left, right, x0, jnp.asarray(False), jnp.int32(0), k_s),
    )
    x_new = jnp.where(ok, x1, x0)
    return x_new, SliceInfo(1 + n_l + n_r + n_s, hi - lo, ok)


class SliceState(NamedTuple):
    widths: jnp.ndarray  # (rows, params)


def slice_block(
    name: str,
    var: str,
    make_conditional: Callable,
    lower,
    upper,
    widths,
    cfg,
    sampled=None,
) -> Block:
    """Coordinate-wise slice sampling of ``state[var]``, shape (rows, params).

    ``make_conditional(state, hyper) -> (fn, aux)`` where
    ``fn(theta_row, aux_row)`` is the log conditional density of one row and
    ``aux`` is a pytree with a leading ``rows`` axis. Rows must be
    conditionally independent given the rest of the state; they are updated in
    parallel. Anything expensive that does not depend on ``state[var]`` should
    be computed once inside ``make_conditional``.

    ``lower/upper/widths`` have shape (params,); ``sampled`` is a static tuple
    of bools (False = parameter held fixed). ``cfg`` is a ``SliceConfig``.
    During warm-up each bracket width tracks a running geometric mean of the
    final shrunk bracket, which is of the order of the slice size.
    """
    lower, upper, widths = (jnp.asarray(v, jnp.result_type(float)) for v in (lower, upper, widths))
    n_par = widths.shape[0]
    sampled = (True,) * n_par if sampled is None else tuple(bool(s) for s in sampled)
    active = [j for j in range(n_par) if sampled[j]]

    def init(state):
        rows = state[var].shape[0]
        return SliceState(jnp.tile(widths, (rows, 1)))

    def step(key, state, bstate, hyper, warmup: bool):
        fn, aux = make_conditional(state, hyper)
        adapting = warmup and cfg.adapt_width

        def update_row(k, row, aux_row, w_row):
            n_evals = jnp.int32(0)
            for _ in range(cfg.n_per_sweep):
                for j in active:
                    k, sub = jax.random.split(k)
                    xj, info = slice_sample_1d(
                        sub,
                        lambda v, row=row, j=j: fn(row.at[j].set(v), aux_row),
                        row[j],
                        w_row[j],
                        lower[j],
                        upper[j],
                        cfg.max_stepout,
                        cfg.max_shrink,
                    )
                    row = row.at[j].set(xj)
                    n_evals = n_evals + info.n_evals
                    if adapting:
                        span = upper[j] - lower[j]
                        target = jnp.clip(info.width, 1e-8 * span, span)
                        new_w = jnp.exp(0.9 * jnp.log(w_row[j]) + 0.1 * jnp.log(target))
                        w_row = w_row.at[j].set(new_w)
            return row, w_row, n_evals

        theta = state[var]
        keys = jax.random.split(key, theta.shape[0])
        theta, new_widths, n_evals = jax.vmap(update_row)(keys, theta, aux, bstate.widths)
        info = {"n_evals": n_evals, "widths": new_widths}
        return {**state, var: theta}, SliceState(new_widths), info

    return Block(name, init, step)
