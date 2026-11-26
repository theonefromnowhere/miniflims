"""Hyper-parameter schedules (likelihood tempering, prior-strength ramps).

Manticore ramps some prior hyper-parameters during burn-in and then freezes
them; samples are only kept once every schedule has reached its end value.
"""
from __future__ import annotations

from typing import Callable

import jax.numpy as jnp


def schedule_value(sched, i: int) -> float:
    """Value of a ``ScheduleConfig`` at sweep ``i``."""
    if sched.n_steps <= 0 or i >= sched.n_steps:
        return float(sched.end)
    t = i / sched.n_steps
    if sched.kind == "linear":
        return float(sched.start + t * (sched.end - sched.start))
    if sched.kind == "geometric":
        if sched.start <= 0 or sched.end <= 0:
            raise ValueError("geometric schedules need positive start and end")
        return float(sched.start * (sched.end / sched.start) ** t)
    raise ValueError(f"unknown schedule kind '{sched.kind}' (linear or geometric)")


def make_hyper_fn(schedules: dict, defaults: dict, n_warmup: int) -> Callable:
    """iteration -> {name: scalar} for every name in ``defaults``."""
    unknown = set(schedules) - set(defaults)
    if unknown:
        raise ValueError(
            f"unknown schedule(s) {sorted(unknown)}; valid: {sorted(defaults)}"
        )
    for name, s in schedules.items():
        if s.n_steps > n_warmup:
            raise ValueError(
                f"schedule '{name}' runs for {s.n_steps} sweeps, longer than "
                f"the warm-up ({n_warmup}); samples would target a moving posterior"
            )
        schedule_value(s, 0)  # validate

    def hyper_fn(i: int) -> dict:
        out = {k: jnp.asarray(v, jnp.result_type(float)) for k, v in defaults.items()}
        for name, s in schedules.items():
            out[name] = jnp.asarray(schedule_value(s, i), jnp.result_type(float))
        return out

    return hyper_fn
