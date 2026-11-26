"""Hamiltonian Monte Carlo: transition kernel, dual-averaging step-size
adaptation, and a Gibbs block wrapping them.

Following BORG / Manticore the field is sampled in its white-noise
parametrisation, where the prior is N(0, 1): the identity mass matrix is then
the prior-preconditioned choice. Step size and trajectory length are
randomised per transition (Jasche & Wandelt 2013) to avoid resonances.
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from .gibbs import Block
from .integrators import INTEGRATORS, integrator_step


class HMCInfo(NamedTuple):
    accept_prob: jnp.ndarray
    accepted: jnp.ndarray
    delta_energy: jnp.ndarray
    logdensity: jnp.ndarray  # at the returned position


def hmc_step(
    key,
    x,
    value_and_grad: Callable,
    step_size,
    n_steps,
    integrator: str = "leapfrog",
    inv_mass=1.0,
):
    """One HMC transition for an array-valued position ``x``.

    value_and_grad(x) -> (log density, gradient). ``n_steps`` may be traced.
    A non-finite energy (divergent trajectory) is rejected.
    """
    coeffs = INTEGRATORS.get(integrator)
    k_mom, k_acc = jax.random.split(key)
    logp0, grad0 = value_and_grad(x)
    p0 = jax.random.normal(k_mom, x.shape, x.dtype) / jnp.sqrt(inv_mass)

    def kinetic(p):
        return 0.5 * jnp.sum(inv_mass * p**2)

    def body(_, carry):
        xc, pc, gc, _ = carry
        xn, pn, lpn, gn = integrator_step(
            value_and_grad, coeffs, step_size, xc, pc, gc, inv_mass
        )
        return xn, pn, gn, lpn

    x1, p1, _, logp1 = jax.lax.fori_loop(0, n_steps, body, (x, p0, grad0, logp0))

    delta = (kinetic(p1) - logp1) - (kinetic(p0) - logp0)
    delta = jnp.where(jnp.isfinite(delta), delta, jnp.inf)
    accept_prob = jnp.minimum(1.0, jnp.exp(-delta))
    accepted = jnp.log(jax.random.uniform(k_acc, dtype=logp0.dtype)) < -delta
    x_new = jnp.where(accepted, x1, x)
    logp_new = jnp.where(accepted, logp1, logp0)
    return x_new, HMCInfo(accept_prob, accepted, delta, logp_new)


# --------------------------------------------------------------------------- #
# Dual averaging (Hoffman & Gelman 2014, Algorithm 5)
# --------------------------------------------------------------------------- #
class DualAveragingState(NamedTuple):
    log_eps: jnp.ndarray  # step size used while adapting
    log_eps_bar: jnp.ndarray  # averaged iterate, used after warm-up
    h_bar: jnp.ndarray
    t: jnp.ndarray
    mu: jnp.ndarray


def dual_averaging_init(step_size) -> DualAveragingState:
    le = jnp.log(jnp.asarray(step_size, dtype=jnp.result_type(float)))
    zero = jnp.zeros_like(le)
    return DualAveragingState(le, le, zero, zero, le + jnp.log(10.0))


def dual_averaging_update(
    s: DualAveragingState, accept_prob, target, gamma=0.05, t0=10.0, kappa=0.75
) -> DualAveragingState:
    t = s.t + 1.0
    h_bar = (1.0 - 1.0 / (t + t0)) * s.h_bar + (target - accept_prob) / (t + t0)
    log_eps = s.mu - jnp.sqrt(t) / gamma * h_bar
    eta = t**-kappa
    log_eps_bar = eta * log_eps + (1.0 - eta) * s.log_eps_bar
    return DualAveragingState(log_eps, log_eps_bar, h_bar, t, s.mu)


# --------------------------------------------------------------------------- #
# Gibbs block
# --------------------------------------------------------------------------- #
def hmc_block(name: str, var: str, logdensity: Callable, cfg) -> Block:
    """HMC update of ``state[var]`` targeting ``logdensity(state, hyper)``
    with the other variables held fixed. ``cfg`` is an ``HMCConfig``."""
    n_max = cfg.n_steps
    n_min = cfg.n_steps if cfg.n_steps_min is None else cfg.n_steps_min
    if not 1 <= n_min <= n_max:
        raise ValueError("need 1 <= hmc.n_steps_min <= hmc.n_steps")
    if not 0.0 <= cfg.step_jitter < 1.0:
        raise ValueError("hmc.step_jitter must be in [0, 1)")
    INTEGRATORS.get(cfg.integrator)  # fail early on a bad name

    def init(state):
        return dual_averaging_init(cfg.step_size)

    def step(key, state, bstate, hyper, warmup: bool):
        def vg(x):
            return jax.value_and_grad(lambda v: logdensity({**state, var: v}, hyper))(x)

        adapting = warmup and cfg.adapt_step_size
        x = state[var]
        info, eps = None, None
        for _ in range(cfg.n_per_sweep):
            key, k_eps, k_n, k_hmc = jax.random.split(key, 4)
            base = jnp.exp(bstate.log_eps if adapting else bstate.log_eps_bar)
            eps = base * (1.0 - cfg.step_jitter * jax.random.uniform(k_eps))
            n = jax.random.randint(k_n, (), n_min, n_max + 1)
            x, info = hmc_step(k_hmc, x, vg, eps, n, cfg.integrator)
            if adapting:
                bstate = dual_averaging_update(bstate, info.accept_prob, cfg.target_accept)
        out = {
            "accept_prob": info.accept_prob,
            "accepted": info.accepted,
            "delta_energy": info.delta_energy,
            "logdensity": info.logdensity,
            "step_size": eps,
            "n_steps": n,
        }
        return {**state, var: x}, bstate, out

    return Block(name, init, step)
