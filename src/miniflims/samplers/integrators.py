"""Symplectic, time-reversible integrators for HMC as palindromic splittings.

An integrator is a pair of coefficient tuples ``(a, b)`` with len(b) =
len(a) + 1. One step of size eps applies

    kick b[0], drift a[0], kick b[1], ..., drift a[-1], kick b[-1]

where kick: p += eps * b * grad and drift: x += eps * a * p / m. The gradient
at the end of a step is reused by the first kick of the next one, so a step
costs len(a) gradient evaluations.
"""
from __future__ import annotations

from ..registry import Registry

INTEGRATORS = Registry("integrator")

# 2nd order, 1 gradient per step
INTEGRATORS.register("leapfrog", ((1.0,), (0.5, 0.5)))

# 2nd order minimum-error 2-stage scheme (McLachlan 1995), 2 gradients per step
_L = 0.1931833275037836
INTEGRATORS.register("mclachlan2", ((0.5, 0.5), (_L, 1.0 - 2.0 * _L, _L)))

# 4th order (Yoshida 1990 / Forest-Ruth), 3 gradients per step
_W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_W0 = 1.0 - 2.0 * _W1
INTEGRATORS.register(
    "yoshida4",
    ((_W1, _W0, _W1), (_W1 / 2, (_W1 + _W0) / 2, (_W0 + _W1) / 2, _W1 / 2)),
)


def integrator_step(value_and_grad, coeffs, eps, x, p, grad, inv_mass=1.0):
    """Advance (x, p) by one step. Returns (x, p, logdensity, grad) at the end."""
    a, b = coeffs
    logp = None
    for ai, bi in zip(a, b[:-1]):
        p = p + eps * bi * grad
        x = x + eps * ai * inv_mass * p
        logp, grad = value_and_grad(x)
    p = p + eps * b[-1] * grad
    return x, p, logp, grad
