"""Flat LCDM background: expansion rate, linear growth factor and growth rate.
All functions are pure JAX and differentiable in the cosmological parameters."""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class Cosmology(NamedTuple):
    h: float
    omega_m: float
    omega_b: float
    n_s: float
    sigma8: float
    t_cmb: float = 2.7255

    @classmethod
    def from_config(cls, c) -> "Cosmology":
        return cls(c.h, c.omega_m, c.omega_b, c.n_s, c.sigma8, c.t_cmb)


_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)
_GL_T = 0.5 * (_GL_X + 1.0)  # nodes on [0, 1]
_GL_W = 0.5 * _GL_W


def efunc(a, cosmo: Cosmology):
    """E(a) = H(a)/H0 for flat LCDM (radiation neglected)."""
    return jnp.sqrt(cosmo.omega_m * a**-3 + (1.0 - cosmo.omega_m))


def _growth_unnorm(a, cosmo: Cosmology):
    # D(a) ∝ E(a) ∫_0^a da' / (a' E(a'))^3 ; substitute a' = a t^2 so the
    # integrand is smooth at the origin.
    t = jnp.asarray(_GL_T)
    ap = a * t**2
    integrand = 2.0 * a * t / (ap * efunc(ap, cosmo)) ** 3
    return efunc(a, cosmo) * jnp.sum(jnp.asarray(_GL_W) * integrand)


def growth_factor(a, cosmo: Cosmology):
    """Linear growth factor normalised to D(a=1) = 1."""
    return _growth_unnorm(a, cosmo) / _growth_unnorm(1.0, cosmo)


def growth_rate(a, cosmo: Cosmology):
    """f = dlnD/dlna."""
    a = jnp.asarray(a, dtype=jnp.result_type(float))
    dlnD = jax.grad(lambda x: jnp.log(_growth_unnorm(x, cosmo)))(a)
    return a * dlnD
