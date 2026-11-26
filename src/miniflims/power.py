"""Linear matter power spectrum at z=0, normalised to sigma8.
k in h/Mpc, P in (Mpc/h)^3."""
from __future__ import annotations

import jax.numpy as jnp

from .cosmology import Cosmology
from .transfer import TRANSFER_FUNCTIONS


def tophat_window(x):
    """Fourier transform of a real-space spherical top-hat."""
    xs = jnp.where(x < 1e-3, 1.0, x)
    w = 3.0 * (jnp.sin(xs) - xs * jnp.cos(xs)) / xs**3
    return jnp.where(x < 1e-3, 1.0 - x**2 / 10.0, w)


def _unnormalised(k, cosmo: Cosmology, transfer: str):
    t = TRANSFER_FUNCTIONS.get(transfer)(k, cosmo)
    ks = jnp.where(k > 0, k, 1.0)
    return jnp.where(k > 0, ks**cosmo.n_s * t**2, 0.0)


def sigma_r(pk_fn, r, k_min=1e-5, k_max=1e2, n_k=2048):
    """RMS of the linear field smoothed with a top-hat of radius r [Mpc/h]."""
    lnk = jnp.linspace(jnp.log(k_min), jnp.log(k_max), n_k)
    k = jnp.exp(lnk)
    integrand = k**3 * pk_fn(k) * tophat_window(k * r) ** 2 / (2.0 * jnp.pi**2)
    dlnk = lnk[1] - lnk[0]
    integral = dlnk * (jnp.sum(integrand) - 0.5 * (integrand[0] + integrand[-1]))
    return jnp.sqrt(integral)


def linear_power(
    k, cosmo: Cosmology, transfer: str = "eisenstein_hu", k_min=1e-5, k_max=1e2, n_k=2048
):
    """P_lin(k, z=0) with sigma(8 Mpc/h) = cosmo.sigma8."""
    k = jnp.asarray(k)
    s8 = sigma_r(lambda q: _unnormalised(q, cosmo, transfer), 8.0, k_min, k_max, n_k)
    return (cosmo.sigma8 / s8) ** 2 * _unnormalised(k, cosmo, transfer)
