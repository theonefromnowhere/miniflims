"""Cosmology: transfer function, matter P(k), growth, Poisson kernel.

Ported from ``gengars/src/cosmo_functions.c``. Units follow the C code: H0 =
100/c in h/Mpc (the dimensionless h cancels), k in h/Mpc, P(k) in (Mpc/h)^3.
All functions are JAX and differentiable in k (and, if a modified Cosmo is
passed, in the cosmological parameters).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import jax.numpy as jnp

C_LIGHT = 299792.458          # km/s
H0 = 100.0 / C_LIGHT          # h/Mpc (hubble drops out, as in the C code)


@dataclass(frozen=True)
class Cosmo:
    """Cosmological parameters + (normalized) transfer function table."""

    omega_m: float
    omega_l: float
    omega_b: float
    hubble: float
    n_s: float
    sigma8: float = 0.834
    A_s: float = 2.215e-9
    k_pivot: float = 0.05          # 1/Mpc (per the C code / parameters.txt)
    sigma8_norm: bool = True
    k_T: Optional[jnp.ndarray] = None   # transfer k (h/Mpc), ascending
    T: Optional[jnp.ndarray] = None     # transfer T, normalized so T(k->0)=1


def load_transfer(tkfile):
    """Load a 2-column (k, T) transfer file; normalize T(k->0)=1. Returns (k, T)."""
    data = np.loadtxt(tkfile, comments="#")
    k, T = data[:, 0], data[:, 1]
    T = T / T[0]
    return jnp.asarray(k), jnp.asarray(T)


def make_cosmo(tkfile, **kwargs):
    """Build a Cosmo from a transfer file and parameter overrides."""
    k_T, T = load_transfer(tkfile)
    return Cosmo(k_T=k_T, T=T, **kwargs)


def trans(cosmo, k):
    """Matter transfer function T(k), log-log interpolated; T=1 below the table,
    T=0 above it (matching gengars/src/cosmo_functions.c::trans)."""
    logk = jnp.log(jnp.maximum(k, 1e-30))
    out = jnp.exp(jnp.interp(logk, jnp.log(cosmo.k_T), jnp.log(cosmo.T)))
    out = jnp.where(k < cosmo.k_T[0], 1.0, out)
    out = jnp.where(k > cosmo.k_T[-1], 0.0, out)
    return out


def Hubble(cosmo, a):
    """H(a) (the C code's H), flat wCDM with w0=-1, wa=0."""
    omega_k = 1.0 - cosmo.omega_m - cosmo.omega_l
    return cosmo.hubble * jnp.sqrt(
        omega_k / a ** 2 + cosmo.omega_m / a ** 3 + cosmo.omega_l
    )


def growth_D(cosmo, a, n=4096):
    """Linear growth factor D(a) = 5/2 Om H(a)/h * int_0^a dx (x H(x)/h)^-3.

    Unnormalized (the C code's D_plus); D_plus(1) is used in the Poisson kernel.
    """
    xs = jnp.linspace(1e-7, a, n)
    integrand = (xs * Hubble(cosmo, xs) / cosmo.hubble) ** (-3.0)
    integ = jnp.trapezoid(integrand, xs)
    return 2.5 * cosmo.omega_m * (Hubble(cosmo, a) / cosmo.hubble) * integ


def _sigma2_R(cosmo, R=8.0, kmin=1e-4, kmax=20.0, n=30000):
    """sigma^2(R) for the UNNORMALIZED spectrum k^n_s T(k)^2 (top-hat window)."""
    k = jnp.logspace(np.log10(kmin), np.log10(kmax), n)
    x = k * R
    W = 3.0 * (jnp.sin(x) / x ** 3 - jnp.cos(x) / x ** 2)
    integrand = k * k * W * W * k ** cosmo.n_s * trans(cosmo, k) ** 2
    return jnp.trapezoid(integrand, k) / (2.0 * np.pi ** 2)


def norm_pk(cosmo):
    """Amplitude so that sigma8 is reproduced (sigma8 normalization branch)."""
    return cosmo.sigma8 ** 2 / _sigma2_R(cosmo, R=8.0)


def Pk_matter(cosmo, k):
    """Linear matter power spectrum P(k) (Mpc/h)^3."""
    k = jnp.asarray(k)
    ks = jnp.where(k > 0, k, 1.0)  # k^n_s at k = 0 would give NaN gradients
    if cosmo.sigma8_norm:
        p = norm_pk(cosmo) * ks ** cosmo.n_s * trans(cosmo, ks) ** 2
    else:
        D1 = growth_D(cosmo, 1.0)
        p = (
            2.0 * np.pi ** 2 * (4.0 / 25.0)
            * (D1 * trans(cosmo, ks) / H0 ** 2 / cosmo.omega_m) ** 2
            * cosmo.A_s
            * (ks * cosmo.hubble / cosmo.k_pivot) ** (cosmo.n_s - 1.0)
            * ks
        )
    return jnp.where(k > 0, p, 0.0)


def poisson_M(cosmo, k):
    """Poisson kernel M(k): delta(k) = M(k) Phi(k), Phi(k) = delta(k)/M(k).

    M(k) = (2/3) k^2 D(z=0) T(k) / (Omega_m H0^2).
    """
    D1 = growth_D(cosmo, 1.0)
    return (2.0 / 3.0) * k ** 2 * D1 * trans(cosmo, k) / (cosmo.omega_m * H0 ** 2)


def power_phi(cosmo, k):
    """Physical primordial-potential power spectrum P_Phi(k) = P_matter(k)/M(k)^2.

    T(k) cancels between P_matter and M(k)^2, so this is evaluated in the
    T-free form (the same numbers inside the transfer table): it stays finite
    above the table, where T = 0 would give 0/0, and it has no 0/0 at k = 0,
    which would make gradients with respect to the cosmology NaN.
        sigma8 branch:  norm k^(n_s-4) (3 Omega_m H0^2 / (2 D1))^2
        A_s branch:     (9/25) 2 pi^2 A_s (k h / k_pivot)^(n_s-1) / k^3
    """
    k = jnp.asarray(k)
    ks = jnp.where(k > 0, k, 1.0)
    if cosmo.sigma8_norm:
        D1 = growth_D(cosmo, 1.0)
        p = norm_pk(cosmo) * ks ** (cosmo.n_s - 4.0) * (1.5 * cosmo.omega_m * H0 ** 2 / D1) ** 2
    else:
        p = (9.0 / 25.0) * 2.0 * np.pi ** 2 * cosmo.A_s * (ks * cosmo.hubble / cosmo.k_pivot) ** (cosmo.n_s - 1.0) / ks ** 3
    return jnp.where(k > 0, p, 0.0)
