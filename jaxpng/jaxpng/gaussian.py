"""Gaussian primordial potential Phi^G from a white-noise field.

Differentiable, parametrized by a real-space white-noise field w(x) ~ N(0,1).
Replaces the GSL-RNG construction in gengars/src/generate_phiG_utils.c; the C
field is not reproduced bit-for-bit (different RNG), only its statistics.
"""
from __future__ import annotations

import jax.numpy as jnp

from .grids import k_grid
from .cosmology import power_phi


def phiG_from_white_noise(white_noise, cosmo, box):
    """Gaussian potential Phi^G(k) (rfftn layout) with P_Phi = power_phi(k).

    Construction: Phi^G(k) = rfftn(w) * sqrt(P_Phi(k)) * (N/L)^{3/2}, DC set to 0.
    The prefactor (N/L)^{3/2} is the discrete-field normalization that makes the
    measured power spectrum (jaxpng.measure convention, P = <|d_k|^2>(L/N^2)^3)
    recover P_Phi(k): a unit white-noise field gives measured P = (L/N)^3.

    Parameters
    ----------
    white_noise : real array (N, N, N), iid N(0,1).
    cosmo : Cosmo
    box : Box

    Returns
    -------
    phi_k : complex array (N, N, N//2+1) -- Phi^G in rfftn layout (real in x-space).
    """
    _, _, _, kmag = k_grid(box)
    Pphi = power_phi(cosmo, kmag)  # (N,N,N//2+1), 0 at k=0
    amp = jnp.sqrt(Pphi) * (box.N / box.L) ** 1.5
    phi_k = jnp.fft.rfftn(white_noise) * amp
    return phi_k.at[0, 0, 0].set(0.0 + 0.0j)
