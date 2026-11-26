"""End-to-end generation: white noise -> Phi = Phi^G + f_NL Phi^NG.

A thin convenience wrapper over :func:`jaxpng.gaussian.phiG_from_white_noise` and
:func:`jaxpng.phing.phiNG`. Pure JAX, so it is jit/grad/vmap-able end to end --
differentiable w.r.t. the white-noise field, f_NL, the bispectrum-shape
parameters (e.g. the oscillatory omega), and cosmology.
"""
from __future__ import annotations

from .gaussian import phiG_from_white_noise
from .phing import phiNG


def generate_phi(white_noise, cosmo, box, template, n_s, f_NL,
                 eps=1e-3, per_efold=20.0, return_parts=False):
    """Primordial potential Phi(k) = Phi^G(k) + f_NL * Phi^NG(k) (rfftn layout).

    Parameters
    ----------
    white_noise : real (N, N, N) ~ N(0,1).
    cosmo : Cosmo
    box : Box
    template : list of separable-factor terms (see jaxpng.bispectra).
    n_s : float -- spectral index (kernel P(k)=k^(n_s-4)).
    f_NL : float
    eps, per_efold : Schwinger t-grid knobs.
    return_parts : if True, also return (Phi^G, Phi^NG).

    Returns
    -------
    phi_k  (and, if return_parts, phiG_k, phiNG_k) -- complex (N, N, N//2+1).
    """
    phiG_k = phiG_from_white_noise(white_noise, cosmo, box)
    phiNG_k = phiNG(phiG_k, template, box, n_s, eps=eps, per_efold=per_efold)
    phi_k = phiG_k + f_NL * phiNG_k
    if return_parts:
        return phi_k, phiG_k, phiNG_k
    return phi_k
