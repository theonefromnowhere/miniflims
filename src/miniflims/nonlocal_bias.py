"""Non-local galaxy bias: operators O_i(x) that are not functions of the local
density, with coefficients b_i sampled together with the local bias
parameters (nuisance block):

    B_tot = B(rho) exp(sum_i b_i O_i)          form "exp" (default, positive)
    B_tot = max(B(rho) + sum_i b_i O_i, floor) form "linear"

To first order both give delta_g = b1 delta + sum_i b_i O_i. ``B_tot`` then
replaces B in lambda = nbar R B / <B> (model.py).

Terms (galaxies.bias.non_local.terms; coefficients in galaxies.bias.params)

    png        b_phi f_NL psi_alpha,   psi_alpha(k) = (k / k_star)^alpha Phi_G(k)

               The scale-dependent bias of primordial non-Gaussianity (Dalal
               et al. 2008; Slosar et al. 2008; Schmidt & Kamionkowski 2010):
               a squeezed limit B ~ (k_L/k_S)^alpha P(k_L) P(k_S) modulates
               the small-scale power, hence the tracer abundance, by the long-
               wavelength primordial potential. In Fourier space
               Delta b(k) = b_phi f_NL (k/k_star)^alpha / M(k), with M(k) the
               Poisson kernel of png.py (delta = M Phi). For the local template
               (alpha = 0) and a universal mass function b_phi = 2 delta_c
               (b1 - 1), delta_c = 1.686. alpha comes from png.template
               (local 0, orthogonal 1, feature 1, equilateral 2,
               equilateral_feature 2) unless non_local.png_alpha is set.
               Phi_G is the Gaussian primordial potential of the white noise
               (the non-Gaussian part would add O(f_NL^2)).

    laplacian  b_lap nabla^2 delta / k_star^2

               The leading higher-derivative bias, applied to the evolved
               density on the mesh.

Only the product b_phi f_NL enters the png term: with both sampled, the
information that separates them comes from the primordial bispectrum.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .grid import Grid
from .params import ParamSpec
from .registry import Registry

NONLOCAL_TERMS = Registry("non-local bias term")
NONLOCAL_TERMS.register("png", ParamSpec("b_phi", 0.0, -100.0, 100.0, 2.0))
NONLOCAL_TERMS.register("laplacian", ParamSpec("b_lap", 0.0, -20.0, 20.0, 0.5))

FORMS = ("exp", "linear")


def term_specs(terms) -> list[ParamSpec]:
    """Coefficient specs of the configured terms, in order."""
    if len(set(terms)) != len(tuple(terms)):
        raise ValueError(f"galaxies.bias.non_local.terms has duplicates: {list(terms)}")
    return [NONLOCAL_TERMS.get(t) for t in terms]


def fourier_resample(field, n_out: int):
    """Band-limited resampling of a periodic real cube to n_out^3 cells (the
    modes with |k_i| < min(n, n_out)/2 per axis; Nyquist planes dropped)."""
    n = field.shape[0]
    if n_out == n:
        return field
    h = min(n, n_out) // 2
    full = np.r_[0:h, -(h - 1):0]
    half = np.arange(h)
    fk = jnp.fft.rfftn(field)
    src = fk[np.ix_(full % n, full % n, half)]
    out = jnp.zeros((n_out, n_out, n_out // 2 + 1), fk.dtype)
    out = out.at[np.ix_(full % n_out, full % n_out, half)].set(src * (n_out / n) ** 3)
    return jnp.fft.irfftn(out, s=(n_out,) * 3)


class NonLocalBias:
    """Operator fields of the configured terms, on the mesh."""

    def __init__(self, cfg, grid: Grid, mesh: Grid, amplitude, png=None):
        """``cfg``: galaxies.bias.non_local; ``amplitude``: white noise ->
        Gaussian linear density in rfftn layout (FieldModel.amplitude);
        ``png``: the PNGModel (needed by the png term)."""
        if cfg.form not in FORMS:
            raise ValueError(f"galaxies.bias.non_local.form must be one of {FORMS}, not '{cfg.form}'")
        if cfg.k_star <= 0:
            raise ValueError("galaxies.bias.non_local.k_star must be positive")
        self.cfg = cfg
        self.terms = tuple(cfg.terms)
        self.specs = term_specs(self.terms)
        self.form = cfg.form
        self.grid, self.mesh = grid, mesh
        self.alpha = None
        if "png" in self.terms:
            if png is None:
                raise ValueError("galaxies.bias.non_local.terms contains 'png' but png.enabled is false")
            self.alpha = float(cfg.png_alpha if cfg.png_alpha is not None else png.template.squeezed_alpha)
            k = np.asarray(grid.kmag())
            scale = np.where(k > 0, (np.where(k > 0, k, 1.0) / cfg.k_star) ** self.alpha, 0.0)
            # white noise -> psi_alpha = (k/k_star)^alpha Phi_G, Phi_G = delta_G / M
            self._psi_filter = jnp.asarray(amplitude) * png.inv_M * jnp.asarray(scale)
        if "laplacian" in self.terms:
            self._lap = -jnp.asarray(np.asarray(mesh.kmag()) ** 2) / cfg.k_star**2

    @property
    def needs_potential(self) -> bool:
        return "png" in self.terms

    def potential(self, x):
        """psi_alpha on the mesh from the white noise (None without the png
        term). Independent of the PNG parameters: compute once per field."""
        if not self.needs_potential:
            return None
        psi = jnp.fft.irfftn(jnp.fft.rfftn(x) * self._psi_filter, s=x.shape)
        return fourier_resample(psi, self.mesh.n)

    def operators(self, rho, psi=None, f_nl=None):
        """(n_terms, n_mesh^3) operator fields, in the order of ``terms``."""
        out = []
        for t in self.terms:
            if t == "png":
                out.append(f_nl * psi)
            elif t == "laplacian":
                out.append(jnp.fft.irfftn(jnp.fft.rfftn(rho - 1.0) * self._lap, s=rho.shape))
        return jnp.stack(out)

    def apply(self, b, coeffs, ops, floor: float):
        """Combine the local bias ``b`` = B(rho) with sum_i coeffs_i O_i."""
        s = jnp.tensordot(coeffs, ops, axes=1)
        if self.form == "exp":
            return b * jnp.exp(s)
        return jnp.maximum(b + s, floor)
