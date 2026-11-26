"""Autodiff correctness: jax.grad through the full generator vs finite differences.

Backs the "differentiable end-to-end" design goal. We differentiate a scalar
summary of the generated field w.r.t. (i) f_NL, (ii) the oscillatory shape
parameter omega (non-trivially, through the Schwinger lax.scan), and (iii) the
white-noise field (directional derivative, since it is high-dimensional), and
check each against a central finite difference. Small grid + coarse t-grid for
speed -- AD and FD differentiate the *same* function, so accuracy of the t-grid
is irrelevant to the comparison.
"""
import numpy as np
import jax
import jax.numpy as jnp

from jaxpng import api, bispectra as B
from jaxpng.config import Box

_BOX = Box(N=16, L=1000.0)
_PE = 8.0


def _stat(phi_k):
    return jnp.sum(jnp.abs(phi_k) ** 2)


def test_grad_wrt_fNL(cosmo):
    n_s = cosmo.n_s
    w = jax.random.normal(jax.random.PRNGKey(0), _BOX.rshape)
    tmpl = B.local(n_s)

    def stat(fNL):
        return _stat(api.generate_phi(w, cosmo, _BOX, tmpl, n_s, fNL, per_efold=_PE))

    f0, h = 50.0, 1.0
    g_ad = float(jax.grad(stat)(f0))
    g_fd = float((stat(f0 + h) - stat(f0 - h)) / (2 * h))
    assert np.isclose(g_ad, g_fd, rtol=1e-4, atol=1e-12 + abs(g_ad) * 1e-4)


def test_grad_wrt_omega_feature(cosmo):
    n_s = cosmo.n_s
    w = jax.random.normal(jax.random.PRNGKey(1), _BOX.rshape)

    def stat(omega):
        tmpl = B.feature(n_s, omega)  # omega flows through the Schwinger scan
        return _stat(api.generate_phi(w, cosmo, _BOX, tmpl, n_s, 100.0, per_efold=_PE))

    w0, h = 20.0, 1e-3
    g_ad = float(jax.grad(stat)(w0))
    g_fd = float((stat(w0 + h) - stat(w0 - h)) / (2 * h))
    assert np.isclose(g_ad, g_fd, rtol=2e-3, atol=abs(g_ad) * 2e-3 + 1e-20)


def test_grad_wrt_white_noise_directional(cosmo):
    n_s = cosmo.n_s
    w = jax.random.normal(jax.random.PRNGKey(2), _BOX.rshape)
    v = jax.random.normal(jax.random.PRNGKey(3), _BOX.rshape)
    v = v / jnp.linalg.norm(v)
    tmpl = B.local(n_s)

    def stat(ww):
        return _stat(api.generate_phi(ww, cosmo, _BOX, tmpl, n_s, 100.0, per_efold=_PE))

    g = jax.grad(stat)(w)
    assert g.shape == _BOX.rshape and bool(jnp.all(jnp.isfinite(g)))
    dd_ad = float(jnp.vdot(g, v))
    h = 1e-3
    dd_fd = float((stat(w + h * v) - stat(w - h * v)) / (2 * h))
    assert np.isclose(dd_ad, dd_fd, rtol=2e-3, atol=abs(dd_ad) * 2e-3 + 1e-20)
