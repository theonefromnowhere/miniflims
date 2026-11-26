"""Tests for the Schwinger Phi^NG core (jaxpng.phing), local template.

Anchor: for the local template W=1, so Phi^NG(x) must equal Phi^G(x)^2 - <Phi^G^2>.
This validates both the absolute normalization and the t-integral convergence.
"""
import numpy as np
import jax
import jax.numpy as jnp

from jaxpng import gaussian, bispectra, phing


def _phiG(cosmo, box, seed=0):
    w = jax.random.normal(jax.random.PRNGKey(seed), box.rshape)
    return gaussian.phiG_from_white_noise(w, cosmo, box)


def test_local_anchor_matches_square(cosmo, box):
    """Phi^NG ~ Phi^G^2 - <Phi^G^2> for the local template (default t-grid)."""
    n_s = cosmo.n_s
    phiG_k = _phiG(cosmo, box)
    phiG_x = jnp.fft.irfftn(phiG_k, s=box.rshape)
    target = phiG_x ** 2 - jnp.mean(phiG_x ** 2)

    ng_k = phing.phiNG(phiG_k, bispectra.local(n_s), box, n_s, per_efold=20.0)
    ng_x = jnp.fft.irfftn(ng_k, s=box.rshape)

    rel = float(jnp.linalg.norm(ng_x - target) / jnp.linalg.norm(target))
    assert rel < 2e-3, f"local anchor rel L2 error = {rel}"
    # amplitude (no factor-of-2 / normalization error)
    assert abs(float(jnp.std(ng_x) / jnp.std(target)) - 1.0) < 3e-3


def test_t_convergence(cosmo, box):
    """Refining the Schwinger t-grid reduces the anchor error (trapezoid order)."""
    n_s = cosmo.n_s
    phiG_k = _phiG(cosmo, box)
    phiG_x = jnp.fft.irfftn(phiG_k, s=box.rshape)
    target = phiG_x ** 2 - jnp.mean(phiG_x ** 2)
    tmpl = bispectra.local(n_s)

    def rel(pe):
        ng = jnp.fft.irfftn(phing.phiNG(phiG_k, tmpl, box, n_s, per_efold=pe), s=box.rshape)
        return float(jnp.linalg.norm(ng - target) / jnp.linalg.norm(target))

    coarse, fine = rel(10.0), rel(40.0)
    assert fine < coarse                      # converges
    assert coarse / fine > 4.0                # ~2nd order (each 2x -> ~4x smaller)


def test_phing_differentiable(cosmo, box):
    n_s = cosmo.n_s
    tmpl = bispectra.local(n_s)
    w = jax.random.normal(jax.random.PRNGKey(1), box.rshape)

    def scalar(w):
        pg = gaussian.phiG_from_white_noise(w, cosmo, box)
        return jnp.sum(jnp.abs(phing.phiNG(pg, tmpl, box, n_s, per_efold=8.0)) ** 2)

    g = jax.grad(scalar)(w)
    assert g.shape == box.rshape and bool(jnp.all(jnp.isfinite(g)))


def test_log_t_grid_converges_exponentially():
    """Trapezoid in ln t (schwinger_t_grid_log): local anchor Phi^NG = Phi^G^2 - <Phi^G^2>
    to < 1e-4 with 32 nodes (the t-trapezoid default needs ~360 nodes for 4e-4)."""
    from jaxpng.config import Box
    from jaxpng.grids import k_grid, P_kernel

    box, n_s = Box(N=32, L=1000.0), 0.9624
    _, _, _, kmag = k_grid(box)
    w = jax.random.normal(jax.random.PRNGKey(0), box.rshape)
    phiG_k = (jnp.fft.rfftn(w) * jnp.sqrt(P_kernel(kmag, n_s))).at[0, 0, 0].set(0.0)
    phiG_x = jnp.fft.irfftn(phiG_k, s=box.rshape)
    target = phiG_x ** 2 - jnp.mean(phiG_x ** 2)
    tmpl = bispectra.local(n_s)

    def rel(n, eps):
        grid = phing.schwinger_t_grid_log(box, n, eps)
        ng = jnp.fft.irfftn(phing.phiNG(phiG_k, tmpl, box, n_s, t_grid=grid), s=box.rshape)
        return float(jnp.linalg.norm(ng - target) / jnp.linalg.norm(target))

    e16, e24, e32 = rel(16, 1e-3), rel(24, 1e-4), rel(32, 1e-5)
    assert e32 < 1e-4 and e24 < 5e-4
    assert e16 / e32 > 30  # far faster than second order (which would give ~4)
    # the default t-grid is unchanged
    t, wt = phing.schwinger_t_grid(box)
    assert len(t) == 361
