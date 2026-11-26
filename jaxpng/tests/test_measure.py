"""Tests for the pure-JAX P(k)/B(k) estimators in jaxpng.measure.

No Pylians dependency: the estimators are validated against the analytic
white-noise power spectrum and against internal consistency, and checked for
jit/grad/vmap compatibility (the differentiability requirement).
"""
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from jaxpng import measure


def test_power_spectrum_white_noise_amplitude():
    """White noise (iid N(0,1) per cell) has P(k) = (BoxSize/N)^3 in all shells."""
    N, L = 64, 1000.0
    field = jnp.asarray(np.random.RandomState(0).standard_normal((N, N, N)))
    k, Pk = measure.power_spectrum(field, L)
    analytic = (L / N) ** 3
    mean_hi = float(np.mean(Pk[5:]))  # well-sampled shells
    assert abs(mean_hi / analytic - 1.0) < 0.05


def test_power_spectrum_shapes_and_bins():
    N, L = 32, 1000.0
    field = jnp.asarray(np.random.RandomState(2).standard_normal((N, N, N)))
    k, Pk = measure.power_spectrum(field, L)
    nbins = int(np.floor(np.sqrt(3.0) * (N // 2)))  # kmax, == Pylians (DC dropped)
    assert k.shape == (nbins,) and Pk.shape == (nbins,)
    assert np.all(np.diff(k) > 0) and np.all(Pk > 0)


def test_power_spectrum_real_vs_kspace_input():
    N, L = 32, 1000.0
    real = jnp.asarray(np.random.RandomState(3).standard_normal((N, N, N)))
    fk = jnp.fft.rfftn(real)
    k1, P1 = measure.power_spectrum(real, L)
    k2, P2 = measure.power_spectrum(fk, L)
    np.testing.assert_allclose(np.asarray(k1), np.asarray(k2), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(P1), np.asarray(P2), rtol=1e-10)


def test_power_spectrum_jit_grad_vmap():
    N, L = 16, 1000.0
    field = jnp.asarray(np.random.RandomState(4).standard_normal((N, N, N)))

    k, Pk = jax.jit(partial(measure.power_spectrum, BoxSize=L))(field)
    assert np.all(np.isfinite(np.asarray(Pk)))

    g = jax.grad(lambda f: jnp.sum(measure.power_spectrum(f, L)[1]))(field)
    assert g.shape == (N, N, N) and bool(jnp.all(jnp.isfinite(g)))

    batch = jnp.asarray(np.random.RandomState(5).standard_normal((3, N, N, N)))
    _, Pks = jax.vmap(partial(measure.power_spectrum, BoxSize=L))(batch)
    assert Pks.shape[0] == 3


def test_bispectrum_equilateral_sides_equal():
    N, L = 32, 1000.0
    field = jnp.asarray(np.random.RandomState(6).standard_normal((N, N, N)))
    out = measure.bispectrum(field, L, 0.1, 0.1, measure.EQUILATERAL_THETA)
    assert set(out) == {"B", "Q", "k", "Pk", "triangles"}
    k = np.asarray(out["k"])
    assert np.allclose(k, k[0], rtol=1e-6)
    assert np.all(np.isfinite(np.asarray(out["B"])))


def test_bispectrum_amplitude_local_model():
    """Validate B's absolute normalization with the local model.

    For f = g + fNL*(g^2 - <g^2>) with a small-amplitude (perturbative) Gaussian g,
    the tree-level equilateral bispectrum is B = 2 fNL * 3 * P_g^2 = 6 fNL P_g^2.
    P(k) is already calibrated (white-noise test), so B/(6 fNL P_g^2) ~ 1 pins B's
    normalization. Uses well-sampled shells and averages over realizations.
    """
    N, L, eps, fNL, nreal = 48, 1000.0, 0.05, 2.0, 40
    ks = [0.12, 0.18]  # python floats: static args to bispectrum under vmap
    keys = jax.random.split(jax.random.PRNGKey(0), nreal)

    def ratio(key):
        g = eps * jax.random.normal(key, (N, N, N))
        f = g + fNL * (g * g - jnp.mean(g * g))
        _, Pg = measure.power_spectrum(g, L)
        Pg = jnp.mean(Pg[5:])  # P_g is flat (white) -> average well-sampled shells
        Bs = jnp.array(
            [measure.bispectrum(f, L, k, k, measure.EQUILATERAL_THETA)["B"][0] for k in ks]
        )
        return Bs / (6.0 * fNL * Pg ** 2)

    r = np.asarray(jnp.mean(jax.vmap(ratio)(keys), axis=0))
    assert np.all(np.abs(r - 1.0) < 0.2), f"B/(6 fNL P_g^2) = {r}, expected ~1"


def test_bispectrum_multi_theta_and_grad():
    N, L = 16, 1000.0
    field = jnp.asarray(np.random.RandomState(7).standard_normal((N, N, N)))
    thetas = jnp.linspace(0.5, np.pi - 0.2, 4)
    out = measure.bispectrum(field, L, 0.05, 0.07, thetas)
    assert np.asarray(out["B"]).shape == (4,)

    gb = jax.grad(lambda f: jnp.sum(measure.bispectrum(f, L, 0.05, 0.07, thetas)["B"]))(field)
    assert bool(jnp.all(jnp.isfinite(gb)))
