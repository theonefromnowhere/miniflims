"""Tests for jaxpng.cosmology (ported from gengars/src/cosmo_functions.c)."""
import numpy as np
import jax.numpy as jnp

from jaxpng import cosmology as C


def test_transfer_limits(cosmo):
    assert np.isclose(float(C.trans(cosmo, 1e-6)), 1.0, atol=1e-6)   # T(k->0)=1
    assert np.isclose(float(C.trans(cosmo, float(cosmo.k_T[0]))), 1.0, atol=1e-3)
    assert float(C.trans(cosmo, 100.0)) == 0.0                       # above table
    # transfer decreases with k over the table
    ks = jnp.array([1e-3, 1e-2, 1e-1, 1.0])
    T = np.asarray(C.trans(cosmo, ks))
    assert np.all(np.diff(T) < 0)


def test_growth_monotonic(cosmo):
    a = np.array([0.1, 0.3, 0.6, 1.0])
    D = np.array([float(C.growth_D(cosmo, ai)) for ai in a])
    assert np.all(np.diff(D) > 0) and D[-1] > 0
    # matter domination: D ~ a at small a
    assert abs(float(C.growth_D(cosmo, 0.05)) / 0.05 - float(C.growth_D(cosmo, 0.1)) / 0.1) < 0.05


def test_sigma8_recovered(cosmo):
    """Integrating the normalized P_matter against the R=8 top-hat returns sigma8."""
    k = jnp.logspace(-4, 2, 40000)
    x = k * 8.0
    W = 3.0 * (jnp.sin(x) / x ** 3 - jnp.cos(x) / x ** 2)
    s8 = jnp.sqrt(jnp.trapezoid(k * k * W * W * C.Pk_matter(cosmo, k), k) / (2.0 * np.pi ** 2))
    assert abs(float(s8) / cosmo.sigma8 - 1.0) < 5e-3


def test_power_phi_low_k_scaling(cosmo):
    """P_Phi(k) ~ k^(n_s-4) on large scales (T->1)."""
    k = jnp.array([1e-3, 2e-3])
    P = np.asarray(C.power_phi(cosmo, k))
    slope = np.log(P[1] / P[0]) / np.log(2.0)
    assert abs(slope - (cosmo.n_s - 4.0)) < 0.05
