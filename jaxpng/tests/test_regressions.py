"""Regression tests for bugs found in the November 2026 audit. They use an
analytic (BBKS) transfer table, so they run without the GENGARS input files.

1. Gradients with respect to cosmological parameters were NaN: the k = 0 mode
   went through 0/0 (power_phi) and 0^(n_s - 4) (P_kernel) in the branch that
   jnp.where discards, whose derivative still propagates.
2. power_phi returned NaN above the transfer table (T = 0 there, 0/0), so
   Phi^G was NaN for boxes resolving k beyond the table.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from jaxpng import api, bispectra, cosmology as C, gaussian, phing
from jaxpng.config import Box
from jaxpng.grids import P_kernel

BASE = dict(omega_m=0.3175, omega_l=0.6825, omega_b=0.049, hubble=0.6711, n_s=0.9624)


def bbks_table(kmax=100.0, omega_m=0.3175, h=0.6711):
    k = np.logspace(-5, np.log10(kmax), 400)
    q = k / (omega_m * h)
    t = np.log(1 + 2.34 * q) / (2.34 * q) * (1 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4) ** -0.25
    return jnp.asarray(k), jnp.asarray(t / t[0])


def cosmo(**over):
    k, t = bbks_table()
    return C.Cosmo(**{**BASE, **over}, k_T=k, T=t)


@pytest.mark.parametrize("sigma8_norm", [True, False])
def test_power_phi_equals_matter_over_poisson_squared(sigma8_norm):
    c = cosmo(sigma8_norm=sigma8_norm)
    k = jnp.logspace(-4, 1, 50)
    direct = C.Pk_matter(c, k) / C.poisson_M(c, k) ** 2
    np.testing.assert_allclose(C.power_phi(c, k), direct, rtol=1e-10)
    assert float(C.power_phi(c, jnp.asarray(0.0))) == 0.0


def test_power_phi_and_phi_g_finite_above_the_transfer_table():
    k, t = bbks_table(kmax=0.8)
    short = C.Cosmo(**BASE, k_T=k, T=t)
    assert np.all(np.isfinite(np.asarray(C.power_phi(short, jnp.asarray([0.5, 2.0, 50.0])))))
    box = Box(N=32, L=100.0)  # k_max = 1.7 h/Mpc, beyond the table
    w = jax.random.normal(jax.random.PRNGKey(0), box.rshape)
    assert bool(jnp.all(jnp.isfinite(gaussian.phiG_from_white_noise(w, short, box))))


def test_p_kernel_gradient_is_finite():
    k = jnp.asarray([0.0, 0.01, 0.1])
    g = jax.grad(lambda ns: jnp.sum(P_kernel(k, ns)))(0.96)
    assert np.isfinite(float(g))


@pytest.mark.parametrize("param", ["omega_m", "sigma8", "n_s"])
def test_cosmology_gradients_match_finite_differences(param):
    box = Box(N=16, L=500.0)
    w = jax.random.normal(jax.random.PRNGKey(1), box.rshape)
    k_t, t = bbks_table()
    x0 = {"omega_m": 0.3175, "sigma8": 0.834, "n_s": 0.9624}[param]

    def summary(x):
        pars = {**BASE, param: x}
        if param == "omega_m":
            pars["omega_l"] = 1.0 - x
        c = C.Cosmo(**pars, k_T=k_t, T=t)
        n_s = pars["n_s"]
        phi = api.generate_phi(w, c, box, bispectra.equilateral(n_s), n_s, 100.0, per_efold=6.0)
        return jnp.sum(jnp.abs(phi) ** 2)

    g = float(jax.grad(summary)(x0))
    h = 1e-5
    fd = float((summary(x0 + h) - summary(x0 - h)) / (2 * h))
    assert np.isfinite(g)
    assert g == pytest.approx(fd, rel=1e-4)
