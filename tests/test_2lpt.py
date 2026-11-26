"""Second-order LPT forward model (forward.model = 2lpt)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from miniflims.config import ForwardConfig, RSDConfig
from miniflims.cosmology import Cosmology
from miniflims.forward import FORWARD_MODELS, second_order_source
from miniflims.grid import Grid

jax.config.update("jax_enable_x64", True)

COSMO = Cosmology(h=0.6711, omega_m=0.3175, omega_b=0.049, n_s=0.9624, sigma8=0.834)
GRID = Grid(16, 100.0)


def make(model, grid=GRID, **kw):
    return FORWARD_MODELS.get(model)(ForwardConfig(model=model, **kw), grid, grid, COSMO)


def gaussian_field(grid, seed, sigma, slope=-1.5):
    """Gaussian delta with P(k) ~ k^slope (no Nyquist modes), rescaled to std sigma."""
    w = np.random.default_rng(seed).standard_normal((grid.n,) * 3)
    k = np.asarray(grid.kmag())
    amp = np.where(k > 0, np.where(k > 0, k, 1.0) ** (slope / 2), 0.0)
    amp *= np.asarray(grid.kvec(zero_nyquist=True)[0] != 0) | (np.asarray(grid.kvec()[0]) == 0)
    d = np.fft.irfftn(np.fft.rfftn(w) * amp, s=w.shape, axes=(0, 1, 2))
    return d * sigma / d.std()


def test_source_of_two_orthogonal_waves_is_exact():
    # delta = A cos(kx) + B cos(ky): phi_xx = A cos(kx), phi_yy = B cos(ky), phi_xy = 0
    k = 2 * GRID.k_fundamental
    x = np.arange(GRID.n) * GRID.cell
    X, Y, _ = np.meshgrid(x, x, x, indexing="ij")
    A, B = 0.3, -0.2
    mu2 = second_order_source(jnp.asarray(A * np.cos(k * X) + B * np.cos(k * Y)), GRID)
    np.testing.assert_allclose(np.asarray(mu2), A * B * np.cos(k * X) * np.cos(k * Y), atol=1e-12)


def test_plane_wave_is_zeldovich():
    # one-dimensional perturbation: mu_2 = 0, so 2LPT = Zel'dovich exactly
    k = GRID.k_fundamental
    x = np.arange(GRID.n) * GRID.cell
    delta = jnp.asarray(0.4 * np.cos(k * x)[:, None, None] * np.ones((GRID.n,) * 3))
    np.testing.assert_allclose(np.asarray(make("2lpt")(delta)), np.asarray(make("zeldovich")(delta)), atol=1e-12)


def test_second_order_term_scales_as_amplitude_squared():
    d = jnp.asarray(gaussian_field(GRID, 0, 1.0))
    m2, mz = make("2lpt"), make("zeldovich")
    diff = [float(jnp.std(m2(e * d) - mz(e * d))) for e in (0.01, 0.02)]
    assert diff[1] / diff[0] == pytest.approx(4.0, rel=0.05)


def test_skewness_gain_over_zeldovich():
    # tree level: S3 = 34/7 - (smoothing) for 2LPT (exact F2), 4 - (same) for Zel'dovich
    g = Grid(48, 400.0)
    m2, mz = make("2lpt", grid=g), make("zeldovich", grid=g)
    gain = []
    for seed in range(3):
        d = jnp.asarray(gaussian_field(g, seed, 0.06))
        s3 = []
        for m in (m2, mz):
            r = np.asarray(m(d)) - 1.0
            s3.append(np.mean(r**3) / np.var(r) ** 2)
        gain.append(s3[0] - s3[1])
    assert np.mean(gain) == pytest.approx(6 / 7, rel=0.35)


def test_gradient_matches_finite_differences():
    m = make("2lpt")
    d = jnp.asarray(gaussian_field(GRID, 1, 0.5))
    w = jnp.asarray(np.random.default_rng(2).standard_normal((GRID.n,) * 3))
    f = jax.jit(lambda v: jnp.sum(w * m(v)))
    g = jax.grad(f)(d)
    v = jnp.asarray(np.random.default_rng(3).standard_normal((GRID.n,) * 3))
    eps = 1e-6
    fd = (f(d + eps * v) - f(d - eps * v)) / (2 * eps)
    assert float(jnp.sum(g * v)) == pytest.approx(float(fd), rel=1e-6)


def test_rsd_uses_second_order_growth_rate():
    m = make("2lpt", rsd=RSDConfig(enabled=True, los="z"))
    d = jnp.asarray(gaussian_field(GRID, 4, 0.3))
    psi1, psi2 = m.displacements(d)
    shift = m.positions(d) - (m.q + psi1 + psi2)
    expect = m.growth_rate * psi1[:, 2] + m.growth_rate_2 * psi2[:, 2]
    np.testing.assert_allclose(np.asarray(shift[:, 2]), np.asarray(expect), atol=1e-12)
    np.testing.assert_allclose(np.asarray(shift[:, :2]), 0.0, atol=1e-12)
