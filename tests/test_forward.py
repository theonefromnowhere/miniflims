import jax
import jax.numpy as jnp
import numpy as np
import pytest

from miniflims.config import ForwardConfig, RSDConfig
from miniflims.cosmology import Cosmology, growth_factor, growth_rate
from miniflims.forward import FORWARD_MODELS, cic_paint, zeldovich_displacement
from miniflims.grid import Grid

COSMO = Cosmology(h=0.681, omega_m=0.306, omega_b=0.0486, n_s=0.967, sigma8=0.807)
GRID = Grid(16, 100.0)


def make(model="zeldovich", grid=GRID, mesh=None, **kw):
    return FORWARD_MODELS.get(model)(ForwardConfig(model=model, **kw), grid, mesh or grid, COSMO)


def plane_wave(amp, axis=0, grid=GRID, harmonic=1):
    k = harmonic * grid.k_fundamental
    x = np.arange(grid.n) * grid.cell
    shape = [1, 1, 1]
    shape[axis] = grid.n
    return amp * np.cos(k * x).reshape(shape) * np.ones((grid.n,) * 3), k


# ------------------------------------------------------------------------- grid
def test_grid_conventions():
    g = Grid(8, 40.0)
    assert g.cell == 5.0 and g.volume == 64000.0
    assert g.kmag().shape == (8, 8, 5)
    assert g.kmag().max() == pytest.approx(np.sqrt(3) * g.k_nyquist)
    # rfft multiplicities add up to the number of modes of the full transform
    assert g.rfft_weights().sum() == 8**3
    assert np.all(g.kvec(zero_nyquist=True)[0][4] == 0)


# -------------------------------------------------------------------------- CIC
def test_cic_particle_on_node_and_between_nodes():
    rho = np.asarray(cic_paint(jnp.array([[2.0, 3.0, 5.0]]), 8, 8.0))
    assert rho[2, 3, 5] == pytest.approx(8**3)
    assert rho.sum() == pytest.approx(8**3)
    rho = np.asarray(cic_paint(jnp.array([[2.5, 3.0, 5.0]]), 8, 8.0))
    assert rho[2, 3, 5] == pytest.approx(0.5 * 8**3) and rho[3, 3, 5] == pytest.approx(0.5 * 8**3)
    rho = np.asarray(cic_paint(jnp.array([[2.25, 3.5, 5.0]]), 8, 8.0))
    assert rho[2, 3, 5] == pytest.approx(0.75 * 0.5 * 8**3)
    assert rho[3, 4, 5] == pytest.approx(0.25 * 0.5 * 8**3)


def test_cic_is_periodic_and_conserves_mass():
    rho = np.asarray(cic_paint(jnp.array([[7.5, 0.0, 0.0]]), 8, 8.0))
    assert rho[7, 0, 0] == pytest.approx(0.5 * 8**3) and rho[0, 0, 0] == pytest.approx(0.5 * 8**3)
    pos = jnp.asarray(np.random.default_rng(0).uniform(-20, 30, (500, 3)))  # outside the box too
    rho = np.asarray(cic_paint(pos, 8, 8.0))
    assert rho.mean() == pytest.approx(1.0)
    assert rho.min() >= 0


def test_cic_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    pos = jnp.asarray(rng.uniform(0, 8, (40, 3)))
    w = jnp.asarray(rng.standard_normal((8, 8, 8)))
    f = lambda p: jnp.sum(w * cic_paint(p, 8, 8.0))
    v = jnp.asarray(rng.standard_normal((40, 3)))
    eps = 1e-6
    fd = (f(pos + eps * v) - f(pos - eps * v)) / (2 * eps)
    assert float(jnp.sum(jax.grad(f)(pos) * v)) == pytest.approx(float(fd), rel=1e-6)


# -------------------------------------------------------------------- Zel'dovich
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_displacement_of_plane_wave(axis):
    delta, k = plane_wave(0.05, axis)
    psi = np.asarray(zeldovich_displacement(jnp.asarray(delta), GRID))
    x = np.arange(GRID.n) * GRID.cell
    shape = [1, 1, 1]
    shape[axis] = GRID.n
    expected = -(0.05 / k) * np.sin(k * x).reshape(shape) * np.ones((GRID.n,) * 3)
    np.testing.assert_allclose(psi[..., axis], expected, atol=1e-12)
    for other in set(range(3)) - {axis}:
        np.testing.assert_allclose(psi[..., other], 0.0, atol=1e-12)


def test_displacement_divergence_recovers_density():
    rng = np.random.default_rng(2)
    delta = rng.standard_normal((16, 16, 16))
    delta -= delta.mean()
    dk = np.fft.rfftn(delta)
    # remove Nyquist planes, where the derivative kernel is zeroed by design
    dk[8], dk[:, 8], dk[:, :, 8] = 0, 0, 0
    ax = (0, 1, 2)
    delta = np.fft.irfftn(dk, s=delta.shape, axes=ax)
    psi = np.asarray(zeldovich_displacement(jnp.asarray(delta), GRID))
    kv = GRID.kvec()
    div = sum(
        np.fft.irfftn(1j * kv[i] * np.fft.rfftn(psi[..., i]), s=delta.shape, axes=ax) for i in range(3)
    )
    np.testing.assert_allclose(-div, delta, atol=1e-10)


def test_zeldovich_unperturbed_lattice_is_uniform():
    for mesh in (GRID, Grid(8, GRID.box)):
        rho = np.asarray(make(mesh=mesh)(jnp.zeros((16, 16, 16))))
        assert rho.shape == (mesh.n,) * 3
        np.testing.assert_allclose(rho, 1.0, atol=1e-12)


def test_mesh_must_divide_grid():
    with pytest.raises(ValueError):
        make(mesh=Grid(12, GRID.box))
    with pytest.raises(ValueError):
        make("linear", mesh=Grid(8, GRID.box))


def test_zeldovich_linear_limit_and_growth():
    delta, k = plane_wave(1e-4)
    for z in (0.0, 1.0):
        fwd = make(redshift=z)
        d = float(growth_factor(1 / (1 + z), COSMO))
        assert fwd.growth == pytest.approx(d)
        rho = np.asarray(fwd(jnp.asarray(delta)))
        # CIC of the half-cell-offset lattice = compact centred difference of psi
        window = np.sin(k * GRID.cell / 2) / (k * GRID.cell / 2)
        np.testing.assert_allclose(rho - 1, d * window * delta, atol=1e-7)
        assert rho.mean() == pytest.approx(1.0)


def test_zeldovich_response_is_linear_at_small_amplitude():
    # no kink at zero displacement: rho(eps * delta) - 1 is odd and linear in
    # eps up to O(eps^2). (Particles starting on mesh nodes would violate this
    # at order one.)
    rng = np.random.default_rng(5)
    delta = jnp.asarray(rng.standard_normal((16, 16, 16)))
    fwd = make()
    plus, minus = np.asarray(fwd(1e-4 * delta)) - 1, np.asarray(fwd(-1e-4 * delta)) - 1
    scale = np.abs(plus).max()
    assert np.abs(plus + minus).max() < 1e-3 * scale
    assert np.abs(np.asarray(fwd(2e-4 * delta)) - 1 - 2 * plus).max() < 1e-3 * scale


def test_displacement_at_offset_lattice_is_fourier_interpolated():
    delta, k = plane_wave(0.05, harmonic=3)
    psi = np.asarray(zeldovich_displacement(jnp.asarray(delta), GRID, offset=0.5))
    x = (np.arange(GRID.n) + 0.5) * GRID.cell
    np.testing.assert_allclose(psi[:, 0, 0, 0], -(0.05 / k) * np.sin(k * x), atol=1e-12)
    fwd = make()
    np.testing.assert_allclose(np.asarray(fwd.q[0]), 0.5 * GRID.cell)
    disp = np.asarray(fwd.displacement(jnp.asarray(delta))).reshape(16, 16, 16, 3)
    np.testing.assert_allclose(disp, fwd.growth * psi, atol=1e-12)


def test_zeldovich_plane_wave_caustic_density():
    # 1D Zel'dovich is exact before shell crossing: rho(x(q)) = 1 / (1 - A cos(kq))
    amp = 0.5
    grid = Grid(64, 100.0)
    delta, k = plane_wave(amp, grid=grid)
    rho = np.asarray(make(grid=grid, mesh=Grid(16, 100.0))(jnp.asarray(delta)))[:, 0, 0]
    assert rho.mean() == pytest.approx(1.0)
    assert rho.argmax() == 0 and rho.argmin() == 8
    # overdense peak at q=0 is compressed, void at q=L/2 is emptied
    assert rho.max() > 1 + amp and rho.min() == pytest.approx(1 / (1 + amp), rel=0.05)


def test_plane_parallel_rsd_boosts_displacement_by_growth_rate():
    delta, _ = plane_wave(0.02, axis=2)
    z = 0.5
    real = make(redshift=z)
    red = make(redshift=z, rsd=RSDConfig(enabled=True, los="z"))
    perp = make(redshift=z, rsd=RSDConfig(enabled=True, los="x"))
    f = float(growth_rate(1 / (1 + z), COSMO))
    d = jnp.asarray(delta)
    shift_real = np.asarray(real.positions(d) - real.q)
    shift_red = np.asarray(red.positions(d) - red.q)
    np.testing.assert_allclose(shift_red, (1 + f) * shift_real, atol=1e-12)
    # wave vector perpendicular to the line of sight: no distortion
    np.testing.assert_allclose(np.asarray(perp.positions(d)), np.asarray(real.positions(d)), atol=1e-12)
    # Kaiser limit for a mode along the line of sight: delta_s = (1 + f) delta
    small = jnp.asarray(delta * 1e-3)
    sel = np.abs(delta) > 0.01
    ratio = (np.asarray(red(small)) - 1)[sel] / (np.asarray(real(small)) - 1)[sel]
    np.testing.assert_allclose(ratio, 1 + f, rtol=1e-3)


def test_radial_rsd_shifts_along_line_of_sight_only():
    rng = np.random.default_rng(3)
    delta = jnp.asarray(0.05 * rng.standard_normal((16, 16, 16)))
    obs = (50.0, 50.0, 50.0)
    real = make()
    red = make(rsd=RSDConfig(enabled=True, los="radial", observer=obs))
    x_real, x_red = np.asarray(real.positions(delta)), np.asarray(red.positions(delta))
    shift = x_red - x_real
    r = x_real - np.array(obs)
    cross = np.cross(shift, r)
    np.testing.assert_allclose(cross, 0.0, atol=1e-9)
    disp = np.asarray(real.displacement(delta))
    rhat = r / np.linalg.norm(r, axis=1, keepdims=True)
    np.testing.assert_allclose(
        np.sum(shift * rhat, axis=1), red.growth_rate * np.sum(disp * rhat, axis=1), atol=1e-10
    )
    with pytest.raises(ValueError):
        make(rsd=RSDConfig(enabled=True, los="diagonal")).positions(delta)


def test_linear_forward():
    delta, _ = plane_wave(0.1)
    fwd = make("linear", redshift=1.0)
    np.testing.assert_allclose(np.asarray(fwd(jnp.asarray(delta))), 1 + fwd.growth * delta)
    with pytest.raises(ValueError):
        make("linear", rsd=RSDConfig(enabled=True))


@pytest.mark.parametrize("rsd", [RSDConfig(), RSDConfig(enabled=True, los="radial")])
def test_forward_gradient_matches_finite_differences(rsd):
    rng = np.random.default_rng(4)
    grid = Grid(8, 100.0)
    fwd = make(grid=grid, rsd=rsd)
    delta = jnp.asarray(0.2 * rng.standard_normal((8, 8, 8)))
    w = jnp.asarray(rng.standard_normal((8, 8, 8)))
    f = lambda d: jnp.sum(w * fwd(d) ** 2)
    v = jnp.asarray(rng.standard_normal((8, 8, 8)))
    eps = 1e-6
    fd = (f(delta + eps * v) - f(delta - eps * v)) / (2 * eps)
    assert float(jnp.sum(jax.grad(f)(delta) * v)) == pytest.approx(float(fd), rel=1e-5)
