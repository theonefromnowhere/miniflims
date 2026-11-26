import jax
import jax.numpy as jnp
import numpy as np
import pytest

from miniflims import diagnostics as diag
from miniflims import load_config, run
from miniflims.cola import make_pm_forces_low_memory, paint, pm_force_kernels, pm_forces, read, tcola_coefficients
from miniflims.config import ColaConfig, ForwardConfig, RSDConfig
from miniflims.cosmology import Cosmology, efunc, growth_factor
from miniflims.forward import FORWARD_MODELS, cic_paint, cic_read
from miniflims.grid import Grid
from pathlib import Path

COSMO = Cosmology(h=0.681, omega_m=0.306, omega_b=0.0486, n_s=0.967, sigma8=0.807)
EDS = COSMO._replace(omega_m=1.0)
GRID = Grid(16, 250.0)
REFERENCE = Path(__file__).parent / "data" / "borgjaxpm_tcola_reference.npz"


def make(model="tcola", grid=GRID, mesh=None, cosmo=COSMO, **kw):
    cola = ColaConfig(**kw.pop("cola", {}))
    return FORWARD_MODELS.get(model)(ForwardConfig(model=model, cola=cola, **kw), grid, mesh or grid, cosmo)


def random_delta(seed=0, amp=1.0, n=16):
    """Smooth random field with unit-ish variance (low-pass filtered noise)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, n, n))
    k = np.fft.fftfreq(n)[:, None, None] ** 2 + np.fft.fftfreq(n)[None, :, None] ** 2 + np.fft.rfftfreq(n)[None, None, :] ** 2
    d = np.fft.irfftn(np.fft.rfftn(x) * np.exp(-k / (2 * 0.12**2)), s=x.shape, axes=(0, 1, 2))
    d -= d.mean()
    return jnp.asarray(amp * d / d.std())


# ----------------------------------------------------------------------- pieces
def test_cic_read_interpolates_and_is_adjoint_of_paint():
    rng = np.random.default_rng(0)
    mesh = jnp.asarray(rng.standard_normal((8, 8, 8)))
    # exact at nodes, linear in between, periodic
    assert float(cic_read(mesh, jnp.array([[2.0, 3.0, 5.0]]), 8.0)[0]) == pytest.approx(float(mesh[2, 3, 5]))
    mid = cic_read(mesh, jnp.array([[2.5, 3.0, 5.0], [7.5, 0.0, 0.0], [-0.5, 8.0, 16.0]]), 8.0)
    assert float(mid[0]) == pytest.approx(0.5 * float(mesh[2, 3, 5] + mesh[3, 3, 5]))
    assert float(mid[1]) == pytest.approx(0.5 * float(mesh[7, 0, 0] + mesh[0, 0, 0]))
    assert float(mid[2]) == pytest.approx(float(mid[1]))
    # <paint(p), m> = (n^3 / Np) sum_p read(m, p)
    pos = jnp.asarray(rng.uniform(0, 8, (300, 3)))
    lhs = float(jnp.sum(cic_paint(pos, 8, 8.0) * mesh))
    assert lhs == pytest.approx(8**3 / 300 * float(jnp.sum(cic_read(mesh, pos, 8.0))))


def test_time_integration_coefficients():
    a0, a1, n = 0.1, 1.0, 6
    co = tcola_coefficients(COSMO, a0, a1, n)
    assert len(co.drift) == 2 * n + 1
    assert sum(k > 0 for k in co.kick) == n and sum(d > 0 for d in co.drift) == n + 1
    assert all((k > 0) != (d > 0) for k, d in zip(co.kick, co.drift))  # each op is a kick or a drift
    # the pieces tile [a0, a1]
    assert sum(co.growth_drift) == pytest.approx(float(growth_factor(a1, COSMO) - growth_factor(a0, COSMO)))
    from scipy.integrate import quad
    e = lambda a: float(efunc(a, COSMO))
    assert sum(co.drift) == pytest.approx(quad(lambda a: 1 / (a**3 * e(a)), a0, a1)[0], rel=1e-8)
    assert sum(co.kick) == pytest.approx(quad(lambda a: 1 / (a**2 * e(a)), a0, a1)[0], rel=1e-8)
    # first kick uses forces at the half-step drift
    assert co.cola_kick[1] == pytest.approx(co.kick[1] * float(growth_factor(a0 + 0.5 * (a1 - a0) / n, COSMO)))
    # Einstein-de Sitter: E = a^-3/2, so ∫ da/(a^2 E) = 2 (sqrt(a1) - sqrt(a0))
    eds = tcola_coefficients(EDS, a0, a1, 3)
    assert sum(eds.kick) == pytest.approx(2 * (np.sqrt(a1) - np.sqrt(a0)), rel=1e-8)
    assert sum(eds.growth_drift) == pytest.approx(a1 - a0, rel=1e-6)
    for bad in ((0.5, 0.5, 3), (0.5, 0.2, 3), (0.1, 1.0, 0)):
        with pytest.raises(ValueError):
            tcola_coefficients(COSMO, *bad)


def test_pm_force_of_a_plane_wave():
    # delta = A cos(w x): force = -grad lap^-1 delta with the finite-difference
    # kernels, times the CIC paint and read windows
    n, box, amp = 16, 16.0, 1e-7  # small: the formula below is first order in amp
    w = 2 * np.pi / n
    q = np.stack(np.meshgrid(*[np.arange(n) + 0.5] * 3, indexing="ij"), -1).reshape(-1, 3)
    psi = -(amp / w) * np.sin(w * q[:, 0])
    pos = jnp.asarray(q + np.stack([psi, 0 * psi, 0 * psi], -1))
    f = np.asarray(pm_forces(pos, n, box, pm_force_kernels(n)))
    # staggered lattice -> CIC density: sin(w/2)/(w/2); CIC read half-way between nodes: cos(w/2)
    window = np.sin(w / 2) / (w / 2) * np.cos(w / 2)
    fd = np.sin(w) / (4 * np.sin(w / 2) ** 2)
    s2, grads = pm_force_kernels(n)
    assert s2[0].shape == (n, 1, 1) and grads[2].shape == (1, 1, n // 2 + 1)  # separable: O(n) storage
    np.testing.assert_allclose(f[:, 0], -amp * window * fd * np.sin(w * q[:, 0]), atol=1e-5 * amp / w)
    np.testing.assert_allclose(f[:, 1:], 0.0, atol=1e-14)
    # momentum conservation
    assert np.abs(f.sum(axis=0)).max() < 1e-12


# ------------------------------------------------------------- against original
def test_matches_original_borgjaxpm_tcola():
    """Reference produced by running the unmodified BorgJaxPM tCOLA
    (prepare_cola_simulation, JAX 0.8, float64) on this input; settings are
    stored in the file. The original computes its time-integration
    coefficients in float32, which sets the agreement level."""
    ref = np.load(REFERENCE)
    fwd = make(lattice_offset=0.0, cola=dict(n_steps=4, z_start=9.0, force_sampling=2))
    pos, mom, _ = fwd.simulate(jnp.asarray(ref["delta"]))
    # the original orders particles with meshgrid's default 'xy' indexing
    reorder = lambda a: np.asarray(a).reshape(16, 16, 16, 3).transpose(1, 0, 2, 3).reshape(-1, 3)
    d = np.mod(reorder(pos) / GRID.cell, 16) - ref["pos"]
    d -= 16 * np.round(d / 16)
    assert np.abs(d).max() < 5e-4  # cells; displacements are O(1) cell
    assert np.abs(reorder(mom) / GRID.cell - ref["vel"]).max() < 1e-3 * np.abs(ref["vel"]).max()
    rho = np.asarray(fwd(jnp.asarray(ref["delta"])))
    assert np.abs(rho - ref["rho"]).max() < 2e-3
    assert np.corrcoef(rho.ravel(), ref["rho"].ravel())[0, 1] > 1 - 1e-6
    assert ref["rho"].std() > 0.5  # a non-trivial, non-linear field


# ---------------------------------------------------------------------- physics
def plane_wave(amp, grid=GRID):
    x = np.arange(grid.n) * grid.cell
    return jnp.asarray(amp * np.cos(grid.k_fundamental * x)[:, None, None] * np.ones((grid.n,) * 3))


def displacement_error(delta, offset, grid=GRID, **cola):
    """Largest deviation of tCOLA from Zel'dovich displacements, relative."""
    co = make(grid=grid, lattice_offset=offset, cola=cola)
    za = make("zeldovich", grid=grid, lattice_offset=co.lattice_offset)  # same particles
    d_za = np.asarray(za.positions(delta) - za.q)
    d_co = np.asarray(co.positions(delta) - co.q)
    return np.abs(d_co - d_za).max() / np.abs(d_za).max()


def test_default_lattice_offset_is_half_a_force_cell():
    assert make(cola=dict(force_sampling=1)).lattice_offset == 0.5
    assert make(cola=dict(force_sampling=2)).lattice_offset == 0.25
    assert make(cola=dict(force_sampling=4)).lattice_offset == 0.125
    assert make(lattice_offset=0.0).lattice_offset == 0.0
    fwd = make(cola=dict(force_sampling=2))
    on_force_mesh = np.asarray(fwd.q) / GRID.box * fwd.n_force
    np.testing.assert_allclose(on_force_mesh - np.floor(on_force_mesh), 0.5)


@pytest.mark.parametrize("amp", [1e-4, 1e-2])
def test_linear_regime_reduces_to_zeldovich(amp):
    # A plane wave follows the Zel'dovich solution exactly before shell
    # crossing, so the COLA residual must vanish. What is left is the
    # resolution of the force mesh.
    delta = plane_wave(amp)
    assert displacement_error(delta, None, force_sampling=1) < 0.06
    assert displacement_error(delta, None, force_sampling=2) < 1e-3
    assert displacement_error(delta, None, force_sampling=4) < 0.012


def test_particles_on_force_mesh_nodes_break_the_linear_regime():
    # Documents why the default offset is not 0 (the BORG / JaxPM layout) and
    # not 0.5 when force_sampling is even: with particles starting on force
    # mesh nodes the CIC kink makes the force non-linear at any amplitude.
    delta = plane_wave(1e-4)
    for offset, fs in ((0.0, 1), (0.0, 2), (0.5, 2)):
        assert displacement_error(delta, offset, force_sampling=fs) > 0.1, (offset, fs)
    assert displacement_error(delta, 0.25, force_sampling=2) < 1e-3


def test_plane_wave_collapse_matches_exact_zeldovich_solution():
    grid = Grid(32, 250.0)
    delta = plane_wave(0.6, grid)
    assert displacement_error(delta, None, grid=grid, n_steps=10, force_sampling=2) < 0.02
    assert np.asarray(make(grid=grid)(delta)).max() > 2.0  # genuinely non-linear


def test_converges_with_number_of_steps():
    delta = random_delta(1, amp=1.0)
    ref = np.asarray(make(cola=dict(n_steps=48))(delta))
    err = [np.sqrt(np.mean((np.asarray(make(cola=dict(n_steps=n))(delta)) - ref) ** 2)) for n in (3, 6, 12, 24)]
    assert all(b < 0.5 * a for a, b in zip(err, err[1:]))  # at least first order; ~second in practice
    assert err[-1] < 0.01 * ref.std()


def lcdm_delta(n, box, seed=2):
    """z = 0 linear density of the default cosmology (Eisenstein-Hu)."""
    from miniflims.data import build_response
    from miniflims.model import FieldModel

    cfg = load_config(overrides={"grid": {"n_grid": n, "box_size": box}})
    model = FieldModel(cfg, build_response(cfg))
    return model.grid, model.linear_density(jnp.asarray(np.random.default_rng(seed).standard_normal((n,) * 3)))


def test_power_relative_to_zeldovich_on_an_lcdm_field():
    # 7.8 Mpc/h cells: non-linear on the smallest scales of the box
    grid, delta = lcdm_delta(32, 250.0)
    k, p_za, _ = diag.power_spectrum(np.asarray(make("zeldovich", grid=grid)(delta)) - 1, grid.box, 6)
    p = {
        fs: diag.power_spectrum(np.asarray(make(grid=grid, cola=dict(force_sampling=fs))(delta)) - 1, grid.box, 6)[1]
        for fs in (1, 2, 4)
    }
    # large scales: both are linear theory
    for fs in (2, 4):
        assert p[fs][0] == pytest.approx(p_za[0], rel=0.06), fs
    # small scales: Zel'dovich particles stream through each other, N-body
    # particles stay bound, so a resolved force gives more power, rising with k
    for fs in (2, 4):
        assert np.all(p[fs][1:] > p_za[1:]) and p[fs][-1] / p_za[-1] > 1.2, fs
    assert np.all(p[4][1:] >= p[2][1:])
    # a force mesh as coarse as the particle grid loses power on every scale
    assert np.all(p[1] < p_za) and np.all(p[1] < p[2])


def test_mass_conservation_meshes_and_validation():
    delta = random_delta(3)
    for mesh in (GRID, Grid(8, GRID.box)):
        rho = np.asarray(make(mesh=mesh)(delta))
        assert rho.shape == (mesh.n,) * 3 and rho.mean() == pytest.approx(1.0) and rho.min() >= 0
    np.testing.assert_allclose(np.asarray(make()(jnp.zeros((16,) * 3))), 1.0, atol=1e-12)
    with pytest.raises(ValueError):
        make(mesh=Grid(12, GRID.box))
    with pytest.raises(ValueError):
        make(cola=dict(force_sampling=0))
    with pytest.raises(ValueError):
        make(redshift=3.0, cola=dict(z_start=2.0))


def test_final_redshift_and_growth():
    delta = random_delta(4, amp=1e-3)
    lo, hi = np.asarray(make(redshift=1.0)(delta)) - 1, np.asarray(make()(delta)) - 1
    d1 = float(growth_factor(0.5, COSMO))
    assert np.sqrt(np.mean(lo**2)) / np.sqrt(np.mean(hi**2)) == pytest.approx(d1, rel=0.01)


@pytest.mark.parametrize("amp", [1e-4, 1e-2])
def test_momentum_and_rsd(amp):
    # plane wave along z: linear theory is exact for its momentum,
    # p = a^2 f E D psi (a = 1 here), and RSD multiplies the displacement by 1 + f
    x = np.arange(16) * GRID.cell
    delta = jnp.asarray(amp * np.cos(GRID.k_fundamental * x)[None, None, :] * np.ones((16,) * 3))
    fwd = make()
    pos, mom, psi = fwd.simulate(delta)
    p_lin = fwd.growth_rate * fwd.e_final * fwd.growth * np.asarray(psi)
    assert np.abs(np.asarray(mom) - p_lin).max() < 1e-8 * np.abs(p_lin).max()
    red = make(rsd=RSDConfig(enabled=True, los="z"))
    shift = np.asarray(red.positions(delta) - pos)
    np.testing.assert_allclose(shift[:, :2], 0.0, atol=1e-14)
    disp = np.asarray(pos - fwd.q)[:, 2]
    assert np.abs(shift[:, 2] - fwd.growth_rate * disp).max() < 1e-8 * fwd.growth_rate * np.abs(disp).max()
    # a line of sight perpendicular to the wave changes nothing
    perp = make(rsd=RSDConfig(enabled=True, los="x"))
    np.testing.assert_allclose(np.asarray(perp.positions(delta)), np.asarray(pos), atol=1e-14)
    radial = make(rsd=RSDConfig(enabled=True, los="radial"))
    assert np.isfinite(np.asarray(radial(random_delta(5)))).all()


@pytest.mark.parametrize("offset", [None, 0.0])
@pytest.mark.parametrize("low_memory", [True, False])
def test_gradient_matches_finite_differences(offset, low_memory):
    rng = np.random.default_rng(6)
    grid = Grid(8, 125.0)
    fwd = make(grid=grid, lattice_offset=offset, cola=dict(n_steps=3, low_memory=low_memory),
               rsd=RSDConfig(enabled=True, los="radial"))
    delta = random_delta(7, amp=0.8, n=8)
    w = jnp.asarray(rng.standard_normal((8, 8, 8)))
    f = lambda d: jnp.sum(w * fwd(d) ** 2)
    v = jnp.asarray(rng.standard_normal((8, 8, 8)))
    eps = 1e-6
    fd = (f(delta + eps * v) - f(delta - eps * v)) / (2 * eps)
    assert float(jnp.sum(jax.grad(f)(delta) * v)) == pytest.approx(float(fd), rel=1e-5)


# ----------------------------------------------------------------- low memory
def test_analytic_cic_adjoints_match_autodiff():
    rng = np.random.default_rng(0)
    p = jnp.asarray(rng.uniform(-3, 11, (200, 3)))  # cells, also outside the box
    mesh = jnp.asarray(rng.standard_normal((8, 8, 8)))
    g = jnp.asarray(rng.standard_normal(200))
    np.testing.assert_allclose(paint(p, 8) * 8**3 / 200, cic_paint(p, 8, 8.0), atol=1e-13)
    np.testing.assert_allclose(read(mesh, p), cic_read(mesh, p, 8.0), atol=1e-13)
    a = jax.grad(lambda q: jnp.sum(mesh * paint(q, 8)))(p)
    b = jax.grad(lambda q: jnp.sum(mesh * cic_paint(q, 8, 8.0)))(p) * 200 / 8**3
    np.testing.assert_allclose(a, b, atol=1e-13)
    a = jax.grad(lambda m, q: jnp.sum(g * read(m, q)), (0, 1))(mesh, p)
    b = jax.grad(lambda m, q: jnp.sum(g * cic_read(m, q, 8.0)), (0, 1))(mesh, p)
    np.testing.assert_allclose(a[0], b[0], atol=1e-13)
    np.testing.assert_allclose(a[1], b[1], atol=1e-13)


def test_low_memory_forces_and_gradients_equal_plain_autodiff():
    rng = np.random.default_rng(1)
    n, box = 8, 100.0
    kernels = pm_force_kernels(2 * n)
    pos = jnp.asarray(rng.uniform(0, box, (n**3, 3)))
    low = make_pm_forces_low_memory(2 * n, box, kernels)
    plain = lambda x: pm_forces(x, 2 * n, box, kernels)
    np.testing.assert_allclose(low(pos), plain(pos), atol=1e-12)
    w = jnp.asarray(rng.standard_normal((n**3, 3)))
    g_low = jax.grad(lambda x: jnp.sum(w * low(x)))(pos)
    g_plain = jax.grad(lambda x: jnp.sum(w * plain(x)))(pos)
    np.testing.assert_allclose(g_low, g_plain, atol=1e-12 * float(jnp.abs(g_plain).max()) + 1e-14)

    # and through the whole simulation
    delta = random_delta(8, amp=1.0, n=8)
    grid = Grid(8, 125.0)
    out = {}
    for mode in (True, False):
        fwd = make(grid=grid, cola=dict(n_steps=4, low_memory=mode))
        out[mode] = (np.asarray(fwd(delta)), np.asarray(jax.grad(lambda d: jnp.sum(fwd(d) ** 3))(delta)))
    np.testing.assert_allclose(out[True][0], out[False][0], atol=1e-12)
    np.testing.assert_allclose(out[True][1], out[False][1], rtol=1e-9, atol=1e-12)


def test_jit_and_vmap():
    fwd = make(grid=Grid(8, 125.0), cola=dict(n_steps=2))
    deltas = jnp.stack([random_delta(s, n=8) for s in (0, 1)])
    out = jax.jit(jax.vmap(fwd))(deltas)
    np.testing.assert_allclose(out[1], fwd(deltas[1]), atol=1e-12)


# --------------------------------------------------------------------- pipeline
@pytest.mark.slow
def test_tcola_in_the_pipeline():
    cfg = load_config(Path(__file__).parents[1] / "configs" / "tcola.yaml", {
        "grid": {"n_grid": 16, "box_size": 400.0},
        "forward": {"cola": {"n_steps": 4}},
        "sampler": {"n_warmup": 150, "n_samples": 150, "thin": 5},
        "output": {"dir": None, "log_every": 0, "n_k_bins": 6},
    })
    assert cfg.forward.model == "tcola"
    res = run(cfg)
    s = res.summary
    assert type(res.model.forward).__name__ == "TColaForward"
    assert np.all(np.isfinite(res.gibbs.trace["log_likelihood"]))
    assert 0.4 < s["hmc"]["accept_rate"] < 0.98
    assert s["density_cross_correlation"]["r"][0] > 0.8
    assert s["nuisance"]["cat0/nbar"]["mean"] == pytest.approx(s["nuisance"]["cat0/nbar"]["truth"], rel=0.1)
    assert s["white_noise_moments"]["variance"] == pytest.approx(1.0, abs=0.12)
