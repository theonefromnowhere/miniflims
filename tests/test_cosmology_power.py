import jax
import jax.numpy as jnp
import numpy as np
import pytest

from miniflims.cosmology import Cosmology, efunc, growth_factor, growth_rate
from miniflims.power import linear_power, sigma_r, tophat_window
from miniflims.transfer import TRANSFER_FUNCTIONS, eh_scales, eisenstein_hu, eisenstein_hu_nowiggle

COSMO = Cosmology(h=0.681, omega_m=0.306, omega_b=0.0486, n_s=0.967, sigma8=0.807)
EDS = Cosmology(h=0.7, omega_m=1.0, omega_b=0.05, n_s=1.0, sigma8=0.8)


# ------------------------------------------------------------------ background
def test_growth_einstein_de_sitter_is_scale_factor():
    for a in (0.1, 0.5, 1.0):
        assert float(growth_factor(a, EDS)) == pytest.approx(a, rel=1e-6)
        assert float(growth_rate(a, EDS)) == pytest.approx(1.0, rel=1e-6)


def test_growth_lcdm():
    assert float(growth_factor(1.0, COSMO)) == pytest.approx(1.0)
    # suppressed relative to EdS at late times, D ∝ a at early times
    assert float(growth_factor(0.5, COSMO)) > 0.5
    ratio = float(growth_factor(0.02, COSMO) / growth_factor(0.01, COSMO))
    assert ratio == pytest.approx(2.0, rel=1e-3)
    # f ≈ Omega_m(a)^0.55
    for a in (0.3, 0.6, 1.0):
        om_a = COSMO.omega_m * a**-3 / float(efunc(a, COSMO)) ** 2
        assert float(growth_rate(a, COSMO)) == pytest.approx(om_a**0.55, rel=0.01)


def test_growth_matches_ode_solution():
    # independent check: integrate D'' + (3/a + E'/E) D' = 1.5 Om D / (a^5 E^2)
    from scipy.integrate import solve_ivp

    om = COSMO.omega_m

    def rhs(a, y):
        e2 = om * a**-3 + 1 - om
        dlne = -1.5 * om * a**-4 / e2
        return [y[1], -(3 / a + dlne) * y[1] + 1.5 * om * y[0] / (a**5 * e2)]

    a0 = 1e-3
    sol = solve_ivp(rhs, (a0, 1.0), [a0, 1.0], rtol=1e-10, atol=1e-12, dense_output=True)
    for a in (0.25, 0.5, 0.8):
        ref = sol.sol(a)[0] / sol.sol(1.0)[0]
        assert float(growth_factor(a, COSMO)) == pytest.approx(ref, rel=1e-4)


# -------------------------------------------------------------------- transfer
def test_transfer_limits():
    for name in TRANSFER_FUNCTIONS.names():
        t = TRANSFER_FUNCTIONS.get(name)
        vals = np.asarray(t(np.array([0.0, 1e-5, 1e-2, 1.0, 10.0]), COSMO))
        assert vals[0] == 1.0, name
        assert vals[1] == pytest.approx(1.0, abs=2e-3), name
        assert np.all(np.diff(vals[1:]) < 0), name
        assert np.all(np.isfinite(vals)), name
    # small-scale slope: T ~ ln(k)/k^2
    hi = np.asarray(eisenstein_hu(np.array([50.0, 100.0]), COSMO))
    slope = np.log(hi[1] / hi[0]) / np.log(2.0)
    assert -2.0 < slope < -1.7


def test_eh_scales_are_physical():
    sc = {k: float(v) for k, v in eh_scales(COSMO).items()}
    assert 3300 < sc["z_eq"] < 3500
    assert 1000 < sc["z_d"] < 1080
    assert 145 < sc["sound_horizon"] < 157  # Mpc
    # EH98 Eq. 26 approximation to the sound horizon agrees to a few per cent
    omh2, obh2 = COSMO.omega_m * COSMO.h**2, COSMO.omega_b * COSMO.h**2
    approx = 44.5 * np.log(9.83 / omh2) / np.sqrt(1 + 10 * obh2**0.75)
    assert sc["sound_horizon"] == pytest.approx(approx, rel=0.03)


def test_wiggles_oscillate_around_smooth_fit():
    k = np.linspace(0.02, 0.4, 600)
    ratio = np.asarray(eisenstein_hu(k, COSMO) / eisenstein_hu_nowiggle(k, COSMO))
    assert np.all(np.abs(ratio - 1) < 0.08)
    osc = ratio - np.convolve(ratio, np.ones(151) / 151, mode="same")
    crossings = np.sum(np.diff(np.sign(osc[100:-100])) != 0)
    assert crossings >= 4
    # BAO period in k is 2 pi / s
    peaks = [i for i in range(101, 499) if osc[i] > osc[i - 1] and osc[i] > osc[i + 1]]
    s = float(eh_scales(COSMO)["sound_horizon"]) * COSMO.h  # Mpc/h
    spacing = np.mean(np.diff(k[peaks]))
    assert spacing == pytest.approx(2 * np.pi / s, rel=0.15)


def test_no_baryon_limit_matches_between_fits():
    c = COSMO._replace(omega_b=1e-4)
    k = np.logspace(-3, 1, 50)
    np.testing.assert_allclose(
        np.asarray(eisenstein_hu(k, c)), np.asarray(eisenstein_hu_nowiggle(k, c)), rtol=0.03
    )


def test_eisenstein_hu_against_colossus():
    cosmology = pytest.importorskip("colossus.cosmology.cosmology")
    ref = cosmology.setCosmology(
        "miniflims_test",
        dict(flat=True, H0=100 * COSMO.h, Om0=COSMO.omega_m, Ob0=COSMO.omega_b,
             sigma8=COSMO.sigma8, ns=COSMO.n_s, relspecies=False, Tcmb0=COSMO.t_cmb),
    )
    k = np.logspace(-3, 1, 60)
    for ours, theirs in (("eisenstein_hu", "eisenstein98"), ("eisenstein_hu_nowiggle", "eisenstein98_zb")):
        np.testing.assert_allclose(
            np.asarray(linear_power(k, COSMO, ours)),
            ref.matterPowerSpectrum(k, model=theirs),
            rtol=2e-3,
        )
    assert float(growth_factor(0.5, COSMO)) == pytest.approx(ref.growthFactor(1.0), rel=1e-3)


# ----------------------------------------------------------------------- power
def test_tophat_window_series_is_continuous():
    x = np.array([0.0, 9.9e-4, 1.01e-3, 0.5])
    w = np.asarray(tophat_window(jnp.asarray(x)))
    assert w[0] == 1.0
    assert w[1] == pytest.approx(w[2], rel=1e-6)
    assert w[3] == pytest.approx(3 * (np.sin(0.5) - 0.5 * np.cos(0.5)) / 0.5**3)


@pytest.mark.parametrize("transfer", ["eisenstein_hu", "eisenstein_hu_nowiggle", "bbks"])
def test_sigma8_normalisation(transfer):
    s8 = sigma_r(lambda k: linear_power(k, COSMO, transfer), 8.0)
    assert float(s8) == pytest.approx(COSMO.sigma8, rel=1e-6)
    # converged with respect to the integration grid
    s8_fine = sigma_r(lambda k: linear_power(k, COSMO, transfer), 8.0, 1e-6, 1e3, 8192)
    assert float(s8_fine) == pytest.approx(COSMO.sigma8, rel=1e-4)


def test_power_shape_and_k0():
    p = np.asarray(linear_power(np.array([0.0, 1e-4, 2e-4]), COSMO))
    assert p[0] == 0.0
    assert np.log(p[2] / p[1]) / np.log(2.0) == pytest.approx(COSMO.n_s, abs=0.01)
    k = np.logspace(-3, 0, 400)
    k_peak = k[np.argmax(np.asarray(linear_power(k, COSMO)))]
    assert 0.01 < k_peak < 0.03  # turnover at matter-radiation equality


def test_power_is_differentiable_in_cosmology():
    def f(om):
        return linear_power(jnp.asarray(0.1), COSMO._replace(omega_m=om))

    g = float(jax.grad(f)(0.306))
    fd = float((f(0.306 + 1e-5) - f(0.306 - 1e-5)) / 2e-5)
    assert np.isfinite(g) and g == pytest.approx(fd, rel=1e-5)
