import numpy as np
import pytest

from miniflims import diagnostics as diag


def ar1(phi, n, chains, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((chains, n))
    x[:, 0] = rng.standard_normal(chains)
    for t in range(1, n):
        x[:, t] = phi * x[:, t - 1] + np.sqrt(1 - phi**2) * rng.standard_normal(chains)
    return x


def test_power_spectrum_of_white_noise_and_parseval():
    rng = np.random.default_rng(0)
    n, box = 32, 200.0
    x = rng.standard_normal((n, n, n))
    k, pk, nmodes = diag.power_spectrum(x, box, 8)
    cell_volume = (box / n) ** 3
    assert np.all(np.diff(k) > 0) and k[0] >= 2 * np.pi / box and k[-1] <= np.pi * n / box
    assert np.all(np.abs(pk / cell_volume - 1) < 5 * np.sqrt(2 / nmodes))
    # single plane wave: all power in its own bin, with the right amplitude
    grid = np.arange(n) * box / n
    wave = 0.3 * np.cos(3 * 2 * np.pi / box * grid)[:, None, None] * np.ones((n, n, n))
    k, pk, nmodes = diag.power_spectrum(wave, box, 15)
    i = np.argmax(pk)
    kf = 2 * np.pi / box  # bins are one fundamental wide here; k is the bin mean
    assert 3 * kf <= k[i] < 4 * kf
    assert (pk * nmodes).sum() / box**3 == pytest.approx(wave.var())  # Parseval
    assert pk[i] * nmodes[i] == pytest.approx((pk * nmodes).sum())


def test_cross_correlation():
    rng = np.random.default_rng(1)
    a, b = rng.standard_normal((2, 16, 16, 16))
    _, r = diag.cross_correlation(a, a, 100.0, 6)
    np.testing.assert_allclose(r, 1.0)
    _, r = diag.cross_correlation(a, -2 * a, 100.0, 6)
    np.testing.assert_allclose(r, -1.0)
    _, r = diag.cross_correlation(a, b, 100.0, 6)
    assert np.all(np.abs(r[1:]) < 0.3)
    _, r = diag.cross_correlation(a, a + b, 100.0, 6)
    np.testing.assert_allclose(r[1:], 1 / np.sqrt(2), atol=0.12)


def test_gaussianity_moments():
    rng = np.random.default_rng(2)
    m = diag.gaussianity_moments(rng.standard_normal(200_000))
    assert abs(m["mean"]) < 0.01 and m["variance"] == pytest.approx(1.0, abs=0.01)
    assert abs(m["skewness"]) < 0.03 and abs(m["kurtosis"]) < 0.05
    m = diag.gaussianity_moments(rng.exponential(size=200_000))
    assert m["skewness"] == pytest.approx(2.0, abs=0.1) and m["kurtosis"] == pytest.approx(6.0, abs=0.6)


def test_gelman_rubin():
    good = ar1(0.5, 2000, 4)
    assert diag.gelman_rubin(good) == pytest.approx(1.0, abs=0.02)
    shifted = good + np.array([0.0, 0.0, 3.0, 3.0])[:, None]
    assert diag.gelman_rubin(shifted) > 1.5
    # the split version also flags a single drifting chain
    assert diag.gelman_rubin(good[0] + np.linspace(0, 5, 2000)) > 1.5
    assert diag.gelman_rubin(good[0]) == pytest.approx(1.0, abs=0.05)
    assert np.isnan(diag.gelman_rubin(np.zeros((2, 3))))
    assert np.isnan(diag.gelman_rubin(np.ones((2, 100))))


@pytest.mark.parametrize("phi", [0.0, 0.5, 0.9])
def test_autocorrelation_time_and_ess_of_ar1(phi):
    x = ar1(phi, 20_000, 4, seed=3)
    tau = (1 + phi) / (1 - phi)
    assert diag.integrated_autocorr_time(x) == pytest.approx(tau, rel=0.15)
    assert diag.effective_sample_size(x) == pytest.approx(x.size / tau, rel=0.15)
    acf = diag.autocorrelation(x[0])
    assert acf[0] == 1.0 and acf[1] == pytest.approx(phi, abs=0.03)


def test_constant_chain_is_handled():
    assert diag.integrated_autocorr_time(np.ones(50)) == 1.0
