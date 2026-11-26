import jax
import jax.numpy as jnp
import numpy as np
import pytest

from miniflims import diagnostics as diag
from miniflims.config import update
from miniflims.data import build_response, load_dataset, make_mock
from miniflims.model import FieldModel
from miniflims.power import linear_power

MANTICORE_LIKE = {
    "forward": {"rsd": {"enabled": True, "los": "radial"}},
    "galaxies": {
        "n_catalogues": 2,
        "bias": {"model": "stdp"},
        "likelihood": {"model": "generalized_poisson", "params": {"b_beta": {"sample": False}}},
    },
    "data": {"mock": {
        "truth": {"nbar": [4.0, 2.0], "alpha": [1.2, 1.6], "b_alpha": 0.3},
        "selections": [{"type": "radial", "r0": 90.0}, {"type": "radial", "r0": 70.0, "r_max": 95.0}],
    }},
}


def build(cfg):
    data = make_mock(cfg)
    return FieldModel(cfg, data.response, data.counts), data


def directional_check(f, x, seed=0, eps=1e-5, rel=1e-5):
    v = jnp.asarray(np.random.default_rng(seed).standard_normal(x.shape))
    fd = (f(x + eps * v) - f(x - eps * v)) / (2 * eps)
    assert float(jnp.sum(jax.grad(f)(x) * v)) == pytest.approx(float(fd), rel=rel)


# ------------------------------------------------------------ initial conditions
def test_linear_field_has_the_configured_power_spectrum(small_cfg):
    cfg = update(small_cfg, {"grid": {"n_grid": 32, "box_size": 600.0}})
    model = FieldModel(cfg, build_response(cfg))
    rng = np.random.default_rng(0)
    fields = [np.asarray(model.linear_density(jnp.asarray(rng.standard_normal((32,) * 3)))) for _ in range(6)]
    # exact expectation of the variance: sum of P(k) over all modes / V
    kfull = np.fft.fftfreq(32, d=1 / 32) * model.grid.k_fundamental
    kmag = np.sqrt(kfull[:, None, None] ** 2 + kfull[None, :, None] ** 2 + kfull[None, None, :] ** 2)
    expected = float(np.sum(np.asarray(linear_power(kmag, model.cosmo)))) / model.grid.volume
    assert np.mean([f.var() for f in fields]) == pytest.approx(expected, rel=0.02)
    assert abs(np.mean(fields)) < 1e-12  # no DC mode
    # scale by scale
    pks = np.mean([diag.power_spectrum(f, 600.0, 10)[1] for f in fields], axis=0)
    k, _, nmodes = diag.power_spectrum(fields[0], 600.0, 10)
    ratio = pks / np.asarray(linear_power(k, model.cosmo))
    assert np.all(np.abs(ratio[nmodes > 200] - 1) < 0.08)


def test_linear_field_is_linear_in_white_noise(small_cfg):
    model, _ = build(small_cfg)
    rng = np.random.default_rng(1)
    a, b = (jnp.asarray(rng.standard_normal((8,) * 3)) for _ in range(2))
    np.testing.assert_allclose(
        model.linear_density(2 * a - 3 * b), 2 * model.linear_density(a) - 3 * model.linear_density(b), atol=1e-12
    )


# ---------------------------------------------------------------------- mock data
def test_mock_is_reproducible_and_consistent(small_cfg):
    d1, d2 = make_mock(small_cfg), make_mock(small_cfg)
    np.testing.assert_array_equal(d1.counts, d2.counts)
    d3 = make_mock(update(small_cfg, ["data.mock.seed=2"]))
    assert not np.array_equal(d1.counts, d3.counts)
    assert d1.counts.shape == (1, 8, 8, 8) and d1.truth["nuisance"].tolist() == [[3.0, 1.4]]
    # ergodic normalisation: the mean count is nbar
    assert d1.counts.mean() == pytest.approx(3.0, rel=0.1)
    assert d1.truth["rho"].mean() == pytest.approx(1.0)


def test_selections_and_masked_voxels(small_cfg):
    cfg = update(small_cfg, MANTICORE_LIKE)
    data = make_mock(cfg)
    assert data.response.shape == (2, 8, 8, 8)
    r0, r1 = data.response
    assert r0[4, 4, 4] == 1.0 and r0[0, 0, 0] < r0[4, 4, 3] < 1.0  # observer at box centre
    assert (r1 == 0).any() and np.all(data.counts[1][r1 == 0] == 0)
    assert data.truth["nuisance"][:, 0].tolist() == [4.0, 2.0]
    with pytest.raises(ValueError):
        build_response(update(cfg, {"galaxies": {"n_catalogues": 3}}))


def test_dataset_from_file(small_cfg, tmp_path):
    mock = make_mock(small_cfg)
    path = tmp_path / "data.npz"
    np.savez(path, counts=mock.counts[0], response=mock.response[0])  # 3D arrays are promoted
    data = load_dataset(update(small_cfg, {"data": {"source": "file", "path": str(path)}}))
    np.testing.assert_array_equal(data.counts, mock.counts)
    assert data.truth is None
    with pytest.raises(ValueError):
        load_dataset(update(small_cfg, {"data": {"source": "file"}}))
    with pytest.raises(ValueError):
        load_dataset(update(small_cfg, {"data": {"source": "catalogue"}}))


def test_model_validates_inputs(small_cfg):
    resp = build_response(small_cfg)
    with pytest.raises(ValueError):
        FieldModel(small_cfg, resp[:, :4])
    with pytest.raises(ValueError):
        FieldModel(small_cfg, -resp)
    with pytest.raises(ValueError):
        FieldModel(small_cfg, resp, np.zeros((1, 4, 4, 4)))
    with pytest.raises(ValueError):
        FieldModel(update(small_cfg, ["grid.n_grid=7"]), np.ones((1, 7, 7, 7)))
    with pytest.raises(ValueError):
        FieldModel(update(small_cfg, {"galaxies": {"bias": {"params": {"beta": {"init": 1.0}}}}}), resp)
    with pytest.raises(KeyError):
        FieldModel(update(small_cfg, ["galaxies.bias.model=quadratic"]), resp)
    with pytest.raises(ValueError):
        FieldModel(small_cfg, resp).nuisance_from_dict({"gamma": 1.0})


# --------------------------------------------------------------------- parameters
def test_parameter_layout_and_initialisation(small_cfg):
    cfg = update(small_cfg, MANTICORE_LIKE)
    model, _ = build(cfg)
    assert model.param_names == ["nbar", "gamma_t", "gamma_s", "alpha", "beta", "beta_0", "b_alpha", "b_beta"]
    assert model.sampled == (True,) * 7 + (False,)
    theta0 = np.asarray(model.init_nuisance(jax.random.PRNGKey(0)))
    np.testing.assert_array_equal(theta0, model.default_nuisance())
    theta = np.asarray(model.init_nuisance(jax.random.PRNGKey(0), jitter=0.3))
    assert theta.shape == (2, 8)
    assert np.all(theta[:, :7] != theta0[:, :7]) and np.all(theta[:, 7] == theta0[:, 7])  # fixed stays
    assert np.all(theta > np.asarray(model.lower)) and np.all(theta < np.asarray(model.upper))


# ---------------------------------------------------------------------- posterior
@pytest.mark.parametrize("overrides", [{}, MANTICORE_LIKE, {"galaxies": {"bias": {"model": "truncated_power_law"}}}])
def test_posterior_gradients_match_finite_differences(small_cfg, overrides):
    model, data = build(update(small_cfg, overrides))
    x = jnp.asarray(data.truth["white_noise"])
    theta = jnp.asarray(data.truth["nuisance"])
    hyper = {"likelihood_temperature": 0.7, "lambda_ps": 50.0}
    assert np.isfinite(float(model.log_posterior(x, theta, hyper)))
    directional_check(lambda v: model.log_posterior(v, theta, hyper), x)
    directional_check(lambda t: model.log_posterior(x, t, hyper), theta, rel=1e-4)


def test_posterior_decomposition_and_tempering(small_cfg):
    cfg = update(small_cfg, {"galaxies": {"bias": {"params": {"alpha": {"prior": "normal", "prior_mean": 1.0, "prior_std": 0.5}}}}})
    model, data = build(cfg)
    x, theta = jnp.asarray(data.truth["white_noise"]), jnp.asarray(data.truth["nuisance"])
    ll = float(model.log_likelihood(x, theta))
    prior_x = -0.5 * float(jnp.sum(x**2))
    prior_t = -0.5 * ((1.4 - 1.0) / 0.5) ** 2
    assert float(model.log_prior_field(x)) == pytest.approx(prior_x)
    assert float(model.log_posterior(x, theta)) == pytest.approx(prior_x + ll + prior_t)
    lp = float(model.log_posterior(x, theta, {"likelihood_temperature": 0.25}))
    assert lp == pytest.approx(prior_x + 0.25 * ll + prior_t)
    mon = model.monitor({"field": x, "nuisance": theta})
    assert float(mon["log_likelihood"]) == pytest.approx(ll)


def test_truth_is_more_likely_than_perturbed_parameters(small_cfg):
    cfg = update(small_cfg, {"grid": {"n_grid": 16, "box_size": 400.0}})
    model, data = build(cfg)
    x, theta = jnp.asarray(data.truth["white_noise"]), data.truth["nuisance"]
    best = float(model.log_likelihood(x, jnp.asarray(theta)))
    for j in range(theta.shape[1]):
        for factor in (0.8, 1.25):
            t = theta.copy()
            t[0, j] *= factor
            assert float(model.log_likelihood(x, jnp.asarray(t))) < best
    other = jnp.asarray(np.random.default_rng(9).standard_normal(x.shape))
    assert float(model.log_likelihood(other, jnp.asarray(theta))) < best


def test_nuisance_conditional_is_the_joint_up_to_a_constant(small_cfg):
    model, data = build(update(small_cfg, MANTICORE_LIKE))
    x = jnp.asarray(data.truth["white_noise"])
    hyper = {"likelihood_temperature": 0.6, "lambda_ps": 10.0}
    fn, aux = model.nuisance_conditional(x, hyper)
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(4):
        theta = jnp.asarray(data.truth["nuisance"] * (1 + 0.05 * rng.standard_normal((2, 8))))
        cond = float(jnp.sum(jax.vmap(fn)(theta, aux)))
        diffs.append(cond - float(model.log_posterior(x, theta, hyper)))
    np.testing.assert_allclose(diffs, diffs[0], rtol=0, atol=1e-8)
    # catalogues are conditionally independent: row 0 does not see row 1's data
    t = jnp.asarray(data.truth["nuisance"])
    row0 = float(fn(t[0], (aux[0][0], aux[1][0])))
    aux_swapped = (aux[0].at[1].set(0.0), aux[1])
    fn2, _ = model.nuisance_conditional(x, hyper)
    assert float(fn2(t[0], (aux_swapped[0][0], aux_swapped[1][0]))) == row0


def test_ergodic_normalisation(small_cfg):
    model, data = build(small_cfg)
    rho = jnp.asarray(data.truth["rho"])
    lam = model.expected_counts(jnp.asarray([3.0, 2.2]), rho, model.response[0])
    assert float(lam.mean()) == pytest.approx(3.0)  # <lambda> = nbar whatever the bias
    raw, _ = build(update(small_cfg, ["galaxies.bias.normalize=false"]))
    lam = raw.expected_counts(jnp.asarray([3.0, 2.2]), rho, raw.response[0])
    np.testing.assert_allclose(lam, 3.0 * (rho + raw.floor) ** 2.2)


# -------------------------------------------------------- power-spectrum prior
def test_power_spectrum_prior(small_cfg):
    cfg = update(small_cfg, {"grid": {"n_grid": 16}, "prior": {"ps_prior_bins": 8}})
    model = FieldModel(cfg, build_response(cfg))
    ps = model._ps
    assert float(jnp.sum(ps["n_i"] * ps["good"])) == 16**3 - 1  # every mode but DC, once
    rng = np.random.default_rng(0)
    ratios = np.array([np.asarray(model.white_noise_power_ratio(jnp.asarray(rng.standard_normal((16,) * 3)))) for _ in range(300)])
    good = np.asarray(ps["good"])
    np.testing.assert_allclose(ratios.mean(axis=0)[good], 1.0, atol=0.02)
    # Manticore Eq. 2 for the scatter of P_i / P_bar
    np.testing.assert_allclose(ratios.var(axis=0)[good], np.asarray(ps["var"])[good], rtol=0.3)

    x = jnp.asarray(rng.standard_normal((16,) * 3))
    base = float(model.log_prior_field(x))
    assert base == pytest.approx(-0.5 * float(jnp.sum(x**2)))
    on = {"lambda_ps": 100.0}
    # amplitude-blind (it constrains the shape), so the Gaussian prior sets the scale
    pen = base - float(model.log_prior_field(x, on))
    pen2 = -0.5 * float(jnp.sum((2 * x) ** 2)) - float(model.log_prior_field(2 * x, on))
    assert pen > 0 and pen2 == pytest.approx(pen)
    # a field with a tilted spectrum is penalised far more than white noise
    red = jnp.fft.irfftn(jnp.fft.rfftn(x) / (1 + jnp.asarray(model.grid.kmag()) / model.grid.k_fundamental), s=x.shape)
    red = red / jnp.std(red)
    assert -0.5 * float(jnp.sum(red**2)) - float(model.log_prior_field(red, on)) > 50 * pen


# ---------------------------------------------------- Gaussianity (bispectrum) prior
def test_bispectrum_prior_calibration_sensitivity_and_gradient(small_cfg):
    cfg = update(small_cfg, {"grid": {"n_grid": 16}, "prior": {"lambda_ps": 100.0,
                 "bispectrum": {"lambda_bs": 100.0, "n_bins": 4, "n_mc": 400}}})
    m = FieldModel(cfg, build_response(cfg))
    assert m.hyper_defaults["lambda_ps"] == 100.0 and m.hyper_defaults["lambda_bs"] == 100.0
    nb = int(m._bs["idx"].shape[0])
    assert nb >= 10
    rng = np.random.default_rng(3)
    chi = [float(m.white_noise_bispectrum_chi2(jnp.asarray(rng.standard_normal((16,) * 3)))) for _ in range(60)]
    assert np.mean(chi) == pytest.approx(nb, rel=0.15)  # calibrated: chi2 ~ number of bins for Gaussian noise
    y = rng.standard_normal((16,) * 3)
    y = (y + 0.2 * (y**2 - 1)) / np.std(y + 0.2 * (y**2 - 1))
    assert float(m.white_noise_bispectrum_chi2(jnp.asarray(y))) > 5 * nb  # skewed noise is penalised
    x = jnp.asarray(rng.standard_normal((16,) * 3))
    base = float(update_free := FieldModel(update(cfg, {"prior": {"lambda_ps": 0.0, "bispectrum": {"lambda_bs": 0.0}}}),
                                         build_response(cfg)).log_prior_field(x))
    full = float(m.log_prior_field(x))
    ps_pen = 100.0 * 0.5 * float(jnp.sum(jnp.where(m._ps["good"], (m.white_noise_power_ratio(x) - 1) ** 2 / m._ps["var"], 0)))
    assert full == pytest.approx(base - ps_pen - 50.0 * float(m.white_noise_bispectrum_chi2(x)), rel=1e-10)
    directional_check(lambda v: m.log_prior_field(v), x)
    # hyper-parameters override the config values (schedules)
    assert float(m.log_prior_field(x, {"lambda_ps": 0.0, "lambda_bs": 0.0})) == pytest.approx(base)


def test_bispectrum_prior_off_skips_calibration(small_cfg):
    m = FieldModel(small_cfg, build_response(small_cfg))
    assert "mean" not in m._bs and m.hyper_defaults["lambda_bs"] == 0.0
    x = jnp.asarray(np.random.default_rng(0).standard_normal((8,) * 3))
    assert float(m.log_prior_field(x)) == pytest.approx(-0.5 * float(jnp.sum(x**2)))
    sched = update(small_cfg, {"sampler": {"n_warmup": 10, "schedules": {"lambda_bs": {"start": 0.0, "end": 50.0, "n_steps": 5}}}})
    assert "mean" in FieldModel(sched, build_response(sched))._bs  # calibrated when a schedule uses it
