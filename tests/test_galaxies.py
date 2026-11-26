"""Bias models and count likelihoods."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from miniflims.bias import BIAS_MODELS, positive
from miniflims.likelihood import LIKELIHOODS
from miniflims.params import ParamSpec, apply_override, resolve_specs
from miniflims.config import ParamOverride

FLOOR = 1e-3
RHO = jnp.asarray(np.linspace(0.0, 6.0, 25))


def defaults(model):
    return {s.name: s.init for s in model.param_specs}


# ------------------------------------------------------------------------ specs
def test_param_spec_validation_and_overrides():
    spec = ParamSpec("a", 1.0, 0.0, 2.0, 0.1)
    new = apply_override(spec, ParamOverride(init=1.5, sample=False))
    assert (new.init, new.sample, new.lower, new.upper) == (1.5, False, 0.0, 2.0)
    assert apply_override(spec, None) is spec
    with pytest.raises(ValueError):
        apply_override(spec, ParamOverride(init=3.0))  # outside bounds
    with pytest.raises(ValueError):
        apply_override(spec, ParamOverride(prior="cauchy"))
    with pytest.raises(ValueError):
        resolve_specs([spec], {"b": ParamOverride(init=1.0)}, "test")


# ------------------------------------------------------------------------- bias
@pytest.mark.parametrize("name", BIAS_MODELS.names())
def test_bias_is_finite_and_differentiable(name):
    model = BIAS_MODELS.get(name)()
    p = defaults(model)
    b = model(RHO, p, FLOOR)
    assert b.shape == RHO.shape and bool(jnp.all(jnp.isfinite(b)))
    if name != "linear":
        assert bool(jnp.all(b >= 0))
    for key in p:
        g = jax.grad(lambda v: jnp.sum(model(RHO, {**p, key: v}, FLOOR)))(p[key])
        assert np.isfinite(float(g)), key
    g = jax.grad(lambda r: jnp.sum(model(r, p, FLOOR)))(RHO)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_bias_special_cases():
    r = positive(RHO, FLOOR)
    lin = BIAS_MODELS.get("linear")()
    np.testing.assert_allclose(lin(RHO, {"b": 2.0}, FLOOR), 1 + 2 * (RHO - 1))
    pl = BIAS_MODELS.get("power_law")()
    np.testing.assert_allclose(pl(RHO, {"alpha": 1.7}, FLOOR), r**1.7)

    tpl = BIAS_MODELS.get("truncated_power_law")()
    b = tpl(RHO, {"alpha": 1.7, "rho_g": 0.5, "epsilon": 1.5}, FLOOR)
    np.testing.assert_allclose(b, r**1.7 * np.exp(-((r / 0.5) ** -1.5)))
    assert float(b[0]) < 1e-30  # voids are emptied
    # truncation scale -> 0 recovers the power law
    b = tpl(RHO[1:], {"alpha": 1.7, "rho_g": 1e-3, "epsilon": 4.0}, FLOOR)
    np.testing.assert_allclose(b, r[1:] ** 1.7, rtol=1e-6)


def test_broken_power_law_limits():
    bpl = BIAS_MODELS.get("broken_power_law")()
    r = positive(RHO, FLOOR)
    p = {"n": 2.5, "log_rhob": 0.3}
    rb = 10**0.3
    np.testing.assert_allclose(bpl(RHO, p, FLOOR), (1 + (r / rb) ** 2) ** 1.25, rtol=1e-12)
    lo, hi = jnp.asarray([1e-3 * rb]), jnp.asarray([1e3 * rb])
    assert float(bpl(lo, p, 0.0)[0]) == pytest.approx(1.0, rel=1e-5)  # flat below the break
    assert float(bpl(hi, p, 0.0)[0]) == pytest.approx(1e3**2.5, rel=1e-5)  # slope n above it


def test_stdp_reductions():
    stdp = BIAS_MODELS.get("stdp")()
    r = positive(RHO, FLOOR)
    base = dict(gamma_t=-50.0, gamma_s=0.5, alpha=1.3, beta=0.0, beta_0=1.0)
    # no truncation, beta = 0 -> single power law (Manticore Sec. 2)
    np.testing.assert_allclose(stdp(RHO, base, FLOOR), r**1.3, rtol=1e-10)
    # sigmoid alone: half suppression at ln(rho) = gamma_t
    p = {**base, "gamma_t": 0.4}
    at_t = stdp(jnp.asarray([np.exp(0.4) - FLOOR]), p, FLOOR)
    assert float(at_t[0]) == pytest.approx(0.5 * np.exp(0.4) ** 1.3, rel=1e-6)
    # second power law: explicit formula and high-density slope alpha - beta^2
    p = {**base, "beta": 0.8, "beta_0": 0.5}
    expect = r**1.3 * (1 + np.exp(0.8 * (np.log(r) - 0.5))) ** -0.8
    np.testing.assert_allclose(stdp(RHO, p, FLOOR), expect, rtol=1e-10)
    hi = stdp(jnp.asarray([1e6, 2e6]), p, FLOOR)
    assert float(jnp.log(hi[1] / hi[0]) / np.log(2)) == pytest.approx(1.3 - 0.8**2, abs=1e-3)


# ------------------------------------------------------------------- likelihood
def pointwise(like, counts, lam, rho, p):
    f = lambda n: like.log_prob(n, lam, rho, p, True, FLOOR)
    return np.asarray(jax.vmap(f)(jnp.asarray(counts, float)))


def test_poisson_matches_scipy():
    like = LIKELIHOODS.get("poisson")()
    n = np.arange(0, 40)
    np.testing.assert_allclose(pointwise(like, n, 3.7, 1.0, {}), stats.poisson.logpmf(n, 3.7), atol=1e-10)


def test_gaussian_matches_scipy():
    like = LIKELIHOODS.get("gaussian")()
    x = np.linspace(-3, 9, 20)
    np.testing.assert_allclose(
        pointwise(like, x, 3.0, 1.0, {"sigma": 1.7}), stats.norm.logpdf(x, 3.0, 1.7), atol=1e-10
    )


@pytest.mark.parametrize("lam,b_alpha,b_beta,rho", [(4.0, 0.3, 0.0, 1.0), (1.5, 0.2, 0.5, 3.0), (6.0, 0.6, -0.5, 2.0)])
def test_generalized_poisson_is_normalised_with_stated_moments(lam, b_alpha, b_beta, rho):
    like = LIKELIHOODS.get("generalized_poisson")()
    p = {"b_alpha": b_alpha, "b_beta": b_beta}
    b = float(like.dispersion(rho, p, FLOOR))
    assert b == pytest.approx(b_alpha * (rho + FLOOR) ** b_beta)
    n = np.arange(0, 2000)
    pmf = np.exp(pointwise(like, n, lam, rho, p))
    assert pmf.sum() == pytest.approx(1.0, abs=1e-8)
    mean = (n * pmf).sum()
    assert mean == pytest.approx(lam, rel=1e-6)  # Manticore Eq. 23
    assert ((n - mean) ** 2 * pmf).sum() == pytest.approx(lam / (1 - b) ** 2, rel=1e-6)  # Eq. 24


def test_generalized_poisson_reduces_to_poisson():
    gp, po = LIKELIHOODS.get("generalized_poisson")(), LIKELIHOODS.get("poisson")()
    n = np.arange(0, 30)
    np.testing.assert_allclose(
        pointwise(gp, n, 2.5, 1.0, {"b_alpha": 0.0, "b_beta": 0.3}), pointwise(po, n, 2.5, 1.0, {}), atol=1e-10
    )


def test_dispersion_is_clipped():
    gp = LIKELIHOODS.get("generalized_poisson")()
    b = gp.dispersion(jnp.asarray([0.0, 1.0, 100.0]), {"b_alpha": 0.9, "b_beta": 2.0}, FLOOR)
    assert float(b.max()) == gp.b_max and float(b.min()) >= 0
    lp = pointwise(gp, np.arange(50), 5.0, 100.0, {"b_alpha": 0.9, "b_beta": 2.0})
    assert np.all(np.isfinite(lp))


@pytest.mark.parametrize("name", LIKELIHOODS.names())
def test_mask_and_empty_voxels(name):
    like = LIKELIHOODS.get(name)()
    p = defaults(like)
    counts = jnp.asarray([2.0, 0.0, 5.0])
    lam = jnp.asarray([1.5, 0.0, 4.0])
    rho = jnp.asarray([1.0, 0.5, 2.0])
    full = like.log_prob(counts, lam, rho, p, jnp.asarray([True, False, True]), FLOOR)
    parts = [like.log_prob(counts[i], lam[i], rho[i], p, True, FLOOR) for i in (0, 2)]
    assert float(full) == pytest.approx(float(parts[0] + parts[1]))
    # zero expected counts in a masked voxel must not poison the gradient
    g = jax.grad(lambda l: like.log_prob(counts, l, rho, p, jnp.asarray([True, False, True]), FLOOR))(lam)
    assert bool(jnp.all(jnp.isfinite(g))) and float(g[1]) == 0.0


@pytest.mark.parametrize(
    "name,p,var_factor",
    [("poisson", {}, 1.0), ("generalized_poisson", {"b_alpha": 0.4, "b_beta": 0.0}, 1 / 0.6**2)],
)
def test_count_samplers_have_model_moments(name, p, var_factor):
    like = LIKELIHOODS.get(name)()
    rng = np.random.default_rng(0)
    lam = np.full(200_000, 3.0)
    x = like.sample(rng, lam, np.ones_like(lam), p, FLOOR)
    assert np.all(x >= 0) and np.all(x == np.round(x))
    assert x.mean() == pytest.approx(3.0, rel=0.02)
    assert x.var() == pytest.approx(3.0 * var_factor, rel=0.03)


def test_generalized_poisson_sampler_matches_pmf():
    like = LIKELIHOODS.get("generalized_poisson")()
    p = {"b_alpha": 0.35, "b_beta": 0.0}
    x = like.sample(np.random.default_rng(1), np.full(100_000, 2.0), np.ones(100_000), p, FLOOR)
    n = np.arange(0, 12)
    pmf = np.exp(pointwise(like, n, 2.0, 1.0, p))
    freq = np.array([(x == k).mean() for k in n])
    np.testing.assert_allclose(freq, pmf, atol=5e-3)


def test_gaussian_sampler():
    like = LIKELIHOODS.get("gaussian")()
    x = like.sample(np.random.default_rng(2), np.full(100_000, 2.0), None, {"sigma": 0.5}, FLOOR)
    assert x.mean() == pytest.approx(2.0, abs=0.01) and x.std() == pytest.approx(0.5, rel=0.02)
