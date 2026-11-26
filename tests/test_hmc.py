import jax
import jax.numpy as jnp
import numpy as np
import pytest

from miniflims.config import HMCConfig
from miniflims.samplers import (
    INTEGRATORS,
    dual_averaging_init,
    dual_averaging_update,
    hmc_block,
    hmc_step,
    integrator_step,
    run_gibbs,
)

SCALES = jnp.asarray([0.5, 1.0, 2.0, 1.0, 0.7])


def gauss_vg(x):
    return jax.value_and_grad(lambda v: -0.5 * jnp.sum((v / SCALES) ** 2))(x)


def integrate(name, eps, n, x, p):
    coeffs = INTEGRATORS.get(name)
    lp, g = gauss_vg(x)
    for _ in range(n):
        x, p, lp, g = integrator_step(gauss_vg, coeffs, eps, x, p, g)
    return x, p, lp


def energy(x, p):
    return 0.5 * float(jnp.sum((x / SCALES) ** 2) + jnp.sum(p**2))


X0 = jnp.asarray([0.3, -1.2, 2.0, 0.1, -0.4])
P0 = jnp.asarray([1.0, 0.5, -0.7, 0.2, -1.1])


# ------------------------------------------------------------------ integrators
@pytest.mark.parametrize("name", INTEGRATORS.names())
def test_integrator_coefficients_are_consistent_and_symmetric(name):
    a, b = INTEGRATORS.get(name)
    assert len(b) == len(a) + 1
    assert sum(a) == pytest.approx(1.0) and sum(b) == pytest.approx(1.0)
    assert tuple(a) == pytest.approx(tuple(reversed(a)))
    assert tuple(b) == pytest.approx(tuple(reversed(b)))


@pytest.mark.parametrize("name", INTEGRATORS.names())
def test_integrator_is_time_reversible(name):
    x1, p1, _ = integrate(name, 0.1, 25, X0, P0)
    x2, p2, _ = integrate(name, 0.1, 25, x1, -p1)
    np.testing.assert_allclose(x2, X0, atol=1e-10)
    np.testing.assert_allclose(-p2, P0, atol=1e-10)


@pytest.mark.parametrize("name,order", [("leapfrog", 2), ("mclachlan2", 2), ("yoshida4", 4)])
def test_integrator_order(name, order):
    # global error at fixed integration time scales as eps^order
    def err(eps, n):
        x, p, _ = integrate(name, eps, n, X0, P0)
        x_ref, p_ref, _ = integrate("yoshida4", 1e-3, 1000, X0, P0)
        return float(jnp.linalg.norm(x - x_ref) + jnp.linalg.norm(p - p_ref))

    rate = np.log2(err(0.1, 10) / err(0.05, 20))
    assert rate == pytest.approx(order, abs=0.25)


def test_minimum_error_integrator_beats_leapfrog_at_equal_cost():
    h0 = energy(X0, P0)
    x, p, _ = integrate("leapfrog", 0.1, 20, X0, P0)  # 20 gradients
    lf = abs(energy(x, p) - h0)
    x, p, _ = integrate("mclachlan2", 0.2, 10, X0, P0)  # 20 gradients
    assert abs(energy(x, p) - h0) < lf


def test_integrator_returns_logdensity_at_endpoint():
    x, _, lp = integrate("mclachlan2", 0.1, 3, X0, P0)
    assert float(lp) == pytest.approx(float(gauss_vg(x)[0]))


# ----------------------------------------------------------------------- kernel
def test_hmc_step_exact_in_small_step_limit():
    x, info = hmc_step(jax.random.PRNGKey(0), X0, gauss_vg, 1e-4, 10)
    assert float(info.accept_prob) > 0.999 and bool(info.accepted)
    assert abs(float(info.delta_energy)) < 1e-6
    assert float(info.logdensity) == pytest.approx(float(gauss_vg(x)[0]))


def test_hmc_step_rejects_divergent_and_nan_trajectories():
    x, info = hmc_step(jax.random.PRNGKey(0), X0, gauss_vg, 50.0, 20)  # wildly unstable
    np.testing.assert_array_equal(x, X0)
    assert not bool(info.accepted) and float(info.accept_prob) == 0.0
    assert float(info.logdensity) == pytest.approx(float(gauss_vg(X0)[0]))

    def nan_vg(v):
        lp, g = gauss_vg(v)
        bad = jnp.any(v != X0)
        return jnp.where(bad, jnp.nan, lp), g

    x, info = hmc_step(jax.random.PRNGKey(1), X0, nan_vg, 0.1, 3)
    np.testing.assert_array_equal(x, X0)
    assert not bool(info.accepted)


@pytest.mark.parametrize("name,eps", [("leapfrog", 0.25), ("mclachlan2", 0.5), ("yoshida4", 0.6)])
def test_hmc_samples_gaussian(name, eps):
    def step(x, key):
        k1, k2 = jax.random.split(key)
        n = jax.random.randint(k1, (), 3, 12)  # traced, randomised trajectory length
        x, info = hmc_step(k2, x, gauss_vg, eps, n, name)
        return x, (x, info.accept_prob)

    keys = jax.random.split(jax.random.PRNGKey(2), 6000)
    _, (xs, acc) = jax.jit(lambda: jax.lax.scan(step, X0, keys))()
    xs = np.asarray(xs)[500:]
    assert 0.6 < float(acc.mean()) < 1.0
    np.testing.assert_allclose(xs.mean(axis=0), 0.0, atol=0.15)
    np.testing.assert_allclose(xs.std(axis=0), SCALES, rtol=0.08)
    assert np.all(np.abs(np.corrcoef(xs.T) - np.eye(5)) < 0.08)


def test_hmc_array_shaped_position_and_mass():
    vg = jax.value_and_grad(lambda v: -0.5 * jnp.sum(v**2) / 4.0)  # N(0, 2^2)
    x0 = jnp.zeros((2, 3))

    def step(x, key):
        x, info = hmc_step(key, x, vg, 0.4, 5, inv_mass=4.0)  # M = Sigma^-1
        return x, x

    _, xs = jax.lax.scan(step, x0, jax.random.split(jax.random.PRNGKey(3), 4000))
    assert xs.shape == (4000, 2, 3)
    np.testing.assert_allclose(np.asarray(xs).std(axis=0), 2.0, rtol=0.1)


# -------------------------------------------------------------- dual averaging
def test_dual_averaging_moves_step_size_in_the_right_direction():
    s = dual_averaging_init(0.1)
    for _ in range(20):
        s = dual_averaging_update(s, 1.0, 0.7)  # always accepted -> grow
    assert float(s.log_eps_bar) > np.log(0.1)
    s = dual_averaging_init(0.1)
    for _ in range(20):
        s = dual_averaging_update(s, 0.0, 0.7)  # always rejected -> shrink
    assert float(s.log_eps_bar) < np.log(0.1)


# ------------------------------------------------------------------------ block
def _block(cfg):
    logdensity = lambda state, hyper: -0.5 * jnp.sum((state["x"] / state["scale"]) ** 2)
    return hmc_block("x", "x", logdensity, cfg)


@pytest.mark.parametrize("target", [0.6, 0.85])
def test_hmc_block_adapts_step_size_to_target_acceptance(target):
    cfg = HMCConfig(step_size=1e-3, n_steps=10, n_steps_min=5, target_accept=target)
    dim = 200
    state = {"x": jnp.ones((3, dim)), "scale": jnp.ones((3, dim))}
    res = run_gibbs(jax.random.PRNGKey(0), [_block(cfg)], state, n_warmup=400, n_samples=600,
                    monitor=lambda s, h: {"var": jnp.var(s["x"])})
    acc = res.trace["x/accept_prob"][:, 400:]
    assert acc.mean() == pytest.approx(target, abs=0.08)
    eps = res.trace["x/step_size"]
    assert eps[:, 400:].mean() > 50 * 1e-3  # grew from the tiny initial value
    # frozen after warm-up: within each chain only the jitter (20 %) remains
    post = eps[:, 400:]
    assert np.all(post.max(axis=1) / post.min(axis=1) < 1.0 / 0.8 + 1e-6)
    assert np.all(post.max(axis=1) / post.min(axis=1) > 1.15)
    n = res.trace["x/n_steps"]
    assert n.min() == 5 and n.max() == 10
    assert res.trace["var"][:, 400:].mean() == pytest.approx(1.0, rel=0.1)
    np.testing.assert_array_equal(np.asarray(res.state["scale"]), 1.0)  # untouched


def test_hmc_block_without_adaptation_uses_configured_step():
    cfg = HMCConfig(step_size=0.05, n_steps=4, step_jitter=0.0, adapt_step_size=False, n_per_sweep=2)
    state = {"x": jnp.ones((1, 10)), "scale": jnp.ones((1, 10))}
    res = run_gibbs(jax.random.PRNGKey(0), [_block(cfg)], state, n_warmup=5, n_samples=5)
    np.testing.assert_allclose(res.trace["x/step_size"], 0.05)
    assert np.all(res.trace["x/n_steps"] == 4)


@pytest.mark.parametrize(
    "bad", [dict(n_steps=5, n_steps_min=9), dict(n_steps_min=0), dict(step_jitter=1.0), dict(integrator="euler")]
)
def test_hmc_block_validates_config(bad):
    with pytest.raises((ValueError, KeyError)):
        _block(HMCConfig(**bad))
