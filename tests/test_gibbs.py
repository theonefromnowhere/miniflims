"""The Gibbs driver, checked against a model with a known joint posterior.

    x_i ~ N(0, 1),  y_i | x_i, sigma ~ N(x_i, sigma^2),  sigma ~ U(0.05, 5)

x is updated with HMC and sigma with slice sampling. Marginally
y_i | sigma ~ N(0, 1 + sigma^2), so the posterior of sigma is available on a
grid and E[x_i | y] = y_i E[1 / (1 + sigma^2) | y].
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from miniflims.config import HMCConfig, ScheduleConfig, SliceConfig
from miniflims.samplers import Block, hmc_block, make_hyper_fn, run_gibbs, schedule_value, slice_block

N = 60
LO, HI = 0.05, 5.0
Y = np.random.default_rng(0).standard_normal(N) * np.sqrt(1 + 0.7**2)


def joint(state, hyper):
    x, sigma = state["x"], state["sigma"][0, 0]
    beta = hyper.get("beta", 1.0)
    return -0.5 * jnp.sum(x**2) + beta * (
        -0.5 * jnp.sum((jnp.asarray(Y) - x) ** 2) / sigma**2 - N * jnp.log(sigma)
    )


def blocks():
    def make_conditional(state, hyper):
        def fn(row, aux_row):
            return joint({**state, "sigma": row[None]}, hyper)

        return fn, jnp.zeros(1)

    return [
        hmc_block("x", "x", joint, HMCConfig(step_size=0.05, n_steps=12, n_steps_min=4)),
        slice_block("sigma", "sigma", make_conditional, [LO], [HI], [0.5], SliceConfig()),
    ]


def init(n_chains, seed=0):
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    return {
        "x": jax.random.normal(k1, (n_chains, N)),
        "sigma": jax.random.uniform(k2, (n_chains, 1, 1), minval=0.2, maxval=3.0),
    }


def monitor(state, hyper):
    return {"sigma": state["sigma"][0, 0], "x": state["x"]}


def grid_posterior():
    s = np.linspace(LO, HI, 20_001)
    logp = -0.5 * (Y**2).sum() / (1 + s**2) - 0.5 * N * np.log(1 + s**2)
    w = np.exp(logp - logp.max())
    w /= w.sum()
    mean = (s * w).sum()
    return mean, np.sqrt(((s - mean) ** 2 * w).sum()), (w / (1 + s**2)).sum()


def test_hmc_within_gibbs_matches_analytic_posterior():
    res = run_gibbs(jax.random.PRNGKey(1), blocks(), init(4), n_warmup=500, n_samples=4000, monitor=monitor)
    sig = res.trace["sigma"][:, 500:]
    mean, std, shrink = grid_posterior()
    assert sig.min() >= LO and sig.max() <= HI
    assert sig.mean() == pytest.approx(mean, abs=0.25 * std)
    assert sig.std() == pytest.approx(std, rel=0.2)
    x_mean = res.trace["x"][:, 500:].mean(axis=(0, 1))
    np.testing.assert_allclose(x_mean, shrink * Y, atol=0.12)
    # all chains agree despite over-dispersed starts
    assert np.ptp(sig.mean(axis=1)) < std


def test_trace_and_sample_shapes_thinning_and_warmup_split():
    res = run_gibbs(
        jax.random.PRNGKey(2), blocks(), init(3), n_warmup=7, n_samples=20,
        monitor=monitor, collect=lambda s: {"x": s["x"]}, thin=4,
    )
    assert res.n_warmup == 7
    assert res.trace["sigma"].shape == (3, 27)
    assert res.trace["x/accept_prob"].shape == (3, 27)
    assert res.trace["sigma/widths"].shape == (3, 27, 1, 1)
    assert res.samples["x"].shape == (3, 5, N)  # sweeps 7, 11, 15, 19, 23
    np.testing.assert_array_equal(res.samples["x"][:, 0], res.trace["x"][:, 7])
    np.testing.assert_array_equal(res.samples["x"][:, -1], res.trace["x"][:, 23])
    assert res.state["x"].shape == (3, N)
    np.testing.assert_array_equal(np.asarray(res.state["x"]), res.trace["x"][:, -1])


def test_reproducible_and_seed_dependent():
    kw = dict(n_warmup=5, n_samples=5, monitor=monitor)
    a = run_gibbs(jax.random.PRNGKey(3), blocks(), init(2), **kw)
    b = run_gibbs(jax.random.PRNGKey(3), blocks(), init(2), **kw)
    c = run_gibbs(jax.random.PRNGKey(4), blocks(), init(2), **kw)
    np.testing.assert_array_equal(a.trace["x"], b.trace["x"])
    assert not np.array_equal(a.trace["x"], c.trace["x"])
    # chains are independent of each other
    assert not np.array_equal(a.trace["x"][0], a.trace["x"][1])


def test_block_order_and_custom_blocks():
    calls = []

    def recorder(name):
        def step(key, state, bstate, hyper, warmup):
            calls.append((name, warmup))  # runs at trace time: once per compiled sweep
            return {**state, "log": state["log"] * 10 + int(name)}, bstate + 1, {"seen": state["log"]}

        return Block(name, lambda state: jnp.zeros(()), step)

    res = run_gibbs(
        jax.random.PRNGKey(0), [recorder("1"), recorder("2")], {"log": jnp.zeros((1,))},
        n_warmup=1, n_samples=1,
    )
    assert calls == [("1", True), ("2", True), ("1", False), ("2", False)]
    assert float(res.state["log"][0]) == 1212.0  # blocks applied in order, twice
    assert float(res.trace["2/seen"][0, 0]) == 1.0
    assert float(res.block_states["1"][0]) == 2.0
    with pytest.raises(ValueError):
        run_gibbs(jax.random.PRNGKey(0), [recorder("1"), recorder("1")], {"log": jnp.zeros((1,))},
                  n_warmup=0, n_samples=1)


def test_hyper_schedule_reaches_blocks_without_recompiling():
    sched = {"beta": ScheduleConfig(start=0.0, end=1.0, n_steps=10)}
    hyper_fn = make_hyper_fn(sched, {"beta": 1.0}, n_warmup=10)
    res = run_gibbs(
        jax.random.PRNGKey(5), blocks(), init(2), n_warmup=10, n_samples=3, hyper_fn=hyper_fn,
        monitor=lambda s, h: {"beta": h["beta"]},
    )
    np.testing.assert_allclose(res.trace["beta"][0], list(np.arange(10) / 10) + [1.0] * 3)


def test_tempered_target_is_the_prior_at_beta_zero():
    hyper_fn = make_hyper_fn({"beta": ScheduleConfig(start=0.0, end=0.0)}, {"beta": 1.0}, 0)
    res = run_gibbs(jax.random.PRNGKey(6), blocks(), init(4), n_warmup=300, n_samples=2500,
                    hyper_fn=hyper_fn, monitor=monitor)
    sig, x = res.trace["sigma"][:, 300:], res.trace["x"][:, 300:]
    assert sig.mean() == pytest.approx((LO + HI) / 2, abs=0.12)  # uniform prior
    assert x.std() == pytest.approx(1.0, rel=0.05) and abs(x.mean()) < 0.03


# -------------------------------------------------------------------- schedules
def test_schedule_values():
    lin = ScheduleConfig(start=2.0, end=0.5, n_steps=3)
    assert [schedule_value(lin, i) for i in range(5)] == pytest.approx([2.0, 1.5, 1.0, 0.5, 0.5])
    geo = ScheduleConfig(start=0.01, end=1.0, n_steps=2, kind="geometric")
    assert [schedule_value(geo, i) for i in range(3)] == pytest.approx([0.01, 0.1, 1.0])
    assert schedule_value(ScheduleConfig(start=3.0, end=7.0, n_steps=0), 0) == 7.0


def test_schedule_validation():
    with pytest.raises(ValueError):  # unknown hyper-parameter
        make_hyper_fn({"gamma": ScheduleConfig()}, {"beta": 1.0}, 10)
    with pytest.raises(ValueError):  # would leak into the sampling phase
        make_hyper_fn({"beta": ScheduleConfig(n_steps=11)}, {"beta": 1.0}, 10)
    with pytest.raises(ValueError):
        make_hyper_fn({"beta": ScheduleConfig(start=0.0, end=1.0, n_steps=5, kind="geometric")}, {"beta": 1.0}, 10)
    with pytest.raises(ValueError):
        make_hyper_fn({"beta": ScheduleConfig(n_steps=5, kind="cosine")}, {"beta": 1.0}, 10)
    assert float(make_hyper_fn({}, {"beta": 0.3}, 0)(5)["beta"]) == pytest.approx(0.3)
