import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from miniflims.config import SliceConfig
from miniflims.samplers import run_gibbs, slice_block, slice_sample_1d


def chain(logp, x0, n, width=1.0, seed=0, **kw):
    def step(x, key):
        x, info = slice_sample_1d(key, logp, x, width, **kw)
        return x, (x, info)

    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    _, (xs, info) = jax.jit(lambda: jax.lax.scan(step, jnp.asarray(x0, float), keys))()
    return np.asarray(xs), info


@pytest.mark.parametrize("width", [0.2, 1.0, 50.0])
def test_standard_normal_for_any_width(width):
    xs, info = chain(lambda x: -0.5 * x**2, 0.3, 20_000, width)
    assert bool(jnp.all(info.moved))
    assert xs.mean() == pytest.approx(0.0, abs=0.06)
    assert xs.var() == pytest.approx(1.0, abs=0.08)
    assert stats.kstest(xs[::10], "norm").pvalue > 1e-3


def test_half_line_support():
    xs, _ = chain(lambda x: -x, 1.0, 20_000, lower=0.0)
    assert xs.min() >= 0.0
    assert stats.kstest(xs[::10], "expon").pvalue > 1e-3


def test_bounded_support_beta():
    logp = lambda x: jnp.log(x) + 4.0 * jnp.log1p(-x)  # Beta(2, 5)
    xs, _ = chain(logp, 0.5, 20_000, width=0.3, lower=0.0, upper=1.0)
    assert xs.min() >= 0.0 and xs.max() <= 1.0
    assert stats.kstest(xs[::10], "beta", args=(2, 5)).pvalue > 1e-3


def test_uniform_on_box_ignores_density_outside():
    # the target is only ever evaluated inside [lower, upper]
    logp = lambda x: jnp.where((x >= -1.0) & (x <= 2.0), 0.0, jnp.nan)
    xs, _ = chain(logp, 0.0, 10_000, width=0.5, lower=-1.0, upper=2.0)
    assert stats.kstest(xs[::5], "uniform", args=(-1, 3)).pvalue > 1e-3


def test_bimodal_target():
    logp = lambda x: jnp.logaddexp(-0.5 * (x - 2.0) ** 2, -0.5 * (x + 2.0) ** 2)
    xs, _ = chain(logp, 2.0, 40_000, width=2.0)
    assert xs.mean() == pytest.approx(0.0, abs=0.2)
    assert xs.var() == pytest.approx(5.0, rel=0.08)
    assert (xs > 0).mean() == pytest.approx(0.5, abs=0.05)


def test_limited_stepout_still_exact():
    xs, _ = chain(lambda x: -0.5 * x**2, 0.0, 30_000, width=0.3, max_stepout=2)
    assert xs.var() == pytest.approx(1.0, abs=0.12)
    assert xs.mean() == pytest.approx(0.0, abs=0.15)


def test_degenerate_target_stays_put():
    # -inf everywhere except the current point: shrinkage must terminate
    logp = lambda x: jnp.where(x == 0.25, 0.0, -jnp.inf)
    x, info = slice_sample_1d(jax.random.PRNGKey(0), logp, 0.25, 1.0, max_shrink=20)
    assert float(x) == 0.25 and not bool(info.moved) and int(info.n_evals) <= 1 + 2 + 20


def test_evaluation_count_and_vmap():
    keys = jax.random.split(jax.random.PRNGKey(1), 64)
    x0 = jnp.linspace(-1, 1, 64)
    f = jax.vmap(lambda k, x: slice_sample_1d(k, lambda v: -0.5 * v**2, x, 1.0))
    x1, info = jax.jit(f)(keys, x0)
    assert x1.shape == (64,) and bool(jnp.all(info.n_evals >= 4))
    assert bool(jnp.all(info.width > 0))


# ------------------------------------------------------------------------ block
def _independent_rows_block(cfg, sampled=None):
    # rows: N(mu_r, 1) x N(0, s_r^2), with per-row aux = (mu_r, s_r)
    def make_conditional(state, hyper):
        def fn(row, aux_row):
            mu, s = aux_row
            return -0.5 * (row[0] - mu) ** 2 - 0.5 * (row[1] / s) ** 2

        return fn, (jnp.asarray([-3.0, 0.0, 4.0]), jnp.asarray([0.1, 1.0, 10.0]))

    return slice_block(
        "theta", "theta", make_conditional, [-100.0, -100.0], [100.0, 100.0], [1.0, 1.0], cfg, sampled
    )


def test_slice_block_rows_and_width_adaptation():
    block = _independent_rows_block(SliceConfig())
    state = {"theta": jnp.zeros((2, 3, 2))}  # 2 chains, 3 rows, 2 params
    res = run_gibbs(
        jax.random.PRNGKey(0), [block], state, n_warmup=300, n_samples=3000,
        monitor=lambda s, h: {"theta": s["theta"]},
    )
    th = res.trace["theta"][:, 300:]
    np.testing.assert_allclose(th[..., 0].mean(axis=(0, 1)), [-3.0, 0.0, 4.0], atol=0.1)
    np.testing.assert_allclose(th[..., 1].std(axis=(0, 1)), [0.1, 1.0, 10.0], rtol=0.1)
    # adapted bracket widths follow the conditional scales over two decades
    w = np.asarray(res.block_states["theta"].widths).mean(axis=0)[:, 1]
    assert w[0] < 1.0 < w[2] and 30 < w[2] / w[0] < 300
    # frozen after warm-up
    widths = res.trace["theta/widths"]
    np.testing.assert_array_equal(widths[:, 300], widths[:, -1])
    assert not np.array_equal(widths[:, 0], widths[:, 299])


def test_slice_block_fixed_parameters_and_no_adaptation():
    block = _independent_rows_block(SliceConfig(adapt_width=False), sampled=(True, False))
    state = {"theta": jnp.full((1, 3, 2), 0.5)}
    res = run_gibbs(
        jax.random.PRNGKey(1), [block], state, n_warmup=20, n_samples=50,
        monitor=lambda s, h: {"theta": s["theta"]},
    )
    assert np.all(res.trace["theta"][..., 1] == 0.5)
    assert np.unique(res.trace["theta"][..., 0]).size > 100
    assert np.all(res.trace["theta/widths"] == 1.0)
