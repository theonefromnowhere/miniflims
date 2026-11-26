"""sampler.gates: a Gibbs block that acts only once a condition on the state holds."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from miniflims.config import ConfigError, GateConfig
from miniflims.pipeline import apply_gates
from miniflims.samplers import Block, gated_block


def _shift_block():
    """Adds 1 to state['theta'] and counts its own steps in its block state."""

    def init(state):
        return jnp.zeros(())

    def step(key, state, bstate, hyper, warmup):
        return {**state, "theta": state["theta"] + 1.0}, bstate + 1.0, {"n": jnp.int32(1)}

    return Block("png", init, step)


@pytest.mark.parametrize("scale, expect_open", [(0.5, False), (1.2, True)])
def test_gated_block_acts_only_when_open(scale, expect_open):
    blk = apply_gates([_shift_block()], {"png": GateConfig(min_field_variance=0.95)}, logger=lambda m: None)[0]
    field = scale * jax.random.normal(jax.random.PRNGKey(0), (16, 16, 16))
    state = {"field": field, "theta": jnp.zeros(3)}
    new, bs, info = jax.jit(blk.step, static_argnums=4)(jax.random.PRNGKey(1), state, jnp.zeros(()), {}, True)
    assert bool(info["active"]) is expect_open
    np.testing.assert_array_equal(np.asarray(new["theta"]), np.full(3, 1.0 if expect_open else 0.0))
    assert float(bs) == (1.0 if expect_open else 0.0)  # adaptation frozen while closed
    np.testing.assert_array_equal(np.asarray(new["field"]), np.asarray(field))


def test_gate_is_per_chain_under_vmap():
    blk = gated_block(_shift_block(), lambda st: jnp.mean(st["field"] ** 2) >= 0.95)
    field = jnp.stack([0.5 * jnp.ones((8, 8, 8)), 1.1 * jnp.ones((8, 8, 8))])
    state = {"field": field, "theta": jnp.zeros((2, 3))}
    step = jax.vmap(lambda k, s, b: blk.step(k, s, b, {}, False))
    new, bs, info = step(jax.random.split(jax.random.PRNGKey(0), 2), state, jnp.zeros(2))
    np.testing.assert_array_equal(np.asarray(info["active"]), [False, True])
    np.testing.assert_array_equal(np.asarray(new["theta"][:, 0]), [0.0, 1.0])


def test_unknown_or_field_gate_rejected():
    with pytest.raises(ConfigError):
        apply_gates([_shift_block()], {"nuisance": GateConfig(0.9)}, logger=lambda m: None)
    fld = Block("field", lambda s: None, lambda *a: a)
    with pytest.raises(ConfigError):
        apply_gates([fld], {"field": GateConfig(0.9)}, logger=lambda m: None)


def test_ungated_and_empty_gate_pass_through():
    b = _shift_block()
    assert apply_gates([b], {}, logger=lambda m: None)[0] is b
    assert apply_gates([b], {"png": GateConfig()}, logger=lambda m: None)[0] is b
