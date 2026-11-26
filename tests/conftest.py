import jax
import pytest

jax.config.update("jax_enable_x64", True)

from miniflims.config import load_config  # noqa: E402


@pytest.fixture
def small_cfg():
    """8^3 Zel'dovich + power-law bias + Poisson; cheap enough for unit tests."""
    return load_config(
        overrides={
            "grid": {"n_grid": 8, "box_size": 200.0},
            "galaxies": {"nbar": {"init": 3.0}},
            "data": {"mock": {"truth": {"alpha": 1.4}}},
            "output": {"log_every": 0},
        }
    )
