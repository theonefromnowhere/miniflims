import pytest
import yaml

from miniflims import config as C


def test_defaults_and_roundtrip():
    cfg = C.load_config()
    assert cfg.grid.n_grid == 32
    assert cfg.sampler.blocks == ("field", "nuisance")
    again = C.from_dict(yaml.safe_load(C.dump_yaml(cfg)))
    assert again == cfg


def test_yaml_file_and_dotted_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "grid: {n_grid: 16}\n"
        "sampler: {hmc: {step_size: 1e-3}}\n"  # YAML reads 1e-3 as a string
        "galaxies: {bias: {model: stdp, params: {alpha: {init: 2.0, sample: false}}}}\n"
    )
    cfg = C.load_config(p, ["sampler.hmc.n_steps=7", "grid.box_size=250", "sampler.blocks=[field]"])
    assert cfg.grid.n_grid == 16 and cfg.grid.box_size == 250.0
    assert cfg.sampler.hmc.step_size == pytest.approx(1e-3)
    assert cfg.sampler.hmc.n_steps == 7
    assert cfg.sampler.blocks == ("field",)
    assert cfg.galaxies.bias.params["alpha"].init == 2.0
    assert cfg.galaxies.bias.params["alpha"].sample is False
    assert cfg.galaxies.bias.params["alpha"].lower is None


def test_file_values_survive_partial_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("sampler: {n_chains: 5, hmc: {n_steps: 11}}\n")
    cfg = C.load_config(p, ["sampler.hmc.step_size=0.5"])
    assert cfg.sampler.n_chains == 5 and cfg.sampler.hmc.n_steps == 11


def test_update_returns_new_config():
    cfg = C.load_config()
    new = C.update(cfg, ["cosmology.h=0.7"])
    assert new.cosmology.h == 0.7 and cfg.cosmology.h == 0.681
    assert new.grid == cfg.grid


def test_nested_lists_and_schedules():
    cfg = C.load_config(
        overrides={
            "data": {"mock": {"selections": [{"type": "radial", "r0": 80}, {"type": "uniform"}]}},
            "sampler": {"schedules": {"lambda_ps": {"start": 10, "end": 100, "n_steps": 5}}},
            "forward": {"rsd": {"observer": [1, 2, 3]}},
        }
    )
    assert [s.type for s in cfg.data.mock.selections] == ["radial", "uniform"]
    assert cfg.data.mock.selections[0].r0 == 80.0
    assert cfg.sampler.schedules["lambda_ps"].end == 100.0
    assert cfg.forward.rsd.observer == (1.0, 2.0, 3.0)


@pytest.mark.parametrize(
    "bad",
    [
        {"grid": {"n_gird": 8}},  # typo
        {"nonsense": 1},
        {"grid": {"n_grid": 8.5}},
        {"grid": {"n_grid": "many"}},
        {"grid": {"box_size": None}},
        {"forward": {"rsd": {"enabled": "yes"}}},
        {"sampler": {"blocks": "field"}},
        {"galaxies": {"bias": {"params": {"alpha": {"initt": 1}}}}},
    ],
)
def test_invalid_config_raises(bad):
    with pytest.raises(C.ConfigError):
        C.from_dict(bad)


def test_bad_override_syntax():
    with pytest.raises(C.ConfigError):
        C.parse_overrides(["grid.n_grid"])
