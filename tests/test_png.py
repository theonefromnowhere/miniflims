"""Primordial non-Gaussianity through jaxpng: physics, conditionals, gradients,
memory, and the png Gibbs block."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from miniflims import load_config, run, update
from miniflims.blocks import BLOCKS
from miniflims.checkpoint import load_samples
from miniflims.config import ConfigError
from miniflims.data import build_response, make_mock
from miniflims.model import FieldModel
from miniflims.png import TEMPLATES, import_jaxpng, poisson_kernel

jp = import_jaxpng()
jax.config.update("jax_enable_x64", True)

FAST = {"n_t": 16}  # coarse Schwinger grid: tests compare like with like


def model_for(template="local", n=8, box=500.0, params=None, png_extra=None, **over):
    png = {"enabled": True, "template": template, "params": params or {}, **FAST, **(png_extra or {})}
    cfg = load_config(overrides={"grid": {"n_grid": n, "box_size": box}, "png": png, **over})
    return cfg, FieldModel(cfg, build_response(cfg))


def white(n=8, seed=0):
    return jnp.asarray(np.random.default_rng(seed).standard_normal((n,) * 3))


# ------------------------------------------------------------------- physics
def test_poisson_kernel_matches_jaxpng():
    from miniflims.cosmology import Cosmology
    from miniflims.transfer import eisenstein_hu

    c = Cosmology.from_config(load_config().cosmology)
    kt = np.logspace(-5, 2, 4000)
    t = np.asarray(eisenstein_hu(kt, c))
    cj = jp.cosmology.Cosmo(omega_m=c.omega_m, omega_l=1 - c.omega_m, omega_b=c.omega_b, hubble=c.h,
                            n_s=c.n_s, k_T=jnp.asarray(kt), T=jnp.asarray(t / t[0]))
    k = jnp.asarray([0.003, 0.02, 0.1, 0.4])
    np.testing.assert_allclose(poisson_kernel(k, c), jp.cosmology.poisson_M(cj, k), rtol=1e-4)
    assert float(poisson_kernel(jnp.asarray(0.0), c)) == 0.0


def test_potential_has_primordial_spectrum_shape():
    # delta_G / M has P_Phi ∝ k^(n_s - 4) wherever the transfer function cancels
    cfg, m = model_for(n=32, box=4000.0)
    from miniflims.power import linear_power

    k = jnp.asarray([2e-3, 4e-3])
    pphi = np.asarray(linear_power(k, m.cosmo) / poisson_kernel(k, m.cosmo) ** 2)
    slope = np.log(pphi[1] / pphi[0]) / np.log(2.0)
    assert slope == pytest.approx(m.cosmo.n_s - 4.0, abs=0.02)


def test_local_kernel_reproduces_the_squared_potential():
    cfg, kern = model_for(n=16, png_extra={"n_t": 32})
    _, exact = model_for(n=16, png_extra={"local_exact": True})
    dk = kern.gaussian_density_k(white(16))
    p = jnp.asarray([1.0])
    a, b = kern.png.ng_density_k(dk, p), exact.png.ng_density_k(dk, p)
    ra, rb = (jnp.fft.irfftn(x, s=(16,) * 3) for x in (a, b))
    assert float(jnp.linalg.norm(ra - rb) / jnp.linalg.norm(rb)) < 2e-3


def test_density_is_linear_in_fnl_and_gaussian_at_zero():
    cfg, m = model_for()
    _, g = model_for(png_extra={"enabled": False})
    x = white()
    d0 = m.linear_density(x, jnp.asarray([0.0]))
    np.testing.assert_allclose(d0, g.linear_density(x), atol=1e-12)
    d1, d2 = m.linear_density(x, jnp.asarray([300.0])), m.linear_density(x, jnp.asarray([600.0]))
    np.testing.assert_allclose(d2 - d0, 2 * (d1 - d0), atol=1e-12)
    assert float(jnp.std(d1 - d0)) > 1e-4
    with pytest.raises(ValueError):
        m.linear_density(x)  # png.enabled needs the parameters


@pytest.mark.parametrize("template", ["feature", "equilateral_feature"])
def test_traced_phase_matches_jaxpng_template(template):
    # phase sampled -> cos(phase) B_sin + sin(phase) B_cos, built in miniflims;
    # phase fixed -> jaxpng's own template with a Python-float phase
    omega, phase = 15.0, 0.7
    _, traced = model_for(template, params={"omega": {"init": omega}, "phase": {"init": phase, "sample": True}})
    _, fixed = model_for(template, params={"omega": {"init": omega}, "phase": {"init": phase}})
    assert traced.png.shape_sampled and not fixed.png.shape_sampled
    dk = traced.gaussian_density_k(white(seed=1))
    p = jnp.asarray([1.0, omega, phase])
    np.testing.assert_allclose(traced.png.ng_density_k(dk, p), fixed.png.ng_density_k(dk, p), rtol=1e-9, atol=1e-14)


def test_feature_template_matches_closed_form():
    # the cos expansion used for the traced phase is exact
    from miniflims.png import _feature_base, _modulate_cos

    n_s, omega = 0.96, 20.0
    terms = _modulate_cos(_feature_base(jp, n_s), omega)
    rng = np.random.default_rng(3)
    for _ in range(5):
        k1, k2, k3 = rng.uniform(0.02, 0.2, 3)
        s = float(sum(t[0](k1) * t[1](k2) * t[2](k3) for t in terms))
        P = [float(jp.grids.P_kernel(jnp.asarray(k), n_s)) for k in (k1, k2, k3)]
        ref = float(jp.bispectra.feature_bispectrum(*P, k1, k2, k3, omega, phase=np.pi / 2))
        assert s == pytest.approx(ref, rel=1e-9)


@pytest.mark.parametrize("template", ["local", "equilateral"])
def test_log_quadrature_agrees_with_jaxpng_default_grid(template):
    """The default (32 nodes in ln t) and jaxpng's own grid (361 nodes in t)
    give the same Phi_NG to within the latter's 4e-4 accuracy."""
    _, a = model_for(template, n=16, png_extra={"n_t": 32})
    _, b = model_for(template, n=16, png_extra={"quadrature": "jaxpng", "per_efold": 20.0})
    assert a.png.t_grid[0].shape == (32,) and b.png.t_grid is None
    dk = a.gaussian_density_k(white(16))
    x, y = (m.png.ng_density_k(dk, jnp.asarray([1.0])) for m in (a, b))
    assert float(jnp.linalg.norm(x - y) / jnp.linalg.norm(y)) < 1e-3
    with pytest.raises(ValueError):
        model_for(png_extra={"quadrature": "simpson"})


def test_templates_and_parameter_validation():
    assert set(TEMPLATES) == {"local", "equilateral", "orthogonal", "feature", "equilateral_feature"}
    _, m = model_for("feature")
    assert m.png.names == ["f_NL", "omega", "phase"] and m.png.sampled == (True, False, False)
    with pytest.raises(ValueError):
        model_for("local", params={"omega": {"init": 3.0}})  # no omega for local
    with pytest.raises(KeyError):
        model_for("squeezed")
    with pytest.raises(ValueError):
        model_for("equilateral", png_extra={"local_exact": True})
    with pytest.raises(ConfigError):
        load_config(overrides={"png": {"templat": "local"}})


# ----------------------------------------------------------- posterior pieces
@pytest.mark.parametrize("template", ["local", "equilateral"])
def test_posterior_gradient_with_png_matches_finite_differences(template):
    cfg, m = model_for(template, data={"mock": {"truth": {"f_NL": 800.0}}})
    data = make_mock(cfg)
    m.set_data(data.counts)
    x = jnp.asarray(data.truth["white_noise"])
    theta, png = jnp.asarray(data.truth["nuisance"]), jnp.asarray(data.truth["png"])
    f = lambda v: m.log_posterior(v, theta, None, png)
    v = jnp.asarray(np.random.default_rng(0).standard_normal(x.shape))
    eps = 1e-5
    fd = (f(x + eps * v) - f(x - eps * v)) / (2 * eps)
    assert float(jnp.sum(jax.grad(f)(x) * v)) == pytest.approx(float(fd), rel=1e-5)
    g = jax.grad(lambda p: m.log_posterior(x, theta, None, p))(png)
    fd = (m.log_posterior(x, theta, None, png + 1e-2) - m.log_posterior(x, theta, None, png - 1e-2)) / 2e-2
    assert float(g[0]) == pytest.approx(float(fd), rel=1e-5)


@pytest.mark.parametrize("sample_omega", [False, True])
def test_png_conditional_is_the_joint_up_to_a_constant(sample_omega):
    params = {"omega": {"init": 20.0, "sample": sample_omega}}
    cfg, m = model_for("feature", params=params, data={"mock": {"truth": {"f_NL": 500.0}}})
    data = make_mock(cfg)
    m.set_data(data.counts)
    x, theta = jnp.asarray(data.truth["white_noise"]), jnp.asarray(data.truth["nuisance"])
    hyper = {"likelihood_temperature": 0.7, "lambda_ps": 5.0}
    cur = jnp.asarray(data.truth["png"])
    fn, aux = m.png_conditional(x, theta, hyper, cur)
    diffs = []
    for f_nl, om in ((-300.0, 20.0), (0.0, 20.0), (900.0, 20.0)) + (((500.0, 25.0),) if sample_omega else ()):
        row = jnp.asarray([f_nl, om, 0.0])
        diffs.append(float(fn(row, aux[0])) - float(m.log_posterior(x, theta, hyper, row)))
    np.testing.assert_allclose(diffs, diffs[0], atol=1e-7)


def test_mock_truth_splits_png_and_nuisance_parameters():
    cfg, _ = model_for(data={"mock": {"truth": {"f_NL": 250.0, "alpha": 1.3, "nbar": 4.0}}})
    d = make_mock(cfg)
    assert d.truth["png"].tolist() == [250.0]
    assert d.truth["nuisance"][0].tolist() == [4.0, 1.3]
    cfg, _ = model_for(data={"mock": {"truth": {"omega": 3.0}}})
    with pytest.raises(ValueError):
        make_mock(cfg)


def test_png_block_requires_png_enabled():
    cfg = load_config(overrides={"grid": {"n_grid": 8}, "sampler": {"blocks": ["field", "png"]}})
    m = FieldModel(cfg, build_response(cfg))
    with pytest.raises(ValueError):
        BLOCKS.get("png")(m, cfg)


def test_gradient_memory_through_phi_ng_is_checkpointed():
    """jaxpng's remat keeps the reverse pass through the Schwinger scan at a few
    fields instead of one per t-node (here: N_t ~ 300)."""
    def temp_mb(remat):
        _, m = model_for(n=16, png_extra={"remat": remat, "quadrature": "jaxpng", "per_efold": 20.0})
        f = jax.grad(lambda v: jnp.sum(m.linear_density(v, jnp.asarray([100.0])) ** 2))
        return jax.jit(f).lower(white(16)).compile().memory_analysis().temp_size_in_bytes / 2**20

    lean, full = temp_mb(True), temp_mb(False)
    assert lean < 3.0 and full > 20 * lean


# ------------------------------------------------------------------ pipeline
def png_run_cfg(tmp_path, name="png", **over):
    base = {
        "grid": {"n_grid": 16, "box_size": 1000.0},
        "galaxies": {"nbar": {"init": 30.0}},
        "png": {"enabled": True, "template": "local", "local_exact": True,
                "params": {"f_NL": {"init": 0.0, "width": 500.0}}},
        "data": {"mock": {"seed": 1, "truth": {"nbar": 30.0, "alpha": 1.5, "f_NL": 3000.0}}},
        "sampler": {"n_chains": 2, "n_warmup": 150, "n_samples": 250, "thin": 10,
                    "blocks": ["field", "nuisance", "png"], "hmc": {"n_steps": 15, "n_steps_min": 5},
                    "checkpoint": {"every": 50}},
        "output": {"dir": str(tmp_path / name), "log_every": 0, "progress": False},
    }
    return update(load_config(), base) if not over else update(update(load_config(), base), over)


@pytest.mark.slow
def test_fnl_is_recovered_and_runs_resume(tmp_path, monkeypatch):
    from miniflims import pipeline

    cfg = png_run_cfg(tmp_path)
    res = run(cfg)
    s = res.summary["png"]["f_NL"]
    assert s["truth"] == 3000.0 and s["sampled"]
    assert abs(s["mean"] - 3000.0) < 3 * s["std"], s
    assert s["std"] < 1000.0 and s["rhat"] < 1.2  # informative and converged (sigma ~ 400 here)
    tr = res.gibbs.trace["png"][:, :, 0]
    assert res.gibbs.trace["png"].shape == (2, 400, 1)  # started at init = 0, see png_run_cfg
    out = Path(cfg.output.dir)
    with np.load(out / "chains.npz") as z:
        assert z["trace/png"].shape == (2, 400, 1) and z["truth/png"].tolist() == [3000.0]
    # posterior-mean density uses each sample's own f_NL: compare with a manual average
    sweeps, fields = load_samples(out)
    m = res.model
    manual = np.mean([np.asarray(m.final_density(jnp.asarray(fields[c, j], jnp.float64), jnp.asarray(tr[c, sw][None])))
                      for c in range(2) for j, sw in enumerate(sweeps)], axis=0)
    np.testing.assert_allclose(res.mean_density, manual, rtol=1e-5, atol=1e-6)

    # interrupted + resumed run with the png state is identical
    calls = {"n": 0}

    def stop(self):
        calls["n"] += 1
        return calls["n"] == 120

    cfg2 = png_run_cfg(tmp_path, "png_split")
    monkeypatch.setattr(pipeline._StopOnSignal, "__call__", stop)
    assert not run(cfg2).completed
    monkeypatch.setattr(pipeline._StopOnSignal, "__call__", lambda self: False)
    res2 = run(cfg2)
    np.testing.assert_array_equal(res2.gibbs.trace["png"], res.gibbs.trace["png"])
    # plots include the PNG panels
    from miniflims.plotting import plot_run

    assert [p.name for p in plot_run(out)] == ["traces.png", "field.png", "diagnostics.png"]


def test_fixed_png_without_block(tmp_path):
    cfg = png_run_cfg(tmp_path, sampler={"blocks": ["field", "nuisance"], "n_warmup": 4, "n_samples": 4},
                      png={"params": {"f_NL": {"init": 700.0}}})
    lines = []
    res = run(cfg, logger=lines.append)
    assert any("PNG parameters fixed at [700.0]" in l for l in lines)
    assert np.all(res.gibbs.trace["png"] == 700.0)
    assert "png" in res.summary and not res.summary["png"]["f_NL"].get("rhat")
