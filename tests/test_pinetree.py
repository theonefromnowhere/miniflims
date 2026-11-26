"""PineTree population model (galaxies.population: pinetree): agreement with the
reference implementation (npe), gauge, whitening, gradients, mock sampler and
pipeline integration."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from miniflims.config import PineTreeConfig, load_config
from miniflims.data import make_mock
from miniflims.model import FieldModel
from miniflims.pinetree import NAMES, PineTreeModel, counts_grid, whitening_matrix

REF = Path(__file__).parent / "data" / "pinetree_reference.npz"
PARAMS = Path(__file__).parents[1] / "configs" / "pinetree" / "quijote_fid_fof_128_m3.3e13_it100000.json"


@pytest.fixture(scope="module")
def ref():
    with np.load(REF) as f:
        return {k: f[k] for k in f.files}


def write_params(tmp_path, theta, threshold):
    p = tmp_path / "pt.json"
    p.write_text(json.dumps({"format": "pinetree-monopole-v1", "theta": dict(zip(NAMES, map(float, theta))),
                             "mass_threshold": float(threshold)}))
    return str(p)


@pytest.fixture
def pt(tmp_path, ref):
    """Raw-parameter model on the reference field, no density floor (as npe)."""
    path = write_params(tmp_path, ref["theta"], ref["mass_threshold"])
    return PineTreeModel(PineTreeConfig(params_path=path), floor=0.0)


def test_reference_parameters_match_the_shipped_export(ref):
    d = json.loads(PARAMS.read_text())
    assert [str(n) for n in ref["names"]] == list(NAMES)
    np.testing.assert_allclose([d["theta"][n] for n in NAMES], ref["theta"], rtol=0, atol=0)


def test_forward_pass_and_likelihood_match_npe(pt, ref):
    rho = jnp.asarray(1.0 + ref["delta"])
    theta = jnp.asarray(ref["theta"])
    log_amp, mu, sig = pt.voxel_params(rho, theta)
    np.testing.assert_allclose(np.exp(log_amp), ref["amplitude"], rtol=1e-11)
    np.testing.assert_allclose(mu, ref["mean"], rtol=1e-12)
    np.testing.assert_allclose(sig, ref["sigma"], rtol=1e-11)
    np.testing.assert_allclose(pt.rate(rho, theta), ref["rate"], rtol=1e-9)
    ll = pt.log_likelihood(rho, theta, jnp.asarray(ref["halo_voxel"]), jnp.log(jnp.asarray(ref["halo_mass"])),
                           jnp.ones(rho.size))
    np.testing.assert_allclose(float(ll), float(ref["log_likelihood"]), rtol=1e-11)


def test_specs_gauge_and_softmax_invariance(pt, ref):
    names = [s.name for s in pt.specs]
    assert tuple(names) == NAMES
    assert [n for s, n in zip(pt.specs, names) if not s.sample] == ["amp_w0", "amp_b0"]
    rho = jnp.asarray(1.0 + ref["delta"])
    args = (jnp.asarray(ref["halo_voxel"]), jnp.log(jnp.asarray(ref["halo_mass"])), jnp.ones(rho.size))
    t = np.array(ref["theta"])
    shifted = t.copy()
    for n in ("amp_b0", "amp_b1"):
        shifted[NAMES.index(n)] += 0.7
    for n in ("amp_w0", "amp_w1"):
        shifted[NAMES.index(n)] -= 0.3
    np.testing.assert_allclose(float(pt.log_likelihood(rho, jnp.asarray(shifted), *args)),
                               float(pt.log_likelihood(rho, jnp.asarray(t), *args)), rtol=1e-12)


def test_whitened_coordinates(tmp_path, ref):
    rng = np.random.default_rng(3)
    sampled = np.array([i for i, n in enumerate(NAMES) if n not in ("amp_w0", "amp_b0")])
    a = rng.standard_normal((sampled.size, sampled.size))
    h = a @ a.T + 0.1 * np.eye(sampled.size)
    L, w = whitening_matrix(h)
    np.testing.assert_allclose(L.T @ h @ L, np.eye(sampled.size), atol=1e-9)
    wpath = tmp_path / "w.npz"
    np.savez(wpath, theta_ref=ref["theta"], L=L, sampled_raw=sampled, names=np.array(NAMES))
    path = write_params(tmp_path, ref["theta"], ref["mass_threshold"])
    m = PineTreeModel(PineTreeConfig(params_path=path, whitening=str(wpath)), floor=0.0)
    assert [s.name for s in m.specs] == [f"z{i}" for i in range(sampled.size)]
    z = rng.standard_normal(sampled.size)
    t = np.asarray(m.theta(jnp.asarray(z)))
    np.testing.assert_allclose(t[sampled], ref["theta"][sampled] + L @ z, rtol=1e-12)
    np.testing.assert_array_equal(np.delete(t, sampled), np.delete(ref["theta"], sampled))
    np.testing.assert_allclose(m.theta_np(z[None]), t[None], rtol=1e-12)


def test_gradients_match_finite_differences(pt, ref):
    rho0 = jnp.asarray(1.0 + ref["delta"])
    vox, lnm = jnp.asarray(ref["halo_voxel"]), jnp.log(jnp.asarray(ref["halo_mass"]))
    resp = jnp.ones(rho0.size)
    theta0 = jnp.asarray(ref["theta"])
    f_t = lambda t: pt.log_likelihood(rho0, t, vox, lnm, resp)
    g = jax.grad(f_t)(theta0)
    for i in (0, 1, 4, 6, 12, 13, 17):
        h = 1e-6 * max(1.0, abs(float(theta0[i])))
        e = jnp.zeros_like(theta0).at[i].set(h)
        fd = (f_t(theta0 + e) - f_t(theta0 - e)) / (2 * h)
        np.testing.assert_allclose(float(g[i]), float(fd), rtol=1e-5, atol=1e-6)
    f_r = lambda r: pt.log_likelihood(r, theta0, vox, lnm, resp)
    g = jax.grad(f_r)(rho0)
    for idx in [(0, 0, 0), (3, 5, 7), (11, 2, 9)]:
        e = jnp.zeros_like(rho0).at[idx].set(1e-6)
        fd = (f_r(rho0 + e) - f_r(rho0 - e)) / 2e-6
        np.testing.assert_allclose(float(g[idx]), float(fd), rtol=1e-5, atol=1e-6)


def test_mock_sampler_counts_and_masses(pt, ref):
    rho = jnp.asarray(np.tile(1.0 + ref["delta"], (2, 2, 2)))  # 24^3, periodic tiling
    theta = jnp.asarray(ref["theta"])
    resp = np.ones(rho.size)
    rng = np.random.default_rng(11)
    lam = float(jnp.sum(pt.rate(rho, theta)))
    n_tot, lnm_all = [], []
    for _ in range(20):
        vox, lnm = pt.sample(rng, rho, theta, resp)
        n_tot.append(vox.size)
        lnm_all.append(lnm)
    assert abs(np.mean(n_tot) - lam) < 4 * np.sqrt(lam / len(n_tot))
    lnm = np.concatenate(lnm_all)
    assert lnm.min() >= pt.ln_threshold
    edges = np.exp(np.array([pt.ln_threshold, np.log(5e13), np.log(1e14), np.log(3e14), 40.0]))
    expected = np.array([float(jnp.sum(pt.rate_in_bin(rho, theta, lo, hi))) for lo, hi in zip(edges[:-1], edges[1:])])
    observed = np.histogram(lnm, np.log(edges))[0] / len(lnm_all)
    np.testing.assert_allclose(expected.sum(), lam, rtol=1e-6)
    chi2 = np.sum((observed - expected) ** 2 / (expected / len(lnm_all)))
    assert stats.chi2.sf(chi2, len(expected)) > 1e-3
    # thinning by the response
    vox, _ = pt.sample(np.random.default_rng(1), rho, theta, np.where(np.arange(rho.size) % 2, 1.0, 0.0))
    assert np.all(vox % 2 == 1)


def pinetree_cfg(**extra):
    over = {
        "grid": {"n_grid": 16, "n_mesh": 16, "box_size": 125.0},  # 7.8125 Mpc/h voxels
        "forward": {"model": "zeldovich"},
        "galaxies": {"population": "pinetree", "pinetree": {"params_path": str(PARAMS)}},
        "data": {"mock": {"seed": 4}},
        "sampler": {"n_chains": 1, "n_warmup": 2, "n_samples": 2, "thin": 1, "blocks": ["nuisance"],
                    "init": {"field": "truth"}},
        "output": {"log_every": 0, "progress": False},
    }
    for k, v in extra.items():
        over[k] = {**over.get(k, {}), **v}
    return load_config(overrides=over)


def test_field_model_integration():
    cfg = pinetree_cfg()
    data = make_mock(cfg)
    assert data.halos is not None and data.halos["voxel"].size > 50
    np.testing.assert_array_equal(data.counts[0], counts_grid(data.halos["voxel"], 16))
    model = FieldModel(cfg, data.response, data.counts, halos=data.halos)
    assert model.param_names == list(NAMES)
    x, theta = jnp.asarray(data.truth["white_noise"]), jnp.asarray(data.truth["nuisance"])
    ll = model.log_likelihood(x, theta)
    direct = model.pinetree.log_likelihood(jnp.asarray(data.truth["rho"]), theta[0], model.halos["voxel"],
                                           model.halos["ln_mass"], model.response[0].ravel())
    np.testing.assert_allclose(float(ll), float(direct), rtol=1e-10)
    g = jax.grad(lambda v: model.log_posterior(v, theta))(x)
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.abs(g).max()) > 0
    lam = model.expected_counts(theta[0], jnp.asarray(data.truth["rho"]), model.response[0])
    assert lam.shape == (16,) * 3
    with pytest.raises(ValueError):
        FieldModel(cfg, data.response, data.counts)  # catalogue missing


def test_pipeline_theta_only_run(tmp_path):
    from miniflims.pipeline import run

    cfg = pinetree_cfg(output={"dir": str(tmp_path / "run")})
    res = run(cfg, logger=lambda *a: None)
    assert res.completed
    nuis = res.summary["nuisance"]
    assert set(nuis) == {f"cat0/{n}" for n in NAMES if n not in ("amp_w0", "amp_b0")}
    assert all(np.isfinite(v["mean"]) for v in nuis.values())
    with np.load(tmp_path / "run" / "chains.npz") as f:
        assert "halo_voxel" in f.files and f["halo_voxel"].size == int(f["counts"].sum())
    # the field never moved from the truth
    from miniflims.checkpoint import load_checkpoint
    ckpt, _ = load_checkpoint(tmp_path / "run")
    np.testing.assert_array_equal(np.asarray(ckpt.state["field"][0]), np.asarray(res.data.truth["white_noise"]))


def test_default_population_is_unchanged(small_cfg):
    assert small_cfg.galaxies.population == "counts"
    assert small_cfg.sampler.init.field == "random"
    data = make_mock(small_cfg)
    assert data.halos is None
    assert FieldModel(small_cfg, data.response, data.counts).param_names == ["nbar", "alpha"]
