"""End-to-end tests of the full pipeline (marked slow: ~2.5 min in total)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from miniflims import load_config, run
from miniflims.blocks import BLOCKS
from miniflims.checkpoint import load_samples
from miniflims.cli import main

ROOT = Path(__file__).resolve().parents[1]
SMALL = {"grid": {"n_grid": 16, "box_size": 400.0}, "output": {"log_every": 0, "n_k_bins": 6}}


@pytest.mark.parametrize("path", sorted((ROOT / "configs").glob("*.yaml")), ids=lambda p: p.name)
def test_shipped_configs_are_valid(path):
    cfg = load_config(path)
    for name in cfg.sampler.blocks:
        assert name in BLOCKS


@pytest.mark.slow
def test_posterior_matches_wiener_filter():
    """Linear forward model, linear bias and Gaussian noise make the posterior
    of the white noise Gaussian with known mean (the Wiener filter) and
    covariance, diagonal in Fourier space:

        d - nbar = A x + n,   A_k = nbar b D sqrt(P_k N^3 / V)
        <x_k | d> = A_k d_k / (A_k^2 + sigma^2),   Var = sigma^2 / (A_k^2 + sigma^2)
    """
    nbar, b, sigma, n = 10.0, 1.5, 12.0, 16
    cfg = load_config(overrides={
        **SMALL,
        "forward": {"model": "linear"},
        "galaxies": {
            "nbar": {"init": nbar, "sample": False},
            "bias": {"model": "linear", "normalize": False, "params": {"b": {"init": b, "sample": False}}},
            "likelihood": {"model": "gaussian", "params": {"sigma": {"init": sigma, "sample": False}}},
        },
        "sampler": {
            "n_chains": 4, "n_warmup": 300, "n_samples": 600, "thin": 2, "blocks": ["field"],
            "hmc": {"step_size": 0.05, "n_steps": 30, "n_steps_min": 10, "target_accept": 0.8},
        },
    })
    res = run(cfg)
    model = res.model
    amp = nbar * b * model.forward.growth * np.asarray(model.amplitude)
    dk = np.fft.rfftn(res.data.counts[0] - nbar)
    wiener = np.fft.irfftn(amp * dk / (amp**2 + sigma**2), s=(n,) * 3, axes=(0, 1, 2))
    post_var = np.sum(model.grid.rfft_weights() * sigma**2 / (amp**2 + sigma**2))
    # signal-to-noise per mode straddles one: neither the prior nor the
    # likelihood alone determines the posterior
    snr = (amp**2 / sigma**2)[amp > 0]
    assert snr.min() < 0.5 and snr.max() > 2.0

    fields = res.fields.astype(float)
    assert fields.shape == (4, 300, n, n, n)
    mean = fields.mean(axis=(0, 1))
    assert np.corrcoef(mean.ravel(), wiener.ravel())[0, 1] > 0.995
    assert (mean - wiener).std() < 0.08 * wiener.std()
    # total posterior second moment: |mean|^2 + trace of the covariance
    norm2 = (fields**2).sum(axis=(2, 3, 4))
    assert norm2.mean() == pytest.approx((wiener**2).sum() + post_var, rel=0.005)
    # per-voxel posterior variance, averaged
    var = fields.reshape(-1, n**3).var(axis=0).mean()
    assert var == pytest.approx(post_var / n**3, rel=0.03)
    assert 0.7 < res.summary["hmc"]["mean_accept_prob"] < 0.9
    assert "nuisance/n_evals" not in res.gibbs.trace  # the block was not run


@pytest.mark.slow
def test_poisson_zeldovich_end_to_end(tmp_path):
    cfg = load_config(overrides={
        **SMALL,
        "galaxies": {"nbar": {"init": 2.0}},
        "data": {"mock": {"truth": {"nbar": 4.0, "alpha": 1.5}}},
        "sampler": {
            "n_chains": 2, "n_warmup": 250, "n_samples": 300, "thin": 5,
            "hmc": {"n_steps": 30, "n_steps_min": 10}, "init": {"nuisance_jitter": 0.02},
        },
        "output": {"dir": str(tmp_path / "run"), "log_every": 100, "n_k_bins": 6},
    })
    lines = []
    res = run(cfg, logger=lines.append)
    s = res.summary

    # sampler health
    assert 0.5 < s["hmc"]["accept_rate"] < 0.95
    assert np.all(np.isfinite(res.gibbs.trace["log_likelihood"]))
    ll = res.gibbs.trace["log_likelihood"]
    assert ll[:, -100:].mean() > ll[:, 0].mean() + 1000  # burned in from the low-amplitude start
    assert s["nuisance"]["cat0/nbar"]["rhat"] < 1.1

    # nuisance parameters recovered (starts were not at the truth)
    start = res.gibbs.trace["nuisance"][:, 0, 0]
    assert np.all(np.abs(start[:, 0] - 4.0) > 0.5)
    assert s["nuisance"]["cat0/nbar"]["mean"] == pytest.approx(4.0, abs=0.2)
    assert s["nuisance"]["cat0/alpha"]["mean"] == pytest.approx(1.5, abs=0.3)
    assert s["nuisance"]["cat0/alpha"]["truth"] == 1.5

    # field recovered on large scales, less so on small scales
    r_ic = np.array(s["ic_cross_correlation"]["r"])
    r_rho = np.array(s["density_cross_correlation"]["r"])
    assert r_ic[0] > 0.8 and r_rho[0] > 0.8 and r_ic[-1] < r_ic[0]
    # the inferred initial conditions look like unit white noise (Manticore App. C)
    mom = s["white_noise_moments"]
    assert mom["variance"] == pytest.approx(1.0, abs=0.1)
    assert abs(mom["mean"]) < 0.05 and abs(mom["skewness"]) < 0.15 and abs(mom["kurtosis"]) < 0.3
    assert res.mean_density.mean() == pytest.approx(1.0)

    # logging and outputs
    assert lines[0].startswith("grid 16^3") and any("warmup" in l for l in lines) and any("sample" in l for l in lines)
    out = tmp_path / "run"
    sweeps, fields = load_samples(out)
    assert fields.shape == (2, 60, 16, 16, 16) and fields.dtype == np.float32
    np.testing.assert_array_equal(sweeps, 250 + 5 * np.arange(60))
    assert res.fields is None and res.run_dir == out
    assert load_config(out / "config.yaml") == cfg
    assert json.loads((out / "summary.json").read_text())["hmc"] == s["hmc"]
    with np.load(out / "chains.npz") as f:
        assert "fields" not in f.files  # streamed to samples/ instead
        assert f["trace/nuisance"].shape == (2, 550, 1, 2)
        assert f["param_names"].tolist() == ["nbar", "alpha"]
        np.testing.assert_array_equal(f["counts"], res.data.counts)
        np.testing.assert_array_equal(f["truth/white_noise"], res.data.truth["white_noise"])
        assert f["mean_density"].shape == (16, 16, 16)


@pytest.mark.slow
def test_manticore_like_config_end_to_end():
    """Two sub-catalogues with radial selections, STDP bias, generalized
    Poisson noise, radial RSD, likelihood tempering and the power-spectrum
    prior, all at once."""
    cfg = load_config(ROOT / "configs" / "manticore_like.yaml", {
        **SMALL,
        "sampler": {"n_warmup": 200, "n_samples": 200,
                    "schedules": {"likelihood_temperature": {"n_steps": 80}}},
        "output": {"dir": None, "log_every": 0, "n_k_bins": 6},
    })
    res = run(cfg)
    s, tr = res.summary, res.gibbs.trace
    assert tr["nuisance"].shape == (2, 400, 2, 8)
    assert np.all(np.isfinite(tr["log_likelihood"])) and np.all(np.isfinite(tr["nuisance"]))
    assert 0.5 < s["hmc"]["accept_rate"] < 0.95

    names = res.model.param_names
    lower, upper = np.asarray(res.model.lower), np.asarray(res.model.upper)
    assert np.all(tr["nuisance"] >= lower) and np.all(tr["nuisance"] <= upper)
    assert np.all(tr["nuisance"][..., names.index("b_beta")] == 0.0)  # sample: false
    assert "cat0/b_beta" not in s["nuisance"] and "cat1/b_alpha" in s["nuisance"]

    # well-constrained parameters are recovered per catalogue
    for cat, nbar, b_alpha in ((0, 4.0, 0.2), (1, 2.0, 0.3)):
        assert s["nuisance"][f"cat{cat}/nbar"]["mean"] == pytest.approx(nbar, rel=0.15)
        assert s["nuisance"][f"cat{cat}/b_alpha"]["mean"] == pytest.approx(b_alpha, abs=0.07)
    # ... and the two catalogues are told apart
    assert s["nuisance"]["cat0/nbar"]["mean"] > s["nuisance"]["cat1/nbar"]["mean"] + 1.0

    # the field is recovered with the right sign (no mirror mode)
    assert s["density_cross_correlation"]["r"][0] > 0.7
    assert s["ic_cross_correlation"]["r"][0] > 0.7
    # power-spectrum prior keeps the white-noise spectrum flat
    ratio = np.asarray(res.model.white_noise_power_ratio(res.gibbs.state["field"][0]))
    assert np.all(np.abs(ratio - 1) < 0.1)


# --------------------------------------------------------------------------- CLI
def test_cli_show_config_and_components(capsys):
    assert main(["show-config", "-o", "grid.n_grid=8", "-o", "sampler.blocks=[field]"]) == 0
    shown = yaml.safe_load(capsys.readouterr().out)
    assert shown["grid"]["n_grid"] == 8 and shown["sampler"]["blocks"] == ["field"]
    assert main(["components"]) == 0
    out = capsys.readouterr().out
    for word in ("eisenstein_hu", "zeldovich", "stdp", "generalized_poisson", "leapfrog", "nuisance"):
        assert word in out


def test_cli_reports_config_errors(capsys):
    assert main(["run", "-o", "grid.n_gird=8"]) == 2
    assert "n_gird" in capsys.readouterr().err


@pytest.mark.slow
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_cli_run_in_subprocess(tmp_path, precision):
    # separate process: precision is a global JAX setting
    out = tmp_path / precision
    cmd = [
        sys.executable, "-m", "miniflims", "run", "-c", str(ROOT / "configs" / "default.yaml"),
        "-o", f"precision={precision}", "-o", "grid.n_grid=8", "-o", "grid.box_size=200",
        "-o", "sampler.n_warmup=30", "-o", "sampler.n_samples=20", "-o", "sampler.thin=5",
        "-o", f"output.dir={out}", "-o", "output.log_every=0",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    summary = json.loads((out / "summary.json").read_text())
    assert 0.2 < summary["hmc"]["accept_rate"] <= 1.0
    assert np.isfinite(summary["log_likelihood"]["mean"])
    with np.load(out / "chains.npz") as f:
        assert "fields" not in f.files
        assert f["trace/log_likelihood"].dtype == np.dtype(precision)
        assert np.all(np.isfinite(f["trace/nuisance"]))
    assert load_samples(out)[1].shape == (2, 4, 8, 8, 8)
