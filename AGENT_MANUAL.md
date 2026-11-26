# miniflims — manual for coding agents

Operational guide for an AI agent (Claude Code or similar) working in this
repository. This file covers how to run, monitor, analyse and extend things
without breaking anything. [README.md](README.md) covers the physics and
the full feature list. Read the README sections named below when you need
them.

State as of 2026-11-25. Check anything time-dependent (running jobs, run
states) yourself before relying on it.

---

## 1. Ground rules

1. **Python environment on the Mac: always `.venv-run`, never `.venv`.**
   - `.venv` contains jax-mps (an Apple-GPU experiment the user installed on
     purpose). It pulled in JAX 0.11.2, and under 0.11.2 PNG inference
     deadlocks silently: 0 % CPU, the main thread blocks forever in
     `np.asarray`, no error.
   - `.venv-run` pins JAX 0.12.1 (`cluster/constraints.txt`) and works.
   - Do not modify, upgrade or uninstall anything in `.venv`.
   - README examples say `.venv/bin/...`; substitute `.venv-run/bin/...`.
2. **No git.** The directory is not a git repository. Do not `git init` or
   commit unless the user asks.
3. **Cluster access needs the user.**
   - Login to CC-IN2P3 is password only. This machine has no key and no
     Kerberos ticket, so you cannot log in on your own.
   - Never ask for the password. Give the user terminal commands (section 8),
     or reuse a ControlMaster socket the user has opened (`-S ~/.ssh/cm-cca`).
4. **Never destroy run outputs.**
   - `miniflims run --fresh` deletes the checkpoint and samples of that run
     directory.
   - Check `miniflims status <dir>` and running processes before starting
     anything that writes to an existing run directory.
5. **Stop means stop.** If the user says stop or wait, end the turn without
   finishing the current step.
6. **Memory-efficient gradients are a requirement.**
   - When you change or port a forward model, keep the custom VJPs, `remat`
     and checkpointing (tCOLA `forward.cola.low_memory`, jaxpng `png.remat`).
   - Measure peak memory before and after the change.
7. **Privacy.** Do not put the user's personal details (email, name) in
   requests to external services.
8. **Long jobs run in the background.**
   - Launch runs detached, logging to a file.
   - Confirm the first new checkpoint appears (about 6 min locally), then
     report and wait for completion notifications. Do not poll in a tight
     loop.

## 2. What the code is

miniflims is a JAX field-level inference pipeline modelled on BORG/Manticore:

- **Forward model:** white noise x → Eisenstein–Hu linear field →
  Zel'dovich/tCOLA/linear forward model → galaxy bias → Poisson,
  generalized-Poisson or Gaussian likelihood on gridded counts.
- **Sampler:** block Gibbs. HMC on the white noise, then slice sampling of
  the bias and nuisance parameters, then (optionally) slice sampling of the
  PNG parameters.
- **PNG:** primordial non-Gaussianity comes from **jaxpng** (`./jaxpng`,
  installed editable in compat mode): Φ = Φ_G + f_NL Φ_NG[Φ_G], δ = M(k) Φ.
  Templates: `local`, `equilateral`, `orthogonal`, `feature`
  (6 f_NL (P1P2P3)^(2/3) sin(ω Σk + φ)), `equilateral_feature`.
- **Non-local bias** (new, see 4.4): a scale-dependent PNG bias term
  b_φ f_NL ψ_α with ψ_α = (k/k_*)^α Φ_G, and a Laplacian term. Their
  coefficients are sampled with the other bias parameters.
- **White-noise priors** (optional): Manticore's power-spectrum prior
  (`prior.lambda_ps`) and a Gaussianity prior on the binned white-noise
  bispectrum (`prior.bispectrum.lambda_bs`).

README sections: *Configuration*, *Primordial non-Gaussianity (jaxpng)*,
*tCOLA forward model*, *Extending*, *What is taken from Manticore / BORG*,
*Conventions* (units Mpc/h and h/Mpc, unnormalised `rfftn`), *Known pitfall*
(the `stdp` mirror mode).

### Repository map

| Path | Contents |
|---|---|
| `src/miniflims/config.py` | Typed config tree: **the schema**. Read it before adding config keys. |
| `src/miniflims/model.py` | `FieldModel`: forward model, likelihood, priors, PNG and non-local hooks, conditionals for each block |
| `src/miniflims/pipeline.py` | `run(cfg, fresh=False)`: data, model, Gibbs loop, checkpoints, `run.log`, summary |
| `src/miniflims/blocks.py`, `samplers/` | Gibbs blocks (`field`, `nuisance`, `png`); HMC (leapfrog, mclachlan2, yoshida4) and slice samplers |
| `src/miniflims/forward.py`, `cola.py` | Zel'dovich, tCOLA (custom VJP), linear |
| `src/miniflims/png.py` | PNG templates and the jaxpng bridge (ln-t quadrature) |
| `src/miniflims/nonlocal_bias.py` | `NONLOCAL_TERMS` registry (`png` → `b_phi`, `laplacian` → `b_lap`) |
| `src/miniflims/bias.py`, `likelihood.py`, `data.py`, `transfer.py` | Component registries; mock generation |
| `src/miniflims/hod.py` | HOD mocks. Zheng+09 SDSS LRG presets (Appendix B, Eqs. B1/B3/B4, verified against the paper: σ₈-dependent, mass scales ×Ω_m/0.24, M_200m, Msun/h). Halo readers, `populate` (NFW satellites, v_rms dispersion), RSD, NGP counts, Tinker+08 completeness check. Tests in `tests/test_hod.py`. |
| `src/miniflims/checkpoint.py` | Checkpoint I/O, resume rules, sample files |
| `src/miniflims/watch.py`, `plotting.py`, `diagnostics.py` | `miniflims watch`, `miniflims plot`, R̂/ESS/moments |
| `configs/` | Shipped configs (table in 4.2) |
| `scripts/` | Analysis and utilities (section 6) |
| `cluster/` | CC-IN2P3 setup and Slurm scripts (section 8) |
| `tests/` | pytest suite. `-m "not slow"` is the fast subset. |
| `jaxpng/` | The jaxpng package: own project, own tests |
| `runs/` | Local run directories (results; do not delete) |
| `meshes/`, `figures/`, `comparisons/` | Generated simulation meshes and figures |

## 3. Environments

### Local (macOS, Apple silicon, 8 cores)

```bash
cd /Users/strusov/miniflims
.venv-run/bin/miniflims components          # sanity check
JAX_PLATFORMS=cpu .venv-run/bin/python -m pytest -m "not slow" -q -p no:cacheprovider
```

- **Tests:** the fast suite is 207 tests. It took 12 min alongside a
  running inference, a few minutes on an idle machine. There are 8 slow
  tests on top.
- **Speed of a 64³ Zel'dovich + feature-PNG run** (2 chains): about 190–210
  Gibbs sweeps per hour (17–19 s/sweep), using about 5 cores.
- **Only one heavy job at a time.** A second one roughly halves the speed of
  both.
- **macOS shell:** there is no `timeout`, and `sed -i` needs `''`.

### Cluster (CC-IN2P3)

The layout is in section 8. The environment there is built from the same
`cluster/constraints.txt`.

## 4. Configuration

### 4.1 Mechanics

- Settings are layered: defaults ← YAML ← `-o dotted.key=value` overrides.
  Unknown keys and wrong types are errors.
- `miniflims show-config -c X.yaml [-o ...]` prints the fully resolved tree.
  Use it to diff two runs:
  `diff <(miniflims show-config -c a.yaml) <(miniflims show-config -c b.yaml)`.
- Each parameter (nuisance, bias, PNG) takes
  `{init, lower, upper, width, sample, prior, prior_mean, prior_std}`:
  - `sample: false` fixes the parameter at `init`.
  - `prior` is `uniform` or `normal`, always within the bounds.
- PNG parameters are only sampled when `png` is in `sampler.blocks`.
  `omega` and `phase` also need `sample: true`.

### 4.2 Shipped configs

| Config | Purpose |
|---|---|
| `default.yaml` | 32³ quick demo, about 2 min |
| `manticore_like.yaml` | Sub-catalogues, `stdp` bias, generalized Poisson |
| `tcola.yaml` | tCOLA gravity |
| `png_local.yaml` | Local f_NL sampled in its own Gibbs block |
| `png_feature_w20.yaml` | Feature PNG, ω=20, **with** λ_ps=100 and λ_bs=100 white-noise priors. Output `runs/png_feature_w20_priors`. That run hung under JAX 0.11.2 and never produced results. Note: this file is **not** the config of `runs/png_feature_w20`. |
| `png_feature_w20_10k.yaml` | Exact settings of the original `runs/png_feature_w20` (no priors), 10 000 sweeps, for the cluster |
| `png_null_w20.yaml` | Null test: same mock with truth f_NL=0, same feature model and sampler |
| `nopng_w20_control.yaml` | Same mock without PNG in the data or the model |
| `pinetree_sc_theta.yaml` | PineTree self-consistency, step 1: PineTree parameters with the field fixed at the truth (section 4.5) |
| `pinetree_sc_field.yaml` | Step 2: initial conditions with the PineTree parameters fixed at the truth |
| `pinetree_sc_joint.yaml` | Step 3: both, from a random low-amplitude start |

The resolved config of any run is in `<run_dir>/config.yaml`. That copy is
authoritative for what the run actually used.

### 4.3 Key knobs

- **`sampler`:**
  - `n_chains`, `n_warmup`, `n_samples`, `thin` (field samples kept every
    `thin` sweeps after warm-up).
  - `blocks` (Gibbs order).
  - `hmc.{step_size, n_steps, n_steps_min, target_accept, integrator}`.
  - `checkpoint.every` (20).
  - `init.field_scale`: **default 0.1**. This is the low-amplitude start
    behind the long burn-in (section 7).
  - `seed`.
- **`sampler.gates.<block>.min_field_variance`:** the block acts only once that chain's ⟨x²⟩ reaches the value. `gates: {png: {min_field_variance: 0.95}}` holds f_NL and ω at their init while the field grows from its low-amplitude start. Without it, the 64³ ω = 20, f_NL = +500 run locked into the ω ≈ 12 alias; with it, the truth was recovered (`configs/png_feature_w20_gated.yaml`). Use a warm-up long enough that the gate opens inside it (1000 sweeps at 64³).
- **`forward.model: 2lpt`:** second-order LPT, with the same lattice and CIC as `zeldovich`, f₂ for RSD, and a checkpointed second-order source (`tests/test_2lpt.py`).
- **`png`:**
  - `template`, `params.{f_NL, omega, phase}`.
  - `quadrature: log`, `n_t: 32`: the ln-t trapezoid, about 3e-5 accuracy.
  - `remat: true`.
  - `local_exact`: cheapest option for local f_NL.
- **`prior`:**
  - `lambda_ps`, `ps_prior_bins`.
  - `bispectrum.{lambda_bs, n_bins, k_max, n_mc, seed}`. 0 means off.
- **`data.mock`:**
  - `seed` sets the white noise.
  - `truth` gives the nuisance and PNG values (`f_NL`, `omega`, `phase`).
- **`output`:** `dir`, `log_every`, `progress_interval` (seconds; use 60 for
  log files), `save_fields`.

### 4.4 Non-local bias (implemented, not yet covered by tests)

```yaml
galaxies:
  bias:
    model: power_law
    params:
      alpha: {init: 1.0, lower: 0.0, upper: 4.0}
      b_phi: {init: 0.0, lower: -100.0, upper: 100.0, width: 2.0}   # coefficient of the png term
    non_local:
      terms: [png]          # and/or laplacian (coefficient b_lap)
      form: exp             # B(rho)·exp(Σ b_i O_i); "linear": max(B + Σ b_i O_i, floor)
      k_star: 0.1           # h/Mpc
      png_alpha: null       # null -> template's squeezed-limit alpha
```

- **The terms:**
  - The `png` term is f_NL ψ_α, with ψ_α = (k/k_*)^α Φ_G.
  - In the density this gives Δb(k) = b_φ f_NL (k/k_*)^α / M(k).
  - Squeezed-limit α by template: local 0, orthogonal 1, feature 1,
    equilateral 2, equilateral_feature 2.
  - Universal-mass-function expectation: b_φ = 2 δ_c (b1 − 1).
- **Sampling:** the coefficients are sampled in the `nuisance` block.
- **Status:** smoke-tested only. `tests/test_nonlocal_bias.py` still needs
  to be written.

### 4.5 PineTree population model (`galaxies.population: pinetree`)

```yaml
galaxies:
  population: pinetree            # default "counts" = everything above
  pinetree:
    params_path: configs/pinetree/quijote_fid_fof_128_m3.3e13_it100000.json
    whitening: configs/pinetree/whitening_sc64_seed1.npz   # null = raw parameters
```

- **What:** PineTree (Ding et al. 2024), a mass-resolved halo catalogue
  conditioned on the evolved density: 3³ monopole kernel on ln ρ, softplus,
  2-log-normal conditional mass function, Poisson process above
  `mass_threshold`. Pure JAX in `src/miniflims/pinetree.py`. The likelihood
  is PineTree's training loss (marked Poisson). The data are the halo
  catalogue itself (`halo_voxel`, `halo_mass` in a data file, or a mock);
  `counts` is only its gridding.
- **Parameters:** 19, exported from a trained PineTree by PineTree's
  `scripts/export_pinetree_params.py`. They replace nbar, bias and likelihood
  parameters. `amp_w0` and `amp_b0` are fixed (softmax gauge). The rest are
  strongly correlated (Hessian condition number ~1e6), so sample them whitened:
  `scripts/pinetree_whitening.py` writes θ = θ_ref + L z from the Hessian at
  the truth, and the nuisance names become `z0…z16`.
- **Mesh:** the kernel is in voxel units. Use the training voxel (7.8125 Mpc/h:
  500 Mpc/h on 64³, or 1000 on 128³). Use `n_grid = 2 n_mesh` so that no
  voxel is empty.
- **Known limit:** a few parameter directions are not constrained by the data.
  Component 0 sits below the mass threshold, so only its tail is observed; on
  the 500 Mpc/h mock three Hessian eigenvalues are ≤ 0. Judge recovery on the
  predictions (`scripts/pinetree_posterior.py`) as well as on the raw parameters.
- **Tests:** `tests/test_pinetree.py`. The reference fixture
  (`tests/data/pinetree_reference.npz`) is the npe forward pass and
  likelihood; it agrees to 1e-11.
- **`sampler.init.field: truth`:** every chain starts at the mock's white
  noise. Without a `field` block the field stays there (θ-only tests).

## 5. Running and monitoring

### 5.1 CLI

```bash
B=.venv-run/bin/miniflims
$B run -c configs/X.yaml [-o key=value ...] [--fresh]   # run or resume (output.dir from the config)
$B status runs/X          # sweep i/total, phase, sweeps/h, ETA, number of field samples
$B watch runs/X [--interval 60] [--once] [--no-plot]    # writes runs/X/progress.log and progress.png
$B plot runs/X            # traces.png, field.png, diagnostics.png (finished runs)
$B show-config -c X.yaml
$B components
```

Exit codes of `run`:
- 0: finished.
- 2: configuration error, including an incompatible resume.
- 3: stopped by a signal after writing a checkpoint.

### 5.2 Resuming and extending

- **Resume:** running the same command again resumes from
  `checkpoint.npz`. If that file is unreadable it falls back to
  `checkpoint.prev.npz`.
- **Allowed changes on resume:** only `output.*`, `sampler.checkpoint.*` and
  a **larger** `sampler.n_samples`. Anything else is refused with exit 2.
- **Extending a finished run** by N sweeps:

  ```bash
  $B run -c runs/X/config.yaml -o sampler.n_samples=<old n_samples + N>
  ```

  `n_samples` counts sweeps after warm-up, so total = `n_warmup + n_samples`.
  The resumed run rewrites `runs/X/config.yaml`, `summary.json` and
  `chains.npz`. Copy them aside first if you need the old numbers, as done
  in `runs/png_feature_w20/at_1300/`.
- **Stopping:** SIGTERM, SIGUSR1 or Ctrl-C checkpoint after the current
  sweep and exit 3. The checkpoint can therefore land at an arbitrary sweep
  (e.g. 333).

### 5.3 Several runs in sequence

`scripts/chain_runs.sh` runs jobs one after another. Jobs are separated by
`--`:

```bash
scripts/chain_runs.sh runs/a/config.yaml -o sampler.n_samples=1000 -- \
                      runs/b/config.yaml -o sampler.n_samples=1000 > runs/chain_a_b.log 2>&1
```

- It uses `.venv-run` (override with `MINIFLIMS_BIN`).
- Each run's stdout and stderr, including the tqdm bar, are appended to
  `<run_dir>/run_stdout.log`.
- It starts `miniflims watch` for each run once that run has written a new
  checkpoint.
- It logs start, end and exit code of each job to its own stdout.

### 5.4 Files in a run directory

| File | Written by | Contents |
|---|---|---|
| `config.yaml` | run | Resolved config (rewritten on every resume) |
| `checkpoint.npz`, `checkpoint.prev.npz` | run, every `checkpoint.every` sweeps | State, adaptation, PRNG key, full traces |
| `samples/field_0000300.npy` … | run | White noise (chains, n, n, n) for kept sweeps; 4.2 MB each at 64³ with 2 chains |
| `run.log` | run | Header, `resuming from sweep i/total`, and every `log_every` sweeps: acceptance, ε, logL, PNG values |
| `run_stdout.log` | `chain_runs.sh` | Console output, including the tqdm bar |
| `progress.log`, `progress.png` | `watch` | One line per checkpoint: acceptance, ε, logL, var(x), and every sampled parameter for each chain (`a/b`). Truth in the header. |
| `summary.json` | run, at the end | `wall_time_s, n_chains, n_sweeps, hmc, log_likelihood, nuisance, png, n_field_samples, white_noise_moments, ic_cross_correlation, density_cross_correlation`. Each parameter has `mean, std, rhat, ess, truth`. |
| `chains.npz` | run, at the end | `trace/...` (chains, sweeps, ...) for every traced quantity; `counts`, `response`, `truth/*`, `mean_density` |
| `traces.png`, `field.png`, `diagnostics.png` | `miniflims plot` | |
| `last_sample/` | `scripts/plot_last_sample.py` | Section 6 |

### 5.5 Checking a run is alive

- `ps -o pid,pcpu,etime,command -p $(pgrep -f "Python.*miniflims run -c <config>")`
  - A healthy local run uses about 480–520 % CPU.
  - 0 % CPU with no new checkpoint after about 2 × the usual interval means
    a hang (section 7).
- **Match `Python.*miniflims run ...` with `pgrep -f`.** A bare
  `miniflims run` also matches the zsh wrapper of your own Bash tool call
  and gives false positives.
- **Waiting for a checkpoint:** `read_meta(run_dir)["iteration"]` from
  `miniflims.checkpoint` gives the latest checkpointed sweep. Use it in an
  `until` loop run in the background.

## 6. Analysis

```python
from miniflims.checkpoint import load_checkpoint, read_meta, load_samples
ckpt, meta = load_checkpoint("runs/X")    # works while running; meta: iteration, total, config, written, wall_time_s
ckpt.state["field"]   # (chains, n, n, n) current white noise
ckpt.state["png"]     # (chains, 1, 3): f_NL, omega, phase
ckpt.state["nuisance"]# (chains, n_catalogues, n_params), e.g. nbar, alpha
ckpt.trace["log_likelihood"], ckpt.trace["png"], ckpt.trace["nuisance"]   # (chains, sweeps, ...)
sweeps, fields = load_samples("runs/X", max_samples=50)  # (chains, kept, n, n, n)
```

**Mock truth:** `miniflims.data.make_mock(cfg)` regenerates it exactly from
the config. `chains.npz` also stores `truth/*`.

**Scripts** (run them with `.venv-run/bin/python -u` so background logs
flush):

| Script | Use |
|---|---|
| `scripts/plot_last_sample.py RUN [--plot-only] [--omega-step 0.05] [--n-gauss 200] [--slab 4]` | Latest checkpoint state against the regenerated truth. Writes `last_sample/{fields,spectra,png_likelihood,gaussianity}.png`, `summary.json` and `profiles.npz`. The PNG likelihood scans (conditional slices, profile over f_NL as a function of ω) take several minutes at 64³. The script checks that the regenerated data reproduce the traced logL and raises otherwise. |
| `scripts/plot_preliminary.py RUN [--max-field-samples 40]` | Preliminary results of a running run from its latest checkpoint: `summary.json`, `chains.npz` and the `miniflims plot` figures, written to `RUN/prelim_<sweep>/`. Uses whichever field samples are in `RUN/samples`; a subset is enough. |
| `scripts/plot_bispectra.py RUN [--n-bins 8] [--n-mc 200] [--from-sweep 1000]` | Binned bispectra (all closed triangles of \|k\| bins) of the white noise, the primordial δ_L (split into white-noise and PNG parts) and ρ − 1. Shows truth, posterior (field samples) and posterior − truth, in units of the Gaussian-IC scatter. Writes `RUN/bispectra/`. About 2 min at 64³. |
| `scripts/make_quijote_data.py CUBE -c CONFIG [--nbar 5 --alpha 1.5 --truth-png 100 --seed 20]` | Galaxy counts from an N-body density cube (Quijote-PNG, 256³ CIC): the cube is CIC-assigned to the mesh, then Poisson-sampled with the config's bias. Writes `data.path` with a partial truth (`truth_rho`, `truth_nuisance`, `truth_png`; no white noise), which `data.source: file` reads. |
| `scripts/make_hod_mock.py HALOS --hod zheng09_lrg_21.2 --sigma8 0.834 -c CONFIG [--matter-cube CUBE] [--rsd z] -o data/....npz` | HOD galaxy mock from a halo catalogue: Rockstar `out_*_pid.list` (host halos, M200b), a Quijote FoF directory (needs Pylians), or `.npz`. Uses `miniflims.hod`. Writes a counts file for `data.source: file` (NGP to the mesh nodes, with `truth_rho` from the matter cube), a galaxy catalogue `_galaxies.npz`, and a diagnostic figure. It reports halo-catalogue incompleteness against the Tinker+08 mass function. |
| `scripts/pinetree_whitening.py -c CONFIG [-o out.npz] [--floor-frac 1e-6]` | Whitening for PineTree parameters (section 4.5): Hessian of −ln L at the truth with the true field, eigenvalues floored; writes the file named by `galaxies.pinetree.whitening`. About 1 min at 64³. |
| `scripts/pinetree_posterior.py RUN [--burn N] [--draws 100]` | PineTree parameters of a run against the truth (mean, std, pull, R̂), plus predictions on the true field: Σλ, λ in bins of ρ, the HMF above threshold. Writes `RUN/pinetree_posterior.png`. Works on a running run. |
| `scripts/compare_burnin.py RUN_A RUN_B ... -o out.png` | Traces of f_NL, ω, α, logL and var(x) (from `progress.log`) for several runs, plus f_NL against var(x) |
| `scripts/make_png_meshes.py`, `make_png_realisations.py`, `plot_realisations.py` | PNG density meshes, Zel'dovich or tCOLA |
| `scripts/compare_public_codes.py`, `compare_quijote_png.py` | Cross-checks against public codes and Quijote-PNG (needs `.[compare]`) |
| `scripts/gpu_backend_probe.py` | Probe experimental JAX GPU backends (jax-mps). Only in `.venv`. |

**Technique: likelihood scans.**
- ln L is close to quadratic in f_NL, because the density is linear in f_NL.
  A 3-point parabola therefore finds f̂_NL(ω).
- Precompute Φ_NG once per ω, then evaluate the f_NL values with
  `jax.lax.map(..., batch_size=8)`.
- **Never `vmap` hundreds of 64³ forward models.** 501 at once took about
  13 GB and died silently.

## 7. Known pitfalls and lessons

1. **JAX 0.11.2 deadlock.** See rule 1. Symptom: the run stops after a few
   sweeps at 0 % CPU, with no error and no new checkpoint.
2. **Slow large-scale burn-in.**
   - Cause: the chains start from `field_scale: 0.1`, so the white-noise
     variance climbs from about 0.94 to 1 only over roughly 600–700 sweeps.
   - Effect: during that time f_NL, ω and the bias compensate. Both the w20
     run and the null run went to f_NL ≈ −600, ω ≈ 10–12, α ≈ 1.9 early on.
   - Proposed fixes (not implemented):
     - `sampler.init.field_scale: 1`.
     - Freeze the PNG block for the first 100–200 warm-up sweeps.
     - Precondition the large-scale modes in HMC.
3. **Secondary ω mode.**
   - Even after var(x) reached 1, the w20 chains stayed at ω ≈ 12 with f_NL
     ≈ −350 to −390. The truth is ω = 20, f_NL = 500.
   - Univariate slice moves in ω at fixed f_NL cannot reach the true mode,
     because getting there means flipping the sign of f_NL at the same time.
   - Proposed fix (not implemented): a joint (ω, f_NL) update with f_NL
     profiled or marginalised analytically, which works because ln L is
     quadratic in f_NL.
4. **The watcher stops early on an extended run.** `watch` exits once
   checkpoint iteration ≥ total. Started on a finished run before the
   extension writes its first checkpoint, it sees the old total and quits.
   `chain_runs.sh` waits for a new checkpoint, and `cluster/slurm/job.slurm`
   restarts the watcher.
5. **Sampled parameters missing from `data.mock.truth`** show their `init`
   as "truth" in `summary.json`. Example: ω = 15 in the null run. This is
   cosmetic.
6. **Memory blow-ups from `vmap`:** see section 6.
7. **`stdp` mirror mode:** see the README's *Known pitfall*.

## 8. Cluster (CC-IN2P3, Slurm)

**Layout** (all under `/sps/euclid/Users/strusov`, set by `MINIFLIMS_BASE`):

| What | Path |
|---|---|
| Code | `/sps/euclid/Users/strusov/miniflims` (rsync copy; no git) |
| Environment | `/sps/euclid/Users/strusov/venvs/miniflims` (GPU build about 4 GB; JAX CUDA 12 from pip, no CUDA module) |
| Runs | `/sps/euclid/Users/strusov/miniflims_runs/<name>/` (logs in `<name>/logs/`) |

**Access.**
- Hostname: `cca.in2p3.fr` (the SSH config sends it to `cca011.in2p3.fr`),
  user `strusov`. Login is password only.
- The user opens a reusable connection in their own terminal:

  ```bash
  ssh -fNM -o ControlPersist=24h -o ServerAliveInterval=60 -o ServerAliveCountMax=10 -S ~/.ssh/cm-cca strusov@cca.in2p3.fr
  ```

- While that socket exists, you can run remote commands with
  `ssh -S ~/.ssh/cm-cca strusov@cca.in2p3.fr '...'` and
  `rsync -e "ssh -S $HOME/.ssh/cm-cca"`. Check it first with
  `ssh -S ~/.ssh/cm-cca -O check strusov@cca.in2p3.fr`.
- Always add `-o BatchMode=yes` to your own ssh/rsync calls. A socket can drop (overnight it did, although `-O check` still said "Master running"); without BatchMode, ssh then falls back to password prompts and the server disconnects after repeated failures.
- If the socket is missing or dead, give the user the commands instead.

**Fetching results back** (all but the field samples, plus every 100th sample):

```bash
rsync -a -e "ssh -S $HOME/.ssh/cm-cca" --exclude .slurm_job --exclude '*.tmp' \
    --include 'samples/' --include 'samples/field_*00.npy' --exclude 'samples/*' \
    strusov@cca.in2p3.fr:/sps/euclid/Users/strusov/miniflims_runs/<name>/ runs/<name>/
```

**Procedure:**

```bash
# copy the code (again after every code change)
cd /Users/strusov
rsync -av -e "ssh -S $HOME/.ssh/cm-cca" \
    --exclude '.venv*' --exclude runs --exclude 'runs_*.log' --exclude figures \
    --exclude meshes --exclude __pycache__ --exclude '*.egg-info' \
    miniflims/ strusov@cca.in2p3.fr:/sps/euclid/Users/strusov/miniflims/

# once, on the login node: build the environment
ssh -S ~/.ssh/cm-cca strusov@cca.in2p3.fr 'cd /sps/euclid/Users/strusov/miniflims && bash cluster/setup_env.sh'

# optional: GPU check (also times the PNG gradient)
ssh -S ~/.ssh/cm-cca strusov@cca.in2p3.fr 'cd /sps/euclid/Users/strusov/miniflims && srun -p gpu_h100_interactive --gpus 1 -c 4 --mem 16G -t 0-00:30 bash -c "source cluster/activate.sh && python cluster/check_gpu.py"'

# submit
ssh -S ~/.ssh/cm-cca strusov@cca.in2p3.fr 'cd /sps/euclid/Users/strusov/miniflims && cluster/slurm/submit.sh configs/png_feature_w20_10k.yaml'

# follow
ssh -S ~/.ssh/cm-cca strusov@cca.in2p3.fr 'squeue -u strusov; tail -3 /sps/euclid/Users/strusov/miniflims_runs/png_feature_w20_10k/progress.log'
```

**`cluster/slurm/submit.sh` options:**
- `--gpu` (the default: partitions `gpu_v100,gpu_h100`, 1 GPU), `--cpu`
  (partition `htc`).
- `-p`, `-t` (default `1-00:00:00`), `-c`, `--mem`.
- `-n NAME` (run name; defaults to the config name).
- `--seeds 0-3` (job array, one run directory per seed).
- `--resubmits N` (10), `--signal-lead S` (900 s).
- `--no-plot`, `--dry-run`.
- Everything after the config is passed to `miniflims run` (`-o ...`).

**Job behaviour (`cluster/slurm/job.slurm`):**
- Checks that JAX sees a GPU, and stops if not.
- Runs `miniflims run` with `output.dir` set to the run directory.
- Keeps `progress.log` and `progress.png` updated (`MINIFLIMS_WATCH=0` turns
  this off).
- At the time limit: USR1, then checkpoint, exit 3, and automatic
  resubmission with `afterany`.
- `scancel`: checkpoints, no resubmission.
- When finished: `miniflims plot`.
- Refuses to share a run directory with another live job.
- Paths under `/sps` add `--licenses=sps` automatically.

**Extending a cluster run:** submit the same config again with
`-o sampler.n_samples=<larger>` and the same `-n`.

## 9. Current state (2026-11-25)

| Run | Where | State |
|---|---|---|
| `runs/png_feature_w20` | local | Stopped at sweep 1677/2300 (it duplicated the much faster cluster run); resumable. Snapshot at 1300 in `at_1300/`; `last_sample/` from sweep 333. |
| `runs/png_null_w20` | local | Finished, 1300 sweeps. `burnin_vs_w20.png` compares it with w20. |
| `runs/nopng_w20_control` | local | Stopped at 121/800, resumable |
| `runs/png_feature_w20_priors` | local | Hung (JAX 0.11.2), no checkpoint; ignore |
| `png_feature_w20_10k` | cluster | **Finished** 2026-11-25 19:46 (4.0 h on a V100). f_NL = −386 ± 91, ω = 12.19 ± 0.62 (truth 500, 20; R̂ 1.04/1.03): converged on the ω ≈ 12 alias. Local mirror `runs/png_feature_w20_10k/` (every 100th sample): `prelim_*/`, `last_sample/` (sweep 1660), `last_sample_final/`, `bispectra/`. |
| `png_feature_w20_128` | cluster | `configs/png_feature_w20_128.yaml`: 128³ in the same box (k_Nyq 0.39 h/Mpc), nbar 0.625 per cell (same galaxy density), thin 50. Job 64855789, started 2026-11-25 17:39 on a V100. 11.3 s/sweep, about 31 h, so two 1-day jobs. Cluster only: never run it locally. |
| `png_feature_w20_fnlm500`, `png_feature_w50_fnl500`, `png_equilateral_fnl500` | cluster | Finished 2026-11-26 (H100, about 6200 sweeps/h for feature). Results: ω=20, f_NL=−500 recovered (−498 ± 103, ω 19.9). Equilateral 500 recovered (492 ± 94). ω=50: chain 1 at the truth (622, 49.8), chain 2 in the alias (−377, 41.3); no gate. |
| `png_feature_w20_gated` | cluster | Finished 2026-11-26 08:17 (job 64879067). **Truth recovered**: f_NL = 440 ± 85, ω = 20.44 ± 0.55 (R̂ ≤ 1.04); gate opened at sweeps 294 and 332. |
| `quijote_eq_2lpt` | cluster | `configs/quijote_eq_2lpt.yaml`: 2LPT inference of the Quijote-PNG equilateral box (`~/Downloads/quijote_png_eq.npy`, assumed EQ_p, f_NL = +100) at 64³, Quijote cosmology, gated. Job 64886692, submitted 2026-11-26. σ(f_NL) at 64³ is about 95, so this tests model-mismatch bias rather than detection. |

**Results so far** (posterior over sweeps 300–1300):

| | w20 (truth) | null (truth) |
|---|---|---|
| f_NL | −389 ± 123 (500), R̂ 1.62, ESS 10 | −145 ± 131 (0), R̂ 1.21, ESS 14 |
| ω | 11.7 ± 0.9 (20), R̂ 1.24 | 18.7 ± 11 (unconstrained) |
| α | 1.512 ± 0.013 (1.5) | 1.510 ± 0.013 (1.5) |
| nbar | 5.012 ± 0.004 (5.0) | 5.005 ± 0.004 (5.0) |

**Interpretation.**
- At the true field the data detect the feature: the profile peaks at
  ω = 20.05, f_NL = 479, with Δln L ≈ 140.
- The chains have not found that mode. They sit in the burn-in-induced
  ω ≈ 12, f_NL < 0 mode (pitfalls 2 and 3).

**HOD mocks (2026-11-26).** `~/Documents/quijote/out_4_pid.list` is a fiducial Quijote Rockstar catalogue at z = 0 from the same simulation as `~/Documents/quijote/df_m_*_CIC_z=0.npy` (r_gm ≈ 0.95). It matches Tinker+08 within 4 % above 10¹⁴ but is **incomplete below about 10^13.8 Msun/h**: about 53 % (M_g<−21.2) or 40 % (M_g<−21.8) of the LRG HOD galaxies would sit in missing halos. The mocks are `data/quijote/hod_lrg212_fid_64.npz` and `hod_lrg218_fid_64.npz`. Mocks of the Quijote-PNG boxes need their Rockstar or FoF catalogues from Globus.

**Open items.** Do none of these unless the user asks.
- Tests for the non-local bias (`tests/test_nonlocal_bias.py`).
- The bias model of Andrews et al. 2024 (arXiv:2412.11945). The user asked
  for it, then paused the work. Do not start it without their go-ahead.
- The sampler fixes in pitfalls 2 and 3. A w20 rerun with them should be
  paired with the null test.
- README gaps: the λ_ps and bispectrum priors, `miniflims watch` and
  `run.log`, the non-local bias, the new scripts, and `.venv-run`. The
  README's cluster section is already up to date.

## 10. Changing the code

- **New components:** register them in the matching registry. See README
  *Extending*. The registries are `TRANSFER_FUNCTIONS`, `FORWARD_MODELS`,
  `BIAS_MODELS`, `NONLOCAL_TERMS`, `LIKELIHOODS`, `SELECTIONS`,
  `INTEGRATORS` and `BLOCKS`.
- **Config changes** go in `config.py` (frozen dataclasses with defaults),
  so old run configs still load.
- **Resume compatibility:** resuming compares the old and new resolved
  configs. New keys need defaults that reproduce the old behaviour, or old
  checkpoints stop resuming.
- **Before handing back:**
  - Run the fast test suite in `.venv-run`.
  - Add tests next to the closest existing file in `tests/`.
  - For anything inside the gradient path, check memory and gradient
    correctness against finite differences. Examples:
    `tests/test_png.py::test_posterior_gradient_with_png_matches_finite_differences`,
    `tests/test_png.py::test_gradient_memory_through_phi_ng_is_checkpointed`,
    `tests/test_cola.py::test_gradient_matches_finite_differences`.
- **Pushing to the cluster:** rsync again (section 8). A job resumed with
  new code uses the new code. Avoid changing numerics under a running
  chain unless the user wants that.
