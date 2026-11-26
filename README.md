# miniflims

A small, modular field-level inference pipeline in JAX, modelled on
BORG / Manticore (McAlpine et al. 2025, arXiv:2505.10682).

```
white noise x ~ N(0,1)
   │  × sqrt(P(k))            Eisenstein–Hu transfer function, sigma8-normalised
   ▼
linear density δ_lin
   │  Zel'dovich (1LPT) or tCOLA particle mesh, + optional RSD + CIC
   ▼
evolved density ρ = 1 + δ
   │  bias B(ρ), ergodic normalisation, survey response R
   ▼
expected counts λ_c = n̄_c R_c B_c/<B_c>     (per sub-catalogue c)
   │  Poisson / generalized Poisson / Gaussian
   ▼
galaxy counts N_c
```

Sampling is block Gibbs: **HMC** on the white-noise field given the nuisance
parameters, then **univariate slice sampling** (stepping-out + shrinkage) of
each n̄, bias and noise parameter given the field.

## Install and run

```bash
python -m venv .venv && .venv/bin/pip install -e ".[test]"

.venv/bin/miniflims run -c configs/default.yaml               # ~2 min on a laptop CPU
.venv/bin/miniflims run -c configs/manticore_like.yaml
.venv/bin/miniflims run -c configs/tcola.yaml                 # tCOLA particle-mesh gravity
.venv/bin/miniflims plot runs/default                         # traces, field, diagnostics .png  (needs ".[plot]")
.venv/bin/miniflims status runs/default                       # progress from the checkpoint, also while running
.venv/bin/miniflims run -c configs/default.yaml -o sampler.hmc.integrator=mclachlan2 -o grid.n_grid=16
.venv/bin/miniflims show-config -c configs/default.yaml       # fully resolved config
.venv/bin/miniflims components                                # registered component names

.venv/bin/pytest -m "not slow"     # unit tests, ~1.5 min
.venv/bin/pytest                   # + end-to-end sampling tests, ~2.5 min
```

```python
from miniflims import load_config, run
res = run(load_config("configs/default.yaml", ["sampler.n_samples=200"]))
res.summary            # acceptance, R-hat, ESS, nuisance vs truth, cross-correlations
res.fields             # white-noise samples (chains, kept, n, n, n)
res.gibbs.trace        # everything traced per sweep, (chains, sweeps, ...)
```

A run writes `config.yaml` (resolved), `summary.json` and `chains.npz` to
`output.dir`, and the kept white-noise samples to `output.dir/samples/` as
they are drawn (one `.npy` per kept sweep; `miniflims.checkpoint.load_samples`
reads them back).

### Checkpoints, resuming and progress

With `output.dir` set, the sampler writes `checkpoint.npz` every
`sampler.checkpoint.every` sweeps (default 20) and keeps the previous one as
`checkpoint.prev.npz`. It holds the sampler state, the HMC step-size and
slice-width adaptation, the PRNG key and the traces, so a resumed run
continues the same computation: an interrupted and resumed run gives the
same chains as an uninterrupted one (bit for bit on CPU).

- Running the same command again resumes automatically. `--fresh` (or
  `sampler.checkpoint.resume: false`) discards the old run instead.
- Resuming refuses a changed configuration, except for `output.*`,
  `sampler.checkpoint.*` and a larger `sampler.n_samples`, which extends a
  finished run.
- SIGTERM, SIGUSR1 or Ctrl-C stop the run after the current sweep with a
  checkpoint; `miniflims run` then exits with code 3. A second Ctrl-C aborts
  at once.
- A tqdm bar on stderr shows sweeps, ETA, acceptance, step size and
  log-likelihood (`output.progress`; in batch logs it is refreshed every 30 s,
  `output.progress_interval`).

## Running on the cluster

Scripts in `cluster/` set up a virtual environment on top of the cluster's
Python module, with JAX's CUDA 12 build from pip (it brings its own CUDA and
cuDNN, so no CUDA module is loaded). Versions are pinned to the ones the
tests passed with (`cluster/constraints.txt`, JAX 0.12.1).

Code, environment and runs live under `/sps/euclid/Users/$USER` (`MINIFLIMS_BASE`;
`MINIFLIMS_VENV` and `MINIFLIMS_RUNS` override the parts):

| | default |
|---|---|
| code | `/sps/euclid/Users/$USER/miniflims` |
| environment | `/sps/euclid/Users/$USER/venvs/miniflims` (GPU build ~4 GB) |
| runs | `/sps/euclid/Users/$USER/miniflims_runs/<name>` |

```bash
# from your machine: copy the code (again after every change)
rsync -av --exclude '.venv*' --exclude runs --exclude 'runs_*.log' --exclude figures \
    --exclude meshes --exclude __pycache__ --exclude '*.egg-info' \
    miniflims/ strusov@cca.in2p3.fr:/sps/euclid/Users/strusov/miniflims/

# on a login node, once
cd /sps/euclid/Users/$USER/miniflims
bash cluster/setup_env.sh                    # or: bash cluster/setup_env.sh cpu

# in every shell / batch job
source cluster/activate.sh
python cluster/check_gpu.py                  # on a GPU node: JAX sees the GPU, gradients (incl. PNG) run in float64
pytest -m "not slow"                         # optional
miniflims run -c configs/tcola.yaml -o output.dir=$PWD/runs/tcola
```

### Slurm jobs at CC-IN2P3

`cluster/slurm/submit.sh` submits `cluster/slurm/job.slurm` with the
resources set (CC-IN2P3 refuses jobs without `-t`, `-c` and `--mem`):

```bash
cluster/slurm/submit.sh configs/tcola.yaml                          # 1 GPU on gpu_v100 or gpu_h100
cluster/slurm/submit.sh -p gpu_h100 -t 2-00:00:00 -n tcola64 configs/tcola.yaml -o grid.n_grid=64
cluster/slurm/submit.sh --seeds 0-3 configs/manticore_like.yaml     # job array: 4 independent chains
cluster/slurm/submit.sh --cpu -c 16 --mem 32G configs/default.yaml  # htc partition
cluster/slurm/submit.sh --dry-run configs/tcola.yaml                # show the sbatch command only

squeue -u $USER
miniflims status $MINIFLIMS_RUNS/tcola                             # progress, compute time, ETA
tail -f $MINIFLIMS_RUNS/tcola/logs/*.out          # or .../tcola/progress.log
```

- Runs go to `--runs-dir` (default `$MINIFLIMS_RUNS`, else
  `/sps/euclid/Users/$USER/miniflims_runs`). Jobs whose code, environment or
  run directory is under `/sps` declare the sps storage licence (`-L sps`)
  automatically.
- While a job runs, `miniflims watch` keeps `progress.log` and `progress.png`
  in the run directory up to date (one line per checkpoint;
  `MINIFLIMS_WATCH=0` switches it off).
- Slurm sends USR1 to the job `--signal-lead` seconds (default 900) before
  its time limit; the job forwards it, miniflims checkpoints, and the job
  resubmits itself with `--dependency=afterany` (up to `--resubmits`, default
  10). A finished run is plotted (`traces/field/diagnostics.png`).
- `scancel <id>` also checkpoints but does not resubmit; continue later by
  submitting the same command again.
- GPU jobs check first that JAX sees the GPU and stop otherwise. Run
  `cluster/check_gpu.py` once on `gpu_v100_interactive` and
  `gpu_h100_interactive` (`srun -p gpu_v100_interactive --gpus 1 -t 0-01:00
  --mem 8G --pty bash -i`); if V100 does not work with this JAX build, use
  `-p gpu_h100`.
- Two jobs never write into the same run directory: a job finds the other
  one through `squeue` and stops.

`activate.sh` loads `Programming_Languages/python/3.13.17`, removes CUDA
directories from `LD_LIBRARY_PATH` (so `nvhpc` or `libtorch.cu118` cannot
shadow JAX's CUDA libraries) and keeps a persistent XLA compilation cache in
the environment directory. The shipped configs run in float64: fine on
data-centre GPUs (V100, A100, H100), slow on GPUs with reduced FP64
throughput.

## Configuration

Everything is set through one typed config tree (`src/miniflims/config.py`):
defaults ← YAML file ← dotted `-o key=value` overrides. Unknown keys and wrong
types are errors. The main switches:

| Section | What it controls |
|---|---|
| `cosmology`, `power` | parameters, transfer function (`eisenstein_hu`, `eisenstein_hu_nowiggle`, `bbks`) |
| `grid` | IC grid, box size, density mesh |
| `forward` | `zeldovich`, `tcola` or `linear`; redshift; RSD (plane-parallel or radial); particle lattice offset; `cola`: steps, starting redshift, force-mesh resolution, low-memory gradients |
| `galaxies` | number of sub-catalogues, bias model (`linear`, `power_law`, `truncated_power_law`, `stdp`), likelihood (`poisson`, `generalized_poisson`, `gaussian`); per parameter: `init`, bounds, `sample: false` to fix it, slice width, `uniform`/`normal` prior |
| `data` | mock (seed, truth, selection functions per catalogue) or `.npz` file with `counts` and `response` |
| `sampler` | chains, warm-up, Gibbs block order, HMC (integrator, step size, trajectory-length range, jitter, dual averaging), slice sampler, initialisation, hyper-parameter schedules |
| `prior` | binning of the power-spectrum prior |
| `output` | directory, logging, summary binning |

## Primordial non-Gaussianity (jaxpng)

`png.enabled: true` makes the initial conditions non-Gaussian with
[jaxpng](jaxpng/) (`src/miniflims/png.py`):

    Phi_G = delta_G / M(k),   Phi = Phi_G + f_NL Phi_NG[Phi_G]   (jaxpng, Schwinger kernel)
    delta = M(k) Phi,          M = (2/3) k^2 T(k) D_md / (Omega_m H0^2)

with M(k) built from miniflims' own transfer function (it agrees with
`jaxpng.cosmology.poisson_M` to 1e-5). Templates: `local`, `equilateral`,
`orthogonal`, `feature` (6 (P1P2P3)^(2/3) sin(ω Σk + φ)) and
`equilateral_feature`.

- **Gibbs step.** Add `png` to `sampler.blocks` (`[field, nuisance, png]`):
  f_NL, and ω / φ if `sample: true`, are slice-sampled given the field and the
  bias. With only f_NL sampled the density is linear in f_NL, so Φ_NG is built
  once per sweep and each slice evaluation costs one forward model. Without
  the block the PNG parameters stay fixed at `init`.
- **HMC.** The field gradient runs through jaxpng's Φ_NG. jaxpng's `remat`
  (on by default) keeps that reverse pass at a few fields: 3.8 MB instead of
  976 MB at 32³.
- **Cost and accuracy.** jaxpng's Schwinger t-integral uses a trapezoid in
  t (second order: 361 nodes for 4e-4). miniflims passes it a trapezoid in
  ln t instead (`png.quadrature: log`, `png.n_t: 32`, via the `t_grid`
  argument added to `jaxpng.phing.phiNG`), which converges exponentially:
  32 nodes give 3e-5. One posterior gradient at 32³:

  | | no PNG | local, `local_exact` | local | equilateral |
  |---|---|---|---|---|
  | jaxpng t-grid (361 nodes) | 3.6 ms | 4.1 ms | 530 ms | 1.86 s |
  | ln t grid, 32 nodes (default) | | | 51 ms | 168 ms |
  | ln t grid, 24 nodes, `eps: 1e-4` (2e-4 error) | | | 39 ms | 130 ms |

  For local f_NL `local_exact` (jaxpng's exact anchor Φ_G² − ⟨Φ_G²⟩) is
  cheapest. `png.quadrature: jaxpng` restores jaxpng's own grid.
- **Mocks.** `data.mock.truth` may contain `f_NL`, `omega`, `phase`.
- Example: `configs/png_local.yaml`.
- Meshes: `scripts/make_png_meshes.py` (one white noise, several PNG
  settings, Zel'dovich and tCOLA) and `scripts/make_png_realisations.py`
  (several realizations).
- Quijote-PNG: `scripts/compare_quijote_png.py` (see "Comparison with public
  codes").

jaxpng must be importable: `pip install -e ./jaxpng --config-settings
editable_mode=compat` (the compat mode avoids the `jaxpng/` project folder
shadowing the package), or set `png.jaxpng_path`.

## tCOLA forward model

`forward.model: tcola` is the tCOLA particle-mesh solver of BorgJaxPM
(`src/miniflims/cola.py`), ported so that it needs neither BORG nor
`jax_cosmo` / `jaxpm` / `jaxdecomp`: Zel'dovich initial conditions at
`cola.z_start`, then `cola.n_steps` drift-kick-drift steps in the LPT frame
with finite-difference PM forces on a mesh of `cola.force_sampling * n_grid`.
It supports RSD from the particle momenta and coarser output meshes.

- **Fidelity.** With `forward.lattice_offset: 0` the port reproduces the
  original (run unmodified under JAX 0.8) to better than 1e-4 cells in
  particle position (4.7e-5 and 7.5e-5 in two 32³ runs); the residual is the
  original's float32 time-integration coefficients. A reference output is
  stored in `tests/data/` and checked by the tests.
- **Memory of the gradient** (`cola.low_memory`, default on). As in the
  original, the CIC operators have analytic adjoints and the force has a
  custom VJP that stores only particle positions and rebuilds the meshes on
  the way back. In addition the time loop is one `lax.scan`. Measured with
  XLA's buffer accounting at 32³ particles and a 64³ force mesh: the forward
  pass needs 16 MB for any number of steps; the gradient needs 37 MB at 10
  steps and 44 MB at 20 (one particle array per step), against 307 MB and
  587 MB with plain autodiff.
- **Force mesh.** `force_sampling: 1` under-resolves the force and loses
  power on all scales relative to Zel'dovich; use 2 or more (default 2).
  With a resolved force, small-scale power relative to Zel'dovich depends on
  the cell size: with 7.8 Mpc/h cells tCOLA has more (up to ×1.36 at the
  Nyquist frequency, non-linear clustering), with 15.6 Mpc/h cells less
  (×0.7-0.8, the force is still smoothed by the mesh). Large scales agree to
  2-4 %.
- **Particle lattice.** By default particles start half a force-mesh cell
  off the nodes (`lattice_offset = 0.5 / force_sampling`). With particles
  starting on force-mesh nodes, which is the BORG / JaxPM layout, the CIC
  kernel is at its kink and a small-amplitude plane wave ends up 8-21 % away
  from the exact Zel'dovich solution. With the default offset the error is
  below 0.01 % at `force_sampling: 2` and 0.8 % at 4; at 1 the coarse force
  alone gives 5 %. See `figures/compare_cola.png`.

Not ported: multi-device sharding, the float32 cancellation guards (use
`precision: float64`), and the other integrators (leapfrog, bullfrog).

## PineTree population model

`galaxies.population: pinetree` replaces the gridded-count galaxy model by
PineTree (Ding, Lavaux & Jasche 2024), a mass-resolved halo model conditioned
on the evolved density. Per voxel, a 3³ monopole kernel on ln ρ and a softplus
feed a two-log-normal conditional mass function n(ln M | v). Halos above
`mass_threshold` are a Poisson process with intensity R_v n(ln M | v):

    ln L = Σ_h ln[R n(ln M_h | v_h)] − Σ_v R_v Σ_k A_k Φ((μ_k − ln M_th)/σ_k)

This is PineTree's training loss with the sign flipped. The data are the halo
catalogue (`halo_voxel`, `halo_mass`), from a mock or a data file.

- **Parameters:** the 19 PineTree weights are the nuisance parameters. They come
  from a trained model via PineTree's `scripts/export_pinetree_params.py`
  (`configs/pinetree/`).
- **Sampling:** in whitened coordinates (`scripts/pinetree_whitening.py`),
  because the raw weights are strongly correlated.
- **Tests:** the implementation (`src/miniflims/pinetree.py`, pure JAX)
  reproduces PineTree's own code to 1e-11 (`tests/test_pinetree.py`).

Self-consistency test (mock drawn from PineTree trained on Gaussian Quijote,
2LPT, 500 Mpc/h on 64³ = PineTree's 7.8 Mpc/h voxel):

```bash
.venv-run/bin/python scripts/pinetree_whitening.py -c configs/pinetree_sc_theta.yaml
.venv-run/bin/miniflims run -c configs/pinetree_sc_theta.yaml     # parameters, field fixed at the truth
.venv-run/bin/python scripts/pinetree_posterior.py runs/pinetree_sc_theta
cluster/slurm/submit.sh configs/pinetree_sc_field.yaml            # field, parameters fixed
cluster/slurm/submit.sh configs/pinetree_sc_joint.yaml            # both
```

## Comparison with public codes

`scripts/compare_public_codes.py` writes figures and `comparison.json` to
`figures/` (install `".[compare]"`):

| Figure | What is compared |
|---|---|
| `compare_power.png` | Eisenstein-Hu P(k) and growth vs colossus, jax_cosmo and CAMB |
| `compare_lpt.png` | Zel'dovich displacement and CIC density vs JaxPM |
| `compare_cola.png` | tCOLA vs the original BorgJaxPM; force mesh; lattice; gradient memory |
| `compare_sampler.png` | HMC-within-Gibbs posterior vs blackjax NUTS on the joint posterior |

`scripts/compare_quijote_png.py` compares with one Quijote-PNG pair
(Coulton et al. 2023; `LC_p/i` and `LC_m/i` share a seed, so their difference
is the primordial signal). Quijote data are only on Globus (login needed):

```bash
.venv/bin/python scripts/compare_quijote_png.py generate --n 128 --seeds 8   # miniflims side (done: comparisons/quijote_png/)
# fetch one pair with Globus (web app or CLI), e.g. the 256^3 CIC density cubes at z = 0
# of LC_p/0 and LC_m/0 (3D_cubes folder; check the exact path with `globus ls`)
.venv/bin/python scripts/compare_quijote_png.py compare --set LC \
    --plus LC_p_0_df_m_256_CIC_z=0.npy --minus LC_m_0_df_m_256_CIC_z=0.npy
```

Snapshots work too (`--plus <snapdir>`, needs `h5py` and `hdf5plugin`).

## Extending

Each component is looked up by name in a registry; adding one does not touch
the pipeline:

```python
from miniflims.bias import BIAS_MODELS, BiasModel
from miniflims.params import ParamSpec

@BIAS_MODELS.register("quadratic")
class Quadratic(BiasModel):
    param_specs = (ParamSpec("b1", 1.0, 0.0, 5.0, 0.2), ParamSpec("b2", 0.0, -2.0, 2.0, 0.2))
    def __call__(self, rho, p, floor):
        d = rho - 1.0
        return 1.0 + p["b1"] * d + p["b2"] * d**2
```

Registries: `transfer.TRANSFER_FUNCTIONS`, `forward.FORWARD_MODELS`,
`bias.BIAS_MODELS`, `likelihood.LIKELIHOODS`, `data.SELECTIONS`,
`samplers.INTEGRATORS`, `blocks.BLOCKS`. The samplers are independent of the
cosmology code: `samplers.run_gibbs` drives any list of `Block`s over a dict
state, and `hmc_block` / `slice_block` wrap any log density (see
`tests/test_gibbs.py` for a non-cosmological example).

## What is taken from Manticore / BORG

| Technique | Here |
|---|---|
| HMC on white-noise ICs, slice sampling of bias parameters conditioned on the density (App. A.2) | `blocks.py`; the forward model runs once per slice sweep, catalogues update in parallel |
| Randomised step size and trajectory length (Jasche & Wandelt 2013) | `sampler.hmc.step_jitter`, `n_steps_min..n_steps` |
| Sub-catalogues with independent bias parameters sharing one field | `galaxies.n_catalogues` |
| λ = n̄ R B/⟨B⟩ with ergodic normalisation (Eqs. 19–21) | `FieldModel.expected_counts` |
| Sigmoid-truncated double power-law bias (Eqs. 13–16) | `bias: stdp` |
| Generalized Poisson likelihood, b = b_α ρ^b_β (Eqs. 22–25) | `likelihood: generalized_poisson` |
| Power-spectrum prior on the white noise, strength λ_ps (Eqs. 2, 6) | `lambda_ps` hyper-parameter |
| Hyper-parameters ramped during burn-in, then frozen (App. D) | `sampler.schedules` (also likelihood tempering) |
| Several chains from random low-amplitude starts, R-hat on the likelihood, autocorrelation/ESS, white-noise Gaussianity moments | `diagnostics.py`, `summary.json` |

Not implemented: the moment priors (they need a calibration suite of ΛCDM
simulations), light-cone effects, resimulation. COLA gravity is available as
`forward.model: tcola`.
The paper does not state its HMC settings or bias-parameter priors; the HMC
defaults here (leapfrog, identity mass matrix, dual-averaging step size) are
this code's choices.

## Conventions

- Lengths in Mpc/h, k in h/Mpc, P in (Mpc/h)³. `rfftn`/`irfftn` unnormalised.
- The linear field, density mesh and response are sampled at nodes `j * cell`.
  Particles start at `(i + offset) * cell` with `offset` chosen so that none
  sits on a CIC kink (0.5 for Zel'dovich, `0.5 / force_sampling` for tCOLA;
  `forward.lattice_offset` overrides it). `n_mesh` must divide `n_grid`.
- Slice sampling enforces each parameter's `[lower, upper]`; priors are
  uniform or normal within those bounds.
- Samples are only collected after warm-up, when step sizes, slice widths and
  schedules are frozen.

## Known pitfall

With the `stdp` bias left fully free the posterior has a mirror mode: the
field with the opposite sign together with a bias that *decreases* with
density. Chains started from a low-amplitude field can land in it.
`configs/manticore_like.yaml` bounds `alpha` and `beta` so that B(ρ) stays
monotonic; keep such bounds (or an informative prior) when using `stdp`.
