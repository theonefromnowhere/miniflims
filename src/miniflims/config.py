"""Typed, validated configuration.

The whole pipeline is driven by one nested, frozen dataclass tree. It is built
from defaults <- YAML file <- dotted overrides (``sampler.hmc.n_steps=30``).
Unknown keys and wrong types raise immediately instead of being ignored.
"""
from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CosmologyConfig:
    # Defaults: DES Y3 3x2pt + All Ext. LCDM, as used by Manticore.
    h: float = 0.681
    omega_m: float = 0.306
    omega_b: float = 0.0486
    n_s: float = 0.967
    sigma8: float = 0.807
    t_cmb: float = 2.7255


@dataclass(frozen=True)
class PowerConfig:
    transfer: str = "eisenstein_hu"  # see transfer.TRANSFER_FUNCTIONS
    # log-k grid for the sigma8 normalisation integral [h/Mpc]
    k_min: float = 1e-5
    k_max: float = 1e2
    n_k: int = 2048


@dataclass(frozen=True)
class GridConfig:
    n_grid: int = 32  # initial-condition grid / particles per dimension
    box_size: float = 500.0  # Mpc/h
    n_mesh: Optional[int] = None  # density mesh per dimension (default n_grid)


@dataclass(frozen=True)
class RSDConfig:
    enabled: bool = False
    los: str = "z"  # "x" | "y" | "z" (plane-parallel) or "radial"
    observer: Optional[tuple[float, ...]] = None  # Mpc/h; default box centre


@dataclass(frozen=True)
class ColaConfig:
    """Settings of the tCOLA particle-mesh model (forward.model = tcola)."""

    n_steps: int = 10  # leapfrog (drift-kick-drift) steps, uniform in a
    z_start: float = 19.0  # particles are initialised with Zel'dovich here
    force_sampling: int = 2  # force mesh = force_sampling * n_grid per dimension
    # Memory-lean reverse mode (as in BorgJaxPM): analytic adjoints for the
    # CIC operators and a force adjoint that stores only particle positions
    # and rebuilds the meshes. False = plain autodiff (same gradient).
    low_memory: bool = True


@dataclass(frozen=True)
class ForwardConfig:
    model: str = "zeldovich"  # see forward.FORWARD_MODELS
    redshift: float = 0.0
    rsd: RSDConfig = field(default_factory=RSDConfig)
    # Offset of the Lagrangian particle lattice from the mesh nodes, in
    # initial-grid cells. null = automatic: half a cell of the finest mesh the
    # particles are painted on (0.5 for zeldovich, 0.5 / cola.force_sampling
    # for tcola), which keeps every particle off the CIC kink at zero
    # displacement. 0 reproduces the BORG / JaxPM layout.
    lattice_offset: Optional[float] = None
    cola: ColaConfig = field(default_factory=ColaConfig)


@dataclass(frozen=True)
class ParamOverride:
    """Per-parameter overrides of a component's default ParamSpec."""

    init: Optional[float] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    width: Optional[float] = None  # initial slice-sampler bracket width
    sample: Optional[bool] = None  # False -> held fixed at ``init``
    prior: Optional[str] = None  # "uniform" | "normal" (truncated to bounds)
    prior_mean: Optional[float] = None
    prior_std: Optional[float] = None


@dataclass(frozen=True)
class NonLocalBiasConfig:
    """Non-local bias terms (see nonlocal_bias.py). Their coefficients (b_phi,
    b_lap) are nuisance parameters, set in galaxies.bias.params and sampled
    with the local bias parameters."""

    # "png": b_phi f_NL (k/k_star)^alpha Phi_G, the scale-dependent PNG bias
    # (needs png.enabled); "laplacian": b_lap nabla^2 delta / k_star^2
    terms: tuple[str, ...] = ()
    form: str = "exp"  # "exp": B exp(sum b_i O_i); "linear": max(B + sum b_i O_i, floor)
    k_star: float = 0.1  # h/Mpc, normalisation scale of the operators
    png_alpha: Optional[float] = None  # squeezed-limit scaling of the png term; null = from png.template


@dataclass(frozen=True)
class BiasConfig:
    model: str = "power_law"  # see bias.BIAS_MODELS
    normalize: bool = True  # ergodic normalisation B -> B/<B>
    # parameters of the local model and of the non-local terms (b_phi, b_lap)
    params: dict[str, ParamOverride] = field(default_factory=dict)
    non_local: NonLocalBiasConfig = field(default_factory=NonLocalBiasConfig)


@dataclass(frozen=True)
class LikelihoodConfig:
    model: str = "poisson"  # see likelihood.LIKELIHOODS
    params: dict[str, ParamOverride] = field(default_factory=dict)


@dataclass(frozen=True)
class GalaxyConfig:
    n_catalogues: int = 1  # independent sub-catalogues sharing the matter field
    nbar: ParamOverride = field(default_factory=ParamOverride)
    bias: BiasConfig = field(default_factory=BiasConfig)
    likelihood: LikelihoodConfig = field(default_factory=LikelihoodConfig)
    density_floor: float = 1e-3  # added to rho before logs/powers


@dataclass(frozen=True)
class PNGConfig:
    """Primordial non-Gaussianity through jaxpng (see png.py)."""

    enabled: bool = False
    # local | equilateral | orthogonal | feature | equilateral_feature
    template: str = "local"
    # f_NL (all), omega and phase (oscillatory templates); same fields as
    # galaxies.*.params. Defaults: f_NL sampled in [-1e4, 1e4]; omega, phase fixed.
    params: dict[str, ParamOverride] = field(default_factory=dict)
    # Schwinger t-integral: "log" = trapezoid in ln t with n_t nodes (exponential
    # convergence: 32 nodes -> 3e-5 error); "jaxpng" = jaxpng's trapezoid in t
    # with per_efold nodes per e-fold (20 -> 361 nodes, 4e-4 error)
    quadrature: str = "log"
    n_t: int = 32
    eps: Optional[float] = None  # t-range truncation; null = 1e-5 (log), 1e-3 (jaxpng)
    per_efold: float = 20.0
    remat: bool = True  # checkpoint the Schwinger scan: O(N^3) gradient memory
    local_exact: bool = False  # local only: Phi_G^2 - <Phi_G^2> instead of the kernel
    jaxpng_path: Optional[str] = None  # project dir containing the jaxpng package


@dataclass(frozen=True)
class BispectrumPriorConfig:
    """Gaussianity prior on the white noise: its binned bispectrum, divided by
    the Gaussian scatter, is pulled towards zero (see FieldModel)."""

    lambda_bs: float = 0.0  # strength; 0 = off. Schedulable as hyper-parameter "lambda_bs"
    n_bins: int = 8  # |k| bins between k_F and k_max (equal width)
    k_max: Optional[float] = None  # h/Mpc; null = Nyquist
    n_mc: int = 256  # Gaussian white-noise realizations used to calibrate the scatter
    seed: int = 12345


@dataclass(frozen=True)
class FieldPriorConfig:
    # Manticore power-spectrum prior on the white noise (Eqs. 2, 6): strength
    # lambda_ps (Manticore: 100), 0 = off. Schedulable as hyper-parameter "lambda_ps".
    lambda_ps: float = 0.0
    ps_prior_bins: int = 12
    bispectrum: BispectrumPriorConfig = field(default_factory=BispectrumPriorConfig)


@dataclass(frozen=True)
class SelectionConfig:
    type: str = "uniform"  # see data.SELECTIONS
    r0: float = 150.0  # Mpc/h, radial scale
    power: float = 2.0  # R = exp(-(r/r0)^power)
    r_max: Optional[float] = None  # hard cut
    observer: Optional[tuple[float, ...]] = None
    path: Optional[str] = None  # for type="file" (.npy)


@dataclass(frozen=True)
class MockConfig:
    seed: int = 1
    # truth for nuisance parameters: name -> value or list (one per catalogue).
    # Parameters not listed use the parameter's ``init``.
    truth: dict[str, Any] = field(default_factory=dict)
    # one selection for all catalogues, or one per catalogue
    selections: tuple[SelectionConfig, ...] = (SelectionConfig(),)


@dataclass(frozen=True)
class DataConfig:
    source: str = "mock"  # "mock" | "file"
    path: Optional[str] = None  # .npz with ``counts`` and ``response``
    mock: MockConfig = field(default_factory=MockConfig)


@dataclass(frozen=True)
class HMCConfig:
    integrator: str = "leapfrog"  # see samplers.integrators.INTEGRATORS
    step_size: float = 0.02  # initial (or fixed) step size
    n_steps: int = 20  # max integration steps per trajectory
    n_steps_min: Optional[int] = None  # n ~ U{n_steps_min..n_steps}; None = fixed
    step_jitter: float = 0.2  # eps ~ U[eps (1 - jitter), eps]
    adapt_step_size: bool = True  # dual averaging during warm-up
    target_accept: float = 0.7
    n_per_sweep: int = 1  # HMC transitions per Gibbs sweep


@dataclass(frozen=True)
class SliceConfig:
    max_stepout: int = 8
    max_shrink: int = 64
    adapt_width: bool = True  # tune bracket widths during warm-up
    n_per_sweep: int = 1  # coordinate sweeps per Gibbs sweep


@dataclass(frozen=True)
class ScheduleConfig:
    """Hyper-parameter ramp from ``start`` to ``end`` over ``n_steps`` sweeps."""

    start: float = 1.0
    end: float = 1.0
    n_steps: int = 0
    kind: str = "linear"  # "linear" | "geometric"


@dataclass(frozen=True)
class GateConfig:
    """When a Gibbs block starts acting (``sampler.gates.<block>``). Checked
    every sweep for each chain separately; until it holds, the block leaves
    its variables (at their ``init``) and its adaptation unchanged."""

    # act only once the chain's white-noise variance <x^2> reaches this, e.g.
    # 0.95 to keep the PNG parameters fixed while the field grows from a
    # low-amplitude start (sampler.init.field_scale)
    min_field_variance: Optional[float] = None


@dataclass(frozen=True)
class InitConfig:
    field_scale: float = 0.1  # initial white noise = field_scale * N(0,1)
    nuisance_jitter: float = 0.0  # fraction of prior range to scatter starts


@dataclass(frozen=True)
class CheckpointConfig:
    """Restart files in ``output.dir`` (ignored when it is null)."""

    every: int = 20  # sweeps between checkpoints; 0 = only at the end / on a signal
    resume: bool = True  # continue from output.dir/checkpoint.npz if it exists
    keep_previous: bool = True  # keep the last-but-one checkpoint as checkpoint.prev.npz


@dataclass(frozen=True)
class SamplerConfig:
    seed: int = 0
    n_chains: int = 2
    n_warmup: int = 300
    n_samples: int = 300
    thin: int = 5  # keep every ``thin``-th field sample after warm-up
    blocks: tuple[str, ...] = ("field", "nuisance")  # Gibbs order
    hmc: HMCConfig = field(default_factory=HMCConfig)
    slice: SliceConfig = field(default_factory=SliceConfig)
    init: InitConfig = field(default_factory=InitConfig)
    # name -> schedule; known names: model.HYPER_DEFAULTS
    schedules: dict[str, ScheduleConfig] = field(default_factory=dict)
    # block name -> condition for that block to act (GateConfig)
    gates: dict[str, GateConfig] = field(default_factory=dict)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


@dataclass(frozen=True)
class OutputConfig:
    dir: Optional[str] = None  # None -> nothing written
    # white-noise samples: with a ``dir`` they are written to dir/samples/
    # as they are drawn (one .npy per kept sweep); otherwise kept in memory
    save_fields: bool = True
    log_every: int = 50  # 0 = silent
    progress: bool = True  # tqdm progress bar on stderr
    # seconds between bar refreshes; null = 0.5 on a terminal, 30 otherwise
    # (batch logs)
    progress_interval: Optional[float] = None
    n_k_bins: int = 12  # for power-spectrum / cross-correlation summaries


@dataclass(frozen=True)
class Config:
    precision: str = "float64"  # "float32" | "float64"
    cosmology: CosmologyConfig = field(default_factory=CosmologyConfig)
    power: PowerConfig = field(default_factory=PowerConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    forward: ForwardConfig = field(default_factory=ForwardConfig)
    galaxies: GalaxyConfig = field(default_factory=GalaxyConfig)
    png: PNGConfig = field(default_factory=PNGConfig)
    prior: FieldPriorConfig = field(default_factory=FieldPriorConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# --------------------------------------------------------------------------- #
# dict -> dataclass with validation
# --------------------------------------------------------------------------- #
class ConfigError(ValueError):
    pass


def _convert(tp: Any, value: Any, path: str) -> Any:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if tp is Any:
        return value
    if origin in (Union, types.UnionType):
        if value is None:
            if type(None) in args:
                return None
            raise ConfigError(f"{path}: may not be null")
        errors = []
        for a in args:
            if a is type(None):
                continue
            try:
                return _convert(a, value, path)
            except ConfigError as e:
                errors.append(str(e))
        raise ConfigError("; ".join(errors))
    if value is None:
        raise ConfigError(f"{path}: may not be null")
    if dataclasses.is_dataclass(tp):
        if isinstance(value, tp):
            return value
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {value!r}")
        return _from_dict(tp, value, path)
    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {value!r}")
        return tuple(_convert(args[0], v, f"{path}[{i}]") for i, v in enumerate(value))
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {value!r}")
        return {str(k): _convert(args[1], v, f"{path}.{k}") for k, v in value.items()}
    if tp is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{path}: expected a bool, got {value!r}")
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ConfigError(f"{path}: expected an int, got {value!r}")
        try:
            f = float(value)
        except ValueError:
            raise ConfigError(f"{path}: expected an int, got {value!r}") from None
        if f != int(f):
            raise ConfigError(f"{path}: expected an int, got {value!r}")
        return int(f)
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ConfigError(f"{path}: expected a float, got {value!r}")
        try:  # YAML reads "1e-3" as a string
            return float(value)
        except ValueError:
            raise ConfigError(f"{path}: expected a float, got {value!r}") from None
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value
    raise ConfigError(f"{path}: unsupported config type {tp}")


def _from_dict(cls: type, d: dict, path: str = "") -> Any:
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - names
    if unknown:
        where = path or "<root>"
        raise ConfigError(
            f"{where}: unknown key(s) {sorted(unknown)}; valid: {sorted(names)}"
        )
    kwargs = {
        k: _convert(hints[k], v, f"{path}.{k}" if path else k) for k, v in d.items()
    }
    return cls(**kwargs)


def _plain(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def _deep_update(base: dict, upd: dict) -> dict:
    out = dict(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _set_dotted(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        nxt = d.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            d[k] = nxt
        d = nxt
    d[keys[-1]] = value


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def to_dict(cfg: Config) -> dict:
    """Plain nested dict (YAML/JSON serialisable)."""
    return _plain(cfg)


def from_dict(d: dict | None = None) -> Config:
    return _from_dict(Config, d or {})


def parse_overrides(overrides: typing.Iterable[str]) -> dict:
    """``["a.b=1", "c=[1, 2]"]`` -> nested dict; values are parsed as YAML."""
    out: dict = {}
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override '{item}' is not of the form key=value")
        key, raw = item.split("=", 1)
        _set_dotted(out, key.strip(), yaml.safe_load(raw))
    return out


def load_config(
    path: str | Path | None = None,
    overrides: typing.Iterable[str] | dict | None = None,
) -> Config:
    """defaults <- YAML file <- overrides (dotted strings or a nested dict)."""
    d: dict = {}
    if path is not None:
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        if not isinstance(d, dict):
            raise ConfigError(f"{path}: top level must be a mapping")
    if overrides:
        upd = overrides if isinstance(overrides, dict) else parse_overrides(overrides)
        d = _deep_update(d, upd)
    return from_dict(d)


def update(cfg: Config, overrides: typing.Iterable[str] | dict) -> Config:
    """Return a new Config with overrides applied."""
    upd = overrides if isinstance(overrides, dict) else parse_overrides(overrides)
    return from_dict(_deep_update(to_dict(cfg), upd))


def dump_yaml(cfg: Config) -> str:
    return yaml.safe_dump(to_dict(cfg), sort_keys=False)
