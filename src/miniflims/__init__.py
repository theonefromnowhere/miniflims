"""miniflims: a small, modular field-level inference pipeline in JAX.

Eisenstein-Hu initial power spectrum, Zel'dovich forward model, galaxy bias
and count likelihoods, sampled with HMC (initial conditions) inside a Gibbs
scheme with slice sampling (nuisance parameters), after BORG / Manticore.
"""
from .config import Config, load_config, to_dict, update
from .model import FieldModel
from .pipeline import RunResult, run

__all__ = ["Config", "load_config", "to_dict", "update", "FieldModel", "RunResult", "run"]
__version__ = "0.1.0"
