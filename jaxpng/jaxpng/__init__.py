"""jaxpng: JAX rewrite of the GENGARS PNG field generator (Phi^G + Phi^NG).

Importing the package enables JAX 64-bit mode (see :mod:`jaxpng.config`).
"""
from . import (
    api,
    bispectra,
    config,
    cosmology,
    gaussian,
    grids,
    io_c,
    measure,
    phing,
)
from .config import Box, complex_dtype, real_dtype, set_precision

__all__ = [
    "api",
    "bispectra",
    "config",
    "cosmology",
    "gaussian",
    "grids",
    "io_c",
    "measure",
    "phing",
    "Box",
    "set_precision",
    "real_dtype",
    "complex_dtype",
]
__version__ = "0.0.1"
