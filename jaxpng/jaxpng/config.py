"""Global configuration for jaxpng.

Importing this module enables JAX 64-bit mode by default: the Schwinger kernel
e^{-t/P(k)} spans a wide dynamic range, so float64 is the safe default for
getting the bispectrum normalization right. A float32 switch is available for
GPU speed once results are validated (see ``set_precision``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import jax

# Enable 64-bit by default. Must happen before any JAX array is created.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402  (after x64 is enabled)

_PRECISION = "float64"


def set_precision(precision: str) -> None:
    """Set the default real/complex dtypes used by jaxpng.

    Parameters
    ----------
    precision : {"float64", "float32"}
        "float64" (default) keeps full fidelity for validation; "float32" is
        GPU-friendly but should only be used after checking accuracy against
        the analytic targets.
    """
    global _PRECISION
    if precision not in ("float64", "float32"):
        raise ValueError("precision must be 'float64' or 'float32'")
    if precision == "float64":
        jax.config.update("jax_enable_x64", True)
    _PRECISION = precision


def real_dtype():
    """Default real dtype (jnp.float64 or jnp.float32)."""
    return jnp.float64 if _PRECISION == "float64" else jnp.float32


def complex_dtype():
    """Default complex dtype (jnp.complex128 or jnp.complex64)."""
    return jnp.complex128 if _PRECISION == "float64" else jnp.complex64


@dataclass(frozen=True)
class Box:
    """Cubic simulation-box geometry.

    Parameters
    ----------
    N : int
        Number of grid points per side.
    L : float
        Box side length, in Mpc/h (matches the GENGARS ``boxsize`` parameter).
    """

    N: int
    L: float

    @property
    def dx(self) -> float:
        """Real-space cell size, L / N."""
        return self.L / self.N

    @property
    def dk(self) -> float:
        """Fundamental wavenumber, 2*pi / L (the GENGARS ``dk``)."""
        return 2.0 * math.pi / self.L

    @property
    def kny(self) -> float:
        """Nyquist wavenumber, pi * N / L."""
        return math.pi * self.N / self.L

    @property
    def volume(self) -> float:
        """Box volume, L**3."""
        return self.L ** 3

    @property
    def rshape(self) -> tuple[int, int, int]:
        """Real-space array shape (N, N, N)."""
        return (self.N, self.N, self.N)

    @property
    def kshape(self) -> tuple[int, int, int]:
        """rfftn k-space array shape (N, N, N//2 + 1)."""
        return (self.N, self.N, self.N // 2 + 1)
