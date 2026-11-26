"""Read/write field files in the GENGARS C binary format.

File layout (native endianness; verified: int=4 bytes, double=8 bytes)::

    int32    ng                          grid points per side
    float64  dk                          fundamental wavenumber 2*pi/L
    float64  data[ng*ng*(ng+2)]          FFTW r2c "padded" half-complex layout

The trailing ``data`` block is a real array of shape ``(ng, ng, ng+2)`` that is
reinterpreted as a complex array of shape ``(ng, ng, ng//2+1)`` -- exactly the
layout produced by ``jnp.fft.rfftn`` over a real ``(ng, ng, ng)`` field. The last
real axis has length ``ng+2 = 2*(ng//2+1)`` (the FFTW padding for even ``ng``).

Written by ``write_phi`` in ``gengars/src/generate_phiG_utils.c`` and by the MPI
``write_phi`` in ``gengars/src/generate_phiNG_utils.c``; read by ``read_phi`` in
``gengars/src/2lpt_utils.c``.

These helpers are for cross-validation against the C code: read a C-generated
Phi^G to feed our generator the *same* Gaussian field, and diff the Phi^NG output.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp


def read_c_field(path: str):
    """Read a GENGARS C field file.

    Returns
    -------
    ng : int
        Grid points per side.
    dk : float
        Fundamental wavenumber 2*pi/L.
    field_k : jax.Array
        Complex array of shape ``(ng, ng, ng//2+1)`` in ``jnp.fft.rfftn`` layout.
    """
    with open(path, "rb") as f:
        ng = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        dk = float(np.fromfile(f, dtype=np.float64, count=1)[0])
        n_real = ng * ng * (ng + 2)
        data = np.fromfile(f, dtype=np.float64, count=n_real)
    if data.size != n_real:
        raise ValueError(
            f"{path}: expected {n_real} doubles after header, got {data.size}"
        )
    real = np.ascontiguousarray(data.reshape(ng, ng, ng + 2))
    field_k = real.view(np.complex128).reshape(ng, ng, ng // 2 + 1)
    return ng, dk, jnp.asarray(field_k)


def write_c_field(path: str, field_k, dk: float) -> None:
    """Write a complex k-space field to a GENGARS C field file.

    Parameters
    ----------
    path : str
        Output path.
    field_k : array_like
        Complex array of shape ``(ng, ng, ng//2+1)`` (``jnp.fft.rfftn`` layout).
    dk : float
        Fundamental wavenumber 2*pi/L (stored in the header).
    """
    arr = np.ascontiguousarray(np.asarray(field_k, dtype=np.complex128))
    ng = arr.shape[0]
    if arr.shape != (ng, ng, ng // 2 + 1):
        raise ValueError(
            f"expected field of shape (ng, ng, ng//2+1), got {arr.shape}"
        )
    real = arr.view(np.float64).reshape(ng, ng, ng + 2)
    with open(path, "wb") as f:
        np.array([ng], dtype=np.int32).tofile(f)
        np.array([dk], dtype=np.float64).tofile(f)
        real.tofile(f)
