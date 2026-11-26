"""3D power spectrum and bispectrum estimators, in pure JAX.

Reimplemented from Pylians3 (``Pk_library``) so that measurement depends only on
JAX: it runs in the ``jaxpng`` env, avoids the macOS multiple-OpenMP crash that
makes in-process Pylians unreliable here, and is ``jit``/``grad``/``vmap``-able.

Conventions match Pylians exactly:
- forward FFT is unnormalized (``jnp.fft.rfftn`` == FFTW forward); the inverse
  ``jnp.fft.irfftn`` divides by N^3 (== pyfftw ``normalise_idft=True``);
- wavenumbers are in units of the fundamental kF = 2*pi/BoxSize;
- integer-k spherical shells: a mode falls in bin ``floor(|k|/kF)``;
- units: P(k) = <|delta_k|^2> * (BoxSize/N^2)^3 per shell;
- the Nyquist component carries +N/2 (Pylians convention); irrelevant for |k|.

`field` may be a real ``(N,N,N)`` array (forward-FFT'd here) or an rfftn k-space
``(N,N,N//2+1)`` array (used directly).

Cross-checked against Pylians (see ``Pk`` in Pk_library.pyx, ``Bk`` in
bispectrum_library.pyx) and against the analytic white-noise P(k) = (BoxSize/N)^3.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

EQUILATERAL_THETA = 2.0 * np.pi / 3.0  # angle between k1,k2 giving k1=k2=k3

_MAS_INDEX = {"None": 0, "NGP": 1, "CIC": 2, "TSC": 3, "PCS": 4}


def _safe(x):
    """Replace zeros by 1 in a denominator (empty shell/bin -> 0, finite grad)."""
    return jnp.where(x == 0, 1.0, x)


def _to_delta_k(field):
    """Return delta_k in rfftn layout (forward-FFT a real field if needed)."""
    f = jnp.asarray(field)
    if jnp.iscomplexobj(f):
        return f
    return jnp.fft.rfftn(f)


def _integer_k(N):
    """Signed integer wavenumbers on the rfftn half-grid, in kF units.

    Returns kx, ky (full, signed, +N/2 at Nyquist), kz (rfft axis, 0..N/2), and
    |k| broadcast to shape (N, N, N//2+1). Matches Pylians' ``kx = kxx-dims if
    kxx>middle else kxx`` convention.
    """
    m = N // 2
    ax = jnp.arange(N)
    ax = jnp.where(ax > m, ax - N, ax)  # +m at Nyquist (not -m), per Pylians
    kx = ax[:, None, None]
    ky = ax[None, :, None]
    kz = jnp.arange(m + 1)[None, None, :]
    kmag = jnp.sqrt((kx * kx + ky * ky + kz * kz).astype(jnp.float64))
    return kx, ky, kz, kmag


def _mas_factor(kx, ky, kz, N, MAS):
    """MAS deconvolution factor (1.0 for MAS='None')."""
    idx = _MAS_INDEX.get(MAS, 0)
    if idx == 0:
        return 1.0
    pre = np.pi / N

    def corr(k):
        x = pre * k
        return jnp.where(x == 0.0, 1.0, (x / jnp.sin(x)) ** idx)

    return corr(kx) * corr(ky) * corr(kz)


def _independent_mode_weight(kx, ky, kz, N):
    """1.0 for Pylians' independent modes, 0.0 for the redundant half.

    On the kz=0 and kz=N/2 planes the rfftn grid holds conjugate pairs; Pylians
    keeps only one of each (``if kx<0: continue`` etc.). Counting both does not
    change a shell's *mean* power (pairs share |k| and |delta|^2), but applying
    this weight reproduces Pylians' Nmodes and bin-averaged values exactly.
    """
    m = N // 2
    special = (kz == 0) | (kz == m)
    excl = special & ((kx < 0) | (((kx == 0) | (kx == m)) & (ky < 0)))
    return jnp.where(excl, 0.0, 1.0)


def power_spectrum(field, BoxSize, MAS="None"):
    """3D power spectrum monopole. Returns ``(k, Pk)`` (DC bin discarded)."""
    delta_k = _to_delta_k(field)
    N = delta_k.shape[0]
    kF = 2.0 * np.pi / BoxSize
    m = N // 2

    kx, ky, kz, kmag = _integer_k(N)
    delta_k = delta_k * _mas_factor(kx, ky, kz, N, MAS)
    delta2 = delta_k.real ** 2 + delta_k.imag ** 2
    w = _independent_mode_weight(kx, ky, kz, N)

    kindex = jnp.floor(kmag).astype(jnp.int32).ravel()
    nbins = int(np.floor(np.sqrt(3.0) * m)) + 1  # kmax+1, == Pylians
    Pk_sum = jnp.bincount(kindex, weights=(delta2 * w).ravel(), length=nbins)
    k_sum = jnp.bincount(kindex, weights=(kmag * w).ravel(), length=nbins)
    Nmodes = jnp.bincount(kindex, weights=w.ravel(), length=nbins)

    Nm = _safe(Nmodes[1:])  # discard DC bin; guard empty bins
    k = (k_sum[1:] / Nm) * kF
    Pk = (Pk_sum[1:] / Nm) * (BoxSize / N ** 2) ** 3
    return k, Pk


def bispectrum(field, BoxSize, k1, k2, theta, MAS="None"):
    """3D bispectrum for triangles (k1, k2, theta), Pylians-style FFT estimator.

    ``k1, k2`` are physical wavenumbers; ``theta`` (scalar or array, radians) is
    the angle between the k1 and k2 vectors, so the third side is
    ``k3 = sqrt((k2 sin t)^2 + (k2 cos t + k1)^2)`` and equilateral => 2*pi/3.

    Returns a dict: ``B``, ``Q`` (reduced), ``k`` (= [k1, k2, *k3]), ``Pk``
    (power in each of those shells), ``triangles`` (number of triangles per
    theta divided by N^6, i.e. sum_x I1 I2 I3 with the irfftn normalisation;
    multiply by N^6 for the count).
    """
    delta_k = _to_delta_k(field)
    N = delta_k.shape[0]
    kF = 2.0 * np.pi / BoxSize

    kx, ky, kz, kmag = _integer_k(N)
    delta_k = delta_k * _mas_factor(kx, ky, kz, N, MAS)

    theta = jnp.atleast_1d(jnp.asarray(theta, dtype=jnp.float64))
    k3 = jnp.sqrt((k2 * jnp.sin(theta)) ** 2 + (k2 * jnp.cos(theta) + k1) ** 2)

    pk_units = (BoxSize / N ** 2) ** 3
    b_units = (BoxSize ** 2 / N ** 3) ** 3

    def shell_fields(ki):
        # spherical shell [ki-kF, ki+kF) in kF units, as in Pylians
        lo = (ki - kF) / kF
        hi = (ki + kF) / kF
        mask = (kmag >= lo) & (kmag < hi)
        dk = jnp.where(mask, delta_k, 0.0 + 0.0j)
        ik = jnp.where(mask, 1.0 + 0.0j, 0.0 + 0.0j)
        d = jnp.fft.irfftn(dk, s=(N, N, N))
        ind = jnp.fft.irfftn(ik, s=(N, N, N))
        return d, ind

    d1, I1 = shell_fields(k1)
    d2, I2 = shell_fields(k2)
    P1 = jnp.sum(d1 * d1) / _safe(jnp.sum(I1 * I1)) * pk_units
    P2 = jnp.sum(d2 * d2) / _safe(jnp.sum(I2 * I2)) * pk_units

    def per_theta(k3i):
        d3, I3 = shell_fields(k3i)
        P3 = jnp.sum(d3 * d3) / _safe(jnp.sum(I3 * I3)) * pk_units
        tri = jnp.sum(I1 * I2 * I3)
        B = jnp.sum(d1 * d2 * d3) / _safe(tri) * b_units
        Q = B / _safe(P1 * P2 + P1 * P3 + P2 * P3)
        return B, Q, P3, tri

    B, Q, P3, tri = jax.vmap(per_theta)(k3)
    return {
        "B": B,
        "Q": Q,
        "k": jnp.concatenate([jnp.array([k1, k2]), k3]),
        "Pk": jnp.concatenate([jnp.array([P1, P2]), P3]),
        "triangles": tri,
    }
