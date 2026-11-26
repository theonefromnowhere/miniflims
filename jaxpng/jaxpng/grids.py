"""3D Fourier-space grids.

Provides the physical wavenumber components (h/Mpc) on the ``jnp.fft.rfftn``
half-grid and the internal kernel power spectrum used by the Schwinger machinery.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp


def k_grid(box):
    """Physical wavenumber components and |k| on the rfftn half-grid.

    Returns ``(KX, KY, KZ, kmag)`` where KX,KY,KZ broadcast to ``box.kshape``
    (N, N, N//2+1) and kmag is the magnitude (h/Mpc). The fundamental is
    ``box.dk = 2*pi/L`` and the Nyquist is ``box.kny = pi*N/L``.
    """
    N, dx = box.N, box.dx
    kx = jnp.fft.fftfreq(N, d=dx) * 2.0 * np.pi  # (N,) signed, -kNy at Nyquist
    kz = jnp.fft.rfftfreq(N, d=dx) * 2.0 * np.pi  # (N//2+1,) >= 0
    KX = kx[:, None, None]
    KY = kx[None, :, None]
    KZ = kz[None, None, :]
    kmag = jnp.sqrt(KX * KX + KY * KY + KZ * KZ)
    return KX, KY, KZ, kmag


def P_kernel(kmag, n_s):
    """Internal kernel power spectrum P(k) = k^(n_s-4), amplitude dropped.

    Used by the Schwinger-parameterized PNG kernel (Phase 2): the absolute
    amplitude of P_Phi cancels in the reduced-bispectrum kernel, so only the
    k^(n_s-4) shape matters. P(0) is set to 1 (as in gengars/generate_phiNG.c).
    """
    # double where: k^(n_s-4) is inf at k = 0, and even in the discarded branch
    # its derivative (e.g. with respect to n_s) would turn gradients into NaN
    ks = jnp.where(kmag > 0, kmag, 1.0)
    return jnp.where(kmag > 0, ks ** (n_s - 4.0), 1.0)


def rk(kmag, n_s):
    """Bispectrum-factor argument rk = P_kernel(k)^(-1/3) (the C code's `keff`)."""
    return P_kernel(kmag, n_s) ** (-1.0 / 3.0)
