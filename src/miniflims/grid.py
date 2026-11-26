"""Periodic cubic grid and Fourier conventions.

Discrete transforms are numpy-style unnormalised ``rfftn`` / ``irfftn``. A
field with continuum power spectrum P(k) then has
``<|delta_hat_k|^2> = P(k) N^6 / V``, and unit white noise has ``N^3``.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Grid(NamedTuple):
    n: int  # cells per dimension
    box: float  # Mpc/h

    @property
    def cell(self) -> float:
        return self.box / self.n

    @property
    def volume(self) -> float:
        return self.box**3

    @property
    def k_fundamental(self) -> float:
        return 2.0 * np.pi / self.box

    @property
    def k_nyquist(self) -> float:
        return np.pi * self.n / self.box

    def kvec(self, zero_nyquist: bool = False):
        """(kx, ky, kz) broadcastable to the rfftn layout (n, n, n//2+1).

        ``zero_nyquist`` zeroes the Nyquist plane of each axis, as needed for
        odd (derivative) kernels to map real fields to real fields.
        """
        kf = self.k_fundamental
        full = np.fft.fftfreq(self.n, d=1.0 / self.n)
        half = np.fft.rfftfreq(self.n, d=1.0 / self.n)
        if zero_nyquist and self.n % 2 == 0:
            full = np.where(np.abs(full) == self.n // 2, 0.0, full)
            half = np.where(np.abs(half) == self.n // 2, 0.0, half)
        return (
            kf * full[:, None, None],
            kf * full[None, :, None],
            kf * half[None, None, :],
        )

    def kmag(self):
        kx, ky, kz = self.kvec()
        return np.sqrt(kx**2 + ky**2 + kz**2)

    def rfft_weights(self):
        """Multiplicity of each rfftn mode in the full FFT (1 or 2)."""
        w = np.full(self.n // 2 + 1, 2.0)
        w[0] = 1.0
        if self.n % 2 == 0:
            w[-1] = 1.0
        return np.broadcast_to(w[None, None, :], (self.n, self.n, self.n // 2 + 1))

    def coords(self):
        """Cell positions (x, y, z), broadcastable to (n, n, n), in Mpc/h."""
        x = np.arange(self.n) * self.cell
        return x[:, None, None], x[None, :, None], x[None, None, :]
