"""Chain and field diagnostics (numpy): power spectra, cross-correlations,
split R-hat, effective sample size, Gaussianity moments."""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Fields
# --------------------------------------------------------------------------- #
def _k_bins(n: int, box: float, n_bins: int):
    kf = 2 * np.pi / box
    f = np.fft.fftfreq(n, d=1.0 / n) * kf
    kmag = np.sqrt(f[:, None, None] ** 2 + f[None, :, None] ** 2 + f[None, None, :] ** 2)
    edges = np.linspace(kf, np.pi * n / box, n_bins + 1)
    idx = np.digitize(kmag.ravel(), edges) - 1
    valid = (idx >= 0) & (idx < n_bins)
    return kmag.ravel(), idx, valid


def cross_power(a, b, box: float, n_bins: int = 12):
    """Binned cross power spectrum of two fields on the same (n, n, n) grid.

    Returns (k_mean, P_ab, n_modes); bins run from the fundamental to the
    Nyquist frequency. Units follow ``box`` (Mpc/h -> (Mpc/h)^3).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = a.shape[0]
    kmag, idx, valid = _k_bins(n, box, n_bins)
    ak, bk = np.fft.fftn(a).ravel(), np.fft.fftn(b).ravel()
    power = (ak * np.conj(bk)).real * box**3 / n**6
    counts = np.bincount(idx[valid], minlength=n_bins)
    safe = np.maximum(counts, 1)
    pk = np.bincount(idx[valid], weights=power[valid], minlength=n_bins) / safe
    km = np.bincount(idx[valid], weights=kmag[valid], minlength=n_bins) / safe
    return km, pk, counts


def power_spectrum(field, box: float, n_bins: int = 12):
    return cross_power(field, field, box, n_bins)


def cross_correlation(a, b, box: float, n_bins: int = 12):
    """r(k) = P_ab / sqrt(P_aa P_bb). Returns (k, r)."""
    k, pab, _ = cross_power(a, b, box, n_bins)
    _, paa, _ = cross_power(a, a, box, n_bins)
    _, pbb, _ = cross_power(b, b, box, n_bins)
    denom = np.sqrt(paa * pbb)
    return k, np.where(denom > 0, pab / np.where(denom > 0, denom, 1.0), 0.0)


def gaussianity_moments(x) -> dict:
    """Mean, variance, skewness and excess kurtosis; (0, 1, 0, 0) for N(0, 1).
    Manticore checks these on the inferred white-noise field."""
    x = np.asarray(x, float).ravel()
    m, s = x.mean(), x.std()
    z = (x - m) / s
    return dict(
        mean=float(m),
        variance=float(s**2),
        skewness=float(np.mean(z**3)),
        kurtosis=float(np.mean(z**4) - 3.0),
    )


# --------------------------------------------------------------------------- #
# Chains: arrays of shape (n_chains, n_samples)
# --------------------------------------------------------------------------- #
def gelman_rubin(chains) -> float:
    """Split R-hat (Gelman et al. 2013): each chain is halved, so it is also
    defined, and sensitive to drifts, for a single chain."""
    x = np.asarray(chains, float)
    if x.ndim == 1:
        x = x[None]
    half = x.shape[1] // 2
    if half < 2:
        return float("nan")
    x = np.concatenate([x[:, :half], x[:, half : 2 * half]], axis=0)
    n = x.shape[1]
    w = x.var(axis=1, ddof=1).mean()
    b = n * x.mean(axis=1).var(ddof=1)
    if w <= 0:
        return float("nan")
    return float(np.sqrt(((n - 1) / n * w + b / n) / w))


def autocorrelation(x) -> np.ndarray:
    """Normalised autocorrelation function of a 1D series."""
    x = np.asarray(x, float)
    n = x.size
    x = x - x.mean()
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    if acf[0] <= 0:
        return np.zeros(n)
    return acf / acf[0]


def integrated_autocorr_time(chains) -> float:
    """tau = 1 + 2 sum_k rho_k, chain-averaged rho, truncated with Geyer's
    initial-positive-sequence rule."""
    x = np.asarray(chains, float)
    if x.ndim == 1:
        x = x[None]
    rho = np.mean([autocorrelation(c) for c in x], axis=0)
    tau = 1.0
    for k in range(1, len(rho) - 1, 2):
        pair = rho[k] + rho[k + 1]
        if pair < 0:
            break
        tau += 2.0 * pair
    return float(max(tau, 1.0))


def effective_sample_size(chains) -> float:
    x = np.asarray(chains, float)
    return float(x.size / integrated_autocorr_time(x))
