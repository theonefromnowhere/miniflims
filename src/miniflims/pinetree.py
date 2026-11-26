"""PineTree (Ding, Lavaux & Jasche 2024, A&A 690 A236) as the halo population
model: a stochastic, mass-resolved halo field conditioned on the evolved
density, in pure JAX (no flax / tensorflow-probability).

Forward pass, per voxel v of the mesh (``galaxies.population: pinetree``)

    x      = ln(rho)                                   (rho floored like the bias models)
    psi_v  = b_c + sum_s w_s sum_{o in shell s} x_{v+o}  3^3 monopole kernel, periodic;
                                                       shells |o| = 0, 1, sqrt2, sqrt3
    a_v    = softplus(psi_v)
    mu_0   = mw_0 a + mb_0,  mu_k = mu_{k-1} + softplus(mw_k a + mb_k)
    sig_k  = exp(sw_k a + sb_k)
    A_k    = exp(nw a + nb) softmax(aw a + ab)_k

    n(ln M | v) = sum_k A_k N(ln M; mu_k, sig_k)       (conditional mass function)

The halos above the threshold M_th are a Poisson process in (voxel, ln M) with
intensity R_v n(ln M | v), R the survey response. Log-likelihood (PineTree's
training loss with the sign flipped):

    ln L = sum_h ln[R_{v_h} n(ln M_h | v_h)] - sum_v R_v lambda_v,
    lambda_v = sum_k A_k Phi((mu_k - ln M_th) / sig_k)

Parameters are a flat vector in the order of ``NAMES``; they come from the JSON
written by the PineTree repository's ``scripts/export_pinetree_params.py``.
Optionally they are sampled in whitened coordinates z, theta = theta_ref + L z
(``galaxies.pinetree.whitening``, from ``scripts/pinetree_whitening.py``),
because the raw parameters are strongly correlated (condition number ~1e6).
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import log_ndtr, logsumexp, ndtr, ndtri

from .bias import positive
from .params import ParamSpec, resolve_specs

N_MIX = 2
NAMES = (
    "conv_w0", "conv_w1", "conv_w2", "conv_w3", "conv_b",
    "nmean_w", "nmean_b",
    "amp_w0", "amp_w1", "amp_b0", "amp_b1",
    "mean_w0", "mean_w1", "mean_b0", "mean_b1",
    "std_w0", "std_w1", "std_b0", "std_b1",
)
# the softmax over the two amplitudes is invariant under a common shift of
# (amp_w0, amp_w1) and of (amp_b0, amp_b1): fixing component 0 removes both
GAUGE = ("amp_w0", "amp_b0")
_IDX = {n: i for i, n in enumerate(NAMES)}
_LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)


def _shells():
    """(27, 3) offsets of the 3^3 kernel and their radial shell (0..3)."""
    off = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    return off, (off**2).sum(1)  # |o|^2 = 0, 1, 2, 3 is the shell index


def load_params(path) -> dict:
    """Read an exported PineTree model: {'theta': {name: value}, 'mass_threshold', ...}."""
    with open(path) as f:
        d = json.load(f)
    if d.get("format") != "pinetree-monopole-v1":
        raise ValueError(f"{path}: not a pinetree-monopole-v1 export")
    missing = set(NAMES) - set(d["theta"])
    if missing:
        raise ValueError(f"{path}: missing parameters {sorted(missing)}")
    return d


def raw_specs(theta0: np.ndarray, span: float, width: float, gauge_fix: bool) -> list[ParamSpec]:
    return [
        ParamSpec(n, float(v), float(v) - span, float(v) + span, width, sample=not (gauge_fix and n in GAUGE))
        for n, v in zip(NAMES, theta0)
    ]


class PineTreeModel:
    """Population model; ``specs`` are the nuisance parameters of the single
    catalogue (raw names, or z0..z{k-1} when whitened)."""

    def __init__(self, cfg, floor: float):
        if cfg.params_path is None:
            raise ValueError("galaxies.population=pinetree needs galaxies.pinetree.params_path")
        meta = load_params(cfg.params_path)
        self.meta = meta
        self.floor = floor
        thr = cfg.mass_threshold if cfg.mass_threshold is not None else meta["mass_threshold"]
        self.mass_threshold = float(thr)
        self.ln_threshold = float(np.log(thr))
        theta0 = np.array([meta["theta"][n] for n in NAMES], dtype=float)
        self.offsets, self.shell = _shells()

        if cfg.whitening is None:
            self.theta_ref = theta0
            self.L = None
            self.sampled_raw = None
            specs = raw_specs(theta0, cfg.raw_span, cfg.raw_width, cfg.gauge_fix)
        else:
            with np.load(cfg.whitening) as w:
                self.theta_ref = np.asarray(w["theta_ref"], float)
                self.L = np.asarray(w["L"], float)  # (n_sampled_raw, n_z)
                self.sampled_raw = np.asarray(w["sampled_raw"], int)
                names = [str(n) for n in w["names"]]
            if tuple(names) != NAMES:
                raise ValueError(f"{cfg.whitening}: parameter names do not match this version")
            specs = [ParamSpec(f"z{i}", 0.0, -cfg.z_bound, cfg.z_bound, 1.0) for i in range(self.L.shape[1])]
        self.specs = resolve_specs(tuple(specs), cfg.params, "galaxies.pinetree")
        self._L = None if self.L is None else jnp.asarray(self.L)
        self._ref = jnp.asarray(self.theta_ref)

    # ------------------------------------------------------------ parameters
    def theta(self, row):
        """Raw parameter vector (19,) from a nuisance row."""
        if self._L is None:
            return row
        return self._ref.at[self.sampled_raw].add(self._L @ row)

    def theta_np(self, rows: np.ndarray) -> np.ndarray:
        """Raw parameters for an array of rows (..., n_specs) -> (..., 19), numpy."""
        rows = np.asarray(rows, float)
        if self.L is None:
            return rows
        out = np.broadcast_to(self.theta_ref, rows.shape[:-1] + (len(NAMES),)).copy()
        out[..., self.sampled_raw] += rows @ self.L.T
        return out

    # --------------------------------------------------------------- forward
    def features(self, rho):
        """a_v = softplus(psi_v), flat (n_vox,), for raw parameters ``t``."""
        x = jnp.log(positive(rho, self.floor))
        # four shell sums of the periodic 3^3 neighbourhood (theta-independent)
        sums = [jnp.zeros_like(x) for _ in range(4)]
        for o, s in zip(self.offsets, self.shell):
            sums[s] = sums[s] + jnp.roll(x, shift=tuple(-o), axis=(0, 1, 2))
        return jnp.stack([s.ravel() for s in sums])  # (4, n_vox)

    def voxel_params(self, rho, row):
        """(log A, mu, sigma), each (n_vox, N_MIX), for a nuisance row."""
        t = self.theta(row)
        S = self.features(rho)
        psi = t[_IDX["conv_b"]] + t[:4] @ S
        a = jax.nn.softplus(psi)[:, None]  # (n_vox, 1)
        w = lambda name: jnp.stack([t[_IDX[f"{name}{k}"]] for k in range(N_MIX)])
        lin_mean = a * w("mean_w") + w("mean_b")
        mu = jnp.cumsum(jnp.concatenate([lin_mean[:, :1], jax.nn.softplus(lin_mean[:, 1:])], axis=1), axis=1)
        sig = jnp.exp(a * w("std_w") + w("std_b"))
        log_amp = jax.nn.log_softmax(a * w("amp_w") + w("amp_b"), axis=1) + (a * t[_IDX["nmean_w"]] + t[_IDX["nmean_b"]])
        return log_amp, mu, sig

    def log_rate_components(self, log_amp, mu, sig):
        """ln of the expected count above threshold of each component, (n_vox, N_MIX)."""
        return log_amp + log_ndtr((mu - self.ln_threshold) / sig)

    def rate(self, rho, row):
        """Expected number of halos above threshold per voxel, flat (n_vox,)."""
        log_amp, mu, sig = self.voxel_params(rho, row)
        return jnp.exp(logsumexp(self.log_rate_components(log_amp, mu, sig), axis=1))

    def rate_in_bin(self, rho, row, m_lo, m_hi):
        """Expected number per voxel with m_lo <= M < m_hi (both >= the threshold)."""
        log_amp, mu, sig = self.voxel_params(rho, row)
        p = ndtr((mu - jnp.log(m_lo)) / sig) - ndtr((mu - jnp.log(m_hi)) / sig)
        return jnp.sum(jnp.exp(log_amp) * p, axis=1)

    # ------------------------------------------------------------ likelihood
    def log_likelihood(self, rho, row, halo_voxel, halo_ln_mass, response):
        """Marked-Poisson log-likelihood of the halo catalogue (flat ``response``)."""
        log_amp, mu, sig = self.voxel_params(rho, row)
        la, m, s = log_amp[halo_voxel], mu[halo_voxel], sig[halo_voxel]
        z = (halo_ln_mass[:, None] - m) / s
        point = jnp.sum(logsumexp(la - jnp.log(s) - _LOG_SQRT_2PI - 0.5 * z**2, axis=1))
        point = point + jnp.sum(jnp.log(response[halo_voxel]))
        lam = jnp.exp(logsumexp(self.log_rate_components(log_amp, mu, sig), axis=1))
        return point - jnp.sum(response * lam)

    # ----------------------------------------------------------------- mocks
    def sample(self, rng: np.random.Generator, rho, row, response):
        """Draw a halo catalogue: (voxel ids, ln M), numpy. Poisson count per
        voxel, component chosen with probability proportional to its rate above
        threshold, ln M from that component truncated at ln M_th."""
        log_amp, mu, sig = (np.asarray(v, float) for v in self.voxel_params(jnp.asarray(rho), jnp.asarray(row)))
        lam_k = np.exp(np.asarray(self.log_rate_components(log_amp, mu, sig)))
        lam = np.asarray(response, float).ravel() * lam_k.sum(1)
        n = rng.poisson(lam)
        vox = np.repeat(np.arange(lam.size), n)
        p = lam_k[vox] / lam_k[vox].sum(1, keepdims=True)
        comp = (rng.random(vox.size)[:, None] > np.cumsum(p, axis=1)[:, :-1]).sum(1)
        m, s = mu[vox, comp], sig[vox, comp]
        q = np.asarray(ndtr((m - self.ln_threshold) / s))
        u = rng.uniform(np.finfo(float).tiny, 1.0, vox.size)
        ln_mass = m - s * np.asarray(ndtri(u * q))
        return vox.astype(np.int64), np.maximum(ln_mass, self.ln_threshold)


def whitening_matrix(hessian, floor_frac: float = 1e-6):
    """L with theta = theta_ref + L z, z ~ N(0, 1) near a Laplace approximation
    of curvature ``hessian`` (of -ln L). Eigenvalues are replaced by their
    absolute value, floored at ``floor_frac`` times the largest, so flat and
    slightly non-convex directions get a finite (wide) scale instead of none."""
    h = np.asarray(hessian, float)
    w, v = np.linalg.eigh(0.5 * (h + h.T))
    wa = np.maximum(np.abs(w), floor_frac * np.abs(w).max())
    return v / np.sqrt(wa), w


def spec_names(gal: dict) -> tuple[list[str], list[bool]]:
    """Nuisance names and sample flags from a plain (dict) galaxies config, for
    tools that read run metadata (watch)."""
    from .config import _from_dict, GalaxyConfig  # local import: config imports nothing from here
    g = _from_dict(GalaxyConfig, gal, "galaxies")
    model = PineTreeModel(g.pinetree, g.density_floor)
    return [s.name for s in model.specs], [s.sample for s in model.specs]


def counts_grid(halo_voxel: np.ndarray, n_mesh: int) -> np.ndarray:
    """Halo counts per voxel, (n_mesh,)*3."""
    return np.bincount(np.asarray(halo_voxel, int), minlength=n_mesh**3).reshape((n_mesh,) * 3).astype(float)
