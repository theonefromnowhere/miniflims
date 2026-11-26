"""The field-level model: white noise -> linear field -> evolved density ->
expected galaxy counts -> likelihood, plus priors.

Latent variables
    field     white-noise initial conditions x, shape (n_grid,)*3, prior N(0, 1)
    nuisance  per-catalogue parameters, shape (n_catalogues, n_params):
              [nbar, <bias params>, <non-local bias coefficients>, <likelihood params>]
    png       primordial non-Gaussianity parameters [f_NL, <shape params>]
              (only with png.enabled; state shape (1, n_png))

Functions of the field take the PNG parameter vector as ``png`` (required
when png.enabled, ignored otherwise).

Galaxies: lambda = nbar R B_tot / <B_tot>, with B_tot the local bias B(rho)
combined with the non-local terms of galaxies.bias.non_local (e.g. the
scale-dependent PNG bias b_phi f_NL psi_alpha; see nonlocal_bias.py). Their
operator fields ``ops`` (n_terms, n_mesh^3) travel with rho; ``ops`` is None
without non-local terms.

Hyper-parameters (scalars that may follow a schedule during warm-up;
defaults from the config, see FieldModel.hyper_defaults)
    likelihood_temperature  beta multiplying the log-likelihood
    lambda_ps               strength of the Manticore power-spectrum prior (prior.lambda_ps)
    lambda_bs               strength of the white-noise bispectrum (Gaussianity)
                            prior (prior.bispectrum.lambda_bs)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import cola  # noqa: F401  (registers the tcola forward model)
from .bias import BIAS_MODELS
from .config import Config
from .cosmology import Cosmology
from .forward import FORWARD_MODELS
from .grid import Grid
from .likelihood import LIKELIHOODS
from .nonlocal_bias import NonLocalBias
from .params import ParamSpec, apply_override, resolve_specs
from .png import PNGModel
from .power import linear_power

HYPER_DEFAULTS = {"likelihood_temperature": 1.0, "lambda_ps": 0.0, "lambda_bs": 0.0}

NBAR_SPEC = ParamSpec("nbar", 1.0, 1e-3, 1e3, 0.1)


class FieldModel:
    def __init__(self, cfg: Config, response, counts=None):
        self.cfg = cfg
        g = cfg.grid
        if g.n_grid % 2:
            raise ValueError("grid.n_grid must be even")
        self.grid = Grid(g.n_grid, g.box_size)
        self.mesh = Grid(g.n_mesh or g.n_grid, g.box_size)
        self.cosmo = Cosmology.from_config(cfg.cosmology)

        # sqrt(P N^3 / V): white noise -> linear density, in rfftn layout
        pc = cfg.power
        pk = linear_power(
            self.grid.kmag(), self.cosmo, pc.transfer, pc.k_min, pc.k_max, pc.n_k
        )
        self.amplitude = jnp.sqrt(pk * self.grid.n**3 / self.grid.volume)

        self.forward = FORWARD_MODELS.get(cfg.forward.model)(
            cfg.forward, self.grid, self.mesh, self.cosmo
        )
        self.png = (
            PNGModel(cfg.png, self.grid, self.cosmo, pc.transfer, cfg.precision)
            if cfg.png.enabled else None
        )
        gal = cfg.galaxies
        self.bias = BIAS_MODELS.get(gal.bias.model)()
        self.likelihood = LIKELIHOODS.get(gal.likelihood.model)()
        self.floor = gal.density_floor
        self.n_catalogues = gal.n_catalogues

        nl = gal.bias.non_local
        self.nonlocal_bias = (
            NonLocalBias(nl, self.grid, self.mesh, self.amplitude, self.png) if nl.terms else None
        )
        nl_specs = self.nonlocal_bias.specs if self.nonlocal_bias is not None else []
        self.specs: list[ParamSpec] = (
            [apply_override(NBAR_SPEC, gal.nbar)]
            + resolve_specs(tuple(self.bias.param_specs) + tuple(nl_specs), gal.bias.params, "galaxies.bias")
            + resolve_specs(
                self.likelihood.param_specs, gal.likelihood.params, "galaxies.likelihood"
            )
        )
        self.param_names = [s.name for s in self.specs]
        if len(set(self.param_names)) != len(self.param_names):
            raise ValueError(f"duplicate nuisance parameter names: {self.param_names}")
        self.lower = jnp.asarray([s.lower for s in self.specs])
        self.upper = jnp.asarray([s.upper for s in self.specs])
        self.widths = jnp.asarray([s.width for s in self.specs])
        self.sampled = tuple(s.sample for s in self.specs)
        self._nl_idx = np.array([self.param_names.index(s.name) for s in nl_specs], dtype=int)

        shape = (self.n_catalogues,) + (self.mesh.n,) * 3
        response = np.asarray(response, dtype=float)
        if response.shape != shape:
            raise ValueError(f"response has shape {response.shape}, expected {shape}")
        if response.min() < 0:
            raise ValueError("response must be non-negative")
        self.response = jnp.asarray(response)
        self.counts = None
        if counts is not None:
            self.set_data(counts)

        self._setup_ps_prior(cfg.prior.ps_prior_bins)
        self._setup_bs_prior(cfg.prior.bispectrum)
        self.hyper_defaults = {
            **HYPER_DEFAULTS,
            "lambda_ps": float(cfg.prior.lambda_ps),
            "lambda_bs": float(cfg.prior.bispectrum.lambda_bs),
        }

    # ------------------------------------------------------------------ data
    def set_data(self, counts) -> "FieldModel":
        counts = np.asarray(counts, dtype=float)
        if counts.shape != self.response.shape:
            raise ValueError(
                f"counts has shape {counts.shape}, expected {self.response.shape}"
            )
        self.counts = jnp.asarray(counts)
        return self

    # ------------------------------------------------------------ parameters
    def params_dict(self, theta_row) -> dict:
        return {name: theta_row[i] for i, name in enumerate(self.param_names)}

    def default_nuisance(self) -> np.ndarray:
        row = np.array([s.init for s in self.specs])
        return np.tile(row, (self.n_catalogues, 1))

    def nuisance_from_dict(self, values: dict) -> np.ndarray:
        """name -> scalar or per-catalogue list; missing names use ``init``."""
        unknown = set(values) - set(self.param_names)
        if unknown:
            raise ValueError(
                f"unknown nuisance parameter(s) {sorted(unknown)}; "
                f"valid: {self.param_names}"
            )
        theta = self.default_nuisance()
        for name, v in values.items():
            v = np.asarray(v, dtype=float)
            if v.ndim not in (0, 1) or (v.ndim == 1 and v.size != self.n_catalogues):
                raise ValueError(
                    f"'{name}': give one value or one per catalogue "
                    f"({self.n_catalogues})"
                )
            theta[:, self.param_names.index(name)] = v
        return theta

    def init_nuisance(self, key, jitter: float = 0.0):
        """Starting point; sampled parameters are scattered by ``jitter`` times
        the prior range around ``init``."""
        theta = jnp.asarray(self.default_nuisance())
        if jitter > 0:
            u = jax.random.uniform(key, theta.shape, minval=-0.5, maxval=0.5)
            span = self.upper - self.lower
            eps = 1e-6 * span
            scat = jnp.clip(theta + jitter * span * u, self.lower + eps, self.upper - eps)
            theta = jnp.where(jnp.asarray(self.sampled), scat, theta)
        return theta

    # --------------------------------------------------------------- physics
    def gaussian_density_k(self, x):
        """Gaussian z=0 linear density from white noise (rfftn layout)."""
        return jnp.fft.rfftn(x) * self.amplitude

    def _check_png(self, png):
        if self.png is not None and png is None:
            raise ValueError("png.enabled: pass the PNG parameters (png=...)")

    def linear_density(self, x, png=None):
        """z=0 linear density contrast from white noise (and PNG parameters)."""
        self._check_png(png)
        dk = self.gaussian_density_k(x)
        if self.png is not None:
            dk = self.png.density_k(dk, png)
        return jnp.fft.irfftn(dk, s=x.shape)

    def final_density(self, x, png=None):
        """Evolved rho = 1 + delta on the mesh."""
        return self.forward(self.linear_density(x, png))

    def bias_potential(self, x):
        """psi_alpha of the png bias term on the mesh, from the white noise
        (None without it). Independent of the PNG parameters."""
        return None if self.nonlocal_bias is None else self.nonlocal_bias.potential(x)

    def bias_operators(self, rho, psi=None, png=None):
        """Non-local bias operator fields (n_terms, n_mesh^3), or None."""
        if self.nonlocal_bias is None:
            return None
        f_nl = png[0] if png is not None else None
        return self.nonlocal_bias.operators(rho, psi, f_nl)

    def galaxy_fields(self, x, png=None):
        """(rho, ops): evolved density and non-local bias operators."""
        rho = self.final_density(x, png)
        return rho, self.bias_operators(rho, self.bias_potential(x), png)

    def expected_counts(self, theta_row, rho, response_c, ops=None):
        p = self.params_dict(theta_row)
        b = self.bias(rho, p, self.floor)
        if self.nonlocal_bias is not None:
            if ops is None:
                raise ValueError("galaxies.bias.non_local: pass the operator fields (ops=...)")
            b = self.nonlocal_bias.apply(b, theta_row[self._nl_idx], ops, self.floor)
        if self.cfg.galaxies.bias.normalize:
            b = b / jnp.mean(b)
        return p["nbar"] * response_c * b

    # ----------------------------------------------------------- likelihood
    def log_likelihood_row(self, theta_row, rho, counts_c, response_c, ops=None):
        """Log-likelihood of one catalogue."""
        lam = self.expected_counts(theta_row, rho, response_c, ops)
        return self.likelihood.log_prob(
            counts_c, lam, rho, self.params_dict(theta_row), response_c > 0, self.floor
        )

    def log_likelihood_rho(self, rho, theta, ops=None):
        ll = jax.vmap(self.log_likelihood_row, in_axes=(0, None, 0, 0, None))(
            theta, rho, self.counts, self.response, ops
        )
        return jnp.sum(ll)

    def log_likelihood(self, x, theta, png=None):
        rho, ops = self.galaxy_fields(x, png)
        return self.log_likelihood_rho(rho, theta, ops)

    # --------------------------------------------------------------- priors
    def log_prior_nuisance_row(self, theta_row):
        """Unnormalised; the box constraint is enforced by the sampler."""
        lp = 0.0
        for i, s in enumerate(self.specs):
            if s.prior == "normal":
                lp = lp - 0.5 * ((theta_row[i] - s.prior_mean) / s.prior_std) ** 2
        return lp

    def _setup_ps_prior(self, n_bins: int):
        kmag = self.grid.kmag()
        w = self.grid.rfft_weights()
        edges = np.linspace(0.0, kmag.max() * (1 + 1e-9), n_bins + 1)
        idx = np.clip(np.digitize(kmag, edges) - 1, 0, n_bins - 1)
        idx = np.where(kmag > 0, idx, n_bins)  # DC mode -> dropped bin
        n_i = np.bincount(idx.ravel(), weights=w.ravel(), minlength=n_bins + 1)[:n_bins]
        n_tot = n_i.sum()
        good = n_i > 0
        # Manticore Eq. 2: Var[P_i / P_bar] = 2 (1/N_i - 1/N_tot)
        var = np.where(good, 2.0 * (1.0 / np.maximum(n_i, 1) - 1.0 / n_tot), 1.0)
        good &= var > 0
        self._ps = dict(
            idx=jnp.asarray(idx.ravel()),
            w=jnp.asarray(w.ravel()),
            n_i=jnp.asarray(np.maximum(n_i, 1.0)),
            n_tot=float(n_tot),
            var=jnp.asarray(np.where(good, var, 1.0)),
            good=jnp.asarray(good),
            n_bins=n_bins,
        )

    def white_noise_power_ratio(self, x):
        """Binned white-noise power relative to its mean, P_i / P_bar."""
        ps = self._ps
        power = ps["w"] * jnp.abs(jnp.fft.rfftn(x)).ravel() ** 2
        binned = jax.ops.segment_sum(power, ps["idx"], num_segments=ps["n_bins"] + 1)
        binned = binned[: ps["n_bins"]]
        return (binned / ps["n_i"]) / (jnp.sum(binned) / ps["n_tot"])

    # ------------------------------------------ white-noise bispectrum prior
    def _setup_bs_prior(self, bc):
        """Binned bispectrum of the white noise over every closed triangle of
        |k| bins (i <= j <= l). Each bin is divided by its scatter for Gaussian
        white noise, calibrated once with ``bc.n_mc`` realizations (this
        includes the exact triangle counts and symmetry factors)."""
        n = self.grid.n
        kmag = np.asarray(self.grid.kmag())
        kf = self.grid.k_fundamental
        kmax = bc.k_max if bc.k_max is not None else self.grid.k_nyquist
        edges = np.linspace(0.5 * kf, kmax, bc.n_bins + 1)
        masks = np.stack([(kmag >= edges[b]) & (kmag < edges[b + 1]) for b in range(bc.n_bins)])
        shape = (n,) * 3
        ones = jnp.fft.irfftn(jnp.asarray(masks, jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64), s=shape, axes=(1, 2, 3))
        trip = [(i, j, l) for i in range(bc.n_bins) for j in range(i, bc.n_bins) for l in range(j, bc.n_bins)]
        counts = np.array([float(jnp.sum(ones[i] * ones[j] * ones[l])) for i, j, l in trip])
        keep = counts > 1e-9 * n**-3  # bins that close a triangle
        trip = [t for t, k in zip(trip, keep) if k]
        self._bs = dict(masks=jnp.asarray(masks), idx=jnp.asarray(np.array(trip, dtype=np.int32).reshape(-1, 3)),
                        n_bins=bc.n_bins, edges=edges)
        if bc.lambda_bs == 0 and "lambda_bs" not in self.cfg.sampler.schedules:
            return  # prior unused: skip the Monte Carlo calibration
        # Gaussian calibration: mean (~0) and scatter of every triangle bin
        key = jax.random.PRNGKey(bc.seed)
        draws = []
        stat = jax.jit(self._white_noise_bispectrum_raw)
        for kk in jax.random.split(key, bc.n_mc):
            draws.append(np.asarray(stat(jax.random.normal(kk, shape))))
        draws = np.array(draws)
        self._bs["mean"] = jnp.asarray(draws.mean(0))
        self._bs["scale"] = jnp.asarray(draws.std(0, ddof=1))

    def _white_noise_bispectrum_raw(self, x):
        """sum_x d_i d_j d_l for every kept triangle bin (d_b = shell-filtered x)."""
        bs = self._bs
        d = jnp.fft.irfftn(jnp.fft.rfftn(x)[None] * bs["masks"], s=x.shape, axes=(1, 2, 3))
        idx = bs["idx"]
        return jnp.sum(d[idx[:, 0]] * d[idx[:, 1]] * d[idx[:, 2]], axis=(1, 2, 3))

    def white_noise_bispectrum_chi2(self, x):
        """sum over triangle bins of (B_bin / Gaussian scatter)^2; ~ number of
        bins for Gaussian white noise, larger when x is non-Gaussian."""
        bs = self._bs
        z = (self._white_noise_bispectrum_raw(x) - bs["mean"]) / bs["scale"]
        return jnp.sum(z**2)

    def log_prior_field(self, x, hyper=None):
        """Gaussian white-noise prior (Manticore Eq. 1) plus, if switched on,
        the power-spectrum prior (Eq. 6) that keeps the binned white-noise
        spectrum flat (lambda_ps), and the Gaussianity prior that keeps its
        binned bispectrum at zero (lambda_bs)."""
        hyper = {**self.hyper_defaults, **(hyper or {})}
        lp = -0.5 * jnp.sum(x**2)
        ps = self._ps
        ratio = self.white_noise_power_ratio(x)
        pen = jnp.sum(jnp.where(ps["good"], (ratio - 1.0) ** 2 / ps["var"], 0.0))
        lp = lp - 0.5 * hyper["lambda_ps"] * pen
        if self._bs["idx"].shape[0] and ("mean" in self._bs):
            lp = lp - 0.5 * hyper["lambda_bs"] * self.white_noise_bispectrum_chi2(x)
        return lp

    # ------------------------------------------------------------ posterior
    def log_posterior(self, x, theta, hyper=None, png=None):
        """Unnormalised log posterior of (field, nuisance, png) at the given hypers."""
        hyper = {**self.hyper_defaults, **(hyper or {})}
        lp = (
            self.log_prior_field(x, hyper)
            + hyper["likelihood_temperature"] * self.log_likelihood(x, theta, png)
            + jnp.sum(jax.vmap(self.log_prior_nuisance_row)(theta))
        )
        if self.png is not None:
            lp = lp + self.png.log_prior(png)
        return lp

    def png_conditional(self, x, theta, hyper=None, png=None):
        """Conditional of the PNG parameters given field and nuisance, in the
        slice-block convention ``(fn(row, aux_row), aux)`` with a single row.

        If only f_NL is sampled, the density is linear in it: M Phi_NG is
        computed once per sweep (from the fixed shape parameters) and each
        evaluation costs one forward model. Sampled shape parameters need a
        new Phi_NG for every evaluation.
        """
        hyper = {**self.hyper_defaults, **(hyper or {})}
        beta = hyper["likelihood_temperature"]
        pm = self.png
        dk = self.gaussian_density_k(x)
        shape = x.shape
        psi = self.bias_potential(x)  # png bias term: f_NL enters through f_NL psi

        def loglike(dens_k, row):
            rho = self.forward(jnp.fft.irfftn(dens_k, s=shape))
            return self.log_likelihood_rho(rho, theta, self.bias_operators(rho, psi, row))

        if pm.shape_sampled:
            def fn(row, aux_row):
                return beta * loglike(pm.density_k(dk, row), row) + pm.log_prior(row)
        else:
            ng = pm.ng_density_k(dk, png if png is not None else jnp.asarray(pm.default()))

            def fn(row, aux_row):
                return beta * loglike(dk + row[0] * ng, row) + pm.log_prior(row)
        return fn, jnp.zeros(1)

    def nuisance_conditional(self, x, hyper=None, png=None):
        """Conditional of the nuisance block given the field.

        The evolved density is computed once; catalogues are conditionally
        independent, so the returned ``fn(theta_row, aux_row)`` is a
        per-catalogue log density and ``aux`` carries the per-catalogue data.
        """
        hyper = {**self.hyper_defaults, **(hyper or {})}
        rho, ops = self.galaxy_fields(x, png)
        beta = hyper["likelihood_temperature"]

        def fn(theta_row, aux_row):
            counts_c, response_c = aux_row
            return beta * self.log_likelihood_row(
                theta_row, rho, counts_c, response_c, ops
            ) + self.log_prior_nuisance_row(theta_row)

        return fn, (self.counts, self.response)

    def monitor(self, state, hyper=None):
        """Quantities traced every sweep."""
        png = state["png"][0] if "png" in state else None
        ll = self.log_likelihood(state["field"], state["nuisance"], png)
        out = {
            "log_likelihood": ll,
            "log_prior_field": self.log_prior_field(state["field"], hyper),
            "nuisance": state["nuisance"],
        }
        if png is not None:
            out["png"] = png
        return out
