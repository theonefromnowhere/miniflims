"""Voxel-wise galaxy-count likelihoods.

``log_prob(counts, lam, rho, p, mask, floor)`` returns the summed log
probability over voxels with ``mask`` True (voxels outside the survey response
carry no information). ``sample`` draws mock data with numpy.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln

from .bias import positive
from .params import ParamSpec
from .registry import Registry

LIKELIHOODS = Registry("likelihood")

_TINY = 1e-30


class Likelihood:
    param_specs: tuple[ParamSpec, ...] = ()

    def log_prob(self, counts, lam, rho, p: dict, mask, floor: float):
        raise NotImplementedError

    def sample(self, rng: np.random.Generator, lam, rho, p: dict, floor: float):
        raise NotImplementedError


@LIKELIHOODS.register("poisson")
class PoissonLikelihood(Likelihood):
    def log_prob(self, counts, lam, rho, p, mask, floor):
        lam = jnp.maximum(lam, _TINY)
        lp = counts * jnp.log(lam) - lam - gammaln(counts + 1.0)
        return jnp.sum(jnp.where(mask, lp, 0.0))

    def sample(self, rng, lam, rho, p, floor):
        return rng.poisson(np.maximum(lam, 0.0)).astype(float)


@LIKELIHOODS.register("gaussian")
class GaussianLikelihood(Likelihood):
    """counts ~ Normal(lambda, sigma^2), homoscedastic."""

    param_specs = (ParamSpec("sigma", 1.0, 1e-3, 1e3, 0.1),)

    def log_prob(self, counts, lam, rho, p, mask, floor):
        s = p["sigma"]
        lp = -0.5 * ((counts - lam) / s) ** 2 - jnp.log(s) - 0.5 * jnp.log(2 * jnp.pi)
        return jnp.sum(jnp.where(mask, lp, 0.0))

    def sample(self, rng, lam, rho, p, floor):
        return lam + p["sigma"] * rng.standard_normal(np.shape(lam))


@LIKELIHOODS.register("generalized_poisson")
class GeneralizedPoissonLikelihood(Likelihood):
    """Manticore's generalized Poisson (McAlpine et al. 2025, Eqs. 22-25):

        P(N | lam, b) = lam (1-b) [lam (1-b) + N b]^(N-1) exp[-lam (1-b) - N b] / N!

    mean lam, variance lam / (1-b)^2, with density-dependent over-dispersion
    b = b_alpha * rho^b_beta clipped to [0, b_max]. b = 0 is Poisson.
    """

    b_max = 0.99
    param_specs = (
        ParamSpec("b_alpha", 0.2, 0.0, 0.95, 0.1),
        ParamSpec("b_beta", 0.0, -2.0, 2.0, 0.3),
    )

    def dispersion(self, rho, p, floor, xp=jnp):
        r = positive(rho, floor) if xp is jnp else np.maximum(rho, 0.0) + floor
        return xp.clip(p["b_alpha"] * r ** p["b_beta"], 0.0, self.b_max)

    def log_prob(self, counts, lam, rho, p, mask, floor):
        b = self.dispersion(rho, p, floor)
        theta = jnp.maximum(lam * (1.0 - b), _TINY)
        lp = (
            jnp.log(theta)
            + (counts - 1.0) * jnp.log(theta + counts * b)
            - theta
            - counts * b
            - gammaln(counts + 1.0)
        )
        return jnp.sum(jnp.where(mask, lp, 0.0))

    def sample(self, rng, lam, rho, p, floor):
        # A generalized Poisson variate is the total progeny of a branching
        # process: Poisson(theta) ancestors, each individual having Poisson(b)
        # offspring.
        b = np.asarray(self.dispersion(np.asarray(rho), p, floor, xp=np))
        theta = np.maximum(np.asarray(lam), 0.0) * (1.0 - b)
        generation = rng.poisson(theta)
        total = generation.copy()
        while generation.any():
            generation = rng.poisson(b * generation)
            total += generation
        return total.astype(float)
