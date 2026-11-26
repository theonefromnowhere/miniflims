"""Galaxy bias models B(rho), rho = 1 + delta.

Each model declares its parameters (``param_specs``) and a pure function
``__call__(rho, p, floor)`` with ``p`` a dict name -> scalar. The expected
galaxy count is ``lambda = nbar * R * B / <B>`` (see model.py).
"""
from __future__ import annotations

import jax.numpy as jnp

from .params import ParamSpec
from .registry import Registry

BIAS_MODELS = Registry("bias model")


def positive(rho, floor):
    """Strictly positive density for logs and powers, smooth for rho >= 0."""
    return jnp.maximum(rho, 0.0) + floor


class BiasModel:
    param_specs: tuple[ParamSpec, ...] = ()

    def __call__(self, rho, p: dict, floor: float):
        raise NotImplementedError


@BIAS_MODELS.register("linear")
class LinearBias(BiasModel):
    """B = 1 + b delta. Can go negative: pair with the Gaussian likelihood."""

    param_specs = (ParamSpec("b", 1.0, 0.0, 10.0, 0.2),)

    def __call__(self, rho, p, floor):
        return 1.0 + p["b"] * (rho - 1.0)


@BIAS_MODELS.register("power_law")
class PowerLawBias(BiasModel):
    """B = rho^alpha."""

    param_specs = (ParamSpec("alpha", 1.0, 0.0, 5.0, 0.2),)

    def __call__(self, rho, p, floor):
        return positive(rho, floor) ** p["alpha"]


@BIAS_MODELS.register("truncated_power_law")
class TruncatedPowerLawBias(BiasModel):
    """B = rho^alpha exp(-(rho/rho_g)^-epsilon)  (Neyrinck et al. 2014; the
    BORG bias of Jasche & Lavaux 2019)."""

    param_specs = (
        ParamSpec("alpha", 1.0, 0.0, 5.0, 0.2),
        ParamSpec("rho_g", 0.5, 1e-3, 10.0, 0.2),
        ParamSpec("epsilon", 1.0, 1e-2, 5.0, 0.3),
    )

    def __call__(self, rho, p, floor):
        r = positive(rho, floor)
        return r ** p["alpha"] * jnp.exp(-((r / p["rho_g"]) ** (-p["epsilon"])))


@BIAS_MODELS.register("broken_power_law")
class BrokenPowerLawBias(BiasModel):
    """Smooth broken power law in density (the form of broken_powerlaw in the
    user's quijote_web.py / tf_broken_powerlaw in quijote_localbias.py):

        B = (1 + (rho / rho_b)^2)^(n/2),   rho_b = 10^log_rhob

    -> 1 for rho << rho_b, -> (rho / rho_b)^n for rho >> rho_b. The amplitude
    is set by the normalisation <B> (galaxies.bias.normalize) and nbar.
    """

    param_specs = (
        ParamSpec("n", 2.0, 0.0, 8.0, 0.3),
        ParamSpec("log_rhob", 0.0, -1.5, 1.5, 0.2),
    )

    def __call__(self, rho, p, floor):
        x = positive(rho, floor) * 10.0 ** (-p["log_rhob"])
        return jnp.exp(0.5 * p["n"] * jnp.log1p(x * x))


@BIAS_MODELS.register("stdp")
class SigmoidTruncatedDoublePowerLaw(BiasModel):
    """Manticore's sigmoid-truncated double power law (McAlpine et al. 2025,
    Eqs. 13-16): B = Theta_sig * Theta_1st * Theta_2nd with

        Theta_sig = 1 / (1 + exp[(gamma_t - ln rho) / gamma_s])
        Theta_1st = rho^alpha
        Theta_2nd = (1 + exp[beta (ln rho - beta_0)])^(-beta)

    beta = 0 recovers a single (sigmoid-truncated) power law.
    """

    param_specs = (
        ParamSpec("gamma_t", -1.0, -5.0, 3.0, 0.5),
        ParamSpec("gamma_s", 0.5, 0.05, 5.0, 0.3),
        ParamSpec("alpha", 1.0, 0.0, 5.0, 0.2),
        ParamSpec("beta", 0.5, 0.0, 3.0, 0.3),
        ParamSpec("beta_0", 1.0, -2.0, 5.0, 0.5),
    )

    def __call__(self, rho, p, floor):
        ln = jnp.log(positive(rho, floor))
        sig = 1.0 / (1.0 + jnp.exp((p["gamma_t"] - ln) / p["gamma_s"]))
        first = jnp.exp(p["alpha"] * ln)
        second = jnp.exp(-p["beta"] * jnp.logaddexp(0.0, p["beta"] * (ln - p["beta_0"])))
        return sig * first * second
