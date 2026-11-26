"""Gibbs blocks of the field-level model, looked up by name from
``sampler.blocks``. A builder takes ``(model, cfg)`` and returns a
``samplers.Block``; register new ones to extend or replace the scheme.

The state holds ``field`` (chains, n, n, n), ``nuisance`` (chains, cats,
params) and, with png.enabled, ``png`` (chains, 1, n_png)."""
from __future__ import annotations

from .registry import Registry
from .samplers import hmc_block, slice_block

BLOCKS = Registry("Gibbs block")


def _png(state):
    return state["png"][0] if "png" in state else None


@BLOCKS.register("field")
def field_block(model, cfg):
    """HMC on the white-noise initial conditions given the nuisance (and PNG)
    parameters. With PNG the gradient runs through jaxpng's Phi_NG."""

    def logdensity(state, hyper):
        return model.log_posterior(state["field"], state["nuisance"], hyper, _png(state))

    return hmc_block("field", "field", logdensity, cfg.sampler.hmc)


@BLOCKS.register("nuisance")
def nuisance_block(model, cfg):
    """Slice sampling of nbar, bias and likelihood parameters given the field.
    The forward model is run once per sweep; catalogues update in parallel."""

    def make_conditional(state, hyper):
        return model.nuisance_conditional(state["field"], hyper, _png(state))

    return slice_block(
        "nuisance",
        "nuisance",
        make_conditional,
        model.lower,
        model.upper,
        model.widths,
        cfg.sampler.slice,
        model.sampled,
    )


@BLOCKS.register("png")
def png_block(model, cfg):
    """Slice sampling of f_NL (and sampled shape parameters such as omega,
    phase) given the field and the nuisance parameters. With only f_NL sampled,
    Phi_NG is computed once per sweep and every slice evaluation costs one
    forward model."""
    if model.png is None:
        raise ValueError("sampler.blocks contains 'png' but png.enabled is false")

    def make_conditional(state, hyper):
        return model.png_conditional(state["field"], state["nuisance"], hyper, _png(state))

    pm = model.png
    return slice_block("png", "png", make_conditional, pm.lower, pm.upper, pm.widths,
                       cfg.sampler.slice, pm.sampled)
