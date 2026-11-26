"""Matter transfer functions T(k). Signature: ``f(k, cosmo)`` with k in h/Mpc.

``eisenstein_hu``           Eisenstein & Hu 1998 (ApJ 496, 605), full fit with
                            baryon acoustic oscillations (their Eqs. 2-24).
``eisenstein_hu_nowiggle``  EH98 smooth "no-wiggle" fit (Eqs. 26, 28-31).
``bbks``                    Bardeen et al. 1986 with Sugiyama's shape parameter.
"""
from __future__ import annotations

import jax.numpy as jnp

from .cosmology import Cosmology
from .registry import Registry

TRANSFER_FUNCTIONS = Registry("transfer function")

_E = jnp.e


def eh_scales(cosmo: Cosmology) -> dict:
    """Characteristic scales of the EH98 fit (lengths in Mpc, k in 1/Mpc)."""
    omh2 = cosmo.omega_m * cosmo.h**2
    obh2 = cosmo.omega_b * cosmo.h**2
    theta = cosmo.t_cmb / 2.7

    z_eq = 2.50e4 * omh2 * theta**-4
    k_eq = 7.46e-2 * omh2 * theta**-2
    b1 = 0.313 * omh2**-0.419 * (1.0 + 0.607 * omh2**0.674)
    b2 = 0.238 * omh2**0.223
    z_d = 1291.0 * omh2**0.251 / (1.0 + 0.659 * omh2**0.828) * (1.0 + b1 * obh2**b2)

    def ratio(z):  # baryon-to-photon momentum density ratio
        return 31.5 * obh2 * theta**-4 * (1e3 / z)

    r_d, r_eq = ratio(z_d), ratio(z_eq)
    s = (
        2.0
        / (3.0 * k_eq)
        * jnp.sqrt(6.0 / r_eq)
        * jnp.log((jnp.sqrt(1.0 + r_d) + jnp.sqrt(r_d + r_eq)) / (1.0 + jnp.sqrt(r_eq)))
    )
    k_silk = 1.6 * obh2**0.52 * omh2**0.73 * (1.0 + (10.4 * omh2) ** -0.95)
    return dict(
        z_eq=z_eq, k_eq=k_eq, z_d=z_d, r_d=r_d, r_eq=r_eq, sound_horizon=s, k_silk=k_silk
    )


@TRANSFER_FUNCTIONS.register("eisenstein_hu")
def eisenstein_hu(k, cosmo: Cosmology):
    k = jnp.asarray(k)
    omh2 = cosmo.omega_m * cosmo.h**2
    fb = cosmo.omega_b / cosmo.omega_m
    fc = 1.0 - fb
    sc = eh_scales(cosmo)
    k_eq, s, k_silk = sc["k_eq"], sc["sound_horizon"], sc["k_silk"]

    kk = jnp.where(k > 0, k, 1.0) * cosmo.h  # 1/Mpc; k=0 handled at the end
    q = kk / (13.41 * k_eq)
    ks = kk * s

    # --- cold dark matter ---
    a1 = (46.9 * omh2) ** 0.670 * (1.0 + (32.1 * omh2) ** -0.532)
    a2 = (12.0 * omh2) ** 0.424 * (1.0 + (45.0 * omh2) ** -0.582)
    alpha_c = a1**-fb * a2 ** (-(fb**3))
    bb1 = 0.944 / (1.0 + (458.0 * omh2) ** -0.708)
    bb2 = (0.395 * omh2) ** -0.0266
    beta_c = 1.0 / (1.0 + bb1 * (fc**bb2 - 1.0))

    def t0(alpha, beta):
        ln = jnp.log(_E + 1.8 * beta * q)
        c = 14.2 / alpha + 386.0 / (1.0 + 69.9 * q**1.08)
        return ln / (ln + c * q**2)

    f = 1.0 / (1.0 + (ks / 5.4) ** 4)
    t_c = f * t0(1.0, beta_c) + (1.0 - f) * t0(alpha_c, beta_c)

    # --- baryons ---
    y = (1.0 + sc["z_eq"]) / (1.0 + sc["z_d"])
    sq = jnp.sqrt(1.0 + y)
    g = y * (-6.0 * sq + (2.0 + 3.0 * y) * jnp.log((sq + 1.0) / (sq - 1.0)))
    alpha_b = 2.07 * k_eq * s * (1.0 + sc["r_d"]) ** -0.75 * g
    beta_b = 0.5 + fb + (3.0 - 2.0 * fb) * jnp.sqrt((17.2 * omh2) ** 2 + 1.0)
    beta_node = 8.41 * omh2**0.435
    s_tilde = s / (1.0 + (beta_node / ks) ** 3) ** (1.0 / 3.0)
    j0 = jnp.sinc(kk * s_tilde / jnp.pi)  # sin(x)/x
    t_b = (
        t0(1.0, 1.0) / (1.0 + (ks / 5.2) ** 2)
        + alpha_b / (1.0 + (beta_b / ks) ** 3) * jnp.exp(-((kk / k_silk) ** 1.4))
    ) * j0

    return jnp.where(k > 0, fb * t_b + fc * t_c, 1.0)


@TRANSFER_FUNCTIONS.register("eisenstein_hu_nowiggle")
def eisenstein_hu_nowiggle(k, cosmo: Cosmology):
    k = jnp.asarray(k)
    omh2 = cosmo.omega_m * cosmo.h**2
    obh2 = cosmo.omega_b * cosmo.h**2
    fb = cosmo.omega_b / cosmo.omega_m
    theta = cosmo.t_cmb / 2.7

    s = 44.5 * jnp.log(9.83 / omh2) / jnp.sqrt(1.0 + 10.0 * obh2**0.75)  # Mpc
    alpha = 1.0 - 0.328 * jnp.log(431.0 * omh2) * fb + 0.38 * jnp.log(22.3 * omh2) * fb**2
    kk = jnp.where(k > 0, k, 1.0)  # h/Mpc
    gamma = cosmo.omega_m * cosmo.h * (
        alpha + (1.0 - alpha) / (1.0 + (0.43 * kk * cosmo.h * s) ** 4)
    )
    q = kk * theta**2 / gamma
    l0 = jnp.log(2.0 * _E + 1.8 * q)
    c0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    return jnp.where(k > 0, l0 / (l0 + c0 * q**2), 1.0)


@TRANSFER_FUNCTIONS.register("bbks")
def bbks(k, cosmo: Cosmology):
    k = jnp.asarray(k)
    gamma = cosmo.omega_m * cosmo.h * jnp.exp(
        -cosmo.omega_b * (1.0 + jnp.sqrt(2.0 * cosmo.h) / cosmo.omega_m)
    )
    q = jnp.where(k > 0, k, 1.0) / gamma
    t = (
        jnp.log(1.0 + 2.34 * q)
        / (2.34 * q)
        * (1.0 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4) ** -0.25
    )
    return jnp.where(k > 0, t, 1.0)
