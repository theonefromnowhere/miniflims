"""Separable bispectrum templates  B_Phi = sum_i b1^i(k1) b2^i(k2) b3^i(k3).

A template is a list of N_i terms; each term is a tuple of three callables
``(g1, g2, g3)``, each mapping a wavenumber-magnitude array to an array (or
scalar). Convention: the factors are written so that

    sum_i b1^i(k1) b2^i(k2) b3^i(k3) = B_Phi(k1,k2,k3) / f_NL,

i.e. f_NL is stripped (applied later as Phi = Phi^G + f_NL Phi^NG) but the
template's numerical prefactors ARE included. With this convention the Schwinger
generator in :mod:`jaxpng.phing` produces the correct reduced-bispectrum kernel
for every template, calibrated by the local anchor Phi^NG = Phi^G^2 - <Phi^G^2>.

Factors are functions of the physical wavenumber magnitude k, with the internal
kernel spectrum ``P_kernel(k) = k^(n_s-4)`` exposed. The primordial-potential
power spectrum is a pure power law P_Phi(k) ∝ k^(n_s-4) (the transfer function
cancels in P_matter/M^2), so the kernel built from P_kernel reproduces the
physical target up to an overall amplitude that cancels in the kernel ratio.

Transcribed from gengars/src/input_bispectrum.c (rk = P^(-1/3): 1/rk^3 = P,
1/rk^2 = P^(2/3), 1/rk = P^(1/3)). Templates: local (N_i=3), equilateral and
orthogonal (N_i=10). Custom shapes: build your own list of separable terms.
"""
from __future__ import annotations

import math

import jax.numpy as jnp

from .grids import P_kernel


def _pw(coeff, power, n_s):
    """Factor  k -> coeff * P_kernel(k)^power."""
    def g(k):
        return coeff * P_kernel(k, n_s) ** power
    return g


def _const(coeff):
    """Constant factor k -> coeff."""
    def g(k):
        return coeff
    return g


def _kpow(coeff, power):
    """Factor  k -> coeff * k**power  (a power of the wavenumber itself).

    Negative powers are set to 0 at k=0 to avoid 1/0 -> inf at the DC mode; this is
    exact, because the DC mode of Phi^G is 0 so it never contributes (inf*0 -> NaN
    is what we are avoiding).
    """
    if power < 0:
        def g(k):
            return coeff * jnp.where(k > 0, k ** power, 0.0)
        return g

    def g(k):
        return coeff * k ** power
    return g


def local(n_s):
    """Local: B/f_NL = 2[P1 P2 + P2 P3 + P1 P3].  W = 1 (anchor: Phi^NG = Phi^G^2)."""
    P = _pw(1.0, 1.0, n_s)
    two = _const(2.0)
    return [(P, P, two), (P, two, P), (two, P, P)]


def equilateral(n_s):
    """Equilateral template (Eq. 2.3), N_i = 10:
    B/f_NL = 6[ -(P1 P2 + 2 perm) - 2(P1 P2 P3)^{2/3} + (P1^{1/3} P2^{2/3} P3 + 5 perm) ]."""
    P, P23, P13 = _pw(1.0, 1.0, n_s), _pw(1.0, 2.0 / 3, n_s), _pw(1.0, 1.0 / 3, n_s)
    m6, m12 = _const(-6.0), _pw(-12.0, 2.0 / 3, n_s)
    P13x6 = _pw(6.0, 1.0 / 3, n_s)
    return [
        (P, P, m6),                # -6 P1 P2
        (P, m6, P),                # -6 P1 P3
        (m6, P, P),                # -6 P2 P3
        (m12, P23, P23),           # -12 (P1 P2 P3)^{2/3}
        (P13x6, P23, P),           # 6 P1^{1/3} P2^{2/3} P3   (+5 perm below)
        (P13x6, P, P23),
        (P23, P13x6, P),
        (P23, P, P13x6),
        (P, P23, P13x6),
        (P, P13x6, P23),
    ]


def orthogonal(n_s):
    """Orthogonal template (Eq. 2.4), N_i = 10:
    B/f_NL = 6[ -3(P1 P2 + 2 perm) - 8(P1 P2 P3)^{2/3} + 3(P1^{1/3} P2^{2/3} P3 + 5 perm) ]."""
    P, P23 = _pw(1.0, 1.0, n_s), _pw(1.0, 2.0 / 3, n_s)
    m18, m48 = _const(-18.0), _pw(-48.0, 2.0 / 3, n_s)
    P13x18 = _pw(18.0, 1.0 / 3, n_s)
    return [
        (P, P, m18),
        (P, m18, P),
        (m18, P, P),
        (m48, P23, P23),
        (P13x18, P23, P),
        (P13x18, P, P23),
        (P23, P13x18, P),
        (P23, P, P13x18),
        (P, P23, P13x18),
        (P, P13x18, P23),
    ]


def feature(n_s, omega, phase=0.0):
    """Oscillatory feature template (paper Eq. 5.1), separable form:

        B/f_NL = 6 [P(k1)P(k2)P(k3)]^{2/3} sin[omega (k1+k2+k3) + phase].

    The sin of a sum expands into separable sin/cos products (4 terms for
    phase=0, up to 8 otherwise). Factors use the TRUE wavenumber k (so the
    oscillation is in physical k, h/Mpc), with [P]^{2/3} from the kernel
    spectrum. omega has units of Mpc/h. (The C code oscillates in rk=P^(-1/3);
    we use k -- see jaxpng.bispectra module docstring.)
    """
    cphi, sphi = math.cos(phase), math.sin(phase)
    # sin(A+B+C) = sA cB cC + cA sB cC + cA cB sC - sA sB sC
    # cos(A+B+C) = cA cB cC - cA sB sC - sA cB sC - sA sB cC
    specs = [
        (6.0 * cphi, "s", "c", "c"), (6.0 * cphi, "c", "s", "c"),
        (6.0 * cphi, "c", "c", "s"), (-6.0 * cphi, "s", "s", "s"),
        (6.0 * sphi, "c", "c", "c"), (-6.0 * sphi, "c", "s", "s"),
        (-6.0 * sphi, "s", "c", "s"), (-6.0 * sphi, "s", "s", "c"),
    ]

    def fac(coeff, kind):
        trig = jnp.sin if kind == "s" else jnp.cos
        return lambda k: coeff * P_kernel(k, n_s) ** (2.0 / 3.0) * trig(omega * k)

    terms = []
    for coeff, a, b, c in specs:
        if coeff != 0.0:  # drop the sin(phase) terms when phase=0
            terms.append((fac(coeff, a), fac(1.0, b), fac(1.0, c)))
    return terms


def equilateral_shape():
    """Geometric equilateral shape  S^eq = (kt1 kt2 kt3)/(k1 k2 k3),
    kt_i = k_j + k_l - k_i, as a separable template (10 terms).

    Expands to  sum_{i!=j} ki/kj  -  sum_i ki^2/(kj kl)  -  2. It is **1 at the
    equilateral configuration** and **-> 0 (linearly) in the squeezed limit**
    (the consistency-relation-friendly suppression). Multiply an amplitude/shape
    by this (via :func:`multiply`) to suppress its squeezed limit without changing
    the equilateral normalization. NOTE: individual terms have 1/k poles that
    cancel in the sum; harmless in float64, and the kernel's 1/P factor suppresses
    low k. This is a dimensionless geometric factor (no P).
    """
    return [
        (_kpow(1.0, 1), _kpow(1.0, -1), _kpow(1.0, 0)),    # k1/k2
        (_kpow(1.0, 1), _kpow(1.0, 0), _kpow(1.0, -1)),    # k1/k3
        (_kpow(1.0, -1), _kpow(1.0, 1), _kpow(1.0, 0)),    # k2/k1
        (_kpow(1.0, 0), _kpow(1.0, 1), _kpow(1.0, -1)),    # k2/k3
        (_kpow(1.0, -1), _kpow(1.0, 0), _kpow(1.0, 1)),    # k3/k1
        (_kpow(1.0, 0), _kpow(1.0, -1), _kpow(1.0, 1)),    # k3/k2
        (_kpow(-1.0, 2), _kpow(1.0, -1), _kpow(1.0, -1)),  # -k1^2/(k2 k3)
        (_kpow(1.0, -1), _kpow(-1.0, 2), _kpow(1.0, -1)),  # -k2^2/(k1 k3)
        (_kpow(1.0, -1), _kpow(1.0, -1), _kpow(-1.0, 2)),  # -k3^2/(k1 k2)
        (_kpow(-2.0, 0), _kpow(1.0, 0), _kpow(1.0, 0)),    # -2
    ]


def equilateral_shape_value(k1, k2, k3):
    """Closed-form S^eq = (kt1 kt2 kt3)/(k1 k2 k3) for validation (1 at equilateral)."""
    kt1, kt2, kt3 = k2 + k3 - k1, k1 + k3 - k2, k1 + k2 - k3
    return (kt1 * kt2 * kt3) / (k1 * k2 * k3)


def multiply(t1, t2):
    """Separable template for the product B1*B2 (distribute slot-wise term-by-term).

    The product of two separable sums is separable, so combining two shapes/
    amplitudes/oscillations is just N1*N2 terms. E.g. an equilateral-shaped
    oscillatory feature with the bare-feature amplitude but squeezed-suppressed:
    ``multiply(feature(n_s, omega), equilateral_shape())``.
    """
    def _m(a, b):
        return lambda k: a(k) * b(k)

    return [
        (_m(x[0], y[0]), _m(x[1], y[1]), _m(x[2], y[2]))
        for x in t1 for y in t2
    ]


def modulate_sin(template, omega, phase=0.0):
    """Multiply a separable template by sin(omega (k1+k2+k3) + phase).

        B_new = B_base * sin(omega (k1+k2+k3) + phase),  still separable.

    The product of two separable sums is separable, so this just distributes the
    sin-of-a-sum expansion over the base terms: N_new = N_base * (4 if phase==0
    else 8). Useful to give an oscillation to a base shape while *inheriting its
    squeezed behaviour* -- e.g. ``modulate_sin(equilateral(n_s), omega)`` oscillates
    but is suppressed in the squeezed limit (unlike the bare feature template, whose
    [P1 P2 P3]^{2/3} grows as k1^{-2}), consistent with single-field consistency
    relations.
    """
    cphi, sphi = math.cos(phase), math.sin(phase)
    osc = [
        (cphi, "s", "c", "c"), (cphi, "c", "s", "c"), (cphi, "c", "c", "s"),
        (-cphi, "s", "s", "s"),
        (sphi, "c", "c", "c"), (-sphi, "c", "s", "s"),
        (-sphi, "s", "c", "s"), (-sphi, "s", "s", "c"),
    ]

    def _mk(g, kind, coeff):
        trig = jnp.sin if kind == "s" else jnp.cos
        return lambda k: coeff * g(k) * trig(omega * k)

    out = []
    for (g1, g2, g3) in template:
        for coeff, a, b, c in osc:
            if coeff != 0.0:
                out.append((_mk(g1, a, coeff), _mk(g2, b, 1.0), _mk(g3, c, 1.0)))
    return out


def template_bispectrum(shape, P1, P2, P3):
    """Analytic target B/f_NL for a named template at given side powers P=P_Phi.

    Closed form (Eqs. 2.2-2.4) for cross-checking measured bispectra; P1,P2,P3 are
    the physical potential power spectra at the three sides. Equilateral (P1=P2=P3)
    gives 6 P^2 for all three shapes; they differ in the squeezed limit.
    """
    pair = P1 * P2 + P2 * P3 + P1 * P3
    p222 = (P1 * P2 * P3) ** (2.0 / 3.0)
    perm = (
        P1 ** (1 / 3) * P2 ** (2 / 3) * P3 + P1 ** (1 / 3) * P3 ** (2 / 3) * P2
        + P2 ** (1 / 3) * P1 ** (2 / 3) * P3 + P2 ** (1 / 3) * P3 ** (2 / 3) * P1
        + P3 ** (1 / 3) * P1 ** (2 / 3) * P2 + P3 ** (1 / 3) * P2 ** (2 / 3) * P1
    )
    if shape == "local":
        return 2.0 * pair
    if shape == "equilateral":
        return 6.0 * (-pair - 2.0 * p222 + perm)
    if shape == "orthogonal":
        return 6.0 * (-3.0 * pair - 8.0 * p222 + 3.0 * perm)
    raise ValueError(f"unknown shape: {shape}")


def feature_bispectrum(P1, P2, P3, k1, k2, k3, omega, phase=0.0):
    """Closed-form oscillatory target B/f_NL = 6 (P1 P2 P3)^{2/3} sin(omega(k1+k2+k3)+phase).

    Unlike the local/equilateral/orthogonal templates this depends on the k's
    (not only the P's), so it has its own evaluator. At equilateral it reduces to
    6 P(k)^2 sin(3 omega k + phase).
    """
    return 6.0 * (P1 * P2 * P3) ** (2.0 / 3.0) * jnp.sin(omega * (k1 + k2 + k3) + phase)
