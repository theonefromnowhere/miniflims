"""Tests for the equilateral & orthogonal templates (jaxpng.bispectra).

The primary check is exact and field-free: the separable factors must sum to the
closed-form template at arbitrary k. A modest field test confirms the generator
reproduces the equilateral normalization (all shapes give 6 f_NL P^2).
"""
import numpy as np
import jax
import jax.numpy as jnp

from jaxpng import bispectra as B, gaussian, phing, measure, cosmology as C
from jaxpng.grids import P_kernel
from jaxpng.config import Box

_BUILDERS = {"local": B.local, "equilateral": B.equilateral, "orthogonal": B.orthogonal}


def test_separable_matches_closedform():
    """sum_i b1^i(k1) b2^i(k2) b3^i(k3) == template_bispectrum(shape, P1,P2,P3) exactly."""
    n_s = 0.96
    rng = np.random.RandomState(0)
    for _ in range(6):
        k1, k2, k3 = rng.uniform(0.01, 0.25, size=3)
        P = [float(P_kernel(jnp.asarray(k), n_s)) for k in (k1, k2, k3)]
        for shape, builder in _BUILDERS.items():
            terms = builder(n_s)
            s = float(sum(t[0](k1) * t[1](k2) * t[2](k3) for t in terms))
            ref = B.template_bispectrum(shape, P[0], P[1], P[2])
            assert np.isclose(s, ref, rtol=1e-9), (shape, s, ref)


def test_all_templates_equal_at_equilateral():
    """At k1=k2=k3 every template gives 6 P^2 (standard normalization)."""
    P = 0.7
    for shape in _BUILDERS:
        assert np.isclose(B.template_bispectrum(shape, P, P, P), 6.0 * P ** 2, rtol=1e-12)


def test_squeezed_shapes_distinct():
    """Templates differ in the squeezed limit (sign/magnitude), unlike at equilateral."""
    Plong, Pshort = 50.0, 1.0  # P(k1)>>P(k2)=P(k3): squeezed
    vals = {s: B.template_bispectrum(s, Plong, Pshort, Pshort) for s in _BUILDERS}
    assert vals["local"] > 0 and vals["local"] > 5 * abs(vals["equilateral"])  # local enhanced
    assert vals["orthogonal"] < 0                                              # orthogonal flips sign


def test_modulate_sin_equals_template_times_sin():
    """modulate_sin(equilateral) sum == equilateral closed-form * sin(omega Sum k)."""
    n_s, omega = 0.96, 20.0
    rng = np.random.RandomState(1)
    tmpl = B.modulate_sin(B.equilateral(n_s), omega)
    for _ in range(5):
        k1, k2, k3 = rng.uniform(0.02, 0.2, size=3)
        P = [float(P_kernel(jnp.asarray(k), n_s)) for k in (k1, k2, k3)]
        s = float(sum(t[0](k1) * t[1](k2) * t[2](k3) for t in tmpl))
        ref = B.template_bispectrum("equilateral", *P) * np.sin(omega * (k1 + k2 + k3))
        assert np.isclose(s, ref, rtol=1e-9)


def test_equilateral_shape_separable_and_value():
    """S^eq = 1 at equilateral; separable factors match the closed form."""
    assert np.isclose(float(B.equilateral_shape_value(0.1, 0.1, 0.1)), 1.0, rtol=1e-12)
    rng = np.random.RandomState(2)
    shape = B.equilateral_shape()
    for _ in range(5):
        k1, k2, k3 = rng.uniform(0.02, 0.2, size=3)
        s = float(sum(t[0](k1) * t[1](k2) * t[2](k3) for t in shape))
        assert np.isclose(s, float(B.equilateral_shape_value(k1, k2, k3)), rtol=1e-9)


def test_scale_invariant_equivalence():
    """At n_s=1: feature x S^eq_geometric == equilateral_template x sin (Planck Eq. 18)."""
    omega = 20.0
    rng = np.random.RandomState(3)
    tg = B.multiply(B.feature(1.0, omega), B.equilateral_shape())
    tt = B.modulate_sin(B.equilateral(1.0), omega)
    for _ in range(5):
        k1, k2, k3 = rng.uniform(0.02, 0.2, size=3)
        sg = float(sum(t[0](k1) * t[1](k2) * t[2](k3) for t in tg))
        st = float(sum(t[0](k1) * t[1](k2) * t[2](k3) for t in tt))
        assert np.isclose(sg, st, rtol=1e-9)


def test_equilateral_normalization_field(cosmo):
    """Generated equil/orthog fields: B(k,k,k) ~ 6 f_NL P_Phi^2 (paired subtraction)."""
    box = Box(N=32, L=1000.0)
    n_s, fNL, nreal = cosmo.n_s, 100.0, 12
    ks = [0.04, 0.07]
    th = measure.EQUILATERAL_THETA
    keys = jax.random.split(jax.random.PRNGKey(5), nreal)
    Btar = 6.0 * fNL * np.asarray(C.power_phi(cosmo, jnp.array(ks))) ** 2

    for shape in ("equilateral", "orthogonal"):
        tmpl = _BUILDERS[shape](n_s)
        dB = np.zeros((nreal, len(ks)))
        for r, key in enumerate(keys):
            ww = jax.random.normal(key, box.rshape)
            pg = gaussian.phiG_from_white_noise(ww, cosmo, box)
            phi = pg + fNL * phing.phiNG(pg, tmpl, box, n_s, per_efold=20.0)
            for j, k in enumerate(ks):
                dB[r, j] = float(
                    measure.bispectrum(phi, box.L, k, k, th)["B"][0]
                    - measure.bispectrum(pg, box.L, k, k, th)["B"][0]
                )
        ratio = dB.mean(0) / Btar
        assert np.all(np.abs(ratio - 1.0) < 0.3), (shape, ratio)
