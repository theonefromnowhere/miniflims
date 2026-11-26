"""Tests for jaxpng.gaussian: Phi^G generation from white noise."""
import numpy as np
import jax
import jax.numpy as jnp

from jaxpng import gaussian, measure, cosmology as C


def test_phiG_measured_pk_matches_target(cosmo, box):
    """Measured P(k) of generated Phi^G matches power_phi(k) (mean over realizations)."""
    keys = jax.random.split(jax.random.PRNGKey(0), 16)

    def one(key):
        w = jax.random.normal(key, box.rshape)
        return measure.power_spectrum(gaussian.phiG_from_white_noise(w, cosmo, box), box.L)

    ks, Pks = jax.vmap(one)(keys)
    k = np.asarray(ks[0])
    Pmeas = np.asarray(jnp.mean(Pks, axis=0))
    Ptar = np.asarray(C.power_phi(cosmo, jnp.asarray(k)))
    sel = (k > 2 * box.dk) & (k < 0.3)
    ratio = Pmeas[sel] / Ptar[sel]
    assert abs(np.median(ratio) - 1.0) < 0.1


def test_phiG_real_and_zero_dc(cosmo, box):
    w = jax.random.normal(jax.random.PRNGKey(1), box.rshape)
    phi_k = gaussian.phiG_from_white_noise(w, cosmo, box)
    assert phi_k.shape == box.kshape
    assert abs(complex(phi_k[0, 0, 0])) == 0.0          # DC removed
    phi_x = jnp.fft.irfftn(phi_k, s=box.rshape)
    # round-trip through irfftn/rfftn is consistent (field is genuinely real)
    np.testing.assert_allclose(
        np.asarray(jnp.fft.rfftn(phi_x)), np.asarray(phi_k), atol=1e-8, rtol=1e-5
    )


def test_phiG_differentiable(cosmo, box):
    w = jax.random.normal(jax.random.PRNGKey(2), box.rshape)
    g = jax.grad(
        lambda w: jnp.sum(jnp.abs(gaussian.phiG_from_white_noise(w, cosmo, box)) ** 2)
    )(w)
    assert g.shape == box.rshape and bool(jnp.all(jnp.isfinite(g)))
