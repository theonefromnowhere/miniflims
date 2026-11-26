import numpy as np
import jax.numpy as jnp

from jaxpng import io_c


def test_roundtrip(tmp_path):
    """write -> read returns the identical k-space field, ng and dk."""
    ng = 16
    rng = np.random.RandomState(0)
    real = rng.standard_normal((ng, ng, ng))
    field_k = jnp.fft.rfftn(jnp.asarray(real))  # (ng, ng, ng//2+1) complex128
    dk = 2.0 * np.pi / 1000.0

    path = str(tmp_path / "phi.dat")
    io_c.write_c_field(path, field_k, dk)
    ng2, dk2, field_k2 = io_c.read_c_field(path)

    assert ng2 == ng
    assert np.isclose(dk2, dk)
    assert field_k2.shape == (ng, ng, ng // 2 + 1)
    np.testing.assert_allclose(
        np.asarray(field_k2), np.asarray(field_k), rtol=0, atol=1e-12
    )


def test_header_and_size(tmp_path):
    """File size = int32 + float64 + ng*ng*(ng+2) doubles (FFTW padded layout)."""
    ng = 8
    field_k = jnp.fft.rfftn(jnp.zeros((ng, ng, ng)))
    path = tmp_path / "z.dat"
    io_c.write_c_field(str(path), field_k, 1.0)
    expected = 4 + 8 + ng * ng * (ng + 2) * 8
    assert path.stat().st_size == expected


def test_read_matches_raw_numpy(tmp_path):
    """Cross-check the reader against a hand-written raw binary file."""
    ng = 4
    rng = np.random.RandomState(1)
    real = np.ascontiguousarray(rng.standard_normal((ng, ng, ng + 2)))
    dk = 0.123
    path = tmp_path / "raw.dat"
    with open(path, "wb") as f:
        np.array([ng], dtype=np.int32).tofile(f)
        np.array([dk], dtype=np.float64).tofile(f)
        real.tofile(f)

    ng2, dk2, field_k = io_c.read_c_field(str(path))
    expected = real.view(np.complex128).reshape(ng, ng, ng // 2 + 1)
    assert ng2 == ng and np.isclose(dk2, dk)
    np.testing.assert_allclose(np.asarray(field_k), expected, rtol=0, atol=0)
