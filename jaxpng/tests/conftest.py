import os

import pytest

from jaxpng.config import Box
from jaxpng import cosmology as C

TKFILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "gengars", "inputs", "planck18_transfer.dat"
)


@pytest.fixture(scope="session")
def cosmo():
    if not os.path.exists(TKFILE):
        pytest.skip(f"transfer file not found: {TKFILE}")
    return C.make_cosmo(
        TKFILE,
        omega_m=0.3175,
        omega_l=0.6825,
        omega_b=0.049,
        hubble=0.6711,
        n_s=0.9624,
        sigma8=0.834,
        sigma8_norm=True,
    )


@pytest.fixture(scope="session")
def box():
    return Box(N=64, L=1000.0)
