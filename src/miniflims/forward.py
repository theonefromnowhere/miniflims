"""Gravity / structure-formation forward models.

A forward model maps the z=0 linear density contrast on the initial-condition
grid to the evolved density ``rho = 1 + delta`` on the mesh. Builders are
registered in ``FORWARD_MODELS`` and receive ``(cfg, grid, mesh, cosmo)``.
"""
from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np

from .cosmology import Cosmology, growth_factor, growth_rate
from .grid import Grid
from .registry import Registry

FORWARD_MODELS = Registry("forward model")


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def cic_paint(pos, n_mesh: int, box: float):
    """Cloud-in-cell mass assignment on a periodic mesh.

    pos: (Np, 3) positions in Mpc/h. Returns rho = 1 + delta, shape (n_mesh,)*3,
    with mean exactly 1. Mesh nodes sit at integer multiples of the cell size,
    so a particle on a node deposits all its mass there. Differentiable in pos.
    """
    p = pos / box * n_mesh
    i0 = jnp.floor(p)
    frac = p - i0
    i0 = i0.astype(jnp.int32)
    mesh = jnp.zeros(n_mesh**3, dtype=pos.dtype)
    for shift in itertools.product((0, 1), repeat=3):
        d = jnp.asarray(shift)
        w = jnp.prod(jnp.where(d == 1, frac, 1.0 - frac), axis=-1)
        idx = jnp.mod(i0 + d, n_mesh)
        flat = (idx[:, 0] * n_mesh + idx[:, 1]) * n_mesh + idx[:, 2]
        mesh = mesh.at[flat].add(w)
    return mesh.reshape(n_mesh, n_mesh, n_mesh) * (n_mesh**3 / pos.shape[0])


def cic_read(mesh, pos, box: float):
    """Cloud-in-cell interpolation of a periodic mesh at particle positions
    (the adjoint of ``cic_paint`` up to its normalisation).

    mesh: (n, n, n); pos: (Np, 3) in Mpc/h. Returns (Np,).
    """
    n = mesh.shape[0]
    p = pos / box * n
    i0 = jnp.floor(p)
    frac = p - i0
    i0 = i0.astype(jnp.int32)
    out = jnp.zeros(pos.shape[0], dtype=mesh.dtype)
    for shift in itertools.product((0, 1), repeat=3):
        d = jnp.asarray(shift)
        w = jnp.prod(jnp.where(d == 1, frac, 1.0 - frac), axis=-1)
        idx = jnp.mod(i0 + d, n)
        out = out + w * mesh[idx[:, 0], idx[:, 1], idx[:, 2]]
    return out


def zeldovich_displacement(delta, grid: Grid, offset: float = 0.0):
    """Zel'dovich displacement psi with delta = -div(psi).

    delta: (n, n, n) linear density contrast sampled at the grid nodes.
    Returns (n, n, n, 3) in Mpc/h, evaluated at the nodes shifted by
    ``offset`` cells along every axis (band-limited Fourier interpolation;
    Nyquist modes, which cannot be shifted unambiguously, are dropped).
    """
    kx, ky, kz = grid.kvec()
    k2 = kx**2 + ky**2 + kz**2
    kernel = np.where(k2 > 0, 1.0 / np.where(k2 > 0, k2, 1.0), 0.0).astype(complex)
    if offset != 0.0:
        for kj, kj_safe in zip(grid.kvec(), grid.kvec(zero_nyquist=True)):
            phase = np.exp(1j * kj * offset * grid.cell)
            kernel = kernel * np.where((kj != 0) & (kj_safe == 0), 0.0, phase)
    dk = jnp.fft.rfftn(delta) * jnp.asarray(kernel)
    shape = (grid.n,) * 3
    comps = [
        jnp.fft.irfftn(1j * jnp.asarray(kj) * dk, s=shape)
        for kj in grid.kvec(zero_nyquist=True)
    ]
    return jnp.stack(comps, axis=-1)


def _los_unit(rsd_cfg, pos, box: float):
    """Unit line-of-sight vectors for each particle, shape (Np, 3) or (3,)."""
    axes = {"x": 0, "y": 1, "z": 2}
    if rsd_cfg.los in axes:
        return jnp.zeros(3, dtype=pos.dtype).at[axes[rsd_cfg.los]].set(1.0)
    if rsd_cfg.los == "radial":
        obs = rsd_cfg.observer if rsd_cfg.observer is not None else (box / 2,) * 3
        r = pos - jnp.asarray(obs, dtype=pos.dtype)
        norm = jnp.sqrt(jnp.sum(r**2, axis=-1, keepdims=True))
        return r / jnp.where(norm > 0, norm, 1.0)
    raise ValueError(f"unknown rsd.los '{rsd_cfg.los}' (x, y, z or radial)")


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class ForwardModel:
    """Interface. ``__call__`` must be a pure, differentiable JAX function."""

    def __init__(self, cfg, grid: Grid, mesh: Grid, cosmo: Cosmology):
        self.cfg, self.grid, self.mesh, self.cosmo = cfg, grid, mesh, cosmo
        a = 1.0 / (1.0 + cfg.redshift)
        self.growth = float(growth_factor(a, cosmo))
        self.growth_rate = float(growth_rate(a, cosmo))

    def __call__(self, delta_lin0):  # -> rho on the mesh
        raise NotImplementedError


@FORWARD_MODELS.register("linear")
class LinearForward(ForwardModel):
    """rho = 1 + D(z) delta_lin. Mostly for tests (analytic posterior)."""

    def __init__(self, cfg, grid, mesh, cosmo):
        super().__init__(cfg, grid, mesh, cosmo)
        if mesh.n != grid.n:
            raise ValueError("forward.model=linear requires n_mesh == n_grid")
        if cfg.rsd.enabled:
            raise ValueError("forward.model=linear does not support RSD")

    def __call__(self, delta_lin0):
        return 1.0 + self.growth * delta_lin0


@FORWARD_MODELS.register("zeldovich")
class ZeldovichForward(ForwardModel):
    """First-order LPT: x = q + D psi(q), optional redshift-space shift
    f D (psi . n) n, then CIC assignment.

    The linear field, the mesh and the survey response are all sampled at the
    nodes j * cell. The Lagrangian lattice is offset by half an initial-grid
    cell, q = (i + 1/2) * cell: particles then never start on a mesh node,
    where the CIC kernel has a kink, so the density responds linearly (and
    differentiably) to small displacements.
    """

    def __init__(self, cfg, grid, mesh, cosmo):
        super().__init__(cfg, grid, mesh, cosmo)
        self.lattice_offset = 0.5 if cfg.lattice_offset is None else float(cfg.lattice_offset)
        if grid.n % mesh.n != 0:
            raise ValueError(
                "n_mesh must divide n_grid (otherwise the unperturbed particle "
                "lattice does not paint to a uniform density)"
            )
        gx, gy, gz = grid.coords()
        q = np.stack(np.broadcast_arrays(gx, gy, gz), axis=-1).reshape(-1, 3)
        self.q = jnp.asarray(q + self.lattice_offset * grid.cell)

    def displacement(self, delta_lin0):
        """Real-space displacement of each particle at the model redshift,
        (Np, 3) Mpc/h."""
        psi = zeldovich_displacement(delta_lin0, self.grid, self.lattice_offset)
        return self.growth * psi.reshape(-1, 3)

    def positions(self, delta_lin0):
        disp = self.displacement(delta_lin0)
        pos = self.q.astype(disp.dtype) + disp
        rsd = self.cfg.rsd
        if rsd.enabled:
            n = _los_unit(rsd, pos, self.grid.box)
            # v / (aH) = f D psi
            pos = pos + self.growth_rate * jnp.sum(disp * n, axis=-1, keepdims=True) * n
        return pos

    def __call__(self, delta_lin0):
        return cic_paint(self.positions(delta_lin0), self.mesh.n, self.mesh.box)


def second_order_source(delta, grid: Grid):
    """mu_2 = sum_{i<j} (phi_ii phi_jj - phi_ij^2) of the potential with
    lap(phi) = delta, at the grid nodes (the source of the 2LPT potential).

    Wrapped in ``jax.checkpoint`` by the caller: the six second-derivative
    fields are recomputed in the backward pass instead of being stored.
    """
    shape = (grid.n,) * 3
    kfull = grid.kvec()
    ksafe = grid.kvec(zero_nyquist=True)  # odd in k_i: Nyquist modes dropped
    k2 = sum(k**2 for k in kfull)
    inv_k2 = jnp.asarray(np.where(k2 > 0, 1.0 / np.where(k2 > 0, k2, 1.0), 0.0))
    dk = jnp.fft.rfftn(delta) * inv_k2  # = -phi_k

    def d2(i, j):  # phi_ij = -k_i k_j phi_k = k_i k_j (delta_k / k^2)
        kk = kfull[i] ** 2 if i == j else ksafe[i] * ksafe[j]
        return jnp.fft.irfftn(dk * jnp.asarray(kk), s=shape)

    pxx, pyy, pzz = d2(0, 0), d2(1, 1), d2(2, 2)
    pxy, pxz, pyz = d2(0, 1), d2(0, 2), d2(1, 2)
    return pxx * pyy + pxx * pzz + pyy * pzz - pxy**2 - pxz**2 - pyz**2


@FORWARD_MODELS.register("2lpt")
class TwoLPTForward(ZeldovichForward):
    """Second-order LPT (Bouchet et al. 1995; Scoccimarro 1998):

        x = q + psi_1 + psi_2,   psi_1 = -grad phi_1,   lap phi_1 = delta_L(z),
        psi_2 = (D_2 / D_1^2) grad phi_2,   lap phi_2 = mu_2[phi_1],
        D_2 / D_1^2 = -(3/7) Omega_m(z)^(-1/143),

    so psi_2 = (3/7) Omega_m(z)^(-1/143) * zeldovich_displacement(mu_2) (the
    homogeneous sphere then collapses with delta = delta_L + (17/21) delta_L^2).
    Redshift space: v / (aH) = f_1 psi_1 + f_2 psi_2, f_2 = 2 Omega_m(z)^(6/11).
    Lattice, offset and CIC assignment as for ``zeldovich``. The gradient
    recomputes the second-order source (``jax.checkpoint``) instead of storing
    its six second-derivative fields.
    """

    def __init__(self, cfg, grid, mesh, cosmo):
        super().__init__(cfg, grid, mesh, cosmo)
        a = 1.0 / (1.0 + cfg.redshift)
        om = cosmo.omega_m / (cosmo.omega_m + (1.0 - cosmo.omega_m) * a**3)  # flat LCDM
        self.d2_factor = (3.0 / 7.0) * om ** (-1.0 / 143.0)
        self.growth_rate_2 = 2.0 * om ** (6.0 / 11.0)
        grid_ = grid
        self._source = jax.checkpoint(lambda d: second_order_source(d, grid_))

    def displacements(self, delta_lin0):
        """(psi_1, psi_2) at the model redshift, each (Np, 3) in Mpc/h."""
        d1 = self.growth * delta_lin0
        psi1 = zeldovich_displacement(d1, self.grid, self.lattice_offset)
        psi2 = self.d2_factor * zeldovich_displacement(self._source(d1), self.grid, self.lattice_offset)
        return psi1.reshape(-1, 3), psi2.reshape(-1, 3)

    def displacement(self, delta_lin0):
        psi1, psi2 = self.displacements(delta_lin0)
        return psi1 + psi2

    def positions(self, delta_lin0):
        psi1, psi2 = self.displacements(delta_lin0)
        pos = self.q.astype(psi1.dtype) + psi1 + psi2
        rsd = self.cfg.rsd
        if rsd.enabled:
            n = _los_unit(rsd, pos, self.grid.box)
            vel = self.growth_rate * psi1 + self.growth_rate_2 * psi2
            pos = pos + jnp.sum(vel * n, axis=-1, keepdims=True) * n
        return pos
