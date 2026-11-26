"""tCOLA particle-mesh forward model.

Ported from BorgJaxPM (``_simulation_cola.py``, ``_src/pm.py``,
``_src/painting.py``; CNRS / Differentiable Universe Initiative, G. Lavaux and
S. Trusov, CeCILL-B and MIT, itself derived from JaxPM) and untied from BORG:
no ``aquila_borg``, ``jax_cosmo``, ``jaxpm``, ``jaxdecomp`` or device
sharding. Background quantities and FFT conventions are miniflims' own.

Algorithm (temporal COLA, Tassev et al. 2013). Particles are evolved in the
frame comoving with their Zel'dovich trajectories, so the particle mesh only
has to integrate the residual:

    x(a)  = q + D(a) psi_1 + residual
    kick:   v  += 3/2 Om [ K F_pm(x) - K D(a_drift) psi_1 ],   K = ∫ da / (a^2 E)
    drift:  x  += v ∫ da / (a^3 E) + [D(a_end) - D(a_begin)] psi_1

with a drift-kick-drift leapfrog on steps uniform in a. ``v`` is the residual
canonical momentum (dx/da = v / (a^3 E)), zero at the start. PM forces use a
CIC density on the force mesh, the finite-difference Poisson kernel
-1 / sum_i 4 sin^2(k_i/2), the finite-difference gradient i sin(k_i), and CIC
interpolation back to the particles.

Memory of the gradient (``cola.low_memory``, on by default; same design as
the original). Reverse mode through n steps must keep the particle state of
every step, 2 x 3 x Np numbers each; everything else is avoidable:

- ``paint`` / ``read`` have hand-written adjoints. Autodiff of a CIC loop
  keeps weights and indices for each of the 8 corners; the analytic adjoint
  recomputes them from the positions inside a ``lax.scan``.
- ``pm_forces`` has a custom VJP whose forward pass saves only the particle
  positions. The backward pass rebuilds density and potential, then
  back-propagates the three force components one after the other, so only
  one force-mesh-sized gradient field is alive at a time. No mesh is stored
  per time step.
- the time loop is a single ``lax.scan`` (the original unrolls it in
  Python). XLA plans buffers per loop body, so the unrolled graph pays for
  the step's mesh temporaries once per step; as a scan it pays once, and
  compilation no longer grows with the number of steps.

Differences from the BorgJaxPM original:
- with ``forward.lattice_offset: 0`` the algorithm is the same operation for
  operation (tests compare against the original run unmodified). The default
  places the Lagrangian lattice half a force-mesh cell off the nodes, see
  ``TColaForward``;
- the particle grid is the initial-condition grid (BorgJaxPM "supersampling"
  is ``n_grid / n_mesh`` here) and the force mesh is
  ``cola.force_sampling * n_grid``;
- multi-device sharding and the float32 cancellation guards (float64 upcasts
  around the kick and drift) are not ported; use ``precision: float64``.
"""
from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .cosmology import Cosmology, efunc, growth_factor
from .forward import FORWARD_MODELS, ForwardModel, _los_unit, cic_paint, cic_read, zeldovich_displacement
from .grid import Grid

# --------------------------------------------------------------------------- #
# CIC operators with analytic adjoints. Positions are in mesh cells.
# --------------------------------------------------------------------------- #
_CORNERS = np.array(
    [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], np.int32
)


def _split(p):
    floor = jnp.floor(p)
    return floor.astype(jnp.int32), p - floor


def _corner(i0, frac, corner, n):
    """Per-axis weights (Np, 3) and wrapped node indices of one cell corner."""
    w = jnp.where(corner == 1, frac, 1.0 - frac)
    return w, jnp.mod(i0 + corner, n)


# The loops over the 8 cell corners are unrolled on purpose. BorgJaxPM runs
# them as ``lax.scan``; XLA then gives each loop its own carry buffers and
# does not share them between time steps, which costs one force mesh plus
# several particle arrays per step. Unrolled, the updates are in-place
# scatter-adds on buffers that are reused from step to step.
@partial(jax.custom_vjp, nondiff_argnums=(1,))
def paint(p, n: int):
    """Sum of CIC weights on an (n, n, n) periodic mesh; p: (Np, 3) cells."""
    i0, frac = _split(p)
    mesh = jnp.zeros((n, n, n), p.dtype)
    for corner in _CORNERS:
        w, idx = _corner(i0, frac, corner, n)
        mesh = mesh.at[idx[:, 0], idx[:, 1], idx[:, 2]].add(w[:, 0] * w[:, 1] * w[:, 2])
    return mesh


def _paint_fwd(p, n):
    return paint(p, n), p


def _position_adjoint(p, n, field, g):
    """d/dp of sum_particles g * (CIC interpolation of ``field`` at p)."""
    i0, frac = _split(p)
    out = jnp.zeros_like(p)
    for corner in _CORNERS:
        w, idx = _corner(i0, frac, corner, n)
        dw = (2 * corner - 1).astype(float)  # d w / d frac per axis
        val = field[idx[:, 0], idx[:, 1], idx[:, 2]] * g
        grad = jnp.stack(
            [dw[0] * w[:, 1] * w[:, 2], w[:, 0] * dw[1] * w[:, 2], w[:, 0] * w[:, 1] * dw[2]], axis=-1
        )
        out = out + val[:, None] * grad
    return out


def _paint_bwd(n, p, g_mesh):
    return (_position_adjoint(p, n, g_mesh, jnp.ones(p.shape[0], p.dtype)),)


paint.defvjp(_paint_fwd, _paint_bwd)


@jax.custom_vjp
def read(mesh, p):
    """CIC interpolation of ``mesh`` at p: (Np, 3) cells -> (Np,)."""
    n = mesh.shape[0]
    i0, frac = _split(p)
    out = jnp.zeros(p.shape[0], mesh.dtype)
    for corner in _CORNERS:
        w, idx = _corner(i0, frac, corner, n)
        out = out + mesh[idx[:, 0], idx[:, 1], idx[:, 2]] * (w[:, 0] * w[:, 1] * w[:, 2])
    return out


def _read_fwd(mesh, p):
    return read(mesh, p), (mesh, p)


def _read_bwd(res, g):
    mesh, p = res
    n = mesh.shape[0]
    i0, frac = _split(p)
    g_mesh = jnp.zeros_like(mesh)
    for corner in _CORNERS:
        w, idx = _corner(i0, frac, corner, n)
        g_mesh = g_mesh.at[idx[:, 0], idx[:, 1], idx[:, 2]].add(g * w[:, 0] * w[:, 1] * w[:, 2])
    return g_mesh, _position_adjoint(p, n, mesh, g)


read.defvjp(_read_fwd, _read_bwd)


# --------------------------------------------------------------------------- #
# Particle-mesh forces
# --------------------------------------------------------------------------- #
def pm_force_kernels(n: int):
    """Finite-difference Poisson and gradient kernels on an n^3 mesh in
    separable form (rfftn layout, wavenumbers in radians per cell).

    Returns ``(s2, grads)``: three broadcastable 1D arrays 4 sin^2(k_i/2)
    whose sum is the discrete Laplacian, and three 1D arrays i sin(k_i).
    Only O(n) numbers are stored; the n^3 kernel is formed on the fly, so the
    compiled program does not carry one mesh-sized constant per use.
    """
    w = 2 * np.pi * np.fft.fftfreq(n)
    wz = 2 * np.pi * np.fft.rfftfreq(n)
    ws = (w[:, None, None], w[None, :, None], wz[None, None, :])
    s2 = tuple(4 * np.sin(wi / 2) ** 2 for wi in ws)
    grads = tuple(1j * np.sin(wi) for wi in ws)
    return s2, grads


def solve_poisson(rho_k, s2):
    """rho_k * (-1 / sum_i 4 sin^2(k_i/2)), zero at k = 0."""
    kk = jnp.asarray(s2[0]) + jnp.asarray(s2[1]) + jnp.asarray(s2[2])
    return -rho_k * jnp.where(kk == 0, 0.0, 1.0 / jnp.where(kk == 0, 1.0, kk))


def pm_forces(pos, n_force: int, box: float, kernels):
    """Particle-mesh force -grad(laplacian^-1 delta) at the particles, in
    force-mesh cells. pos: (Np, 3) in Mpc/h. Plain-autodiff version."""
    s2, grads = kernels
    pot_k = solve_poisson(jnp.fft.rfftn(cic_paint(pos, n_force, box)), s2)
    shape = (n_force,) * 3
    comps = [
        cic_read(jnp.fft.irfftn(-jnp.asarray(g) * pot_k, s=shape), pos, box) for g in grads
    ]
    return jnp.stack(comps, axis=-1)


def make_pm_forces_low_memory(n_force: int, box: float, kernels):
    """Same function as ``pm_forces`` with a memory-lean reverse mode: the
    forward pass saves the positions only; the backward pass rebuilds the
    potential and handles one force component at a time."""
    s2, grads = kernels
    shape = (n_force,) * 3
    scale = n_force / box

    def potential(p):
        rho = paint(p, n_force) * (n_force**3 / p.shape[0])
        return solve_poisson(jnp.fft.rfftn(rho), s2)

    def component(pot_k, p, i):
        return read(jnp.fft.irfftn(-jnp.asarray(grads[i]) * pot_k, s=shape), p)

    def impl(pos):
        p = pos * scale
        pot_k = potential(p)
        return jnp.stack([component(pot_k, p, i) for i in range(3)], axis=-1)

    @jax.custom_vjp
    def forces(pos):
        return impl(pos)

    def fwd(pos):
        return impl(pos), pos

    def bwd(pos, g):
        p = pos * scale
        pot_k, vjp_potential = jax.vjp(potential, p)
        pot_bar = jnp.zeros_like(pot_k)
        p_bar = jnp.zeros_like(p)
        for i in range(3):
            _, vjp_i = jax.vjp(partial(component, i=i), pot_k, p)
            a, b = vjp_i(g[:, i])
            pot_bar = pot_bar + a
            p_bar = p_bar + b
        p_bar = p_bar + vjp_potential(pot_bar)[0]
        return (p_bar * scale,)

    forces.defvjp(fwd, bwd)
    return forces


# --------------------------------------------------------------------------- #
# Time integration
# --------------------------------------------------------------------------- #
class TColaCoefficients(NamedTuple):
    """One entry per leapfrog operation (drift or kick; unused ones are 0)."""

    drift: tuple  # ∫ da / (a^3 E) over the drift
    kick: tuple  # ∫ da / (a^2 E) over the kick
    cola_kick: tuple  # kick * D(a of the positions used for the forces)
    growth_drift: tuple  # D(a_end) - D(a_begin) over the drift


_GL_X, _GL_W = np.polynomial.legendre.leggauss(32)


def _integrate(fn, lo: float, hi: float) -> float:
    x = 0.5 * (hi - lo) * _GL_X + 0.5 * (hi + lo)
    return float(0.5 * (hi - lo) * np.sum(_GL_W * np.array([fn(v) for v in x])))


def tcola_coefficients(cosmo: Cosmology, a_start: float, a_final: float, n_steps: int) -> TColaCoefficients:
    """Drift-kick-drift programme: D(1/2), [K(1), D(1)] * (n-1), K(1), D(1/2)."""
    if n_steps < 1:
        raise ValueError("cola.n_steps must be at least 1")
    if not 0 < a_start < a_final:
        raise ValueError("need 0 < a_start < a_final (cola.z_start > forward.redshift)")
    e = lambda a: float(efunc(a, cosmo))
    d1 = lambda a: float(growth_factor(a, cosmo))
    programme = [("D", 0.5)] + [("K", 1.0), ("D", 1.0)] * (n_steps - 1) + [("K", 1.0), ("D", 0.5)]
    da = (a_final - a_start) / n_steps
    now = {"K": a_start, "D": a_start}
    drift, kick, cola_kick, growth_drift = [], [], [], []
    for op, frac in programme:
        a0, a1 = now[op], now[op] + frac * da
        if op == "K":
            k = _integrate(lambda a: 1.0 / (a**2 * e(a)), a0, a1)
            kick.append(k)
            cola_kick.append(k * d1(now["D"]))  # forces are evaluated at the last drift time
            drift.append(0.0)
            growth_drift.append(0.0)
        else:
            drift.append(_integrate(lambda a: 1.0 / (a**3 * e(a)), a0, a1))
            growth_drift.append(d1(a1) - d1(a0))
            kick.append(0.0)
            cola_kick.append(0.0)
        now[op] = a1
    assert abs(now["D"] - a_final) < 1e-9 and abs(now["K"] - a_final) < 1e-9
    return TColaCoefficients(tuple(drift), tuple(kick), tuple(cola_kick), tuple(growth_drift))


@FORWARD_MODELS.register("tcola")
class TColaForward(ForwardModel):
    """Zel'dovich initial conditions at ``cola.z_start`` followed by
    ``cola.n_steps`` tCOLA steps to ``forward.redshift``.

    By default the Lagrangian lattice sits half a *force-mesh* cell off the
    nodes (``lattice_offset = 0.5 / force_sampling`` initial-grid cells).
    Particles that start exactly on force-mesh nodes sit on the kink of the
    CIC kernel, which makes the PM force respond non-linearly to small
    displacements: a single plane wave then misses the exact Zel'dovich
    solution by 10-20 % however small its amplitude (see tests).
    """

    def __init__(self, cfg, grid: Grid, mesh: Grid, cosmo: Cosmology):
        super().__init__(cfg, grid, mesh, cosmo)
        if grid.n % mesh.n != 0:
            raise ValueError("n_mesh must divide n_grid")
        c = cfg.cola
        if c.force_sampling < 1:
            raise ValueError("cola.force_sampling must be at least 1")
        self.lattice_offset = (
            0.5 / c.force_sampling if cfg.lattice_offset is None else float(cfg.lattice_offset)
        )
        self.low_memory = c.low_memory
        self.a_start = 1.0 / (1.0 + c.z_start)
        self.a_final = 1.0 / (1.0 + cfg.redshift)
        self.coefficients = tcola_coefficients(cosmo, self.a_start, self.a_final, c.n_steps)
        self.growth_start = float(growth_factor(self.a_start, cosmo))
        self.e_final = float(efunc(self.a_final, cosmo))
        self.n_force = c.force_sampling * grid.n
        self.kernels = pm_force_kernels(self.n_force)
        gx, gy, gz = grid.coords()
        q = np.stack(np.broadcast_arrays(gx, gy, gz), axis=-1).reshape(-1, 3)
        self.q = jnp.asarray(q + self.lattice_offset * grid.cell)
        if self.low_memory:
            self._forces = make_pm_forces_low_memory(self.n_force, grid.box, self.kernels)
        else:
            self._forces = lambda pos: pm_forces(pos, self.n_force, grid.box, self.kernels)

    def simulate(self, delta_lin0):
        """Returns (positions, momenta, psi_1): comoving positions in Mpc/h
        (not wrapped), total canonical momentum in Mpc/h (dx/da = p/(a^3 E))
        and the Zel'dovich displacement at D = 1."""
        om = self.cosmo.omega_m
        co = self.coefficients
        psi = zeldovich_displacement(delta_lin0, self.grid, self.lattice_offset).reshape(-1, 3)
        # a force of one force-mesh cell is a displacement of box / n_force
        to_mpc = self.grid.box / self.n_force

        pos = self.q.astype(psi.dtype) + self.growth_start * psi
        vel = jnp.zeros_like(psi)  # residual momentum in the LPT frame
        if not self.low_memory:
            for i in range(len(co.drift)):
                if co.kick[i] > 0:
                    force = self._forces(pos) * to_mpc
                    vel = vel + 1.5 * om * (co.kick[i] * force - co.cola_kick[i] * psi)
                pos = pos + co.drift[i] * vel + co.growth_drift[i] * psi
        else:
            # Same programme as one scan over (kick, drift) pairs after the
            # opening half drift. The step body is compiled once and its mesh
            # buffers exist once; reverse mode stacks only what the body
            # saves per step: the particle state and, through the custom
            # force VJP, the positions.
            dtype = psi.dtype
            pos = pos + co.growth_drift[0] * psi  # vel = 0 during the first half drift
            steps = tuple(
                jnp.asarray(c, dtype)
                for c in (co.kick[1::2], co.cola_kick[1::2], co.drift[2::2], co.growth_drift[2::2])
            )

            def step(carry, c):
                x, v = carry
                kick, cola_kick, drift, growth_drift = c
                v = v + 1.5 * om * (kick * self._forces(x) * to_mpc - cola_kick * psi)
                x = x + drift * v + growth_drift * psi
                return (x, v), None

            (pos, vel), _ = jax.lax.scan(step, (pos, vel), steps)
        # add back the Zel'dovich momentum a^2 f E D psi
        p_lpt = self.a_final**2 * self.growth_rate * self.e_final * self.growth * psi
        return pos, vel + p_lpt, psi

    def positions(self, delta_lin0):
        pos, mom, _ = self.simulate(delta_lin0)
        rsd = self.cfg.rsd
        if rsd.enabled:
            n = _los_unit(rsd, pos, self.grid.box)
            shift = mom / (self.a_final**2 * self.e_final)  # v / (aH) = p / (a^2 E)
            pos = pos + jnp.sum(shift * n, axis=-1, keepdims=True) * n
        return pos

    def __call__(self, delta_lin0):
        pos = self.positions(delta_lin0)
        if self.low_memory:
            n = self.mesh.n
            return paint(pos * (n / self.mesh.box), n) * (n**3 / pos.shape[0])
        return cic_paint(pos, self.mesh.n, self.mesh.box)
