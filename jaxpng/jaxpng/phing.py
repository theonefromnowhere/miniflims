"""Non-Gaussian potential Phi^NG via the Schwinger-parameterized kernel.

The core of GENGARS (paper Eq. 2.16), ported to JAX. Generates Phi^NG(k) from a
Gaussian Phi^G(k) and a separable target bispectrum (see :mod:`jaxpng.bispectra`):

    Phi^NG(k) = sum_t  weight(t)  sum_i  w1^i(t,k) * rfftn[ irfftn(w2^i(t) Phi^G)
                                                          * irfftn(w3^i(t) Phi^G) ]
    w_j^i(t,k) = b_j^i(k) * exp(-t / P(k)) / P(k),   P(k) = k^(n_s-4).

Normalization (derived, not fitted): with the template convention
sum_i b1 b2 b3 = B_Phi/f_NL and the trapezoid weight (dt_prev+dt)/4 (the extra
1/2 vs a plain trapezoid is the kernel's 1/2 = W_true/W_C), the t-integral
reconstructs 1/(P1 P2 + P2 P3 + P1 P3) and the FFT convolution is the cyclic
convolution / N^3 = rfftn of the real-space product. Hence for the local template
(W=1) this returns exactly rfftn(Phi^G^2), and Phi^NG(x) = Phi^G(x)^2 - <Phi^G^2>
after the DC mode is zeroed. That is the field-level anchor used to validate it.

t-grid follows gengars/src/generate_phiNG.c: log-spaced, ~per_efold points per
e-fold, tmin = -ln(1-eps) k_max^-3, tmax = -ln(eps) k_min^-3, k_max = dk(N/2)sqrt3,
k_min = dk. N_t and eps are convergence knobs.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import lax

from .grids import k_grid, P_kernel


def schwinger_t_grid(box, eps=1e-3, per_efold=20.0):
    """Log-spaced Schwinger t nodes and their trapezoid(/2) weights.

    Returns ``(t, weight)`` arrays of length N_t+1. ``weight`` already includes
    the kernel's factor 1/2 (it is the plain trapezoid node weight divided by 2).
    """
    dk = box.dk
    m = box.N // 2
    kmin = dk
    kmax = dk * m * math.sqrt(3.0)
    tmin = -math.log(1.0 - eps) * kmax ** -3
    tmax = -math.log(eps) * kmin ** -3
    it_min = int(math.log(tmin))
    it_max = int(math.log(tmax))
    n_t = int((it_max - it_min) * per_efold)

    it = jnp.arange(n_t + 1)
    t = jnp.exp(it_min + it / per_efold)
    dt = jnp.exp(it_min + (it + 1) / per_efold) - t
    dt_prev = jnp.concatenate([jnp.array([2.0 * math.exp(it_min)]), dt[:-1]])
    weight = (dt_prev + dt) / 4.0
    return t, weight


def schwinger_t_grid_log(box, n_nodes=32, eps=1e-5):
    """Schwinger t nodes and weights for a trapezoid rule in u = ln t.

    The t-integral  int_0^inf dt exp(-t S)  becomes  int du e^u exp(-e^u S),
    whose integrand is smooth and decays exponentially at both ends, so the
    trapezoid rule in u converges exponentially with the number of nodes
    (against second order for the trapezoid in t of ``schwinger_t_grid``).
    The range is the same as in ``schwinger_t_grid`` for the given ``eps``.
    Measured on a 32^3 local / equilateral Phi^NG (relative L2 error):
    24 nodes, eps=1e-4: 2e-4;  32 nodes, eps=1e-5: 3e-5;  48 nodes, eps=1e-6: 7e-6
    (``schwinger_t_grid`` default: 361 nodes, 4e-4).

    Returns ``(t, weight)``; ``weight`` includes the kernel's factor 1/2 as in
    ``schwinger_t_grid``.
    """
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2")
    kmin = box.dk
    kmax = box.dk * (box.N // 2) * math.sqrt(3.0)
    u_lo = math.log(-math.log(1.0 - eps) * kmax ** -3)
    u_hi = math.log(-math.log(eps) * kmin ** -3)
    u = jnp.linspace(u_lo, u_hi, n_nodes)
    h = (u_hi - u_lo) / (n_nodes - 1)
    end = jnp.where((jnp.arange(n_nodes) == 0) | (jnp.arange(n_nodes) == n_nodes - 1), 0.5, 1.0)
    t = jnp.exp(u)
    return t, 0.5 * h * t * end


def phiNG(phiG_k, template, box, n_s, eps=1e-3, per_efold=20.0, remat=False, t_grid=None):
    """Non-Gaussian potential Phi^NG(k) (rfftn layout) for a separable template.

    Parameters
    ----------
    phiG_k : complex (N, N, N//2+1) -- Gaussian potential Phi^G in rfftn layout.
    template : list of (g1, g2, g3) separable-factor callables (see jaxpng.bispectra).
    box : Box
    n_s : float -- spectral index (sets the internal kernel P(k)=k^(n_s-4)).
    eps, per_efold : Schwinger t-grid convergence knobs.
    remat : if True, wrap the scan body in ``jax.checkpoint`` so reverse-mode
        autodiff recomputes each node's FFTs instead of storing them (gradient
        memory O(N^3) instead of O(N_t * N_terms * N^3)). Forward values unchanged.
    t_grid : optional ``(t, weight)``, e.g. from :func:`schwinger_t_grid_log`;
        overrides ``eps`` / ``per_efold``. Cost is proportional to len(t).

    Returns
    -------
    phiNG_k : complex (N, N, N//2+1) -- Phi^NG in rfftn layout (DC mode zeroed,
        i.e. zero spatial mean). Combine as Phi = Phi^G + f_NL * Phi^NG.
    """
    N = box.N
    s = (N, N, N)
    _, _, _, kmag = k_grid(box)
    invP = 1.0 / P_kernel(kmag, n_s)
    terms = [(g1(kmag), g2(kmag), g3(kmag)) for (g1, g2, g3) in template]
    t, weight = t_grid if t_grid is not None else schwinger_t_grid(box, eps, per_efold)

    def body(acc, tw):
        tl, wl = tw
        E = jnp.exp(-tl * invP) * invP  # exp(-t/P)/P
        contrib = jnp.zeros_like(phiG_k)
        for (b1, b2, b3) in terms:
            a = jnp.fft.irfftn((b2 * E) * phiG_k, s=s)
            b = jnp.fft.irfftn((b3 * E) * phiG_k, s=s)
            contrib = contrib + (b1 * E) * jnp.fft.rfftn(a * b)
        return acc + wl * contrib, None

    if remat:
        body = jax.checkpoint(body)
    phiNG_k, _ = lax.scan(body, jnp.zeros_like(phiG_k), (t, weight))
    return phiNG_k.at[0, 0, 0].set(0.0 + 0.0j)
