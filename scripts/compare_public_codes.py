"""Cross-checks of miniflims against public codes, with figures.

    .venv/bin/python scripts/compare_public_codes.py [--out figures] [--skip-sampler]

    power    Eisenstein-Hu P(k) and growth vs colossus, jax_cosmo and CAMB
    lpt      Zel'dovich displacement and CIC density vs JaxPM
    cola     tCOLA port vs the original BorgJaxPM, force mesh, lattice, memory
    sampler  HMC-within-Gibbs posterior vs blackjax NUTS on the joint posterior

Extra packages: matplotlib colossus camb jax_cosmo jaxpm blackjax
(jax_cosmo needs setuptools<81).
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from miniflims import diagnostics as diag  # noqa: E402
from miniflims import load_config, run  # noqa: E402
from miniflims.cosmology import Cosmology, growth_factor, growth_rate  # noqa: E402
from miniflims.data import build_response  # noqa: E402
from miniflims.forward import cic_paint, zeldovich_displacement  # noqa: E402
from miniflims.model import FieldModel  # noqa: E402
from miniflims.plotting import AXIS, GRID, INK, INK2, SERIES, _mpl, image, logx  # noqa: E402
from miniflims.power import linear_power  # noqa: E402

plt, SEQ, DIV = _mpl()
DASH = (0, (4, 3))


def jc_cosmology(c: Cosmology):
    import jax_cosmo as jc

    return jc.Cosmology(Omega_c=c.omega_m - c.omega_b, Omega_b=c.omega_b, h=c.h, n_s=c.n_s,
                        sigma8=c.sigma8, Omega_k=0.0, w0=-1.0, wa=0.0)


# --------------------------------------------------------------------------- #
def compare_power(c: Cosmology, out: Path) -> dict:
    import camb
    import jax_cosmo as jc
    from colossus.cosmology import cosmology as col

    k = np.logspace(-3.5, 1, 400)
    ours = np.asarray(linear_power(k, c, "eisenstein_hu"))
    ours_nw = np.asarray(linear_power(k, c, "eisenstein_hu_nowiggle"))

    ref = col.setCosmology("cmp", dict(flat=True, H0=100 * c.h, Om0=c.omega_m, Ob0=c.omega_b,
                                       sigma8=c.sigma8, ns=c.n_s, relspecies=False, Tcmb0=c.t_cmb))
    p_col = ref.matterPowerSpectrum(k, model="eisenstein98")
    p_jc = np.asarray(jc.power.linear_matter_power(jc_cosmology(c), k, a=1.0,
                                                   transfer_fn=jc.transfer.Eisenstein_Hu))

    # CAMB: massless neutrinos, same Omega_m, rescaled to the same sigma8
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100 * c.h, ombh2=c.omega_b * c.h**2, omch2=(c.omega_m - c.omega_b) * c.h**2,
                       mnu=0.0, omk=0.0, TCMB=c.t_cmb)
    pars.InitPower.set_params(As=2.1e-9, ns=c.n_s)
    zs = [0.0, 0.5, 1.0, 2.0, 4.0]
    pars.set_matter_power(redshifts=zs, kmax=20.0)
    pars.NonLinear = camb.model.NonLinear_none
    res = camb.get_results(pars)
    s8 = res.get_sigma8()  # ordered by increasing a
    interp = res.get_matter_power_interpolator(nonlinear=False, hubble_units=True, k_hunit=True)
    p_camb = interp.P(0.0, k) * (c.sigma8 / s8[-1]) ** 2

    a = 1 / (1 + np.array(zs))
    d_ours = np.array([float(growth_factor(x, c)) for x in a])
    f_ours = np.array([float(growth_rate(x, c)) for x in a])
    d_col = np.array([ref.growthFactor(z) for z in zs])
    d_camb = (s8 / s8[-1])[::-1]  # large-scale-structure growth incl. radiation
    # CAMB's f sigma8 (from the velocity-density correlation) over sigma8
    f_camb = (res.get_fsigma8() / res.get_sigma8())[::-1]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    ax = axes[0, 0]
    ax.plot(k, ours, color=SERIES[0], label="miniflims Eisenstein–Hu")
    ax.plot(k, p_camb, color=INK, lw=1.2, ls=DASH, label="CAMB (Boltzmann)")
    logx(ax)
    ax.set(yscale="log", xlabel="k  [h/Mpc]", ylabel="P(k)  [(Mpc/h)³]",
           title="Linear matter power spectrum, z = 0")
    ax.legend(loc="lower left")

    ax = axes[0, 1]
    ax.axhspan(-0.05, 0.05, color=GRID, alpha=0.6, lw=0)
    ax.plot(k, ours / p_camb - 1, color=SERIES[0], label="EH with wiggles")
    ax.plot(k, ours_nw / p_camb - 1, color=SERIES[1], label="EH no-wiggle")
    ax.axhline(0, color=INK, lw=1.0, ls=DASH)
    logx(ax)
    ax.set(ylim=(-0.12, 0.12), xlabel="k  [h/Mpc]", ylabel="P / P_CAMB − 1",
           title="Accuracy of the fitting formulae vs CAMB (band: ±5%)")
    ax.legend(loc="lower left")

    ax = axes[1, 0]
    ax.plot(k, ours / p_col - 1, color=SERIES[0], label="vs colossus eisenstein98")
    ax.plot(k, ours / p_jc - 1, color=SERIES[1], label="vs jax_cosmo Eisenstein_Hu")
    ax.axhline(0, color=INK, lw=1.0, ls=DASH)
    logx(ax)
    ax.set(xlabel="k  [h/Mpc]", ylabel="P_miniflims / P_public − 1",
           title="Same formula, other implementations")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.legend(loc="best")

    ax = axes[1, 1]
    ax.plot(zs, d_ours / d_col - 1, "o-", ms=6, color=SERIES[0], label="D(z): vs colossus")
    ax.plot(zs, d_ours / d_camb - 1, "s-", ms=6, color=SERIES[1], label="D(z): vs CAMB σ8(z)/σ8(0)")
    ax.plot(zs, f_ours / f_camb - 1, "^-", ms=6, color=SERIES[2], label="f(z): vs CAMB fσ8(z)/σ8(z)")
    ax.axhline(0, color=INK, lw=1.0, ls=DASH)
    ax.set(xlabel="redshift", ylabel="miniflims / public − 1", title="Linear growth factor and rate")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "compare_power.png")
    plt.close(fig)

    bao = (k > 0.02) & (k < 0.3)
    return {
        "max|P/P_colossus - 1|": float(np.abs(ours / p_col - 1).max()),
        "max|P/P_jax_cosmo - 1|": float(np.abs(ours / p_jc - 1).max()),
        "max|P_EH/P_CAMB - 1| (0.02<k<0.3)": float(np.abs(ours / p_camb - 1)[bao].max()),
        "max|P_EH/P_CAMB - 1| (all k)": float(np.abs(ours / p_camb - 1).max()),
        "max|P_nowiggle/P_CAMB - 1| (0.02<k<0.3)": float(np.abs(ours_nw / p_camb - 1)[bao].max()),
        "max|D/D_colossus - 1|": float(np.abs(d_ours / d_col - 1).max()),
        "max|D/D_CAMB - 1|": float(np.abs(d_ours / d_camb - 1).max()),
        "max|f/f_CAMB - 1|": float(np.abs(f_ours / f_camb - 1).max()),
    }


# --------------------------------------------------------------------------- #
def compare_lpt(out: Path, n: int = 64, box: float = 1000.0) -> dict:
    import jaxpm.painting as jpaint
    import jaxpm.pm as jpm

    cfg = load_config(overrides={"grid": {"n_grid": n, "box_size": box}})
    model = FieldModel(cfg, build_response(cfg))
    grid = model.grid
    x = jnp.asarray(np.random.default_rng(0).standard_normal((n,) * 3))
    delta = model.linear_density(x)

    # displacement at the grid nodes, in cells, at a = 1 (D = 1 in both codes)
    ours = np.asarray(zeldovich_displacement(delta, grid)) / grid.cell
    theirs = np.asarray(jpm.lpt(jc_cosmology(model.cosmo), delta, a=1.0, order=1)[0])

    # JaxPM differentiates with a 4th-order finite-difference kernel instead
    # of ik: per axis it multiplies our displacement by (8 sin w - sin 2w)/(6w)
    w = 2 * np.pi * np.fft.fftfreq(n)
    fd = np.where(w != 0, (8 * np.sin(w) - np.sin(2 * w)) / (6 * np.where(w != 0, w, 1.0)), 1.0)
    ours_fd = np.stack([
        np.fft.ifftn(np.fft.fftn(ours[..., i]) * fd.reshape([n if j == i else 1 for j in range(3)])).real
        for i in range(3)
    ], axis=-1)

    ok = np.fft.fftn(ours[..., 0])
    tk = np.fft.fftn(theirs[..., 0])
    kx = np.abs(np.broadcast_to(w[:, None, None], ok.shape))
    edges = np.linspace(0, np.pi, 17)
    idx = np.digitize(kx.ravel(), edges) - 1
    valid = (idx >= 0) & (idx < 16) & (kx.ravel() > 0) & (kx.ravel() < np.pi * (1 - 1e-9))
    num = np.bincount(idx[valid], weights=(tk * np.conj(ok)).real.ravel()[valid], minlength=16)
    den = np.bincount(idx[valid], weights=(np.abs(ok) ** 2).ravel()[valid], minlength=16)
    transfer = num / den
    wc = 0.5 * (edges[1:] + edges[:-1])

    # same particles through both painters
    q = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), axis=-1).astype(float)
    pos = q + ours
    rho_theirs = np.asarray(jpaint.cic_paint(jnp.zeros((n,) * 3), jnp.asarray(pos)))
    rho_ours = np.asarray(cic_paint(jnp.asarray(pos.reshape(-1, 3) * grid.cell), n, box))
    # each code's own Zel'dovich density (node-centred lattice for both)
    rho_jaxpm = np.asarray(jpaint.cic_paint(jnp.zeros((n,) * 3), jnp.asarray(q + theirs)))

    nb = 20
    k, p_ours, _ = diag.power_spectrum(rho_ours - 1, box, nb)
    p_jaxpm = diag.power_spectrum(rho_jaxpm - 1, box, nb)[1]
    p_lin = diag.power_spectrum(np.asarray(delta), box, nb)[1]
    r = diag.cross_correlation(rho_ours - 1, rho_jaxpm - 1, box, nb)[1]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6))
    ax = axes[0, 0]
    wfine = np.linspace(1e-3, np.pi, 200)
    ax.plot(wfine / np.pi, (8 * np.sin(wfine) - np.sin(2 * wfine)) / (6 * wfine), color=INK, lw=1.2,
            ls=DASH, label="4th-order finite-difference kernel")
    ax.plot(wc / np.pi, transfer, "o", ms=6, color=SERIES[0], label="measured  ψ_JaxPM / ψ_miniflims")
    ax.set(xlabel="k_x / k_Nyquist", ylabel="displacement ratio (x component)", ylim=(0, 1.1),
           title="Zel'dovich displacement: JaxPM vs miniflims")
    ax.legend(loc="lower left")

    ax = axes[0, 1]
    ax.plot(k, p_lin, color=INK, lw=1.2, ls=DASH, label="linear input field")
    ax.plot(k, p_ours, color=SERIES[0], label="miniflims Zel'dovich + CIC")
    ax.plot(k, p_jaxpm, color=SERIES[1], label="JaxPM LPT (order 1) + CIC")
    logx(ax)
    ax.set(yscale="log", xlabel="k  [h/Mpc]", ylabel="P(k)  [(Mpc/h)³]",
           title="Evolved density power spectrum")
    ax.legend(loc="lower left")

    ax = axes[0, 2]
    ax.plot(k, p_jaxpm / p_ours, color=SERIES[0], label="P_JaxPM / P_miniflims")
    ax.plot(k, r, color=SERIES[1], label="cross-correlation r(k)")
    ax.axhline(1.0, color=INK, lw=1.0, ls=DASH)
    logx(ax)
    ax.set(ylim=(0.8, 1.05), xlabel="k  [h/Mpc]", ylabel="ratio",
           title="Agreement of the two density fields")
    ax.legend(loc="lower left")

    ext = (0, box, 0, box)
    sl = n // 2
    a_, b_ = np.log10(np.maximum(rho_ours[:, :, sl], 1e-2)), np.log10(np.maximum(rho_jaxpm[:, :, sl], 1e-2))
    vmin, vmax = min(a_.min(), b_.min()), max(a_.max(), b_.max())
    image(axes[1, 0], a_, SEQ, ext, "log₁₀ ρ", vmin, vmax)
    axes[1, 0].set(title="miniflims", xlabel="x  [Mpc/h]", ylabel="y  [Mpc/h]")
    image(axes[1, 1], b_, SEQ, ext, "log₁₀ ρ", vmin, vmax)
    axes[1, 1].set(title="JaxPM", xlabel="x  [Mpc/h]", ylabel="y  [Mpc/h]")
    d = (rho_jaxpm - rho_ours)[:, :, sl]
    lim = np.abs(d).max()
    image(axes[1, 2], d, DIV, ext, "ρ_JaxPM − ρ_miniflims", -lim, lim)
    axes[1, 2].set(title="Difference (finite-difference vs spectral gradient)", xlabel="x  [Mpc/h]",
                   ylabel="y  [Mpc/h]")
    fig.tight_layout()
    fig.savefig(out / "compare_lpt.png")
    plt.close(fig)

    return {
        "corr(psi_jaxpm, psi_miniflims)": float(np.corrcoef(theirs.ravel(), ours.ravel())[0, 1]),
        "max|psi_jaxpm - FDkernel*psi_miniflims| / rms(psi)": float(np.abs(theirs - ours_fd).max() / ours.std()),
        "max|CIC_jaxpm - CIC_miniflims| (same particles)": float(np.abs(rho_theirs - rho_ours).max()),
        "corr(rho_jaxpm, rho_miniflims)": float(np.corrcoef(rho_jaxpm.ravel(), rho_ours.ravel())[0, 1]),
        "P_jaxpm/P_miniflims at lowest k": float(p_jaxpm[0] / p_ours[0]),
        "P_jaxpm/P_miniflims at highest k": float(p_jaxpm[-1] / p_ours[-1]),
    }


# --------------------------------------------------------------------------- #
def compare_cola(out: Path, n: int = 64, box: float = 500.0) -> dict:
    """tCOLA port: agreement with the original BorgJaxPM, effect of the force
    mesh and of the particle lattice, and memory of the gradient."""
    from miniflims.config import ColaConfig, ForwardConfig
    from miniflims.forward import FORWARD_MODELS
    from miniflims.grid import Grid

    cosmo = Cosmology.from_config(load_config().cosmology)

    def make(model, grid, offset=None, **cola):
        fc = ForwardConfig(model=model, lattice_offset=offset, cola=ColaConfig(**cola))
        return FORWARD_MODELS.get(model)(fc, grid, grid, cosmo)

    stats = {}

    # (1) against the original, on the reference shipped with the tests
    ref = np.load(Path(__file__).resolve().parents[1] / "tests" / "data" / "borgjaxpm_tcola_reference.npz")
    g16 = Grid(16, 250.0)
    port = make("tcola", g16, 0.0, n_steps=4, z_start=9.0, force_sampling=2)
    pos, _, _ = port.simulate(jnp.asarray(ref["delta"]))
    mine = np.mod(np.asarray(pos).reshape(16, 16, 16, 3).transpose(1, 0, 2, 3).reshape(-1, 3) / g16.cell, 16)
    dpos = mine - ref["pos"]
    dpos -= 16 * np.round(dpos / 16)
    total = ref["pos"] - np.stack(np.meshgrid(*[np.arange(16)] * 3), -1).reshape(-1, 3)
    total -= 16 * np.round(total / 16)
    rho_port = np.asarray(port(jnp.asarray(ref["delta"])))
    stats["max|x_port - x_BorgJaxPM| [cells]"] = float(np.abs(dpos).max())
    stats["rms displacement in that run [cells]"] = float(total.std())
    stats["max|rho_port - rho_BorgJaxPM|"] = float(np.abs(rho_port - ref["rho"]).max())

    # (2) force-mesh resolution, on a realistic field
    cfg = load_config(overrides={"grid": {"n_grid": n, "box_size": box}})
    model = FieldModel(cfg, build_response(cfg))
    grid = model.grid
    delta = model.linear_density(jnp.asarray(np.random.default_rng(0).standard_normal((n,) * 3)))
    nb = 20
    rho_za = np.asarray(make("zeldovich", grid)(delta))
    k, p_za, _ = diag.power_spectrum(rho_za - 1, box, nb)
    rho_cola, p_cola = {}, {}
    for fs in (1, 2, 4):
        rho_cola[fs] = np.asarray(jax.jit(make("tcola", grid, force_sampling=fs))(delta))
        p_cola[fs] = diag.power_spectrum(rho_cola[fs] - 1, box, nb)[1]
        stats[f"P_tcola/P_zeldovich at lowest k, force_sampling={fs}"] = float(p_cola[fs][0] / p_za[0])

    # (3) plane wave in the linear regime: exact answer is Zel'dovich
    x = np.arange(16) * g16.cell
    wave = jnp.asarray(1e-4 * np.cos(g16.k_fundamental * x)[:, None, None] * np.ones((16,) * 3))
    layouts = [(0.0, 1), (0.0, 2), (0.0, 4), (0.5, 2), (0.5, 1), (0.25, 2), (0.125, 4)]
    errors = []
    for off, fs in layouts:
        za, co = make("zeldovich", g16, off), make("tcola", g16, off, force_sampling=fs)
        d_za, d_co = np.asarray(za.positions(wave) - za.q), np.asarray(co.positions(wave) - co.q)
        errors.append(float(np.abs(d_co - d_za).max() / np.abs(d_za).max()))
    on_node = [((off * fs) % 1.0) == 0.0 for off, fs in layouts]
    stats["plane-wave displacement error by (lattice_offset, force_sampling)"] = {
        f"({off}, {fs})": e for (off, fs), e in zip(layouts, errors)}

    # (4) gradient memory as planned by XLA, 32^3 particles, 64^3 force mesh
    g32 = Grid(32, 500.0)
    steps = [2, 5, 10, 20]

    def temp_mb(mode, n_steps):
        fwd = make("tcola", g32, n_steps=n_steps, low_memory=(mode != "plain"))
        fn = fwd if mode == "forward" else jax.grad(lambda d: jnp.sum(fwd(d) ** 2))
        return jax.jit(fn).lower(jnp.zeros((32,) * 3)).compile().memory_analysis().temp_size_in_bytes / 2**20

    mem = {m: [temp_mb(m, s_) for s_ in (steps if m != "plain" else steps[:3])] for m in ("forward", "low", "plain")}
    stats["XLA temp memory [MB] vs steps " + str(steps)] = mem

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.8))
    ax = axes[0, 0]
    for fs, col in zip((1, 2, 4), SERIES):
        ax.plot(k, p_cola[fs] / p_za, color=col, label=f"force mesh = {fs} × particle grid")
    ax.axhline(1.0, color=INK, lw=1.0, ls=DASH)
    logx(ax)
    ax.set(xlabel="k  [h/Mpc]", ylabel="P_tCOLA / P_Zel'dovich", title="tCOLA vs Zel'dovich: force-mesh resolution")
    ax.legend(loc="upper left")

    ax = axes[0, 1]
    labels = [f"{off:g}\n×{fs}" for off, fs in layouts]
    cols = [SERIES[1] if o else SERIES[0] for o in on_node]
    bars = ax.bar(range(len(layouts)), [100 * e for e in errors], width=0.62, color=cols)
    for b, e in zip(bars, errors):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4, f"{100 * e:.1f}%" if e > 5e-4 else "<0.05%",
                ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(len(layouts)), labels)
    ax.grid(axis="x", visible=False)
    ax.set(xlabel="lattice offset [grid cells]  /  force sampling", ylabel="displacement error  [%]",
           title="Linear plane wave: error vs exact solution")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SERIES[1], label="particles start on force-mesh nodes"),
                       Patch(color=SERIES[0], label="particles start between nodes")], loc="upper right")

    ax = axes[0, 2]
    ax.plot(steps[:3], mem["plain"], "o-", ms=6, color=SERIES[1], label="plain autodiff")
    ax.plot(steps, mem["low"], "o-", ms=6, color=SERIES[0], label="low-memory mode (default)")
    ax.plot(steps, mem["forward"], "o-", ms=6, color=SERIES[2], label="forward pass only")
    ax.set(yscale="log", xlabel="tCOLA steps", ylabel="XLA temporary buffers  [MB]",
           title="Memory of one gradient (32³ particles, 64³ force mesh)")
    ax.legend(loc="center right")

    ext, sl = (0, box, 0, box), n // 2
    lg = lambda a: np.log10(np.maximum(a[:, :, sl], 1e-2))
    a_, b_ = lg(rho_za), lg(rho_cola[2])
    vmin, vmax = min(a_.min(), b_.min()), max(a_.max(), b_.max())
    image(axes[1, 0], a_, SEQ, ext, "log₁₀ ρ", vmin, vmax)
    axes[1, 0].set(title="Zel'dovich", xlabel="x  [Mpc/h]", ylabel="y  [Mpc/h]")
    image(axes[1, 1], b_, SEQ, ext, "log₁₀ ρ", vmin, vmax)
    axes[1, 1].set(title="tCOLA, 10 steps, force sampling 2", xlabel="x  [Mpc/h]", ylabel="y  [Mpc/h]")
    ax = axes[1, 2]
    bins = np.logspace(-9, -2, 50)
    ax.hist(np.abs(dpos).ravel() + 1e-12, bins, color=SERIES[0], histtype="stepfilled", alpha=0.85, lw=0)
    ax.text(0.02, 0.95, f"rms particle displacement\nin this run: {total.std():.2f} cells\n(off scale, right)",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color=INK2)
    logx(ax)
    ax.set(xlim=(1e-9, 1e-2), xlabel="|x_miniflims − x_BorgJaxPM|  [cells]", ylabel="particle coordinates",
           title="Port vs original BorgJaxPM tCOLA (same layout)")
    fig.tight_layout()
    fig.savefig(out / "compare_cola.png")
    plt.close(fig)
    return stats


# --------------------------------------------------------------------------- #
def compare_sampler(out: Path) -> dict:
    """Same posterior, two unrelated samplers: miniflims (HMC on the field +
    slice sampling of nbar and alpha) vs blackjax NUTS on all variables jointly
    (nuisance parameters mapped to the real line with a logistic transform)."""
    import blackjax

    cfg = load_config(overrides={
        "grid": {"n_grid": 8, "box_size": 200.0},
        "galaxies": {"nbar": {"init": 2.0}},
        "data": {"mock": {"truth": {"nbar": 3.0, "alpha": 1.4}}},
        "sampler": {"n_chains": 4, "n_warmup": 500, "n_samples": 5000, "thin": 1,
                    "hmc": {"n_steps": 20, "n_steps_min": 8}},
        "output": {"log_every": 0},
    })
    t0 = time.time()
    res = run(cfg)
    t_ours = time.time() - t0
    model, nw = res.model, cfg.sampler.n_warmup
    th_ours = res.gibbs.trace["nuisance"][:, nw:, 0, :]
    x_ours = res.fields.reshape(-1, 8**3).astype(float)

    lower, span = model.lower, model.upper - model.lower

    def logdensity(z):
        s = jax.nn.sigmoid(z["u"])
        theta = (lower + span * s)[None]
        return model.log_posterior(z["x"], theta) + jnp.sum(jnp.log(span * s * (1 - s)))

    n_chains, n_warm, n_samp = 4, 1000, 5000
    t0 = time.time()
    th_nuts, x_nuts, acc = [], [], []
    for c in range(n_chains):
        key = jax.random.PRNGKey(100 + c)
        k_init, k_warm, k_run = jax.random.split(key, 3)
        s0 = (2.0 - lower[0]) / span[0], (1.0 - lower[1]) / span[1]
        init = {"x": 0.1 * jax.random.normal(k_init, (8, 8, 8)),
                "u": jnp.log(jnp.asarray(s0) / (1 - jnp.asarray(s0)))}
        warm = blackjax.window_adaptation(blackjax.nuts, logdensity, target_acceptance_rate=0.8)
        (state, params), _ = warm.run(k_warm, init, num_steps=n_warm)
        step = blackjax.nuts(logdensity, **params).step

        def body(st, k):
            st, info = step(k, st)
            return st, (st.position, info.acceptance_rate)

        _, (pos, a) = jax.jit(lambda s, ks: jax.lax.scan(body, s, ks))(state, jax.random.split(k_run, n_samp))
        th_nuts.append(np.asarray(lower + span * jax.nn.sigmoid(pos["u"])))
        x_nuts.append(np.asarray(pos["x"]).reshape(n_samp, -1))
        acc.append(float(a.mean()))
    t_nuts = time.time() - t0
    th_nuts = np.stack(th_nuts)
    x_nuts = np.concatenate(x_nuts)

    truth = res.data.truth["nuisance"][0]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9))
    for ax, j, name in ((axes[0], 0, "n̄"), (axes[1], 1, "α")):
        a_, b_ = th_ours[..., j].ravel(), th_nuts[..., j].ravel()
        bins = np.linspace(min(a_.min(), b_.min()), max(a_.max(), b_.max()), 45)
        ax.hist(a_, bins, density=True, histtype="step", lw=2, color=SERIES[0], label="miniflims HMC + slice")
        ax.hist(b_, bins, density=True, histtype="step", lw=2, color=SERIES[1], label="blackjax NUTS (joint)")
        ax.axvline(truth[j], color=INK, lw=1.0, ls=DASH, label="truth")
        ax.set(xlabel=name, ylabel="posterior density", title=f"Marginal posterior of {name}")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.45)  # headroom for the legend
        ax.legend(loc="upper right")
    for ax, fn, name in ((axes[2], np.mean, "mean"), (axes[3], np.std, "standard deviation")):
        a_, b_ = fn(x_nuts, axis=0), fn(x_ours, axis=0)
        lo, hi = min(a_.min(), b_.min()), max(a_.max(), b_.max())
        ax.plot([lo, hi], [lo, hi], color=INK, lw=1.0, ls=DASH)
        ax.scatter(a_, b_, s=14, color=SERIES[0], edgecolor="#fcfcfb", linewidth=0.5)
        ax.set(xlabel="blackjax NUTS", ylabel="miniflims", aspect="equal",
               title=f"White-noise posterior {name}, per voxel")
    fig.tight_layout()
    fig.savefig(out / "compare_sampler.png")
    plt.close(fig)

    stats = {"wall time miniflims [s]": t_ours, "wall time blackjax [s]": t_nuts,
             "NUTS mean acceptance": float(np.mean(acc))}
    for j, name in ((0, "nbar"), (1, "alpha")):
        a_, b_ = th_ours[..., j], th_nuts[..., j]
        se = np.sqrt(a_.var() / diag.effective_sample_size(a_) + b_.var() / diag.effective_sample_size(b_))
        stats[name] = {
            "miniflims": [float(a_.mean()), float(a_.std())],
            "blackjax": [float(b_.mean()), float(b_.std())],
            "mean difference / MC error": float((a_.mean() - b_.mean()) / se),
            "ESS miniflims": diag.effective_sample_size(a_), "ESS blackjax": diag.effective_sample_size(b_),
            "rhat pooled (both codes as chains)": diag.gelman_rubin(np.concatenate([a_, b_])),
        }
    stats["corr of per-voxel posterior means"] = float(np.corrcoef(x_ours.mean(0), x_nuts.mean(0))[0, 1])
    stats["rms difference of per-voxel means / posterior std"] = float(
        (x_ours.mean(0) - x_nuts.mean(0)).std() / x_nuts.std(0).mean())
    stats["mean ratio of per-voxel std (miniflims/blackjax)"] = float((x_ours.std(0) / x_nuts.std(0)).mean())
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--skip-sampler", action="store_true")
    ap.add_argument("--only", choices=["power", "lpt", "cola", "sampler"], action="append")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cosmo = Cosmology.from_config(load_config().cosmology)

    tasks = {"power": lambda: compare_power(cosmo, out), "lpt": lambda: compare_lpt(out),
             "cola": lambda: compare_cola(out), "sampler": lambda: compare_sampler(out)}
    names = args.only or [t for t in tasks if not (t == "sampler" and args.skip_sampler)]
    path = out / "comparison.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    for name in names:
        report[name] = tasks[name]()
    (out / "comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
