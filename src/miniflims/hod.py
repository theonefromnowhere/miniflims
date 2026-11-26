"""Halo occupation distribution (HOD) galaxy mocks from halo catalogues.

    halos = read_halos("out_4_pid.list")                     # Rockstar, Quijote FoF or .npz
    hod = zheng09_lrg("Mg<-21.2", sigma8=0.834, omega_m=0.3175)
    gal = populate(halos, hod, np.random.default_rng(1))
    pos = redshift_space(gal.pos, gal.vel, "z", halos)       # optional
    counts = ngp_counts(pos, 64, halos.box)

Centrals sit at the halo centre with the halo velocity; their number is
drawn from the nearest-integer distribution (Bernoulli for <N_cen> <= 1).
Satellites are Poisson with mean <N_sat(M)>, placed on an NFW profile
truncated at R_200m (concentration R_200m / r_s from the catalogue, or the
Duffy et al. 2008 c(M) for 200 x mean density), with the halo velocity plus an
isotropic Gaussian of 1D dispersion v_rms / sqrt(3) (Rockstar) or
sqrt(G M / 2R) (virial estimate). Counts are assigned to the nearest mesh node
j * cell, the convention of miniflims' gridded data.

HOD presets: Zheng et al. (2009, ApJ 707, 554; arXiv:0809.1868), Appendix B:
the five-parameter model (Eq. B1)

    <N(M)> = 1/2 [1 + erf((log M - log M_min) / sigma_logM)] [1 + ((M - M_0) / M_1')^alpha]

fitted to the SDSS LRG samples M_g < -21.2 (n_g = 9.73e-5 h^3 Mpc^-3) and
M_g < -21.8 (2.40e-5), as quadratics in sigma_8 - 0.8 (Eqs. B3, B4), for
(Omega_m, Omega_L, Omega_b, n_s, h) = (0.24, 0.76, 0.04, 0.95, 0.73); "for
results with cosmological models with Omega_m different than 0.24, one only
needs to change the three mass scales by a factor of Omega_m/0.24". Masses in
Msun/h for halos with a mean density of 200 times the background (M_200m,
Rockstar's 'M200b'). The LRG samples are at z ~ 0.3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.special import erf

RHO_CRIT0 = 2.77536627e11  # (Msun/h) / (Mpc/h)^3
G_NEWTON = 4.30091727e-9  # Mpc (km/s)^2 / Msun

# Zheng et al. 2009, Eqs. (B3), (B4): (constant, coefficient of x, of x^2), x = sigma_8 - 0.8
ZHENG09_LRG = {
    "Mg<-21.2": dict(
        n_g=9.73e-5,
        log_mmin=(13.673, 1.419, -1.706),
        sigma_logm=(0.621, 0.908, -0.935),
        log_m0=(12.339, 0.658, 9.206),
        log_m1p=(14.533, 1.248, -1.394),
        alpha=(1.832, -1.326, 0.523),
        errors=dict(log_mmin=0.06, sigma_logm=0.07, log_m0=0.6, log_m1p=0.025, alpha=0.08),
    ),
    "Mg<-21.8": dict(
        n_g=2.40e-5,
        log_mmin=(14.304, 1.694, -1.810),
        sigma_logm=(0.797, 0.761, -0.614),
        log_m0=(12.491, -1.476, 29.983),
        log_m1p=(14.946, 1.616, -0.712),
        alpha=(1.717, -0.589, -7.437),
        errors=dict(log_mmin=0.06, sigma_logm=0.055, log_m0=0.7, log_m1p=0.1, alpha=0.4),
    ),
}
ZHENG09_OMEGA_M = 0.24


# --------------------------------------------------------------------------- #
# HOD model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Zheng07HOD:
    """Five-parameter HOD (Zheng et al. 2005, 2007, 2009 Eq. B1); masses in Msun/h,
    log = log10. Satellites are modulated by the central occupation."""

    log_mmin: float
    sigma_logm: float
    log_m0: float
    log_m1p: float
    alpha: float
    label: str = ""

    def mean_ncen(self, mass):
        return 0.5 * (1.0 + erf((np.log10(mass) - self.log_mmin) / self.sigma_logm))

    def mean_nsat(self, mass):
        mass = np.asarray(mass, float)
        x = np.clip((mass - 10.0**self.log_m0) / 10.0**self.log_m1p, 0.0, None)
        return self.mean_ncen(mass) * x**self.alpha

    def mean_n(self, mass):
        return self.mean_ncen(mass) + self.mean_nsat(mass)

    def as_dict(self) -> dict:
        return dict(log_mmin=self.log_mmin, sigma_logm=self.sigma_logm, log_m0=self.log_m0,
                    log_m1p=self.log_m1p, alpha=self.alpha, label=self.label)


def zheng09_lrg(sample: str = "Mg<-21.2", sigma8: float = 0.8, omega_m: float = ZHENG09_OMEGA_M) -> Zheng07HOD:
    """Zheng et al. (2009) Appendix B LRG HOD at the given sigma_8 (fit range
    0.65-1.00), mass scales rescaled by omega_m / 0.24."""
    if sample not in ZHENG09_LRG:
        raise ValueError(f"unknown Zheng09 sample {sample!r}; choose from {list(ZHENG09_LRG)}")
    if not 0.65 <= sigma8 <= 1.0:
        raise ValueError(f"sigma8 = {sigma8} is outside the fitted range 0.65-1.00")
    c = ZHENG09_LRG[sample]
    x = sigma8 - 0.8

    def q(name):
        a0, a1, a2 = c[name]
        return a0 + a1 * x + a2 * x * x

    shift = np.log10(omega_m / ZHENG09_OMEGA_M)
    return Zheng07HOD(q("log_mmin") + shift, q("sigma_logm"), q("log_m0") + shift, q("log_m1p") + shift,
                      q("alpha"), label=f"Zheng+09 LRG {sample}, sigma8={sigma8:g}, Om={omega_m:g}")


HOD_PRESETS = {
    "zheng09_lrg_21.2": lambda s8, om: zheng09_lrg("Mg<-21.2", s8, om),
    "zheng09_lrg_21.8": lambda s8, om: zheng09_lrg("Mg<-21.8", s8, om),
}


# --------------------------------------------------------------------------- #
# Halo catalogues
# --------------------------------------------------------------------------- #
@dataclass
class HaloCatalog:
    pos: np.ndarray  # (N, 3) comoving Mpc/h, in [0, box)
    vel: np.ndarray  # (N, 3) peculiar km/s
    mass: np.ndarray  # (N,) Msun/h, the mass the HOD is defined with
    box: float  # Mpc/h
    redshift: float = 0.0
    omega_m: float = 0.3175
    mass_def: str = "200m"
    rs: Optional[np.ndarray] = None  # NFW scale radius, comoving Mpc/h
    vrms: Optional[np.ndarray] = None  # 3D velocity dispersion of the halo particles, km/s
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.mass)

    def radius(self) -> np.ndarray:
        """Comoving R_200m from the mass (200 times the mean matter density)."""
        rho = 200.0 * self.omega_m * RHO_CRIT0  # comoving mean density is constant
        return (3.0 * self.mass / (4.0 * np.pi * rho)) ** (1.0 / 3.0)


def _rockstar_header(path):
    names, info = None, {}
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            s = line[1:].strip()
            if names is None and s.startswith("ID "):
                names = s.split()
            elif s.startswith("a ="):
                info["a"] = float(s.split("=")[1])
            elif s.startswith("Om ="):
                for part in s.split(";"):
                    k, v = part.split("=")
                    info[k.strip()] = float(v)
            elif s.startswith("Box size:"):
                info["box"] = float(s.split(":")[1].split()[0])
            elif s.startswith("Particle mass:"):
                info["particle_mass"] = float(s.split(":")[1].split()[0])
    if names is None:
        raise ValueError(f"{path}: no '#ID ...' column header (not a Rockstar out_*.list?)")
    return names, info


def read_rockstar(path, mass_column: str = "M200b", hosts_only: bool = True, box: Optional[float] = None,
                  omega_m: Optional[float] = None) -> HaloCatalog:
    """Rockstar ASCII catalogue (out_<snap>.list / out_<snap>_pid.list, as for
    Quijote). Positions Mpc/h, velocities km/s (physical peculiar), masses
    Msun/h, radii kpc/h in the file. ``hosts_only`` keeps PID == -1."""
    names, info = _rockstar_header(path)
    col = {n: i for i, n in enumerate(names)}
    want = ["X", "Y", "Z", "VX", "VY", "VZ", mass_column] + [n for n in ("Rs", "Vrms", "PID") if n in col]
    missing = [n for n in want if n not in col]
    if missing:
        raise ValueError(f"{path}: columns {missing} not in the header")
    d = np.loadtxt(path, comments="#", usecols=[col[n] for n in want], ndmin=2)
    v = {n: d[:, i] for i, n in enumerate(want)}
    keep = np.ones(len(d), bool)
    if hosts_only and "PID" in v:
        keep &= v["PID"] == -1
    keep &= v[mass_column] > 0
    box = box or info.get("box")
    if box is None:
        raise ValueError("box size not in the header; pass box=")
    a = info.get("a", 1.0)
    pos = np.mod(np.stack([v["X"], v["Y"], v["Z"]], 1)[keep], box)
    return HaloCatalog(
        pos=pos, vel=np.stack([v["VX"], v["VY"], v["VZ"]], 1)[keep], mass=v[mass_column][keep], box=box,
        redshift=1.0 / a - 1.0, omega_m=omega_m or info.get("Om", 0.3175),
        mass_def={"M200b": "200m", "Mvir": "vir", "M200c": "200c"}.get(mass_column, mass_column),
        rs=v["Rs"][keep] / 1e3 if "Rs" in v else None, vrms=v["Vrms"][keep] if "Vrms" in v else None,
        meta=dict(source=str(path), format="rockstar", mass_column=mass_column, hosts_only=hosts_only,
                  n_objects=int(len(d)), **info),
    )


def read_quijote_fof(snapdir, snapnum: int = 4, box: float = 1000.0, omega_m: float = 0.3175,
                     redshift: Optional[float] = None) -> HaloCatalog:
    """Quijote FoF catalogue (linking length 0.2) through Pylians' readfof,
    with the unit conversions of the Quijote documentation. FoF masses are not
    M_200m; the HOD presets are defined for M_200m."""
    try:
        import readfof  # Pylians
    except ImportError as e:  # pragma: no cover - depends on Pylians
        raise ImportError("reading Quijote FoF catalogues needs Pylians (readfof); "
                          "use the Rockstar out_*_pid.list, or convert to .npz") from e
    z = {4: 0.0, 3: 0.5, 2: 1.0, 1: 2.0, 0: 3.0}[snapnum] if redshift is None else redshift
    fof = readfof.FoF_catalog(str(snapdir), snapnum, long_ids=False, swap=False, SFR=False, read_IDs=False)
    return HaloCatalog(pos=np.mod(fof.GroupPos / 1e3, box), vel=fof.GroupVel * (1.0 + z),
                       mass=fof.GroupMass * 1e10, box=box, redshift=z, omega_m=omega_m, mass_def="fof0.2",
                       meta=dict(source=str(snapdir), format="fof", snapnum=snapnum))


def read_npz(path) -> HaloCatalog:
    """.npz with pos (N,3) Mpc/h, vel (N,3) km/s, mass (N,) Msun/h, box and
    optionally rs (Mpc/h), vrms (km/s), redshift, omega_m, mass_def."""
    with np.load(path) as f:
        g = {k: f[k] for k in f.files}
    opt = lambda k, d=None: (g[k].item() if g[k].ndim == 0 else g[k]) if k in g else d  # noqa: E731
    return HaloCatalog(pos=np.mod(g["pos"], float(g["box"])), vel=g["vel"], mass=g["mass"], box=float(g["box"]),
                       redshift=float(opt("redshift", 0.0)), omega_m=float(opt("omega_m", 0.3175)),
                       mass_def=str(opt("mass_def", "200m")), rs=opt("rs"), vrms=opt("vrms"),
                       meta=dict(source=str(path), format="npz"))


def read_halos(path, fmt: str = "auto", **kw) -> HaloCatalog:
    path = Path(path).expanduser()
    if fmt == "auto":
        fmt = "npz" if path.suffix == ".npz" else ("fof" if path.is_dir() else "rockstar")
    return {"rockstar": read_rockstar, "fof": read_quijote_fof, "npz": read_npz}[fmt](path, **kw)


# --------------------------------------------------------------------------- #
# Populating halos
# --------------------------------------------------------------------------- #
def duffy08_concentration(mass, redshift: float = 0.0):
    """c_200m(M) of Duffy et al. (2008), full sample, 200 x mean density:
    c = 10.14 (M / 2e12 Msun/h)^-0.081 (1 + z)^-1.01."""
    return 10.14 * (np.asarray(mass) / 2e12) ** -0.081 * (1.0 + redshift) ** -1.01


def nfw_mass_fraction(x):
    return np.log1p(x) - x / (1.0 + x)


def sample_nfw_radius(conc, rng, n_iter: int = 60):
    """r / R for one point per entry of ``conc`` from an NFW profile truncated
    at r = R (x = r / r_s in [0, c]), by bisection of the enclosed-mass CDF."""
    conc = np.asarray(conc, float)
    target = rng.random(conc.shape) * nfw_mass_fraction(conc)
    lo, hi = np.zeros_like(conc), conc.copy()
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        below = nfw_mass_fraction(mid) < target
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return 0.5 * (lo + hi) / conc


def random_directions(n, rng):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@dataclass
class Galaxies:
    pos: np.ndarray  # (N, 3) real-space comoving Mpc/h, wrapped into the box
    vel: np.ndarray  # (N, 3) peculiar km/s
    central: np.ndarray  # (N,) bool
    host: np.ndarray  # (N,) index into the halo catalogue

    def __len__(self) -> int:
        return len(self.central)


def populate(halos: HaloCatalog, hod: Zheng07HOD, rng, concentration: str = "auto",
             velocity_dispersion: str = "auto", c_range=(1.0, 50.0)) -> Galaxies:
    """One HOD realisation. ``concentration``: 'halo' (R_200m / r_s from the
    catalogue), 'duffy08', or 'auto' (halo if r_s is available).
    ``velocity_dispersion``: 'vrms' (v_rms / sqrt 3), 'virial' (sqrt(G M / 2R)),
    or 'auto' (vrms if available)."""
    m = halos.mass
    ncen = (rng.random(len(m)) < hod.mean_ncen(m)).astype(int)  # nearest-integer, <N_cen> <= 1
    nsat = rng.poisson(hod.mean_nsat(m))
    ic = np.flatnonzero(ncen)
    isat = np.repeat(np.arange(len(m)), nsat)

    radius = halos.radius()
    if concentration == "auto":
        concentration = "halo" if halos.rs is not None else "duffy08"
    if concentration == "halo":
        if halos.rs is None:
            raise ValueError("concentration='halo' needs r_s in the catalogue")
        conc = radius[isat] / halos.rs[isat]
    elif concentration == "duffy08":
        conc = duffy08_concentration(m[isat], halos.redshift)
    else:
        raise ValueError(f"unknown concentration {concentration!r}")
    conc = np.clip(conc, *c_range)
    r = sample_nfw_radius(conc, rng) * radius[isat]
    sat_pos = halos.pos[isat] + r[:, None] * random_directions(len(isat), rng)

    if velocity_dispersion == "auto":
        velocity_dispersion = "vrms" if halos.vrms is not None else "virial"
    if velocity_dispersion == "vrms":
        if halos.vrms is None:
            raise ValueError("velocity_dispersion='vrms' needs v_rms in the catalogue")
        sigma = halos.vrms[isat] / np.sqrt(3.0)
    elif velocity_dispersion == "virial":
        r_phys = radius[isat] / (1.0 + halos.redshift)
        sigma = np.sqrt(G_NEWTON * m[isat] / (2.0 * r_phys))
    else:
        raise ValueError(f"unknown velocity_dispersion {velocity_dispersion!r}")
    sat_vel = halos.vel[isat] + sigma[:, None] * rng.normal(size=(len(isat), 3))

    pos = np.mod(np.concatenate([halos.pos[ic], sat_pos]), halos.box)
    vel = np.concatenate([halos.vel[ic], sat_vel])
    central = np.concatenate([np.ones(len(ic), bool), np.zeros(len(isat), bool)])
    return Galaxies(pos, vel, central, np.concatenate([ic, isat]))


def redshift_space(pos, vel, los, halos: HaloCatalog):
    """Plane-parallel redshift-space positions along axis ``los`` ('x', 'y',
    'z' or 0-2): s = x + v_los (1 + z) / H(z), H in h km/s/Mpc."""
    ax = {"x": 0, "y": 1, "z": 2}.get(los, los)
    z = halos.redshift
    hubble = 100.0 * np.sqrt(halos.omega_m * (1 + z) ** 3 + 1 - halos.omega_m)
    out = np.array(pos, float, copy=True)
    out[:, ax] = np.mod(out[:, ax] + vel[:, ax] * (1 + z) / hubble, halos.box)
    return out


def ngp_counts(pos, n: int, box: float) -> np.ndarray:
    """Number of objects per mesh node j * box / n (nearest node, periodic)."""
    idx = np.mod(np.floor(np.asarray(pos) / box * n + 0.5).astype(np.int64), n)
    flat = (idx[:, 0] * n + idx[:, 1]) * n + idx[:, 2]
    return np.bincount(flat, minlength=n**3).reshape(n, n, n).astype(float)


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def mass_function(halos: HaloCatalog, dlogm: float = 0.125, lo: float = 11.0, hi: float = 16.0):
    """(edges, dn/dlog10 M) in (h/Mpc)^3 per dex."""
    edges = np.arange(lo, hi + dlogm / 2, dlogm)
    h, _ = np.histogram(np.log10(halos.mass), edges)
    return edges, h / dlogm / halos.box**3


def tinker08_dndlogm(log_m, cosmo, redshift: float = 0.0, transfer: str = "eisenstein_hu"):
    """Tinker et al. (2008) halo mass function for 200 x mean density,
    dn/dlog10 M in (h/Mpc)^3 per dex at the masses 10**log_m [Msun/h], from
    miniflims' linear power spectrum (``cosmo``: miniflims.cosmology.Cosmology).
    f(sigma) = A [(sigma/b)^-a + 1] exp(-c / sigma^2), (A, a, b, c) =
    (0.186, 1.47, 2.57, 1.19) at z = 0, with their redshift scalings."""
    from .cosmology import growth_factor
    from .power import linear_power

    log_m = np.asarray(log_m, float)
    lnk = np.linspace(np.log(1e-4), np.log(1e2), 4096)
    k = np.exp(lnk)
    pk = np.asarray(linear_power(k, cosmo, transfer))
    rho_m = cosmo.omega_m * RHO_CRIT0
    lm = np.concatenate([log_m - 0.01, log_m + 0.01])
    r = (3.0 * 10.0**lm / (4.0 * np.pi * rho_m)) ** (1.0 / 3.0)
    x = k[None, :] * r[:, None]
    w = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
    sig = np.sqrt(np.trapezoid(k**3 * pk * w**2 / (2 * np.pi**2), lnk, axis=1))
    sig *= float(growth_factor(1.0 / (1.0 + redshift), cosmo))
    s_lo, s_hi = sig[: len(log_m)], sig[len(log_m):]
    s = np.sqrt(s_lo * s_hi)
    dlnsig_dlnm = (np.log(s_hi) - np.log(s_lo)) / (0.02 * np.log(10.0))
    alpha_z = 10.0 ** (-((0.75 / np.log10(200.0 / 75.0)) ** 1.2))
    A = 0.186 * (1 + redshift) ** -0.14
    a = 1.47 * (1 + redshift) ** -0.06
    b = 2.57 * (1 + redshift) ** -alpha_z
    f = A * ((s / b) ** -a + 1.0) * np.exp(-1.19 / s**2)
    m = 10.0**log_m
    return np.log(10.0) * f * rho_m / m * np.abs(dlnsig_dlnm)


def completeness(halos: HaloCatalog, hod: Zheng07HOD, cosmo=None, dlogm: float = 0.125) -> dict:
    """Low-mass incompleteness of a halo catalogue, measured against the
    Tinker et al. (2008) mass function (``cosmo`` given), and the HOD's
    galaxies expected in the missing halos. ``n_g_expected`` is the HOD number
    density for the full mass function."""
    edges, dn = mass_function(halos, dlogm)
    c = 0.5 * (edges[1:] + edges[:-1])
    out = dict(log_m_peak=float(c[int(np.argmax(dn))]), n_halos=len(halos))
    if cosmo is None:
        return out
    dn_t = tinker08_dndlogm(c, cosmo, halos.redshift)
    ratio = np.where(dn_t > 0, dn / dn_t, np.nan)
    hi = c >= out["log_m_peak"] + 0.5
    good = hi & (dn > 0) & (c < c[dn > 0].max() - 0.5)
    m = 10.0**c
    n_exp = float(np.sum(hod.mean_n(m) * dn_t * dlogm))
    n_cat = float(np.sum(hod.mean_n(m) * dn * dlogm))
    n_miss = float(np.sum(hod.mean_n(m) * np.clip(dn_t - dn, 0, None) * dlogm))
    above = c[(ratio >= 0.8) & (c >= out["log_m_peak"])]
    return {**out, "log_m_complete_80": float(above.min()) if above.size else float("nan"),
            "catalogue_over_tinker_high_mass": float(np.nanmedian(ratio[good])) if good.any() else float("nan"),
            "n_g_expected": n_exp, "n_g_catalogue_hod": n_cat, "n_g_missing": n_miss,
            "missing_fraction": n_miss / n_exp if n_exp > 0 else float("nan"),
            "log_m": c.tolist(), "dndlogm_tinker08": dn_t.tolist()}
