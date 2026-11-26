"""HOD mocks from halo catalogues (miniflims.hod)."""
import numpy as np
import pytest

from miniflims.data import cic_regrid
from miniflims.hod import (ZHENG09_LRG, HaloCatalog, Zheng07HOD, completeness, ngp_counts, nfw_mass_fraction,
                           populate, read_halos, read_rockstar, redshift_space, sample_nfw_radius, zheng09_lrg)


def test_zheng09_preset_reproduces_eq_b3_and_omega_m_scaling():
    h = zheng09_lrg("Mg<-21.2", sigma8=0.8, omega_m=0.24)
    assert (h.log_mmin, h.sigma_logm, h.log_m0, h.log_m1p, h.alpha) == (13.673, 0.621, 12.339, 14.533, 1.832)
    x = 0.9 - 0.8
    h2 = zheng09_lrg("Mg<-21.8", sigma8=0.9, omega_m=0.48)
    assert h2.log_mmin == pytest.approx(14.304 + 1.694 * x - 1.810 * x**2 + np.log10(2.0))
    assert h2.alpha == pytest.approx(1.717 - 0.589 * x - 7.437 * x**2)  # alpha not rescaled
    assert h2.sigma_logm == pytest.approx(0.797 + 0.761 * x - 0.614 * x**2)
    with pytest.raises(ValueError):
        zheng09_lrg("Mg<-21.2", sigma8=1.2)
    assert set(ZHENG09_LRG) == {"Mg<-21.2", "Mg<-21.8"}


def test_occupation_functions():
    h = Zheng07HOD(13.5, 0.5, 13.0, 14.2, 1.0)
    assert h.mean_ncen(10**13.5) == pytest.approx(0.5)
    assert h.mean_nsat(10**12.9) == 0.0  # below M_0
    m = 10**14.8
    assert h.mean_nsat(m) == pytest.approx(h.mean_ncen(m) * (m - 1e13) / 10**14.2)  # modulated by <N_cen>


def synthetic_halos(n=40000, mass=10**14.3, box=500.0, seed=0, rs=True):
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 3)) * box
    vel = rng.normal(0, 300, (n, 3))
    masses = np.full(n, mass)
    cat = HaloCatalog(pos, vel, masses, box, omega_m=0.3175)
    if rs:
        cat.rs = cat.radius() / 6.0
        cat.vrms = np.full(n, 900.0)
    return cat


def test_populate_matches_mean_occupations_and_nfw_profile():
    halos = synthetic_halos()
    hod = Zheng07HOD(14.0, 0.4, 13.0, 14.0, 1.0)
    gal = populate(halos, hod, np.random.default_rng(1))
    n = len(halos)
    ncen, nsat = gal.central.sum(), (~gal.central).sum()
    m = halos.mass[0]
    assert ncen == pytest.approx(n * hod.mean_ncen(m), rel=0.03)
    assert nsat == pytest.approx(n * hod.mean_nsat(m), rel=0.03)
    # centrals at the halo centre with the halo velocity
    np.testing.assert_allclose(gal.pos[gal.central], halos.pos[gal.host[gal.central]])
    np.testing.assert_allclose(gal.vel[gal.central], halos.vel[gal.host[gal.central]])
    # satellites inside R_200m, NFW with c = 6: fraction inside r_s
    d = gal.pos[~gal.central] - halos.pos[gal.host[~gal.central]]
    d -= halos.box * np.round(d / halos.box)
    r = np.linalg.norm(d, axis=1) / halos.radius()[0]
    assert r.max() <= 1.0 + 1e-9
    assert np.mean(r < 1 / 6) == pytest.approx(nfw_mass_fraction(1.0) / nfw_mass_fraction(6.0), abs=0.01)
    # satellite velocity dispersion v_rms / sqrt(3) around the halo velocity
    dv = gal.vel[~gal.central] - halos.vel[gal.host[~gal.central]]
    assert dv.std() == pytest.approx(900 / np.sqrt(3), rel=0.02)


def test_nfw_radius_sampler_cdf():
    rng = np.random.default_rng(2)
    c = np.full(200000, 10.0)
    x = sample_nfw_radius(c, rng) * 10.0
    for q in (0.5, 2.0, 7.0):
        assert np.mean(x < q) == pytest.approx(nfw_mass_fraction(q) / nfw_mass_fraction(10.0), abs=0.004)


def test_duffy_fallback_and_virial_dispersion():
    halos = synthetic_halos(n=2000, rs=False)
    gal = populate(halos, Zheng07HOD(13.0, 0.3, 12.0, 13.5, 1.0), np.random.default_rng(3))
    assert (~gal.central).sum() > 0 and np.all(np.isfinite(gal.vel))


def test_redshift_space_shift_and_ngp_counts():
    halos = synthetic_halos(n=10)
    pos = np.array([[10.0, 20.0, 30.0]])
    vel = np.array([[0.0, 0.0, 250.0]])
    s = redshift_space(pos, vel, "z", halos)  # z = 0: shift v / 100 Mpc/h
    np.testing.assert_allclose(s, [[10.0, 20.0, 32.5]])
    s = redshift_space(np.array([[1.0, 1.0, 499.0]]), vel, 2, halos)
    assert s[0, 2] == pytest.approx(1.5)  # periodic
    p = np.random.default_rng(4).random((1000, 3)) * 500.0
    c = ngp_counts(p, 16, 500.0)
    assert c.sum() == 1000 and c.shape == (16, 16, 16)
    assert ngp_counts(np.array([[499.0, 0.0, 15.0]]), 16, 500.0)[0, 0, 0] == 1  # nearest node, wrapped


def test_read_rockstar_hosts_units_and_header(tmp_path):
    f = tmp_path / "out_4_pid.list"
    names = ("ID DescID Mvir Vmax Vrms Rvir Rs Np X Y Z VX VY VZ M200b PID").split()
    rows = [
        [1, -1, 2e14, 700, 800, 1000, 200, 300, 10.0, 20.0, 30.0, 100, -50, 5, 2.2e14, -1],
        [2, -1, 5e13, 400, 500, 700, 150, 80, 999.5, 1.0, 2.0, 0, 0, 0, 5.5e13, 1],  # subhalo
        [3, -1, 8e13, 500, 600, 800, 120, 120, 1000.2, 3.0, 4.0, 1, 2, 3, 9e13, -1],
    ]
    header = "#" + " ".join(names) + "\n#a = 1.000000\n#Om = 0.317500; Ol = 0.682500; h = 0.671100\n" \
             "#Particle mass: 6.56561e+11 Msun/h\n#Box size: 1000.000000 Mpc/h\n"
    f.write_text(header + "\n".join(" ".join(str(v) for v in r) for r in rows) + "\n")
    cat = read_halos(f)
    assert len(cat) == 2 and cat.box == 1000.0 and cat.redshift == 0.0 and cat.omega_m == 0.3175
    np.testing.assert_allclose(cat.mass, [2.2e14, 9e13])
    np.testing.assert_allclose(cat.rs, [0.2, 0.12])  # kpc/h -> Mpc/h
    assert cat.pos[1, 0] == pytest.approx(0.2)  # wrapped into the box
    assert len(read_rockstar(f, hosts_only=False)) == 3
    assert read_rockstar(f, mass_column="Mvir").mass_def == "vir"


def test_completeness_against_tinker08():
    from miniflims.cosmology import Cosmology
    from miniflims.hod import tinker08_dndlogm

    cosmo = Cosmology(h=0.6711, omega_m=0.3175, omega_b=0.049, n_s=0.9624, sigma8=0.834)
    rng = np.random.default_rng(5)
    box = 400.0
    grid = np.linspace(12.5, 15.5, 601)
    dn = tinker08_dndlogm(grid, cosmo)
    cdf = np.concatenate([[0], np.cumsum(0.5 * (dn[1:] + dn[:-1]) * np.diff(grid))])
    n = rng.poisson(cdf[-1] * box**3)
    logm = np.interp(rng.random(n) * cdf[-1], cdf, grid)
    full = HaloCatalog(np.zeros((n, 3)), np.zeros((n, 3)), 10**logm, box)
    hod = Zheng07HOD(13.5, 0.4, 13.0, 14.5, 1.0)
    c = completeness(full, hod, cosmo)
    assert c["catalogue_over_tinker_high_mass"] == pytest.approx(1.0, abs=0.1)
    assert c["missing_fraction"] < 0.05
    cut = logm > 14.0  # catalogue truncated at 1e14
    trunc = HaloCatalog(np.zeros((cut.sum(), 3)), np.zeros((cut.sum(), 3)), 10 ** logm[cut], box)
    c = completeness(trunc, hod, cosmo)
    assert c["log_m_complete_80"] == pytest.approx(14.0, abs=0.13)
    assert c["missing_fraction"] > 0.5
    assert completeness(trunc, hod)["log_m_peak"] >= 13.9  # without a cosmology: peak only


def test_cic_regrid_preserves_mean_and_positivity():
    rho = 1.0 + np.random.default_rng(6).exponential(1.0, (32, 32, 32))
    r = cic_regrid(rho, 8)
    assert r.shape == (8, 8, 8) and r.min() > 0
    assert r.mean() == pytest.approx(rho.mean())
    np.testing.assert_allclose(cic_regrid(np.ones((16,) * 3), 4), 1.0)
