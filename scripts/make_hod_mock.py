"""HOD galaxy mock from a (Quijote) halo catalogue, gridded for miniflims.

    .venv-run/bin/python scripts/make_hod_mock.py ~/Documents/quijote/out_4_pid.list \
        --hod zheng09_lrg_21.2 --sigma8 0.834 -c configs/quijote_eq_2lpt.yaml \
        --matter-cube ~/Documents/quijote/df_m_256_CIC_z=0.npy -o data/quijote/hod_lrg212_fid_64.npz

Halo catalogue: Rockstar out_<snap>[_pid].list (host halos, M200b), a Quijote
FoF directory (needs Pylians; FoF masses), or .npz (see miniflims.hod.read_npz).
HOD presets: Zheng et al. (2009) SDSS LRGs, M_g < -21.2 or -21.8 (Appendix B),
at the given sigma_8, mass scales rescaled to the catalogue's Omega_m.

Writes
  <out>.npz           counts (1, n, n, n) on the mesh nodes (NGP), response,
                      truth_rho (if --matter-cube: CIC-regridded 1 + delta),
                      meta (JSON: HOD, numbers, completeness) -> data.source: file
  <out>_galaxies.npz  galaxy catalogue: pos (real space), pos_s (if --rsd), vel,
                      central, host, host_mass
  <out>.png           occupation, mass function and completeness, P(k) and bias, slices
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from miniflims.config import load_config
from miniflims.data import cic_regrid
from miniflims.hod import (HOD_PRESETS, Zheng07HOD, completeness, mass_function, ngp_counts, populate,
                           read_halos, redshift_space)


def power(field, box, n_bins=24):
    """Shell-averaged auto power of a real field on a periodic grid (Mpc/h)^3."""
    n = field.shape[0]
    fk = np.fft.rfftn(field)
    kf = 2 * np.pi / box
    kx = np.fft.fftfreq(n, 1 / n) * kf
    kz = np.fft.rfftfreq(n, 1 / n) * kf
    k = np.sqrt(kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kz[None, None, :] ** 2)
    w = np.full(kz.shape, 2.0)
    w[0] = 1.0
    if n % 2 == 0:
        w[-1] = 1.0
    w = np.broadcast_to(w, k.shape)
    p = np.abs(fk) ** 2 * box**3 / n**6
    edges = np.linspace(kf / 2, kf * n / 2, n_bins + 1)
    i = np.digitize(k, edges) - 1
    ok = (i >= 0) & (i < n_bins)
    cnt = np.bincount(i[ok], weights=w[ok], minlength=n_bins)
    pk = np.bincount(i[ok], weights=(w * p)[ok], minlength=n_bins) / np.maximum(cnt, 1)
    kk = np.bincount(i[ok], weights=(w * k)[ok], minlength=n_bins) / np.maximum(cnt, 1)
    return kk[cnt > 0], pk[cnt > 0]


def cross(a, b, box, n_bins=24):
    n = a.shape[0]
    fa, fb = np.fft.rfftn(a), np.fft.rfftn(b)
    kf = 2 * np.pi / box
    kx = np.fft.fftfreq(n, 1 / n) * kf
    kz = np.fft.rfftfreq(n, 1 / n) * kf
    k = np.sqrt(kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kz[None, None, :] ** 2)
    w = np.full(kz.shape, 2.0)
    w[0] = 1.0
    if n % 2 == 0:
        w[-1] = 1.0
    w = np.broadcast_to(w, k.shape)
    edges = np.linspace(kf / 2, kf * n / 2, n_bins + 1)
    i = np.digitize(k, edges) - 1
    ok = (i >= 0) & (i < n_bins)
    cnt = np.bincount(i[ok], weights=w[ok], minlength=n_bins)
    pab = np.bincount(i[ok], weights=(w * np.real(fa * np.conj(fb)))[ok], minlength=n_bins)
    paa = np.bincount(i[ok], weights=(w * np.abs(fa) ** 2)[ok], minlength=n_bins)
    pbb = np.bincount(i[ok], weights=(w * np.abs(fb) ** 2)[ok], minlength=n_bins)
    good = cnt > 0
    return pab[good] / np.sqrt(paa[good] * pbb[good])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("halos")
    ap.add_argument("--format", default="auto", choices=["auto", "rockstar", "fof", "npz"])
    ap.add_argument("--mass-column", default="M200b", help="Rockstar mass column for the HOD [M200b = M_200m]")
    ap.add_argument("--snapnum", type=int, default=4, help="FoF snapshot number [4: z = 0]")
    ap.add_argument("--hod", default="zheng09_lrg_21.2", choices=sorted(HOD_PRESETS))
    ap.add_argument("--hod-params", default=None,
                    help="JSON with log_mmin, sigma_logm, log_m0, log_m1p, alpha (overrides --hod)")
    ap.add_argument("--sigma8", type=float, default=0.834, help="sigma_8 for the preset [Quijote fiducial 0.834]")
    ap.add_argument("--omega-m", type=float, default=None, help="Omega_m [from the catalogue]")
    ap.add_argument("--cosmology", default="0.6711,0.049,0.9624",
                    help="h, Omega_b, n_s for the Tinker08 completeness check [Quijote fiducial]")
    ap.add_argument("--rsd", default="none", choices=["none", "x", "y", "z"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-c", "--config", default=None, help="miniflims config: mesh size and box")
    ap.add_argument("--n", type=int, default=64, help="mesh size if no config")
    ap.add_argument("--matter-cube", default=None, help="matter density cube of the same simulation (truth_rho)")
    ap.add_argument("--truth-png", type=float, nargs="*", default=None,
                    help="PNG parameters of the simulation (e.g. 100 for an EQ_p box), stored as truth_png")
    ap.add_argument("-o", "--out", required=True, help="output .npz (data.source: file)")
    args = ap.parse_args()

    kw = {}
    fmt = args.format
    if fmt in ("auto", "rockstar") and not Path(args.halos).expanduser().is_dir() and not args.halos.endswith(".npz"):
        kw["mass_column"] = args.mass_column
    if fmt == "fof" or (fmt == "auto" and Path(args.halos).expanduser().is_dir()):
        kw["snapnum"] = args.snapnum
    halos = read_halos(args.halos, fmt, **kw)
    om = args.omega_m or halos.omega_m
    if args.hod_params:
        hod = Zheng07HOD(**json.loads(args.hod_params), label="custom")
    else:
        hod = HOD_PRESETS[args.hod](args.sigma8, om)
    print(f"{len(halos)} halos ({halos.mass_def}), box {halos.box} Mpc/h, z = {halos.redshift:.3g}, Om = {om}")
    print(f"HOD {hod.label}: " + ", ".join(f"{k} {v:.4g}" for k, v in hod.as_dict().items() if k != "label"))
    if halos.mass_def != "200m":
        print(f"note: the presets are defined for M_200m; this catalogue's masses are '{halos.mass_def}'")

    from miniflims.cosmology import Cosmology

    h, ob, ns = (float(v) for v in args.cosmology.split(","))
    cosmo = Cosmology(h=h, omega_m=om, omega_b=ob, n_s=ns, sigma8=args.sigma8)
    comp = completeness(halos, hod, cosmo)
    print(f"mass function: catalogue / Tinker08 = {comp['catalogue_over_tinker_high_mass']:.2f} at high mass, "
          f">= 80% complete above 10^{comp['log_m_complete_80']:.2f} Msun/h (peak at 10^{comp['log_m_peak']:.2f})")
    print(f"HOD number density for the full (Tinker08) mass function {comp['n_g_expected']:.3g} (h/Mpc)^3, "
          f"from this catalogue {comp['n_g_catalogue_hod']:.3g}: {comp['missing_fraction']:.0%} of the galaxies "
          f"would sit in halos missing from the catalogue")

    rng = np.random.default_rng(args.seed)
    gal = populate(halos, hod, rng)
    n_gal, n_cen = len(gal), int(gal.central.sum())
    n_g = n_gal / halos.box**3
    print(f"{n_gal} galaxies ({n_cen} centrals, {n_gal - n_cen} satellites, f_sat {1 - n_cen / n_gal:.3f}); "
          f"n_g = {n_g:.3g} (h/Mpc)^3, mean host mass 10^{np.log10(np.mean(halos.mass[gal.host])):.2f}")

    pos = gal.pos if args.rsd == "none" else redshift_space(gal.pos, gal.vel, args.rsd, halos)
    if args.config:
        cfg = load_config(args.config)
        n, box = cfg.grid.n_mesh or cfg.grid.n_grid, cfg.grid.box_size
        if abs(box - halos.box) > 1e-6 * box:
            raise SystemExit(f"config box {box} differs from the catalogue's {halos.box}")
    else:
        n, box = args.n, halos.box
    counts = ngp_counts(pos, n, box)
    print(f"mesh {n}^3 ({box / n:.2f} Mpc/h cells): mean {counts.mean():.4f} galaxies per cell, max {counts.max():.0f}")

    rho = None
    if args.matter_cube:
        cube = np.load(Path(args.matter_cube).expanduser(), mmap_mode="r", allow_pickle=False)
        x = np.asarray(cube, np.float64)
        rho = cic_regrid(x + 1.0 if abs(x.mean()) < 0.5 else x / x.mean(), n)
        r = cross(counts / counts.mean() - 1.0, rho - 1.0, box)
        print(f"galaxy-matter cross-correlation r(k) in the first bins: {np.round(r[:4], 3).tolist()} "
              "(~0.9+ if the cube and the halos come from the same simulation, ~0 otherwise)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(halos=str(args.halos), halo_meta={k: v for k, v in halos.meta.items() if np.isscalar(v)},
                mass_def=halos.mass_def, hod=hod.as_dict(), hod_preset=None if args.hod_params else args.hod,
                sigma8=args.sigma8, omega_m=om, seed=args.seed, rsd=args.rsd, n_mesh=n, box=box,
                truth_png=args.truth_png,
                n_galaxies=n_gal, n_centrals=n_cen, n_g=n_g,
                completeness={k: v for k, v in comp.items() if not isinstance(v, list)},
                matter_cube=args.matter_cube)
    extra = {"truth_rho": rho} if rho is not None else {}
    if args.truth_png is not None:
        extra["truth_png"] = np.asarray(args.truth_png, float)
    np.savez_compressed(out, counts=counts[None].astype(np.float32), response=np.ones((1, n, n, n), np.float32),
                        meta=json.dumps(meta), **extra)
    gal_extra = {"pos_s": pos.astype(np.float32)} if args.rsd != "none" else {}
    np.savez_compressed(out.with_name(out.stem + "_galaxies.npz"), pos=gal.pos.astype(np.float32),
                        vel=gal.vel.astype(np.float32), central=gal.central, host=gal.host,
                        host_mass=halos.mass[gal.host], meta=json.dumps(meta), **gal_extra)
    print(f"written {out} and {out.with_name(out.stem + '_galaxies.npz')}")
    figure(out.with_suffix(".png"), halos, hod, gal, counts, rho, comp, box, n_g)


def figure(path, halos, hod, gal, counts, rho, comp, box, n_g):
    from miniflims.plotting import INK, MUTED, SERIES, _mpl

    plt, _, _ = _mpl()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))

    ax = axes[0, 0]  # occupation
    lm = np.log10(halos.mass)
    edges = np.arange(np.floor(lm.min() * 4) / 4, lm.max() + 0.25, 0.125)
    c = 0.5 * (edges[1:] + edges[:-1])
    nh, _ = np.histogram(lm, edges)
    ncen, _ = np.histogram(lm[gal.host[gal.central]], edges)
    nsat, _ = np.histogram(lm[gal.host[~gal.central]], edges)
    ok = nh > 0
    m = 10**c
    ax.plot(c, hod.mean_ncen(m), color=SERIES[0], lw=1.5, label="<N_cen> model")
    ax.plot(c, hod.mean_nsat(m), color=SERIES[1], lw=1.5, label="<N_sat> model")
    ax.plot(c[ok], ncen[ok] / nh[ok], "o", ms=4, color=SERIES[0], label="centrals, this mock")
    ax.plot(c[ok], nsat[ok] / nh[ok], "o", ms=4, color=SERIES[1], label="satellites, this mock")
    ax.set(yscale="log", ylim=(1e-3, 30), xlabel="log10 M [Msun/h]", ylabel="<N | M>",
           title=f"Occupation: {hod.label}")
    ax.legend(fontsize=8)

    ax = axes[0, 1]  # mass function and completeness
    e, dn = mass_function(halos)
    cc = 0.5 * (e[1:] + e[:-1])
    ax.step(cc, dn, where="mid", color=INK, lw=1.2, label="catalogue dn/dlogM")
    if "dndlogm_tinker08" in comp:
        lt, nt = np.asarray(comp["log_m"]), np.asarray(comp["dndlogm_tinker08"])
        ax.plot(lt, nt, color=MUTED, ls=(0, (4, 3)), label="Tinker+08 (200m)")
        ax.plot(lt, nt * hod.mean_n(10**lt), color=SERIES[2], ls=(0, (4, 3)), lw=1.2,
                label="Tinker+08 x <N> (expected galaxies)")
        ax.axvline(comp["log_m_complete_80"], color=MUTED, lw=0.8)
    ax.plot(cc, dn * hod.mean_n(10**cc), color=SERIES[2], lw=1.4, label="catalogue x <N> (mock galaxies)")
    ax.set(yscale="log", xlabel="log10 M [Msun/h]", ylabel="dn/dlog10M [(h/Mpc)^3]",
           title=f"Mass function: {comp.get('missing_fraction', float('nan')):.0%} of the HOD galaxies in missing halos")
    ax.set_ylim(1e-9, None)
    ax.legend(fontsize=8)

    ax = axes[0, 2]  # power spectra
    n = counts.shape[0]
    dg = counts / counts.mean() - 1.0
    k, pg = power(dg, box)
    shot = 1.0 / n_g
    ax.loglog(k, pg, color=SERIES[0], label="galaxies (incl. shot noise)")
    ax.axhline(shot, color=SERIES[0], ls=(0, (2, 2)), lw=0.9, label="1 / n_g")
    if rho is not None:
        _, pm = power(rho - 1.0, box)
        ax.loglog(k, pm, color=INK, label="matter")
        ax2 = ax.inset_axes([0.12, 0.08, 0.42, 0.35])
        b = np.sqrt(np.clip(pg - shot, 0, None) / pm)
        ax2.semilogx(k, b, color=SERIES[0])
        ax2.semilogx(k, cross(dg, rho - 1.0, box), color=SERIES[2])
        ax2.set_title("bias (P_gg - 1/n)^1/2 / P_mm^1/2, r_gm", fontsize=7)
        ax2.set_ylim(0, max(3.0, np.nanmax(b[:6]) * 1.2))
        ax2.tick_params(labelsize=7)
    ax.set(xlabel="k [h/Mpc]", ylabel="P(k) [(Mpc/h)^3]", title="Power spectra on the mesh")
    ax.legend(fontsize=8, loc="upper right")

    sl = slice(0, max(1, n // 16))
    proj = counts[:, :, sl].sum(-1)
    ax = axes[1, 0]
    ax.imshow(proj.T, origin="lower", extent=(0, box, 0, box), cmap="Greys")
    ax.set(title=f"Galaxy counts, {sl.stop} cell(s) thick", xlabel="x [Mpc/h]", ylabel="y [Mpc/h]")
    ax = axes[1, 1]
    if rho is not None:
        ax.imshow(np.log10(rho[:, :, sl].mean(-1)).T, origin="lower", extent=(0, box, 0, box), cmap="Blues")
        ax.set(title="Matter log10(1 + delta), same slab", xlabel="x [Mpc/h]")
    else:
        ax.set_visible(False)
    ax = axes[1, 2]
    hm = np.log10(halos.mass[gal.host])
    ax.hist(hm[gal.central], bins=40, histtype="step", color=SERIES[0], label="centrals")
    ax.hist(hm[~gal.central], bins=40, histtype="step", color=SERIES[1], label="satellites")
    ax.set(xlabel="log10 M_host [Msun/h]", ylabel="galaxies", title="Host halo masses")
    ax.legend(fontsize=8)
    fig.suptitle(f"HOD mock: {len(gal)} galaxies, n_g = {n_g:.3g} (h/Mpc)^3, f_sat = {1 - gal.central.mean():.3f}",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"figure {path}")


if __name__ == "__main__":
    main()
