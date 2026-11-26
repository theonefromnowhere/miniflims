"""Gallery plots of a set of realizations written by make_png_realisations.py.

    .venv/bin/python scripts/plot_realisations.py [meshes/cola64_L1024_x10] [--slab 4]

Writes, next to the meshes:
    gallery_density.png   projected log density of every realization (Gaussian ICs)
    gallery_png.png       PNG imprint per realization: rho(f_NL) - rho(Gaussian), projected
    gallery_spectra.png   P(k) and the PNG responses of every realization (thin) and their mean
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from miniflims.plotting import AXIS, INK, INK2, SERIES, _mpl, image, logx  # noqa: E402

LABEL = {"gaussian": "Gaussian", "local_p100": "local f_NL = +100", "equil_p500": "equilateral f_NL = +500"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", nargs="?", default="meshes/cola64_L1024_x10")
    ap.add_argument("--slab", type=int, default=4, help="cells projected along z")
    args = ap.parse_args()
    d = Path(args.directory)
    manifest = json.loads((d / "manifest.json").read_text())
    reals = sorted({m["realisation"] for m in manifest})
    settings = list(dict.fromkeys(m["setting"] for m in manifest))
    meta = manifest[0]
    box, n = meta["box_size"], meta["n_grid"]
    rho = {(m["realisation"], m["setting"]): np.load(d / f"r{m['realisation']:02d}" / f"{m['setting']}.npz")["rho"]
           .astype(np.float64) for m in manifest}
    sl = slice(n // 2 - args.slab // 2, n // 2 - args.slab // 2 + args.slab)
    proj = {key: v[:, :, sl].mean(axis=2) for key, v in rho.items()}
    depth = args.slab * box / n
    ext = (0, box, 0, box)
    grav = "tCOLA" if meta["gravity"] == "cola" else "Zel'dovich"
    head = f"{n}³, L = {box:g} Mpc/h, {grav}, z = 0; projection over {depth:g} Mpc/h"
    plt, seq, div = _mpl()
    ncol = 5
    nrow = int(np.ceil(len(reals) / ncol))

    # ---------------------------------------------------------- density gallery
    lg = {r: np.log10(np.maximum(proj[r, "gaussian"], 1e-2)) for r in reals}
    vmin = np.percentile(np.concatenate([v.ravel() for v in lg.values()]), 0.5)
    vmax = np.percentile(np.concatenate([v.ravel() for v in lg.values()]), 99.8)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol + 1.2, 3.3 * nrow + 0.6), squeeze=False,
                             layout="constrained")
    for ax, r in zip(axes.ravel(), reals):
        im = ax.imshow(lg[r].T, origin="lower", cmap=seq, extent=ext, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.grid(False)
        ax.set_title(f"realization {r}  (seed {meta['seed'] - meta['realisation'] + r})", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
    for ax in axes.ravel()[len(reals):]:
        ax.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("log₁₀ ρ (projected)", color=INK2)
    cb.outline.set_edgecolor(AXIS)
    fig.suptitle(f"Gaussian ICs — {head}", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.savefig(d / "gallery_density.png")
    plt.close(fig)

    # -------------------------------------------------------------- PNG imprint
    png_settings = [s for s in settings if s != "gaussian"]
    fig, axes = plt.subplots(len(png_settings) * nrow, ncol,
                             figsize=(3.3 * ncol + 1.2, 3.3 * nrow * len(png_settings) + 0.8), squeeze=False,
                             layout="constrained")
    for i, s in enumerate(png_settings):
        diffs = {r: proj[r, s] - proj[r, "gaussian"] for r in reals}
        lim = np.percentile(np.abs(np.concatenate([v.ravel() for v in diffs.values()])), 99.5)
        block = axes[i * nrow:(i + 1) * nrow]
        for ax, r in zip(block.ravel(), reals):
            im = ax.imshow(diffs[r].T, origin="lower", cmap=div, extent=ext, vmin=-lim, vmax=lim,
                           interpolation="nearest")
            ax.grid(False)
            ax.set_title(f"{LABEL.get(s, s)} — r{r}", fontsize=9)
            ax.set_xticks([]), ax.set_yticks([])
        for ax in block.ravel()[len(reals):]:
            ax.set_visible(False)
        cb = fig.colorbar(im, ax=block, fraction=0.015, pad=0.01)
        cb.set_label("ρ_PNG − ρ_Gauss", color=INK2)
        cb.outline.set_edgecolor(AXIS)
    fig.suptitle(f"PNG imprint (same white noise, minus the Gaussian mesh) — {head}",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.savefig(d / "gallery_png.png")
    plt.close(fig)

    # ------------------------------------------------------------------ spectra
    from miniflims.png import import_jaxpng

    jp = import_jaxpng()
    jax.config.update("jax_enable_x64", True)
    kny = np.pi * n / box

    def pk(f):
        k, p = jp.measure.power_spectrum(jnp.asarray(f - 1.0), box, MAS="CIC")
        k, p = np.asarray(k), np.asarray(p)
        return k[k < kny], p[k < kny]

    k, _ = pk(rho[reals[0], "gaussian"])
    P = {key: pk(v)[1] for key, v in rho.items()}
    kk = 2 * np.pi / box * np.arange(2, n // 3)

    def beq(f):
        return np.array([float(jp.measure.bispectrum(jnp.asarray(f - 1.0), box, kv, kv,
                                                     jp.measure.EQUILATERAL_THETA, MAS="CIC")["B"][0]) for kv in kk])

    B = {key: beq(v) for key, v in rho.items()}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    for r in reals:
        ax.plot(k, P[r, "gaussian"], color=SERIES[0], lw=0.8, alpha=0.45)
    ax.plot(k, np.mean([P[r, "gaussian"] for r in reals], axis=0), color=INK, lw=2, label=f"mean of {len(reals)}")
    ax.plot([], [], color=SERIES[0], lw=0.8, label="each realization")
    logx(ax)
    ax.set(yscale="log", xlabel="k  [h/Mpc]", ylabel="P(k)  [(Mpc/h)³]  (CIC compensated)",
           title="Power spectrum, Gaussian ICs")
    ax.legend(loc="lower left")

    ax = axes[1]
    for col, s in zip(SERIES[1:], png_settings):
        ratios = np.array([P[r, s] / P[r, "gaussian"] - 1 for r in reals])
        for row in ratios:
            ax.plot(k, row, color=col, lw=0.7, alpha=0.4)
        ax.plot(k, ratios.mean(0), color=col, lw=2.2, label=LABEL.get(s, s))
    ax.axhline(0, color=INK, lw=1.0, ls=(0, (4, 3)))
    logx(ax)
    ax.set(xlabel="k  [h/Mpc]", ylabel="P / P_Gauss − 1  (same white noise)",
           title="Power-spectrum response, per realization")
    ax.legend(loc="upper left")

    ax = axes[2]
    for col, s in zip(SERIES[1:], png_settings):
        dB = np.array([B[r, s] - B[r, "gaussian"] for r in reals])
        for row in dB:
            ax.plot(kk, row, color=col, lw=0.7, alpha=0.4)
        ax.plot(kk, dB.mean(0), color=col, lw=2.2, label=LABEL.get(s, s))
    ax.set_yscale("symlog", linthresh=1e5)
    ax.axhline(0, color=AXIS, lw=1.0)
    ax.set(xlabel="k  [h/Mpc]", ylabel="B(k,k,k) − B_Gauss  [(Mpc/h)⁶]",
           title="Equilateral bispectrum response, per realization")
    ax.legend(loc="upper right")
    fig.suptitle(head.split(";")[0] + f" — {len(reals)} realizations", x=0.01, ha="left", fontsize=11,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(d / "gallery_spectra.png")
    plt.close(fig)
    for name in ("gallery_density.png", "gallery_png.png", "gallery_spectra.png"):
        print(d / name)


if __name__ == "__main__":
    main()
