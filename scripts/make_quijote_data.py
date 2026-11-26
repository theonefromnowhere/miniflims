"""Galaxy counts on the inference mesh drawn from a Quijote(-PNG) N-body density cube.

    .venv-run/bin/python scripts/make_quijote_data.py ~/Downloads/quijote_png_eq.npy \
        -c configs/quijote_eq_2lpt.yaml --nbar 5 --alpha 1.5 --truth-png 100 --seed 20

1. The cube (density contrast or 1 + delta, n_fine^3, the config's box) is
   assigned to the config's mesh with the cloud-in-cell kernel of a coarse
   cell: every fine cell, at its node i * L / n_fine, deposits its mass on the
   8 surrounding coarse nodes j * L / n_mesh. That is the window of miniflims'
   particle CIC, and the result is positive with mean exactly 1.
2. Counts are drawn from the config's own bias model and likelihood with the
   given nbar and bias parameters, and a uniform response.

Writes ``data.path`` of the config (or -o): counts, response and the partial
truth read by data.source = file (truth_rho, truth_nuisance, truth_png; no
white noise), plus the source file name and the settings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from miniflims.config import load_config  # noqa: E402
from miniflims.data import cic_regrid  # noqa: E402
from miniflims.model import FieldModel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cube")
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-o", "--out", default=None, help="output .npz [data.path of the config]")
    ap.add_argument("--nbar", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=1.5, help="power-law bias exponent (bias model power_law)")
    ap.add_argument("--truth-png", type=float, nargs="*", default=None,
                    help="PNG parameters of the simulation (f_NL, ...), stored as truth_png")
    ap.add_argument("--seed", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out or cfg.data.path)
    cube = np.load(Path(args.cube).expanduser(), mmap_mode="r", allow_pickle=False)
    x = np.asarray(cube, dtype=np.float64)
    m = x.mean()
    rho_fine = x + 1.0 if abs(m) < 0.5 else x / m  # delta or 1 + delta (or counts)
    print(f"cube {cube.shape} {cube.dtype}: mean {m:.4g}, std {x.std():.4g} -> 1 + delta, min {rho_fine.min():.4g}")

    n = cfg.grid.n_mesh or cfg.grid.n_grid
    rho = cic_regrid(rho_fine, n)
    print(f"{n}^3 CIC: mean {rho.mean():.6f}, std {rho.std():.4f}, min {rho.min():.4f}, max {rho.max():.2f}")

    response = np.ones((1,) + (n,) * 3)
    model = FieldModel(cfg, response, np.zeros_like(response))
    if cfg.galaxies.bias.model != "power_law":
        raise SystemExit("this script sets the power_law exponent; extend it for other bias models")
    theta = model.nuisance_from_dict({"nbar": args.nbar, "alpha": args.alpha})
    lam = np.asarray(model.expected_counts(jnp.asarray(theta[0]), jnp.asarray(rho), jnp.asarray(response[0])))
    p = {name: float(theta[0, i]) for i, name in enumerate(model.param_names)}
    counts = model.likelihood.sample(np.random.default_rng(args.seed), lam, jnp.asarray(rho), p, model.floor)[None]
    print(f"counts: {counts.sum():.0f} galaxies, mean {counts.mean():.3f} per cell, max {counts.max():.0f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    extra = {}
    if args.truth_png is not None:
        extra["truth_png"] = np.asarray(args.truth_png, float)
    meta = dict(source=str(Path(args.cube).name), n_fine=int(cube.shape[0]), n_mesh=n, box=cfg.grid.box_size,
                nbar=args.nbar, alpha=args.alpha, seed=args.seed, truth_png=args.truth_png)
    np.savez_compressed(out, counts=np.asarray(counts, np.float32), response=response.astype(np.float32),
                        truth_rho=rho, truth_nuisance=theta, meta=json.dumps(meta), **extra)
    print(f"written {out}")


if __name__ == "__main__":
    main()
