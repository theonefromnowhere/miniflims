"""Whitening of the PineTree parameters (galaxies.pinetree.whitening).

The raw PineTree parameters are strongly correlated (Hessian condition number
~1e6 on a 1 (Gpc/h)^3 box), which univariate slice sampling cannot cross. This
computes the Hessian of -ln L in the sampled raw parameters at theta_ref (the
mock truth, or init) with the density field fixed at the truth (truth/rho of a
mock, or truth_rho of a data file), and writes L such that theta = theta_ref + L z
has z ~ N(0, 1) under the Laplace approximation (eigenvalues |w| floored at
--floor-frac times the largest: flat or slightly non-convex directions get a
wide but finite scale). The run then samples z (names z0, z1, ...).

The preconditioner changes only how the chain moves, not the target. With the
true field it is an oracle choice, fine for self-consistency tests; for real
data use the field of a previous run.

    .venv-run/bin/python scripts/pinetree_whitening.py -c configs/pinetree_sc_theta.yaml
    (writes the file named by galaxies.pinetree.whitening; -o to choose another path)
"""
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from miniflims.config import load_config, parse_overrides, update
from miniflims.data import load_dataset
from miniflims.model import FieldModel
from miniflims.pinetree import NAMES, whitening_matrix
from miniflims.pipeline import set_precision


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-o", "--out", default=None, help="output .npz [galaxies.pinetree.whitening of the config]")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    ap.add_argument("--floor-frac", type=float, default=1e-6)
    a = ap.parse_args()

    cfg = load_config(a.config, parse_overrides(a.set))
    out = a.out or cfg.galaxies.pinetree.whitening
    if out is None:
        raise SystemExit("give -o or set galaxies.pinetree.whitening in the config")
    raw = update(cfg, {"galaxies": {"pinetree": {"whitening": None}}})  # raw parameters, same mock
    if raw.galaxies.population != "pinetree":
        raise SystemExit("galaxies.population must be pinetree")
    set_precision(raw.precision)
    data = load_dataset(raw)
    if not data.truth or "rho" not in data.truth:
        raise SystemExit("needs the true density (a mock, or truth_rho in the data file)")
    model = FieldModel(raw, data.response, data.counts, halos=data.halos)
    rho = jnp.asarray(data.truth["rho"])
    theta0 = jnp.asarray(data.truth["nuisance"][0] if "nuisance" in data.truth else model.default_nuisance()[0])
    sampled = np.flatnonzero(model.sampled)

    def nll(ts):
        return -model.log_likelihood_row(theta0.at[sampled].set(ts), rho, model.counts[0], model.response[0])

    print(f"{int(data.counts.sum())} haloes; Hessian in {sampled.size} parameters ...", flush=True)
    grad = np.asarray(jax.grad(nll)(theta0[sampled]))
    hess = np.asarray(jax.hessian(nll)(theta0[sampled]))
    L, w = whitening_matrix(hess, a.floor_frac)
    print("eigenvalues of the Hessian:", np.array2string(w, precision=3, max_line_width=110))
    print(f"|grad| at theta_ref: {np.abs(grad).max():.3g} (0 only at the maximum-likelihood point)")
    sd = np.sqrt(np.sum(L**2, axis=1))  # marginal scale of each raw parameter implied by L
    print(f"{'parameter':>10s} {'theta_ref':>12s} {'scale':>10s}")
    for i, j in enumerate(sampled):
        print(f"{NAMES[j]:>10s} {float(theta0[j]):12.5f} {sd[i]:10.3g}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, theta_ref=np.asarray(theta0), L=L, sampled_raw=sampled, names=np.array(NAMES),
             eigenvalues=w, hessian=hess, grad=grad, n_halos=int(data.counts.sum()),
             config=str(Path(a.config).name), mock_seed=int(raw.data.mock.seed))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
