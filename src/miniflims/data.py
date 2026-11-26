"""Survey response (selection) functions, mock data and data loading."""
from __future__ import annotations

from typing import NamedTuple, Optional

import jax.numpy as jnp
import numpy as np

from .config import Config
from .grid import Grid
from .model import FieldModel
from .pinetree import counts_grid as pinetree_counts_grid
from .registry import Registry

SELECTIONS = Registry("selection")


class Dataset(NamedTuple):
    counts: np.ndarray  # (n_catalogues, n_mesh, n_mesh, n_mesh)
    response: np.ndarray  # same shape, >= 0
    truth: Optional[dict] = None  # mocks only: white_noise, rho, nuisance
    # population pinetree: {"voxel": flat mesh ids, "ln_mass": ln(M / (M_sun/h))};
    # ``counts`` is then the gridded catalogue
    halos: Optional[dict] = None


# --------------------------------------------------------------------------- #
# Selections: f(sel_cfg, mesh: Grid) -> response array on the mesh
# --------------------------------------------------------------------------- #
@SELECTIONS.register("uniform")
def uniform_selection(sel, mesh: Grid):
    return np.ones((mesh.n,) * 3)


@SELECTIONS.register("radial")
def radial_selection(sel, mesh: Grid):
    """R = exp(-(r/r0)^power) around the observer, zero beyond r_max."""
    obs = sel.observer if sel.observer is not None else (mesh.box / 2,) * 3
    x, y, z = mesh.coords()
    r = np.sqrt((x - obs[0]) ** 2 + (y - obs[1]) ** 2 + (z - obs[2]) ** 2)
    resp = np.exp(-((r / sel.r0) ** sel.power))
    if sel.r_max is not None:
        resp = np.where(r <= sel.r_max, resp, 0.0)
    return resp


@SELECTIONS.register("file")
def file_selection(sel, mesh: Grid):
    if sel.path is None:
        raise ValueError("selection type 'file' needs a path")
    return np.load(sel.path)


def build_response(cfg: Config) -> np.ndarray:
    n_cat = cfg.galaxies.n_catalogues
    mesh = Grid(cfg.grid.n_mesh or cfg.grid.n_grid, cfg.grid.box_size)
    sels = cfg.data.mock.selections
    if len(sels) == 1:
        sels = sels * n_cat
    if len(sels) != n_cat:
        raise ValueError(
            f"data.mock.selections has {len(sels)} entries; need 1 or {n_cat}"
        )
    return np.stack([np.asarray(SELECTIONS.get(s.type)(s, mesh), float) for s in sels])


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def make_mock(cfg: Config) -> Dataset:
    """Draw white noise from the prior, run the forward model and sample counts
    from the configured likelihood at the configured truth."""
    mock = cfg.data.mock
    rng = np.random.default_rng(mock.seed)
    response = build_response(cfg)
    model = FieldModel(cfg, response)

    x = rng.standard_normal((cfg.grid.n_grid,) * 3)
    # truth: PNG parameters (f_NL, omega, phase) vs galaxy nuisance parameters
    png_names = set(model.png.names) if model.png is not None else set()
    truth_png = {k: v for k, v in mock.truth.items() if k in png_names}
    png = model.png.from_dict(truth_png) if model.png is not None else None
    rho, ops = model.galaxy_fields(jnp.asarray(x), None if png is None else jnp.asarray(png))
    rho = np.asarray(rho)
    theta = model.nuisance_from_dict({k: v for k, v in mock.truth.items() if k not in png_names})
    truth = dict(white_noise=x, rho=rho, nuisance=theta)
    if png is not None:
        truth["png"] = png
    if model.pinetree is not None:  # mass-resolved halo catalogue
        vox, ln_mass = model.pinetree.sample(rng, rho, theta[0], response[0])
        counts = pinetree_counts_grid(vox, model.mesh.n)[None]
        return Dataset(counts, response, truth, {"voxel": vox, "ln_mass": ln_mass})
    counts = np.empty_like(response)
    for c in range(model.n_catalogues):
        lam = np.asarray(model.expected_counts(jnp.asarray(theta[c]), jnp.asarray(rho), response[c], ops))
        p = {name: float(theta[c, i]) for i, name in enumerate(model.param_names)}
        counts[c] = model.likelihood.sample(rng, lam, rho, p, model.floor)
        counts[c] = np.where(response[c] > 0, counts[c], 0.0)
    return Dataset(counts, response, truth)


def cic_regrid(rho_fine: np.ndarray, n_out: int) -> np.ndarray:
    """Density on a fine periodic grid (values at nodes i * L / n_fine)
    re-assigned to a coarser grid (nodes j * L / n_out) with the cloud-in-cell
    kernel of a coarse cell, i.e. the window of particle CIC on the coarse
    mesh. Positive input stays positive; the mean is preserved."""
    n = rho_fine.shape[0]
    if n % n_out:
        raise ValueError(f"grid size {n} is not a multiple of {n_out}")
    r = n // n_out
    w = {m: 1.0 - abs(m) / r for m in range(-(r - 1), r)}  # sums to r
    out = np.asarray(rho_fine, np.float64)
    for axis in range(3):
        acc = 0.0
        for m, wm in w.items():
            acc = acc + wm * np.take(np.roll(out, -m, axis=axis), np.arange(0, out.shape[axis], r), axis=axis)
        out = acc / r
    return out


def load_dataset(cfg: Config) -> Dataset:
    if cfg.data.source == "mock":
        return make_mock(cfg)
    if cfg.data.source == "file":
        if cfg.data.path is None:
            raise ValueError("data.source=file needs data.path")
        with np.load(cfg.data.path) as f:
            response = np.asarray(f["response"], float)
            # optional, partial truth (e.g. data drawn from a simulation):
            # truth_rho (evolved density on the mesh), truth_nuisance
            # (n_catalogues, n_params), truth_png; no white noise
            truth = {k[len("truth_"):]: np.asarray(f[k]) for k in f.files if k.startswith("truth_")}
            # population pinetree: the catalogue itself (halo_voxel = flat mesh
            # index, halo_mass in M_sun/h); counts may then be omitted
            halos = None
            if "halo_voxel" in f.files:
                halos = {"voxel": np.asarray(f["halo_voxel"], np.int64),
                         "ln_mass": np.log(np.asarray(f["halo_mass"], float))}
            if "counts" in f.files:
                counts = np.asarray(f["counts"], float)
            elif halos is not None:
                counts = pinetree_counts_grid(halos["voxel"], response.shape[-1])
            else:
                raise ValueError(f"{cfg.data.path}: needs counts (or halo_voxel, halo_mass)")
        if counts.ndim == 3:
            counts, response = counts[None], response[None]
        return Dataset(counts, response, truth or None, halos)
    raise ValueError(f"unknown data.source '{cfg.data.source}' (mock or file)")
