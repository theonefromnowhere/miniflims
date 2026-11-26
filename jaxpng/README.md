# jaxpng

JAX rewrite of the field-generation core of **GENGARS** — generation of a primordial
gravitational potential Φ = Φ^G + f_NL·Φ^NG with an arbitrary *separable* target
bispectrum, via the Wagner–Verde reduced-bispectrum kernel + Schwinger parameterization.

Fully 3D, differentiable (`jit`/`grad`/`vmap`), GPU-ready. See `../task.md` for the plan
and logbook, and `../CLAUDE.md` for project conventions.

## Install (into the `conda_pylians` env)

```sh
conda activate conda_pylians
pip install -e .            # editable install; jax/Pylians already present in the env
pytest                      # run the test suite
```

## Layout
- `jaxpng/config.py`   — float64 default, dtype switch, `Box` geometry.
- `jaxpng/grids.py`    — 3D k-grids, P_Φ(k), rk *(stub)*.
- `jaxpng/cosmology.py`— T(k), P(k), growth, Poisson M(k) *(stub)*.
- `jaxpng/gaussian.py` — Φ^G from white noise *(stub)*.
- `jaxpng/bispectra.py`— separable templates *(stub)*.
- `jaxpng/phing.py`    — Schwinger-parameterized Φ^NG core *(stub)*; `schwinger_t_grid_log` +
  `phiNG(..., t_grid=...)`: trapezoid in ln t, exponentially convergent (32 nodes → 3e-5).
- `jaxpng/io_c.py`     — read/write GENGARS C binary field files.
- `jaxpng/measure.py`  — Pylians wrappers for 3D P(k)/B(k).
