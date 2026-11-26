"""Check that JAX sees the GPU and can run the miniflims forward models and
gradients on it, in float64 as the shipped configs do.

    source cluster/activate.sh && python cluster/check_gpu.py [CONFIG ...]

Without arguments: configs/default.yaml, configs/tcola.yaml and, if jaxpng is
installed, configs/png_feature_w20_10k.yaml (gradient through Phi_NG).
"""
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

try:
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30,
    )
    print("nvidia-smi:", smi.stdout.strip() or smi.stderr.strip())
except (OSError, subprocess.TimeoutExpired) as e:
    print("nvidia-smi not available:", e)

print("jax", jax.__version__, "| backend:", jax.default_backend(), "| devices:", jax.devices())
if jax.default_backend() != "gpu":
    raise SystemExit("JAX is not using a GPU on this node (see the devices above).")

from miniflims import load_config  # noqa: E402
from miniflims.data import make_mock  # noqa: E402
from miniflims.model import FieldModel  # noqa: E402

configs = Path(__file__).resolve().parents[1] / "configs"
paths = [Path(a) for a in sys.argv[1:]] or [configs / "default.yaml", configs / "tcola.yaml"] + (
    [configs / "png_feature_w20_10k.yaml"] if importlib.util.find_spec("jaxpng") else [])
for path in paths:
    name = path.stem
    cfg = load_config(path, {"output": {"dir": None}})
    data = make_mock(cfg)
    model = FieldModel(cfg, data.response, data.counts)
    x = jnp.asarray(data.truth["white_noise"])
    theta = jnp.asarray(data.truth["nuisance"])
    png = data.truth.get("png")
    png = None if png is None else jnp.asarray(png)
    vg = jax.jit(jax.value_and_grad(lambda v: model.log_posterior(v, theta, png=png)))
    t = time.time()
    lp, g = vg(x)
    g.block_until_ready()
    t_compile = time.time() - t
    t = time.time()
    for _ in range(5):
        lp, g = vg(x)
    g.block_until_ready()
    print(f"{name:20s} {cfg.grid.n_grid}^3: log posterior {float(lp):.6e}, |grad| {float(jnp.linalg.norm(g)):.6e}, "
          f"first call {t_compile:.1f} s, then {(time.time() - t) / 5 * 1e3:.1f} ms per gradient")
    assert np.isfinite(float(lp)) and bool(jnp.all(jnp.isfinite(g)))
print("GPU check passed")
