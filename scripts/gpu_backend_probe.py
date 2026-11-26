"""Check an experimental JAX GPU backend (e.g. jax-mps, metaljax on Apple
Silicon) for what miniflims needs, against the CPU in the same process.

    python scripts/gpu_backend_probe.py [--n 64]

1. Primitive checks: rfftn / irfftn accuracy, complex arithmetic, scatter-add
   (CIC painting), gather, gradients through FFTs, lax.scan.
2. The real workload: one float32 posterior value + gradient at n^3 with the
   Zel'dovich forward model and the feature PNG kernel (jaxpng), on the
   accelerator and on the CPU: relative difference and time per gradient.

Apple GPUs have no float64, so everything runs in float32. A backend is only
usable here if the gradient agrees with the CPU to ~1e-4 relative.
"""
from __future__ import annotations

import argparse
import time

import jax

jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402


REPEAT = 10


def timed(fn, *args, repeat=None):
    """(output, timing dict). The first call (JIT compilation + first run) is
    timed separately; then every call is blocked on before the next one, as
    in HMC where each gradient depends on the previous step."""
    repeat = repeat or REPEAT
    t = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    first = time.perf_counter() - t
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t)
    times = np.array(times)
    return out, {"first": first, "median": float(np.median(times)), "min": float(times.min()), "max": float(times.max())}


def rel(a, b):
    a, b = np.asarray(a, np.complex128), np.asarray(b, np.complex128)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--cpu-only", action="store_true", help="dry run: use the CPU as the 'accelerator'")
    ap.add_argument("--repeat", type=int, default=10, help="timed gradients per device (after the compiling call)")
    args = ap.parse_args()
    global REPEAT
    REPEAT = args.repeat
    n = args.n
    cpu = jax.devices("cpu")[0]
    acc = cpu if args.cpu_only else jax.devices()[0]
    print("jax", jax.__version__, "| default device:", acc, "| platform:", acc.platform)
    if acc.platform == "cpu" and not args.cpu_only:
        print("no accelerator found: is the plugin installed in this environment?")
        return

    def on(dev, f, *a):
        with jax.default_device(dev):
            return f(*[jax.device_put(x, dev) for x in a])

    rng = np.random.default_rng(0)
    x = rng.standard_normal((n,) * 3).astype(np.float32)
    idx = rng.integers(0, n**3, 300_000).astype(np.int32)
    checks = {
        "rfftn": lambda v: jnp.fft.rfftn(v),
        "irfftn(rfftn)": lambda v: jnp.fft.irfftn(jnp.fft.rfftn(v) * (1.0 + 0.5j), s=v.shape),
        "complex arithmetic": lambda v: jnp.abs(jnp.fft.rfftn(v) * jnp.exp(1j * 0.3)) ** 2,
        "scatter-add (CIC)": lambda v: jnp.zeros(n**3, jnp.float32).at[jnp.asarray(idx)].add(jnp.tile(v.ravel(), 2)[: len(idx)]),
        "gather": lambda v: v.ravel()[jnp.asarray(idx)],
        "grad through FFT": lambda v: jax.grad(lambda u: jnp.sum(jnp.abs(jnp.fft.rfftn(u)) ** 2))(v),
        "lax.scan": lambda v: jax.lax.scan(lambda c, t: (c * 0.9 + t, None), jnp.zeros_like(v), jnp.stack([v] * 8))[0],
    }
    print("\n1. primitives (relative difference accelerator vs CPU, float32):")
    for name, f in checks.items():
        try:
            ref = on(cpu, jax.jit(f), x)
            got = on(acc, jax.jit(f), x)
            r = rel(got, ref)
            print(f"   {'OK  ' if r < 1e-4 else 'BAD '} {name:22s} {r:.1e}")
        except Exception as e:  # unsupported op
            print(f"   FAIL {name:22s} {type(e).__name__}: {str(e).splitlines()[0][:120]}")

    print(f"\n2. miniflims posterior gradient, {n}^3, Zel'dovich + feature PNG (float32):")
    from miniflims import load_config
    from miniflims.data import make_mock
    from miniflims.model import FieldModel

    over = {"precision": "float32", "grid": {"n_grid": n, "box_size": 1024.0},
            "png": {"enabled": True, "template": "feature", "params": {"omega": {"init": 20.0}}},
            "data": {"mock": {"truth": {"nbar": 5.0, "alpha": 1.5, "f_NL": 500.0}}}}
    cfg = load_config(overrides=over)
    with jax.default_device(cpu):
        d = make_mock(cfg)
        m = FieldModel(cfg, d.response, d.counts)
    jax.config.update("jax_enable_x64", False)  # jaxpng's import switches it on
    xw = d.truth["white_noise"].astype(np.float32)
    th = d.truth["nuisance"].astype(np.float32)
    p = d.truth["png"].astype(np.float32)
    vg = jax.jit(jax.value_and_grad(lambda v, t, q: m.log_posterior(v, t, None, q)))
    res = {}
    for label, dev in (("cpu", cpu), ("accelerator", acc)):
        try:
            with jax.default_device(dev):
                args_ = [jax.device_put(a, dev) for a in (xw, th, p)]
                (val, grad), dt = timed(vg, *args_)
            res[label] = (float(val), np.asarray(grad), dt)
            print(f"   {label:12s} log posterior {float(val):.6e}   per gradient: median {dt['median'] * 1e3:7.1f} ms "
                  f"(min {dt['min'] * 1e3:.1f}, max {dt['max'] * 1e3:.1f}, {REPEAT} calls);  "
                  f"first call incl. JIT {dt['first']:.1f} s")
        except Exception as e:
            print(f"   {label:12s} FAIL {type(e).__name__}: {str(e).splitlines()[0][:160]}")
    if len(res) == 2:
        (v0, g0, t0), (v1, g1, t1) = res["cpu"], res["accelerator"]
        print(f"   value rel. diff {abs(v1 - v0) / abs(v0):.1e}, gradient rel. diff {rel(g1, g0):.1e}, "
              f"speed-up (median) {t0['median'] / t1['median']:.2f}x")
        print("   usable" if rel(g1, g0) < 1e-4 and np.all(np.isfinite(g1)) else "   NOT usable: gradients disagree")

    print("\n3. float32 vs float64 on the CPU (the precision cost of any Apple-GPU route):")
    if "cpu" in res:
        jax.config.update("jax_enable_x64", True)
        cfg64 = load_config(overrides={**over, "precision": "float64"})
        with jax.default_device(cpu):
            m64 = FieldModel(cfg64, d.response, d.counts)
            jax.config.update("jax_enable_x64", True)
            v64, g64 = jax.jit(jax.value_and_grad(lambda v: m64.log_posterior(v, jnp.asarray(d.truth["nuisance"]), None,
                                                                              jnp.asarray(d.truth["png"]))))(jnp.asarray(d.truth["white_noise"]))
        v32, g32, _ = res["cpu"]
        print(f"   log posterior: float32 {v32:.8e}, float64 {float(v64):.8e} (rel. diff {abs(v32 - float(v64)) / abs(float(v64)):.1e})")
        print(f"   gradient rel. diff float32 vs float64: {rel(g32, np.asarray(g64)):.1e}")


if __name__ == "__main__":
    main()
