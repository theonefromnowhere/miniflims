# Source this in every shell or batch job that runs miniflims:
#
#     source cluster/activate.sh
#
# Same variables as cluster/setup_env.sh; override them before sourcing.

MINIFLIMS_PYTHON_MODULE="${MINIFLIMS_PYTHON_MODULE:-Programming_Languages/python/3.13.17}"
MINIFLIMS_BASE="${MINIFLIMS_BASE:-/sps/euclid/Users/$USER}"
MINIFLIMS_VENV="${MINIFLIMS_VENV:-$MINIFLIMS_BASE/venvs/miniflims}"
MINIFLIMS_RUNS="${MINIFLIMS_RUNS:-$MINIFLIMS_BASE/miniflims_runs}"
export MINIFLIMS_BASE MINIFLIMS_VENV MINIFLIMS_RUNS

if ! command -v module >/dev/null 2>&1 && [ -f /etc/profile.d/modules.sh ]; then
    . /etc/profile.d/modules.sh
fi
module load "$MINIFLIMS_PYTHON_MODULE" || return 1 2>/dev/null || exit 1

if [ ! -f "$MINIFLIMS_VENV/bin/activate" ]; then
    echo "miniflims: no environment at $MINIFLIMS_VENV; run cluster/setup_env.sh first" >&2
    return 1 2>/dev/null || exit 1
fi
. "$MINIFLIMS_VENV/bin/activate"

# JAX uses the CUDA libraries installed by pip. CUDA directories on
# LD_LIBRARY_PATH (nvhpc, libtorch.cu118, ...) would be loaded first and can
# break it, so drop them for this shell only.
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    LD_LIBRARY_PATH="$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -viE 'cuda|nvhpc|cudnn|libtorch' | paste -sd: -)"
    export LD_LIBRARY_PATH
fi

# Reuse compiled XLA programs across runs (tCOLA gradients take a while to
# compile). Safe to delete at any time.
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$MINIFLIMS_VENV/xla-cache}"

# By default JAX reserves 75% of GPU memory at start-up. Uncomment to
# allocate on demand instead (slower, but lets several jobs share a GPU):
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
