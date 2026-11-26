#!/usr/bin/env bash
# One-time installation of miniflims on the cluster (RHEL 9, environment
# modules). Run it on a login node, which has internet access:
#
#     bash cluster/setup_env.sh            # GPU build: JAX with CUDA 12 from pip
#     bash cluster/setup_env.sh cpu        # CPU-only build
#
# Then, in every shell or job:  source cluster/activate.sh
#
# Settings (environment variables, all optional):
#   MINIFLIMS_BASE           base directory for the environment and runs
#                            [/sps/euclid/Users/$USER]
#   MINIFLIMS_VENV           where the environment goes   [$MINIFLIMS_BASE/venvs/miniflims]
#                            The GPU build is ~4 GB (CUDA and cuDNN wheels).
#   MINIFLIMS_PYTHON_MODULE  Python module                [Programming_Languages/python/3.13.17]
#   MINIFLIMS_EXTRAS         package extras               [test,plot]
#                            add "compare" for scripts/compare_public_codes.py
set -euo pipefail

TARGET="${1:-gpu}"
case "$TARGET" in gpu | cpu) ;; *) echo "usage: $0 [gpu|cpu]" >&2; exit 2 ;; esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MINIFLIMS_PYTHON_MODULE="${MINIFLIMS_PYTHON_MODULE:-Programming_Languages/python/3.13.17}"
export MINIFLIMS_BASE="${MINIFLIMS_BASE:-/sps/euclid/Users/$USER}"
export MINIFLIMS_VENV="${MINIFLIMS_VENV:-$MINIFLIMS_BASE/venvs/miniflims}"
EXTRAS="${MINIFLIMS_EXTRAS:-test,plot}"
CONSTRAINTS="$REPO/cluster/constraints.txt"

if ! command -v module >/dev/null 2>&1 && [ -f /etc/profile.d/modules.sh ]; then
    . /etc/profile.d/modules.sh
fi
# Only Python is needed. Do not load nvhpc or a CUDA-flavoured libtorch: JAX
# brings its own CUDA libraries.
module load "$MINIFLIMS_PYTHON_MODULE"
echo "python: $(command -v python3) ($(python3 --version))"

if [ ! -d "$MINIFLIMS_VENV" ]; then
    mkdir -p "$(dirname "$MINIFLIMS_VENV")"
    python3 -m venv "$MINIFLIMS_VENV"
fi
. "$MINIFLIMS_VENV/bin/activate"
export PIP_CACHE_DIR="$MINIFLIMS_VENV/.pip-cache"   # keep the download cache out of $HOME

python -m pip install --upgrade pip
if [ "$TARGET" = gpu ]; then
    python -m pip install -c "$CONSTRAINTS" "jax[cuda12]"
else
    python -m pip install -c "$CONSTRAINTS" jax
fi
python -m pip install -c "$CONSTRAINTS" -e "$REPO[$EXTRAS]"
# jaxpng (primordial non-Gaussianity, png.enabled) if its project folder is present;
# compat mode stops the jaxpng/ folder from shadowing the package
if [ -f "$REPO/jaxpng/pyproject.toml" ]; then
    python -m pip install -c "$CONSTRAINTS" -e "$REPO/jaxpng" --config-settings editable_mode=compat
fi
rm -rf "$PIP_CACHE_DIR"

echo
echo "installed into $MINIFLIMS_VENV"
python - <<'EOF'
import jax, jaxlib, numpy
print(f"jax {jax.__version__}, jaxlib {jaxlib.__version__}, numpy {numpy.__version__}")
print("default backend:", jax.default_backend())
print("devices:", jax.devices())
EOF
if [ "$TARGET" = gpu ]; then
    cat <<'EOF'

The login node usually has no GPU, so "cpu" above is expected there. On a GPU
node, check with:
    source cluster/activate.sh && python cluster/check_gpu.py
EOF
fi
