#!/bin/bash
# Start Claude Code in an isolated container that can ONLY see the project folder(s) you pass.
# Reusable across projects; pass one OR several folders.
#
# Usage:   export ANTHROPIC_API_KEY=sk-ant-...        # (or use /login once, with a subscription)
#          bash run_claude.sh /path/to/project                 # -> that folder is /work
#          bash run_claude.sh ~/png ~/other-project            # -> each appears as /work/<name>
#
# Everything Claude (and anything it runs) can touch is limited to those folders.
# Login/config is shared across projects in ~/.claude-cluster (override with CLAUDE_STATE).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SIF="${CLAUDE_SIF:-$HERE/claude.sif}"
STATE="${CLAUDE_STATE:-$HOME/.claude-cluster}"        # shared login/config (NOT inside any project)

# build the bind list: 1 folder -> /work ; several -> /work/<basename> each
ARGS=("$@"); [ ${#ARGS[@]} -eq 0 ] && ARGS=("$PWD")
BINDS=()
if [ ${#ARGS[@]} -eq 1 ]; then
    P="$(cd "${ARGS[0]}" && pwd)"; BINDS+=(--bind "$P:/work")
else
    for a in "${ARGS[@]}"; do P="$(cd "$a" && pwd)"; BINDS+=(--bind "$P:/work/$(basename "$P")"); done
fi

CMD="$(command -v apptainer || command -v singularity || true)"
[ -n "$CMD" ] || { echo "ERROR: neither 'apptainer' nor 'singularity' found (try 'module load apptainer')"; exit 1; }
[ -f "$SIF" ] || { echo "ERROR: $SIF not found -- build it first: $CMD build --fakeroot $SIF $HERE/claude.def"; exit 1; }

mkdir -p "$STATE"
[ -n "${ANTHROPIC_API_KEY:-}" ] && { export APPTAINERENV_ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"
                                     export SINGULARITYENV_ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"; }

exec "$CMD" run --cleanenv "${BINDS[@]}" --pwd /work \
    --bind "$STATE:/state" --env HOME=/state \
    "$SIF"
