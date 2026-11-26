#!/bin/zsh
# Run (or resume) several miniflims runs one after another, each followed by
# `miniflims watch` (progress.log / progress.png) once it has written a new
# checkpoint.  Each job is "<config> [-o key=value ...]" separated by "--":
#
#   scripts/chain_runs.sh runs/a/config.yaml -o sampler.n_samples=1000 -- \
#                         runs/b/config.yaml -o sampler.n_samples=1000
#
# The run directory is taken from output.dir of the resolved config.
set -u
cd "${0:A:h}/.."
BIN=${MINIFLIMS_BIN:-.venv-run/bin/miniflims}
PY=${BIN:h}/python

iteration() {  # current checkpoint sweep of run dir $1 (-1 if none)
  "$PY" -c 'import sys; from miniflims.checkpoint import read_meta
m = read_meta(sys.argv[1]); print(-1 if m is None else m["iteration"])' "$1"
}

run_one() {
  local args=("$@")
  local dir
  dir=$("$BIN" show-config -c "${args[@]}" | "$PY" -c 'import sys, yaml; print(yaml.safe_load(sys.stdin)["output"]["dir"])')
  local start=$(iteration "$dir")
  echo "$(date '+%F %T')  start $dir (checkpoint at sweep $start): ${args[*]}"
  "$BIN" run -c "${args[@]}" >> "$dir/run_stdout.log" 2>&1 &
  local pid=$!
  (
    while kill -0 $pid 2>/dev/null; do
      [[ $(iteration "$dir") -gt $start ]] && exec "$BIN" watch "$dir" --interval 60
      sleep 30
    done
  ) &
  local wpid=$!
  wait $pid
  local rc=$?
  sleep 5; kill $wpid 2>/dev/null; "$BIN" watch "$dir" --once >/dev/null 2>&1
  echo "$(date '+%F %T')  end   $dir: exit $rc"
}

job=()
for a in "$@"; do
  if [[ $a == -- ]]; then run_one "${job[@]}"; job=(); else job+=("$a"); fi
done
(( ${#job[@]} )) && run_one "${job[@]}"
