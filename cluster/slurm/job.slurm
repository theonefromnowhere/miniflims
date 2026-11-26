#!/bin/bash
# miniflims batch job for the CC-IN2P3 Slurm cluster.
#
# Normally submitted through cluster/slurm/submit.sh, which sets the
# resources, the run directory and the resubmission options. The #SBATCH lines
# below are only the fallback for a bare `sbatch cluster/slurm/job.slurm
# CONFIG [-o key=value ...]` from the repository root.
#
# What it does:
#   - runs `miniflims run`, which resumes from RUN_DIR/checkpoint.npz if present
#     and writes a checkpoint every sampler.checkpoint.every sweeps;
#   - before the time limit Slurm sends USR1 to this script (--signal=B:USR1@...),
#     which is forwarded to miniflims: it finishes the current sweep, writes a
#     checkpoint and exits with code 3;
#   - on code 3 the job resubmits itself (afterany dependency) until the run is
#     finished or MINIFLIMS_RESUBMITS_LEFT reaches 0;
#   - while it runs, `miniflims watch` keeps RUN_DIR/progress.log and
#     progress.png up to date (one entry per checkpoint; MINIFLIMS_WATCH=0 to
#     switch off);
#   - when the run is finished it writes the plots.
#
#SBATCH --job-name=miniflims
#SBATCH --partition=gpu_v100,gpu_h100
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --signal=B:USR1@900
#SBATCH --output=slurm-%x-%j.out

set -uo pipefail   # no -e: the exit code of miniflims decides what happens next

if [ $# -lt 1 ]; then
    echo "usage: sbatch job.slurm CONFIG [-o key=value ...]" >&2
    exit 2
fi
CONFIG="$1"
shift
ARGS=("$@")

REPO="${MINIFLIMS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
JOB_SCRIPT="${MINIFLIMS_JOB_SCRIPT:-$REPO/cluster/slurm/job.slurm}"
RUN_BASE="${MINIFLIMS_RUN_DIR:-${MINIFLIMS_RUNS:-${MINIFLIMS_BASE:-/sps/euclid/Users/$USER}/miniflims_runs}/$(basename "${CONFIG%.*}")}"
LOG_DIR="${MINIFLIMS_LOG_DIR:-$RUN_BASE/logs}"
SBATCH_OPTS="${MINIFLIMS_SBATCH_OPTS:---partition=gpu_v100,gpu_h100 --gpus=1 --time=1-00:00:00 --cpus-per-task=4 --mem=32G --signal=B:USR1@900}"
RESUBMITS_LEFT="${MINIFLIMS_RESUBMITS_LEFT:-10}"
JOB_ID="${SLURM_JOB_ID:-local}"

# independent chains: one seed (and run directory) per array task
SEED="${MINIFLIMS_SEED:-${SLURM_ARRAY_TASK_ID:-}}"
RUN_DIR="$RUN_BASE"
SEED_ARGS=()
if [ -n "$SEED" ]; then
    RUN_DIR="$RUN_BASE/seed_$SEED"
    SEED_ARGS=(-o "sampler.seed=$SEED")
fi
mkdir -p "$RUN_DIR" "$LOG_DIR"
cd "$REPO" || exit 1

log() { echo "[job $JOB_ID $(date '+%F %T')] $*"; }

# refuse to share a run directory with another live job
LOCK="$RUN_DIR/.slurm_job"
if [ -f "$LOCK" ]; then
    OTHER="$(cat "$LOCK")"
    if [ "$OTHER" != "$JOB_ID" ] && command -v squeue >/dev/null \
        && squeue -h -j "$OTHER" -t PENDING,RUNNING,CONFIGURING 2>/dev/null | grep -q .; then
        log "job $OTHER is still using $RUN_DIR; not starting a second one"
        exit 1
    fi
fi
echo "$JOB_ID" > "$LOCK"

set +u  # module functions may use unset variables
# shellcheck source=../activate.sh
source "$REPO/cluster/activate.sh" || exit 1
set -u

log "node $(hostname), partition ${SLURM_JOB_PARTITION:-?}, ${SLURM_CPUS_PER_TASK:-?} CPUs"
log "config $CONFIG ${ARGS[*]+${ARGS[*]}} ${SEED_ARGS[*]+${SEED_ARGS[*]}}"
log "run directory $RUN_DIR, $RESUBMITS_LEFT automatic resubmission(s) left"
if command -v nvidia-smi >/dev/null && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/    GPU: /'
fi
if [ "${MINIFLIMS_REQUIRE_GPU:-0}" = 1 ]; then
    if ! python -c "import jax, sys; print('    JAX devices:', jax.devices()); sys.exit(jax.default_backend() != 'gpu')"; then
        log "JAX does not see a GPU on this node; stopping (run cluster/check_gpu.py on an interactive GPU node)"
        exit 1
    fi
fi

miniflims run -c "$CONFIG" ${ARGS[@]+"${ARGS[@]}"} ${SEED_ARGS[@]+"${SEED_ARGS[@]}"} -o "output.dir=$RUN_DIR" &
PID=$!
WATCH_PID=""
if [ "${MINIFLIMS_WATCH:-1}" = 1 ]; then
    # restarted if it returns early (e.g. right after n_samples was increased,
    # when the old checkpoint still counts as finished)
    ( while kill -0 "$PID" 2>/dev/null; do
          miniflims watch "$RUN_DIR" --interval "${MINIFLIMS_WATCH_INTERVAL:-120}" >/dev/null 2>&1
          sleep 60
      done ) &
    WATCH_PID=$!
fi
stop_watch() {
    [ -n "$WATCH_PID" ] || return 0
    pkill -P "$WATCH_PID" 2>/dev/null
    kill "$WATCH_PID" 2>/dev/null
    wait "$WATCH_PID" 2>/dev/null
    miniflims watch "$RUN_DIR" --once >/dev/null 2>&1 || true
}
# USR1 comes from --signal before the time limit: checkpoint and continue in a
# new job. TERM comes from scancel (or the limit itself): checkpoint and stop.
REASON=""
trap 'REASON=${REASON:-timelimit}; log "time limit approaching: asking miniflims to checkpoint"; kill -USR1 "$PID" 2>/dev/null' USR1
trap 'REASON=${REASON:-terminated}; log "terminated: asking miniflims to checkpoint, no continuation"; kill -USR1 "$PID" 2>/dev/null' TERM
while :; do
    wait "$PID"
    STATUS=$?
    if [ "$STATUS" -gt 128 ] && kill -0 "$PID" 2>/dev/null; then
        continue  # wait was interrupted by the trap; miniflims is still running
    fi
    if [ "$STATUS" -gt 128 ]; then
        wait "$PID" 2>/dev/null
        S2=$?
        [ "$S2" -ne 127 ] && STATUS=$S2
    fi
    break
done
trap - USR1 TERM
rm -f "$LOCK"
stop_watch

case "$STATUS" in
0)
    log "run finished"
    if [ "${MINIFLIMS_PLOT:-1}" = 1 ]; then
        miniflims plot "$RUN_DIR" || log "plotting failed (the run itself is fine)"
    fi
    miniflims status "$RUN_DIR"
    ;;
3)
    miniflims status "$RUN_DIR"
    if [ "$REASON" = terminated ]; then
        log "checkpointed after scancel/termination; not resubmitting (submit again to continue)"
        exit 3
    elif [ "$RESUBMITS_LEFT" -gt 0 ]; then
        # shellcheck disable=SC2086  # SBATCH_OPTS is a list of options
        NEW_ID=$(env -u SLURM_ARRAY_TASK_ID -u SLURM_ARRAY_JOB_ID \
            MINIFLIMS_REPO="$REPO" MINIFLIMS_JOB_SCRIPT="$JOB_SCRIPT" MINIFLIMS_RUN_DIR="$RUN_BASE" \
            MINIFLIMS_LOG_DIR="$LOG_DIR" MINIFLIMS_SBATCH_OPTS="$SBATCH_OPTS" MINIFLIMS_SEED="$SEED" \
            MINIFLIMS_RESUBMITS_LEFT=$((RESUBMITS_LEFT - 1)) \
            sbatch --parsable --export=ALL $SBATCH_OPTS --job-name="${SLURM_JOB_NAME:-miniflims}" \
            --output="$LOG_DIR/%x${SEED:+-seed$SEED}-%j.out" --dependency="afterany:$JOB_ID" \
            "$JOB_SCRIPT" "$CONFIG" ${ARGS[@]+"${ARGS[@]}"})
        if [ -n "$NEW_ID" ]; then
            log "checkpointed; continuation submitted as job $NEW_ID"
        else
            log "checkpointed, but resubmission failed: resubmit by hand to continue"
            exit 3
        fi
    else
        log "checkpointed; no automatic resubmissions left: submit again to continue"
        exit 3
    fi
    ;;
*)
    log "miniflims failed with exit code $STATUS; not resubmitting"
    exit "$STATUS"
    ;;
esac
