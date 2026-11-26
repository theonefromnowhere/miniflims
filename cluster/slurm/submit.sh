#!/usr/bin/env bash
# Submit a miniflims run to the CC-IN2P3 Slurm cluster.
#
#   cluster/slurm/submit.sh [options] CONFIG [-o key=value ...]
#
# Examples
#   cluster/slurm/submit.sh configs/tcola.yaml
#   cluster/slurm/submit.sh -p gpu_h100 -t 2-00:00:00 -n tcola128 configs/tcola.yaml -o grid.n_grid=128
#   cluster/slurm/submit.sh --seeds 0-3 configs/manticore_like.yaml        # 4 independent chains
#   cluster/slurm/submit.sh --cpu -c 16 --mem 32G configs/default.yaml
#
# Options
#   --gpu              GPU job (default): partitions gpu_v100,gpu_h100, 1 GPU
#   --cpu              CPU-only job on the htc partition
#   -p, --partition P  partition(s), e.g. gpu_h100 or gpu_v100,gpu_h100
#   -t, --time T       wall time of each job [1-00:00:00 GPU, 2-00:00:00 CPU]; site max 7 days
#   -c, --cpus N       CPU cores [4 GPU, 8 CPU]
#   --mem M            memory [32G GPU, 16G CPU]
#   -n, --name NAME    run name [config file name]; output goes to RUNS_DIR/NAME
#   --runs-dir DIR     parent of the run directories [$MINIFLIMS_RUNS, else /sps/euclid/Users/$USER/miniflims_runs]
#   --seeds LIST       job array over sampler.seed (e.g. 0-3 or 1,5,9): one run dir per seed
#   --resubmits N      automatic continuations after a time-limit checkpoint [10]
#   --signal-lead S    seconds before the time limit at which to checkpoint [900]
#   -A, --account A    Slurm account [your main group]
#   --sps              declare the sps storage licence (automatic if a path is under /sps)
#   --no-plot          do not run `miniflims plot` at the end
#   --dry-run          print the sbatch command and submit nothing
#
# Everything after CONFIG is passed to `miniflims run` (use -o key=value
# overrides). The run directory is set with output.dir by the job.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB_SCRIPT="$REPO/cluster/slurm/job.slurm"

MODE=gpu PARTITION="" TIME="" CPUS="" MEM="" NAME="" RUNS_DIR="${MINIFLIMS_RUNS:-${MINIFLIMS_BASE:-/sps/euclid/Users/$USER}/miniflims_runs}"
SEEDS="" RESUBMITS=10 LEAD=900 ACCOUNT="" SPS=0 PLOT=1 DRY=0

die() { echo "submit.sh: $*" >&2; exit 2; }
need() { [ $# -ge 2 ] || die "option $1 needs a value"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) MODE=gpu ;;
        --cpu) MODE=cpu ;;
        -p | --partition) need "$@"; PARTITION="$2"; shift ;;
        -t | --time) need "$@"; TIME="$2"; shift ;;
        -c | --cpus) need "$@"; CPUS="$2"; shift ;;
        --mem) need "$@"; MEM="$2"; shift ;;
        -n | --name) need "$@"; NAME="$2"; shift ;;
        --runs-dir) need "$@"; RUNS_DIR="$2"; shift ;;
        --seeds) need "$@"; SEEDS="$2"; shift ;;
        --resubmits) need "$@"; RESUBMITS="$2"; shift ;;
        --signal-lead) need "$@"; LEAD="$2"; shift ;;
        -A | --account) need "$@"; ACCOUNT="$2"; shift ;;
        --sps) SPS=1 ;;
        --no-plot) PLOT=0 ;;
        --dry-run) DRY=1 ;;
        -h | --help) sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) die "unknown option $1 (see --help)" ;;
        *) break ;;
    esac
    shift
done
[ $# -ge 1 ] || die "no config file given (see --help)"
CONFIG="$1"
shift
[ -f "$CONFIG" ] || die "config file not found: $CONFIG"
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
[[ "$RESUBMITS" =~ ^[0-9]+$ ]] || die "--resubmits must be a non-negative integer"
[[ "$LEAD" =~ ^[0-9]+$ ]] || die "--signal-lead must be a number of seconds"

if [ "$MODE" = gpu ]; then
    PARTITION="${PARTITION:-gpu_v100,gpu_h100}" TIME="${TIME:-1-00:00:00}" CPUS="${CPUS:-4}" MEM="${MEM:-32G}"
else
    PARTITION="${PARTITION:-htc}" TIME="${TIME:-2-00:00:00}" CPUS="${CPUS:-8}" MEM="${MEM:-16G}"
fi
NAME="${NAME:-$(basename "${CONFIG%.*}")}"
mkdir -p "$RUNS_DIR"
RUNS_DIR="$(cd "$RUNS_DIR" && pwd)"
RUN_DIR="$RUNS_DIR/$NAME"
LOG_DIR="$RUN_DIR/logs"

for path in "$REPO" "$RUN_DIR" "${MINIFLIMS_VENV:-}"; do
    [[ "$path" == /sps/* ]] && SPS=1
    [[ "$path" =~ [[:space:],] ]] && die "paths with spaces or commas are not supported: $path"
done

OPTS=(--partition="$PARTITION" --time="$TIME" --cpus-per-task="$CPUS" --mem="$MEM" --signal="B:USR1@$LEAD")
[ "$MODE" = gpu ] && OPTS+=(--gpus=1)
[ -n "$ACCOUNT" ] && OPTS+=(--account="$ACCOUNT")
[ "$SPS" = 1 ] && OPTS+=(--licenses=sps)

SUBMIT=(sbatch --parsable --export=ALL "${OPTS[@]}" --job-name="mf-$NAME")
if [ -n "$SEEDS" ]; then
    SUBMIT+=(--array="$SEEDS" --output="$LOG_DIR/%x-seed%a-%A_%a.out")
else
    SUBMIT+=(--output="$LOG_DIR/%x-%j.out")
fi
SUBMIT+=("$JOB_SCRIPT" "$CONFIG" "$@")

ENVS=(
    MINIFLIMS_REPO="$REPO" MINIFLIMS_JOB_SCRIPT="$JOB_SCRIPT" MINIFLIMS_RUN_DIR="$RUN_DIR"
    MINIFLIMS_LOG_DIR="$LOG_DIR" MINIFLIMS_SBATCH_OPTS="${OPTS[*]}" MINIFLIMS_RESUBMITS_LEFT="$RESUBMITS"
    MINIFLIMS_REQUIRE_GPU="$([ "$MODE" = gpu ] && echo 1 || echo 0)" MINIFLIMS_PLOT="$PLOT"
)

if [ "$DRY" = 1 ]; then
    printf '%q ' env "${ENVS[@]}" "${SUBMIT[@]}"
    echo
    exit 0
fi

# check the config before queueing anything
set +u  # module functions may use unset variables
source "$REPO/cluster/activate.sh" >/dev/null
set -u
miniflims show-config -c "$CONFIG" "$@" >/dev/null

mkdir -p "$LOG_DIR"
JOB_ID="$(env "${ENVS[@]}" "${SUBMIT[@]}")"
echo "submitted job $JOB_ID ($PARTITION, $TIME, ${CPUS} CPUs, $MEM${SEEDS:+, seeds $SEEDS})"
echo "  run directory  $RUN_DIR${SEEDS:+/seed_<seed>}"
echo "  logs           $LOG_DIR"
echo "  follow with    squeue -u \$USER   |   tail -f $LOG_DIR/*   |   miniflims status $RUN_DIR${SEEDS:+/seed_<seed>}"
