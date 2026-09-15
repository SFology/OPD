#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/remote-home/liufengkai/projects/OPD
CONDA_SH=/remote-home/share/anaconda3/etc/profile.d/conda.sh
TMUX_BIN=/attached/remote-home1/liufengkai/tools/tmux-env/bin/tmux
STORAGE_ROOT=/attached/remote-home1/liufengkai/opd
CONFIG=$REPO/configs/trustworthy_opd/lcb_reliability_calibration.yaml
SESSION=opd-lcb-reliability-calibration
RUN_DIR=
WORKER=false
ORCH_LOG=

usage() {
    echo "Usage: $0 [--session NAME] [--config PATH] [--run-dir PATH]"
}

while (($#)); do
    case "$1" in
        --session) SESSION=$2; shift 2 ;;
        --config) CONFIG=$2; shift 2 ;;
        --run-dir) RUN_DIR=$2; shift 2 ;;
        --worker) WORKER=true; shift ;;
        --orchestration-log) ORCH_LOG=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

SCRIPT=$(readlink -f "${BASH_SOURCE[0]}")
if [[ "$WORKER" != true ]]; then
    [[ -x "$TMUX_BIN" ]] || TMUX_BIN=$(command -v tmux || true)
    [[ -n "$TMUX_BIN" && -x "$TMUX_BIN" ]] || { echo "tmux not found" >&2; exit 1; }
    [[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
    if [[ -n "$RUN_DIR" && ! -f "$RUN_DIR/config.yaml" ]]; then
        echo "Resume run has no config.yaml: $RUN_DIR" >&2
        exit 2
    fi
    if "$TMUX_BIN" has-session -t "$SESSION" 2>/dev/null; then
        echo "tmux session already exists: $SESSION" >&2
        exit 1
    fi
    stamp=$(date -u +%Y%m%d_%H%M%S)
    log_dir=$STORAGE_ROOT/experiments/launch_logs
    mkdir -p "$log_dir"
    ORCH_LOG=$log_dir/${stamp}_${SESSION}.log
    worker=(
        "$SCRIPT" --worker --orchestration-log "$ORCH_LOG"
        --session "$SESSION" --config "$CONFIG"
    )
    [[ -z "$RUN_DIR" ]] || worker+=(--run-dir "$RUN_DIR")
    printf -v worker_command '%q ' "${worker[@]}"
    "$TMUX_BIN" new-session -d -s "$SESSION" \
        "bash -lc '$worker_command; rc=\$?; echo lcb_reliability_exit=\$rc; exec bash'"
    echo "Started tmux session: $SESSION"
    echo "Orchestration log: $ORCH_LOG"
    echo "Attach: $TMUX_BIN attach -t '$SESSION'"
    echo "Progress: tail -f '$ORCH_LOG'"
    exit 0
fi

[[ -n "$ORCH_LOG" ]] || { echo "--orchestration-log is required in worker mode" >&2; exit 2; }
ORCH_DIR=${ORCH_LOG%.log}_artifacts
mkdir -p "$ORCH_DIR"
exec > >(tee -a "$ORCH_LOG") 2>&1

set +u
source "$CONDA_SH"
conda activate opd
set -u
cd "$REPO"
export OPD_ROOT=$REPO
export OPD_STORAGE_ROOT=$STORAGE_ROOT
export OPD_MODEL_DIR=$STORAGE_ROOT/models

FROZEN_HEAD=$(git rev-parse HEAD)
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Repository must be clean before calibration" >&2
    git status --short >&2
    exit 1
fi
printf '%s\n' "$FROZEN_HEAD" > "$ORCH_DIR/frozen_git_revision.txt"
export OPD_FROZEN_GIT_REVISION=$FROZEN_HEAD

echo "Starting offline LCB reliability discovery at $(date -u --iso-8601=seconds)"
echo "Frozen revision: $FROZEN_HEAD"
command=(python -u scripts/trustworthy_opd/run_lcb_reliability_calibration.py)
if [[ -n "$RUN_DIR" ]]; then
    command+=(--run-dir "$RUN_DIR")
else
    command+=(--config "$CONFIG")
fi
set +e
"${command[@]}" 2>&1 | tee "$ORCH_DIR/pipeline.log"
exit_code=${PIPESTATUS[0]}
set -e
resolved_run=$(sed -n 's/^RUN_DIR=//p' "$ORCH_DIR/pipeline.log" | head -n 1)
if [[ -n "$resolved_run" ]]; then
    printf '%s\n' "$resolved_run" > "$ORCH_DIR/run_dir.txt"
fi
if ((exit_code != 0)); then
    echo "Pipeline failed with exit code $exit_code; resume with --run-dir '$resolved_run'" >&2
    exit "$exit_code"
fi
echo "Offline LCB reliability discovery completed at $(date -u --iso-8601=seconds)"
echo "RUN_DIR=$resolved_run"
