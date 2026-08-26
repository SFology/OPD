#!/usr/bin/env bash
set -euo pipefail

REPO=/remote-home/liufengkai/projects/OPD
CONDA_SH=/remote-home/share/anaconda3/etc/profile.d/conda.sh
TMUX_BIN=/attached/remote-home1/liufengkai/tools/tmux-env/bin/tmux
STORAGE_ROOT=/attached/remote-home1/liufengkai/opd
MODEL_DIR=$STORAGE_ROOT/models

MODE=probe
GPU_COUNT=4
MIN_FREE_MIB=43000
MAX_UTIL=10
STABLE_SAMPLES=3
POLL_SECONDS=30
SESSION=opd-dense-discrete-comparison
WORKER=false
ORCH_LOG=

usage() {
    echo "Usage: $0 [--mode probe|full] [--session NAME]"
    echo "          [--gpu-count N] [--min-free-mib N] [--max-util N]"
    echo "          [--stable-samples N] [--poll-seconds N]"
}

while (($#)); do
    case "$1" in
        --mode) MODE=$2; shift 2 ;;
        --session) SESSION=$2; shift 2 ;;
        --gpu-count) GPU_COUNT=$2; shift 2 ;;
        --min-free-mib) MIN_FREE_MIB=$2; shift 2 ;;
        --max-util) MAX_UTIL=$2; shift 2 ;;
        --stable-samples) STABLE_SAMPLES=$2; shift 2 ;;
        --poll-seconds) POLL_SECONDS=$2; shift 2 ;;
        --worker) WORKER=true; shift ;;
        --orchestration-log) ORCH_LOG=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$MODE" in
    probe|full) ;;
    *) echo "Invalid --mode: $MODE" >&2; exit 2 ;;
esac
for value in "$GPU_COUNT" "$MIN_FREE_MIB" "$MAX_UTIL" "$STABLE_SAMPLES" "$POLL_SECONDS"; do
    [[ "$value" =~ ^[0-9]+$ ]] || { echo "Numeric options must be non-negative integers" >&2; exit 2; }
done
((GPU_COUNT >= 1 && STABLE_SAMPLES >= 1 && POLL_SECONDS >= 1)) || {
    echo "gpu-count, stable-samples, and poll-seconds must be positive" >&2
    exit 2
}

SCRIPT=$(readlink -f "${BASH_SOURCE[0]}")

if [[ "$WORKER" != true ]]; then
    [[ -x "$TMUX_BIN" ]] || { echo "tmux not found: $TMUX_BIN" >&2; exit 1; }
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
        --mode "$MODE" --session "$SESSION" --gpu-count "$GPU_COUNT"
        --min-free-mib "$MIN_FREE_MIB" --max-util "$MAX_UTIL"
        --stable-samples "$STABLE_SAMPLES" --poll-seconds "$POLL_SECONDS"
    )
    printf -v worker_command '%q ' "${worker[@]}"
    "$TMUX_BIN" new-session -d -s "$SESSION" "bash -lc '$worker_command'"
    echo "Started tmux session: $SESSION"
    echo "Orchestration log: $ORCH_LOG"
    echo "Attach: $TMUX_BIN attach -t $SESSION"
    echo "Progress: tail -f '$ORCH_LOG'"
    exit 0
fi

[[ -n "$ORCH_LOG" ]] || { echo "--orchestration-log is required in worker mode" >&2; exit 2; }
mkdir -p "$(dirname "$ORCH_LOG")"
exec > >(tee -a "$ORCH_LOG") 2>&1

source "$CONDA_SH"
conda activate opd
cd "$REPO"
export OPD_ROOT=$REPO
export OPD_STORAGE_ROOT=$STORAGE_ROOT
export OPD_MODEL_DIR=$MODEL_DIR

echo "Started at $(date -u --iso-8601=seconds)"
echo "Mode=$MODE GPU_COUNT=$GPU_COUNT MIN_FREE_MIB=$MIN_FREE_MIB MAX_UTIL=$MAX_UTIL"

if pgrep -u "$(id -u)" -af 'verl[.]trainer[.]main_ppo' >/dev/null; then
    echo "Refusing to start because another OPD/verl training process is running for this user:" >&2
    pgrep -u "$(id -u)" -af 'verl[.]trainer[.]main_ppo' >&2 || true
    exit 1
fi

select_stable_gpus() {
    local stable=0 previous= current= records= count=
    while true; do
        records=$(nvidia-smi \
            --query-gpu=index,memory.free,utilization.gpu \
            --format=csv,noheader,nounits | \
            awk -F',' -v free="$MIN_FREE_MIB" -v util="$MAX_UTIL" '
                {
                    for (i=1; i<=NF; i++) gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
                    if (($2 + 0) >= free && ($3 + 0) <= util) print $1, $2, $3
                }
            ' | sort -k2,2nr | head -n "$GPU_COUNT")
        current=$(awk '{print $1}' <<<"$records" | paste -sd, -)
        count=$(awk 'NF {count++} END {print count+0}' <<<"$records")
        echo "$(date -u --iso-8601=seconds) eligible=$count/$GPU_COUNT candidates=${current:-none}"
        if ((count == GPU_COUNT)); then
            if [[ "$current" == "$previous" ]]; then
                stable=$((stable + 1))
            else
                stable=1
                previous=$current
            fi
            if ((stable >= STABLE_SAMPLES)); then
                SELECTED_GPUS=$current
                echo "Selected stable GPUs: $SELECTED_GPUS"
                return 0
            fi
        else
            stable=0
            previous=
        fi
        sleep "$POLL_SECONDS"
    done
}

run_managed() {
    local config=$1 label=$2
    select_stable_gpus
    echo "Starting $label with GPUs $SELECTED_GPUS"
    python -u scripts/run_opd_experiment.py "$config" \
        --set "trainer.n_gpus_per_node=$GPU_COUNT" \
        --set "runtime.cuda_visible_devices=$SELECTED_GPUS" \
        --set "runtime.min_free_gpu_memory_mb=$MIN_FREE_MIB"
    echo "Completed $label at $(date -u --iso-8601=seconds)"
}

if [[ "$MODE" == probe ]]; then
    run_managed configs/experiments/opd_dense_discrete_opd_probe.yaml "OPD control probe"
    run_managed configs/experiments/opd_dense_discrete_ropd_probe.yaml "ROPD treatment probe"
else
    run_managed configs/experiments/opd_dense_discrete_opd.yaml "OPD control full run"
    run_managed configs/experiments/opd_dense_discrete_ropd.yaml "ROPD treatment full run"
fi
