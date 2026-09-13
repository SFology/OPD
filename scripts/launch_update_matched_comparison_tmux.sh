#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/remote-home/liufengkai/projects/OPD
CONDA_SH=/remote-home/share/anaconda3/etc/profile.d/conda.sh
TMUX_BIN=/attached/remote-home1/liufengkai/tools/tmux-env/bin/tmux
STORAGE_ROOT=/attached/remote-home1/liufengkai/opd
MODEL_DIR=$STORAGE_ROOT/models

MODE=probe
GPU_COUNT=auto
MIN_GPUS=2
MAX_GPUS=4
MIN_FREE_MIB=43000
MAX_UTIL=10
STABLE_SAMPLES=3
POLL_SECONDS=30
PROBE_STEPS=5
SESSION=opd-update-matched-seed43
WORKER=false
ORCH_LOG=

usage() {
    echo "Usage: $0 [--mode probe|full] [--session NAME]"
    echo "          [--gpu-count auto|N] [--min-gpus N] [--max-gpus N]"
    echo "          [--min-free-mib N] [--max-util N] [--probe-steps N]"
    echo "          [--stable-samples N] [--poll-seconds N]"
}

while (($#)); do
    case "$1" in
        --mode) MODE=$2; shift 2 ;;
        --session) SESSION=$2; shift 2 ;;
        --gpu-count) GPU_COUNT=$2; shift 2 ;;
        --min-gpus) MIN_GPUS=$2; shift 2 ;;
        --max-gpus) MAX_GPUS=$2; shift 2 ;;
        --min-free-mib) MIN_FREE_MIB=$2; shift 2 ;;
        --max-util) MAX_UTIL=$2; shift 2 ;;
        --probe-steps) PROBE_STEPS=$2; shift 2 ;;
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
for value in "$MIN_GPUS" "$MAX_GPUS" "$MIN_FREE_MIB" "$MAX_UTIL" \
    "$STABLE_SAMPLES" "$POLL_SECONDS" "$PROBE_STEPS"; do
    [[ "$value" =~ ^[0-9]+$ ]] || { echo "Numeric options must be integers" >&2; exit 2; }
done
[[ "$GPU_COUNT" == auto || "$GPU_COUNT" =~ ^[1-9][0-9]*$ ]] || {
    echo "--gpu-count must be auto or a positive integer" >&2
    exit 2
}
((MIN_GPUS >= 1 && MAX_GPUS >= MIN_GPUS && STABLE_SAMPLES >= 1 && POLL_SECONDS >= 1 && PROBE_STEPS >= 1)) || {
    echo "Invalid GPU/probe bounds" >&2
    exit 2
}

SCRIPT=$(readlink -f "${BASH_SOURCE[0]}")
if [[ "$WORKER" != true ]]; then
    [[ -x "$TMUX_BIN" ]] || TMUX_BIN=$(command -v tmux || true)
    [[ -n "$TMUX_BIN" && -x "$TMUX_BIN" ]] || { echo "tmux not found" >&2; exit 1; }
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
        --min-gpus "$MIN_GPUS" --max-gpus "$MAX_GPUS"
        --min-free-mib "$MIN_FREE_MIB" --max-util "$MAX_UTIL"
        --stable-samples "$STABLE_SAMPLES" --poll-seconds "$POLL_SECONDS"
        --probe-steps "$PROBE_STEPS"
    )
    printf -v worker_command '%q ' "${worker[@]}"
    "$TMUX_BIN" new-session -d -s "$SESSION" "bash -lc '$worker_command; rc=\$?; echo update_matched_exit=\$rc; exec bash'"
    echo "Started tmux session: $SESSION"
    echo "Orchestration log: $ORCH_LOG"
    echo "Attach: $TMUX_BIN attach -t '$SESSION'"
    echo "Progress: tail -f '$ORCH_LOG'"
    exit 0
fi

[[ -n "$ORCH_LOG" ]] || { echo "--orchestration-log is required in worker mode" >&2; exit 2; }
mkdir -p "$(dirname "$ORCH_LOG")"
exec > >(tee -a "$ORCH_LOG") 2>&1

set +u
source "$CONDA_SH"
conda activate opd
set -u
cd "$REPO"
export OPD_ROOT=$REPO
export OPD_STORAGE_ROOT=$STORAGE_ROOT
export OPD_MODEL_DIR=$MODEL_DIR

if pgrep -u "$(id -u)" -af 'verl[.]trainer[.]main_ppo' >/dev/null; then
    echo "Another OPD/verl training process is already running for this user" >&2
    pgrep -u "$(id -u)" -af 'verl[.]trainer[.]main_ppo' >&2 || true
    exit 1
fi

eligible_gpu_records() {
    nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | \
        awk -F',' -v free="$MIN_FREE_MIB" -v util="$MAX_UTIL" '
            {
                for (i=1; i<=NF; i++) gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
                if (($2 + 0) >= free && ($3 + 0) <= util) print $1, $2, $3
            }
        ' | sort -k2,2nr
}

if [[ "$GPU_COUNT" == auto ]]; then
    while true; do
        available=$(eligible_gpu_records | awk 'NF {count++} END {print count+0}')
        if ((available >= MIN_GPUS)); then
            RESOLVED_GPU_COUNT=$available
            ((RESOLVED_GPU_COUNT > MAX_GPUS)) && RESOLVED_GPU_COUNT=$MAX_GPUS
            break
        fi
        echo "$(date -u --iso-8601=seconds) waiting for at least $MIN_GPUS idle GPUs; found $available"
        sleep "$POLL_SECONDS"
    done
else
    RESOLVED_GPU_COUNT=$GPU_COUNT
fi
echo "Fixed GPU count for all four paired arms: $RESOLVED_GPU_COUNT"

select_stable_gpus() {
    local stable=0 previous= current= count= records=
    while true; do
        records=$(eligible_gpu_records | head -n "$RESOLVED_GPU_COUNT")
        current=$(awk '{print $1}' <<<"$records" | paste -sd, -)
        count=$(awk 'NF {count++} END {print count+0}' <<<"$records")
        echo "$(date -u --iso-8601=seconds) eligible=$count/$RESOLVED_GPU_COUNT candidates=${current:-none}"
        if ((count == RESOLVED_GPU_COUNT)); then
            if [[ "$current" == "$previous" ]]; then
                stable=$((stable + 1))
            else
                stable=1
                previous=$current
            fi
            if ((stable >= STABLE_SAMPLES)); then
                SELECTED_GPUS=$current
                return 0
            fi
        else
            stable=0
            previous=
        fi
        sleep "$POLL_SECONDS"
    done
}

run_arm() {
    local config=$1 label=$2
    local extra=()
    select_stable_gpus
    if [[ "$MODE" == probe ]]; then
        extra=(
            --set "experiment.name=${label}_probe${PROBE_STEPS}"
            --set "trainer.total_training_steps=$PROBE_STEPS"
            --set "trainer.save_freq=$PROBE_STEPS"
            --set "trainer.max_actor_ckpt_to_keep=1"
        )
    fi
    echo "Starting $label on GPUs $SELECTED_GPUS at $(date -u --iso-8601=seconds)"
    python -u scripts/run_opd_experiment.py "$config" \
        --set "trainer.n_gpus_per_node=$RESOLVED_GPU_COUNT" \
        --set "runtime.cuda_visible_devices=$SELECTED_GPUS" \
        --set "runtime.min_free_gpu_memory_mb=$MIN_FREE_MIB" \
        "${extra[@]}"
    echo "Completed $label at $(date -u --iso-8601=seconds)"
}

configs=(
    configs/experiments/opd_update_matched_seed43_opd.yaml
    configs/experiments/opd_update_matched_seed43_scaled_opd.yaml
    configs/experiments/opd_update_matched_seed43_ropd.yaml
    configs/experiments/opd_update_matched_seed43_normalized_ropd.yaml
)
labels=(opd scaled_opd ropd normalized_ropd)
for index in "${!configs[@]}"; do
    run_arm "${configs[$index]}" "${labels[$index]}"
done

echo "All four update-matched arms completed at $(date -u --iso-8601=seconds)"
