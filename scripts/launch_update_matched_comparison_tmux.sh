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
CALIBRATION_STEPS=5
MAX_STARTUP_RETRIES=6
RETRY_DELAY_SECONDS=60
SESSION=opd-update-matched-seed43
WORKER=false
ORCH_LOG=

usage() {
    echo "Usage: $0 [--mode probe|full] [--session NAME]"
    echo "          [--gpu-count auto|N] [--min-gpus N] [--max-gpus N]"
    echo "          [--min-free-mib N] [--max-util N] [--probe-steps N]"
    echo "          [--calibration-steps N] [--stable-samples N] [--poll-seconds N]"
    echo "          [--max-startup-retries N] [--retry-delay-seconds N]"
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
        --calibration-steps) CALIBRATION_STEPS=$2; shift 2 ;;
        --stable-samples) STABLE_SAMPLES=$2; shift 2 ;;
        --poll-seconds) POLL_SECONDS=$2; shift 2 ;;
        --max-startup-retries) MAX_STARTUP_RETRIES=$2; shift 2 ;;
        --retry-delay-seconds) RETRY_DELAY_SECONDS=$2; shift 2 ;;
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
    "$STABLE_SAMPLES" "$POLL_SECONDS" "$PROBE_STEPS" "$CALIBRATION_STEPS" \
    "$MAX_STARTUP_RETRIES" "$RETRY_DELAY_SECONDS"; do
    [[ "$value" =~ ^[0-9]+$ ]] || { echo "Numeric options must be integers" >&2; exit 2; }
done
[[ "$GPU_COUNT" == auto || "$GPU_COUNT" =~ ^[1-9][0-9]*$ ]] || {
    echo "--gpu-count must be auto or a positive integer" >&2
    exit 2
}
((MIN_GPUS >= 1 && MAX_GPUS >= MIN_GPUS && STABLE_SAMPLES >= 1 && POLL_SECONDS >= 1 \
    && PROBE_STEPS >= 1 && CALIBRATION_STEPS >= 3 && RETRY_DELAY_SECONDS >= 1)) || {
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
        --calibration-steps "$CALIBRATION_STEPS"
        --max-startup-retries "$MAX_STARTUP_RETRIES"
        --retry-delay-seconds "$RETRY_DELAY_SECONDS"
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
ORCH_DIR=${ORCH_LOG%.log}_artifacts
mkdir -p "$ORCH_DIR"
RUN_MANIFEST=$ORCH_DIR/runs.tsv
printf 'role\trun_dir\n' > "$RUN_MANIFEST"

set +u
source "$CONDA_SH"
conda activate opd
set -u
cd "$REPO"
export OPD_ROOT=$REPO
export OPD_STORAGE_ROOT=$STORAGE_ROOT
export OPD_MODEL_DIR=$MODEL_DIR

FROZEN_HEAD=$(git rev-parse HEAD)
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Repository must be clean before a paired multi-arm run" >&2
    git status --short >&2
    exit 1
fi
printf '%s\n' "$FROZEN_HEAD" > "$ORCH_DIR/frozen_git_revision.txt"

assert_frozen_revision() {
    local current_head
    current_head=$(git rev-parse HEAD)
    if [[ "$current_head" != "$FROZEN_HEAD" || -n "$(git status --porcelain)" ]]; then
        echo "Repository changed after orchestration started; refusing the next paired arm" >&2
        echo "frozen=$FROZEN_HEAD current=$current_head" >&2
        git status --short >&2
        exit 1
    fi
}

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
    local config=$1 label=$2 role=$3
    shift 3
    local extra=("$@")
    local probe_overrides=()
    local attempt=0 exit_code=0 attempt_log= run_dir=
    assert_frozen_revision
    if [[ "$MODE" == probe ]]; then
        probe_overrides=(
            --set "experiment.name=${label}_probe${PROBE_STEPS}"
            --set "trainer.total_training_steps=$PROBE_STEPS"
            --set "trainer.save_freq=$PROBE_STEPS"
            --set "trainer.max_actor_ckpt_to_keep=1"
        )
    fi
    while true; do
        attempt=$((attempt + 1))
        select_stable_gpus
        attempt_log=$ORCH_DIR/${role}_attempt${attempt}.log
        echo "Starting $label attempt=$attempt on GPUs $SELECTED_GPUS at $(date -u --iso-8601=seconds)"
        set +e
        python -u scripts/run_opd_experiment.py "$config" \
            --set "trainer.n_gpus_per_node=$RESOLVED_GPU_COUNT" \
            --set "runtime.cuda_visible_devices=$SELECTED_GPUS" \
            --set "runtime.min_free_gpu_memory_mb=$MIN_FREE_MIB" \
            "${probe_overrides[@]}" "${extra[@]}" 2>&1 | tee "$attempt_log"
        exit_code=${PIPESTATUS[0]}
        set -e
        run_dir=$(sed -n 's/^RUN_DIR=//p' "$attempt_log" | head -n 1)
        if ((exit_code == 0)); then
            [[ -n "$run_dir" ]] || { echo "Could not recover RUN_DIR for $label" >&2; exit 1; }
            LAST_RUN_DIR=$run_dir
            printf '%s\t%s\n' "$role" "$run_dir" >> "$RUN_MANIFEST"
            echo "Completed $label at $(date -u --iso-8601=seconds): $run_dir"
            return 0
        fi
        if ! grep -Eq \
            'No available memory for the cache blocks|below the configured [0-9]+ MiB safety threshold' \
            "$attempt_log"; then
            echo "$label failed with a non-retryable error; inspect $attempt_log" >&2
            return "$exit_code"
        fi
        if [[ -n "$run_dir" && -s "$run_dir/metrics/ropd_step_metrics.jsonl" ]]; then
            echo "$label reached training metrics; refusing an automatic from-scratch retry" >&2
            return "$exit_code"
        fi
        if [[ -n "$run_dir" ]] && find "$run_dir/checkpoints" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
            echo "$label wrote checkpoints; refusing an automatic from-scratch retry" >&2
            return "$exit_code"
        fi
        if ((attempt > MAX_STARTUP_RETRIES)); then
            echo "$label exhausted $MAX_STARTUP_RETRIES startup retries; inspect $attempt_log" >&2
            return "$exit_code"
        fi
        echo "Retryable pre-training GPU memory failure; reselecting after ${RETRY_DELAY_SECONDS}s."
        sleep "$RETRY_DELAY_SECONDS"
    done
}

echo "Running independent FP32 label-free calibration before the paired arms."
calibration_mode=$MODE
MODE=full
run_arm configs/experiments/opd_update_matched_scale_calibration.yaml \
    "scale_calibration_${CALIBRATION_STEPS}" calibration \
    --set "experiment.name=opd_update_matched_scale_calibration_${CALIBRATION_STEPS}" \
    --set "trainer.total_training_steps=$CALIBRATION_STEPS" \
    --set "trainer.save_freq=$CALIBRATION_STEPS" \
    --set "trainer.max_actor_ckpt_to_keep=1"
MODE=$calibration_mode

CALIBRATION_JSON=$ORCH_DIR/fixed_opd_scale.json
python -u scripts/calibrate_update_matched_scale.py \
    --metrics "$LAST_RUN_DIR/metrics/ropd_step_metrics.jsonl" \
    --minimum-steps "$CALIBRATION_STEPS" \
    --output "$CALIBRATION_JSON" | tee "$ORCH_DIR/calibration.log"
FIXED_OPD_SCALE=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["fixed_opd_scale"])' "$CALIBRATION_JSON")
echo "Frozen FP32 fixed_opd_scale=$FIXED_OPD_SCALE"

run_arm configs/experiments/opd_update_matched_seed43_opd.yaml opd opd
run_arm configs/experiments/opd_update_matched_seed43_scaled_opd.yaml scaled_opd scaled_opd \
    --set "distillation.robust_opd.fixed_opd_scale=$FIXED_OPD_SCALE"
run_arm configs/experiments/opd_update_matched_seed43_ropd.yaml ropd ropd
run_arm configs/experiments/opd_update_matched_seed43_normalized_ropd.yaml \
    normalized_ropd normalized_ropd

echo "All four update-matched arms completed at $(date -u --iso-8601=seconds)"
echo "Run manifest: $RUN_MANIFEST"
echo "Frozen calibration: $CALIBRATION_JSON"
