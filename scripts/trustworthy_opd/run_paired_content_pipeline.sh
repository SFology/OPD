#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$REPO_ROOT/configs/trustworthy_opd/paired_content_120.yaml"
RUN_DIR=""
MIN_FREE_MB=30000
MAX_UTIL=15
GPU_WAIT_SECONDS=86400
GPU_POLL_SECONDS=60
GPU_STABILITY_SECONDS=10
PYTHON_BIN="${PYTHON_BIN:-python}"

while (($#)); do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --min-free-mb) MIN_FREE_MB="$2"; shift 2 ;;
        --max-util) MAX_UTIL="$2"; shift 2 ;;
        --gpu-wait-seconds) GPU_WAIT_SECONDS="$2"; shift 2 ;;
        --gpu-poll-seconds) GPU_POLL_SECONDS="$2"; shift 2 ;;
        --gpu-stability-seconds) GPU_STABILITY_SECONDS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

cd "$REPO_ROOT"
export OPD_ROOT="${OPD_ROOT:-$REPO_ROOT}"
export OPD_STORAGE_ROOT="${OPD_STORAGE_ROOT:-/attached/remote-home1/${USER}/opd}"
export OPD_MODEL_DIR="${OPD_MODEL_DIR:-$OPD_STORAGE_ROOT/models}"
mkdir -p "$OPD_STORAGE_ROOT/trustworthy_opd_launch_logs"

select_gpu() {
    local selected confirmed deadline
    deadline=$((SECONDS + GPU_WAIT_SECONDS))
    while true; do
        selected="$({
            nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
                --format=csv,noheader,nounits |
            awk -F',' -v min_free="$MIN_FREE_MB" -v max_util="$MAX_UTIL" '
                {
                    for (i=1; i<=3; i++) gsub(/^[ \t]+|[ \t]+$/, "", $i)
                    if ($2 >= min_free && $3 <= max_util) print $1, $2, $3
                }
            ' |
            sort -k2,2nr -k3,3n | awk 'NR == 1 {print $1}'
        } || true)"
        if [[ -n "$selected" ]]; then
            sleep "$GPU_STABILITY_SECONDS"
            confirmed="$(
                nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
                    --format=csv,noheader,nounits |
                awk -F',' -v wanted="$selected" -v min_free="$MIN_FREE_MB" \
                    -v max_util="$MAX_UTIL" '
                    {
                        for (i=1; i<=3; i++) gsub(/^[ \t]+|[ \t]+$/, "", $i)
                        if ($1 == wanted && $2 >= min_free && $3 <= max_util) print $1
                    }
                '
            )"
            if [[ "$confirmed" == "$selected" ]]; then
                printf '%s\n' "$selected"
                return 0
            fi
        fi
        if ((SECONDS >= deadline)); then
            echo "No GPU meets the requested thresholds" >&2
            nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
                --format=csv >&2
            return 1
        fi
        echo "Waiting for GPU: >=${MIN_FREE_MB} MiB free, <=${MAX_UTIL}% utilization" >&2
        sleep "$GPU_POLL_SECONDS"
    done
}

if [[ -z "$RUN_DIR" ]]; then
    launch_log="$OPD_STORAGE_ROOT/trustworthy_opd_launch_logs/paired_$(date -u +%Y%m%d_%H%M%S).log"
    bash "$SCRIPT_DIR/run_pipeline.sh" \
        --config "$CONFIG" \
        --stop-after stability \
        --min-free-mb "$MIN_FREE_MB" \
        --max-util "$MAX_UTIL" \
        --gpu-wait-seconds "$GPU_WAIT_SECONDS" \
        --gpu-poll-seconds "$GPU_POLL_SECONDS" \
        --gpu-stability-seconds "$GPU_STABILITY_SECONDS" \
        2>&1 | tee "$launch_log"
    RUN_DIR="$(sed -n 's/^TRUST_OPD_RUN=//p' "$launch_log" | tail -n 1)"
else
    RUN_DIR="$(realpath "$RUN_DIR")"
    if [[ ! -f "$RUN_DIR/results/stability.parquet" ]]; then
        current_status="$(awk '/^status:/ {print $2; exit}' "$RUN_DIR/status.yaml")"
        if [[ "$current_status" == "collection_rejected" ]]; then
            CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -u \
                "$SCRIPT_DIR/collect_states.py" \
                --config "$CONFIG" \
                --run-dir "$RUN_DIR" \
                --rebuild-states-only \
                --valid-only \
                --min-valid-rollouts-per-prompt 3 \
                2>&1 | tee "$RUN_DIR/logs/01c_rebuild_valid_states.log"
        fi
        bash "$SCRIPT_DIR/run_pipeline.sh" \
            --run-dir "$RUN_DIR" \
            --stop-after stability \
            --min-free-mb "$MIN_FREE_MB" \
            --max-util "$MAX_UTIL" \
            --gpu-wait-seconds "$GPU_WAIT_SECONDS" \
            --gpu-poll-seconds "$GPU_POLL_SECONDS" \
            --gpu-stability-seconds "$GPU_STABILITY_SECONDS"
    fi
fi

[[ -n "$RUN_DIR" && -f "$RUN_DIR/results/stability.parquet" ]] || {
    echo "Paired pipeline did not produce a valid stability run" >&2
    exit 1
}
echo "PAIRED_RUN_DIR=$RUN_DIR"

for role in teacher student; do
    gpu_id="$(select_gpu)"
    echo "[paired-$role] physical GPU $gpu_id -> process cuda:0"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" -u \
        "$SCRIPT_DIR/validate_paired_content.py" \
        --run-dir "$RUN_DIR" --role "$role" \
        2>&1 | tee "$RUN_DIR/logs/paired_validate_${role}.log"
done

CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -u \
    "$SCRIPT_DIR/analyze_paired_content.py" --run-dir "$RUN_DIR" \
    2>&1 | tee "$RUN_DIR/logs/paired_content_analysis.log"

echo "Paired content pipeline complete"
echo "PAIRED_RUN_DIR=$RUN_DIR"
echo "Summary: $RUN_DIR/results/paired_content_summary.json"
