#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$REPO_ROOT/configs/trustworthy_opd/teacher_degradation_diagnostic.yaml"
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
mkdir -p "$OPD_STORAGE_ROOT/teacher_degradation_diagnostic_launch_logs"

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

gpu_id="$(select_gpu)"
stamp="$(date -u +%Y%m%d_%H%M%S)"
log="$OPD_STORAGE_ROOT/teacher_degradation_diagnostic_launch_logs/${stamp}.log"
echo "physical GPU $gpu_id -> process cuda:0"
if [[ -n "$RUN_DIR" ]]; then
    command=("$PYTHON_BIN" -u "$SCRIPT_DIR/diagnose_teacher_degradation.py" --run-dir "$RUN_DIR")
else
    command=("$PYTHON_BIN" -u "$SCRIPT_DIR/diagnose_teacher_degradation.py" --config "$CONFIG")
fi
CUDA_VISIBLE_DEVICES="$gpu_id" "${command[@]}" 2>&1 | tee "$log"
