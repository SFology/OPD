#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$REPO_ROOT/configs/trustworthy_opd/pilot.yaml"
RUN_DIR="${TRUST_OPD_RUN:-}"
FROM_STAGE=""
MIN_FREE_MB=30000
MAX_UTIL=15
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
    cat <<'EOF'
Usage: run_pipeline.sh [options]

Options:
  --config PATH       Experiment config (default: configs/trustworthy_opd/pilot.yaml)
  --run-dir PATH      Resume an existing run directory
  --from-stage NAME   Start at collect, check, student-features, teacher-features,
                      stability, student-validation, teacher-validation, or analysis
  --min-free-mb N     Minimum free GPU memory in MiB (default: 30000)
  --max-util N        Maximum GPU utilization percent (default: 15)
  -h, --help          Show this help
EOF
}

while (($#)); do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --from-stage) FROM_STAGE="$2"; shift 2 ;;
        --min-free-mb) MIN_FREE_MB="$2"; shift 2 ;;
        --max-util) MAX_UTIL="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

cd "$REPO_ROOT"
export OPD_ROOT="${OPD_ROOT:-$REPO_ROOT}"
export OPD_STORAGE_ROOT="${OPD_STORAGE_ROOT:-/attached/remote-home1/${USER}/opd}"
export OPD_MODEL_DIR="${OPD_MODEL_DIR:-$OPD_STORAGE_ROOT/models}"
mkdir -p "$OPD_STORAGE_ROOT/trustworthy_opd_launch_logs"

select_gpu() {
    local selected
    selected="$({
        nvidia-smi \
            --query-gpu=index,memory.free,utilization.gpu \
            --format=csv,noheader,nounits |
        awk -F',' -v min_free="$MIN_FREE_MB" -v max_util="$MAX_UTIL" '
            {
                for (i=1; i<=3; i++) gsub(/^[ \t]+|[ \t]+$/, "", $i)
                if ($2 >= min_free && $3 <= max_util) print $1, $2, $3
            }
        ' |
        sort -k2,2nr -k3,3n |
        awk 'NR == 1 {print $1}'
    } || true)"
    if [[ -z "$selected" ]]; then
        echo "No GPU meets free-memory/utilization thresholds" >&2
        nvidia-smi \
            --query-gpu=index,memory.used,memory.free,utilization.gpu \
            --format=csv >&2
        return 1
    fi
    printf '%s\n' "$selected"
}

run_gpu_stage() {
    local label="$1"
    shift
    local gpu_id
    gpu_id="$(select_gpu)"
    echo "[$label] physical GPU $gpu_id -> process cuda:0"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$@" 2>&1 |
        tee "$RUN_DIR/logs/${label}.log"
}

run_cpu_stage() {
    local label="$1"
    shift
    echo "[$label] CPU-only"
    CUDA_VISIBLE_DEVICES="" "$@" 2>&1 |
        tee "$RUN_DIR/logs/${label}.log"
}

stage_number() {
    case "$1" in
        collect) echo 0 ;;
        check) echo 1 ;;
        student-features) echo 2 ;;
        teacher-features) echo 3 ;;
        stability) echo 4 ;;
        student-validation) echo 5 ;;
        teacher-validation) echo 6 ;;
        analysis) echo 7 ;;
        *) echo "Unknown stage: $1" >&2; exit 2 ;;
    esac
}

infer_resume_stage() {
    local status
    status="$(awk '/^status:/ {print $2; exit}' "$RUN_DIR/status.yaml")"
    case "$status" in
        collecting) echo collect ;;
        collected) echo check ;;
        collection_rejected)
            echo "Rejected collections must be restarted as a new run" >&2
            exit 2
            ;;
        collection_checked) echo student-features ;;
        extracting_student) echo student-features ;;
        extracted_student) echo teacher-features ;;
        extracting_teacher) echo teacher-features ;;
        extracted_teacher) echo stability ;;
        computing_stability) echo stability ;;
        stability_computed) echo student-validation ;;
        validating_student) echo student-validation ;;
        validated_student_partial) echo teacher-validation ;;
        validating_teacher) echo teacher-validation ;;
        validated) echo analysis ;;
        analyzed) echo done ;;
        failed)
            case "$(awk '/^stage:/ {print $2; exit}' "$RUN_DIR/status.yaml")" in
                collect_states) echo collect ;;
                extract_student) echo student-features ;;
                extract_teacher) echo teacher-features ;;
                validate_student) echo student-validation ;;
                validate_teacher) echo teacher-validation ;;
                *)
                    echo "Cannot infer failed stage; use --from-stage" >&2
                    exit 2
                    ;;
            esac
            ;;
        *)
            echo "Cannot infer resume stage from status '$status'; use --from-stage" >&2
            exit 2
            ;;
    esac
}

if [[ -n "$RUN_DIR" ]]; then
    RUN_DIR="$(realpath "$RUN_DIR")"
    [[ -f "$RUN_DIR/config.yaml" ]] || {
        echo "Invalid run directory: $RUN_DIR" >&2
        exit 2
    }
fi

if [[ -n "$FROM_STAGE" ]]; then
    START_STAGE="$FROM_STAGE"
elif [[ -n "$RUN_DIR" ]]; then
    START_STAGE="$(infer_resume_stage)"
    if [[ "$START_STAGE" == done ]]; then
        echo "Run is already complete: $RUN_DIR"
        exit 0
    fi
else
    START_STAGE=collect
fi
START_NUMBER="$(stage_number "$START_STAGE")"

if ((START_NUMBER <= 0)); then
    gpu_id="$(select_gpu)"
    launch_log="$OPD_STORAGE_ROOT/trustworthy_opd_launch_logs/collect_$(date -u +%Y%m%d_%H%M%S).log"
    echo "[01_collect_states] physical GPU $gpu_id -> process cuda:0"
    if [[ -n "$RUN_DIR" ]]; then
        CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" -u \
            "$SCRIPT_DIR/collect_states.py" --config "$CONFIG" --run-dir "$RUN_DIR" \
            2>&1 | tee "$launch_log"
    else
        CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" -u \
            "$SCRIPT_DIR/collect_states.py" --config "$CONFIG" \
            2>&1 | tee "$launch_log"
        RUN_DIR="$(sed -n 's/^RUN_DIR=//p' "$launch_log" | tail -n 1)"
    fi
    [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]] || {
        echo "Collector did not produce a valid RUN_DIR" >&2
        exit 1
    }
    export TRUST_OPD_RUN="$RUN_DIR"
    cp "$launch_log" "$RUN_DIR/logs/01_collect_states.log"
fi

export TRUST_OPD_RUN="$RUN_DIR"

if ((START_NUMBER <= 1)); then
    run_cpu_stage 01b_check_collection "$PYTHON_BIN" -u \
        "$SCRIPT_DIR/check_collection.py" --run-dir "$RUN_DIR"
fi
if ((START_NUMBER <= 2)); then
    run_gpu_stage 02_extract_student "$PYTHON_BIN" -u \
        "$SCRIPT_DIR/extract_features.py" --run-dir "$RUN_DIR" --role student
fi
if ((START_NUMBER <= 3)); then
    run_gpu_stage 03_extract_teacher "$PYTHON_BIN" -u \
        "$SCRIPT_DIR/extract_features.py" --run-dir "$RUN_DIR" --role teacher
fi
if ((START_NUMBER <= 4)); then
    run_cpu_stage 04_compute_stability "$PYTHON_BIN" -u \
        "$SCRIPT_DIR/compute_stability.py" --run-dir "$RUN_DIR"
fi
if ((START_NUMBER <= 5)); then
    run_gpu_stage 05_validate_student "$PYTHON_BIN" -u \
        "$SCRIPT_DIR/validate_reliability.py" --run-dir "$RUN_DIR" --role student
fi
if ((START_NUMBER <= 6)); then
    run_gpu_stage 06_validate_teacher "$PYTHON_BIN" -u \
        "$SCRIPT_DIR/validate_reliability.py" --run-dir "$RUN_DIR" --role teacher
fi
if ((START_NUMBER <= 7)); then
    run_cpu_stage 07_analyze_results "$PYTHON_BIN" -u \
        "$SCRIPT_DIR/analyze_results.py" --run-dir "$RUN_DIR"
fi

echo
echo "Pipeline complete"
echo "TRUST_OPD_RUN=$RUN_DIR"
echo "Summary: $RUN_DIR/results/summary.json"
