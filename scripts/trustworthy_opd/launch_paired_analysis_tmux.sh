#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_DIR=""
MINIMUM_FRACTION="0.70"
SESSION="opd-paired-analysis-$(date -u +%Y%m%d-%H%M%S)"
ATTACH=false

while (($#)); do
    case "$1" in
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --minimum-complete-state-fraction) MINIMUM_FRACTION="$2"; shift 2 ;;
        --session) SESSION="$2"; shift 2 ;;
        --attach) ATTACH=true; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$RUN_DIR" && -f "$RUN_DIR/config.yaml" ]] || {
    echo "A valid --run-dir is required" >&2
    exit 2
}
RUN_DIR="$(realpath "$RUN_DIR")"
command -v tmux >/dev/null || {
    echo "tmux is required but was not found" >&2
    exit 1
}

LOG_PATH="$RUN_DIR/logs/paired_content_analysis_exploratory.log"
worker="set -o pipefail; \
source /remote-home/share/anaconda3/etc/profile.d/conda.sh && \
conda activate opd && cd $REPO_ROOT && \
CUDA_VISIBLE_DEVICES='' python -u scripts/trustworthy_opd/analyze_paired_content.py \
--run-dir $RUN_DIR \
--minimum-complete-state-fraction $MINIMUM_FRACTION \
2>&1 | tee $LOG_PATH; \
rc=\${PIPESTATUS[0]}; echo paired_analysis_exit=\$rc; exec bash"

tmux new-session -d -s "$SESSION" "$worker"
echo "Started tmux session: $SESSION"
echo "Log: $LOG_PATH"
echo "Attach: tmux attach -t $SESSION"
if [[ "$ATTACH" == true ]]; then
    tmux attach -t "$SESSION"
fi
