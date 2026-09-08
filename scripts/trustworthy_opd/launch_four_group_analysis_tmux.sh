#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_DIR=""
COMPLETED_ROUNDS=1
SESSION="opd-four-group-analysis-$(date -u +%Y%m%d-%H%M%S)"
ATTACH=false

while (($#)); do
    case "$1" in
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --completed-rounds) COMPLETED_ROUNDS="$2"; shift 2 ;;
        --session) SESSION="$2"; shift 2 ;;
        --attach) ATTACH=true; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

command -v tmux >/dev/null || {
    echo "tmux is required but was not found" >&2
    exit 1
}
[[ -n "$RUN_DIR" && -f "$RUN_DIR/config.yaml" ]] || {
    echo "A valid --run-dir is required" >&2
    exit 2
}
[[ -f "$RUN_DIR/artifacts/frozen_rounds.json" ]] || {
    echo "Freeze the collection before starting analysis: $RUN_DIR" >&2
    exit 2
}

RUN_DIR="$(realpath "$RUN_DIR")"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG="$RUN_DIR/logs/analysis_only_${STAMP}.log"
printf -v RUN_DIR_Q '%q' "$RUN_DIR"
printf -v REPO_ROOT_Q '%q' "$REPO_ROOT"
printf -v LOG_Q '%q' "$LOG"

worker="set -o pipefail; \
source /remote-home/share/anaconda3/etc/profile.d/conda.sh && \
conda activate opd && cd $REPO_ROOT_Q && \
export OPD_ROOT=$REPO_ROOT_Q && \
export OPD_STORAGE_ROOT=/attached/remote-home1/${USER}/opd && \
export OPD_MODEL_DIR=/attached/remote-home1/${USER}/opd/models && \
python scripts/trustworthy_opd/freeze_four_group_rounds.py \
--run-dir $RUN_DIR_Q --completed-rounds $COMPLETED_ROUNDS --verify-only && \
python -u scripts/trustworthy_opd/run_four_group_pipeline.py \
--run-dir $RUN_DIR_Q --analysis-only --completed-rounds $COMPLETED_ROUNDS \
2>&1 | tee $LOG_Q; \
rc=\${PIPESTATUS[0]}; echo four_group_analysis_exit=\$rc; exec bash"

tmux new-session -d -s "$SESSION" "$worker"
echo "Started tmux session: $SESSION"
echo "Log: $LOG"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl+B, then D"
if [[ "$ATTACH" == true ]]; then
    tmux attach -t "$SESSION"
fi
