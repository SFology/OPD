#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$REPO_ROOT/configs/trustworthy_opd/ppl_branch_comparison.yaml"
RUN_DIR=""
SESSION="opd-ppl-branches-$(date -u +%Y%m%d-%H%M%S)"
ATTACH=false

while (($#)); do
    case "$1" in
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --session) SESSION="$2"; shift 2 ;;
        --attach) ATTACH=true; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$RUN_DIR" && -f "$RUN_DIR/config.yaml" ]] || {
    echo "A valid --run-dir is required" >&2
    exit 2
}
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 1; }

RUN_DIR="$(realpath "$RUN_DIR")"
CONFIG="$(realpath "$CONFIG")"
mkdir -p "$RUN_DIR/ppl_branch_comparison/logs"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG="$RUN_DIR/ppl_branch_comparison/logs/launch_${STAMP}.log"
printf -v RUN_Q '%q' "$RUN_DIR"
printf -v CONFIG_Q '%q' "$CONFIG"
printf -v REPO_Q '%q' "$REPO_ROOT"
printf -v LOG_Q '%q' "$LOG"

worker="set -o pipefail; \
source /remote-home/share/anaconda3/etc/profile.d/conda.sh && \
conda activate opd && cd $REPO_Q && \
export OPD_ROOT=$REPO_Q && \
export OPD_STORAGE_ROOT=/attached/remote-home1/${USER}/opd && \
export OPD_MODEL_DIR=/attached/remote-home1/${USER}/opd/models && \
python scripts/trustworthy_opd/freeze_four_group_rounds.py \
--run-dir $RUN_Q --completed-rounds 1 --verify-only && \
python -u scripts/trustworthy_opd/run_ppl_branch_comparison.py \
--run-dir $RUN_Q --config $CONFIG_Q \
2>&1 | tee $LOG_Q; \
rc=\${PIPESTATUS[0]}; echo ppl_branch_comparison_exit=\$rc; exec bash"

tmux new-session -d -s "$SESSION" "$worker"
echo "Started tmux session: $SESSION"
echo "Log: $LOG"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl+B, then D"
if [[ "$ATTACH" == true ]]; then
    tmux attach -t "$SESSION"
fi
