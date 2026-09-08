#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SESSION="opd-teacher-degradation-$(date -u +%Y%m%d-%H%M%S)"
ATTACH=false

while (($#)); do
    case "$1" in
        --attach) ATTACH=true; shift ;;
        --session) SESSION="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

command -v tmux >/dev/null || {
    echo "tmux is required but was not found" >&2
    exit 1
}
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session already exists: $SESSION" >&2
    exit 1
fi

worker="source /remote-home/share/anaconda3/etc/profile.d/conda.sh && \
conda activate opd && \
cd $REPO_ROOT && \
export OPD_STORAGE_ROOT=/attached/remote-home1/${USER}/opd && \
export OPD_MODEL_DIR=/attached/remote-home1/${USER}/opd/models && \
bash scripts/trustworthy_opd/run_teacher_degradation_diagnostic.sh \
  --gpu-wait-seconds 86400 --gpu-poll-seconds 60 \
  --gpu-stability-seconds 10 --min-free-mb 30000 --max-util 15; \
rc=\$?; echo diagnostic_exit=\$rc; exec bash"

tmux new-session -d -s "$SESSION" "$worker"
echo "Started tmux session: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl+B, then D"

if [[ "$ATTACH" == true ]]; then
    tmux attach -t "$SESSION"
fi
