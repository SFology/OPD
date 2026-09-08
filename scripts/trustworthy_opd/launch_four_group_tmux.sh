#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$REPO_ROOT/configs/trustworthy_opd/four_group_ppl_stability.yaml"
RUN_DIR=""
SESSION="opd-four-group-$(date -u +%Y%m%d-%H%M%S)"
ATTACH=false

while (($#)); do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --session) SESSION="$2"; shift 2 ;;
        --attach) ATTACH=true; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

command -v tmux >/dev/null || {
    echo "tmux is required but was not found" >&2
    exit 1
}
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }

run_dir_arg=""
if [[ -n "$RUN_DIR" ]]; then
    [[ -f "$RUN_DIR/config.yaml" ]] || {
        echo "Invalid run directory: $RUN_DIR" >&2
        exit 2
    }
    printf -v run_dir_arg ' --run-dir %q' "$(realpath "$RUN_DIR")"
fi

worker="set -o pipefail; \
source /remote-home/share/anaconda3/etc/profile.d/conda.sh && \
conda activate opd && cd $REPO_ROOT && \
export OPD_ROOT=$REPO_ROOT && \
export OPD_STORAGE_ROOT=/attached/remote-home1/${USER}/opd && \
export OPD_MODEL_DIR=/attached/remote-home1/${USER}/opd/models && \
python -u scripts/trustworthy_opd/run_four_group_pipeline.py \
--config $CONFIG${run_dir_arg} \
2>&1 | tee /attached/remote-home1/${USER}/opd/four_group_launch_$(date -u +%Y%m%d_%H%M%S).log; \
rc=\${PIPESTATUS[0]}; echo four_group_exit=\$rc; exec bash"

tmux new-session -d -s "$SESSION" "$worker"
echo "Started tmux session: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl+B, then D"
if [[ "$ATTACH" == true ]]; then
    tmux attach -t "$SESSION"
fi
