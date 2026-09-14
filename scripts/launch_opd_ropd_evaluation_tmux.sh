#!/usr/bin/env bash
set -Ee -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="$REPO_ROOT/configs/experiments/opd_ropd_seed42_evaluation.yaml"
RUN_DIR=""
SESSION=""
ATTACH=false
TMUX_BIN="/attached/remote-home1/${USER}/tools/tmux-env/bin/tmux"

while (($#)); do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --session) SESSION="$2"; shift 2 ;;
        --attach) ATTACH=true; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -x "$TMUX_BIN" ]] || TMUX_BIN="$(command -v tmux || true)"
[[ -n "$TMUX_BIN" && -x "$TMUX_BIN" ]] || {
    echo "tmux is required but was not found" >&2
    exit 1
}
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }

source /remote-home/share/anaconda3/etc/profile.d/conda.sh
conda activate opd
set -u
cd "$REPO_ROOT"
export OPD_ROOT="$REPO_ROOT"
export OPD_STORAGE_ROOT="/attached/remote-home1/${USER}/opd"
export OPD_MODEL_DIR="$OPD_STORAGE_ROOT/models"

if [[ -z "$RUN_DIR" ]]; then
    CONFIG_STEM="$(basename "${CONFIG%.yaml}")"
    RUN_ID="$(date -u +%Y%m%d_%H%M%S)_${CONFIG_STEM}"
    RUN_DIR="$OPD_STORAGE_ROOT/evaluations/$RUN_ID"
fi
if [[ -z "$SESSION" ]]; then
    SESSION="opd-eval-$(basename "$RUN_DIR" | cut -c1-32)"
fi

python -u scripts/val/run_formal_evaluation.py \
    --config "$CONFIG" \
    --run-dir "$RUN_DIR" \
    --prepare-only

COMMAND_FILE="$RUN_DIR/command.sh"
cat >"$COMMAND_FILE" <<EOF
#!/usr/bin/env bash
set -Ee -o pipefail
source /remote-home/share/anaconda3/etc/profile.d/conda.sh
conda activate opd
set -u
cd '$REPO_ROOT'
export OPD_ROOT='$REPO_ROOT'
export OPD_STORAGE_ROOT='$OPD_STORAGE_ROOT'
export OPD_MODEL_DIR='$OPD_MODEL_DIR'
python -u scripts/val/run_formal_evaluation.py --config '$CONFIG' --run-dir '$RUN_DIR'
EOF
chmod +x "$COMMAND_FILE"

LOG_FILE="$RUN_DIR/logs/launcher.log"
WORKER="set -o pipefail; '$COMMAND_FILE' 2>&1 | tee -a '$LOG_FILE'; rc=\${PIPESTATUS[0]}; echo formal_evaluation_exit=\$rc; exec bash"
"$TMUX_BIN" new-session -d -s "$SESSION" "$WORKER"

echo "Started formal evaluation"
echo "Run directory: $RUN_DIR"
echo "tmux session: $SESSION"
echo "Attach: $TMUX_BIN attach -t '$SESSION'"
echo "Progress: python scripts/val/run_formal_evaluation.py --run-dir '$RUN_DIR' --status"
echo "Log: tail -f '$LOG_FILE'"
echo "Detach: Ctrl+B, then D"

if [[ "$ATTACH" == true ]]; then
    "$TMUX_BIN" attach -t "$SESSION"
fi
