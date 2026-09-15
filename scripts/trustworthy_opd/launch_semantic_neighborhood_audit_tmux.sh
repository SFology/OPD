#!/usr/bin/env bash
set -euo pipefail

SESSION="opd-semantic-neighborhood-audit"
RUN_DIR=""
PORT=8094

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session)
      SESSION="$2"
      shift 2
      ;;
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$RUN_DIR" ]]; then
  echo "--run-dir is required" >&2
  exit 2
fi

REPO_DIR="/remote-home/liufengkai/projects/OPD"
PYTHON="/remote-home/liufengkai/.conda/envs/opd/bin/python"
TMUX="/attached/remote-home1/liufengkai/tools/tmux-env/bin/tmux"
RUN_DIR="$(realpath "$RUN_DIR")"

if [[ ! -f "$RUN_DIR/artifacts/blinded_pairs.jsonl" ]]; then
  echo "Missing prepared blinded pairs: $RUN_DIR" >&2
  exit 1
fi
if "$TMUX" has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

mkdir -p "$RUN_DIR/logs"
COMMAND="cd '$REPO_DIR' && exec '$PYTHON' -u scripts/trustworthy_opd/serve_semantic_neighborhood_audit.py --run-dir '$RUN_DIR' --host 127.0.0.1 --port '$PORT' 2>&1 | tee -a '$RUN_DIR/logs/annotation_server.log'"
"$TMUX" new-session -d -s "$SESSION" "$COMMAND"

echo "SESSION=$SESSION"
echo "RUN_DIR=$RUN_DIR"
echo "URL=http://127.0.0.1:$PORT/?annotator=liufengkai"
echo "ATTACH=$TMUX attach -t $SESSION"
