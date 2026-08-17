#!/usr/bin/env bash
# Start the full FSTTM voice → HVAC e2e on the Jetson in a tmux session.
#   pane 0: hvac-react backend  (FastAPI :8000)
#   pane 1: hvac-react frontend (Vite :5173, bound 0.0.0.0)
#   pane 2: fsttm voice server  (Jabra mic → Phi-3 intent → backend)
# Usage:  contrib/hvac/scripts/start-e2e.sh   then:  tmux attach -t fsttm
set -e
# repo root (this script lives in contrib/hvac/scripts/)
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SESS=fsttm

# Ensure the Jabra capture level is usable (33% default is too quiet for VAD)
JABRA_SRC=$(pactl list short sources 2>/dev/null | awk "/Jabra.*mono-fallback/ {print $2}" | head -1)
if [ -n "$JABRA_SRC" ]; then pactl set-source-mute "$JABRA_SRC" 0; pactl set-source-volume "$JABRA_SRC" 90%; echo "Jabra capture: $JABRA_SRC @90%"; fi

tmux kill-session -t "$SESS" 2>/dev/null || true
tmux new-session -d -s "$SESS" -n e2e

# pane 0 — backend
tmux send-keys -t "$SESS":e2e.0 \
  "cd $ROOT/contrib/hvac/hvac-react/backend && . .venv/bin/activate && uvicorn server:app --host 127.0.0.1 --port 8000" C-m

# pane 1 — frontend (split below)
tmux split-window -v -t "$SESS":e2e
tmux send-keys -t "$SESS":e2e.1 \
  "cd $ROOT/contrib/hvac/hvac-react/frontend && npm run dev -- --host 0.0.0.0" C-m

# pane 2 — fsttm voice (split right of backend)
tmux split-window -h -t "$SESS":e2e.0
sleep 3   # let backend bind before bridge connects
tmux send-keys -t "$SESS":e2e.2 \
  "$ROOT/scripts/run-voice.sh" C-m

tmux select-layout -t "$SESS":e2e tiled
echo \"Started tmux session '$SESS'. Attach with:  tmux attach -t $SESS\"
echo \"  Backend : http://<jetson-ip>:8000/state\"
echo \"  UI      : http://<jetson-ip>:5173\"
