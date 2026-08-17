#!/usr/bin/env bash
# Start the HVAC backend (FastAPI :8000) and frontend dev server (Vite :5173)
# together. Frontend binds 0.0.0.0 so the UI is reachable from another machine.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Free our ports if a previous run left something listening (avoids the
# "address already in use" / "Port 5173 is in use" fallbacks).
for port in 8000 5173; do
  pids=$(lsof -ti tcp:"$port" 2>/dev/null || true)
  [ -n "$pids" ] && { echo "freeing port $port (pids: $pids)"; kill $pids 2>/dev/null || true; }
done

# backend
cd "$ROOT/backend"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  . .venv/bin/activate
  pip install -q -r requirements.txt
else
  . .venv/bin/activate
fi
uvicorn server:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!
trap "kill $BACKEND_PID 2>/dev/null" EXIT

# frontend — bind all interfaces so the UI is reachable from your browser
cd "$ROOT/frontend"
[ -d node_modules ] || npm install
npm run dev -- --host 0.0.0.0
