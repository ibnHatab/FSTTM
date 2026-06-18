#!/usr/bin/env bash
# Launch the fsttm voice server with the environment whisper.cpp (CUDA) needs:
#   - llama_cpp imports first (binds its OWN ggml) before whisper preloads
#     its bundled ggml RTLD_GLOBAL — avoids ggml symbol collision / abort
#   - CUDA runtime libs in /usr/local/cuda-11.4/lib64
#   - FSTTM_NO_AEC=1 (this PulseAudio module-echo-cancel outputs silence)
# Pass --tui for the 3-panel Rich interface (chat / intents / state+perf):
#   scripts/run-voice.sh --tui
set -e
cd "$(dirname "$0")/.."
. .venv/bin/activate
export LD_LIBRARY_PATH="/usr/local/cuda-11.4/lib64:${LD_LIBRARY_PATH:-}"
export FSTTM_NO_AEC=1
exec python -u -m fsttm.server --config config.jetson.yaml "$@"
