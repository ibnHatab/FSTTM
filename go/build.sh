#!/usr/bin/env bash
# Build fsttm-go against local llama.cpp + whisper.cpp checkouts.
# Both must be built already (cmake, optionally -DGGML_CUDA=ON).
#   LLAMA_DIR   (default ~/repo/vox/llama.cpp)
#   WHISPER_DIR (default ~/repo/vox/whisper.cpp)
set -euo pipefail
cd "$(dirname "$0")"
LLAMA_DIR="${LLAMA_DIR:-$HOME/repo/vox/llama.cpp}"
WHISPER_DIR="${WHISPER_DIR:-$HOME/repo/vox/whisper.cpp}"

WLIBS="$WHISPER_DIR/build/src:$WHISPER_DIR/build/ggml/src:$WHISPER_DIR/build/ggml/src/ggml-cuda"
LLIBS="$LLAMA_DIR/build/bin"

export CGO_CFLAGS="-I$LLAMA_DIR/include -I$LLAMA_DIR/ggml/include -I$WHISPER_DIR/include -I$WHISPER_DIR/ggml/include"
export CGO_LDFLAGS="-L$LLIBS -Wl,-rpath,$LLIBS -L${WLIBS//:/ -L} -Wl,-rpath,${WLIBS//:/ -Wl,-rpath,}"
# whisper bindings resolve -lwhisper/-lggml* via LIBRARY_PATH
export LIBRARY_PATH="$LLIBS:$WLIBS"
export C_INCLUDE_PATH="$WHISPER_DIR/include:$WHISPER_DIR/ggml/include:$LLAMA_DIR/include:$LLAMA_DIR/ggml/include"

if [ "${1:-}" = "test" ]; then
  shift
  export LD_LIBRARY_PATH="$LLIBS:$WLIBS"
  exec go test "${@:-./...}"
fi
if [ $# -gt 0 ]; then targets=(); for t in "$@"; do targets+=("./cmd/$t"); done; else targets=(./cmd/...); fi
go build -o bin/ "${targets[@]}"
echo "built → go/bin/ (rpath-embedded; no LD_LIBRARY_PATH needed)"
