#!/usr/bin/env bash
# Build fsttm-go against local llama.cpp + whisper.cpp checkouts.
# Both must be built already (cmake, optionally -DGGML_CUDA=ON).
#   LLAMA_DIR   (default ~/repo/vox/llama.cpp)
#   WHISPER_DIR (default ~/repo/vox/whisper.cpp)
set -euo pipefail
cd "$(dirname "$0")"
LLAMA_DIR="${LLAMA_DIR:-$HOME/repo/vox/llama.cpp}"
WHISPER_DIR="${WHISPER_DIR:-$HOME/repo/vox/whisper.cpp}"

# whisper.cpp lib layout differs by vintage: older builds emit under
# build/src + build/ggml/src; current master collects everything in build/bin
WLIBS="$WHISPER_DIR/build/src:$WHISPER_DIR/build/ggml/src:$WHISPER_DIR/build/ggml/src/ggml-cuda:$WHISPER_DIR/build/bin"
LLIBS="$LLAMA_DIR/build/bin"

export CGO_CFLAGS="-I$LLAMA_DIR/include -I$LLAMA_DIR/ggml/include -I$WHISPER_DIR/include -I$WHISPER_DIR/ggml/include"
export CGO_LDFLAGS="-L$LLIBS -Wl,-rpath,$LLIBS -L${WLIBS//:/ -L} -Wl,-rpath,${WLIBS//:/ -Wl,-rpath,}"
# whisper bindings resolve -lwhisper/-lggml* via LIBRARY_PATH
export LIBRARY_PATH="$LLIBS:$WLIBS"
export C_INCLUDE_PATH="$WHISPER_DIR/include:$WHISPER_DIR/ggml/include:$LLAMA_DIR/include:$LLAMA_DIR/ggml/include"

# pin the whisper bindings replace to WHISPER_DIR (the committed go.mod
# carries a relative fallback that only works in the vox/ sibling layout)
go mod edit -replace github.com/ggerganov/whisper.cpp/bindings/go="$WHISPER_DIR/bindings/go"

if [ "${1:-}" = "test" ]; then
  shift
  export LD_LIBRARY_PATH="$LLIBS:$WLIBS"
  exec go test "${@:-./...}"
fi
TAGS=()
if [ "${ROS:-}" = "1" ]; then
  TAGS=(-tags ros)
  # merge rclgo-gen's cgo flags (ROS typesupport headers/libs) with ours
  if [ -f ros-cgo-flags.env ]; then
    eval "$(sed 's/^export CGO_CFLAGS=/ROS_CGO_CFLAGS=/;s/^export CGO_LDFLAGS=/ROS_CGO_LDFLAGS=/' ros-cgo-flags.env)"
    export CGO_CFLAGS="$CGO_CFLAGS $ROS_CGO_CFLAGS"
    export CGO_LDFLAGS="$CGO_LDFLAGS $ROS_CGO_LDFLAGS"
  fi
fi
if [ $# -gt 0 ]; then targets=(); for t in "$@"; do targets+=("./cmd/$t"); done; else targets=(./cmd/...); fi
go build "${TAGS[@]}" -o bin/ "${targets[@]}"
echo "built → go/bin/ (rpath-embedded; no LD_LIBRARY_PATH needed)"
