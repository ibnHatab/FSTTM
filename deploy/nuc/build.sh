#!/usr/bin/env bash
# Build the Go engine on the NUC, natively (no containers): whisper.cpp (CPU) and go/bin/fsttm-go with
# -tags nollama,norhvoice (the LLM is the cloud one: no llama.cpp linked, so whisper's ggml is the only ggml).
#   deploy/nuc/build.sh            whisper.cpp if not built, then fsttm-go
#   deploy/nuc/build.sh test       the Go tests of the packages the NUC build uses
# Needs: build-essential cmake golang-go pkg-config libopus-dev libopusfile-dev libasound2-dev (install.sh).
set -euo pipefail
FSTTM="$(cd "$(dirname "$0")/../.." && pwd)"
WHISPER_DIR="${WHISPER_DIR:-$HOME/metis/whisper.cpp}"
if [ ! -d "$WHISPER_DIR/.git" ]; then
  git clone -q --depth 1 https://github.com/ggml-org/whisper.cpp "$WHISPER_DIR"
fi
if [ ! -f "$WHISPER_DIR/build/bin/libwhisper.so" ]; then
  cmake -S "$WHISPER_DIR" -B "$WHISPER_DIR/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
        -DWHISPER_BUILD_EXAMPLES=OFF -DWHISPER_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=ON >/dev/null
  cmake --build "$WHISPER_DIR/build" -j"$(nproc)" --target whisper
fi
# go.mod's replace names ../../whisper.cpp (beside the FSTTM checkout): a link to the build
[ -e "$FSTTM/../whisper.cpp" ] || ln -s "$WHISPER_DIR" "$FSTTM/../whisper.cpp"
WLIB="$WHISPER_DIR/build/bin"
export CGO_CFLAGS="-I$WHISPER_DIR/include -I$WHISPER_DIR/ggml/include"
export CGO_LDFLAGS="-L$WLIB -Wl,-rpath,$WLIB"
export LIBRARY_PATH="$WLIB" C_INCLUDE_PATH="$WHISPER_DIR/include:$WHISPER_DIR/ggml/include"
cd "$FSTTM/go"
if [ "${1:-}" = test ]; then
  export LD_LIBRARY_PATH="$WLIB"
  exec go test -tags nollama,norhvoice ./pkg/... ./internal/cloud/... ./internal/attention/... ./internal/audio/... \
    ./internal/tts/... ./internal/fsm/... ./internal/panel/... ./internal/pipeline/...
fi
go build -tags nollama,norhvoice -o bin/ ./cmd/fsttm-go
echo "built $FSTTM/go/bin/fsttm-go (rpath: $WLIB)"
