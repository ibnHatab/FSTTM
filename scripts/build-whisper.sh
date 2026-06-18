#!/usr/bin/env bash
# Build whisper.cpp with CUDA + the pywhispercpp Python binding, and fetch the
# GGML model fsttm uses (ggml-base.en-q5_1) into models/.
#
# fsttm/whisper.py resolves a short STT model name ("base") against
# ~/repo/vox/whisper.cpp/models, preferring the q5_1 quant; config.*.yaml may
# instead point at a full path under the project models/ dir. This script
# populates both: it builds whisper.cpp at $WHISPER_DIR and symlinks/copies the
# model into the project models/ so either config style works.
#
# Usage:
#   scripts/build-whisper.sh [CUDA_ARCH]
#     CUDA_ARCH   GPU compute capability (default: auto-detect; Jetson Xavier=72)
# Env:
#   CUDA_HOME     CUDA toolkit root (default: auto from nvcc/PATH)
#   WHISPER_DIR   where to clone/build whisper.cpp (default ~/repo/vox/whisper.cpp)
#   MODEL         ggml model name (default base.en-q5_1)
#   PYWHISPER_VER pywhispercpp version (default 1.5.0)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WHISPER_DIR="${WHISPER_DIR:-$HOME/repo/vox/whisper.cpp}"
MODEL="${MODEL:-base.en-q5_1}"
PYWHISPER_VER="${PYWHISPER_VER:-1.5.0}"

# ── CUDA toolkit ──────────────────────────────────────────────────────────────
if [ -z "${CUDA_HOME:-}" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  elif [ -d /usr/local/cuda ]; then
    CUDA_HOME=/usr/local/cuda
  else
    echo "ERROR: CUDA not found. Set CUDA_HOME (e.g. /usr/local/cuda-11.4)." >&2
    exit 1
  fi
fi
NVCC="$CUDA_HOME/bin/nvcc"
[ -x "$NVCC" ] || { echo "ERROR: nvcc not at $NVCC" >&2; exit 1; }
echo "CUDA: $CUDA_HOME ($("$NVCC" --version | grep -oE 'release [0-9.]+' | head -1))"

# ── CUDA arch ─────────────────────────────────────────────────────────────────
ARCH="${1:-}"
if [ -z "$ARCH" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')"
    ARCH="${CC:-72}"
  else
    ARCH=72   # Jetson AGX Xavier default
  fi
fi
echo "CUDA arch: sm_$ARCH"

# ── cmake (whisper.cpp needs >=3.18) ──────────────────────────────────────────
if ! cmake --version 2>/dev/null | head -1 | grep -qE '3\.(1[89]|[2-9][0-9])|[4-9]\.'; then
  echo "Installing cmake>=3.22 into the active venv…"
  pip install -q "cmake>=3.22,<4"
fi

# ── clone + build whisper.cpp ─────────────────────────────────────────────────
if [ ! -d "$WHISPER_DIR/.git" ]; then
  echo "Cloning whisper.cpp → $WHISPER_DIR"
  git clone --depth 1 https://github.com/ggerganov/whisper.cpp "$WHISPER_DIR"
fi
cd "$WHISPER_DIR"
PATH="$CUDA_HOME/bin:$PATH" CUDACXX="$NVCC" \
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
        -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
echo "whisper.cpp built (build/)."

# ── model ─────────────────────────────────────────────────────────────────────
echo "Fetching ggml model: $MODEL"
bash ./models/download-ggml-model.sh "$MODEL"
SRC="$WHISPER_DIR/models/ggml-$MODEL.bin"
mkdir -p "$ROOT/models"
DST="$ROOT/models/ggml-$MODEL.bin"
if [ -f "$SRC" ] && [ ! -e "$DST" ]; then
  ln -s "$SRC" "$DST"
  echo "Linked $DST → $SRC"
fi

# ── pywhispercpp (CUDA) ───────────────────────────────────────────────────────
echo "Building pywhispercpp==$PYWHISPER_VER with CUDA…"
PATH="$CUDA_HOME/bin:$PATH" CUDACXX="$NVCC" \
  CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$ARCH" GGML_CUDA=1 \
  pip install --no-binary pywhispercpp "pywhispercpp==$PYWHISPER_VER"

# pywhispercpp's Python uses 3.10+ union syntax in a couple of signatures; on
# Python 3.8 add a future import so it loads. No-op on 3.9+.
PYVER="$(python -c 'import sys;print("%d%d"%sys.version_info[:2])')"
if [ "$PYVER" -lt 39 ]; then
  SP="$(python -c 'import pywhispercpp,os;print(os.path.dirname(pywhispercpp.__file__))')"
  for f in utils model; do
    if [ -f "$SP/$f.py" ] && ! head -1 "$SP/$f.py" | grep -q "from __future__"; then
      sed -i '1i from __future__ import annotations' "$SP/$f.py"
      echo "  patched $SP/$f.py for Py3.8 (PEP 604)"
    fi
  done
fi

echo
echo "Done. whisper.cpp + pywhispercpp (CUDA sm_$ARCH) ready; model at $DST"
echo "run-voice.sh sets LD_LIBRARY_PATH for the bundled .so libs."
