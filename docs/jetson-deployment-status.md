# FSTTM Jetson Deployment — Actual Status

> Reproducible record of the remote provisioning done on **2026-05-30**.
> The companion [`jetson-orin-nx.md`](jetson-orin-nx.md) is a forward-looking
> *optimization plan* for the target **Orin NX** (sm_87 / CUDA 12.2 / TRT 10).
> This file documents what is **actually deployed and verified today** on the
> reachable hardware, which is a different board.

## Deployed host

| | Value |
|---|---|
| SSH alias | `jetson` → `10.0.0.67`, user `nvidia` |
| Board | **Jetson AGX Xavier** (not the Orin NX in the plan) |
| Compute capability | **sm_72** (Xavier), CUDA arch `72` |
| JetPack / L4T | R35.2.1 |
| OS | Ubuntu 20.04.6, aarch64 |
| Python | **3.8.10** (system) |
| CUDA toolkit | 11.4 (`/usr/local/cuda-11.4`, `nvcc` V11.4.315) |
| Repo path | `~/repo/vox/fsttm` |
| Git remote auth | Jetson key already authorized on `cicd.skyway.porsche.com` |

## Branch

Everything runs on branch **`py38-backport`** (pushed, no MR). `main` uses
Python 3.10+ syntax (`match`/`case`, PEP 585 generics) that does **not** import
on the Jetson's Python 3.8. The branch backports:

- `grammar.py`: `match`/`case` → `if`/`elif` (semantics preserved).
- `grammar.py`, `llama.py`, `server.py`, `two_pass.py` + three test files:
  `from __future__ import annotations` to make subscripted-generic annotations
  lazy on 3.8 (all such uses are annotation-only).
- `headless.py`: new **`--no-aec`** flag — skips PipeWire `module-echo-cancel`
  so headless LLM/intent testing runs on a server with no audio session.

## Environment setup (reproduce)

```bash
ssh jetson
cd ~/repo/vox/fsttm
git checkout py38-backport

python3 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip setuptools wheel

# Core (pure-python / prebuilt aarch64 wheels)
pip install "reactivex>=4.0" "cyclotron>=2.0.0" "cyclotron-std>=2.0.0" \
  "webrtcvad>=2.0.10" "numpy>=1.24.0,<2" "scipy>=1.10.0" \
  "pydantic>=2.0.0" "pyyaml>=6.0" pytest httpx
pip install -e . --no-deps        # install fsttm package itself

# Audio I/O
sudo apt-get install -y portaudio19-dev
pip install pyaudio

# STT — faster-whisper. Pin tokenizers to an aarch64/cp38 wheel; the default
# (0.21) needs a Rust build (puccinialin/maturin) that has no wheel here.
pip install "tokenizers==0.19.1" "faster-whisper==1.0.3"

# LLM — llama-cpp-python built from source with CUDA for sm_72 (~20-30 min)
export PATH=/usr/local/cuda-11.4/bin:$PATH
export CUDACXX=/usr/local/cuda-11.4/bin/nvcc CUDA_HOME=/usr/local/cuda-11.4
export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=72 -DLLAVA_BUILD=off"
export FORCE_CMAKE=1
pip install --no-cache-dir "llama-cpp-python==0.2.90"
```

## Models (`~/repo/vox/fsttm/models/`)

| File | Size | Status |
|---|---|---|
| `Phi-3-mini-4k-instruct-Q6_K.gguf` | 3.14 GB | ✅ loads, 33/33 layers on GPU |
| `en_US-lessac-medium.onnx` (+`.json`) | 63 MB | ✅ loads via onnxruntime |
| whisper `base` | — | auto-downloaded by faster-whisper on first use |

A Jetson config is generated from the sample with corrected paths:
`config.jetson.yaml` (untracked) — `sed 's#/home/axadmin/repo/vox/FSTTM#/home/nvidia/repo/vox/fsttm#g' config.sample.yaml`.

## Verified working

- All `fsttm.*` modules import on Python 3.8 (except `rhvoice` — unused).
- 16 pure-logic tests pass (`pytest`, excluding hardware/e2e).
- **End-to-end headless intent pipeline on GPU** (`--intent --no-aec`):
  - `found 1 CUDA devices: Device 0: Xavier, compute capability 7.2`
  - `offloaded 33/33 layers to GPU`, CUDA0 buffer 2912 MiB
  - Two-pass grammar-constrained generation + KV rollback + TTS text all run.
  - Latency: JSON ~0.9–1.3 s, TTS text ~0.4 s per turn (Xavier; Orin NX expected faster).

Run it:
```bash
printf "its too cold\n" | python -m fsttm.headless \
  --config config.jetson.yaml \
  --model models/Phi-3-mini-4k-instruct-Q6_K.gguf \
  --prompt prompts/hvac-intentions-phi3.txt \
  --intent --no-aec
```

## Full voice e2e (hvac-react + fsttm voice)

All three services run **on the Jetson** (the Jabra EVOLVE 20 mic+speaker is
plugged into it). Topology: Jabra → fsttm voice (STT→Phi-3 intent→TTS) →
HTTP POST → hvac-react backend → WebSocket → React UI.

One-time extra setup beyond the base provision:

```bash
# Node 20 LTS (NodeSource arm64) — apt's nodejs is too old for Vite 5
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - && sudo apt-get install -y nodejs
sudo apt-get install -y tmux

# hvac-react backend venv (Py3.8-compatible pins + eval_type_backport so
# pydantic can evaluate the `str | None` model fields on 3.8)
cd hvac-react/backend && python3 -m venv .venv && . .venv/bin/activate
pip install "fastapi>=0.110,<0.116" "uvicorn[standard]==0.33.0" \
            "pydantic>=2.7,<2.11" websockets eval_type_backport
deactivate

# frontend deps
cd ../frontend && npm install
```

Audio: the Jetson runs **PulseAudio** (no PipeWire). `fsttm/aec.py` now
auto-detects the Jabra source by name and loads `module-echo-cancel` (falling
back to no `aec_args`, which this PulseAudio build rejects), creating the
`fsttm_ec_source` virtual mic the config points at. Set `FSTTM_NO_AEC=1` to
capture the raw mic instead.

### STT on GPU — whisper.cpp via pywhispercpp

STT was the latency bottleneck (CPU faster-whisper ~2500 ms/utterance). Now
**whisper.cpp built with CUDA (sm_72) + pywhispercpp**, in-process and warm:
**~150 ms/utterance** (≈18× faster). `fsttm/whisper.py` loads a GGML model
(prefers `ggml-base.en-q5_1.bin`) and does a CUDA warmup pass (the first encode
JITs kernels, ~1.2 s one-time).

Build (one-time):
```bash
# whisper.cpp CUDA build (needs cmake ≥3.18 — system 3.16 too old)
pip install "cmake>=3.22,<4"
cd ~/repo/vox && git clone --depth 1 https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
PATH=/usr/local/cuda-11.4/bin:$PATH CUDACXX=/usr/local/cuda-11.4/bin/nvcc \
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=72 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j6
bash ./models/download-ggml-model.sh base.en-q5_1   # 57 MB quant

# pywhispercpp built against CUDA (vendors its own whisper.cpp submodule)
PATH=/usr/local/cuda-11.4/bin:$PATH CUDACXX=/usr/local/cuda-11.4/bin/nvcc \
  CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=72" GGML_CUDA=1 \
  pip install --no-binary pywhispercpp pywhispercpp==1.5.0
# its Python uses 3.10+ syntax — add `from __future__ import annotations` to
# site-packages/pywhispercpp/{utils,model}.py (PEP 604 union in a signature).
```
pywhispercpp's bundled `.so` libs land in site-packages, off the loader path —
**`scripts/run-voice.sh` sets `LD_LIBRARY_PATH` (site-packages + CUDA libs)**.

Benchmark whisper.cpp variants: `~/repo/vox/whisper.cpp/bench_stt.sh <wav>`.
(`whisper-cli` per-run shows ~1.2 s because each cold process re-JITs CUDA;
`whisper-bench` shows the true warm encode: base-q5 ~63 ms, tiny ~31 ms.)

### Start the e2e

```bash
# backend (:8000) + frontend (:5173, bound 0.0.0.0)
~/repo/vox/fsttm/hvac-react/start.sh
# voice server (sets LD_LIBRARY_PATH + FSTTM_NO_AEC for whisper.cpp CUDA)
~/repo/vox/fsttm/scripts/run-voice.sh
```

Or all three in a tmux session: `~/repo/vox/fsttm/scripts/start-e2e.sh`
(update it to call run-voice.sh for the voice pane). Open the UI from your
browser at `http://<jetson-ip>:5173` (Jetson is 10.0.0.67). Speak an HVAC
command into the Jabra; the panel updates live and the assistant speaks.

Verified piecewise: backend `/state` + `/command` apply intents
(`bump_temperature` → temp 22.0→22.5 in both zones), the fsttm bridge command
format matches the backend, Phi-3 intent generation runs on GPU, and AEC
creates `fsttm_ec_source`.

## Audio (mic + TTS) — important

**AEC is broken on this PulseAudio.** `module-echo-cancel` (loaded without the
`aec_args` this build rejects) outputs pure silence — the EC source reads RMS=0
while the raw Jabra is live. So **run with `FSTTM_NO_AEC=1`**, which captures the
raw Jabra directly (server auto-sets it as default source @90% volume). The
EVOLVE 20 is a headset, so echo bleed is negligible.

**piper TTS now works** (lessac voice, spoken output):
```bash
pip install piper-phonemize-cross==1.2.0   # community aarch64/cp38 wheel; no build
sudo apt-get install -y espeak-ng
# the cross wheel looks for espeak data at /usr/share/espeak-ng-data:
sudo ln -sfn /usr/lib/aarch64-linux-gnu/espeak-ng-data /usr/share/espeak-ng-data
```
`fsttm/piper.py` uses this wheel's `synthesize_stream_raw()` API and, under
`FSTTM_NO_AEC`, plays to the default `pulse` sink (the Jabra). Verified: lessac
synthesizes and plays (RMS ~5700). Test audio out standalone with
`scripts/tts_check.py [tone|piper|espeak]`.

`Bad voice attribute: option` from espeak-ng is a harmless warning.

## Known gaps / deferred

- **piper → TensorRT**: still the eventual target (see `jetson-orin-nx.md` §1)
  for ~3× TTS speedup; the CPU lessac path above is the working stopgap.
- **whisper on CUDA**: faster-whisper installed; switch `device="cuda",
  compute_type="float16"` in `fsttm/whisper.py` per `jetson-orin-nx.md` §2 to use GPU.
- **`rhvoice`**: unused, not installed.

## Disk note

The board's eMMC was at 88 % (3.4 GB free) on arrival. Aggressive cleanup
(removed nsight GUI profilers ~2.1 GB, firefox, pip/apt caches ~1.3 GB,
`/opt/ota_package`, `/usr/share/doc`) freed ~4.5 GB. **Kept**: CUDA 11.4,
TensorRT, gcc/g++/make/cmake, DeepStream. After models + venv + llama build:
~85 % used. Watch free space before adding the whisper CUDA cache and TRT engines.
