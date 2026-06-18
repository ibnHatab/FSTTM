# FSTTM on Nvidia Jetson Orin NX — Optimisation Plan

## Target Hardware

| Spec | Jetson Orin NX 16 GB |
|------|----------------------|
| GPU | Ampere (sm_87), 1024 CUDA cores |
| Unified memory | 16 GB (GPU+CPU shared) |
| Memory bandwidth | 102 GB/s |
| TensorRT | 10.x |
| CUDA | 12.2 |
| Power envelope | 10–25 W |

No discrete VRAM boundary — all 16 GB is available to both GPU and CPU, but bandwidth is shared.

---

## Component Budget

| Component | x86 now | Jetson target | Notes |
|-----------|---------|--------------|-------|
| Phi-3-mini Q6_K | 2.9 GB | 2.9 GB | sm_87 CUDA build |
| KV cache (4096 ctx) | ~0.5 GB | ~0.5 GB | both on GPU |
| Whisper base (float16) | 145 MB CPU | 145 MB GPU | faster-whisper cuda |
| Piper TTS (ONNX) | 61 MB CPU | ~40 MB TRT engine | FP16 |
| **Total** | **~3.5 GB** | **~3.6 GB / 16 GB** | 77% headroom |

---

## Step 1 — Piper TTS → TensorRT

The piper ONNX model runs on CPU with `piper-tts`. Export to a TRT engine for ~3× speedup:

```bash
# Prerequisites
pip install tensorrt

# Convert (on Jetson)
trtexec \
  --onnx=models/en_US-lessac-medium.onnx \
  --saveEngine=models/en_US-lessac-medium-fp16.engine \
  --fp16 \
  --minShapes=input:1x1,input_lengths:1 \
  --optShapes=input:1x300,input_lengths:1 \
  --maxShapes=input:1x1000,input_lengths:1
```

Then replace `PiperVoice.synthesize()` in `fsttm/piper.py` with a TRT inference session:

```python
import tensorrt as trt
import cuda   # pycuda or cupy

class TRTPiper:
    def __init__(self, engine_path: str):
        # load .engine, allocate input/output buffers
        ...
    def synthesize(self, text: str) -> bytes:
        # phonemize → mel → TRT inference → PCM
        ...
```

Expected latency: CPU ONNX ~120 ms → TRT FP16 ~35 ms

---

## Step 2 — Whisper on CUDA

Switch `faster-whisper` from CPU int8 to CUDA float16:

```python
# fsttm/whisper.py
whisper_model = WhisperModel(
    model_name,
    device="cuda",        # was "cpu"
    compute_type="float16",  # was "int8"
)
```

Build whisper.cpp for Jetson if Python binding latency is too high:
```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build --target main -j4
```

Expected: faster-whisper base CUDA ~80 ms (vs CPU int8 ~280 ms)

---

## Step 3 — llama.cpp for Jetson (sm_87)

Current build uses sm_120 (Blackwell x86). Rebuild for Orin:

```bash
cd /home/axadmin/repo/vox/llama.cpp
cmake -B build-jetson \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DLLAMA_BUILD_SERVER=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-jetson --target llama-server -j4
```

llama-cpp-python rebuild for Jetson:
```bash
PATH=/usr/local/cuda/bin:$PATH \
CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87" \
pip install --force-reinstall --no-cache-dir \
  git+https://github.com/abetlen/llama-cpp-python.git
```

**Model selection for 16 GB budget:**

| Model | Size | Fits? | Inference speed (est.) |
|-------|------|-------|------------------------|
| Phi-3-mini Q6_K | 2.9 GB | ✓ | ~40 tok/s |
| Phi-3-mini Q4_K_M | 1.8 GB | ✓ | ~60 tok/s |
| Llama-3.2-3B Q6_K | 2.5 GB | ✓ | ~45 tok/s |

Recommended: **Q4_K_M** on Jetson for best latency/quality trade-off.

---

## Step 4 — Single-Process Deployment

Replace Python reactive pipeline + subprocess servers with a single C++/Go binary for production:
- whisper.cpp for STT (no Python overhead)
- llama.cpp for LLM (same binary)
- piper TRT for TTS (via C++ bindings)
- AEC: use `libechocancel` or Jetson's hardware echo-cancel DSP

This eliminates asyncio overhead, IPC, and Python GIL issues.

---

## Power Budget

| Mode | GPU watts | CPU watts | Total |
|------|-----------|-----------|-------|
| Idle | 1 W | 1 W | ~5 W (platform) |
| STT only | 3 W | 2 W | ~10 W |
| LLM generating | 8 W | 3 W | ~15 W |
| All active | 10 W | 4 W | ~20 W |

Jetson NX 15W mode: all components fit.

---

## Summary Timeline

| Step | Effort | Benefit |
|------|--------|---------|
| Whisper CUDA | 1 hour | 3× STT speedup |
| llama.cpp sm_87 rebuild | 2 hours | GPU inference on Jetson |
| Piper TRT conversion | 1 day | 3× TTS speedup |
| Single binary | 2 weeks | production-grade deploy |
