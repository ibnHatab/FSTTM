# FSTTM on the Unitree Go2 (Jetson Orin NX 16G)

Deploying the dialog engine next to the perception stack (DINOv3 + detector +
LiDAR + nav2) on `unitree-jetson-payload` (JetPack 6.2 / L4T r36.4.7, Ubuntu
22.04, Python 3.10, CUDA 12.6, TensorRT 10.3, 8× A78AE, 15.3 GiB shared RAM).

## Execution model: temporal separation, not DLA

- **DINOv3-S + detector + LiDAR: GPU, continuous.**
- **Whisper + Phi-3: GPU, event-driven.** They run only on/after an utterance
  (STT burst, then an INT4 LLM burst of tens–hundreds of ms) and then sleep —
  the perception loop never competes with them at steady state:

```text
normal operation:      GPU ████████████████ DINOv3 + detector
user speaks:           GPU ████ DINOv3 ████ Whisper ███ Phi-3 ███ DINOv3
```

- **Leave DLA alone.** Whisper/Phi-3 are transformer-dominated (QKV,
  attention, LayerNorm/RMSNorm, KV cache) — the wrong workload for DLA's
  CNN-oriented layer set, and TensorRT dropped DLA support after 10.7.
- CPU-only FSTTM is the fallback profile: `gpt.n_gpu_layers: 0` + a CPU
  whisper.cpp build. Keep both profiles config-side; never hardwire.

## Budget

The dialog pipeline is serial (VAD → speaker-verify → STT → LLM → TTS):
peak demand is max-of-stages (~4–6 threads for seconds per turn),
steady-state <0.5 core.

| Stage | Component | RAM | Notes |
|---|---|---|---|
| VAD | webrtcvad | ~0 | continuous, negligible |
| Speaker verify | sherpa-onnx embedding (CPU) | ~30 MB | 10–50 ms/utterance; aarch64 wheel, bundled runtime, **no torch** (the robot has none) |
| STT | whisper.cpp base.en-q5_1 | ~300 MB | CPU RTF 0.3–0.5; GPU burst optional |
| LLM | Phi-3-mini Q4_K_M | 2.4 GB + KV | see below |
| TTS | RHVoice (statistical) | ~50 MB | near-instant, CPU |

- **KV cache**: Phi-3-mini ≈ 0.4 MB/token f16 → use `n_ctx: 2048` on the
  robot (≈0.8 GB). The two-pass KV-prefix cache evaluates the intent prompt
  ONCE at boot (~1–2 min CPU — schedule pre-mission); each turn evals only
  the utterance + generates the JSON/ack → ~4–8 s/turn CPU, sub-second GPU.
- **Total**: FSTTM ≈ 4 GB; DINOv3+GLIM+nav2+ROS+system ≈ 5–7 GB → ~9–11 of
  15.3 GiB shared RAM.
- **Share one Phi-3 instance** between dialog and the DINOv3 semi-open
  dictionary queries: the llama driver already serializes typed requests
  (IntentGenerate / ManualGenerate) through one worker — add a SceneQuery
  event rather than loading a second 2.4 GB model.
- Containment: run FSTTM under `nice 10` + a cpuset (e.g. cores 0–5),
  `gpt.n_threads: 4–6`; leave the nav2 controller cores alone.
- If turn latency disappoints: tiny.en STT → shorter prompt variant
  (`system.prompt_variant: one-shot`) → partial `n_gpu_layers`.

## Install

```bash
# Python 3.10 venv on the robot
pip install fsttm[voicefilter] fsttm-dog
```

- **llama-cpp-python / pywhispercpp**: build with CUDA on-device
  (`scripts/build-whisper.sh 87` for the Orin's sm_87), or CPU-only for the
  fallback profile.
- **RHVoice**: no arm64 apt package on Ubuntu 22.04 — build from source
  (CMake, CPU-only, small): <https://github.com/RHVoice/RHVoice>. The
  backend shells out to `RHVoice-client`, so only the binaries + an English
  voice are needed. Verify: `echo test | RHVoice-client -s SLT | aplay`.
- **Speaker filter**: drop a WeSpeaker/3D-Speaker ONNX under
  `models/speaker/`, enroll with `fsttm-enroll --record 3 --name <you>`,
  tune the threshold in `voice_filter.mode: shadow`, then switch to
  `enforce`.

## Prerequisites / open items

- **Audio hardware**: the Go2 head unit exposes HDMI audio only — a USB
  mic + speaker (or a USB conference puck) is required for both STT and TTS.
  PulseAudio is already running for the `unitree` user.
- AEC: enable `aec.enabled` only after the USB audio path exists; the
  headset-style setup may not need it (`soft_duck` handles barge-in).
- The dog domain ships logging stubs; the real backends implement the
  `fsttm_dog.actions` Protocols:
  - `ActionBackend` → Go2 wireless-controller/action interface,
  - `SemanticMemory.query(target, constraints) → candidates[]` → the DINOv3
    semantic map / object-instance memory,
  - `NavigationBackend.navigate(PoseGoal)` / `explore()` → nav2 action
    clients (`NavigateToPose`, frontier exploration).
  Config: `domains.dog.backend` selects the implementation.

Start from `contrib/dog/config.dog.sample.yaml`.
