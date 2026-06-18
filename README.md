# Finite-State Turn-Taking Machine (FSTTM)

A spoken dialog system implementing the N09-1071 turn-taking model with a fully local LLM+STT+TTS pipeline, PipeWire acoustic echo cancellation, and grammar-constrained intent generation.

**Paper:** [A Finite-State Turn-Taking Model for Spoken Dialog Systems](https://aclanthology.org/N09-1071.pdf) — Raux & Eskenazi, NAACL 2009

---

## Architecture

```
Microphone (Jabra)
    │
    ▼
PipeWire AEC (module-echo-cancel, WebRTC)
    │  fsttm_ec_source — echo-cancelled mic input
    │
    ▼
VAD (WebRTC VAD, energy-based utterance segmentation)
    │  vad_collector: frames → None delimiters
    │
    ▼
STT (faster-whisper, CPU int8)
    │  TextResult(text)
    │
    ▼
FSM Gate (N09-1071 6-state turn-taking model)
    │  only passes utterances when user holds floor
    │
    ├─[normal mode]──────────────────────────────────────────────┐
    │  LLM (llama-cpp-python, CUDA)                              │
    │  streaming tokens → ResponseDone                           │
    │                                                            │
    ├─[intent mode]──────────────────────────────────────────────┤
    │  Pass 1: grammar-constrained JSON (LlamaGrammar)           │
    │  Pass 2: TTS text via KV-cache rollback (eval+save_state)  │
    │  → IntentResult(intent_json, tts_text)                     │
    │                                                            │
    ▼                                                            │
TTS (piper-tts → PyAudio → pulse sink)◄─────────────────────────┘
    │
    ▼
Speaker output
```

---

## Quick Start

### Requirements

- Ubuntu 24.04, Python 3.12
- NVIDIA GPU with CUDA 12.9 (CPU-only also works, slower)
- PipeWire 1.0+ (default on Ubuntu 22.04+)
- Jabra Link 370 or any USB microphone

### Install

```bash
git clone git@cicd.skyway.porsche.com:PG50/fsttm.git
cd fsttm

python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[test]"

# Build llama-cpp-python from source with CUDA (required for recent models)
PATH=/usr/local/cuda/bin:$PATH \
CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native" \
pip install --force-reinstall --no-cache-dir \
  git+https://github.com/abetlen/llama-cpp-python.git
```

### Download Models

> **Models are not stored in git** (too large). Download into `models/` before running.

```bash
mkdir -p models
```

#### LLM — choose one

| Model | Size | Use case | Download |
|-------|------|----------|----------|
| **Phi-3-mini-4k-instruct Q6_K** ← recommended | 2.9 GB | Voice assistant, intent mode | [bartowski/Phi-3-mini-4k-instruct-GGUF](https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF) |
| Llama-3.2-3B-Instruct Q6_K | 2.5 GB | General chat | [bartowski/Llama-3.2-3B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) |
| Phi-4-mini-reasoning Q6_K | 2.9 GB | Math/logic (verbose) | [lmstudio-community/Phi-4-mini-reasoning-GGUF](https://huggingface.co/lmstudio-community/Phi-4-mini-reasoning-GGUF) |

```bash
# Phi-3-mini — recommended for HVAC intent mode
wget -O models/Phi-3-mini-4k-instruct-Q6_K.gguf \
  "https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF/resolve/main/Phi-3-mini-4k-instruct-Q6_K.gguf"

# Llama-3.2-3B (alternative)
wget -O models/Llama-3.2-3B-Instruct-Q6_K.gguf \
  "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K.gguf"

# Phi-4-mini-reasoning (math/logic)
wget -O models/Phi-4-mini-reasoning-Q6_K.gguf \
  "https://huggingface.co/lmstudio-community/Phi-4-mini-reasoning-GGUF/resolve/main/Phi-4-mini-reasoning-Q6_K.gguf"
```

#### TTS voice — Piper en_US-lessac-medium

| File | Size | Source |
|------|------|--------|
| `en_US-lessac-medium.onnx` | 61 MB | [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium) |
| `en_US-lessac-medium.onnx.json` | 4 KB | same |

```bash
wget -O models/en_US-lessac-medium.onnx \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
wget -O models/en_US-lessac-medium.onnx.json \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
```

### Configure

Edit `config.sample.yaml` (copy to `config.yaml` for custom deployments):

```yaml
vad:
    vad_aggressiveness: 3
    device: null                          # null = system default
    device_name: "fsttm_ec_source"        # PipeWire AEC virtual mic
    rate: 16000
    padding_ms: 700                       # end-of-utterance silence threshold

stt:
    model: "base"                         # faster-whisper model size
    language: "en"                        # null = auto-detect

tts:
    model: "/abs/path/to/models/en_US-lessac-medium.onnx"
    sample_rate: 22050

gpt:
    model: "/abs/path/to/models/Phi-3-mini-4k-instruct-Q6_K.gguf"
    n_ctx: 2048
    n_threads: 6
    temp: 0.7

    # Intent / grammar mode
    intent_mode: false                    # true = structured JSON output
    intent_approach: "a"                  # a = KV rollback, b = translation prompt
    intent_prompt: null                   # path to system prompt file
```

### Run

```bash
source .venv/bin/activate

# Full pipeline (voice in → LLM → voice out)
python -m fsttm.server --config config.sample.yaml

# Headless test (keyboard input, no audio hardware)
python -m fsttm.headless --config config.sample.yaml
```

---

## Intent Mode (Grammar-Constrained Generation)

Intent mode replaces open-ended LLM responses with structured JSON commands suitable for application integration (e.g. HVAC control, navigation).

### How It Works

Two-pass generation using llama-cpp-python's low-level API:

**Pass 1 — Grammar-constrained JSON**
```
model.eval(base_prompt_tokens)    # fill KV cache with system+user prompt
state = model.save_state()        # snapshot after prompt evaluation
model.generate([], grammar=g)     # sample JSON tokens (guaranteed valid)
```

**Pass 2 — TTS text via KV rollback**
```
model.load_state(state)           # roll back to after-prompt state
model.eval(tts_cue_tokens)        # append "[json]\nSpoken response:" tokens
model.generate([], grammar=None)  # sample natural speech text
```

The prompt is evaluated **once**; both generations share the KV cache. Measured latency: **150–400 ms/turn** on RTX PRO 2000 (Blackwell, sm_120).

### Example — HVAC Control

Enable in config:
```yaml
gpt:
    intent_mode: true
    intent_approach: "a"
    intent_prompt: "prompts/hvac-intentions-phi3.txt"
```

Benchmark (headless):
```bash
python -m fsttm.headless --config config.sample.yaml \
  --model models/Phi-3-mini-4k-instruct-Q6_K.gguf \
  --prompt prompts/hvac-intentions-phi3.txt \
  --intent both
```

Sample output:
```
[user] it's too cold
  JSON  → {'intent': 'WARMER', 'delta': 1, 'zone': 'both'}
  Voice → "I'm warming things up for you."

[user] defrost the windshield
  JSON  → {'intent': 'VENT_DEFROST', 'zone': 'both'}
  Voice → 'Activating defrost for clear windshield.'
```

### Intent Approaches

| Flag | Approach | Latency | Notes |
|------|----------|---------|-------|
| `a` | eval + KV rollback | ~300 ms | Prompt evaluated once; both passes share cache. Default. |
| `b` | translation prompt | ~450 ms | Separate few-shot translation; simpler but slower and less reliable |
| `both` | benchmark | — | Runs both and prints comparison table |

### HVAC Intent Catalog

See `prompts/hvac-intentions-phi3.txt` for the full catalog. Supported intents:

`WARMER` · `COOLER` · `SET_TEMPERATURE` · `SET_FAN` · `FAN_UP` · `FAN_DOWN` ·
`AC_ON` · `AC_OFF` · `VENT_FACE` · `VENT_FEET` · `VENT_DEFROST` · `VENT_SPLIT` ·
`RECIRCULATE_ON` · `RECIRCULATE_OFF` · `AUTO_ON` · `AUTO_OFF` · `STATUS` · `UNKNOWN`

Each response includes `zone` (`both`/`driver`/`passenger`) and optional parameters (`temp`, `delta`, `level`, `unit`).

---

## Turn-Taking FSM

Six-state model from N09-1071. States represent floor *ownership* (intent/obligation), not surface speech/silence.

```
         FREEs (gap after system)
           │
    ┌──────┼──────┐
    │      │      │
 SYSTEM  BOTHs  USER
    │      │      │
    └──────┼──────┘
           │
         BOTHu
           │
         FREEu (gap after user)
```

| State | Meaning |
|-------|---------|
| `USER` | User holds floor |
| `SYSTEM` | System holds floor |
| `BOTHs` | Overlap: user barged in on system |
| `BOTHu` | Overlap: system interrupted user |
| `FREEs` | Gap after system spoke |
| `FREEu` | Gap after user spoke |

**Cost model** (N09-1071 §3.2, implemented in `fsttm/fsttm.py`):
- `C_u = 5000` — cost of cutting in on user
- `C_gap(τ) = 1×τ` — gap wait cost grows linearly with pause duration
- `C_o(τ) = exp((τ+100)/1000)` — overlap cost grows exponentially

**FSM wire-up** in `fsttm/server.py`:
- VAD speech start → `user_action('G')`
- Utterance end (silence) → `user_action('R')`
- LLM/intent response ready → `system_action('G')`
- TTS playback done → `system_action('R')`
- Barge-in during TTS → `user_action('G')` + `system_action('R')` (with 600 ms grace period)

---

## Acoustic Echo Cancellation

PipeWire `module-echo-cancel` (WebRTC AEC) loaded dynamically at app start via `EchoCancelSession` (`fsttm/aec.py`). Unloaded cleanly on exit; stale modules from crashes are auto-cleaned on next start.

```
Jabra mic ──▶ fsttm_ec_source (AEC virtual mic) ──▶ PyAudio ──▶ VAD
                        ▲ reference signal
TTS PCM ──▶ PyAudio (pulse, PULSE_SINK=fsttm_ec_sink) ──▶ speaker
```

TTS plays straight from a PyAudio output stream — no `aplay` subprocess. The
output device and PulseAudio sink are config-driven (`tts.device` / `tts.sink`,
matched by name), so barge-in stops audio mid-chunk by flipping a cancel flag
rather than killing a process. With AEC active, `tts.sink` defaults to
`fsttm_ec_sink` (via `PULSE_SINK`) so the canceller still sees the TTS reference;
set it to e.g. `"Jabra"` to force the headset speaker, or set `device` to a card
name from `aplay -l` to bypass PulseAudio (the device must accept `sample_rate`).

Tested: RMS after cancellation = 0.0 on a 440 Hz sine test tone.

---

## Headless Mode

Test without audio hardware — keyboard input, console output:

```bash
python -m fsttm.headless --config config.sample.yaml [options]

Options:
  --model PATH      Override model path from config
  --prompt FILE     Load system prompt from text file
  --intent a|b|both Enable two-pass grammar intent mode
                      a    = KV rollback (faster, default)
                      b    = translation prompt
                      both = benchmark both approaches
```

Special commands during interactive session:
- `STOP` — simulate barge-in, cancel current generation
- `STATE` — print current FSM state and cost model values

---

## Testing

```bash
source .venv/bin/activate
pytest tests/ --asyncio-mode=auto -v

# 23 tests total:
#   tests/fsttm_test.py             9  FSM state machine (N09-1071 scenarios)
#   tests/test_async_conversation.py 8  async pipeline FSM event mapping
#   tests/test_aec.py               5  PipeWire AEC (requires PipeWire)
#   tests/config_test.py            1  config YAML parsing
```

---

## Models

| Model | Size | Use case |
|-------|------|----------|
| `Phi-3-mini-4k-instruct-Q6_K.gguf` | 2.9 GB | Voice assistant, intent mode (**recommended**) |
| `Llama-3.2-3B-Instruct-Q6_K.gguf` | 2.5 GB | General chat |
| `Phi-4-mini-reasoning-Q6_K.gguf` | 2.9 GB | Math/logic tasks (verbose, less suitable for voice) |

Switch model by editing `gpt.model` in config. The chat template is auto-detected from the filename.

---

## HVAC React Simulator

`hvac-react/` contains a simulated car HVAC panel (React + FastAPI) for testing the intent pipeline without hardware.

### GUI

![HVAC cockpit UI](hvac-react/docs/cockpit.png)

Dual-zone climate (temperature, fan, air direction, per-zone AUTO), global
controls, seat comforts, door/window telemetry, lighting, and a live VHAL
protocol diagnostics console. Every control maps to a VHAL command over the
WebSocket/REST API, and external changes — e.g. from the FSTTM intent
pipeline — are reflected in the UI live.

### Stack

```
hvac-react/
├── backend/      FastAPI + VhalStore (in-memory VHAL property map)
│                 REST: GET /state  GET /config  POST /command  POST /set/{name}
│                 WebSocket /ws    — pushes delta frames to all clients
├── frontend/     React (Vite + Tailwind) dark cockpit dashboard
│                 dual-zone climate, global toggles, seat comfort,
│                 door/window telemetry, lighting, VHAL diagnostics console
└── start.sh      Starts both servers (backend :8000, frontend :5173)
```

### Start

```bash
cd hvac-react

# First run: install frontend deps
cd frontend && npm install && cd ..

bash start.sh
# → http://127.0.0.1:5173   (React UI)
# → http://127.0.0.1:8000   (REST API / WebSocket)
```

### Connect FSTTM

Set `hvac_backend.url` in `config.sample.yaml`:
```yaml
hvac_backend:
    url: "http://127.0.0.1:8000"
```

Every recognized HVAC intent is translated to a VHAL protocol command and POSTed to the backend. The React UI updates in real time via WebSocket delta.

### E2E Tests

```bash
PYTHONPATH=hvac-react/backend pytest tests/test_e2e_hvac.py -v --asyncio-mode=auto
# 9 tests: backend REST + HvacBridge/grammar intent pipeline → backend state
```

---

## References

- [N09-1071 — Finite-State Turn-Taking Model](https://aclanthology.org/N09-1071.pdf)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [piper-tts](https://github.com/rhasspy/piper)
- [PhiCookBook](https://github.com/microsoft/PhiCookBook)
- [Reactive Programming intro](https://gist.github.com/staltz/868e7e9bc2a7b8c1f754)
- [ReSpeaker 4-Mic Array](https://wiki.seeedstudio.com/ReSpeaker_4_Mic_Array_for_Raspberry_Pi/)
