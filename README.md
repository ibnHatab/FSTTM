# Finite-State Turn-Taking Machine (FSTTM)

A spoken dialog **engine** implementing the N09-1071 turn-taking model with a
fully local LLM+STT+TTS pipeline, PipeWire acoustic echo cancellation,
grammar-constrained intent generation, and **pluggable intent domains** —
the same engine drives a car HVAC cockpit today and a Unitree Go2 robot dog
next.

**Paper:** [A Finite-State Turn-Taking Model for Spoken Dialog Systems](https://aclanthology.org/N09-1071.pdf) — Raux & Eskenazi, NAACL 2009

---

## Repo layout — engine + contrib domains

One git repo, three pip distributions:

| Package | Path | What it is |
|---|---|---|
| **fsttm** | `fsttm/` | the engine: FSM, VAD/STT/LLM/TTS drivers, AEC, attention, narrator, TUI, RAG store, plugin seams |
| **fsttm-hvac** | `contrib/hvac/` | HVAC/vehicle intent domain + `hvac-react/` simulated cockpit (FastAPI VHAL + React UI) |
| **fsttm-dog** | `contrib/dog/` | Go2 robot-dog intent language ([spec](contrib/dog/spec.md)); typed nav2/DINOv3 seams, logging stubs, no ROS2 deps |

The engine knows **no domain**: intent schemas, prompts, command translation
and backend dispatch come from the active provider, selected per deployment
via `system.domain` and registered through the `fsttm.domains` entry-point
group. TTS voices (`fsttm.tts_backends`: piper, rhvoice) and voice filters
(`fsttm.voice_filters`: speaker) plug in the same way.

```
Microphone
    │
    ▼
PipeWire AEC (module-echo-cancel, WebRTC)
    │
    ▼
VAD (webrtcvad, utterance segmentation)
    │
    ▼
Voice filter (optional: speaker verification — "only my voice")
    │
    ▼
STT (whisper.cpp, CUDA or CPU)
    │
    ▼
FSM Gate (N09-1071 6-state turn-taking)
    │
    ├─[chat mode]───► LLM streaming → narrator
    │
    └─[intent mode]─► two-pass grammar generation (KV rollback)
                        │  IntentResult(intent_json, tts_text)
                        ▼
                      DOMAIN PROVIDER (hvac / dog / …)
                        │  translate() → command dicts
                        ▼
                      DOMAIN DISPATCHER → backend (VHAL REST / Go2 actions)
    ▼
TTS backend (piper ONNX | RHVoice) → PyAudio → speaker
```

---

## Quick Start

### Requirements

- Ubuntu 22.04+/24.04, Python **3.10+**
- NVIDIA GPU with CUDA (CPU-only also works, slower)
- PipeWire 1.0+ (for AEC); any USB microphone

### Install

```bash
git clone git@cicd.skyway.porsche.com:PG50/fsttm.git
cd fsttm

python3 -m venv .venv
source .venv/bin/activate

# engine + extras + both domains (dev box)
pip install -e ".[piper,tui,voicefilter,test]" -e contrib/hvac -e contrib/dog

# Build llama-cpp-python from source with CUDA (required for recent models)
PATH=/usr/local/cuda/bin:$PATH \
CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native" \
pip install --force-reinstall --no-cache-dir \
  git+https://github.com/abetlen/llama-cpp-python.git

# whisper.cpp + pywhispercpp with CUDA + the STT model
scripts/build-whisper.sh
```

### Download Models

> **Models are not stored in git.** Download into `models/`.

| Model | Size | Use case |
|-------|------|----------|
| **Phi-3-mini-4k-instruct Q4_K_M / Q6_K** ← recommended | 2.4–2.9 GB | intent mode |
| Llama-3.2-3B-Instruct Q6_K | 2.5 GB | general chat |
| `en_US-lessac-medium.onnx` (+ .json) | 61 MB | piper voice |
| `ggml-base.en-q5_1.bin` | 60 MB | STT (fetched by build-whisper.sh) |

```bash
mkdir -p models
wget -O models/Phi-3-mini-4k-instruct-Q4_K_M.gguf \
  "https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF/resolve/main/Phi-3-mini-4k-instruct-Q4_K_M.gguf"
wget -O models/en_US-lessac-medium.onnx \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
wget -O models/en_US-lessac-medium.onnx.json \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
```

### Run

```bash
# Full pipeline (voice in → LLM → voice out); --tui for the 3-panel interface
fsttm --config config.sample.yaml
fsttm --config config.taycann.yaml --tui

# Headless (keyboard input, no audio hardware)
fsttm-headless --config config.sample.yaml
fsttm-headless --config config.taycann.yaml --intent --no-aec        # hvac
fsttm-headless --config contrib/dog/config.dog.sample.yaml --intent --no-aec  # dog
```

---

## Configuration (config.sample.yaml)

```yaml
system:
    name: "Nina"
    intent_mode: true          # two-pass grammar intents vs plain chat
    domain: "hvac"             # fsttm.domains entry point: hvac | dog | null
    intent_domains: null       # provider sub-domains (hvac: climate,lights,body,manual)
    prompt_variant: "few-shot" # one-shot | few-shot | few-shot-extra
    attention: false           # wake-word layer ("Nina, …")
domains:                       # per-domain block, passed to the provider
    hvac:
        backend_url: "http://127.0.0.1:8000"
        manual: {enabled: true, store: models/taycan-manual.npz,
                 embed: models/nomic-embed-text-v1.5.Q4_K_M.gguf}
tts:
    backend: piper             # piper | rhvoice (fsttm.tts_backends)
    piper:   {model: models/en_US-lessac-medium.onnx, sample_rate: 22050}
    rhvoice: {voice: SLT, rate: 0.3, volume: -0.1}
voice_filter:                  # optional "only my voice" (speaker verification)
    enabled: false
    model: models/speaker/wespeaker_en_voxceleb_resnet34.onnx
    profiles: models/speaker/profiles.npz
    threshold: 0.45
    mode: shadow               # tune scores first, then enforce
```

Pre-0.2 configs (hvac_intent / hvac_backend / system.manual*) still boot —
`normalize_config` maps them with deprecation warnings.

---

## Intent Mode (grammar-constrained generation)

Two-pass generation with llama-cpp-python's low-level API — the prompt is
evaluated **once**, both passes share the KV cache (150–400 ms/turn on GPU):

```
model.eval(base_prompt_tokens)    # Pass 1: fill KV cache
state = model.save_state()
model.generate([], grammar=g)     # sample JSON (guaranteed valid vs the
                                  #   ACTIVE DOMAIN's schema)
model.load_state(state)           # Pass 2: rollback
model.eval(tts_cue_tokens)        # "[json]\nSpoken response:"
model.generate([], grammar=None)  # natural speech ack
```

The domain provider owns the schema. HVAC is a flat command format
(`{"intent":"WARMER","area":0,"delta":1}`); the dog domain is a **nested
grounded format** (open-vocabulary targets + spatial relations):

```json
{"intent": "FIND",
 "target": {"type": "OBJECT", "description": "chair",
            "attributes": {"color": "red"}},
 "constraints": [{"relation": "NEAR",
                  "reference": {"type": "OBJECT", "description": "window"}}]}
```

---

## Writing a domain (plugin how-to)

A domain is a pip package with one entry point:

```toml
[project.entry-points."fsttm.domains"]
mydomain = "my_pkg.provider:PROVIDER"
```

`PROVIDER` implements `fsttm.domain.DomainProvider`:

| Hook | Job |
|---|---|
| `build_schema/build_grammar` | the complete intent JSON schema (nested OK) |
| `build_prompt(enabled, variant)` | the system prompt teaching it |
| `translate(intent)` | intent JSON → command dicts (`"cmd"` discriminator) |
| `meta_intent(intent)` | map onto engine behaviours (TIME/DATE/CHITCHAT/UNKNOWN) |
| `make_dispatcher(ctx)` | a `DomainDispatcher`: `handle()` side effects, `local_answer()` deterministic answers, `interpolate()` placeholders, `defer_narration()` e.g. RAG |

The engine never sees intent names — it dispatches on `translate()` output
and `meta_intent()`. See `contrib/hvac/fsttm_hvac/` (modular, flat schema,
REST backend, manual RAG) and `contrib/dog/fsttm_dog/` (monolithic, nested
schema, typed robot seams) as the two reference shapes.

---

## Voice filter — "only my voice"

Utterance-level speaker verification between VAD and STT: a sherpa-onnx
speaker embedding (CPU, no torch) is cosine-matched against enrolled
profiles; non-matching utterances are dropped before transcription — and
because barge-in confirmation requires a transcript, strangers can neither
command nor interrupt the narration.

```bash
fsttm-enroll --model models/speaker/wespeaker.onnx \
             --profiles models/speaker/profiles.npz \
             --name axadmin --record 3
fsttm-enroll --model ... --profiles ... --test-mic   # cosine scores per profile
```

Run with `voice_filter.mode: shadow` first (scores logged, nothing dropped),
then `enforce`.

---

## Turn-Taking FSM

Six-state model from N09-1071. States represent floor *ownership*
(intent/obligation), not surface speech/silence.

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

**Cost model** (N09-1071 §3.2, `fsttm/fsttm.py`): `C_u = 5000`,
`C_gap(τ) = 1×τ`, `C_o(τ) = exp((τ+100)/1000)`.

**Wire-up** (`fsttm/pipeline wiring in server.py`): VAD speech start →
`user_action('G')`; utterance end → `user_action('R')`; response ready →
`system_action('G')`; playback done → `system_action('R')`; barge-in during
TTS → tentative floor flip, confirmed only by a real transcript (600 ms
grace + auto-resume on false alarms).

---

## Acoustic Echo Cancellation

PipeWire `module-echo-cancel` (WebRTC) loaded at app start via
`EchoCancelSession` (`fsttm/aec.py`); TTS plays into `fsttm_ec_sink` so the
canceller sees the reference. `FSTTM_NO_AEC=1` captures the raw mic.

---

## HVAC domain (contrib/hvac)

The simulated car cockpit and the intent domain that drives it:

```bash
cd contrib/hvac/hvac-react
cd frontend && npm install && cd ..    # first run
bash start.sh                          # backend :8000 + React UI :5173
```

Voice commands → intent JSON → PROTOCOL.md commands → VHAL store → live UI
update via WebSocket. STATUS questions are answered from **real** backend
telemetry; manual questions run RAG over an ingested PDF
(`python -m fsttm.rag.ingest manual.pdf --embed <gguf> --out <npz>`).

E2E: `pytest contrib/hvac/tests/test_e2e_hvac.py` (spawns the backend).

## Dog domain (contrib/dog)

The Go2 natural-language intent & semantic-navigation language — see
[contrib/dog/spec.md](contrib/dog/spec.md) (GOAT/HomeRobot-style: the LLM
produces an intent; semantic perception resolves it into a spatial goal;
classical navigation executes it). Ships typed seams
(`ActionBackend`/`SemanticMemory`/`NavigationBackend`) with logging stubs;
nav2/DINOv3 backends plug in on the robot. Deployment notes:
[docs/orin-deployment.md](docs/orin-deployment.md).

---

## Testing

```bash
source .venv/bin/activate
pytest tests/ -m "not hardware"        # engine (fast; hardware = live PipeWire)
pytest contrib/hvac/tests              # hvac domain (some tests need models/)
pytest contrib/dog/tests               # dog domain (headless)
```

---

## References

- [N09-1071 — Finite-State Turn-Taking Model](https://aclanthology.org/N09-1071.pdf)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) ·
  [whisper.cpp](https://github.com/ggerganov/whisper.cpp) ·
  [piper-tts](https://github.com/rhasspy/piper) ·
  [RHVoice](https://github.com/RHVoice/RHVoice) ·
  [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- [GOAT — GO to Any Thing](https://theophilegervet.github.io/projects/goat/) ·
  [HomeRobot](https://github.com/facebookresearch/home-robot)
