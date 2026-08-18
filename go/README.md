# fsttm-go

The FSTTM spoken-dialog engine in Go: a message-passing pipeline over
whisper.cpp, llama.cpp and Linux RHVoice, built for the Orin deployment where
**idle must mean idle** — every goroutine blocks on a channel receive; a
silent room costs one native VAD call per 20 ms frame, the GPU schedules no
kernels between utterances, and TTS is a subprocess that exists only while
speaking.

```
capture ──frames──▶ vad ──events──▶ orchestrator ──speak──▶ rhvoice|aplay
(malgo/pulse)   (webrtcvad)      (owns the N09-1071 FSM;
                                  whisper + llama called inline, bursty)
```

## Bindings audit (why this shape)

| Component | Binding | Verdict |
|---|---|---|
| whisper.cpp | official `bindings/go` (in-tree, API-current) | **on par**: 42 ms vs pywhispercpp's 44 ms for 11 s audio, same CUDA lib |
| llama.cpp | go-llama.cpp | **rejected**: frozen 2024-03, pinned llama.cpp predates Phi-3, no sampler-chain / `llama_memory_*` API |
| llama.cpp | `internal/llm` — ~300-line cgo shim over the current C API | **on par**: two-pass JSON byte-identical to `fsttm/two_pass.py`, latency equal (188–523 ms/turn, tail eval 1–6 ms with KV-prefix reuse) |
| TTS | **librhvoice in-process** (default) — `play_speech` callback per PCM chunk | mid-synthesis abort (callback returns 0) + EXACT fraction heard; `RHVoice-client | aplay` subprocess kept as fallback (`tts.engine: subprocess`) |

Measured idle (30 s, mic live, silent room, this dev box): **1.2 % CPU,
0 % GPU util, 2 threads** (Python engine: 1.6 % CPU, 7–11 threads,
+1.1 GB GPU memory). On the Orin's slower cores the gap is larger.

## Build

Requires Go ≥1.23 and sibling checkouts of llama.cpp + whisper.cpp, each
built with CMake (`-DGGML_CUDA=ON` where wanted):

```bash
LLAMA_DIR=~/repo/vox/llama.cpp WHISPER_DIR=~/repo/vox/whisper.cpp ./build.sh
# binaries land in go/bin/ with rpath embedded — no LD_LIBRARY_PATH needed
```

The intent grammar/prompt are generated from the Python dog domain
(single source of truth):

```bash
cd .. && source .venv/bin/activate && python - <<'PY'
import json
from llama_cpp.llama_grammar import json_schema_to_gbnf
from fsttm_dog.provider import DOG_SCHEMA, PROVIDER
open('go/grammar/dog.gbnf', 'w').write(json_schema_to_gbnf(json.dumps(DOG_SCHEMA)))
open('go/grammar/dog-prompt.txt', 'w').write(PROVIDER.build_prompt())
PY
```

## Run

```bash
./bin/fsttm-go -config config.dog.yaml            # voice: mic → intent → voice
./bin/fsttm-go -config config.dog.yaml -headless  # stdin text turns
./bin/llmbench -model <gguf> -prompt grammar/dog-prompt.txt \
    -gbnf grammar/dog.gbnf "go to the chair next to the window"
```

- `wake_word: rex` — asleep until heard, then stays awake.
- `barge_in: false` (default) — half-duplex: mic events are ignored while the
  system speaks, and utterances whose audio overlaps our own playback are
  dropped as echo. Enable barge-in ONLY with AEC in the audio path (AEC
  virtual mic / USB conference speakerphone). A confirmed barge-in cuts
  synthesis mid-stream (librhvoice callback abort) and logs the exact
  fraction heard — the N09-1071 transition-8 semantics.
- `n_gpu_layers: 0` — CPU-only fallback profile.
- **System-initiated narration**: `Engine.Announce("Battery low.")` (or
  `kill -USR1 <pid>` for a live test) — the system takes an unclaimed floor
  (paper transition 5, cost 0) and speaks; while the user holds the floor
  the announcement is deferred, never cutting them. Works from cold boot:
  the FSM initializes in FREEu (nobody claims the floor at start).

## Barge-in: the full solution

`barge_in: "confirm"` is the production mode for the robot — the Python
engine's soft-duck design completed with voice imprinting:

1. **AEC + noise suppression** (`aec:` block, port of the Python
   EchoCancelSession): PulseAudio `module-echo-cancel` with webrtc — echo
   cancellation AND built-in noise suppression — plus an optional **RNNoise
   LADSPA chain** on the AEC output for the twelve-motor noise floor. The EC
   source/sink become the PulseAudio defaults; restored on exit.
2. **Tentative, not trigger-happy**: a VAD onset during narration cuts
   nothing — the overlapping utterance is captured and transcribed while
   the robot keeps talking.
3. **Confirmation**: only a real transcript (whisper noise/parasite filters)
   **in the imprinted owner's voice** cuts the narration — librhvoice aborts
   synthesis mid-stream, the exact fraction heard is logged, and the floor
   walks SYSTEM→BOTHs→USER (paper transitions 7+8). AEC residue rarely
   survives STT; whatever does can never match the owner imprint, because
   the robot's own TTS voice is not the owner.

`"vad"` (cut on bare onset) remains for setups with hardware AEC;
`"off"` (half-duplex) for no echo control at all.

## Owner imprinting (voice_id)

The robot listens to ONE person. A speaker-embedding model (sherpa-onnx,
CPU, no torch) scores every gated utterance against the enrolled profile:

```bash
./bin/fsttm-imprint -model models/speaker/wespeaker_en_voxceleb_CAM++.onnx \
    -profile models/speaker/owner.json -owner you -record 3
```

- `voice_id.gate: wake` — imprinting: the wake word ("hello rex", "hey
  rex", … — the attention layer ports the Python wake/sleep machine, incl.
  the rule that only a wake-prefixed utterance can put the robot to sleep)
  only works in the owner's voice; `always` gates every command.
- Barge-in confirmation always requires the owner when a profile exists.
- **Ownership transfer = re-run enrollment** — the new profile atomically
  replaces the old; nothing else changes. Proven by e2e tests that enroll
  one RHVoice voice as owner, reject another, then transfer between them.

## Semantic verification

The FSM is machine-checked against the N09-1071 paper (canonical 12-
transition table, structural constraints, the six §3.1 phenomena, Table-1
action availability) in `internal/fsm/paper_test.go` — audit narrative in
`../docs/fsttm-n09-1071-verification.md`. The concurrency invariants have
e2e tests:

- **context rollback** (`internal/llm/rollback_test.go`, model-gated):
  byte-identical JSON for the same utterance across interleaved turns and
  prefix switches — the KV rewind leaves no trace.
- **output cutting** (`internal/tts/librhvoice_test.go`, voice-gated):
  Speak blocks until audio truly finished; Cancel cuts mid-utterance within
  one device period and reports the exact fraction heard.
- **turn behavior** (`internal/pipeline/e2e_test.go`, fake drivers): full
  gap-transition traces, barge-in walks SYSTEM→BOTHs→USER and cuts the
  speaker, grace-window suppression, half-duplex echo drop, deterministic
  QUERY answers, JSON-parrot never spoken, wake word.

Run: `./build.sh test ./...`

## Orin notes

- Build llama.cpp/whisper.cpp on-device (CUDA sm_87, or CPU-only), then
  `./build.sh` — pure cgo, no Python at runtime.
- RHVoice: build from source (no arm64 apt package); the engine shells out to
  `RHVoice-client`, so binaries + one English voice suffice.
- The robot seams live in `internal/intent` (`ActionBackend`,
  `SemanticMemory`, `NavigationBackend`) — implement them against the Go2
  action interface, the DINOv3 semantic map, and nav2; the logging stubs show
  the contract. See ../contrib/dog/spec.md and ../docs/orin-deployment.md.
