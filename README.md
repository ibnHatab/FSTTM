# Finite-State Turn-Taking Machine (FSTTM)

A spoken dialog **engine** in Go implementing the N09-1071 turn-taking model: the microphone stream is
split into utterances (WebRTC VAD), each utterance is transcribed by whisper.cpp **in-process** (cgo,
the official Go bindings), the floor is kept by the N09-1071 FSM, the reply comes from an LLM (a
local llama.cpp two-pass intent model, or any OpenAI-compatible chat API such as Gemini), and it is
spoken in-process (Paradee / Kokoro af_heart over onnxruntime, or RHVoice), with barge-in.

**Paper:** [A Finite-State Turn-Taking Model for Spoken Dialog Systems](https://aclanthology.org/N09-1071.pdf) — Raux & Eskenazi, NAACL 2009

The engine and everything about it: [`go/README.md`](go/README.md). The Python implementation that
preceded it is retired (it lives in git history before this commit).

| where | what |
|---|---|
| `go/cmd/fsttm-go` | the engine: capture, VAD, Whisper, the FSM, the LLM, TTS, the panel |
| `go/cmd/fsttm-hass` | Home Assistant → the ENVIRONMENT stream |
| `go/cmd/respeaker` | the ReSpeaker 4 Mic Array's XVF-3000 parameters over USB |
| `go/cmd/paradee-say`, `go2say`, `llmbench`, `fsttm-imprint` | the voice from the command line, the Go2's speaker, LLM benchmarks, the voice-filter imprint |
| `go/pkg/streams`, `go/pkg/hometools` | the stream door and the rule engine's tool port, for other services to import |
| `go/docs/dog-spec.md` | the Go2 robot-dog intent language |
| `deploy/nuc/` | the NUC home assistant: build, install, systemd user units, audio setup |
| `docs/` | the FSM's verification against N09-1071; the Orin / Go2 deployment |

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

**Cost model** (N09-1071 §3.2, `go/pkg/fsm`): `C_u = 5000`,
`C_gap(τ) = 1×τ`, `C_o(τ) = exp((τ+100)/1000)`.

**Wire-up** (`go/internal/pipeline`): VAD speech start →
`user_action('G')`; utterance end → `user_action('R')`; response ready →
`system_action('G')`; playback done → `system_action('R')`; barge-in during
TTS → tentative floor flip, confirmed only by a real transcript (600 ms
grace + auto-resume on false alarms).

---

## References

- [N09-1071 — Finite-State Turn-Taking Model](https://aclanthology.org/N09-1071.pdf)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) ·
  [piper-tts](https://github.com/rhasspy/piper) ·
  [RHVoice](https://github.com/RHVoice/RHVoice) ·
  [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- [GOAT — GO to Any Thing](https://theophilegervet.github.io/projects/goat/) ·
  [HomeRobot](https://github.com/facebookresearch/home-robot)
