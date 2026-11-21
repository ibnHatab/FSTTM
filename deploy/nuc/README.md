# The NUC's voice stack

Nina on the NUC (`nuc`, i3-10110U, no GPU, Ubuntu 26.04): the ReSpeaker 4 Mic Array, Whisper on the CPU,
Gemini over its API, Paradee (the af_heart voice, in-process), beside Home Assistant (Docker, the only container) and the home's rule engine.
Everything native, as systemd **user** units, from this folder.

| file | what |
|---|---|
| `fsttm-go.yaml` | the Go engine's config (`go/bin/fsttm-go`): cloud LLM, streams door `127.0.0.1:8766`, tools `127.0.0.1:8768`, panel `0.0.0.0:8767`, the array from ALSA (6 ch, keep ch 0), Whisper base.en, the TTS (`tts.paradee`), "hey Nina" with a 30 s idle; its `hass:` block is fsttm-hass's |
| `units/fsttm-go.service` | the Go engine; `FSTTM_TTS_ENGINE=paradee` (over `tts.engine`) |
| `units/fsttm-hass.service` | Home Assistant → ENVIRONMENT, `go/bin/fsttm-hass` |
| `paradee-NOTICE.txt` | the licences of `models/paradee/` (Paradee-8M v1.2 and misaki's lexicons, Apache-2.0), fetched by `install.sh` |
| `audio/` | the array: why its capture is ALSA's and its PipeWire profile output-only (`README.md`), the WirePlumber rule, `respeaker-rec.sh` (six channels with arecord, the engine paused) |
| `build.sh` | whisper.cpp (CPU, `~/metis/whisper.cpp`), `go/bin/fsttm-go` and `paradee-say` (`-tags nollama,norhvoice`; onnxruntime, espeak-ng), `fsttm-hass`, `respeaker`; `build.sh test` the Go tests |
| `install.sh` | packages, the Paradee models, the audio setup, the build; `enable` writes the units and (re)starts the stack |

The rule engine is metis-agent's `metis-home` (branch `home-cgo`, `~/metis/metis-agent-home`, unit
`metis-home.service`); `enable` installs its build to `~/.local/bin`.

```sh
cd ~/repo/FSTTM && git checkout go-nuc
deploy/nuc/install.sh          # packages, models, audio, build; nothing running changes
deploy/nuc/install.sh enable   # the units written, fsttm-go + fsttm-hass (+ metis-home) (re)started
journalctl --user -fu fsttm-go
```

The page: http://nuc:8767/ (LAN and tailnet): the dialog on the left (what the array heard, her replies,
a text line for typed turns), the ENVIRONMENT, REASONING (the rules' conclusions and the LLM's tool calls)
and FLOOR docks on the right.

Measured on the NUC (2026-11-22, the Go engine): Whisper base.en q5_1 with `audio_ctx: 768` and no
fallback, 1.2 s for 2.5 s of speech and 1.5 s for 10 s (2.3 s / 2.9 s at the full 1500); Paradee 15–21×
real time on one thread; Gemini 1.2–1.6 s to the first sentence without a tool call, 2.9 s with one.
