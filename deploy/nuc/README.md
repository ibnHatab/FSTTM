# The NUC's voice stack

Nina on the NUC (`nuc`, i3-10110U, no GPU, Ubuntu 26.04): the ReSpeaker 4 Mic Array, Whisper on the CPU,
Gemini over its API, Paradee (the af_heart voice, in-process), beside Home Assistant (Docker, the only container) and the home's rule engine.
Everything native, as systemd **user** units, from this folder.

| file | what |
|---|---|
| `fsttm-go.yaml` | the Go engine's config (`go/bin/fsttm-go`): cloud LLM, streams door `127.0.0.1:8766`, tools `127.0.0.1:8768`, panel `0.0.0.0:8767`, the array from ALSA (6 ch, keep ch 0), Whisper base.en, the TTS (`tts.paradee`), "hey Nina" with a 30 s idle; its `hass:` block is fsttm-hass's |
| `units/fsttm-go.service` | the Go engine; `FSTTM_TTS_ENGINE=paradee` (over `tts.engine`) |
| `units/fsttm-hass.service` | Home Assistant → ENVIRONMENT, `go/bin/fsttm-hass` (the port of `fsttm/feeds/hass.py`) |
| `units/legacy/` | the Python engine, its rule-engine feed and Piper's server: `swap-py` / `swap-piper` install them for a rollback, `purge-python` removes them |
| `paradee-NOTICE.txt` | the licences of `models/paradee/` (Paradee-8M v1.2 and misaki's lexicons, Apache-2.0), fetched by `install.sh` |
| `audio/` | the array: why its capture is ALSA's and its PipeWire profile output-only (`README.md`), the WirePlumber rule, `respeaker-rec.sh` (six channels with arecord, the engine paused) |
| `build.sh` | whisper.cpp (CPU, `~/metis/whisper.cpp`), `go/bin/fsttm-go` and `paradee-say` (`-tags nollama,norhvoice`; onnxruntime, espeak-ng), `fsttm-hass`, `respeaker`; `build.sh test` the Go tests |
| `install.sh` | packages, the Paradee models, the audio setup, the build; `swap-go` (the Go stack on), `swap-piper` / `swap-py` (rollbacks), `purge-python [-n]` (the Python legacy off the disk; `-n` lists it) |

The rule engine is metis-agent's `metis-home` (branch `home-cgo`, `~/metis/metis-agent-home`, unit
`metis-home.service`); `swap-go` installs its build (surprises written in Go, no Python) to `~/.local/bin`.

```sh
cd ~/repo/FSTTM && git checkout go-nuc
deploy/nuc/install.sh          # nothing running changes
deploy/nuc/install.sh swap-go  # the Go stack on: Paradee's voice, the Go HA feed; the Python units off
deploy/nuc/install.sh swap-piper   # back to Piper's voice only (until purge-python)
deploy/nuc/install.sh swap-py  # back to the Python engine (until purge-python)
deploy/nuc/install.sh purge-python -n   # what purge-python removes
journalctl --user -fu fsttm-go
```

The page: http://nuc:8767/ (LAN and tailnet): the dialog on the left (what the array heard, her replies,
a text line for typed turns), the ENVIRONMENT, REASONING (the rules' conclusions and the LLM's tool calls)
and FLOOR docks on the right. The Python engine's panel was `127.0.0.1:8767` (local only).

Measured on the NUC (2026-11-21, the Python engine running beside, load 6-9): Whisper base.en 1.9 s for
2.2 s of speech, the right text (`audio_ctx: 512` on whisper.cpp 1.9.6: 3.7 s, or 1.7 s with a degenerate
"4, 4, 4, ..." without fallback; the Python engine's pywhispercpp gained from it); Piper 2.1 s of speech in 0.3 s;
Gemini 1.8–3.9 s for a reply with one tool call (8.7 s once).
