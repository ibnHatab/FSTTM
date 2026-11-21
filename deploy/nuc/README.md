# The NUC's voice stack

Nina on the NUC (`nuc`, i3-10110U, no GPU, Ubuntu 26.04): the ReSpeaker 4 Mic Array, Whisper on the CPU,
Gemini over its API, Piper, beside Home Assistant (Docker, the only container) and the home's rule engine.
Everything native, as systemd **user** units, from this folder.

| file | what |
|---|---|
| `fsttm-go.yaml` | the Go engine's config (`go/bin/fsttm-go`): cloud LLM, streams door `127.0.0.1:8766`, tools `127.0.0.1:8768`, panel `0.0.0.0:8767`, the array from ALSA (6 ch, keep ch 0), Whisper base.en, Piper, "hey Nina" with a 30 s idle |
| `units/fsttm-go.service` | the Go engine (`Conflicts=fsttm.service`) |
| `units/piper-tts.service` | Piper's voice over HTTP on `127.0.0.1:5150` (FSTTM's venv: `python -m piper.http_server`) |
| `units/fsttm.service` | the Python engine (`config.nuc.yaml`; `Conflicts=fsttm-go.service`): the rollback |
| `units/fsttm-hass.service`, `units/fsttm-metis.service` | Home Assistant → ENVIRONMENT; the rule engine (metis-worlds `home`) → REASONING and the tools. Python, either engine |
| `audio/` | the array: why its capture is ALSA's and its PipeWire profile output-only (`README.md`), the WirePlumber rule, `respeaker-rec.sh` (six channels with arecord, the engine paused) |
| `build.sh` | whisper.cpp (CPU, `~/metis/whisper.cpp`) and `go/bin/fsttm-go` (`-tags nollama,norhvoice`); `build.sh test` the Go tests |
| `install.sh` | packages, flask for Piper's server, the audio setup, the units (written, not started), the build; `install.sh swap-go` / `swap-py` switch the engine |

```sh
cd ~/repo/FSTTM && git checkout go-nuc
deploy/nuc/install.sh          # nothing running changes
deploy/nuc/install.sh swap-go  # the Go engine on (and Piper's server), the Python one off
deploy/nuc/install.sh swap-py  # back
journalctl --user -fu fsttm-go
```

The page: http://nuc:8767/ (LAN and tailnet): the dialog on the left (what the array heard, her replies,
a text line for typed turns), the ENVIRONMENT, REASONING (the rules' conclusions and the LLM's tool calls)
and FLOOR docks on the right. The Python engine's panel was `127.0.0.1:8767` (local only).

Measured on the NUC (2026-11-21, the Python engine running beside, load 6-9): Whisper base.en 1.9 s for
2.2 s of speech, the right text (`audio_ctx: 512` on whisper.cpp 1.9.6: 3.7 s, or 1.7 s with a degenerate
"4, 4, 4, ..." without fallback; the Python engine's pywhispercpp gained from it); Piper 2.1 s of speech in 0.3 s;
Gemini 1.8–3.9 s for a reply with one tool call (8.7 s once).
