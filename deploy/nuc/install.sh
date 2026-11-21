#!/usr/bin/env bash
# Install the NUC's voice stack from this folder: packages, the models, the audio setup, the user units, the
# build. Nothing is started or swapped unless asked:
#   deploy/nuc/install.sh               install and build; the running engine is left alone
#   deploy/nuc/install.sh swap-go       the Go stack on: fsttm-go speaking with Paradee (af_heart), the Go
#                                       fsttm-hass, metis-home's Go surprise writer; Python's units off
#   deploy/nuc/install.sh swap-piper    rollback of the voice only: fsttm-go back on Piper (piper-tts.service)
#   deploy/nuc/install.sh swap-py       rollback of everything: the Python engine and feed back on
#   deploy/nuc/install.sh purge-python  after swap-go: the Python legacy removed (the list: purge_list below)
# Run as the desktop user (sudo is asked for the apt packages only).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FSTTM="$(cd "$HERE/../.." && pwd)"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
UNITS="$HOME/.config/systemd/user"
CARD=alsa_card.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00
METIS_HOME_SRC="${METIS_HOME_SRC:-$HOME/metis/metis-agent-home/go/bin/metis-home}"   # branch home-cgo, its build.sh
MODELS="$FSTTM/models/paradee"
PARADEE_HF=https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.2/resolve/main
MISAKI_DATA=https://raw.githubusercontent.com/hexgrad/misaki/main/misaki/data

dropin() {  # dropin UNIT NAME TEXT: a drop-in (TEXT empty: removed)
  local d="$UNITS/$1.d"
  if [ -n "$3" ]; then mkdir -p "$d"; printf '%s\n' "$3" > "$d/$2.conf"; else rm -f "$d/$2.conf"; rmdir "$d" 2>/dev/null || true; fi
}
legacy_units() { for u in "$HERE"/units/legacy/fsttm.service "$HERE"/units/legacy/fsttm-metis.service "$HERE"/units/legacy/piper-tts.service; do
  install -m 644 "$u" "$UNITS/"; done; }

# what purge-python removes, one path or unit per line (printed by `purge-python -n`)
purge_list() {
  cat <<L
unit fsttm.service           the Python engine
unit fsttm-metis.service     the Python rule-engine feed (metis-home, Go, serves the home)
unit piper-tts.service       Piper's HTTP server (FSTTM's venv)
dropin fsttm-hass.service.d/python.conf   swap-py's Python feed, if left
dropin fsttm-go.service.d/piper.conf      swap-piper's, if left
path $FSTTM/.venv                         FSTTM's Python venv (piper-tts, pywhispercpp, the Python engine)
path $FSTTM/models/en_US-lessac-medium.onnx       Piper's voice
path $FSTTM/models/en_US-lessac-medium.onnx.json
path $HOME/repo/acasa/.run                acasa's run state (Gemma 4 E2B via metis-agent's py: logs, sessions, config)
path $HOME/.config/pipewire/pipewire.conf.d/respeaker-asr.conf.off
path $HOME/.config/pipewire/pipewire-pulse.conf.d/respeaker-asr.conf.off-remap
path $HOME/.local/bin/metis-home.prev     swap-go's backup of the Python-calling metis-home
L
}

case "${1:-install}" in
  swap-go)
    [ -x "$FSTTM/go/bin/fsttm-hass" ] && [ -f "$MODELS/paradee_int8.onnx" ] || { echo "run install.sh first (the build, the models)" >&2; exit 1; }
    for u in "$HERE"/units/*.service; do install -m 644 "$u" "$UNITS/"; done
    dropin fsttm-go.service piper ""; dropin fsttm-hass.service python ""
    systemctl --user daemon-reload
    if [ -x "$METIS_HOME_SRC" ] && ! cmp -s "$METIS_HOME_SRC" "$HOME/.local/bin/metis-home"; then
      cp -p "$HOME/.local/bin/metis-home" "$HOME/.local/bin/metis-home.prev"
      install -m 755 "$METIS_HOME_SRC" "$HOME/.local/bin/metis-home"
      systemctl --user restart metis-home.service
    fi
    systemctl --user disable --now fsttm.service fsttm-metis.service piper-tts.service 2>/dev/null || true
    systemctl --user enable fsttm-go.service fsttm-hass.service
    systemctl --user restart fsttm-go.service fsttm-hass.service
    systemctl --user --no-pager status fsttm-go.service fsttm-hass.service | grep -E "●|Active"
    exit 0 ;;
  swap-piper)
    [ -f "$UNITS/piper-tts.service" ] || { echo "Piper is purged: no rollback to it" >&2; exit 1; }
    dropin fsttm-go.service piper "[Service]
Environment=FSTTM_TTS_ENGINE=piper"
    systemctl --user daemon-reload
    systemctl --user start piper-tts.service
    systemctl --user restart fsttm-go.service
    exit 0 ;;
  swap-py)
    [ -x "$FSTTM/.venv/bin/fsttm-hass" ] || { echo "the Python stack is purged: no rollback to it" >&2; exit 1; }
    legacy_units
    dropin fsttm-hass.service python "[Service]
Environment=PYTHONUNBUFFERED=1
ExecStart=
ExecStart=%h/repo/FSTTM/.venv/bin/fsttm-hass --config config.nuc.yaml"
    systemctl --user daemon-reload
    [ -x "$HOME/.local/bin/metis-home.prev" ] && install -m 755 "$HOME/.local/bin/metis-home.prev" "$HOME/.local/bin/metis-home" \
      && systemctl --user restart metis-home.service
    systemctl --user enable fsttm.service
    systemctl --user disable fsttm-go.service
    systemctl --user start fsttm.service                 # Conflicts= stops the Go engine
    systemctl --user restart fsttm-hass.service
    systemctl --user --no-pager status fsttm.service | head -5
    exit 0 ;;
  purge-python)
    if [ "${2:-}" = -n ]; then purge_list; exit 0; fi
    systemctl --user is-active -q fsttm-go.service || { echo "the Go engine is not running: swap-go first" >&2; exit 1; }
    purge_list | while read -r kind what _; do
      case $kind in
        unit)   systemctl --user disable --now "$what" 2>/dev/null || true; rm -f "$UNITS/$what"; echo "removed unit $what" ;;
        dropin) rm -f "$UNITS/$what" && echo "removed $UNITS/$what"; rmdir "$UNITS/$(dirname "$what")" 2>/dev/null || true ;;
        path)   [ -e "$what" ] && rm -rf -- "$what" && echo "removed $what" || true ;;
      esac
    done
    systemctl --user daemon-reload
    echo "kept: metis-agent's py/.venv (the metis plugin's MCP server), Home Assistant's container, the agent-bus venv,"
    echo "      Gemma 4 E2B (~/metis/metis-agent/go/models/gemma4, 2.9 GB: metis-agent's, acasa's run.sh was its only user here)."
    echo "acasa's run.sh is not a unit (./run.sh start|stop): nothing of it runs."
    exit 0 ;;
  install) ;;
  *) echo "usage: install.sh [install|swap-go|swap-piper|swap-py|purge-python [-n]]" >&2; exit 2 ;;
esac

sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential cmake golang-go pkg-config \
  libopus-dev libopusfile-dev libasound2-dev libonnxruntime-dev libespeak-ng-dev pulseaudio-utils sox >/dev/null

# Paradee (Apache-2.0) and misaki's US lexicons (Apache-2.0): models/paradee/, not in git
mkdir -p "$MODELS"
for f in onnx/paradee_int8.onnx config.json; do
  [ -s "$MODELS/$(basename $f)" ] || curl -fsSL -o "$MODELS/$(basename $f)" "$PARADEE_HF/$f"
done
for f in us_gold.json us_silver.json; do [ -s "$MODELS/$f" ] || curl -fsSL -o "$MODELS/$f" "$MISAKI_DATA/$f"; done
install -m 644 "$HERE/paradee-NOTICE.txt" "$MODELS/NOTICE.txt"

# the audio: the array's capture is ALSA's (the engine's), PipeWire keeps its playback
mkdir -p "$HOME/.config/wireplumber/wireplumber.conf.d" "$HOME/bin"
install -m 644 "$HERE/audio/51-respeaker.conf" "$HOME/.config/wireplumber/wireplumber.conf.d/51-respeaker.conf"
install -m 755 "$HERE/audio/respeaker-rec.sh" "$HOME/bin/respeaker-rec.sh"
# the array's USB node for the plugdev group (fsttm-go sets its XVF-3000 parameters at start)
if ! cmp -s "$HERE/audio/99-respeaker.rules" /etc/udev/rules.d/99-respeaker.rules; then
  sudo install -m 644 "$HERE/audio/99-respeaker.rules" /etc/udev/rules.d/99-respeaker.rules
  sudo udevadm control --reload && sudo udevadm trigger --attr-match=idVendor=2886
fi
pactl set-card-profile "$CARD" output:analog-stereo 2>/dev/null || echo "(the ReSpeaker is not plugged in: set its profile later)"

# the units are written by swap-go (their new ExecStart would take effect at the next restart otherwise)
"$HERE/build.sh"
echo "installed. The engine running now: $(systemctl --user is-active fsttm-go >/dev/null 2>&1 && echo Go || echo Python)."
echo "Swap: $0 swap-go   (back: $0 swap-piper, $0 swap-py; then $0 purge-python)"
