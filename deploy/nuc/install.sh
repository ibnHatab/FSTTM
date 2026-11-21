#!/usr/bin/env bash
# Install the NUC's voice stack from this folder: packages, the Piper HTTP server's dependency, the audio
# setup, the user units, the build. Nothing is started or swapped unless asked:
#   deploy/nuc/install.sh             install and build; the running engine is left alone
#   deploy/nuc/install.sh swap-go     the Go engine on (fsttm-go + piper-tts), the Python one off
#   deploy/nuc/install.sh swap-py     back: the Python engine on, the Go one off
# Run as the desktop user (sudo is asked for the apt packages only).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FSTTM="$(cd "$HERE/../.." && pwd)"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
UNITS="$HOME/.config/systemd/user"
CARD=alsa_card.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00

case "${1:-install}" in
  swap-go)
    systemctl --user enable piper-tts.service fsttm-go.service
    systemctl --user disable fsttm.service
    systemctl --user start piper-tts.service
    systemctl --user start fsttm-go.service              # Conflicts= stops the Python engine
    systemctl --user restart fsttm-hass.service fsttm-metis.service   # their door is the new engine's
    systemctl --user --no-pager status fsttm-go.service | head -5
    exit 0 ;;
  swap-py)
    systemctl --user enable fsttm.service
    systemctl --user disable fsttm-go.service piper-tts.service
    systemctl --user start fsttm.service                 # Conflicts= stops the Go engine
    systemctl --user stop piper-tts.service
    systemctl --user restart fsttm-hass.service fsttm-metis.service
    systemctl --user --no-pager status fsttm.service | head -5
    exit 0 ;;
  install) ;;
  *) echo "usage: install.sh [install|swap-go|swap-py]" >&2; exit 2 ;;
esac

sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential cmake golang-go pkg-config \
  libopus-dev libopusfile-dev libasound2-dev pulseaudio-utils sox >/dev/null
# the Piper HTTP server runs in FSTTM's venv (piper-tts is there; it needs flask)
"$HOME/.local/bin/uv" pip install -q --python "$FSTTM/.venv/bin/python" flask 2>/dev/null \
  || "$FSTTM/.venv/bin/python" -m pip install -q flask

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

# the units: written, daemon reloaded; enabling/starting is swap-go / swap-py's
mkdir -p "$UNITS"
for u in "$HERE"/units/*.service; do install -m 644 "$u" "$UNITS/"; done
systemctl --user daemon-reload

"$HERE/build.sh"
echo "installed. The engine running now: $(systemctl --user is-active fsttm-go >/dev/null 2>&1 && echo Go || echo Python)."
echo "Swap: $0 swap-go   (back: $0 swap-py)"
