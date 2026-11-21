#!/usr/bin/env bash
# Install the NUC's voice stack from this folder: packages, the models, the audio setup, the user units, the
# build. Nothing is started or swapped unless asked:
#   deploy/nuc/install.sh               install and build; the running engine is left alone
#   deploy/nuc/install.sh enable      the units written and the stack (re)started: fsttm-go (Paradee),
#                                     fsttm-hass, and metis-home from metis-agent's home-cgo build
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
case "${1:-install}" in
  enable|swap-go)
    [ -x "$FSTTM/go/bin/fsttm-hass" ] && [ -f "$MODELS/paradee_int8.onnx" ] || { echo "run install.sh first (the build, the models)" >&2; exit 1; }
    for u in "$HERE"/units/*.service; do install -m 644 "$u" "$UNITS/"; done
    systemctl --user daemon-reload
    if [ -x "$METIS_HOME_SRC" ] && ! cmp -s "$METIS_HOME_SRC" "$HOME/.local/bin/metis-home"; then
      install -m 755 "$METIS_HOME_SRC" "$HOME/.local/bin/metis-home"
      systemctl --user restart metis-home.service
    fi
    systemctl --user enable fsttm-go.service fsttm-hass.service
    systemctl --user restart fsttm-go.service fsttm-hass.service
    systemctl --user --no-pager status fsttm-go.service fsttm-hass.service | grep -E "●|Active"
    exit 0 ;;
  install) ;;
  *) echo "usage: install.sh [install|enable]" >&2; exit 2 ;;
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
echo "installed and built. Start or restart the stack: $0 enable"
