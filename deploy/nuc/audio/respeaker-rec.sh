#!/usr/bin/env bash
# Record the ReSpeaker straight from ALSA (all six channels), fsttm and PipeWire out of the way meanwhile.
#   respeaker-rec.sh [seconds] [out.wav]      default 6 s, respeaker6.wav
# ch0 = processed speech (ASR), ch1..4 = raw mics, ch5 = playback reference.
# Split with: sox respeaker6.wav asr.wav remix 1   (remix 2..5 the mics, remix 6 the playback)
set -u
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
SECS=${1:-6}; OUT=${2:-respeaker6.wav}
C=alsa_card.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00
HW=hw:ArrayUAC10,0
VOICE=$(systemctl --user is-active fsttm-go >/dev/null 2>&1 && echo fsttm-go || echo fsttm)   # the engine running now
back() {
  pactl set-card-profile $C output:analog-stereo      # PipeWire keeps only the playback; the engine reads ALSA
  systemctl --user start $VOICE
  echo "PipeWire has the array again; $VOICE started"
}
trap back EXIT
systemctl --user stop $VOICE                         # the engine reads the capture straight from ALSA
pactl set-card-profile $C off || { echo "pactl failed (XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR?)"; exit 1; }
for _ in $(seq 1 20); do fuser /dev/snd/pcmC*D0c 2>/dev/null | grep -q . || break; sleep 0.2; done
echo "recording $SECS s from $HW -> $OUT (speak now)"
arecord -D $HW -c 6 -r 16000 -f S16_LE -d "$SECS" "$OUT"
