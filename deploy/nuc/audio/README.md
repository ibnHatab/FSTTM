# The NUC's ReSpeaker 4 Mic Array

The array's capture is read **straight from ALSA** by fsttm (`config.nuc.yaml`: `vad.device_name:
ReSpeaker`, `channels: 6`, `channel: 0`): its firmware sends six channels at 16 kHz, ch0 the processed
speech (beamformed, noise-suppressed, echo-cancelled against what plays through its own jack), ch1-4 the
raw mics, ch5 the playback reference. PipeWire keeps only the **playback**: the card's profile is
`output:analog-stereo`, so fsttm's speech plays through the array's jack (the echo canceller's
reference) and the capture is free.

    pactl set-card-profile alsa_card.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00 output:analog-stereo

Why: on the NUC (PipeWire 1.6.2, 2026-11-21) every PipeWire capture of the array was broken: the
pro-audio profile's ch0 transcribed as noise ("[SOUND]", "water splashing") and its raw mics were exact
zeros, through pulse's remap and through a native loopback alike, with or without mmap and larger
periods (`51-respeaker.conf`, kept: harmless for the playback); `arecord -D hw:ArrayUAC10,0 -c 6` on the
same device, the same phrase, was clean ("Hey, Nina, turn on the lap ventilation."), and so is fsttm now.

Keep the array away from the NUC's fan (its 1-2 kHz whine masked speech).
