#!/usr/bin/env bash

pactl unload-module module-echo-cancel
pactl load-module module-echo-cancel aec_method=webrtc source_name=echocancel sink_name=echocancel1
pacmd set-default-source echocancel
pacmd set-default-sink echocancel1

pactl list sources short

echo "Done. In pavucontroll select 'input sources: Build-in Audio (echo canceled)"

#1       alsa_output.pci-0000_00_1f.3.analog-stereo.monitor      module-alsa-card.c      s16le 2ch 48000Hz       RUNNING
#2       alsa_input.pci-0000_00_1f.3.analog-stereo       module-alsa-card.c      s16le 2ch 48000Hz       RUNNING
#27      alsa_output.pci-0000_01_00.1.hdmi-stereo.monitor        module-alsa-card.c      s16le 2ch 44100Hz       RUNNING

#Used to perform acoustic echo cancellation between a designated sink and source
#load-module module-echo-cancel source_name=alsa_input.pci-0000_00_1f.3.analog-stereo source_properties=device.description=MicrofoneNoise
#set-default-source MicrofoneNoise