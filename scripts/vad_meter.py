#!/usr/bin/env python3
"""Live RMS + webrtcvad verdict on the EXACT path fsttm uses (pulse @16k, 20ms
frames, aggressiveness 3). Speak and watch: SPEECH should light up when you talk.
Usage: python scripts/vad_meter.py [aggressiveness 0-3]"""
import sys, time, audioop, pyaudio, webrtcvad

AGG = int(sys.argv[1]) if len(sys.argv) > 1 else 3
RATE = 16000
CHUNK = 320  # 20ms @16k = 640 bytes, valid webrtcvad frame
vad = webrtcvad.Vad(AGG)

pa = pyaudio.PyAudio()
idx = next((i for i in range(pa.get_device_count())
            if pa.get_device_info_by_index(i)["maxInputChannels"]>0
            and "pulse" in pa.get_device_info_by_index(i)["name"].lower()), None)
if idx is None:
    print("no pulse device"); sys.exit(1)
print("device idx=%d (pulse), aggressiveness=%d. Speak now (Ctrl-C to stop):" % (idx, AGG))
s = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
            input_device_index=idx, frames_per_buffer=CHUNK)
peak=1; speech_frames=0; total=0
try:
    while True:
        d = s.read(CHUNK, exception_on_overflow=False)
        rms = audioop.rms(d, 2); peak=max(peak,rms)
        sp = vad.is_speech(d, RATE); total+=1; speech_frames += 1 if sp else 0
        bar = int(30*rms/max(peak,500))
        sys.stdout.write("\rRMS %5d |%-30s| %s   speech%%=%3d" %
                         (rms, "#"*bar, "SPEECH" if sp else "  ----", 100*speech_frames//max(total,1)))
        sys.stdout.flush()
except KeyboardInterrupt:
    print()
finally:
    s.stop_stream(); s.close(); pa.terminate()
