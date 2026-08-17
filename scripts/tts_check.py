#!/usr/bin/env python3
"""Test audio OUTPUT + TTS separately from the voice pipeline.

Stages (each independent):
  1. tone   — play a 440Hz beep to confirm the speaker works
  2. piper  — try piper PiperVoice (needs piper_phonemize)
  3. espeak — try espeak-ng直接 synthesis (needs apt espeak-ng)

Usage: python scripts/tts_check.py [tone|piper|espeak|all] ["text to speak"]
"""
import sys, subprocess, math, struct, wave, tempfile, os

STAGE = sys.argv[1] if len(sys.argv) > 1 else "all"
TEXT  = sys.argv[2] if len(sys.argv) > 2 else "Increasing the temperature slightly."
MODEL = os.path.expanduser("~/repo/vox/fsttm/models/en_US-lessac-medium.onnx")

def play_wav(path):
    # play via PulseAudio default sink (the Jabra)
    subprocess.run(["aplay", "-q", path], check=False)

def stage_tone():
    print("[tone] playing 440Hz beep on default sink (Jabra)...")
    rate=22050; secs=1.0; freq=440
    f=tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    w=wave.open(f.name,"w"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
    for i in range(int(rate*secs)):
        v=int(16000*math.sin(2*math.pi*freq*i/rate)); w.writeframes(struct.pack("<h",v))
    w.close(); play_wav(f.name); os.unlink(f.name)
    print("[tone] done — did you hear a beep?")

def stage_piper():
    print("[piper] trying PiperVoice (%s)..." % os.path.basename(MODEL))
    try:
        from piper import PiperVoice
    except Exception as e:
        print("[piper] UNAVAILABLE:", e); return
    voice=PiperVoice.load(MODEL)
    f=tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(f.name,"w") as wf:
        voice.synthesize(TEXT, wf)
    play_wav(f.name); os.unlink(f.name)
    print("[piper] spoke:", repr(TEXT))

def stage_espeak():
    print("[espeak] trying espeak-ng...")
    from shutil import which
    if not which("espeak-ng"):
        print("[espeak] UNAVAILABLE: install with: sudo apt-get install -y espeak-ng"); return
    f=tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    subprocess.run(["espeak-ng","-w",f.name,TEXT], check=False)
    play_wav(f.name); os.unlink(f.name)
    print("[espeak] spoke:", repr(TEXT))

def stage_rhvoice():
    """fsttm.tts RhvoiceBackend: probe → sample rate from the WAV header,
    synthesize, play through aplay."""
    print("[rhvoice] trying fsttm.tts rhvoice backend...")
    try:
        from fsttm.tts.rhvoice_backend import RhvoiceBackend
        b = RhvoiceBackend()
        b.load({})
    except Exception as e:
        print("[rhvoice] UNAVAILABLE:", e); return
    pcm = b.synthesize(TEXT)
    dur = len(pcm) / (b.sample_rate * 2)
    print(f"[rhvoice] rate={b.sample_rate} Hz  pcm={len(pcm)}B  dur={dur:.2f}s")
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    w = wave.open(f.name, "w"); w.setnchannels(1); w.setsampwidth(2)
    w.setframerate(b.sample_rate); w.writeframes(pcm); w.close()
    play_wav(f.name); os.unlink(f.name)
    print("[rhvoice] spoke:", repr(TEXT))

stages={"tone":stage_tone,"piper":stage_piper,"espeak":stage_espeak,
        "rhvoice":stage_rhvoice}
if STAGE=="all":
    for s in ["tone","piper","espeak","rhvoice"]: stages[s]()
elif STAGE in stages:
    stages[STAGE]()
else:
    print("unknown stage:", STAGE, "(use tone|piper|espeak|rhvoice|all)")
