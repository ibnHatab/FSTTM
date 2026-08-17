"""fsttm-enroll — build and test voice profiles for the speaker filter.

Enroll from recordings:
    fsttm-enroll --model wespeaker.onnx --profiles profiles.npz --name alice \
        --record 3 --seconds 5
Enroll from wav files (16 kHz mono s16le):
    fsttm-enroll --model wespeaker.onnx --profiles profiles.npz --name alice \
        --wav a.wav b.wav c.wav
Score a wav (or the mic) against all enrolled profiles (threshold tuning):
    fsttm-enroll --model wespeaker.onnx --profiles profiles.npz --test x.wav
    fsttm-enroll --model wespeaker.onnx --profiles profiles.npz --test-mic
"""
from __future__ import annotations

import argparse
import os
import sys
import wave

import numpy as np

from fsttm.voicefilter.speaker import embed, make_extractor

_RATE = 16000


def _read_wav(path: str) -> bytes:
    with wave.open(path) as w:
        if w.getsampwidth() != 2:
            sys.exit(f"{path}: need 16-bit PCM")
        pcm = w.readframes(w.getnframes())
        if w.getnchannels() == 2:
            pcm = np.frombuffer(pcm, np.int16)[::2].tobytes()
        if w.getframerate() != _RATE:
            from scipy.signal import resample_poly
            a = np.frombuffer(pcm, np.int16).astype(np.float32)
            a = resample_poly(a, _RATE, w.getframerate())
            pcm = a.astype(np.int16).tobytes()
    return pcm


def _record(seconds: float) -> bytes:
    import pyaudio
    from fsttm.utils import ignoreStderr
    with ignoreStderr():
        pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=_RATE,
                     input=True, frames_per_buffer=1024)
    print(f"  recording {seconds:.0f}s — speak naturally …", flush=True)
    frames = [stream.read(1024, exception_on_overflow=False)
              for _ in range(int(_RATE / 1024 * seconds))]
    stream.stop_stream(); stream.close(); pa.terminate()
    print("  done")
    return b"".join(frames)


def _load_profiles(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with np.load(path) as npz:
        return {k: np.asarray(npz[k], np.float32) for k in npz.files}


def main():
    ap = argparse.ArgumentParser("fsttm-enroll",
                                 description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True,
                    help='speaker-embedding ONNX (sherpa-onnx compatible)')
    ap.add_argument('--profiles', required=True,
                    help='profiles .npz to create/update')
    ap.add_argument('--name', help='profile name to enroll')
    ap.add_argument('--wav', nargs='+', help='enrollment wav files')
    ap.add_argument('--record', type=int, metavar='N',
                    help='record N enrollment utterances from the mic')
    ap.add_argument('--seconds', type=float, default=5.0,
                    help='seconds per recorded utterance (default 5)')
    ap.add_argument('--test', metavar='WAV',
                    help='score a wav against all enrolled profiles')
    ap.add_argument('--test-mic', action='store_true',
                    help='record one utterance and score it')
    args = ap.parse_args()

    extractor = make_extractor(args.model)
    profiles = _load_profiles(args.profiles)

    if args.test or args.test_mic:
        if not profiles:
            sys.exit(f"no profiles in {args.profiles}")
        pcm = _read_wav(args.test) if args.test else _record(args.seconds)
        emb = embed(extractor, pcm, _RATE)
        print("cosine scores:")
        for name, prof in sorted(profiles.items()):
            s = float(np.dot(emb, prof) /
                      (np.linalg.norm(emb) * np.linalg.norm(prof)))
            print(f"  {name:20s} {s:+.3f}")
        return

    if not args.name or not (args.wav or args.record):
        ap.error("enrollment needs --name and --wav/--record "
                 "(or use --test/--test-mic)")

    embs = []
    if args.wav:
        for p in args.wav:
            print(f"  {p}")
            embs.append(embed(extractor, _read_wav(p), _RATE))
    for i in range(args.record or 0):
        print(f"[{i + 1}/{args.record}]")
        embs.append(embed(extractor, _record(args.seconds), _RATE))

    mean = np.mean(np.stack(embs), axis=0).astype(np.float32)
    # Report enrollment consistency — low self-similarity means bad takes.
    for i, e in enumerate(embs):
        s = float(np.dot(e, mean) / (np.linalg.norm(e) * np.linalg.norm(mean)))
        print(f"  take {i + 1}: self-similarity {s:+.3f}")
    profiles[args.name] = mean
    np.savez(args.profiles, **profiles)
    print(f"enrolled {args.name!r} → {args.profiles} "
          f"({len(profiles)} profile(s))")


if __name__ == '__main__':
    main()
