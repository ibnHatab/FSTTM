"""RHVoice synth backend — statistical TTS, CPU-cheap (Orin-friendly).

Runs the system `RHVoice-client` per utterance (client/service model; the
service is spawned on demand): text on stdin → streaming WAV on stdout.
No Python dependency — the distro package (or an arm64 source build on the
Orin) provides the binaries and voices.

Config block (tts.rhvoice):
    voice: SLT           # -s; installed English voices incl. alan, bdl, clb,
                         #     evgeniy-eng, lyubov, slt
    rate: 0.3            # -r, range -1..1
    volume: -0.1         # -v, range -1..1
    client: RHVoice-client   # binary name/path override
"""
from __future__ import annotations

import io
import subprocess
import wave

from fsttm.tts.base import SynthBackend

_SYNTH_TIMEOUT_S = 30


class RhvoiceBackend(SynthBackend):
    def __init__(self):
        self._argv = None
        self.sample_rate = 24000

    def load(self, cfg: dict) -> None:
        cfg = cfg or {}
        client = cfg.get("client", "RHVoice-client")
        self._argv = [client,
                      "-s", str(cfg.get("voice", "SLT")),
                      "-r", str(cfg.get("rate", 0.3)),
                      "-v", str(cfg.get("volume", -0.1))]
        # Probe: synthesize one character to verify the client + service work
        # and read the ACTUAL sample rate from the WAV header (24000 Hz for the
        # bundled English voices — but never hardcoded).
        pcm, rate = self._run(".")
        if not pcm:
            raise RuntimeError("RHVoice probe produced no audio")
        self.sample_rate = rate
        print(f"RHVoice ready: voice={cfg.get('voice', 'SLT')} "
              f"rate={self.sample_rate} Hz")

    def _run(self, text: str):
        out = subprocess.run(self._argv, input=text.encode("utf-8"),
                             stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL,
                             timeout=_SYNTH_TIMEOUT_S).stdout
        # RHVoice-client streams a WAV with a placeholder length in the header;
        # wave tolerates it for reading params, then we take the raw payload
        # past the data chunk header.
        try:
            with wave.open(io.BytesIO(out)) as w:
                rate = w.getframerate()
                nch = w.getnchannels()
                width = w.getsampwidth()
                pcm = w.readframes(w.getnframes())
        except wave.Error:
            # Streaming header wave can't parse → find the data chunk manually.
            i = out.find(b"data")
            if i < 0:
                return b"", self.sample_rate
            return bytes(out[i + 8:]), self.sample_rate
        if not pcm:
            # 0x7fffffff-length streaming header → readframes saw "0" frames;
            # take everything after the data chunk header instead.
            i = out.find(b"data")
            pcm = bytes(out[i + 8:]) if i >= 0 else b""
        if nch == 2:
            import numpy as np
            pcm = np.frombuffer(pcm, dtype=np.int16)[::2].tobytes()
        if width != 2:
            raise RuntimeError(f"RHVoice produced {width * 8}-bit audio; "
                               f"expected 16-bit")
        return pcm, rate

    def synthesize(self, text: str) -> bytes:
        pcm, _ = self._run(text)
        return pcm
