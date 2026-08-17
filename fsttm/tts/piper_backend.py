"""Piper (ONNX) synth backend — the default voice.

Config block (tts.piper):
    model: /abs/path/to/en_US-lessac-medium.onnx
    sample_rate: 22050
    cuda: false          # onnxruntime CUDA EP (needs onnxruntime-gpu + cuDNN)
"""
from __future__ import annotations

from fsttm.tts.base import SynthBackend
from fsttm.utils import ignoreStderr


class PiperBackend(SynthBackend):
    def __init__(self):
        self._voice = None
        self.sample_rate = 22050

    def load(self, cfg: dict) -> None:
        cfg = cfg or {}
        model_path = cfg.get("model")
        self.sample_rate = int(cfg.get("sample_rate", 22050))
        cuda = bool(cfg.get("cuda", False))
        # piper (piper_phonemize) may be unavailable on some platforms (e.g. no
        # matching wheel). Raising here makes the driver degrade gracefully:
        # Speak items become no-op lifecycle events, so STT→LLM→intent still
        # runs without spoken responses.
        from piper import PiperVoice
        print(f"Loading piper voice: {model_path} (cuda={cuda})")
        try:
            self._voice = PiperVoice.load(model_path, use_cuda=cuda)
        except Exception as exc:
            if cuda:
                print(f"WARNING: CUDA piper load failed ({exc}); "
                      f"falling back to CPU")
                self._voice = PiperVoice.load(model_path, use_cuda=False)
            else:
                raise
        print("Piper ready")

    def synthesize(self, text: str) -> bytes:
        buf = bytearray()
        # piper API differs by version:
        #   1.3+: voice.synthesize(text) → chunks with .audio_int16_bytes
        #   1.2 : voice.synthesize_stream_raw(text) → Iterable[bytes] (PCM)
        # ignoreStderr silences piper/espeak's per-call "Bad voice attribute:
        # option" chatter (harmless phonemizer noise, was 35x/session).
        with ignoreStderr():
            if hasattr(self._voice, 'synthesize_stream_raw'):
                for pcm in self._voice.synthesize_stream_raw(text):
                    buf.extend(pcm)
            else:
                for chunk in self._voice.synthesize(text):
                    buf.extend(chunk.audio_int16_bytes)
        return bytes(buf)
