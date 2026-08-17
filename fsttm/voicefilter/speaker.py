"""Speaker verification — cosine match against enrolled voice profiles.

Embedding model: any sherpa-onnx speaker-embedding ONNX (WeSpeaker /
3D-Speaker exports). sherpa-onnx bundles its own runtime — no torch, no
system onnxruntime — and ships aarch64 wheels, so the same path runs on the
Orin. Profiles are mean embeddings stored in one .npz (name → vector),
written by the fsttm-enroll CLI.
"""
from __future__ import annotations

import logging
from collections import namedtuple

import numpy as np

_log = logging.getLogger("fsttm.voicefilter")

FilterResult = namedtuple('FilterResult', ['accepted', 'score', 'speaker'])
FilterResult.__new__.__defaults__ = (True, 0.0, None)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


class SpeakerVerifier:
    """VoiceFilter provider `speaker` (fsttm.voice_filters entry point)."""

    def __init__(self):
        self._extractor = None
        self._profiles: dict[str, np.ndarray] = {}
        self.threshold = 0.45
        self.min_utterance_s = 0.5

    def load(self, cfg: dict) -> None:
        cfg = cfg or {}
        self.threshold = float(cfg.get("threshold", 0.45))
        self.min_utterance_s = float(cfg.get("min_utterance_s", 0.5))
        model = cfg.get("model")
        profiles = cfg.get("profiles")
        if not model or not profiles:
            raise ValueError("voice_filter needs `model` (embedding onnx) "
                             "and `profiles` (enrolled .npz)")
        self._extractor = make_extractor(model)
        with np.load(profiles) as npz:
            self._profiles = {k: np.asarray(npz[k], dtype=np.float32)
                              for k in npz.files}
        if not self._profiles:
            raise ValueError(f"no enrolled profiles in {profiles} "
                             f"— run fsttm-enroll first")
        _log.info("speaker filter ready: %d profile(s) (%s), threshold=%.2f",
                  len(self._profiles), ", ".join(self._profiles), self.threshold)

    def check(self, pcm: bytes, sample_rate: int) -> FilterResult:
        """Score one utterance (s16le mono PCM) against the enrolled profiles.
        Too-short utterances bypass (accept) — not enough signal to verify,
        and it keeps short wake words responsive; the score is still logged
        by the driver."""
        n_samples = len(pcm) // 2
        if n_samples < self.min_utterance_s * sample_rate:
            return FilterResult(accepted=True, score=float("nan"), speaker=None)
        emb = embed(self._extractor, pcm, sample_rate)
        best_name, best = None, -1.0
        for name, prof in self._profiles.items():
            s = _cosine(emb, prof)
            if s > best:
                best_name, best = name, s
        return FilterResult(accepted=best >= self.threshold,
                            score=best, speaker=best_name)


# ── sherpa-onnx plumbing (module-level so enroll.py shares it) ───────────────

def make_extractor(model_path: str, num_threads: int = 2):
    import sherpa_onnx
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=model_path, num_threads=num_threads, provider="cpu")
    if not cfg.validate():
        raise ValueError(f"invalid speaker-embedding model: {model_path}")
    return sherpa_onnx.SpeakerEmbeddingExtractor(cfg)


def embed(extractor, pcm: bytes, sample_rate: int) -> np.ndarray:
    """s16le mono PCM → L2-normalizable embedding vector."""
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
    stream.input_finished()
    return np.asarray(extractor.compute(stream), dtype=np.float32)
