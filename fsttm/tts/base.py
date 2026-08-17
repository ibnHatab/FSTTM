"""SynthBackend — the text→PCM contract every TTS backend implements.

The driver (fsttm.tts.driver) owns everything else: the serial queue, the
cancel flag, playback, and the event lifecycle. A backend only turns a string
into mono 16-bit PCM at its sample rate.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class SynthBackend(ABC):
    #: authoritative AFTER load(); the driver opens the output stream with it
    #: and computes exact playback durations from it.
    sample_rate: int = 22050

    @abstractmethod
    def load(self, cfg: dict) -> None:
        """Load voices/models from the backend's config block (tts.<name>).
        Raise on failure — the driver degrades to silent lifecycle events,
        exactly like the historical missing-piper path."""

    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """Complete utterance → s16le MONO PCM at self.sample_rate. Blocking;
        the driver calls it on an executor thread."""

    def close(self) -> None:
        pass


def load_backend(name: str) -> SynthBackend:
    """Resolve a TTS backend by its fsttm.tts_backends entry-point name."""
    from importlib.metadata import entry_points
    eps = {ep.name: ep for ep in entry_points(group="fsttm.tts_backends")}
    if name not in eps:
        raise LookupError(f"TTS backend {name!r} not installed "
                          f"(available: {sorted(eps) or 'none'})")
    return eps[name].load()()
