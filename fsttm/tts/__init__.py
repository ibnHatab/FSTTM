"""Pluggable TTS: backend-agnostic driver + entry-point synth backends.

Backends register in the fsttm.tts_backends entry-point group (the engine
ships `piper` and `rhvoice`); config `tts.backend` selects one per
deployment. See driver.py for the floor-critical event contract.
"""
from fsttm.tts.base import SynthBackend, load_backend  # noqa: F401
from fsttm.tts.driver import (  # noqa: F401
    Sink, Source,
    Initialize, Speak, CancelPlayback, ClearQueue,
    PlaybackStarted, AudioPlaybackStarted, PlaybackDone, TtsError,
    make_driver,
)
from fsttm.tts.player import PcmPlayer  # noqa: F401
