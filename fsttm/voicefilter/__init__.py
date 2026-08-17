"""Utterance-level voice filtering ("only my voice").

A pluggable filter stage between the VAD utterance buffer and STT: the full
16 kHz mono utterance is scored by a VoiceFilter provider (entry-point group
``fsttm.voice_filters``; the engine ships ``speaker`` — speaker verification
against enrolled profiles via a sherpa-onnx embedding model, CPU-only, no
torch). Non-matching utterances are dropped before transcription.

Barge-in semantics: the SpeechDuringPlayback sentinel carries no audio, so a
tentative floor flip still happens for any voice — but confirmation requires
a transcript, and the confirming utterance passes through this filter first.
A non-enrolled speaker therefore never confirms, and the narrator's
auto-resume timer restores the floor ~1 s later. Net effect: strangers can
neither command nor barge in, with zero extra barge-in code.

Enroll with the ``fsttm-enroll`` CLI; tune the threshold in ``mode: shadow``
(scores logged, nothing dropped) before switching to ``enforce``.
"""
from fsttm.voicefilter.driver import (  # noqa: F401
    Sink, Source,
    Initialize, Filter,
    Accepted, Rejected,
    make_driver,
)
from fsttm.voicefilter.speaker import FilterResult, SpeakerVerifier  # noqa: F401
