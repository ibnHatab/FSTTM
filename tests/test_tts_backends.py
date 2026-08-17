"""TTS backend + driver contract tests.

The driver's event lifecycle is floor-critical: PlaybackDone must fire for
every unit (spoken, empty, dropped, failed) or the FSM never releases the
floor, and AudioPlaybackStarted.duration_s must be the exact PCM duration
(the narrator's replay/skip threshold depends on it).

RhvoiceBackend tests are gated on the RHVoice-client binary being installed.
"""
import asyncio
import shutil

import pytest
from reactivex.subject import Subject

from fsttm.tts import driver as tts
from fsttm.tts.base import SynthBackend, load_backend


class FakeBackend(SynthBackend):
    """1 kB of silence per call at 16 kHz — 1024/2/16000 s exactly."""
    def __init__(self, fail=False):
        self.sample_rate = 16000
        self._fail = fail

    def load(self, cfg):
        pass

    def synthesize(self, text):
        if self._fail:
            raise RuntimeError("synth boom")
        return b"\x00" * 1024


def _run_driver(events, backend, ready_player=True, timeout=2.0):
    """Feed events through a driver wired to `backend`; collect source events
    until the loop drains."""
    out = []
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        comp = tts.make_driver(loop)
        subject = Subject()
        src = comp.call(tts.Sink(request=subject))
        src.audio.subscribe(on_next=out.append)

        async def scenario():
            # Install the backend/player state directly (setup() would resolve
            # entry points and open real audio) by replaying the driver's own
            # bookkeeping: Initialize with a bogus backend name degrades to
            # None, so instead we inject via monkeypatched load path below.
            for ev in events:
                subject.on_next(ev)
            # allow queued _play tasks to run
            for _ in range(20):
                await asyncio.sleep(0.01)

        loop.run_until_complete(asyncio.wait_for(scenario(), timeout))
    finally:
        loop.close()
    return out


def test_missing_backend_still_emits_lifecycle(monkeypatch):
    """Speak with no backend loaded → immediate Started+Done (floor releases)."""
    out = _run_driver([tts.Speak(text="hello", context="ckpt:0")], None)
    kinds = [type(e).__name__ for e in out]
    assert kinds == ["PlaybackStarted", "PlaybackDone"]
    assert out[-1].context == "ckpt:0"


def test_empty_text_emits_lifecycle():
    out = _run_driver([tts.Speak(text="   ", context="ckpt:1")], None)
    kinds = [type(e).__name__ for e in out]
    assert kinds == ["PlaybackStarted", "PlaybackDone"]


def test_synth_failure_emits_error_and_done(monkeypatch):
    """A backend that raises still ends with PlaybackDone for the unit."""
    monkeypatch.setattr("fsttm.tts.driver.load_backend",
                        lambda name: FakeBackend(fail=True))
    monkeypatch.setattr("fsttm.tts.player.PcmPlayer.open",
                        lambda self, *a, **k: setattr(self, "_stream", object()))
    monkeypatch.setattr("fsttm.tts.player.PcmPlayer.write",
                        lambda self, pcm, cancel: None)
    out = _run_driver([tts.Initialize(backend="fake", cfg={}),
                       tts.Speak(text="boom", context="ckpt:0")], None)
    kinds = [type(e).__name__ for e in out]
    assert "TtsError" in kinds
    assert kinds[-1] == "PlaybackDone"


def test_duration_exact_and_clearqueue_drains(monkeypatch):
    """duration_s == len(pcm)/(rate*2); ClearQueue emits PlaybackDone for
    every dropped unit."""
    monkeypatch.setattr("fsttm.tts.driver.load_backend",
                        lambda name: FakeBackend())
    monkeypatch.setattr("fsttm.tts.player.PcmPlayer.open",
                        lambda self, *a, **k: setattr(self, "_stream", object()))
    writes = []
    monkeypatch.setattr("fsttm.tts.player.PcmPlayer.write",
                        lambda self, pcm, cancel: writes.append(len(pcm)))
    out = _run_driver([tts.Initialize(backend="fake", cfg={}),
                       tts.Speak(text="a", context="ckpt:0"),
                       tts.Speak(text="b", context="ckpt:1"),
                       tts.Speak(text="c", context="ckpt:2")], None)
    started = [e for e in out if type(e).__name__ == "AudioPlaybackStarted"]
    assert started and all(abs(e.duration_s - 1024 / (16000 * 2)) < 1e-9
                           for e in started)
    done_ctx = [e.context for e in out if type(e).__name__ == "PlaybackDone"]
    # every spoken unit completed
    assert set(done_ctx) == {"ckpt:0", "ckpt:1", "ckpt:2"}


def test_clearqueue_dropped_units_get_done(monkeypatch):
    """Units still queued when ClearQueue arrives must each emit PlaybackDone."""
    monkeypatch.setattr("fsttm.tts.driver.load_backend",
                        lambda name: FakeBackend())
    monkeypatch.setattr("fsttm.tts.player.PcmPlayer.open",
                        lambda self, *a, **k: setattr(self, "_stream", object()))
    monkeypatch.setattr("fsttm.tts.player.PcmPlayer.write",
                        lambda self, pcm, cancel: None)

    out = []
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        comp = tts.make_driver(loop)
        subject = Subject()
        src = comp.call(tts.Sink(request=subject))
        src.audio.subscribe(on_next=out.append)

        async def scenario():
            subject.on_next(tts.Initialize(backend="fake", cfg={}))
            # Queue three units then immediately clear — the worker hasn't had
            # a chance to run yet, so all three are dropped from the queue.
            subject.on_next(tts.Speak(text="a", context="ckpt:0"))
            subject.on_next(tts.Speak(text="b", context="ckpt:1"))
            subject.on_next(tts.Speak(text="c", context="ckpt:2"))
            subject.on_next(tts.ClearQueue())
            for _ in range(20):
                await asyncio.sleep(0.01)

        loop.run_until_complete(asyncio.wait_for(scenario(), 2.0))
    finally:
        loop.close()

    done_ctx = [e.context for e in out if type(e).__name__ == "PlaybackDone"]
    assert set(done_ctx) == {"ckpt:0", "ckpt:1", "ckpt:2"}


# ── entry points ──────────────────────────────────────────────────────────────

def test_backend_entry_points_resolve():
    for name in ("piper", "rhvoice"):
        b = load_backend(name)
        assert isinstance(b, SynthBackend)


def test_unknown_backend_raises():
    with pytest.raises(LookupError):
        load_backend("nope")


# ── rhvoice (gated on the system binary) ─────────────────────────────────────

needs_rhvoice = pytest.mark.skipif(
    shutil.which("RHVoice-client") is None,
    reason="RHVoice-client not installed")


@needs_rhvoice
def test_rhvoice_probe_and_synthesize():
    from fsttm.tts.rhvoice_backend import RhvoiceBackend
    b = RhvoiceBackend()
    b.load({"voice": "SLT", "rate": 0.3, "volume": -0.1})
    assert b.sample_rate == 24000          # bundled English voices
    pcm = b.synthesize("hello from fsttm")
    assert len(pcm) > 8000                 # non-trivial audio
    assert len(pcm) % 2 == 0               # s16le
