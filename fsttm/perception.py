import asyncio
from collections import deque, namedtuple

import reactivex as rx
from cyclotron import Component

from fsttm.mic_vad import VADAudio, find_device_index
import pyaudio as _pa

Sink = namedtuple('Sink', ['control'])
Source = namedtuple('Source', ['voice'])

# Sink events
# device: int index OR None (system default)
# device_name: string to search for (e.g. "fsttm_ec_source"); overrides device
Initialize = namedtuple('Initialize', ['vad_aggressiveness', 'device', 'rate', 'device_name', 'padding_ms'])
Initialize.__new__.__defaults__ = (3, None, 16000, None, 700)

Start  = namedtuple('Start', [])
Stop   = namedtuple('Stop', [])
Duck   = namedtuple('Duck', [])     # full mute: block VAD output to STT
Unduck = namedtuple('Unduck', [])
# SoftDuck: keep VAD running for barge-in detection but suppress STT frames
SoftDuck = namedtuple('SoftDuck', [])


def make_driver(loop=None):
    def driver(sink):
        vad_audio = None

        def setup_vad(aggressiveness, device, rate, device_name, padding_ms):
            if device_name:
                from fsttm.utils import ignoreStderr
                with ignoreStderr():
                    tmp_pa = _pa.PyAudio()
                try:
                    idx = find_device_index(tmp_pa, device_name)
                finally:
                    tmp_pa.terminate()
                if idx is not None:
                    device = idx
                    print(f"Initialising VAD (device_name={device_name!r} → index={idx}, rate={rate})")
                else:
                    print(f"WARNING: device_name={device_name!r} not found, using default")
            else:
                print(f"Initialising VAD (device={device}, rate={rate})")
            return VADAudio(loop,
                            aggressiveness=aggressiveness,
                            device=device,
                            input_rate=rate,
                            padding_ms=padding_ms)

        def on_subscribe(observer, scheduler):
            nonlocal vad_audio
            # Soft-duck routing (pure logic in SoftDuckRouter): while ducked, real
            # speech is suppressed to STT and a SpeechDuringPlayback sentinel is
            # emitted for barge-in; a bounded pre-roll keeps the interrupting
            # utterance's onset, and a delimiter is preserved between ducked
            # utterances so they don't merge.
            _router = SoftDuckRouter(preroll_frames=30)   # ~600 ms of 20 ms frames

            async def read_frames():
                async for frame in vad_audio.vad_collector():
                    for ev in _router.on_frame(frame):
                        loop.call_soon(observer.on_next, ev)

            def on_control(item):
                nonlocal vad_audio
                if type(item) is Initialize:
                    vad_audio = setup_vad(item.vad_aggressiveness,
                                          item.device, item.rate,
                                          item.device_name, item.padding_ms)
                    asyncio.ensure_future(read_frames())
                elif type(item) is Start:
                    if vad_audio:
                        vad_audio.start()
                        print(f"VAD streaming started (device={vad_audio.device})")
                elif type(item) is Stop:
                    if vad_audio:
                        vad_audio.stop()
                elif type(item) is Duck:
                    if vad_audio:
                        vad_audio.duck()
                    _router.duck()
                elif type(item) is SoftDuck:
                    # Keep VAD running (unduck) but suppress output to STT
                    if vad_audio:
                        vad_audio.unduck()
                    _router.soft_duck()
                elif type(item) is Unduck:
                    if vad_audio:
                        vad_audio.unduck()
                    for ev in _router.unduck():   # recover the interrupting onset
                        loop.call_soon(observer.on_next, ev)
                else:
                    observer.on_error(f"Unknown control: {type(item)}")

            sink.control.subscribe(on_next=on_control,
                                   on_error=lambda e: observer.on_error(e))

        return Source(voice=rx.create(on_subscribe))

    return Component(call=driver, input=Sink)


# Additional source event emitted during SoftDuck when speech is detected
SpeechDuringPlayback = namedtuple('SpeechDuringPlayback', [])


# ── soft-duck frame routing (pure, unit-testable) ─────────────────────────────
class SoftDuckRouter:
    """Decides what the perception source emits for each VAD frame, given the
    soft-duck state. Pure (no I/O) so the duck/pre-roll/delimiter logic can be
    tested without a mic. Returns a list of emissions per input:
      - a VAD frame (bytes)            → forward to STT
      - None                           → utterance delimiter (closes a buffer)
      - SpeechDuringPlayback()         → barge-in sentinel

    While ducked, real speech frames are stashed in a bounded pre-roll (so the
    onset of the interrupting utterance isn't lost) and a sentinel is emitted for
    barge-in. A None *while ducked* with pre-rolled content flushes that buffer
    plus the delimiter — so consecutive ducked utterances stay SEPARATE instead
    of merging into one (the bug where 'utterance 1' surfaced glued to
    'utterance 2'). On Unduck the remaining pre-roll is flushed.
    """

    def __init__(self, preroll_frames=30):
        self.ducked = False
        self._preroll = deque(maxlen=preroll_frames)

    def soft_duck(self):
        self.ducked = True
        self._preroll.clear()      # fresh window per response

    def duck(self):
        self.ducked = False
        self._preroll.clear()

    def unduck(self):
        self.ducked = False
        out = list(self._preroll)
        self._preroll.clear()
        return out                 # flush onset of the interrupting utterance

    def on_frame(self, frame):
        """Return the list of things to emit for this VAD frame."""
        if not self.ducked:
            return [frame]         # live: forward frame (incl. None delimiters)
        if frame is not None:
            # ducked speech → barge-in sentinel + stash for pre-roll
            self._preroll.append(frame)
            return [SpeechDuringPlayback()]
        # ducked None = end of a (suppressed) utterance. If we buffered speech,
        # flush it + the delimiter so this utterance doesn't merge with the next.
        if self._preroll:
            out = list(self._preroll) + [None]
            self._preroll.clear()
            return out
        return []                  # silence while ducked → nothing
