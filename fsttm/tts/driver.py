"""
TTS driver — backend-agnostic queue/worker/playback lifecycle.

All Speak events are processed by a single serial asyncio worker so synthesis
and playback never overlap. The synth backend (piper, rhvoice, …) is selected
by entry-point name at Initialize; the driver owns the floor-critical event
contract:

  - PlaybackDone fires for EVERY unit — spoken, empty-text, backend-missing,
    failed, and each unit dropped by ClearQueue — or the FSM never releases
    the floor.
  - AudioPlaybackStarted.duration_s is the EXACT PCM duration
    (len(pcm) / (sample_rate * 2)); the narrator's replay/skip threshold
    depends on it.
  - ClearQueue stops the active unit (cancel flag, checked between playback
    chunks) and drains all pending units.
"""
import asyncio
import threading
from collections import namedtuple

import reactivex as rx
from cyclotron import Component

from fsttm.tts.base import load_backend
from fsttm.tts.player import PcmPlayer

Sink = namedtuple('Sink', ['request'])
Source = namedtuple('Source', ['audio'])

# Sink events. `backend` is the fsttm.tts_backends entry-point name; `cfg` is
# that backend's config block (tts.<backend>).
Initialize     = namedtuple('Initialize',     ['backend', 'cfg', 'device', 'sink'])
Initialize.__new__.__defaults__ = ('piper', None, None, None)
Speak          = namedtuple('Speak',          ['text', 'context'])
Speak.__new__.__defaults__ = (None, None)
CancelPlayback = namedtuple('CancelPlayback', [])
ClearQueue     = namedtuple('ClearQueue',     [])   # cancel active + discard all pending

# Source events
PlaybackStarted      = namedtuple('PlaybackStarted',      ['context'])
AudioPlaybackStarted = namedtuple('AudioPlaybackStarted', ['context', 'duration_s'])
PlaybackDone         = namedtuple('PlaybackDone',         ['context'])
TtsError             = namedtuple('TtsError',             ['error', 'context'])

_SENTINEL = object()   # marks end of queue on shutdown


def make_driver(loop=None):
    def driver(sink):
        backend = [None]          # loaded SynthBackend, or None (degraded)
        player = PcmPlayer()

        def setup(name, cfg, device, snk):
            try:
                b = load_backend(name or 'piper')
                b.load(cfg or {})
            except Exception as exc:
                # Backend unavailable (missing wheel, missing RHVoice binary,
                # bad model path …). Degrade gracefully: Speak items become
                # immediate lifecycle no-ops, so STT→LLM→intent still runs
                # without spoken responses.
                backend[0] = None
                print(f"WARNING: TTS backend {name!r} unavailable ({exc}); "
                      f"running without spoken output")
                return
            backend[0] = b
            try:
                player.open(b.sample_rate, device=device, sink=snk)
            except Exception as exc:
                print(f"WARNING: could not open audio output ({exc}); "
                      f"running without spoken output")

        def on_subscribe(observer, scheduler):
            # ── Serial worker state ────────────────────────────────────────
            _queue = asyncio.Queue()
            # Threading flag (not asyncio.Event): the write loop runs in an
            # executor thread, so the cancel must be visible across threads.
            _cancel = threading.Event()

            async def _worker():
                """Process one Speak at a time — no overlapping synthesis/playback."""
                while True:
                    item = await _queue.get()
                    if item is _SENTINEL:
                        break
                    if not isinstance(item, Speak):
                        continue
                    await _play(item)

            async def _play(item):
                ctx = item.context
                loop.call_soon(observer.on_next, PlaybackStarted(context=ctx))
                try:
                    b = backend[0]
                    # Synthesis in thread pool (blocking)
                    pcm = await loop.run_in_executor(None, b.synthesize, item.text)

                    # Cancelled during synthesis, or no output stream available
                    if _cancel.is_set() or not player.ready:
                        loop.call_soon(observer.on_next, PlaybackDone(context=ctx))
                        return

                    # Emit exact timing so the narrator can decide replay/skip
                    duration_s = len(pcm) / (b.sample_rate * 2)
                    loop.call_soon(observer.on_next,
                                   AudioPlaybackStarted(context=ctx,
                                                        duration_s=duration_s))

                    # Blocking chunked write in the executor; barge-in flips
                    # _cancel and the loop returns within a chunk (~50 ms).
                    await loop.run_in_executor(None, player.write, pcm, _cancel)

                except Exception as exc:
                    if not isinstance(exc, asyncio.CancelledError):
                        loop.call_soon(observer.on_next,
                                       TtsError(error=exc, context=ctx))
                finally:
                    loop.call_soon(observer.on_next, PlaybackDone(context=ctx))
                    _cancel.clear()   # reset after each unit completes

            # Start the serial worker
            asyncio.ensure_future(_worker())

            def on_request(item):
                if type(item) is Initialize:
                    setup(item.backend, item.cfg, item.device, item.sink)

                elif type(item) is Speak:
                    if backend[0] is None or not player.ready \
                            or not (item.text and item.text.strip()):
                        # No TTS (backend/output unavailable) or nothing to
                        # say: emit the playback lifecycle immediately so the
                        # FSM releases the floor (the pipeline reacts to
                        # PlaybackDone) and turn-taking continues without
                        # spoken output.
                        ctx = item.context
                        loop.call_soon(observer.on_next, PlaybackStarted(context=ctx))
                        loop.call_soon(observer.on_next, PlaybackDone(context=ctx))
                    else:
                        _queue.put_nowait(item)

                elif type(item) is CancelPlayback:
                    # Stop only the currently-playing unit; the queue continues.
                    _cancel.set()

                elif type(item) is ClearQueue:
                    # Stop active playback and discard all pending units.
                    _cancel.set()
                    while not _queue.empty():
                        try:
                            dropped = _queue.get_nowait()
                            if isinstance(dropped, Speak):
                                # Emit PlaybackDone for each dropped unit
                                loop.call_soon(observer.on_next,
                                               PlaybackDone(context=dropped.context))
                        except asyncio.QueueEmpty:
                            break

                else:
                    observer.on_error(f"Unknown item type: {type(item)}")

            sink.request.subscribe(on_next=on_request,
                                   on_error=lambda e: observer.on_error(e))

        return Source(audio=rx.create(on_subscribe))

    return Component(call=driver, input=Sink)
