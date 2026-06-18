"""
Piper TTS driver.

All Speak events are processed by a single serial asyncio worker so synthesis
and playback never overlap. Playback writes PCM straight to a PyAudio output
stream (no aplay subprocess) in small chunks; ClearQueue/CancelPlayback set a
cancel flag the write loop checks between chunks, so barge-in stops audio almost
immediately and drains the pending queue, producing a clean silence before the
next utterance.

Output device & sink are config-driven (tts.device / tts.sink), resolved by name
the same way the mic is. Going through the PulseAudio `pulse` device (the default
for tts.device) lets PulseAudio handle rate/channel conversion and, via the sink
name, routes TTS into module-echo-cancel's reference sink (fsttm_ec_sink) when
AEC is active.
"""
import asyncio
import os
from collections import namedtuple

import numpy as np
import reactivex as rx
from cyclotron import Component

from fsttm.utils import ignoreStderr

Sink = namedtuple('Sink', ['request'])
Source = namedtuple('Source', ['audio'])

# Sink events
Initialize     = namedtuple('Initialize',     ['model_path', 'sample_rate',
                                               'device', 'sink', 'cuda'])
Initialize.__new__.__defaults__ = (None, 22050, None, None, False)
Speak          = namedtuple('Speak',          ['text', 'context'])
Speak.__new__.__defaults__ = (None, None)
CancelPlayback = namedtuple('CancelPlayback', [])
ClearQueue     = namedtuple('ClearQueue',     [])   # cancel active + discard all pending

# Source events
PlaybackStarted      = namedtuple('PlaybackStarted',      ['context'])
AudioPlaybackStarted = namedtuple('AudioPlaybackStarted', ['context', 'duration_s'])
PlaybackDone         = namedtuple('PlaybackDone',         ['context'])
PiperError           = namedtuple('PiperError',           ['error', 'context'])

_SENTINEL = object()   # marks end of queue on shutdown


# Device buffer must be deep enough to ride out scheduling hiccups (asyncio
# loop, TUI refresh, GPU STT/LLM) without starving ALSA → underrun glitches.
# ~340 ms @22 kHz. Barge-in does NOT wait for this to drain: cancel flushes it
# via stop_stream(), so a big buffer costs nothing in cutoff latency.
_OUT_BUFFER_FRAMES = 16384   # ~740ms @22kHz — deeper buffer rides out the gaps
                             # between checkpoint synthesis on the Jetson (was
                             # 8192/~370ms → ALSA underruns mid-narration)
# Write granularity — how often the write loop checks the cancel flag. Small
# enough to notice a barge-in promptly, large enough to keep per-write overhead
# negligible (~46 ms @22 kHz).
_CHUNK_FRAMES = 1024


def _find_output_index(pa, name):
    """PyAudio output device index for the first device whose name contains
    `name` (case-insensitive). None if not found → PyAudio default."""
    if not name:
        return None
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info['maxOutputChannels'] > 0 and name.lower() in info['name'].lower():
            return i
    return None


def make_driver(loop=None):
    def driver(sink):
        voice       = None
        sample_rate = 22050

        pa          = [None]    # PyAudio instance
        out_stream  = [None]    # persistent output stream
        out_channels = [1]      # channels the stream was opened with

        def _open_output(device, snk):
            """Open (or reopen) the PyAudio output stream for the configured
            device/sink. Routing to a named PulseAudio sink (e.g. fsttm_ec_sink
            for AEC, or the Jabra) is done via PULSE_SINK, which the pulse device
            honours; PulseAudio then handles rate/channel conversion."""
            import pyaudio
            if pa[0] is None:
                with ignoreStderr():
                    pa[0] = pyaudio.PyAudio()
            if snk:
                os.environ['PULSE_SINK'] = snk      # picked up by the pulse device
            idx = _find_output_index(pa[0], device)
            kwargs = dict(format=pyaudio.paInt16, channels=1, rate=sample_rate,
                          output=True, frames_per_buffer=_OUT_BUFFER_FRAMES)
            if idx is not None:
                kwargs['output_device_index'] = idx
            try:
                with ignoreStderr():
                    out_stream[0] = pa[0].open(**kwargs)
                out_channels[0] = 1
            except Exception:
                # Some hardware devices reject mono; retry stereo (mono is
                # duplicated to both channels at write time).
                kwargs['channels'] = 2
                with ignoreStderr():
                    out_stream[0] = pa[0].open(**kwargs)
                out_channels[0] = 2
            print(f"Piper output: device={device or 'default'!r} "
                  f"sink={snk or 'default'!r} "
                  f"rate={sample_rate} ch={out_channels[0]}")

        def setup(model_path, rate, device, snk, cuda):
            nonlocal voice, sample_rate
            sample_rate = rate
            try:
                from piper import PiperVoice
            except Exception as exc:
                # piper (piper_phonemize) may be unavailable on some platforms
                # (e.g. no aarch64/Py3.8 wheel on Jetson; TTS slated for TRT).
                # Degrade gracefully: voice stays None, Speak items are no-ops,
                # so STT→LLM→intent→HVAC still runs without spoken responses.
                voice = None
                print(f"WARNING: piper TTS unavailable ({exc}); "
                      f"running without spoken output")
                return
            print(f"Loading piper voice: {model_path} (cuda={cuda})")
            try:
                voice = PiperVoice.load(model_path, use_cuda=cuda)
            except Exception as exc:
                if cuda:
                    print(f"WARNING: CUDA piper load failed ({exc}); falling back to CPU")
                    voice = PiperVoice.load(model_path, use_cuda=False)
                else:
                    raise
            try:
                _open_output(device, snk)
            except Exception as exc:
                print(f"WARNING: could not open audio output ({exc}); "
                      f"running without spoken output")
                out_stream[0] = None
            print("Piper ready")

        def synthesize(text):
            buf = bytearray()
            # piper API differs by version:
            #   1.3+: voice.synthesize(text) → chunks with .audio_int16_bytes
            #   1.2 : voice.synthesize_stream_raw(text) → Iterable[bytes] (PCM)
            # ignoreStderr silences piper/espeak's per-call "Bad voice attribute:
            # option" chatter (harmless phonemizer noise, was 35x/session).
            with ignoreStderr():
                if hasattr(voice, 'synthesize_stream_raw'):
                    for pcm in voice.synthesize_stream_raw(text):
                        buf.extend(pcm)
                else:
                    for chunk in voice.synthesize(text):
                        buf.extend(chunk.audio_int16_bytes)
            return bytes(buf)

        def on_subscribe(observer, scheduler):
            nonlocal voice, sample_rate

            # ── Serial worker state ────────────────────────────────────────
            _queue        = asyncio.Queue()
            # Threading flag (not asyncio.Event): the write loop runs in an
            # executor thread, so the cancel must be visible across threads.
            import threading
            _cancel       = threading.Event()

            def _write_pcm_blocking(pcm):
                """Write PCM to the output stream in chunks, bailing out as soon
                as _cancel is set (barge-in). Runs in an executor thread.

                A deep device buffer (_OUT_BUFFER_FRAMES) absorbs scheduling
                jitter so playback doesn't underrun; the cancel path flushes that
                buffer via stop/start so barge-in is still immediate. The whole
                buffer is converted to the output channel layout ONCE up front so
                each write is a plain slice with no per-chunk numpy work."""
                stream = out_stream[0]
                if stream is None:
                    return
                if out_channels[0] == 2:
                    # mono → interleaved stereo, once
                    a = np.frombuffer(pcm, dtype=np.int16)
                    pcm = np.repeat(a, 2).tobytes()
                frame_bytes = 2 * out_channels[0]
                step = _CHUNK_FRAMES * frame_bytes
                cancelled = False
                for off in range(0, len(pcm), step):
                    if _cancel.is_set():
                        cancelled = True
                        break
                    try:
                        stream.write(pcm[off:off + step])
                    except Exception:
                        break
                if cancelled:
                    # Flush audio already queued in the device buffer so playback
                    # stops now instead of draining the remaining ~340 ms.
                    try:
                        stream.stop_stream()
                        stream.start_stream()
                    except Exception:
                        pass

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
                    # Synthesis in thread pool (blocking)
                    pcm = await loop.run_in_executor(None, synthesize, item.text)

                    # Cancelled during synthesis, or no output stream available
                    if _cancel.is_set() or out_stream[0] is None:
                        loop.call_soon(observer.on_next, PlaybackDone(context=ctx))
                        return

                    # Emit exact timing so the narrator can decide replay/skip
                    duration_s = len(pcm) / (sample_rate * 2)
                    loop.call_soon(observer.on_next,
                                   AudioPlaybackStarted(context=ctx,
                                                        duration_s=duration_s))

                    # Blocking chunked write in the executor; barge-in flips
                    # _cancel and the loop returns within a chunk (~50 ms).
                    await loop.run_in_executor(None, _write_pcm_blocking, pcm)

                except Exception as exc:
                    if not isinstance(exc, asyncio.CancelledError):
                        loop.call_soon(observer.on_next,
                                       PiperError(error=exc, context=ctx))
                finally:
                    loop.call_soon(observer.on_next, PlaybackDone(context=ctx))
                    _cancel.clear()   # reset after each unit completes

            # Start the serial worker
            asyncio.ensure_future(_worker())

            def on_request(item):
                if type(item) is Initialize:
                    setup(item.model_path, item.sample_rate, item.device, item.sink, item.cuda)

                elif type(item) is Speak:
                    if voice is None or out_stream[0] is None \
                            or not (item.text and item.text.strip()):
                        # No TTS (piper/output unavailable) or nothing to say:
                        # emit the playback lifecycle immediately so the FSM
                        # releases the floor (server.py reacts to PlaybackDone)
                        # and turn-taking continues without spoken output.
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
