"""PcmPlayer — PyAudio output stream shared by all TTS backends.

Playback writes PCM straight to a PyAudio output stream (no aplay subprocess)
in small chunks; a cancel flag checked between chunks stops audio almost
immediately, and the device buffer is flushed via stop/start so barge-in
doesn't drain the remaining ~340 ms.

Routing to a named PulseAudio sink (e.g. fsttm_ec_sink for AEC, or the Jabra)
is done via PULSE_SINK, which the `pulse` device honours; PulseAudio then
handles rate/channel conversion.
"""
from __future__ import annotations

import os
import time

import numpy as np

from fsttm.utils import ignoreStderr

# Device buffer must be deep enough to ride out scheduling hiccups (asyncio
# loop, TUI refresh, GPU STT/LLM) without starving ALSA → underrun glitches.
# Barge-in does NOT wait for this to drain: cancel flushes it via
# stop_stream(), so a big buffer costs nothing in cutoff latency.
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


class PcmPlayer:
    def __init__(self):
        self._pa = None
        self._stream = None
        self._channels = 1
        self._sample_rate = 22050

    @property
    def ready(self) -> bool:
        return self._stream is not None

    def open(self, sample_rate: int, device=None, sink=None) -> None:
        """Open (or reopen) the output stream for the configured device/sink."""
        import pyaudio
        self._sample_rate = sample_rate
        if self._pa is None:
            with ignoreStderr():
                self._pa = pyaudio.PyAudio()
        if sink:
            os.environ['PULSE_SINK'] = sink      # picked up by the pulse device
        idx = _find_output_index(self._pa, device)
        kwargs = dict(format=pyaudio.paInt16, channels=1, rate=sample_rate,
                      output=True, frames_per_buffer=_OUT_BUFFER_FRAMES)
        if idx is not None:
            kwargs['output_device_index'] = idx
        try:
            with ignoreStderr():
                self._stream = self._pa.open(**kwargs)
            self._channels = 1
        except Exception:
            # Some hardware devices reject mono; retry stereo (mono is
            # duplicated to both channels at write time).
            kwargs['channels'] = 2
            with ignoreStderr():
                self._stream = self._pa.open(**kwargs)
            self._channels = 2
        print(f"TTS output: device={device or 'default'!r} "
              f"sink={sink or 'default'!r} "
              f"rate={sample_rate} ch={self._channels}")

    def write(self, pcm: bytes, cancel_event) -> None:
        """Write PCM in chunks, bailing out as soon as cancel_event is set
        (barge-in). Runs in an executor thread. The whole buffer is converted
        to the output channel layout ONCE up front so each write is a plain
        slice with no per-chunk numpy work.

        Returns only when the audio has actually FINISHED PLAYING (or was
        cancelled): stream.write merely queues frames, and the PulseAudio
        plugin can absorb seconds of audio into its own buffer, so the chunk
        loop may complete almost instantly. Without the drain wait below,
        PlaybackDone fires while the speaker is still talking — the narrator
        un-ducks, the open mic transcribes the assistant's own voice, and the
        pipeline feeds itself (observed live with AEC off)."""
        stream = self._stream
        if stream is None:
            return
        # Wall-clock duration of the MONO payload (before channel conversion).
        duration = len(pcm) / (2.0 * self._sample_rate)
        t0 = time.monotonic()
        if self._channels == 2:
            # mono → interleaved stereo, once
            a = np.frombuffer(pcm, dtype=np.int16)
            pcm = np.repeat(a, 2).tobytes()
        frame_bytes = 2 * self._channels
        step = _CHUNK_FRAMES * frame_bytes
        cancelled = False
        for off in range(0, len(pcm), step):
            if cancel_event.is_set():
                cancelled = True
                break
            try:
                stream.write(pcm[off:off + step])
            except Exception:
                break
        if not cancelled:
            # Drain wait: hold until the playback clock catches up with the
            # queued audio, still honouring barge-in cancel every 50 ms.
            while not cancel_event.is_set():
                remaining = duration - (time.monotonic() - t0)
                if remaining <= 0:
                    break
                time.sleep(min(0.05, remaining))
            cancelled = cancel_event.is_set() and \
                (time.monotonic() - t0) < duration
        if cancelled:
            # Flush audio already queued in the device buffer so playback
            # stops now instead of draining the remainder.
            try:
                stream.stop_stream()
                stream.start_stream()
            except Exception:
                pass

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
