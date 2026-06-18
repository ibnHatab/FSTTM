from __future__ import annotations  # PEP 563: lazy annotations for Python 3.8 (int | None)
import asyncio
from collections import deque

import numpy as np
import pyaudio
import webrtcvad
from scipy import signal

from fsttm.utils import ignoreStderr


def find_device_index(pa: pyaudio.PyAudio, name: str) -> int | None:
    """Return PyAudio device index for the first input device whose name contains `name`."""
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0 and name.lower() in info['name'].lower():
            return i
    return None


class Audio:
    FORMAT = pyaudio.paInt16
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self, loop, device=None, input_rate=RATE_PROCESS):
        self.buffer_queue = asyncio.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND))
        with ignoreStderr():
            self.pa = pyaudio.PyAudio()

        def proxy_callback(in_data, frame_count, time_info, status):
            loop.call_soon_threadsafe(
                self.buffer_queue.put_nowait, (bytearray(in_data), status)
            )
            return (None, pyaudio.paContinue)

        kwargs = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self.input_rate,
            'input': True,
            'frames_per_buffer': self.block_size_input,
            'stream_callback': proxy_callback,
        }
        if self.device is not None:
            kwargs['input_device_index'] = self.device

        self.stream = self.pa.open(**kwargs)

    def start(self):
        self.stream.start_stream()

    def stop(self):
        self.stream.stop_stream()

    def resample(self, data, input_rate):
        data16 = np.frombuffer(data, dtype=np.int16)
        resample_size = int(len(data16) / input_rate * self.RATE_PROCESS)
        resampled = signal.resample(data16, resample_size)
        return np.array(resampled, dtype=np.int16).tobytes()

    async def read_resampled(self):
        data, _ = await self.buffer_queue.get()
        return self.resample(data=data, input_rate=self.input_rate)

    async def read(self):
        data, _ = await self.buffer_queue.get()
        return data

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)


class VADAudio(Audio):
    def __init__(self, loop, aggressiveness=3, device=None, input_rate=None,
                 padding_ms=700):
        super().__init__(loop, device=device, input_rate=input_rate)
        self.vad = webrtcvad.Vad(aggressiveness)
        self.padding_ms = padding_ms   # end-of-utterance silence before closing turn
        # ducking: clear → muted (TTS playing), set → active (normal)
        self._duck_event = asyncio.Event()
        self._duck_event.set()

    def duck(self):
        self._duck_event.clear()

    def unduck(self):
        self._duck_event.set()

    async def frame_generator(self):
        if self.input_rate == self.RATE_PROCESS:
            while True:
                yield await self.read()
        else:
            while True:
                yield await self.read_resampled()

    async def vad_collector(self, padding_ms=None, ratio=0.75,
                            onset_ms=200, onset_ratio=0.40):
        """
        Yields audio frames for each utterance, separated by None delimiters.

        Onset and end-of-utterance use separate parameters so short syllables
        at the start of speech ("no", "ok") can still trigger recording:
          onset_ms / onset_ratio  — small window (200ms, 40%) for fast trigger
          padding_ms / ratio      — large window (700ms, 75%) for end-of-turn
        """
        padding_ms = padding_ms if padding_ms is not None else self.padding_ms
        onset_frames   = onset_ms  // self.frame_duration_ms   # e.g. 10 frames
        hangover_frames = padding_ms // self.frame_duration_ms  # e.g. 35 frames

        # onset_buf  — short window; triggers recording when > onset_ratio voiced
        # hang_buf   — long window;  ends utterance when > ratio silent
        # pre_roll   — frames buffered before onset fires (provides context)
        onset_buf  = deque(maxlen=onset_frames)
        hang_buf   = deque(maxlen=hangover_frames)
        pre_roll   = deque(maxlen=onset_frames)   # same depth as onset window
        triggered  = False

        async for frame in self.frame_generator():
            await self._duck_event.wait()

            if len(frame) < 640:
                yield None
                continue

            is_speech = self.vad.is_speech(bytes(frame), self.sample_rate)

            if not triggered:
                pre_roll.append(frame)
                onset_buf.append(is_speech)
                voiced = sum(onset_buf)
                if voiced > onset_ratio * onset_buf.maxlen:
                    triggered = True
                    hang_buf.clear()
                    for f in pre_roll:
                        yield f
                    pre_roll.clear()
                    onset_buf.clear()
            else:
                yield frame
                hang_buf.append(is_speech)
                unvoiced = hang_buf.maxlen - sum(hang_buf)
                if len(hang_buf) == hang_buf.maxlen and unvoiced > ratio * hang_buf.maxlen:
                    triggered = False
                    pre_roll.clear()
                    onset_buf.clear()
                    yield None
                    hang_buf.clear()


if __name__ == '__main__':
    import sys
    import time

    async def amain(loop):
        vad_audio = VADAudio(loop, aggressiveness=3, device=None, input_rate=16000)
        print("Listening (ctrl-C to exit)...")
        vad_audio.start()
        n = 0
        t = time.time_ns()
        async for frame in vad_audio.vad_collector():
            if frame is not None:
                n += 1
            else:
                tt = (time.time_ns() - t) / 1e9
                print(f"\nutterance: {n} frames / {tt:.2f}s")
                n = 0
                t = time.time_ns()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    asyncio.ensure_future(amain(loop=loop))
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        sys.exit('\nInterrupted')
