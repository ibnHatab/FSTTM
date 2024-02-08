import os
import sys
import asyncio
from collections import deque

import numpy as np
import pyaudio
import webrtcvad
from scipy import signal


from utils import ignore_stderr

class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread,
    and stored in a buffer, to be read from."""

    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self, loop, device=None, input_rate=RATE_PROCESS):
        self.loop = loop
        self.buffer_queue = asyncio.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND))

        with ignore_stderr():
            self.pa = pyaudio.PyAudio()

        def proxy_callback(in_data, frame_count, time_info, status):
            #pylint: disable=unused-argument
            self.loop.call_soon_threadsafe(self.buffer_queue.put_nowait, (bytearray(in_data), status))
            return (None, pyaudio.paContinue)

        kwargs = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self.input_rate,
            'input': True,
            'frames_per_buffer': self.block_size_input,
            'stream_callback': proxy_callback,
        }

        # if not default device
        if self.device:
            kwargs['input_device_index'] = self.device

        self.stream = self.pa.open(**kwargs)

    def start(self):
        self.stream.start_stream()

    def stop(self):
        self.stream.stop_stream()

    def resample(self, data, input_rate):
        """
        Microphone may not support our native processing sampling rate, so
        resample from input_rate to RATE_PROCESS here for webrtcvad and
        deepspeech

        Args:
            data (binary): Input audio stream
            input_rate (int): Input audio rate to resample from
        """
        data16 = np.fromstring(string=data, dtype=np.int16)
        resample_size = int(len(data16) / self.input_rate * self.RATE_PROCESS)
        resample = signal.resample(data16, resample_size)
        resample16 = np.array(resample, dtype=np.int16)
        return resample16.tostring()

    async def read_resampled(self):
        """Return a block of audio data resampled to 16000hz, blocking if necessary."""
        data, status = await self.buffer_queue.get()
        return self.resample(data=data, input_rate=self.input_rate)

    async def read(self):
        """Return a block of audio data, blocking if necessary."""
        data, status = await self.buffer_queue.get()
        return data

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)


class VADAudio(Audio):
    """Filter & segment audio with voice activity detection."""

    def __init__(self, loop=None, aggressiveness=3, device=None, input_rate=None):
        if loop is None:
            loop = asyncio.get_running_loop()
        super().__init__(loop, device=device, input_rate=input_rate)
        self.vad = webrtcvad.Vad(aggressiveness)

    async def frame_generator(self):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == self.RATE_PROCESS:
            while True:
                yield await self.read()
        else:
            while True:
                yield await self.read_resampled()

    async def vad_collector(self, padding_ms=300, ratio=0.75):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = deque(maxlen=num_padding_frames)
        triggered = False

        async for frame in self.frame_generator():
            if len(frame) < 640:
                print('frame fenerator < 640')
                yield None

            is_speech = self.vad.is_speech(frame, self.sample_rate)
            os.write(sys.stdout.fileno(), b'1' if is_speech else b'0')
            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        yield f
                    ring_buffer.clear()

            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield None
                    ring_buffer.clear()


if __name__ == '__main__':
    import os
    import sys
    import time

    print(sys.path)
    async def amain():
        vad_audio = VADAudio(aggressiveness=3,
                            device=0,
                            input_rate=16000)
        print("Listening (ctrl-C to exit)...")
        vad_audio.start()

        n = 0
        t = time.time_ns()
        async for frame in  vad_audio.vad_collector():
            if frame is not None:
                if not t: t = time.time_ns()
                n += 1
                os.write(sys.stdout.fileno(), b'.')
                # print("streaming frame: {}".format(len(frame)))
            else:
                tt = time.time_ns() - t
                tt = tt/1e9
                print()
                print("end of utterence: {}f / {}s = {}f/s".format(n, tt, int(n/tt)))
                n = 0
                t = 0 # time.time_ns()

    asyncio.run(amain())




