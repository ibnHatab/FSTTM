
import os
import logging
import threading
import queue
import asyncio

from collections import namedtuple, deque

import numpy as np
import pyaudio
import wave
import webrtcvad
from scipy import signal

import rx
from cyclotron import Component
from cyclotron_std.logging import Log

class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, and stored in a buffer, to be read from."""

    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS):
        def proxy_callback(in_data, frame_count, time_info, status):
            #pylint: disable=unused-argument
            if self.chunk is not None:
                in_data = self.wf.readframes(self.chunk)
            callback(in_data)
            return (None, pyaudio.paContinue)

        if callback is None: callback = lambda in_data: self.buffer_queue.put(in_data)
        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND))
        self.pa = pyaudio.PyAudio()

        kwargs = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self.input_rate,
            'input': True,
            'frames_per_buffer': self.block_size_input,
            'stream_callback': proxy_callback,
        }

        self.chunk = None
        # if not default device
        if self.device:
            kwargs['input_device_index'] = self.device

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

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

    def read_resampled(self):
        """Return a block of audio data resampled to 16000hz, blocking if necessary."""
        return self.resample(data=self.buffer_queue.get(),
                             input_rate=self.input_rate)

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)


class VADAudio(Audio):
    """Filter & segment audio with voice activity detection."""

    def __init__(self, aggressiveness=3, device=None, input_rate=None):
        super().__init__(device=device, input_rate=input_rate)
        self.vad = webrtcvad.Vad(aggressiveness)

    def frame_generator(self):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == self.RATE_PROCESS:
            while True:
                yield self.read()
        else:
            while True:
                yield self.read_resampled()

    def vad_collector(self, padding_ms=300, ratio=0.75, frames=None):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        if frames is None: frames = self.frame_generator()
        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = deque(maxlen=num_padding_frames)
        triggered = False

        for frame in frames:
            if len(frame) < 640:
                return

            is_speech = self.vad.is_speech(frame, self.sample_rate)

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


# Component
Sink = namedtuple('Sink', ['request'])
Source = namedtuple('Source', ['response', 'log'])

# Sink events
Initialize = namedtuple('Initialize', ['vad_aggressiveness', 'device', 'rate'])
Initialize.__new__.__defaults__ = (None,)
Start = namedtuple('Start', [])

# Sourc events
Utterence = namedtuple('Utterence', ['data', 'context'])

def make_driver(loop=None):
    def driver(sink):
        vad_audio = None
        log_observer = None

        def on_log_subscribe(observer, scheduler):
            nonlocal log_observer
            log_observer = observer

        def log(message, level=logging.DEBUG):
            if log_observer is not None:
                log_observer.on_next(Log(
                    logger=__name__,
                    level=level,
                    message=message,
                ))

        def setup_vad(vad_aggressiveness, device, rate):
            vad = VADAudio(aggressiveness=vad_aggressiveness, device=device, input_rate=rate)
            log("Listening on {} (ctrl-C to exit)...".format(device))
            return vad

        def on_subscribe(observer, scheduler):

            # FIXME: rewrite VAD from yield based to async
            async def read_events():
                nonlocal vad_audio
                frames = vad_audio.vad_collector()
                for frame in frames:
                    if frame is not None:
                        print("streaming frame")
                        observer.on_next(Utterence(data=frame, context=None))
                    else:
                        print("end of utterence")
                        observer.on_next(Utterence(data=None, context=None))

            def on_next(item):
                nonlocal vad_audio
                if type(item) is Initialize:
                    vad_audio = setup_vad(item.vad_aggressiveness, item.device, item.rate)
                elif type(item) is Start:
                    asyncio.ensure_future(read_events())
                else:
                    log("unknown item: {}".format(item), level=logging.CRITICAL)
                    observer.on_error(
                        "Unknown item type: {}".format(type(item)))

            sink.request.subscribe(
                on_next=on_next,
                on_error=lambda e: observer.on_error(e))

        return Source(
            response=rx.create(on_subscribe),
            log=rx.create(on_log_subscribe),
        )

    return Component(call=driver, input=Sink)


if __name__ == '__main__':
    ARGS = Initialize(3, 0, 16000)
    vad_audio = VADAudio(aggressiveness=ARGS.vad_aggressiveness,
                         device=ARGS.device,
                         input_rate=ARGS.rate)
    print("Listening (ctrl-C to exit)...")

    frames = vad_audio.vad_collector()

    for frame in frames:
        if frame is not None:
            print("streaming frame")
        else:
            print("end of utterence")


