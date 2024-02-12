import os
import queue
import sys
import time
from collections import deque
import threading

import numpy as np
import pyaudio
import pykka
import webrtcvad
from scipy import signal


from utils import ignore_stderr

DEBUG = True

class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread,
    and stored in a buffer, to be read from."""

    FORMAT = pyaudio.paInt16
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self, device=None, input_rate=RATE_PROCESS):
        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND))

        with ignore_stderr():
            self.pa = pyaudio.PyAudio()

        def proxy_callback(in_data, frame_count, time_info, status):
            #pylint: disable=unused-argument
            self.buffer_queue.put_nowait(bytearray(in_data))
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

    def start_audio(self):
        self.stream.start_stream()

    def stop_audio(self):
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

    def read_resampled(self, timeout=None):
        """Return a block of audio data resampled to 16000hz, blocking if necessary."""
        data = self.buffer_queue.get(timeout=timeout)
        return self.resample(data=data, input_rate=self.input_rate)

    def read(self, timeout=None):
        """Return a block of audio data, blocking if necessary."""
        data = self.buffer_queue.get(timeout=timeout)
        return data

    def destroy_audio(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()


    frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)


class VADAudioProducer(threading.Thread):
    """Filter & segment audio with voice activity detection."""

    def __init__(self, consumer: pykka.ThreadingActor, aggressiveness=3, device=None, input_rate=None):
        super().__init__()
        self.audio = Audio(device=device, input_rate=input_rate)
        self.consumer = consumer
        self.vad = webrtcvad.Vad(aggressiveness)

        self._stopped = threading.Event()
        self._lock = threading.Lock()
        self.audio.start_audio()
        self.start()

    def run(self):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """

        if self.audio.input_rate == self.audio.RATE_PROCESS:
            _read = self.audio.read
        else:
            _read = self.audio.read_resampled

        padding_ms = 300
        ratio = 0.75

        num_padding_frames = padding_ms // self.audio.frame_duration_ms
        ring_buffer = deque(maxlen=num_padding_frames)
        triggered = False


        def _send_uterance(data, stamp):
            tt = time.time_ns() - stamp
            tt = tt / 1e9
            self.consumer.tell({'uterance': data.copy(), 'time': tt})
            data.clear()
            stamp = None

        ts = None
        uterance = bytearray()

        while not self._stopped.is_set():
            try:
                frame = _read(timeout=1)
            except queue.Empty:
                continue

            if len(frame) < 640:
                if DEBUG:
                    print('frame fenerator < 640')
                _send_uterance(uterance, ts)

            is_speech = self.vad.is_speech(frame, self.audio.sample_rate)
            if DEBUG:
                os.write(sys.stdout.fileno(), b'1' if is_speech else b'0')

            if not triggered:
                if not ts:
                    ts = time.time_ns()

                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        uterance.extend(frame)
                    ring_buffer.clear()

            else:
                uterance.extend(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    _send_uterance(uterance, ts)
                    ring_buffer.clear()



    def stop(self):
        """
        Stops the thread.
        """
        self.audio.destroy_audio()
        self._stopped.set()
        self.join()

if __name__ == '__main__':

    class PlainActor(pykka.ThreadingActor):
        def __init__(self):
            super().__init__()
            self.stored_messages = []

        def on_receive(self, message):
            print("Received: ", len(message['uterance']), message['time'])
            self.stored_messages.append(message)
            return None

    consumer = PlainActor.start()
    vad_audio = VADAudioProducer(consumer, aggressiveness=3, device=None, input_rate=16000)

    print("Listening (ctrl-C to exit)...")

    time.sleep(5)

    print("Stopping...")
    consumer.stop()
    vad_audio.stop()



