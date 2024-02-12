
import os
import re
import sys
import time
import asyncio
import numpy as np
from dataclasses import dataclass
import pykka

import whispercpp as w
from mic_vad import VADAudio
from mic_vad_thread import VADAudioProducer

from utils import ignore_stderr

DEBUG = True

class Whisper:
    """
    A class that performs speech-to-text conversion using the Whisper model.

    Args:
        model_name (str): The name of the pretrained Whisper model.
        n_threads (int, optional): The number of threads to use for processing. Defaults to 7.
    """

    def __init__(self, model_name, n_threads=7):
        with ignore_stderr():
            self.model = w.Whisper.from_pretrained(model_name)
        params = ( # noqa # type: ignore
            self.model.params
            .with_print_realtime(False)
            .with_num_threads(n_threads)
            .with_suppress_blank(True)
            .with_language('en')
            .build()
        )
            #asas.with_suppress_none_speech_tokens(True)

    def transcribe(self, data: bytes):
        """
        Processes the input speech data and returns the transcribed text.

        Args:
            data (bytes): The input speech data as a byte array.

        Returns:
            str: The transcribed text.
        """
        text = None
        try:
            audio = np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0
            if len(data)/640 < 50:
                audio = np.pad(audio, (0, 16000), 'constant')
            text = self.model.transcribe(audio)
        except Exception as e:
            print(f"Whisper error: {e}")

        return text


@dataclass
class SpeechVars:
    Uterance: str
    Count: int
    Stamp: int

class SpeechToTextSvc(pykka.ThreadingActor):
    """
    A high-performance inference of OpenAI's Whisper with automatic speech recognition (ASR) model.

    Attributes:
        audio (VADAudio): The VADAudio object for audio processing.
        stt (Whisper): The Whisper object for speech-to-text processing.

    Methods:
        start(): Starts the audio processing.
        stop(): Stops the audio processing.
        async_generator(): Asynchronously generates speech variables.

    """

    def __init__(self, consumer: pykka.ThreadingActor, stt: Whisper) -> None:
        super().__init__()
        self.consumer = consumer
        self.stt = stt
        pattern = r'\{([^{}]*)\}|\(([^()]*)\)|\[([^[\]]*)\]'
        self.non_speech_tokens = re.compile(pattern)


    def on_receive(self, message):

        self.process_data(message['uterance'])

    def process_data(self, uterance: bytes):
        """
        Asynchronously generates speech variables.

        Yields:
            SpeechVars: A named tuple containing the processed text, count, and processing time.

        """
        if DEBUG:
            print('<<', end='', flush=True)
        ts = time.time_ns()
        wc = 0
        text_query = self.stt.transcribe(uterance)
        tt = time.time_ns() - ts
        tt = tt / 1e9
        if DEBUG:
            print(f"\n>> {text_query}", flush=True)
        text = self.non_speech_tokens.sub('', text_query).strip()
        if text:
            self.consumer.tell(SpeechVars(text, wc, tt))


if __name__ == '__main__':

    class PlainActor(pykka.ThreadingActor):
        def __init__(self):
            super().__init__()

        def on_receive(self, message):
            print("Received: ", message)
            return None

    consumer = PlainActor.start()

    whisper = Whisper(model_name='base.en')
    stt_svc = SpeechToTextSvc.start(consumer, whisper)

    vad_audio = VADAudioProducer(stt_svc, aggressiveness=3, device=None, input_rate=16000)

    print("Listening (ctrl-C to exit)...")

    time.sleep(5)

    print("Stopping...")

    pykka.ActorRegistry.stop_all()
    vad_audio.stop()





