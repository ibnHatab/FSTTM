
import os
import time
import asyncio
import numpy as np
from dataclasses import dataclass

import whispercpp as w
from mic_vad import VADAudio
from utils import ignore_stderr

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
        params = (
            self.model.params
            .with_print_realtime(False)
            .with_num_threads(n_threads)
            .with_suppress_blank(True)
            .build()
        )

    async def process_data(self, data):
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

class SpeechToTextProxy:
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

    def __init__(self, vad: VADAudio, stt: Whisper) -> None:
        self.audio = vad
        self.stt = stt

    def start(self):
        """
        Starts the audio processing.
        """
        self.audio.start()

    def stop(self):
        """
        Stops the audio processing.
        """
        self.audio.stop()

    async def async_generator(self):
        """
        Asynchronously generates speech variables.

        Yields:
            SpeechVars: A named tuple containing the processed text, count, and processing time.

        """
        ts = time.time_ns()
        count = 0
        uterance = bytearray()
        async for frame in self.audio.vad_collector():
            if frame is not None:
                if not ts:
                    ts = time.time_ns()
                os.write(sys.stdout.fileno(), b'.')
                uterance.extend(frame)
            else:
                tt = time.time_ns() - ts
                tt = tt / 1e9
                text = await self.stt.process_data(uterance)
                uterance.clear()
                yield SpeechVars(text, count, tt)
                ts = 0  # time.time_ns()
                count += 1


if __name__ == '__main__':
    import sys

    async def amain():

        vad_audio = VADAudio(aggressiveness=3,
                            device=0,
                            input_rate=16000)

        whisper = Whisper(model_name='base')

        stt_svc = SpeechToTextProxy(vad_audio, whisper)
        stt_svc.start()

        async for text in stt_svc.async_generator():
                print(f"\n{text}")

    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')




