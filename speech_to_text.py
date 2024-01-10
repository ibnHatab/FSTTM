
import os
import re
import sys
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
        self.loop = asyncio.get_event_loop()

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

    async def process_data(self, data):
        text = await self.loop.run_in_executor(None, self.transcribe, data)
        return text

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
        self.vad_active = False

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
        wc = 0
        uterance = bytearray()
        pattern = r'\{([^{}]*)\}|\(([^()]*)\)|\[([^[\]]*)\]'
        non_speech_tokens = re.compile(pattern)
        async for frame in self.audio.vad_collector():
            if frame is not None:
                if not ts:
                    ts = time.time_ns()
                uterance.extend(frame)
                if not self.vad_active:
                    self.voice_active = True
            else:
                tt = time.time_ns() - ts
                tt = tt / 1e9
                text_query = await self.stt.process_data(uterance)
                print(f"\n>> {text_query}")
                text = non_speech_tokens.sub('', text_query).strip()
                if text:
                    self.voice_active = False
                    yield SpeechVars(text, wc, tt)
                else:
                    self.voice_active = False # bare noice
                ts = 0
                wc += 1
                uterance.clear()

    @property
    def voice_active(self):
        return self.vad_active

    @voice_active.setter
    def voice_active(self, val: bool):
        self.vad_active = val
        self.voice_active_ind(self.vad_active)

    def voice_active_ind(self, active: bool):
        print(f"{'*' if active else '.'}", end='', flush=True)
        pass

# FIXME: Implement dramatical pause of 0.5 seconds using asyncio.queue and asyncio.sleep

if __name__ == '__main__':

    async def amain():

        vad_audio = VADAudio(aggressiveness=3,
                            device=0,
                            input_rate=16000)

        whisper = Whisper(model_name='base.en')
        stt_svc = SpeechToTextProxy(vad_audio, whisper)
        stt_svc.start()

        async for text in stt_svc.async_generator():
                print(f"\n{text}")

    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')




