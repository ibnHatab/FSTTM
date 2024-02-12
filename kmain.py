#!/usr/bin/env python3

import sys
import pykka
import logging

from mic_vad_thread import VADAudioProducer
from speech_to_text import SpeechToTextSvc




def kmain():

    stt = SpeechToTextSvc.start()
    vad_audio = VADAudioProducer(stt, aggressiveness=3, device=0, input_rate=16000)


    vad_audio.audio.start_audio()

    pykka.ActorRegistry.stop_all()



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s] %(asctime)-15s | %(threadName)-14s| %(message)s')

    logging.info("Listening (ctrl-C to exit)...")

    try:
        kmain()
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')

