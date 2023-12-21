
import asyncio
import numpy as np

import whispercpp as w

# High-performance inference of OpenAI's Whisper
# automatic speech recognition (ASR) model

class SpeechToText:
    def __init__(self, loop, model_name, n_threads=7):
        self.loop = loop
        self.model = w.Whisper.from_pretrained(model_name)
        params = (self.model.params
                      .with_print_realtime(False)
                      .with_num_threads(n_threads)
                      .with_suppress_blank(True)
                      .build())

    async def process_data(self, data):
        text = None
        try:
            # print(f"\nReceive speech: {len(item.data)/640}")
            audio = np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0
            # Compensate for the fact that the model cant process short audio
            # https://github.com/ggerganov/whisper.cpp/issues/39
            if len(data)/640 < 50:
                audio = np.pad(audio, (0, 16000), 'constant')
            text = self.model.transcribe(audio)
        except Exception as e:
            print(f"Whisper error: {e}")

        return text

if __name__ == '__main__':
    import os
    import sys
    import time

    from mic_vad import VADAudio


    print(sys.path)
    async def amain(loop):
        stt_svc = SpeechToText(loop=loop, model_name='base')

        vad_audio = VADAudio(loop,
                            aggressiveness=3,
                            device=0,
                            input_rate=16000)
        print("Listening (ctrl-C to exit)...")
        vad_audio.start()

        n = 0
        t = time.time_ns()
        uterance = bytearray()
        async for frame in  vad_audio.vad_collector():
            if frame is not None:
                if not t: t = time.time_ns()
                n += 1
                os.write(sys.stdout.fileno(), b'.')
                # print("streaming frame: {}".format(len(frame)))
                uterance.extend(frame)
            else:
                tt = time.time_ns() - t
                tt = tt/1e9
                print()
                print("end of utterence: {}f / {}s = {}f/s".format(n, tt, int(n/tt)))
                n = 0
                t = 0 # time.time_ns()
                text = await stt_svc.process_data(uterance)
                uterance.clear()
                print(f"Text: {text}")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    asyncio.ensure_future(amain(loop=loop))
    loop.set_debug(False)
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')




