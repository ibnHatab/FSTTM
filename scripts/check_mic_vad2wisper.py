import os
import asyncio
import numpy as np

from gpt_fsttm_server.mic_vad import VADAudio
from gpt_fsttm_server.rhvoice import Player

import whispercpp as w


async def amain(loop):
    model = w.Whisper.from_pretrained('base.en')
    params = (model.params
              .with_print_realtime(True)
              .with_translate(False)
              .build())
    print('>>', params)
    # player = Player()

    vad_audio = VADAudio(loop,
                        aggressiveness=3,
                        device=0,
                        input_rate=16000)
    print("Listening (ctrl-C to exit)...")
    vad_audio.start()

    data = []


    async for frame in  vad_audio.vad_collector():
        if frame is not None:
            ##print("streaming frame: ", len(frame))
            os.write(1, b'.')
            data.append(frame)
        else:
            print()
            print(">> end of utterence: ", len(data))
            buffer = b''.join(data)
            audio = np.frombuffer(buffer, np.int16).flatten().astype(np.float32) / 32768.0
            if len(data) < 50:
                audio = np.pad(audio, (0, 16000), 'constant')
            text = model.transcribe(audio)
            print('>>', text)
            # player.say(text)
            text = ''
            data = []

if __name__ == '__main__':
    import sys

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    asyncio.ensure_future(amain(loop=loop))
    # loop.set_debug(True)
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')
