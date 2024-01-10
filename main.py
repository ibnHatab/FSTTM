
from datetime import datetime
import itertools
import time
from typing import Optional, Dict, Any, AsyncGenerator
import asyncio
import os
from chat import Model, PromptVars, LlamaSvcProxy
from fsttm import Dialog
from mic_vad import VADAudio
from speech_to_text import SpeechToTextProxy, Whisper
from text_to_speech import TTSPlayThread



model = Model(
    Name="phi-2.Q5_K_S",
    ShortName="phi-2",
    ModelPath="./models/phi-2.Q5_K_S.gguf",
    Template="User: {Prompt}\nAssistant:",
    Designation="System:",
    System="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful answers to the user's questions.",
    Options={
        "chat_format": "llama-2",
        "n_ctx": 2048,
        "n_threads": 8,
        "n_gpu_layers": 35,
        "stop": ["<|endoftext|>", "User:", "Assistant:", "System:"]
    },
)


class SpeechToText(SpeechToTextProxy):
    def __init__(self, dialog: Dialog, vad: VADAudio, stt: Whisper) -> None:
        super().__init__(vad, stt)
        self.dialog = dialog
        self.dialog.user_cb = self.floor_switch_ind

    def floor_switch_ind(self, action: str, floor: bool):
        pass

    def voice_active_ind(self, active: bool):
        if active:
            self.dialog.user_action('G')
        else:
            self.dialog.user_action('R')

class LlamaSvc(LlamaSvcProxy):
    def __init__(self, dialog: Dialog, model: Model, speaker: TTSPlayThread) -> None:
        super().__init__(model)
        self.dialog = Dialog()
        self.dialog.system_cb = self.floor_switch_ind
        self.speaker = speaker

    def floor_switch_ind(self, action: str, floor: bool):
        print('++ system > ' + action + ' ' + str(floor))
        # if floor:
        #     self.speaker.clear()

    def generator_active_ind(self, active: bool):
        if active:
            self.dialog.system_action('G')
            # if not self.can_speak:
            #     self.stop()
        else:
            self.dialog.system_action('R')


async def amain():

    dialog = Dialog()

    vad_audio = VADAudio(aggressiveness=3, device=0, input_rate=16000)
    whisper = Whisper(model_name='base.en')
    stt_svc = SpeechToText(dialog, vad_audio, whisper)

    aplay = TTSPlayThread()
    llama_svc = LlamaSvc(dialog, model, aplay)

    # dialog = Model()
    # dialog.state = 'USER'


    # Start the periodic generator in a separate task
    asyncio.create_task(llama_svc.run_periodic_generator())

    stt_svc.start()
    first = True

    user_input = stt_svc.async_generator()

    # Define a starting timestamp
    start_timestamp = datetime.now()

    # Infinite loop generating an index and timestamp
    # for index, timestamp in enumerate(itertools.count()):
    #     current_time = start_timestamp + datetime.timedelta(seconds=index)
    #     print(f"Index: {index}, Timestamp: {current_time}")

    #     try:
    #         user_say = await  user_input.__anext__()
    #     except StopAsyncIteration:
    #         break

    async for user_say in user_input:

        print(f"\n{user_say}")

        await llama_svc.send(PromptVars.create(prompt=user_say.Uterance, first=first))
        first = False

        async_gen = llama_svc.async_generator()
        sentence = ""
        async for value in async_gen:
            word = value.Response
            sentence += word
            if word.endswith('.') or word.endswith(',') or word.endswith('!') or word.endswith('?'):
                print(f"Received value: {sentence}")
                aplay.say(sentence)
                sentence = ""

        time.sleep(.3)  # Adjust this delay as needed

    # Stop the generator
    await llama_svc.stop()
    all_tasks = asyncio.all_tasks()
    await asyncio.wait(all_tasks)

# Run the event loop
asyncio.run(amain())
