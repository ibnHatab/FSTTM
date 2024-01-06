
from datetime import datetime
import itertools
import time
from typing import Optional, Dict, Any, AsyncGenerator
import asyncio
import os
from chat import Model, PromptVars, LlamaSvcProxy
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

async def main():
    vad_audio = VADAudio(aggressiveness=3, device=0, input_rate=16000)
    whisper = Whisper(model_name='base')
    stt_svc = SpeechToTextProxy(vad_audio, whisper)
    llama_svc = LlamaSvcProxy(model)
    aplay = TTSPlayThread()

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
            sentence += value.Response
        print(f"Received value: {sentence}")
        aplay.say(sentence)

        # Simulate some processing time
        time.sleep(1)  # Adjust this delay as needed

    # Stop the generator
    await llama_svc.stop()

# Run the event loop
asyncio.run(main())
