
from typing import Optional, Dict, Any, AsyncGenerator
import asyncio
import os
from chat import Model, PromptVars, LlamaSvcProxy
from mic_vad import VADAudio
from speech_to_text import SpeechToTextProxy, Whisper
from text_to_speech import APlayThread



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
    vad_audio = VADAudio(aggressiveness=3,
                        device=0,
                        input_rate=16000)

    whisper = Whisper(model_name='base')
    stt_proxy = SpeechToTextProxy(vad_audio, whisper)
    llama_proxy = LlamaSvcProxy(model)
    aplay = APlayThread()

    # Start the periodic generator in a separate task
    asyncio.create_task(llama_proxy.run_periodic_generator())

    stt_proxy.start()
    first = True

    async for user_say in stt_proxy.async_generator():
            print(f"\n{user_say}")

            await llama_proxy.send(PromptVars.create(prompt=user_say.Uterance, first=first))
            first = False

            async_gen = llama_proxy.async_generator()
            sentence = ""
            async for value in async_gen:
                sentence += value.Response
            print(f"Received value: {sentence}")
            aplay.say(sentence)

    # Stop the generator
    await llama_proxy.stop()

# Run the event loop
asyncio.run(main())
