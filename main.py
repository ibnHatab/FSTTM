
from typing import Optional, Dict, Any, AsyncGenerator
import asyncio
import os
from chat import Model, PromptVars, LlamaSvcProxy



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
    async_proxy = LlamaSvcProxy(model)

    # Start the periodic generator in a separate task
    asyncio.create_task(async_proxy.run_periodic_generator())

    await async_proxy.send(PromptVars.create(prompt="How are you?", first=True))
    async_gen = async_proxy.async_generator()
    async for value in async_gen:
        print(f"Received value: {value}")

    await async_proxy.send(PromptVars.create(prompt="What is the weather today?", first=False))
    async_gen = async_proxy.async_generator()
    async for value in async_gen:
        print(f"Received value: {value}")

    # Stop the generator
    await async_proxy.stop()

# Run the event loop
asyncio.run(main())
