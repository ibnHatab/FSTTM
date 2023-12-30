
import asyncio
import os
import sys
from threading import Thread
import threading
import time
import llama_cpp

from dataclasses import dataclass
import queue
from typing import List, Dict


@dataclass
class Model:
    Name: str
    ShortName: str
    ModelPath: str
    Template: str
    Designation: str
    System: str
    Options: Dict[str, any]

@dataclass
class ResponseVars:
    Response: str
    Last: bool = False

@dataclass
class PromptVars:
    System: str
    Prompt: str
    First: bool = False

    @classmethod
    def create(cls, *, system: str = '', prompt: str = '', first: bool = False):
        return cls(
            System=system,
            Prompt=prompt,
            First=first,
        )

    def format(self, m: Model):
        if self.First and self.System == "":
            system = f"{m.Designation} {m.System}\n"
        elif self.System != "":
            system = f"{m.Designation} {self.System}\n"
        else:
            system = ""
        return system + model.Template.format(Prompt=self.Prompt)


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

class LlamaSvcThread:

    def __init__(self, model: Model, output_queue: queue.Queue):
        super().__init__()
        self.model = model
        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = output_queue

        self.data_buffer = []

        self.llm = llama_cpp.Llama(model_path=model.ModelPath,
                                n_ctx=model.Options["n_ctx"],
                                n_threads=model.Options["n_threads"],
                                n_gpu_layers=model.Options["n_gpu_layers"],
                                verbose=False,)

        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.paused = threading.Event()
        self.thread = threading.Thread(target=self.generate_values)
        self.thread.start()

    def send(self, data: PromptVars):
        """Send data to the input queue."""
        self.input_queue.put_nowait(data)

    def stop(self):
        self.stopped.set()
        self.thread.join()

    def pause(self):
        self.paused.set()
        # FIXme: self.llm.reset()

    def resume(self):
        self.paused.unset()

    def generate_values(self):
        """Run the thread. Read data from the input queue and dispatch it."""

        def stopping_criteria(a, b):
            return self.paused.is_set()

        while not self.stopped.is_set():
            try:
                data = self.input_queue.get(timeout=.1,)
                self.data_buffer.append(data)
            except queue.Empty:
                continue

            if self.paused.is_set():
                continue

            if len(self.data_buffer):
                fst = self.data_buffer[0]
                fst.Prompt = ' '.join([i.Prompt for i in self.data_buffer])
                prompt = fst.format(self.model)
                self.data_buffer.clear()
                print('gen >>', prompt)
                buf = []
                for res in self.llm.create_completion(prompt=prompt,
                                                    stream=True,
                                                    stop=model.Options["stop"],
                                                    max_tokens=256,
                                                    echo=True,
                                                    stopping_criteria=stopping_criteria):
                    t = res["choices"][0]["text"]
                    buf.append(t)
                print(res)
                last = res["choices"][0]["finish_reason"] == "stop"
                sentence = ''.join(buf)
                print('>>', sentence)
                #FIXME: split into sentences
                out = ResponseVars(Response=sentence, Last=last)
                self.output_queue.put_nowait(out)


class LlamaSvcProxy:
    def __init__(self, model) -> None:
        self.queue = asyncio.Queue()

        self.thread_queue = queue.Queue()
        self.periodic_generator = LlamaSvcThread(model, self.thread_queue)

    async def async_generator(self):
        while True:
            value = await self.queue.get()
            yield value

    async def run_periodic_generator(self):
        while True:
            with self.periodic_generator.lock:
                try:
                    value = self.thread_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.1)
                    continue
            await self.queue.put(value)
            await asyncio.sleep(0.1)

    async def stop(self):
        self.periodic_generator.stop()

    async def send(self, data: PromptVars):
        self.periodic_generator.send(data)

async def main():
    async_proxy = LlamaSvcProxy(model)

    # Start the periodic generator in a separate task
    asyncio.create_task(async_proxy.run_periodic_generator())

    # Get async generator from the proxy
    async_gen = async_proxy.async_generator()

    await async_proxy.send(PromptVars.create(prompt="How are you?", first=True))
    # Consume values from the async generator
    for _ in range(10):
        value = await async_gen.__anext__()
        print(f"Received value: {value}")

    # Stop the generator
    await async_proxy.stop()

# Run the event loop
asyncio.run(main())


