
import asyncio
import threading
import queue

from dataclasses import dataclass
import time
from typing import List, Dict

import llama_cpp


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
        return system + m.Template.format(Prompt=self.Prompt)

@dataclass
class ResponseVars:
    Response: str
    Last: bool = False

import threading
import queue

class LlamaSvcThread(threading.Thread):
    """
    A thread class for handling communication with the Llama model.

    Args:
        model (Model): The Llama model.
        output_queue (queue.Queue): The queue for storing output responses.

    Attributes:
        model (Model): The Llama model.
        input_queue (queue.Queue): The queue for storing input data.
        output_queue (queue.Queue): The queue for storing output responses.
        data_buffer (list): A buffer to collect user inputs.
        llm (llama_cpp.Llama): The Llama model instance.
        _lock (threading.Lock): A lock for thread synchronization.
        _stopped (threading.Event): An event to indicate if the thread is stopped.
        _paused (threading.Event): An event to indicate if the thread is paused.
    """

    def __init__(self, model: Model, output_queue: queue.Queue):
        super().__init__()
        self.model = model
        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = output_queue

        self.data_buffer = [] # collect User: inputs
        # FIXME: don't collect, just send to llama

        self.llm = llama_cpp.Llama(model_path=model.ModelPath,
                                n_ctx=model.Options["n_ctx"],
                                n_threads=model.Options["n_threads"],
                                n_gpu_layers=model.Options["n_gpu_layers"],
                                verbose=False,)

        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._paused = threading.Event()
        self.start()

    def send(self, data: PromptVars):
        """
        Sends input data to the input queue.

        Args:
            data (PromptVars): The input data to be sent.
        """
        self.input_queue.put_nowait(data)

    def stop(self):
        """
        Stops the thread.
        """
        self._stopped.set()
        self.join()

    def pause(self):
        """
        Pauses the model generator.
        """
        self._paused.set()
        # FIXme: self.llm.reset()

    def resume(self):
        """
        Resumes the model generator.
        """
        self._paused.clear()

    def run(self):
        """
        The main execution logic of the thread.
        """

        def stopping_criteria(a, b):
            return self._paused.is_set()

        while not self._stopped.is_set():
            try:
                data = self.input_queue.get(timeout=.1,)
                self.data_buffer.append(data)
            except queue.Empty:
                continue

            if self._paused.is_set():
                continue

            if len(self.data_buffer):
                fst = self.data_buffer[0]
                fst.Prompt = ' '.join([i.Prompt for i in self.data_buffer])
                prompt = fst.format(self.model)
                self.data_buffer.clear()
                for res in self.llm.create_completion(prompt=prompt,
                                                    stream=True,
                                                    stop=self.model.Options["stop"],
                                                    max_tokens=256,
                                                    echo=True,
                                                    stopping_criteria=stopping_criteria):
                    text = res["choices"][0]["text"]
                    last = res["choices"][0]["finish_reason"] == "stop"
                    out = ResponseVars(Response=text, Last=last)
                    print('x', end='', flush=True)
                    self.output_queue.put_nowait(out)
                    time.sleep(0.1) # yield to other threads



class LlamaSvcProxy:
    """
    A class representing a proxy for a Llama service.

    Attributes:
        queue (asyncio.Queue): An asyncio queue to store values.
        thread_queue (queue.Queue): A thread-safe queue to store values.
        periodic_generator (LlamaSvcThread): An instance of LlamaSvcThread for generating values periodically.
    """

    def __init__(self, model) -> None:
        """
        Initializes a new instance of the LlamaSvcProxy class.

        Args:
            model: The model to be used for generating values.
        """
        self.queue = asyncio.Queue()
        self.thread_queue = queue.Queue()
        self.periodic_generator = LlamaSvcThread(model, self.thread_queue)
        self._generator_active = False

    async def async_generator(self):
        """
        An asynchronous generator that yields values from the queue.

        Yields:
            value: The value retrieved from the queue.
        """
        while True:
            value = await self.queue.get()
            if not self.generator_active:
                self.generator_active = True
            yield value
            if value.Last:
                self.generator_active = False
                break

    async def run_periodic_generator(self):
        """
        Runs the periodic generator to generate values and put them into the queue.
        """
        while True:
            with self.periodic_generator._lock:
                try:
                    value = self.thread_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.1)
                    continue
            await self.queue.put(value)

    def stop(self):
        """
        Stops the periodic generator.
        """
        self.periodic_generator.stop()

    async def send(self, data: PromptVars):
        """
        Sends data to the periodic generator.

        Args:
            data (PromptVars): The data to be sent.
        """
        self.periodic_generator.send(data)

    @property
    def generator_active(self):
        return self._generator_active

    @generator_active.setter
    def generator_active(self, val: bool):
        self._generator_active = val
        self.generator_active_ind(self._generator_active)

    def generator_active_ind(self, active: bool):
        print(f"{'*' if active else '.'}", end='', flush=True)
        pass



if __name__ == "__main__":

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
        #await asyncio.sleep(1)
        await async_proxy.stop()

    # Run the event loop
    asyncio.run(main())


