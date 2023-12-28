
import os
import sys
from threading import Thread
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
class PromptVars:
    System: str
    Prompt: str
    Response: str
    First: bool = False

    @classmethod
    def create(cls, *, system: str = '', prompt: str = '', first: bool = False):
        return cls(
            System=system,
            Prompt=prompt,
            Response="",
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

class LlamaSvc(Thread):

    def __init__(self, model: Model, stream_callback=None):
        super().__init__()
        self.model = model
        self.input_queue = queue.Queue(maxsize=10)

        self.is_running = True
        self.is_paused = False
        self.data_buffer = []

        self.llm = llama_cpp.Llama(model_path=model.ModelPath,
                                n_ctx=model.Options["n_ctx"],
                                n_threads=model.Options["n_threads"],
                                n_gpu_layers=model.Options["n_gpu_layers"],
                                verbose=False,)

    def send_data(self, data: PromptVars):
        """Send data to the input queue."""
        self.input_queue.put_nowait(data)

    def stop(self):
        self.is_running = False

    def pause_wait(self):
        self.is_paused = True
        # self.llm.reset()

    def resume(self):
        self.is_paused = False

    def run(self):
        """Run the thread. Read data from the input queue and dispatch it."""

        def stopping_criteria(a, b):
            return self.is_paused

        while self.is_running:
            try:
                data = self.input_queue.get(timeout=.1,)
                self.data_buffer.append(data)
            except queue.Empty:
                continue

            if self.is_paused:
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
                sentence = ''.join(buf)

                print('>>', sentence)


chat = LlamaSvc(model=model)
chat.start()

vars = PromptVars.create(prompt="How are you?", first=True)
chat.send_data(vars)
time.sleep(1)

chat.pause_wait()
chat.send_data(PromptVars.create(prompt="X=2"))
chat.send_data(PromptVars.create(prompt="Y=2"))
chat.send_data(PromptVars.create(prompt="X+Y="))
#chat.send_data(PromptVars.create(prompt="2 + 2 = "))
time.sleep(1)

chat.resume()
chat.send_data(PromptVars.create(prompt="What is the answer?"))
time.sleep(1)

chat.stop()
chat.join()

del chat.llm
time.sleep(1)
