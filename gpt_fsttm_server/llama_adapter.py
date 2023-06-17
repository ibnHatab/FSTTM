
from threading import Thread
import queue
from collections import namedtuple

import llama_cpp

import gpt_fsttm_server.conversation as conversation

Generate = namedtuple('Generate', ['text', 'context'])
Generate.__new__.__defaults__ = (None, None)

StopGenerate = namedtuple('StopGenerate', [])
ContinueGenerate = namedtuple('ContinueGenerate', ['max_tokens'])

AddSystem = namedtuple('AddSystem', ['prompt', 'id'])
AddSystem.__new__.__defaults__ = (None,)
RemoveSystem = namedtuple('RemoveSystem', ['prompt', 'id'])
RemoveSystem.__new__.__defaults__ = (None, None)


class LlamaSvc(Thread):

    def __init__(self, params, stream_callback=None):
        super().__init__()
        self.stream_callback = stream_callback
        self.input_queue = queue.Queue(maxsize=10)
        self.is_running = True
        self.params = params
        self.conversation = conversation.get_conv_template(params.conversation)


    def send_data(self, data):
        self.input_queue.put_nowait(data)

    def run(self):
        while self.is_running:
            try:
                data = self.input_queue.get(timeout=1, )
            except queue.Empty:
                continue
            if not data:
                continue
            self.dispatch(data)
        #    self.stream_callback(data)

    def generate(self, data):
        self.conversation.add_prompt(data.text, data.context)
        self.conversation.generate(self.params)
        self.stream_callback(self.conversation.get_last_generated())

    def stop_generate(self, data):
        self.conversation.stop_generate()

    def continue_generate(self, data):
        self.conversation.continue_generate(data.max_tokens)
        self.stream_callback(self.conversation.get_last_generated())

    def add_system(self, data):
        self.conversation.add_system(data.prompt, data.id)

    def remove_system(self, data):
        self.conversation.remove_system(data.prompt, data.id)

    def dispatch(self, data):
        if isinstance(data, Generate):
            self.generate(data)
        elif isinstance(data, StopGenerate):
            self.stop_generate(data)
        elif isinstance(data, ContinueGenerate):
            self.continue_generate(data)
        elif isinstance(data, AddSystem):
            self.add_system(data)
        elif isinstance(data, RemoveSystem):
            self.remove_system(data)
        else:
            raise ValueError(f"Unknown data type: {data}")

    def stop(self):
        self.is_running = False



if __name__ == '__main__':
    import time
    from gpt_fsttm_server.config import GptParams

    def cb(data):
        print(f">>: {data}")

    params = GptParams(
        n_ctx=2048,
        temp=0.7,
        top_k=40,
        top_p=0.5,
        repeat_last_n=256,
        n_batch=1024,
        repeat_penalty=1.17647,
        n_threads=8,
        n_predict=2048,
        model='./models/7B/ggml-vicuna-7b-1.1-q5_1.bin',
        safeword='cow',
        conversation='vicuna_v1.1',
    )


    server = LlamaSvc(params, stream_callback=cb)

    server.start()
    server.send_data("Hello")
    server.send_data("World")
    server.send_data("!")
    time.sleep(1)
    server.stop()
    server.join()





