
from threading import Thread
import queue
from collections import namedtuple, deque

import llama_cpp

import gpt_fsttm_server.conversation as conversation

# Serrvice protocol
AddUser = namedtuple('AddUser', ['text', 'context'])
AddUser.__new__.__defaults__ = (None, None)

StopGenerate = namedtuple('StopGenerate', [])
# backchannel messages
ContinueGenerate = namedtuple('ContinueGenerate', ['max_tokens'])

AddSystem = namedtuple('AddSystem', ['prompt', 'idx'])
AddSystem.__new__.__defaults__ = (None,)

RemoveSystem = namedtuple('RemoveSystem', ['prompt', 'idx'])
RemoveSystem.__new__.__defaults__ = (None, None)


class LlamaSvc(Thread):

    def __init__(self, params, stream_callback=None):
        super().__init__()
        self.stream_callback = stream_callback
        self.input_queue = queue.Queue(maxsize=10)
        self.is_running = True
        self.params = params
        self.conversation = conversation.get_conv_template(params.conversation)

		# model load
        self.lparams = llama_cpp.llama_context_default_params()
        self.lparams.n_ctx = self.params.n_ctx
        self.lparams.seed = self.params.seed
        self.lparams.memory_f16 = True
        self.lparams.use_mlock = False
        self.lparams.use_mmap = True

        self.ctx = llama_cpp.llama_init_from_file(self.params.model.encode("utf8"), self.lparams)
        if (not self.ctx):
            raise RuntimeError(f"error: failed to load model '{self.params.model}'")

        # create internal context
        self.n_ctx = llama_cpp.llama_n_ctx(self.ctx)
        self.n_past = 0

        # Add a space in front of the first character to match OG llama tokenizer behavior
        prompt = " " + self.conversation.get_prompt()

        self.last_n_tokens = deque(maxlen=self.n_ctx)

		# tokenize the prompt
        self.embd = []
        self.embd_inp = self._tokenize(self.params.prompt)
        self.n_keep = len(self.embd_inp)

        # determine newline token
        self.llama_token_newline = self._tokenize("\n", False)
        self.llama_token_eot = self._tokenize(" [end of text]\n", False)

    def _tokenize(self, prompt, bos=True):
        _arr = (llama_cpp.llama_token * ((len(prompt) + 1) * 4))()
        _n = llama_cpp.llama_tokenize(self.ctx, prompt.encode("utf8", errors="ignore"), _arr, len(_arr), bos)
        return _arr[:_n]

    def send_data(self, data):
        """Send data to the input queue."""
        self.input_queue.put_nowait(data)

    def run(self):
        """Run the thread. Read data from the input queue and dispatch it."""
        while self.is_running:
            try:
                data = self.input_queue.get(timeout=.1,)
            except queue.Empty:
                continue
            if data:
                self.dispatch(data)
            self.generate_tokens(100)

    def generate_tokens(self, n_tokens=1):
        output_tokens = []

        for _ in range(n_tokens):

            if (llama_cpp.llama_eval(
                self.ctx, (llama_cpp.llama_token * len(self.embd))(*self.embd), len(self.embd), self.n_past, self.params.n_threads
            ) != 0):
                raise Exception("Failed to llama_eval!")

            self.n_past += len(self.embd)
            self.embd = []

            id = 0

            logits = llama_cpp.llama_get_logits(self.ctx)
            n_vocab = llama_cpp.llama_n_vocab(self.ctx)


            _arr = (llama_cpp.llama_token_data * n_vocab)(*[
                llama_cpp.llama_token_data(token_id, logits[token_id], 0.0)
                for token_id in range(n_vocab)
            ])
            candidates_p = llama_cpp.ctypes.pointer(llama_cpp.llama_token_data_array(_arr, len(_arr), False))

            # Apply penalties
            nl_logit = logits[llama_cpp.llama_token_nl()]
            last_n_repeat = min(len(self.last_n_tokens), self.params.repeat_last_n, self.n_ctx)

            _arr = (llama_cpp.llama_token * last_n_repeat)(*self.last_n_tokens[len(self.last_n_tokens) - last_n_repeat:])
            llama_cpp.llama_sample_repetition_penalty(self.ctx, candidates_p,
                _arr,
                last_n_repeat, llama_cpp.c_float(self.params.repeat_penalty))
            llama_cpp.llama_sample_frequency_and_presence_penalties(self.ctx, candidates_p,
                _arr,
                last_n_repeat, llama_cpp.c_float(self.params.frequency_penalty), llama_cpp.c_float(self.params.presence_penalty))

            # Temperature sampling
            llama_cpp.llama_sample_top_k(self.ctx, candidates_p, top_k, min_keep=llama_cpp.c_size_t(1))
            llama_cpp.llama_sample_tail_free(self.ctx, candidates_p, llama_cpp.c_float(self.params.tfs_z), min_keep=llama_cpp.c_size_t(1))
            llama_cpp.llama_sample_typical(self.ctx, candidates_p, llama_cpp.c_float(self.params.typical_p), min_keep=llama_cpp.c_size_t(1))
            llama_cpp.llama_sample_top_p(self.ctx, candidates_p, llama_cpp.c_float(self.params.top_p), min_keep=llama_cpp.c_size_t(1))
            llama_cpp.llama_sample_temperature(self.ctx, candidates_p, llama_cpp.c_float(self.params.temp))
            id = llama_cpp.llama_sample_token(self.ctx, candidates_p)

            if id == llama_cpp.llama_token_eos():
                id = self.llama_token_newline[0]
                self.embd.append(id)
                self.embd += self.first_antiprompt[0]
                self.embd_inp += self.first_antiprompt[0]
            else:
                self.embd.append(id)


        last_generated = self.conversation.get_last_generated()
        if not last_generated:
            last_generated = self.conversation.roles[1], ""

        last_generated.test.extend(output_tokens)
        self.stream_callback(self.conversation.get_last_generated())

    def add_user(self, data):
        self.conversation.add_user(data.text)

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
        if isinstance(data, AddUser):
            self.add_user(data)
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
    server.send_data(AddUser("Hello Word!"))
    time.sleep(1)
    server.send_data(StopGenerate())
    time.sleep(1)
    server.stop()
    server.join()





