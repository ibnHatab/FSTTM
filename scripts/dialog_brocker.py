"""
This is an example implementation of main.cpp from llama.cpp
Quirks:
 * Its not exactly alike since this port is designed around programmatic I/O
 * Input is always echoed if on, so it should be turned off when using "input()"
 * The first antiprompt should be the userprompt like "\nUser:",
   because its added when n_predict is reached (aka generation ended prematurely)
 * n_predict can be set to -1 for unlimited length responses (or just a really high value)
 * Instruction mode adds its own antiprompt.
   You should also still be feeding the model with a "primer" prompt that
   shows it the expected format.
"""
import ctypes
import sys
from time import time
from os import cpu_count, path

import llama_cpp
from common import GptParams
import util

# A LLaMA interactive session
class LLaMAInteract:
    def __init__(self, params: GptParams) -> None:
        # input args
        self.params = params

        if (self.params.seed <= 0):
            self.params.seed = int(time())

        print(f"seed = {self.params.seed}", file=sys.stderr)

        # runtime args
        self.input_consumed = 0
        self.n_past = 0
        self.n_session_consumed = 0
        self.first_antiprompt = []
        self.remaining_tokens = self.params.n_predict
        self.output_echo = self.params.input_echo
        self.multibyte_fix = []

        # model load
        self.lparams = llama_cpp.llama_context_default_params()
        self.lparams.n_ctx = self.params.n_ctx
        self.lparams.n_parts = self.params.n_parts
        self.lparams.seed = self.params.seed
        self.lparams.memory_f16 = self.params.memory_f16
        self.lparams.use_mlock = self.params.use_mlock
        self.lparams.use_mmap = self.params.use_mmap

        self.ctx = llama_cpp.llama_init_from_file(self.params.model.encode("utf8"), self.lparams)
        if (not self.ctx):
            raise RuntimeError(f"error: failed to load model '{self.params.model}'")

        if (self.params.ignore_eos):
            self.params.logit_bias[llama_cpp.llama_token_eos()] = -float("inf")

        if (len(self.params.lora_adapter) > 0):
            if (llama_cpp.llama_apply_lora_from_file(
                self.ctx,
                self.params.lora_adapter.encode("utf8"),
                self.params.lora_base.encode("utf8") if len(self.params.lora_base) > 0 else None,
                self.params.n_threads
            ) != 0):
                print("error: failed to apply lora adapter")
                return

        print(file=sys.stderr)
        print(f"system_info: n_threads = {self.params.n_threads} / {cpu_count()} \
| {llama_cpp.llama_print_system_info().decode('utf8')}", file=sys.stderr)

        # determine the required inference memory per token:
        if (self.params.mem_test):
            tmp = [0, 1, 2, 3]
            llama_cpp.llama_eval(self.ctx, (llama_cpp.c_int * len(tmp))(*tmp), len(tmp), 0, self.n_threads)
            llama_cpp.llama_print_timings(self.ctx)
            self.exit()
            return

        # create internal context
        self.n_ctx = llama_cpp.llama_n_ctx(self.ctx)

        # Add a space in front of the first character to match OG llama tokenizer behavior
        self.params.prompt = " " + self.params.prompt

        # tokenize the prompt
        self.embd = []
        self.embd_inp = self._tokenize(self.params.prompt)

        if (len(self.embd_inp) > self.n_ctx - 4):
            raise RuntimeError(f"error: prompt is too long ({len(self.embd_inp)} tokens, max {self.params.n_ctx - 4})")

        # number of tokens to keep when resetting context
        self.params.n_keep = len(self.embd_inp)


        # determine newline token
        self.llama_token_newline = self._tokenize("\n", False)
        self.llama_token_eot = self._tokenize(" [end of text]\n", False)

        if (self.params.verbose_prompt):
            print(f"""
prompt: '{self.params.prompt}'
number of tokens in prompt = {len(self.embd_inp)}""", file=sys.stderr)

            for i in range(len(self.embd_inp)):
                print(f"{self.embd_inp[i]} -> '{llama_cpp.llama_token_to_str(self.ctx, self.embd_inp[i])}'", file=sys.stderr)

            if (self.params.n_keep > 0):
                print("static prompt based on n_keep: '")
                for i in range(self.params.n_keep):
                    print(llama_cpp.llama_token_to_str(self.ctx, self.embd_inp[i]), file=sys.stderr)
                print("'", file=sys.stderr)
            print(file=sys.stderr)


        if (len(self.params.antiprompt) > 0):
            for antiprompt in self.params.antiprompt:
                print(f"Reverse prompt: '{antiprompt}'", file=sys.stderr)

        if len(self.params.input_prefix) > 0:
            print(f"Input prefix: '{self.params.input_prefix}'", file=sys.stderr)

        print(f"""sampling: repeat_last_n = {self.params.repeat_last_n},\
repeat_penalty = {self.params.repeat_penalty},\
presence_penalty = {self.params.presence_penalty},\
frequency_penalty = {self.params.frequency_penalty},\
top_k = {self.params.top_k},\
tfs_z = {self.params.tfs_z},\
top_p = {self.params.top_p},\
typical_p = {self.params.typical_p},\
temp = {self.params.temp},\
mirostat = {self.params.mirostat},\
mirostat_lr = {self.params.mirostat_eta},\
mirostat_ent = {self.params.mirostat_tau},\

generate: n_ctx = {self.n_ctx},\
n_batch = {self.params.n_batch},\
n_predict = {self.params.n_predict},\
n_keep = {self.params.n_keep}

""", file=sys.stderr)

        # determine antiprompt tokens
        for i in self.params.antiprompt:
            self.first_antiprompt.append(self._tokenize(i, False))

        self.last_n_tokens = [0]*self.n_ctx #TODO: deque doesnt support slices

        self.set_color(util.CONSOLE_COLOR_PROMPT)

    # tokenize a prompt
    def _tokenize(self, prompt, bos=True):
        _arr = (llama_cpp.llama_token * ((len(prompt) + 1) * 4))()
        _n = llama_cpp.llama_tokenize(self.ctx, prompt.encode("utf8", errors="ignore"), _arr, len(_arr), bos)
        return _arr[:_n]

    def set_color(self, c):
        if (self.params.use_color):
            print(c, end="")

    def use_antiprompt(self):
        return len(self.first_antiprompt) > 0

    # generate tokens
    def generate(self):
        while self.remaining_tokens > 0 or self.params.n_predict == -1:
            # predict
            if len(self.embd) > 0:
                # infinite text generation via context swapping
                # if we run out of context:
                # - take the n_keep first tokens from the original prompt (via n_past)
                # - take half of the last (n_ctx - n_keep) tokens and recompute the logits in a batch
                if (self.n_past + len(self.embd) > self.n_ctx):
                    n_left = self.n_past - self.params.n_keep
                    self.n_past = self.params.n_keep

                    # insert n_left/2 tokens at the start of embd from last_n_tokens
                    _insert = self.last_n_tokens[
                        self.n_ctx - int(n_left/2) - len(self.embd):-len(self.embd)
                    ]
                    self.embd = _insert + self.embd
                    self.params.path_session = ""


                if (llama_cpp.llama_eval(
                    self.ctx, (llama_cpp.llama_token * len(self.embd))(*self.embd), len(self.embd), self.n_past, self.params.n_threads
                ) != 0):
                    raise Exception("Failed to llama_eval!")


            self.n_past += len(self.embd)
            self.embd = []
            if len(self.embd_inp) <= self.input_consumed: #&& !is_interacting
                # out of user input, sample next token
                top_k = llama_cpp.llama_n_vocab(self.ctx) if self.params.top_k <= 0 else self.params.top_k
                repeat_last_n = self.n_ctx if self.params.repeat_last_n < 0 else self.params.repeat_last_n


                id = 0

                logits = llama_cpp.llama_get_logits(self.ctx)
                n_vocab = llama_cpp.llama_n_vocab(self.ctx)

                # Apply params.logit_bias map
                for key, value in self.params.logit_bias.items():
                    logits[key] += value

                _arr = (llama_cpp.llama_token_data * n_vocab)(*[
                    llama_cpp.llama_token_data(token_id, logits[token_id], 0.0)
                    for token_id in range(n_vocab)
                ])
                candidates_p = llama_cpp.ctypes.pointer(llama_cpp.llama_token_data_array(_arr, len(_arr), False))

                # Apply penalties
                nl_logit = logits[llama_cpp.llama_token_nl()]
                last_n_repeat = min(len(self.last_n_tokens), repeat_last_n, self.n_ctx)

                _arr = (llama_cpp.llama_token * last_n_repeat)(*self.last_n_tokens[len(self.last_n_tokens) - last_n_repeat:])
                llama_cpp.llama_sample_repetition_penalty(self.ctx, candidates_p,
                    _arr,
                    last_n_repeat, llama_cpp.c_float(self.params.repeat_penalty))
                llama_cpp.llama_sample_frequency_and_presence_penalties(self.ctx, candidates_p,
                    _arr,
                    last_n_repeat, llama_cpp.c_float(self.params.frequency_penalty), llama_cpp.c_float(self.params.presence_penalty))

                if not self.params.penalize_nl:
                    logits[llama_cpp.llama_token_nl()] = nl_logit

                if self.params.temp <= 0:
                    # Greedy sampling
                    id = llama_cpp.llama_sample_token_greedy(self.ctx, candidates_p)
                else:
                    if self.params.mirostat == 1:
                        mirostat_mu = 2.0 * self.params.mirostat_tau
                        mirostat_m = 100
                        llama_cpp.llama_sample_temperature(self.ctx, candidates_p, llama_cpp.c_float(self.params.temp))
                        id = llama_cpp.llama_sample_token_mirostat(self.ctx, candidates_p, llama_cpp.c_float(self.params.mirostat_tau), llama_cpp.c_float(self.params.mirostat_eta), llama_cpp.c_int(mirostat_m), llama_cpp.c_float(mirostat_mu))
                    elif self.params.mirostat == 2:
                        mirostat_mu = 2.0 * self.params.mirostat_tau
                        llama_cpp.llama_sample_temperature(self.ctx, candidates_p, llama_cpp.c_float(self.params.temp))
                        id = llama_cpp.llama_sample_token_mirostat_v2(self.ctx, candidates_p, llama_cpp.c_float(self.params.mirostat_tau), llama_cpp.c_float(self.params.mirostat_eta), llama_cpp.c_float(mirostat_mu))
                    else:
                        # Temperature sampling
                        llama_cpp.llama_sample_top_k(self.ctx, candidates_p, top_k, min_keep=llama_cpp.c_size_t(1))
                        llama_cpp.llama_sample_tail_free(self.ctx, candidates_p, llama_cpp.c_float(self.params.tfs_z), min_keep=llama_cpp.c_size_t(1))
                        llama_cpp.llama_sample_typical(self.ctx, candidates_p, llama_cpp.c_float(self.params.typical_p), min_keep=llama_cpp.c_size_t(1))
                        llama_cpp.llama_sample_top_p(self.ctx, candidates_p, llama_cpp.c_float(self.params.top_p), min_keep=llama_cpp.c_size_t(1))
                        llama_cpp.llama_sample_temperature(self.ctx, candidates_p, llama_cpp.c_float(self.params.temp))
                        id = llama_cpp.llama_sample_token(self.ctx, candidates_p)
                # print("`{}`".format(candidates_p.size))

                self.last_n_tokens.pop(0)
                self.last_n_tokens.append(id)

                # replace end of text token with newline token when in interactive mode
                if (id == llama_cpp.llama_token_eos()):
                    id = self.llama_token_newline[0]
                    self.embd.append(id)
                    if (self.use_antiprompt()):
                        # tokenize and inject first reverse prompt
                        self.embd_inp += self.first_antiprompt[0]
                        for id in self.first_antiprompt[0]:
                            self.embd.append(id)
                else:
                    # add it to the context
                    self.embd.append(id)

                # echo this to console
                self.output_echo = True

                # decrement remaining sampling budget
                self.remaining_tokens -= 1
            else:
                # output to console if input echo is on
                self.output_echo = self.params.input_echo

                # some user input remains from prompt or interaction, forward it to processing
                while len(self.embd_inp) > self.input_consumed:
                    self.embd.append(self.embd_inp[self.input_consumed])
                    self.last_n_tokens.pop(0)
                    self.last_n_tokens.append(self.embd_inp[self.input_consumed])
                    self.input_consumed += 1
                    if len(self.embd) >= self.params.n_batch:
                        break

            # display tokens
            if self.output_echo:
                for id in self.embd:
                    yield id

            # reset color to default if we there is no pending user input
            if (self.params.input_echo and len(self.embd_inp) == self.input_consumed):
                self.set_color(util.CONSOLE_COLOR_DEFAULT)

            if (len(self.embd_inp) <= self.input_consumed):
                # if antiprompt is present, stop
                if (self.use_antiprompt()):
                    if True in [
                        i == self.last_n_tokens[-len(i):]
                        for i in self.first_antiprompt
                    ]:
                        break

            # end of text token
            if len(self.embd) > 0 and self.embd[-1] == llama_cpp.llama_token_eos():
                for i in self.llama_token_eot:
                    yield i
                break

            # respect n_predict even if antiprompt is present
            if (self.remaining_tokens <= 0 and self.params.n_predict != -1):
                self.embd_inp += self.first_antiprompt[0]
                self.n_remain = self.params.n_predict
                break

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.exit()

    def exit(self):
        llama_cpp.llama_free(self.ctx)
        self.set_color(util.CONSOLE_COLOR_DEFAULT)

    # return past text
    def past(self):
        for id in self.last_n_tokens[-self.n_past:]:
            yield llama_cpp.llama_token_to_str(self.ctx, id).decode("utf8", errors="ignore")

    # write input
    def input(self, prompt: str):
        self.embd_inp += self._tokenize(prompt)

    # write output
    def output(self):
        self.remaining_tokens = self.params.n_predict
        for id in self.generate():
            cur_char = llama_cpp.llama_token_to_str(self.ctx, id)

            # Add remainder of missing bytes
            if None in self.multibyte_fix:
                self.multibyte_fix[self.multibyte_fix.index(None)] = cur_char

            # Return completed utf char
            if len(self.multibyte_fix) > 0 and not None in self.multibyte_fix:
                yield (b"".join(self.multibyte_fix)).decode("utf8")
                self.multibyte_fix = []
                continue

            # Contains multi-byte UTF8
            for num, pattern in [(2, 192), (3, 224), (4, 240)]:
                # Bitwise AND check
                if pattern & int.from_bytes(cur_char, 'little') == pattern:
                    self.multibyte_fix = [cur_char] + ([None] * (num-1))

            # Stop incomplete bytes from passing
            if len(self.multibyte_fix) > 0:
                continue

            yield cur_char.decode("utf8")

    # read user input
    def read_input(self):
        out = ""
        while (t := input()).endswith("\\"):
            out += t[:-1] + "\n"
        return out + t + "\n"

    # interactive mode
    def interact(self):
        for i in self.output():
            print(i,end="",flush=True)
        self.params.input_echo = False

        while True:
            self.set_color(util.CONSOLE_COLOR_USER_INPUT)
            print(self.params.input_prefix, end="")
            self.input(f"{self.params.input_prefix}{self.read_input()}{self.params.input_suffix}")
            print(self.params.input_suffix,end="")
            self.set_color(util.CONSOLE_COLOR_DEFAULT)

            try:
                for i in self.output():
                    print(i,end="",flush=True)
            except KeyboardInterrupt:
                self.set_color(util.CONSOLE_COLOR_DEFAULT)
                print(self.params.fix_prefix,end="")
                self.input(self.params.fix_prefix)

if __name__ == "__main__":
    from datetime import datetime
    MODEL = "./models/7B/ggml-vicuna-7b-1.1-q5_1.bin"
    N_PREDICTS = 2048
    N_THREAD = 8

    prompt = """A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's questions."""

    USER_NAME="USER"
    AI_NAME="ASSISTANT"
    print("Loading model...")
    params = GptParams(
        seed=42,
        n_ctx=2048,
        temp=0.7,
        top_k=40,
        top_p=0.5,
        repeat_last_n=256,
        n_batch=1024,
        repeat_penalty=1.17647,
        model=MODEL,
        n_threads=N_THREAD,
        n_predict=N_PREDICTS,
        use_color=True,
#        interactive=True,
        antiprompt=[f"{USER_NAME}:"],
        input_prefix=" ",
        input_suffix=f"{AI_NAME}:",
        prompt=prompt,
    )

    with LLaMAInteract(params) as m:
        m.interact()
