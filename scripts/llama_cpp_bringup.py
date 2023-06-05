from llama_cpp import Llama

import llama_cpp as lpp
import ctypes

from gpt_fsttm_server.utils import ignoreStderr

model_path="./models/7B/ggml-vicuna-7b-1.1-q5_1.bin"

with ignoreStderr():
    llama = Llama(model_path, embedding=True)

embedings = llama.create_embedding("Hello world!")


output = llama("Q: Name the planets in the solar system? A: ",
               max_tokens=32,
               stop=["Q:", "\n"],
               echo=True)
print(output)



def trace_stop_callback(trace, scores):
    ret = False
    print("stop callback")
    print(llama.detokenize(trace))
    print(len(scores))
    if len(trace) > 10:
        ret = True
    return ret

tokens = llama.tokenize(b"Hello, world!")
for token in llama.generate(tokens, top_k=40, top_p=0.95, temp=1.0, repeat_penalty=1.1, stopping_criteria=trace_stop_callback):
    print(llama.detokenize([token]))


llama.detokenize([
    llama.token_bos(),
    llama.token_eos(),
    llama.token_nl(),
    ])




# params = llama_cpp.llama_context_default_params()

# ctx = llama_cpp.llama_init_from_file(b"./models/7B/ggml-model-q4_0.bin", params)
# max_tokens = params.n_ctx

# tokens = (llama_cpp.llama_token * int(max_tokens))()
# n_tokens = llama_cpp.llama_tokenize(ctx, b"Q: Name the planets in the solar system? A: ", tokens, max_tokens, add_bos=llama_cpp.c_bool(True))

# for token in llama.generate(tokens, top_k=40, top_p=0.95, temp=1.0, repeat_penalty=1.1):
# ...     print(llama.detokenize([token]))


# llama_cpp.llama_free(ctx)



