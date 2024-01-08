from llama_cpp import Llama

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.

llm = Llama(
  model_path="./models/phi-2.Q5_K_S.gguf",  # Download the model file first
  n_ctx=2048,  # The max sequence length to use - note that longer sequence lengths require much more resources
  n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
  n_gpu_layers=35         # The number of layers to offload to GPU, if you have GPU acceleration available
)


def stopping_criteria(a, b):
    return False

system = "System: A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful answers to the user's questions."
tmpl = "User: {prompt}\nAssistant:"
txt = "How are you?"
prompt = tmpl.format(prompt=txt)
prompt = system + "\n" + prompt

def completion(prompt, max_tokens=256):
    iter = llm.create_completion(prompt=prompt,
                            stream=False,
                            stop=["<|endoftext|>", "User:","Assistant:", "System:"],
                            max_tokens=max_tokens,
                            echo=True,
                            stopping_criteria=stopping_criteria
                            )
    text = iter["choices"][0]["text"]
    print(text)

completion(prompt)

to_eval = [
    "User:  What is the weather today?\n",
    "Q:      What is the difference between Texas and yogurt?\n",
    "A:      Yogurt has culture.\n",
    "Assistant:",
]

prompt = "".join(to_eval)
completion(prompt)

to_gen = [
"Assistant: So, ",
]

prompt = "".join(to_gen)
completion(prompt)

del llm