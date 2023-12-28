from llama_cpp import Llama

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
# llm = Llama(
#   model_path="./models/phi-2.Q5_K_S.gguf",  # Download the model file first
#   n_ctx=2048,  # The max sequence length to use - note that longer sequence lengths require much more resources
#   n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
#   n_gpu_layers=35         # The number of layers to offload to GPU, if you have GPU acceleration available
# )
# print(output)

# Chat Completion API

llm = Llama(model_path="./models/phi-2.Q5_K_S.gguf", chat_format="llama-2")
out = llm.create_chat_completion(
    messages = [
        {"role": "system",
         "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful answers to the user's questions."
        },
        {
            "role": "user",
            "content": "How are you?"
        }
    ],
    stop=["<|endoftext|>", "User:","Assistant:","System:"],
    stream=True
)

for t in out:
    print(t)



# llm = Llama(model_path="./phi-2.Q4_K_M.gguf", chat_format="llama-2")  # Set chat_format according to the model you are using
# llm.create_chat_completion(
#     messages = [
#         {"role": "system", "content": "You are a story writing assistant."},
#         {
#             "role": "user",
#             "content": "Write a story about llamas."
#         }
#     ]
# )

# "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful answers to the user's questions."

# {"stop":["<|endoftext|>", "User:","Assistant:", "System:"]}

"""{{ if .System }}System: {{ .System }}{{ end }}
User: {{ .Prompt }}
Assistant:"""

system_message="You are a helpful assistant."
system_template="<|im_start|>system\n{system_message}"
system_message=system_template.format(system_message=system_message)