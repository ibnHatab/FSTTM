#  Looping through sentences

from llama_cpp.llama import _LlamaModel as LlamaModel
from llama_cpp.llama import _LlamaContext as LamaContext


class Chat:
    def __init__(self, model: Model, verbose: bool = False):
        self.model = model
        self.verbose = verbose

        model_params = llama_cpp.llama_model_default_params()
        n_gpu_layers = model.Options["n_gpu_layers"]
        model_params.n_gpu_layers = (0x7FFFFFFF if n_gpu_layers == -1 else n_gpu_layers)
        model_params.use_mmap = True
        model_params.use_mlock = False

        context_params = llama_cpp.llama_context_default_params()
        context_params.n_ctx = model.Options["n_ctx"]
        context_params.n_threads = model.Options["n_threads"]

        model_path = model.ModelPath
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")


        self.lm = LlamaModel(path_model=model_path, params=model_params, verbose=verbose)
        self.ctx = LamaContext(model=self.lm, params=context_params, verbose=verbose)

        self.vars = PromptVars(
            System=model.System,
            Prompt="",
            Response="",
            First=True,
        )
        if self.verbose:
            print(llama_cpp.llama_print_system_info().decode("utf-8"), file=sys.stderr)

    def __call__(self, prompt: str) -> str:
        self.vars.Prompt = prompt
        self.vars.Response = self.ctx.create_chat_completion(
            messages=[
                {"role": "system", "content": self.model.System},
                {"role": "user", "content": self.vars.Prompt},
            ],
            stop=self.model.Options["stop"],
        )
        return self.vars.Response


vars = [PromptVars.create(prompt="How are you?", first=True)]
vars += [PromptVars.create(prompt=p) for p in ["What is your name?",
                                            "What is your favorite color?",
                                            "Is there Dog?",]]

dia = []

for v in vars:
    prompt = v.format(model)
    print(prompt)
    v.Response = llm(prompt, stop=model.Options["stop"], echo=True)
    dia += [v]

expr = PromptVars.create(prompt="""
{
   "intent": {
      "intentName": "searchWeatherForecast",
      "probability": 0.95
   },
   "slots": [
      {
         "value": "paris",
         "entity": "locality",
         "slotName": "forecast_locality"
      },
      {
         "value": {
            "kind": "InstantTime",
            "value": "2018-02-08 20:00:00 +00:00"
         },
         "entity": "snips/datetime",
         "slotName": "forecast_start_datetime"
      }
   ]
}
""", system="Summarize it into plain English using one short sentence. Ask confirmation for intended action. If confirmed, execute the action."
).format(model)

r = llm(expr, stop=model.Options["stop"], echo=True, max_tokens=20)
print(r["choices"][0]["text"])


vars = PromptVars.create(prompt="Which city I'm asking for weather?", first=True).format(model)
r = llm(vars, stop=model.Options["stop"], echo=True)
print(r["choices"][0]["text"])



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
