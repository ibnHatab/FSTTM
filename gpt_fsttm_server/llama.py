import asyncio
from collections import namedtuple
from cyclotron import Component

import reactivex as rx
import reactivex.operators as ops

from llama_cpp import Llama
from gpt_fsttm_server.utils import ignoreStderr

Sink = namedtuple('Sink', ['request'])
Source = namedtuple('Source', ['system',])

# sink events
Initialize = namedtuple('Initialize', [
    'model_path',
])
Initialize.__new__.__defaults__ = (None,)

Generate = namedtuple('Generate', ['text', 'context'])
Generate.__new__.__defaults__ = (None, None)

StopGenerate = namedtuple('StopGenerate', [])
ContinueGenerate = namedtuple('ContinueGenerate', ['max_tokens'])

AddSystem = namedtuple('AddSystem', ['prompt',])
AddSystem.__new__.__defaults__ = (None,)

# source events
Response = namedtuple('Response', ['text', 'context', ])
Response.__new__.__defaults__ = (None, None, None)
LlamaError = namedtuple('LlamaError', ['error', 'context'])

def make_driver(loop=None):
    def driver(sink):
        model = None
        stop = False

        def setup_model(model_path,):
            nonlocal model
            with ignoreStderr(False):
                model = Llama(model_path,
                              embedding=True,
                              n_threads=6,
                              verbose=True,
                              )
            return model

        def stop_callback(trace, scores):
            nonlocal stop
            return stop

        def on_subscribe(observer, scheduler):
            def on_llama_request(item):
                nonlocal model, stop

                if type(item) is Generate:
                    if model is not None:
                        try:
                            stop = False
                            print('1>>', item)
                            tokens = model.tokenize(item.text.encode('utf-8'))
                            print('2>>', tokens)
                            for token in model.generate(tokens, top_k=40, top_p=0.95,
                                                        temp=1.0, repeat_penalty=1.1,
                                                        stopping_criteria=stop_callback):
                                observer.on_next(Response(
                                    text=model.detokenize([token]).decode('utf-8'),
                                    context=item.context,
                                ))

                        except Exception as e:
                            print(f"Llama error: {e}")
                            observer.on_next(rx.throw(LlamaError(
                                error=e,
                                context=item.context,
                            )))
                elif type(item) is AddSystem:
                    if model is not None:
                        try:
                            tokens = model.tokenize(item.prompt)
                            params_n_keep = len(tokens)
                            model.eval(tokens)
                        except Exception as e:
                            print(f"Llama error: {e}")
                            observer.on_next(rx.throw(LlamaError(
                                error=e,
                                context=None,
                            )))
                elif type(item) is StopGenerate:
                    print(f"Receive stop generate: {item}")
                    stop = True
                elif type(item) is Initialize:
                    print(f"Receive initialize: {item}")
                    model = setup_model(item.model_path,)
                    print('0>>', model, model._candidates_data)
                else:
                    print(f"unknown item: {item}")
                    observer.on_error(
                        "Unknown item type: {}".format(type(item)))
            sink.request.subscribe(lambda item: on_llama_request(item))

        return Source(
            system=rx.create(on_subscribe),
        )

    return Component(call=driver, input=Sink)


