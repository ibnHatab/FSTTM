from collections import namedtuple
import numpy as np

import reactivex as rx
from cyclotron import Component
from cyclotron_std.logging import Log

import whispercpp as w

# High-performance inference of OpenAI's Whisper
# automatic speech recognition (ASR) model

Sink = namedtuple('Sink', ['request'])
Source = namedtuple('Source', ['text'])

# Sink events
Initialize = namedtuple('Initialize', ['model',])
Initialize.__new__.__defaults__ = (None,)

SpeechToText = namedtuple('SpeechToText', ['data', 'context'])

# Sourc eevents
TextResult = namedtuple('TextResult', ['text', 'context'])
TextError = namedtuple('TextError', ['error', 'context'])


def make_driver(loop=None):
    def driver(sink):
        model = None

        def setup_model(model_name):
            print(f"Initialize Whisper model: {model_name}")
            model = w.Whisper.from_pretrained(model_name)
            # FIXME: params = model.params.with_print_realtime(True).build()
            return model

        def on_subscribe(observer, scheduler):
            def on_whisper_request(item):
                nonlocal model

                if type(item) is SpeechToText:
                    if model is not None:
                        try:
                            audio = np.frombuffer(item.data, np.int16).flatten().astype(np.float32) / 32768.0
                            text = model.transcribe(audio)
                            print('STT>>', len(item.data), audio.shape, text)
                            observer.on_next(rx.just(TextResult(
                                text=text,
                                context=item.context,
                            )))
                        except Exception as e:
                            print(f"Whisper error: {e}")
                            observer.on_next(rx.throw(TextError(
                                error=e,
                                context=item.context,
                            )))
                elif type(item) is Initialize:
                    print(f"Receive initialize: {item}")
                    model = setup_model(item.model)
                else:
                    print(f"unknown item: {item}")
                    observer.on_error(
                        "Unknown item type: {}".format(type(item)))

            sink.request.subscribe(lambda item: on_whisper_request(item))

        return Source(
            text=rx.create(on_subscribe),
        )

    return Component(call=driver, input=Sink)
