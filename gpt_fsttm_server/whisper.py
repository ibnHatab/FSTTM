
import io
import logging
from collections import namedtuple
import numpy as np

import reactivex as rx
from cyclotron import Component
from cyclotron_std.logging import Log

import whispercpp as w

# High-performance inference of OpenAI's Whisper
# automatic speech recognition (ASR) model

Sink = namedtuple('Sink', ['speech'])
Source = namedtuple('Source', ['text', 'log'])

# Sink events
Initialize = namedtuple('Initialize', ['model', 'scorer', 'beam_width'])
Initialize.__new__.__defaults__ = (None,)

SpeechToText = namedtuple('SpeechToText', ['data', 'context'])

# Sourc eevents
TextResult = namedtuple('TextResult', ['text', 'context'])
TextError = namedtuple('TextError', ['error', 'context'])


def make_driver(loop=None):
    def driver(sink):
        model = None
        params = None
        log_observer = None

        def on_log_subscribe(observer, scheduler):
            nonlocal log_observer
            log_observer = observer

        def log(message, level=logging.DEBUG):
            if log_observer is not None:
                log_observer.on_next(Log(
                    logger=__name__,
                    level=level,
                    message="{}: {}".format(__name__, message),
                ))
        def setup_model(model_name, scorer, beam_width):
            log("creating model {} with scorer {}...".format(model_name, scorer))

            model = w.Whisper.from_pretrained(model_name)
            params = model.params.with_print_realtime(True).build()
            log("model {} with params {}...".format(model_name, params))

            return model

        def subscribe(observer, scheduler):
            def on_whisper_request(item):
                nonlocal model

                if type(item) is SpeechToText:
                    if model is not None:
                        try:
                            audio = np.frombuffer(item.data, np.int16).flatten().astype(np.float32) / 32768.0
                            text = model.transcribe(audio)
                            log("STT result: {}".format(text))
                            observer.on_next(rx.just(TextResult(
                                text=text,
                                context=item.context,
                            )))
                        except Exception as e:
                            log("STT error: {}".format(e), level=logging.ERROR)
                            observer.on_next(rx.throw(TextError(
                                error=e,
                                context=item.context,
                            )))
                elif type(item) is Initialize:
                    log("initialize: {}".format(item))
                    model = setup_model(
                        item.model, item.scorer, item.beam_width)
                else:
                    log("unknown item: {}".format(item), level=logging.CRITICAL)
                    observer.on_error(
                        "Unknown item type: {}".format(type(item)))

            sink.speech.subscribe(lambda item: on_whisper_request(item))

        return Source(
            text=rx.create(subscribe),
            log=rx.create(on_log_subscribe),
        )

    return Component(call=driver, input=Sink)
