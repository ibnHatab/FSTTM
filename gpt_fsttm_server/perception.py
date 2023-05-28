import asyncio
import logging
from collections import namedtuple

import reactivex as rx
from cyclotron import Component
from cyclotron_std.logging import Log

from mic_vad import VADAudio

# Component that interfaces with the microphone and the VAD

Sink = namedtuple('Sink', ['request'])
Source = namedtuple('Source', ['response', 'log'])

# Sink events
Initialize = namedtuple('Initialize', ['vad_aggressiveness', 'device', 'rate'])
Initialize.__new__.__defaults__ = (None,)

Start = namedtuple('Start', [])
Stop = namedtuple('Stop', [])

# Sourc events
Utterence = namedtuple('Utterence', ['data', 'context'])

def make_driver(loop=None):
    loop = asyncio.get_event_loop() if loop is None else loop

    def driver(sink):
        vad_audio = None
        log_observer = None

        def on_log_subscribe(observer, scheduler):
            nonlocal log_observer
            log_observer = observer

        def log(message, level=logging.DEBUG):
            print('>>2', message)
            if log_observer is not None:
                log_observer.on_next(Log(
                    logger=__name__,
                    level=level,
                    message="{}: {}".format(__name__, message),
                ))

        def setup_vad(vad_aggressiveness, device, rate):
            vad = VADAudio(loop,
                           aggressiveness=vad_aggressiveness,
                           device=device,
                           input_rate=rate)
            log("Listening on {}".format(device))
            return vad

        def on_subscribe(observer, scheduler):

            async def read_events():
                nonlocal vad_audio
                async for frame in vad_audio.vad_collector():
                    if frame is not None:
                        log("streaming frame")
                        loop.call_soon(observer.on_next, Utterence(data=frame, context=None))
                    else:
                        log("end of utterence")
                        loop.call_soon(observer.on_next, Utterence(data=None, context=None))

            def on_next(item):
                nonlocal vad_audio
                if type(item) is Initialize:
                    vad_audio = setup_vad(item.vad_aggressiveness,
                                          item.device,
                                          item.rate)
                    asyncio.ensure_future(read_events())
                elif type(item) is Start:
                    vad_audio.start()
                    log("Start listening on {}".format(vad_audio.device), level=logging.INFO)

                elif type(item) is Stop:
                    vad_audio.stop()
                    log("Stop listening on {}".format(vad_audio.device), level=logging.INFO)

                else:
                    log("unknown item: {}".format(item), level=logging.CRITICAL)
                    observer.on_error(
                        "Unknown item type: {}".format(type(item)))

            sink.request.subscribe(
                on_next=on_next,
                on_error=lambda e: observer.on_error(e))

        return Source(
            response=rx.create(on_subscribe),
            log=rx.create(on_log_subscribe),
        )

    return Component(call=driver, input=Sink)
