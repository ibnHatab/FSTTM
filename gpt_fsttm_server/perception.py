import asyncio
import logging
from collections import namedtuple

import reactivex as rx
from cyclotron import Component

from gpt_fsttm_server.mic_vad import VADAudio

# Component that interfaces with the microphone and the VAD

Sink = namedtuple('Sink', ['control'])
Source = namedtuple('Source', ['voice'])

# Sink events
Initialize = namedtuple('Initialize', ['vad_aggressiveness', 'device', 'rate'])
Initialize.__new__.__defaults__ = (None, None, None,)

Start = namedtuple('Start', [])
Stop = namedtuple('Stop', [])

# Source events
Chunk = bytearray

def make_driver(loop=None):
    def driver(sink):
        vad_audio = None

        def setup_vad(vad_aggressiveness, device, rate):
            print("fListening on audio {device}")
            vad = VADAudio(loop,
                           aggressiveness=vad_aggressiveness,
                           device=device,
                           input_rate=rate)
            return vad

        def on_subscribe(observer, scheduler):
            async def read_events():
                nonlocal vad_audio
                async for frame in vad_audio.vad_collector():
                    loop.call_soon(observer.on_next, frame)
                    #print(f">> streaming frame: {type(frame)}")

            def on_perception_request(item):
                nonlocal vad_audio
                if type(item) is Initialize:
                    vad_audio = setup_vad(item.vad_aggressiveness,
                                          item.device,
                                          item.rate)
                    asyncio.ensure_future(read_events())
                elif type(item) is Start:
                    vad_audio.start()
                    print(f"Start streaming on {vad_audio.device}")

                elif type(item) is Stop:
                    vad_audio.stop()
                    print(f"Stop streaming on {vad_audio.device}")
                else:
                    print("unknown item: {item}")
                    observer.on_error(
                        "Unknown item type: {}".format(type(item)))

            sink.control.subscribe(
                on_next=on_perception_request,
                on_error=lambda e: observer.on_error(e))

        return Source(
            voice=rx.create(on_subscribe),
        )

    return Component(call=driver, input=Sink)
