import asyncio

from collections import namedtuple

import reactivex as rx
import reactivex.operators as ops
from reactivex.scheduler import ImmediateScheduler

import cyclotron_std.sys.stdout as stdout
from cyclotron.debug.trace import trace_observable as trace
from cyclotron import Component
from cyclotron.asyncio.runner import run

import gpt_fsttm_server.perception as perception
import gpt_fsttm_server.whisper as whisper

EchoSource = namedtuple('EchoSource', ['perception', 'stt'])
EchoSink = namedtuple('EchoSink',     ['stdout', 'perception', 'stt'])
EchoDriver = namedtuple('EchoDriver', ['stdout', 'perception', 'stt'])


def echo_server(sources):
    vad_init = rx.from_([
        perception.Initialize(3, None, 16000),
        perception.Start(),
        ],
        scheduler=ImmediateScheduler()
    ).pipe( trace('vad_init'), )

    stt_init = rx.from_([
        whisper.Initialize('base.en'),
        ],
        scheduler=ImmediateScheduler()
    ).pipe( trace('stt_init'), )

    # Split the voice stream into utterances and frames
    vad = sources.perception.voice.pipe(
            ops.publish(),
            ops.ref_count(),
    )

    utterance = vad.pipe(ops.filter(lambda r: r == None), )
    frames = vad.pipe(
        ops.window(utterance),
        ops.flat_map(lambda w: w.pipe(
            ops.filter(lambda r: r != None),
            ops.reduce(lambda acc, cur: acc + cur, bytearray()))
        ),
        ops.map(lambda r: whisper.SpeechToText(data=r, context=None)),
        trace('frames', trace_next_payload=False),
    )

    # Convert the utterance into a text
    stt_request = rx.merge(stt_init, frames).pipe(
        trace('stt_request', trace_next_payload=False),
    )

    stt_response = sources.stt.text.pipe(
        trace('stt_response', trace_next_payload=False),
    )

    # Print text to stdout
    value = rx.merge(utterance, stt_response).pipe(
        trace('value', trace_next_payload=False),
        ops.filter(lambda r: type(r) is whisper.TextResult),
        ops.map(lambda r: r.text),
    )

    return EchoSink(
        stdout=stdout.Sink(data=value),
        perception=perception.Sink(control=vad_init),
        stt=whisper.Sink(request=stt_request),
    )

def main():
    loop = asyncio.get_event_loop()
    loop.set_debug(True)
    run(entry_point=Component(call=echo_server, input=EchoSource),
        drivers=EchoDriver(
            stdout = stdout.make_driver(),
            perception = perception.make_driver(loop),
            stt = whisper.make_driver(loop),
        ),
        loop=loop,
    )


if __name__ == '__main__':
    main()
