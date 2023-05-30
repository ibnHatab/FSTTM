
import asyncio
from collections import namedtuple
from functools import partial

import reactivex as rx
import reactivex.operators as ops
from reactivex.scheduler.eventloop import AsyncIOScheduler
from reactivex.scheduler import ImmediateScheduler

from cyclotron import Component
from cyclotron.asyncio.runner import run
from cyclotron.router import make_error_router
import cyclotron_std.logging as logging
import cyclotron_std.sys.stdout as stdout
import cyclotron_std.io.file as file
import cyclotron_std.sys.argv as argv
import cyclotron_std.argparse as argparse

from gpt_fsttm_server.config import parse_config
import gpt_fsttm_server.perception as perception
import gpt_fsttm_server.llama as llama
import gpt_fsttm_server.rhvoice as rhvoice
import gpt_fsttm_server.whisper as whisper

import gpt_fsttm_server.trace as trace

FSTTMSink = namedtuple('Sink', [
    'perception', 'stt', 'logging', 'file', 'stdout'
    ])
FSTTMSource = namedtuple('Source', [
    'perception', 'stt', 'file', 'argv'
    ])
FSTTMDrivers = namedtuple('Drivers', [
    'perception', 'stt', 'stdout', 'logging', 'file', 'argv'
    ])


def parse_arguments(argv):
    parser = argparse.ArgumentParser("Finite-State Turn-Taking Machine")
    parser.add_argument('--config', required=True, help="Path of the server configuration file")
    return argv.pipe(
        ops.skip(1),
        argparse.parse(parser),
    )


    # asr_error, route_asr_error = make_error_router()
    # tts_error, route_tts_error = make_error_router()

def fsttm_server(aio_scheduler, sources):

    argv = sources.argv.argv
    args = parse_arguments(argv)

    read_request, read_response = args.pipe(
        ops.map(lambda i: file.Read(id='config', path=i.value)),
        file.read(sources.file.response),
    )
    read_request = read_request.pipe(
        ops.subscribe_on(aio_scheduler),
    )
    config = parse_config(read_response)

    logs_config = config.pipe(
        ops.flat_map(lambda i: rx.from_(i.log.level, scheduler=ImmediateScheduler())),
        ops.map(lambda i: logging.SetLevel(logger=i.logger, level=i.level)),
    )
    logs = rx.merge(logs_config, 
                    sources.perception.log,
                    sources.stt.log,
                    )
    
    ##
    perception_init = config.pipe(
        ops.flat_map(lambda i: rx.from_([
            perception.Initialize(i.vad.vad_aggressiveness,
                            i.vad.device,
                            i.vad.rate),
            perception.Start(),
        ])),
        #ops.share(),
        trace.trace('p init+'),
    )

    utterance_end = sources.perception.response.pipe(
        ops.filter(lambda i: i.data == None)
    )

    utterance = sources.perception.response.pipe(
        ops.map(lambda i: i.data if i.data else bytes()),
        ops.buffer(utterance_end),
        ops.map(lambda xs: bytearray().join(xs)),
    )

    ###
    # stt_init = config.pipe(
    #     ops.flat_map(lambda i: rx.from_([
    #         whisper.Initialize(i.stt.model),
    #     ])),
    # )

    # stt_request = utterance.pipe(
    #     ops.map(lambda i: whisper.SpeechToText(data=i, context=None)),
    # )
    
    # stt_subjects = rx.merge(stt_request, stt_init)
    # stt_response = sources.stt.text
    stt_subjects = rx.empty()
    ###

    # text_response = stt_response.pipe(
    #     ops.map(lambda i: i.text.encode('utf-8')),
    # )

    # log_utterance_length = utterance.pipe(
    #     ops.map(lambda i: "recv: {}\n".format(len(i))),
    # )
    
#    std_out = rx.merge(log_utterance_length, text_response)
    log_utterance_length = rx.just('bla-bla\n')
    std_out = rx.merge(log_utterance_length, )

#    init = rx.merge(perception_init, stt_init)
    return FSTTMSink(
        perception=perception.Sink(request=perception_init),
        stt = whisper.Sink(speech=stt_subjects),
        logging=logging.Sink(request=logs),
        file=file.Sink(request=read_request),
        stdout=stdout.Sink(data=std_out),
    )

def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_debug(True)
    aio_scheduler = AsyncIOScheduler(loop=loop)

    run(
        Component(
            call=partial(fsttm_server, aio_scheduler),
            input=FSTTMSource
        ),
        FSTTMDrivers(
            perception=perception.make_driver(loop),
            stt = whisper.make_driver(loop),
            stdout=stdout.make_driver(),
            logging=logging.make_driver(),
            file=file.make_driver(),
            argv=argv.make_driver(),
        ),
        loop=loop,
    )


if __name__ == '__main__':
    import sys
    sys.argv.append('--config')
    sys.argv.append('config.sample.yaml')

    main()
