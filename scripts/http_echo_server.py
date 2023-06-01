
from collections import namedtuple

from cyclotron import Component
from cyclotron.asyncio.runner import run, setup
import cyclotron_aiohttp.httpd as httpd

import reactivex as rx
import reactivex.operators as ops

from gpt_fsttm_server.trace import *

EchoSource = namedtuple('EchoSource', ['httpd'])
EchoSink = namedtuple('EchoSink', ['httpd'])
EchoDriver = namedtuple('EchoDriver', ['httpd'])

def echo_server(source):
    init = rx.from_([
        httpd.Initialize(),
        httpd.AddRoute(methods=['GET'], path='/echo/{what}', id='echo'),
        httpd.StartServer(host='127.1', port=8080),
    ])

    echo = source.httpd.route.pipe(
        trace('echo'),
        ops.filter(lambda r: r.id == 'echo'),
        ops.flat_map(lambda r: r.request),
        ops.map(lambda r: httpd.Response(
            context=r.context,
            data=r.match_info['what'].encode('utf-8')),        
        ),
        
    )
    
    control = rx.merge(init, echo).pipe(
        trace('control'),        
    )   
    return EchoSink(httpd=httpd.Sink(control))

def main():
    run(entry_point=Component(call=echo_server, input=EchoSource),
        drivers=EchoDriver(httpd.make_driver())
        )
    pass
    
if __name__ == '__main__':
    main()  
