import asyncio
import os
import io
import reactivex as rx
import reactivex.operators as ops
from reactivex.notification import OnNext, OnError, OnCompleted
from reactivex.scheduler.eventloop import AsyncIOScheduler

# 
# # https://blog.oakbits.com/rxpy-and-asyncio.html

async def to_agen(obs, loop):
    queue = asyncio.Queue()

    def on_next(i):
        queue.put_nowait(i)

    disposable = obs.pipe(ops.materialize()).subscribe(
        on_next=on_next,
        scheduler=AsyncIOScheduler(loop=loop)
    )

    while True:
        i = await queue.get()
        if isinstance(i, OnNext):
            yield i.value
            queue.task_done()
        elif isinstance(i, OnError):
            disposable.dispose()
            raise(Exception(i.value))
        else:
            disposable.dispose()
            break


async def main(loop):
    gen = to_agen(rx.from_([1, 2, 3, 4]), loop)
    async for i in gen:
        print(i)

    print("done")


loop = asyncio.get_event_loop()
loop.run_until_complete(main(loop))