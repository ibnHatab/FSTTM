import asyncio
import functools
import reactivex as rx
from reactivex.scheduler.eventloop import AsyncIOScheduler
from reactivex.disposable import Disposable


def from_aiter(iter, loop):
    def on_subscribe(observer, scheduler):
        async def _aio_sub():
            try:
                async for i in iter:
                    observer.on_next(i)
                loop.call_soon(
                    observer.on_completed)
            except Exception as e:
                loop.call_soon(
                    functools.partial(observer.on_error, e))

        task = asyncio.ensure_future(_aio_sub(), loop=loop)
        return Disposable(lambda: task.cancel())

    return rx.create(on_subscribe)


async def ticker(delay, to):
    """Yield numbers from 0 to `to` every `delay` seconds."""
    for i in range(to):
        yield i
        await asyncio.sleep(delay)


async def main(loop):
    done = asyncio.Future()

    def on_completed():
        print("completed")
        done.set_result(0)

    disposable = from_aiter(ticker(0.1, 10), loop).subscribe(
        on_next=lambda i: print("next: {}".format(i)),
        on_error=lambda e: print("error: {}".format(e)),
        on_completed=on_completed,
    )

    await done
    disposable.dispose()

loop = asyncio.get_event_loop()
loop.run_until_complete(main(loop))