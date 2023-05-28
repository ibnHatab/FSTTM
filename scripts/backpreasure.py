import asyncio
import functools
import reactivex as rx
from reactivex.scheduler.eventloop import AsyncIOScheduler
from reactivex.disposable import Disposable
from reactivex.subject import Subject
import reactivex.operators as ops


def from_aiter(iter, feedback, loop):
    def on_subscribe(observer, scheduler):
        async def _aio_sub():
            try:
                async for i in iter:
                    observer.on_next(i)
                loop.call_soon(observer.on_completed)
            except Exception as e:
                loop.call_soon(functools.partial(
                    observer.on_error, e))

        async def _aio_next():
            try:
                i = await iter.__anext__()
                observer.on_next(i)
            except StopAsyncIteration:
                observer.on_completed()
            except Exception as e:
                observer.on_error(e)

        if feedback is not None:
            return feedback.subscribe(
                on_next=lambda i: asyncio.ensure_future(
                    _aio_next(), loop=loop)
            )
        else:
            task = asyncio.ensure_future(_aio_sub(), loop=loop)
            return Disposable(lambda: task.cancel())

    return rx.create(on_subscribe)


async def ticker(delay, to):
    """Yield numbers from 0 to `to` every `delay` seconds."""
    for i in range(to):
        yield i
        await asyncio.sleep(delay)


async def main(loop):
    fb = Subject()
    done = asyncio.Future()

    def on_completed():
        print("completed")
        done.set_result(0)

    fb_bootstap = rx.from_([True])
    obs = from_aiter(
        ticker(0.1, 5),
        rx.concat(fb_bootstap, fb), loop).pipe(
            ops.share(),
            ops.delay(1.0),
        )

    disposable1 = obs.subscribe(fb,
        scheduler=AsyncIOScheduler(loop=loop))
    disposable2 = obs.subscribe(
        on_next=lambda i: print("next: {}".format(i)),
        on_error=lambda e: print("error: {}".format(e)),
        on_completed=on_completed,
        scheduler=AsyncIOScheduler(loop=loop),
    )

    await done
    disposable1.dispose()
    disposable2.dispose()

loop = asyncio.get_event_loop()
loop.run_until_complete(main(loop))