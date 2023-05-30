import pytest

import reactivex
from reactivex import operators as ops
from reactivex.testing import ReactiveTest, TestScheduler

on_next = ReactiveTest.on_next
on_completed = ReactiveTest.on_completed
on_error = ReactiveTest.on_error
subscribe = ReactiveTest.subscribe
subscribed = ReactiveTest.subscribed
disposed = ReactiveTest.disposed
created = ReactiveTest.created


from gpt_fsttm_server.trace import *

def test_window():
    scheduler = TestScheduler()
    xs = scheduler.create_hot_observable(
        on_next(90, 1),
        on_next(180, 2),
        on_next(250, 3),
        on_next(260, 4),
        on_next(310, 5),
        on_next(340, 6),
        on_next(410, 7),
        on_next(420, 8),
        on_next(470, 9),
        on_next(550, 10),
        on_completed(590),
    )
    window = [1]

    def create():
        def closing():
            curr = window[0]
            window[0] += 1
            return reactivex.timer(curr * 100)

        def mapper(w, i):
            return w.pipe(ops.map(lambda x: str(i) + " " + str(x)))

        return xs.pipe(
            ops.window_when(closing),
            ops.map_indexed(mapper),
            ops.merge_all(),
        )

    results = scheduler.start(create=create)
    assert results.messages == [
        on_next(250, "0 3"),
        on_next(260, "0 4"),
        on_next(310, "1 5"),
        on_next(340, "1 6"),
        on_next(410, "1 7"),
        on_next(420, "1 8"),
        on_next(470, "1 9"),
        on_next(550, "2 10"),
        on_completed(590),
    ]
    assert xs.subscriptions == [subscribe(200, 590)]


def test_window_when():
    scheduler = TestScheduler()
    xs = scheduler.create_hot_observable(
        on_next(90, 1),
        on_next(180, 2),
        on_next(250, 3),
        on_next(260, 4),
        on_next(310, 5),
        on_next(340, 6),
        on_next(410, 7),
        on_next(420, 8),
        on_next(470, 9),
        on_next(550, 10),
        on_completed(590),
    )
 
    ys = xs.pipe(ops.filter(lambda x: x % 5 == 0))

    def create():
        return xs.pipe(
            ops.window_when(lambda: ys),
            ops.map_indexed(lambda w, i: w.pipe(ops.map(lambda x: str(i) + " " + str(x)))),
            ops.merge_all(),
            # concat_with
        )

    results = scheduler.start(create=create)
    assert results.messages == [
        on_next(250, "0 3"),
        on_next(260, "0 4"),
        on_next(310, "1 5"),
        on_next(340, "1 6"),
        on_next(410, "1 7"),
        on_next(420, "1 8"),
        on_next(470, "1 9"),
        on_next(550, "2 10"),
        on_completed(590),
    ]
    assert xs.subscriptions == [subscribe(200, 590)]
    assert ys.subscriptions == [subscribe(200, 590)]



