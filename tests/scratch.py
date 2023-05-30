
import reactivex as rx
import reactivex.operators as ops

from gpt_fsttm_server.trace import *


xs = rx.from_marbles("1-2-3-4-5-6-7-n-8-9-10-11-12-13-14-n-15-16-17-|")
ys = xs.pipe( ops.filter(lambda x: x == "n"),)
xs.pipe(
    ops.window_when(lambda: ys),
    ops.map_indexed(lambda w, i: w.pipe(ops.map(lambda x: str(i) + " " + str(x)))),
    ops.map(lambda w: w.pipe(ops.buffer_with_count(4, 3),)),
    ops.do_action(lambda x: print(x)),    
    ops.merge_all(),
    ops.to_marbles(),
    ).run()

s1 = rx.from_marbles('1---2-3|')
s2 = rx.from_marbles('-a-b-c-|')
s = rx.merge(s1, s2)
s.pipe(ops.to_marbles()).run()

s = rx.from_marbles('1---2-3|')
s.pipe(
    ops.flat_map(lambda x: rx.from_(range(x, x + 3))),
    ops.timestamp(),
    ops.pluck_attr("value"),
    ops.to_marbles()
    ).run()

s = rx.from_marbles('1-2-3-4-5-----|')
s.pipe(
    ops.time_interval(),
    #ops.pluck_attr("interval"),    
).subscribe(pry())

d = rx.from_marbles('1-2-3-4-5-|')
c = rx.from_marbles('a-b-c-d-e-f-|')
dd = d.pipe(
    ops.delay(0.300),
)
rx.merge(dd, c).pipe(
    ops.to_marbles()
).run()

s = rx.from_marbles('1-2-3-4-5-|')
s.pipe(
    ops.materialize(),
    ops.to_marbles()
).run()

from reactivex.notification import OnNext, OnCompleted
s = rx.from_list([OnNext(1), OnNext(2), OnNext(3), OnCompleted()])
s.pipe(
    ops.start_with(OnNext(0)),
    ops.dematerialize(),
#    ops.ignore_elements(),
    ops.to_marbles()
).run()

e = rx.empty()
e = rx.cold('1-2-e-4-e-|', lookup={"e": rx.empty()})
e.pipe(
    ops.default_if_empty(42),
    ops.to_marbles()
).run()

h = rx.hot("a--b--c-", lookup={'a': 1, 'b': 2, 'c': 3})
h.subscribe(pry())



xs = rx.cold('1-2-3-4-5-6-7-8-9|')
cs = rx.cold('---e---e----------|')
xs.pipe(    
    ops.buffer(boundaries=cs),
    ops.to_marbles()
).run()

xs = rx.cold('1-2-3-4-5-6-7-8-9-1-2-3-4-5-6-7-8-9|')
xs.pipe(
    ops.buffer_with_count(5, skip=4),
    ops.to_marbles()
    ).run()

xs = rx.cold('1-2-3-4-5-6-7-8-9-|')    
xs.pipe(
    ops.take_last_with_time(1.3),
    ops.to_marbles()
).run()

xs = rx.from_iterable([1, 2, 3, 4, 5, 6, 7, 8, 9])
xs.pipe(
    ops.single(lambda x: x == 5),
    ops.to_marbles()
).run()