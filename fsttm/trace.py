import traceback

import reactivex as rx
from reactivex import Observer

class pry(Observer):
    def on_next(self, value):
        print("Received {0}".format(value))

    def on_error(self, error):
        print("Error Occurred: {0}".format(error))

    def on_completed(self):
        print("Done!")


def id(s):
    print("id: {}".format(s))
    return s


def trace(prefix):
    def _trace(source):
        def on_subscribe(observer, scheduler):
            def on_next(i):
                print("{} - on next: {}".format(prefix, i))
                observer.on_next(i)

            def on_error(e):
                if isinstance(e, Exception):
                    print("{} - on error: {}, {}".format(prefix,  e, traceback.print_tb(e.__traceback__)))
                else:
                    print("{} - on error: {}".format(prefix,  e))
                observer.on_error(e)

            def on_completed():
                print("{} - completed".format(prefix))
                observer.on_completed()

            source.subscribe(
                on_next=on_next,
                on_error=on_error,
                on_completed=on_completed
            )
        return rx.create(on_subscribe)

    return _trace
