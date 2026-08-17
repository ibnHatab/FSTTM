"""Voice-filter cyclotron driver.

Sink: Initialize(cfg) / Filter(data, context) — data is one full utterance of
16 kHz s16le mono PCM (the VAD buffer). Source: Accepted(data, context) /
Rejected(score, speaker, context).

Disabled (the default) or failed-to-load → synchronous passthrough: every
Filter immediately emits Accepted, so the pipeline wiring is unconditional
and adds no latency. Enabled → the embedding (50–200 ms CPU) runs on a
dedicated worker thread; results are marshalled back via
loop.call_soon_threadsafe.

`mode: shadow` scores and logs every utterance but drops nothing — for
threshold tuning before switching to `enforce`.
"""
import logging
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import reactivex as rx
from cyclotron import Component

_log = logging.getLogger("fsttm.voicefilter")

Sink = namedtuple('Sink', ['request'])
Source = namedtuple('Source', ['result'])

# Sink events
Initialize = namedtuple('Initialize', ['cfg'])
Initialize.__new__.__defaults__ = (None,)
Filter = namedtuple('Filter', ['data', 'context'])
Filter.__new__.__defaults__ = (None, None)

# Source events
Accepted = namedtuple('Accepted', ['data', 'context'])
Rejected = namedtuple('Rejected', ['score', 'speaker', 'context'])

_RATE = 16000   # the VAD utterance rate


def _load_provider(name):
    from importlib.metadata import entry_points
    eps = {ep.name: ep for ep in entry_points(group="fsttm.voice_filters")}
    if name not in eps:
        raise LookupError(f"voice filter {name!r} not installed "
                          f"(available: {sorted(eps) or 'none'})")
    return eps[name].load()()


def make_driver(loop=None):
    def driver(sink):
        verifier = [None]        # loaded provider, or None → passthrough
        shadow = [False]
        # One worker: utterances are serial anyway (VAD closes one at a time).
        executor = ThreadPoolExecutor(max_workers=1,
                                      thread_name_prefix='voicefilter')

        def on_subscribe(observer, scheduler):

            def setup(cfg):
                cfg = cfg or {}
                if not cfg.get('enabled'):
                    verifier[0] = None
                    return
                shadow[0] = (cfg.get('mode', 'enforce') == 'shadow')
                try:
                    v = _load_provider(cfg.get('provider', 'speaker'))
                    v.load(cfg)
                    verifier[0] = v
                    print(f"Voice filter ready (mode="
                          f"{'shadow' if shadow[0] else 'enforce'})")
                except Exception as exc:
                    verifier[0] = None
                    print(f"WARNING: voice filter unavailable ({exc}); "
                          f"passing all utterances")

            def _check_sync(item):
                """Runs on the worker thread."""
                try:
                    res = verifier[0].check(item.data, _RATE)
                except Exception as exc:
                    _log.warning("voice filter check failed (%s) — accepting",
                                 exc)
                    loop.call_soon_threadsafe(
                        observer.on_next,
                        Accepted(data=item.data, context=item.context))
                    return
                if res.accepted or shadow[0]:
                    if not res.accepted:   # shadow-mode would-have-dropped
                        _log.info("[voice-filter] SHADOW would drop "
                                  "(score=%.2f best=%s)", res.score, res.speaker)
                    elif res.score == res.score:   # not NaN (bypass)
                        _log.debug("[voice-filter] accept score=%.2f (%s)",
                                   res.score, res.speaker)
                    loop.call_soon_threadsafe(
                        observer.on_next,
                        Accepted(data=item.data, context=item.context))
                else:
                    _log.info("[voice-filter] REJECT utterance "
                              "(score=%.2f best=%s)", res.score, res.speaker)
                    loop.call_soon_threadsafe(
                        observer.on_next,
                        Rejected(score=res.score, speaker=res.speaker,
                                 context=item.context))

            def on_request(item):
                if type(item) is Initialize:
                    setup(item.cfg)
                elif type(item) is Filter:
                    if verifier[0] is None:
                        # Passthrough: synchronous, zero added latency.
                        observer.on_next(Accepted(data=item.data,
                                                  context=item.context))
                    else:
                        executor.submit(_check_sync, item)
                else:
                    observer.on_error(f"Unknown item type: {type(item)}")

            sink.request.subscribe(on_next=on_request,
                                   on_error=lambda e: observer.on_error(e))

        return Source(result=rx.create(on_subscribe))

    return Component(call=driver, input=Sink)
