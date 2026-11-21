"""A conversation with the streams beside it, against the configured API backend: what the
model says when the home's state and the rule engine's conclusions arrive between turns.
    python scripts/streams_probe.py config.nuc.yaml"""
import asyncio
import sys
import time

import yaml
from reactivex.subject import Subject

from fsttm import llama
from fsttm.config import llama_cloud_fields
from fsttm.llama import Initialize, AddSystem, Generate, ResponseDone, LlamaError, StreamEvent, Sink

SCRIPT = [
    ("env", "light.living_room", "on"), ("env", "media_player.tv", "on"),
    ("env", "fan.vent_bedroom", "off"),
    ("say", "Hi Nina, is the living room light on?"),
    ("say", "Turn off the TV, please."),
    ("reasoning", None, "done off(tv): I turn the TV off."), ("env", "media_player.tv", "off"),
    ("say", "Did that work?"),
    ("say", "Turn on the heating, I'm cold."),
    ("reasoning", None, "refused on(heating): I can't turn that on, it is not a device I know in this home."),
    ("say", "Okay. What else could warm the room?"),
    ("env", "light.living_room", "off"),
    ("say", "I'm reading now, is everything fine with the lights?"),
]


async def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    g = cfg["gpt"]
    loop = asyncio.get_running_loop()
    sub, done = Subject(), asyncio.Queue()
    src = llama.make_driver(loop).call(Sink(request=sub))
    src.system.subscribe(on_next=lambda e: done.put_nowait(e) if type(e) in (ResponseDone, LlamaError) else None)
    sub.on_next(Initialize(model_path=g["model"], **llama_cloud_fields(g)))
    sub.on_next(AddSystem(prompt="You are Nina, the voice of a flat's smart home."))
    for step in SCRIPT:
        if step[0] == "say":
            t = time.monotonic()
            sub.on_next(Generate(text=step[1], context=step[1]))
            ev = await done.get()
            print("YOU : %s\nNINA: %s   [%.1f s]\n" % (step[1], getattr(ev, "full_text", None) or ev.error, time.monotonic() - t))
        else:
            sub.on_next(StreamEvent(stream=step[0], key=step[1], text=step[2], ts=time.time()))
            print("  <%s> %s%s" % (step[0], (step[1] + ": ") if step[1] else "", step[2]))


asyncio.run(main(sys.argv[1]))
