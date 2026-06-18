"""Body intent domain: doors (lock/unlock), windows (open/close), seat comfort
(heat/cool up/down)."""
from fsttm.intents.base import IntentModule, register, AREA_BOTH, AREA_BOTH_BM

INTENTS = [
    "DOOR_LOCK", "DOOR_UNLOCK",
    "WINDOW_OPEN", "WINDOW_CLOSE",
    "SEAT_HEAT_UP", "SEAT_HEAT_DOWN",
    "SEAT_COOL_UP", "SEAT_COOL_DOWN",
]

PROMPT = """\
## Door / Window / Seat Intents

| Intent | Trigger phrases | Parameters |
|--------|-----------------|------------|
| DOOR_LOCK | "lock", "secure the car" | area (default 0=all) |
| DOOR_UNLOCK | "unlock", "open the lock" | area |
| WINDOW_OPEN | "open the window", "roll down" | position 0–100 (default 50) |
| WINDOW_CLOSE | "close the window", "roll up" | — |
| SEAT_HEAT_UP | "warm/heat my seat", "seat warmer", "more seat heat" | delta 1–3 |
| SEAT_HEAT_DOWN | "less seat heat", "seat too warm", "cooler seat" | delta 1–3 |
| SEAT_COOL_UP | "cool/chill my seat", "seat cooler on", "more seat cooling" | delta 1–3 |
| SEAT_COOL_DOWN | "less seat cooling", "seat cooling off" | delta 1–3 |

UP = start/increase that function; DOWN = reduce it. "cool the seat" turns
cooling ON (SEAT_COOL_UP), it does NOT mean less cooling.

Zone: area 1=driver/left, 4=passenger/right, 0=all. "lock the passenger door" → area=4
"""


def _action(act, **args):
    d = {"cmd": "action", "action": act}
    if args:
        d["args"] = args
    return d


def translate(intent):
    name  = intent.get("intent")
    area  = intent.get("area", AREA_BOTH)
    delta = intent.get("delta", 1)
    za = AREA_BOTH_BM if area == AREA_BOTH else area

    if name == "DOOR_LOCK":
        return [{"cmd": "set", "name": "DOOR_LOCK", "area": area, "value": True}]
    if name == "DOOR_UNLOCK":
        return [{"cmd": "set", "name": "DOOR_LOCK", "area": area, "value": False}]
    if name == "WINDOW_OPEN":
        pos = intent.get("position", 50)
        return [_action("window_move", area=za, position=int(pos))]
    if name == "WINDOW_CLOSE":
        return [_action("window_move", area=za, position=0)]
    if name == "SEAT_HEAT_UP":
        return [_action("bump_seat_temp", up=True, area=za)] * delta
    if name == "SEAT_HEAT_DOWN":
        return [_action("bump_seat_temp", up=False, area=za)] * delta
    if name == "SEAT_COOL_UP":
        return [_action("bump_seat_vent", up=True, area=za)] * delta
    if name == "SEAT_COOL_DOWN":
        return [_action("bump_seat_vent", up=False, area=za)] * delta
    return []


# Body needs the "position" prop for windows.
MODULE = register(IntentModule("body", INTENTS, PROMPT, translate,
                               extra_props={"position": {"type": "integer"}}))
