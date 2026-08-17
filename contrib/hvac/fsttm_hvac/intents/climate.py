"""Climate (HVAC) intent domain: temperature, fan, A/C, vents, recirc, auto,
power, rear defrost."""
from fsttm_hvac.registry import IntentModule, register, AREA_BOTH, AREA_BOTH_BM

# Fan direction bitmask (PROTOCOL.md).
FAN_FACE    = 1   # 0x1
FAN_FLOOR   = 2   # 0x2
FAN_DEFROST = 4   # 0x4
FAN_SPLIT   = 3   # FACE | FLOOR

INTENTS = [
    "WARMER", "COOLER", "SET_TEMPERATURE",
    "FAN_UP", "FAN_DOWN", "SET_FAN",
    "AC_ON", "AC_OFF", "MAX_AC_TOGGLE",
    "VENT_FACE", "VENT_FEET", "VENT_DEFROST", "VENT_SPLIT", "VENT_DEFROST_MAX",
    "RECIRCULATE_ON", "RECIRCULATE_OFF",
    "AUTO_ON", "AUTO_OFF", "SYNC_TOGGLE",
    "POWER_ON", "POWER_OFF",
    "REAR_DEFROST_TOGGLE",
]

PROMPT = """\
## Climate Intents

| Intent | Trigger phrases | Parameters |
|--------|-----------------|------------|
| WARMER | "too cold", "warmer", "heat up", "it's freezing" | delta 1–3 |
| COOLER | "too hot", "cooler", "cool down", "it's hot" | delta 1–3 |
| SET_TEMPERATURE | "set to X degrees", "X degrees please" | temp: 16–28 °C |
| FAN_UP | "more air", "faster fan", "blow harder" | delta 1–3 |
| FAN_DOWN | "less air", "quieter fan", "softer" | delta 1–3 |
| SET_FAN | "fan level X", "set fan to X" | fan_level: 1–7 |
| AC_ON / AC_OFF | "turn on/off AC", "air conditioning" | — |
| MAX_AC_TOGGLE | "max AC", "maximum cooling", "blast it" | — |
| VENT_FACE | "air to face", "blow on me" | — |
| VENT_FEET | "air to feet", "warm my feet" | — |
| VENT_DEFROST | "defrost windshield", "clear the windshield" | — |
| VENT_SPLIT | "face and feet", "both vents" | — |
| VENT_DEFROST_MAX | "max defrost", "full defrost" | — |
| RECIRCULATE_ON / RECIRCULATE_OFF | "recirculate" / "fresh air" | — |
| AUTO_ON / AUTO_OFF | "auto mode" / "manual mode" | — |
| SYNC_TOGGLE | "sync zones", "link both sides", "DUAL" | — |
| POWER_ON / POWER_OFF | "turn on/off climate" | — |
| REAR_DEFROST_TOGGLE | "rear window defrost" | — |
"""


def _action(act, **args):
    d = {"cmd": "action", "action": act}
    if args:
        d["args"] = args
    return d


def translate(intent):
    name  = intent.get("intent")
    area  = intent.get("area", AREA_BOTH)
    temp  = intent.get("temp")
    fan   = intent.get("fan_level")
    delta = intent.get("delta", 1)
    za = AREA_BOTH_BM if area == AREA_BOTH else area

    if name == "WARMER":
        return [_action("bump_temperature", up=True, area=za)] * delta
    if name == "COOLER":
        return [_action("bump_temperature", up=False, area=za)] * delta
    if name == "SET_TEMPERATURE":
        return [_action("update_temperature", value=float(temp), area=za)] if temp is not None else []
    if name == "FAN_UP":
        return [_action("bump_fan_speed", up=True, area=za)] * delta
    if name == "FAN_DOWN":
        return [_action("bump_fan_speed", up=False, area=za)] * delta
    if name == "SET_FAN":
        return [_action("update_fan_speed", value=int(fan), area=za)] if fan is not None else []
    if name in ("AC_ON", "AC_OFF"):
        return [_action("ac_toggle")]
    if name == "MAX_AC_TOGGLE":
        return [_action("ac_max_toggle")]
    if name == "VENT_FACE":
        return [_action("fan_direction_toggle", direction=FAN_FACE, area=za)]
    if name == "VENT_FEET":
        return [_action("fan_direction_toggle", direction=FAN_FLOOR, area=za)]
    if name == "VENT_DEFROST":
        return [_action("fan_direction_toggle", direction=FAN_DEFROST, area=za)]
    if name == "VENT_SPLIT":
        return [_action("fan_direction_toggle", direction=FAN_SPLIT, area=za)]
    if name == "VENT_DEFROST_MAX":
        return [_action("max_defrost_toggle")]
    if name in ("RECIRCULATE_ON", "RECIRCULATE_OFF"):
        return [_action("recirc_toggle")]
    if name in ("AUTO_ON", "AUTO_OFF"):
        return [_action("auto_toggle", area=area)]
    if name == "SYNC_TOGGLE":
        return [_action("dual_toggle")]
    if name in ("POWER_ON", "POWER_OFF"):
        return [_action("power_toggle")]
    if name == "REAR_DEFROST_TOGGLE":
        return [_action("window_defrost_toggle")]
    return []


MODULE = register(IntentModule("climate", INTENTS, PROMPT, translate))
