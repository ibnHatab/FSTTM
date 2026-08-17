"""Lights intent domain: head / fog / hazard / cabin (interior) lights."""
from fsttm_hvac.registry import IntentModule, register

INTENTS = ["LIGHTS_ON", "LIGHTS_OFF"]

# light_type selects which lamp the LIGHTS_ON/OFF intent controls.
EXTRA_PROPS = {
    "light_type": {"type": "string", "enum": ["head", "fog", "hazard", "cabin"]},
}

PROMPT = """\
## Light Intents

LIGHTS_ON / LIGHTS_OFF with a light_type:
| light_type | Trigger phrases |
|------------|-----------------|
| head   | "headlights on/off", "lights on/off", "turn on/off the lights" |
| fog    | "fog lights on/off", "turn on/off fog" |
| hazard | "hazard lights", "emergency lights", "flashers on/off" |
| cabin  | "cabin light on/off", "interior light on/off", "dome light", "reading light" |

Example: "cabin light on" → {"intent":"LIGHTS_ON","area":0,"light_type":"cabin"}
"""

_PROP_MAP = {
    "head":   "HEADLIGHTS_SWITCH",
    "fog":    "FOG_LIGHTS_SWITCH",
    "hazard": "HAZARD_LIGHTS_SWITCH",
    "cabin":  "CABIN_LIGHTS_SWITCH",
}


def translate(intent):
    name = intent.get("intent")
    if name not in INTENTS:
        return []
    on = (name == "LIGHTS_ON")
    lt = intent.get("light_type", "head")
    prop = _PROP_MAP.get(lt, "HEADLIGHTS_SWITCH")
    # set (not toggle) so on/off is idempotent
    return [{"cmd": "set", "name": prop, "area": 0, "value": on}]


MODULE = register(IntentModule("lights", INTENTS, PROMPT, translate,
                               extra_props=EXTRA_PROPS))
