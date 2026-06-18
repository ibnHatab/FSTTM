"""
Intent domains, assembled on demand from the enabled set.

Importing this package registers all built-in domains (climate, lights, body, manual).
The server passes the config-enabled domain names to build_schema / build_grammar
/ build_prompt / translate; pass None for "all".

Public API:
    INTENT_DOMAINS                       — registered domain names, in order
    build_schema(enabled)  -> dict
    build_grammar(enabled) -> LlamaGrammar   (compiled + cached)
    build_prompt(enabled)  -> str            (header + fragments + footer)
    translate(intent, enabled) -> list[dict] (PROTOCOL.md commands)
    TTS_TRANSLATION_EXAMPLES                 — 2nd-pass spoken-ack examples
"""
from fsttm.intents.base import (  # noqa: F401
    build_schema as _build_schema,
    build_grammar,
    translate,
    all_names,
    _enabled_modules,
)
from fsttm.intents.base import build_prompt as _build_prompt

# Register the built-in domains (import side effect).
from fsttm.intents import climate, lights, body, manual  # noqa: F401,E402

INTENT_DOMAINS = all_names()

# Prompt header (zone addressing) + footer is added by the server when it builds
# the full prompt; kept here so the wording lives with the domains.
PROMPT_HEADER = """\
You are a voice assistant for a car's control system. Map what the driver says
to exactly one structured intent JSON per utterance. Respond with only the JSON.

Only the listed car-control intents are valid. Use STATUS ONLY for a question
about the cabin CLIMATE settings (current/set temperature, fan speed, AC on/off).
Any question about the clock or time of day ("what time is it", "what is the
time now", "do you have the time") is TIME, NOT STATUS. A question about the
calendar date or day ("what's the date", "what day is it") is DATE. A greeting
or social remark to the assistant ("hello", "hi Nina", "how are you", "thanks",
"good morning") is CHITCHAT. Something the car genuinely cannot do (weather,
jokes, phone calls, navigation, web search) is UNKNOWN. Never invent an intent
or force an unrelated request into a control intent.

## Zone / Area Addressing

| Zone | area | Trigger phrases |
|------|------|-----------------|
| All / global | 0 | "all", "every", no zone mentioned |
| Driver (front-left) | 1 | "my", "driver", "left", "my side" |
| Passenger (front-right) | 4 | "passenger", "right", "his/her side" |
| Rear-left | 16 | "rear left", "back left" |
| Rear-right | 64 | "rear right", "back right" |
| Trunk | 256 | "trunk", "boot" |
"""


# Few-shot footer — worked examples that teach the PATTERNS the model gets wrong
# under grammar pressure (not specific test phrases): possessive→area, "by N"→
# delta, always-set light_type, and refusing out-of-domain to UNKNOWN/STATUS.
# Each line is tagged with the intent it demonstrates so we only show examples
# for ENABLED domains (showing a disabled intent would teach a bad mapping).
# Adding the few-shot block lifted intent+field accuracy ~70%→90% in the
# scripts/opt_intent.py sweep (see memory: intent-optimization).
_FEWSHOT = [
    # (intent_label, "utterance" → JSON) — label gates inclusion by enabled domain.
    # Trimmed to one example per PATTERN (the KV-prefix cache makes prompt length
    # cheap, but a tight prompt still classifies better). Kept: possessive→area,
    # "by N"→delta, light_type, seat up-direction, manual how-to, and refusal.
    ("WARMER",          '"raise the temperature by two" → {"intent":"WARMER","area":0,"delta":2}'),
    ("COOLER",          '"it\'s too hot in here" → {"intent":"COOLER","area":0,"delta":1}'),
    ("SET_TEMPERATURE", '"set my temperature to 22" → {"intent":"SET_TEMPERATURE","area":1,"temp":22}'),
    ("AC_ON",           '"turn on the air conditioning" → {"intent":"AC_ON","area":0}'),
    ("VENT_DEFROST",    '"defrost the windshield" → {"intent":"VENT_DEFROST","area":0}'),
    ("LIGHTS_ON",       '"cabin light on" → {"intent":"LIGHTS_ON","area":0,"light_type":"cabin"}'),
    ("LIGHTS_ON",       '"headlights on" → {"intent":"LIGHTS_ON","area":0,"light_type":"head"}'),
    ("WINDOW_OPEN",     '"open my window" → {"intent":"WINDOW_OPEN","area":1,"position":50}'),
    ("SEAT_HEAT_UP",    '"warm my seat" → {"intent":"SEAT_HEAT_UP","area":1,"delta":1}'),
    ("SEAT_COOL_UP",    '"cool the passenger seat" → {"intent":"SEAT_COOL_UP","area":4,"delta":1}'),
    # Manual/RAG how-to questions — without an example the model maps these to a
    # random control intent ("explain how to open trunk" → VENT_FACE).
    ("HOWTO",   '"how do I open the trunk" → {"intent":"HOWTO","area":0,"topic":"open the trunk"}'),
    ("EXPLAIN", '"explain the tyre pressure light" → {"intent":"EXPLAIN","area":0,"topic":"tyre pressure light"}'),
    # Meta — answer time/date, refuse true out-of-domain.
    ("TIME",    '"what time is it" → {"intent":"TIME","area":0}'),
    ("TIME",    '"what is the time now" → {"intent":"TIME","area":0}'),
    ("DATE",    '"what day is it" → {"intent":"DATE","area":0}'),
    ("STATUS",  '"what\'s the temperature in here" → {"intent":"STATUS","area":0}'),
    ("CHITCHAT",'"hello Nina" → {"intent":"CHITCHAT","area":0}'),
    ("CHITCHAT",'"how are you" → {"intent":"CHITCHAT","area":0}'),
    ("UNKNOWN", '"tell me a joke" → {"intent":"UNKNOWN","area":0}'),
]

# EXTRA few-shot — a second tier of examples appended only for the
# "few-shot-extra" variant (more coverage at the cost of a longer prompt).
# Same (label → line) tagging so they stay domain-gated.
_FEWSHOT_EXTRA = [
    ("WARMER",          '"warmer please" → {"intent":"WARMER","area":0,"delta":1}'),
    ("FAN_UP",          '"more air" → {"intent":"FAN_UP","area":0,"delta":1}'),
    ("LIGHTS_OFF",      '"turn off the headlights" → {"intent":"LIGHTS_OFF","area":0,"light_type":"head"}'),
    ("DOOR_UNLOCK",     '"unlock my door" → {"intent":"DOOR_UNLOCK","area":1}'),
    ("WINDOW_CLOSE",    '"close all windows" → {"intent":"WINDOW_CLOSE","area":0}'),
    ("SEAT_COOL_DOWN",  '"less seat cooling" → {"intent":"SEAT_COOL_DOWN","area":1,"delta":1}'),
    ("TIME",            '"do you have the time" → {"intent":"TIME","area":0}'),
    ("DATE",            '"what\'s the date today" → {"intent":"DATE","area":0}'),
    ("CHITCHAT",        '"good morning" → {"intent":"CHITCHAT","area":0}'),
    ("CHITCHAT",        '"thank you Nina" → {"intent":"CHITCHAT","area":0}'),
    ("UNKNOWN",         '"what\'s the weather like" → {"intent":"UNKNOWN","area":0}'),
    ("UNKNOWN",         '"call my wife" → {"intent":"UNKNOWN","area":0}'),
]

# Meta intents present in every grammar (base.py appends them) — their few-shot
# lines are always relevant regardless of which control domains are enabled.
_META_INTENTS = {"TIME", "DATE", "STATUS", "CHITCHAT", "UNKNOWN"}

# Prompt variants — switchable via config (system.prompt_variant) so we can A/B
# accuracy vs latency and bisect regressions:
#   one-shot       : header + domain tables only (the old lean prompt; fastest)
#   few-shot       : + curated worked examples (production default; +accuracy)
#   few-shot-extra : few-shot + a second tier of examples (max coverage)
PROMPT_VARIANTS = ("one-shot", "few-shot", "few-shot-extra")
_DEFAULT_VARIANT = "few-shot"


def _filter_lines(pairs, enabled):
    allowed = set(_META_INTENTS)
    for m in _enabled_modules(enabled):
        allowed.update(getattr(m, "intents", ()))
    return [text for label, text in pairs if label in allowed]


def _fewshot_footer(enabled, extra=False):
    """Few-shot footer for the enabled domains. extra=True also appends the
    second-tier _FEWSHOT_EXTRA examples."""
    lines = _filter_lines(_FEWSHOT, enabled)
    if extra:
        lines += _filter_lines(_FEWSHOT_EXTRA, enabled)
    if not lines:
        return None
    return ("## Examples (utterance → JSON; copy the field discipline)\n"
            "Always include light_type for lights. Use area 1 for \"my/driver\", "
            "4 for \"passenger\". Read \"by N\" as delta:N. STATUS is for CLIMATE "
            "readings only; clock/time → TIME. If not a car control, answer "
            "UNKNOWN.\n"
            + "\n".join(lines))


def build_prompt(enabled=None, fewshot=True, variant=None):
    """Assemble the intent system prompt for the given variant.

    variant: "one-shot" | "few-shot" | "few-shot-extra". If None, falls back to
    the `fewshot` bool for back-compat (True→few-shot, False→one-shot).
    """
    if variant is None:
        variant = _DEFAULT_VARIANT if fewshot else "one-shot"
    if variant not in PROMPT_VARIANTS:
        variant = _DEFAULT_VARIANT
    if variant == "one-shot":
        footer = None
    else:
        footer = _fewshot_footer(enabled, extra=(variant == "few-shot-extra"))
    return _build_prompt(enabled, header=PROMPT_HEADER, footer=footer)


def build_schema(enabled=None):
    return _build_schema(enabled)


# 2nd-pass (intent JSON → spoken acknowledgment) few-shot examples. Domain-
# agnostic enough to keep as one block; only the lines for enabled domains
# matter at inference but extra examples are harmless context.
TTS_TRANSLATION_EXAMPLES = """\
Translate this intent JSON to a spoken acknowledgment (8 words max, no JSON):
{"intent":"WARMER","area":0,"delta":1} → Turning up the heat.
{"intent":"COOLER","area":1,"delta":2} → Cooling driver side by two steps.
{"intent":"SET_TEMPERATURE","area":4,"temp":22.0} → Setting passenger temperature to twenty-two.
{"intent":"VENT_DEFROST","area":0} → Activating windshield defrost.
{"intent":"AC_OFF","area":0} → Turning off the air conditioning.
{"intent":"AUTO_ON","area":0} → Switching to automatic climate mode.
{"intent":"DOOR_LOCK","area":0} → All doors locked.
{"intent":"LIGHTS_ON","area":0,"light_type":"head"} → Headlights on.
{"intent":"LIGHTS_ON","area":0,"light_type":"cabin"} → Cabin light on.
{"intent":"WINDOW_OPEN","area":1,"position":50} → Driver window opened halfway.
{"intent":"SEAT_HEAT_UP","area":1,"delta":1} → Driver seat heat increased.
{"intent":"STATUS","area":0} → Checking your current settings.
{"intent":"UNKNOWN","area":0} → Sorry, I can't do that.\
"""
