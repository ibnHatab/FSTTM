"""
Intent-module registry.

Each *intent domain* (climate, lights, body, …) is an ``IntentModule`` that owns:
  - its slice of the intent enum,
  - any extra JSON-schema properties it needs (e.g. ``light_type``),
  - a prompt fragment teaching the model those intents,
  - a translate() mapping each of its intents → PROTOCOL.md command dict(s).

The active schema, LlamaGrammar, prompt, and protocol dispatcher are ASSEMBLED
from the enabled modules — so domains can be toggled in config without touching
the others, and new domains drop in by registering a module.
"""

# Shared zone/area constants (PROTOCOL.md addressing).
AREA_BOTH    = 0   # broadcast for set commands; store expands to all zones
AREA_BOTH_BM = 5   # ROW_1_LEFT | ROW_1_RIGHT = 0x5, used for per-zone action args
AREA_DRIVER  = 1   # ROW_1_LEFT  = 0x0001
AREA_PASS    = 4   # ROW_1_RIGHT = 0x0004

# Common schema fragments every domain may reference.
_AREA_PROP = {"type": "integer", "enum": [0, 1, 4, 16, 64, 256]}


class IntentModule:
    """One toggleable intent domain. ``translate`` is f(intent_dict)->list|None;
    returning None means 'not mine' (lets the dispatcher try the next module)."""
    def __init__(self, name, intents, prompt, translate, extra_props=None):
        self.name = name
        self.intents = list(intents)
        self.prompt = prompt.strip("\n")
        self.translate = translate
        self.extra_props = dict(extra_props or {})


# Registration order = assembly order (schema enum order, prompt order).
_REGISTRY = []
_BY_NAME = {}


def register(module):
    if module.name in _BY_NAME:
        raise ValueError(f"intent module {module.name!r} already registered")
    _REGISTRY.append(module)
    _BY_NAME[module.name] = module
    return module


def all_names():
    return [m.name for m in _REGISTRY]


def _enabled_modules(enabled):
    """enabled: None → all; else an iterable of module names (in registry order)."""
    if enabled is None:
        return list(_REGISTRY)
    want = set(enabled)
    return [m for m in _REGISTRY if m.name in want]


# ── assembly ──────────────────────────────────────────────────────────────────

def build_schema(enabled=None):
    """Assemble the JSON schema from the enabled modules.

    Property ORDER matters: LlamaGrammar generates fields in this order, and the
    model must commit to `intent` FIRST (then area/params) — otherwise it emits
    `{"area": …` before knowing the intent, which truncates or picks wrong areas.
    So `intent` leads, matching the original unified schema.
    """
    mods = _enabled_modules(enabled)
    intent_enum = []
    extra = {}
    for m in mods:
        intent_enum.extend(m.intents)
        extra.update(m.extra_props)
    # Meta intents always available. TIME/DATE → local clock; CHITCHAT → a real
    # conversational reply (server routes it to a plain chat Generate); UNKNOWN →
    # polite refusal of genuine out-of-domain requests.
    intent_enum += ["TIME", "DATE", "STATUS", "CHITCHAT", "UNKNOWN"]

    props = {"intent": {"type": "string", "enum": intent_enum},
             "area": dict(_AREA_PROP),
             "temp": {"type": "number"},
             "fan_level": {"type": "integer"},
             "delta": {"type": "integer"}}
    props.update(extra)   # light_type, position, … after the common params
    return {
        "type": "object",
        "properties": props,
        "required": ["intent", "area"],
        "additionalProperties": False,
    }


def build_prompt(enabled=None, header=None, footer=None):
    """Assemble the intent system prompt from the enabled modules' fragments."""
    mods = _enabled_modules(enabled)
    parts = []
    if header:
        parts.append(header.strip("\n"))
    for m in mods:
        parts.append(m.prompt)
    if footer:
        parts.append(footer.strip("\n"))
    return "\n\n".join(parts)


def translate(intent, enabled=None):
    """Dispatch an intent dict to the first enabled module that owns it."""
    name = intent.get("intent", "UNKNOWN")
    for m in _enabled_modules(enabled):
        if name in m.intents:
            out = m.translate(intent)
            return out or []
    return []   # STATUS, UNKNOWN, unknown, or disabled-domain → no command


# ── grammar (compiled + cached by the engine's shared helper) ─────────────────

def build_grammar(enabled=None):
    """Compile (and cache) the GBNF grammar for the enabled module set."""
    from fsttm.domain import compile_grammar
    return compile_grammar(build_schema(enabled))
