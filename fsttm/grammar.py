"""
GBNF/JSON-Schema grammars for llama-cpp-python constrained generation.

The HVAC intent schema now carries:
  - abstract intent tag (for display / logging)
  - area (0=both fronts, 1=driver/ROW_1_LEFT, 4=passenger/ROW_1_RIGHT)
  - optional parameters: temp, fan_level, delta

intent_to_protocol_cmd() translates an intent dict into the JSON command
understood by hvac-react/backend (PROTOCOL.md §4).  The caller POSTs that
command to POST /command or sends it over WebSocket.
"""
from __future__ import annotations  # PEP 563: lazy annotations for Python 3.8 (list[]/tuple[])
import json
from typing import Optional
from llama_cpp import LlamaGrammar

# The HVAC/vehicle intent schema, grammar, prompt and protocol translation now
# live in the per-domain modules under fsttm/intents/ (climate, lights, body),
# assembled on demand from the config-enabled set. This module keeps a thin
# compatibility surface (make_hvac_grammar / intent_to_protocol_cmd /
# TTS_TRANSLATION_EXAMPLES) plus the unrelated system-intent (attention) grammar.

# ── System-intent schema (attention / sleep) ──────────────────────────────────
# Classifies an AWAKE utterance into a system action. "command" means "not a
# system control — pass it on as a normal request". Kept deliberately tiny and
# separate from the HVAC schema so future system intents (volume, repeat, …) can
# be added here without touching domain intents.
SYSTEM_INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["command", "sleep", "mute"],
        },
    },
    "required": ["action"],
    "additionalProperties": False,
}

_SYSTEM_GRAMMAR = None


def _compile_silently(schema):
    import os
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(1)
    try:
        os.dup2(devnull, 1)
        return LlamaGrammar.from_json_schema(json.dumps(schema))
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)


def make_system_grammar() -> LlamaGrammar:
    """Compile the system-intent schema → GBNF (cached)."""
    global _SYSTEM_GRAMMAR
    if _SYSTEM_GRAMMAR is None:
        _SYSTEM_GRAMMAR = _compile_silently(SYSTEM_INTENT_SCHEMA)
    return _SYSTEM_GRAMMAR


SYSTEM_INTENT_PROMPT = (
    "You control a voice assistant named Nina. The user has just addressed Nina "
    "by name. Decide if they are EXPLICITLY turning the assistant off, or giving "
    "a normal request.\n"
    'Answer "mute" or "sleep" ONLY for an explicit, unambiguous request to '
    'disable/dismiss the assistant itself — e.g. "voice off", "Nina off", '
    '"mute", "go to sleep", "stop listening", "that\'s all, goodbye".\n'
    'Answer "command" for EVERYTHING else, including any request to control the '
    'car or stop a car function — e.g. "stop the climate", "stop the music", '
    '"turn it off", "stop", "cancel". When in doubt, answer "command". A request '
    'to stop a CAR FUNCTION is never sleep/mute.\n'
    'Reply with ONLY the JSON {"action": "..."}.'
)


def make_hvac_grammar(enabled=None) -> LlamaGrammar:
    """Compile the intent grammar for the enabled domains (cached per set).

    Delegates to fsttm.intents, which assembles the schema from the enabled
    intent modules (climate / lights / body). enabled=None → all domains.
    """
    from fsttm import intents
    return intents.build_grammar(enabled)


# ── Protocol translation ──────────────────────────────────────────────────────
# Translates an intent dict to the command objects accepted by
# POST /command (PROTOCOL.md §4).  Returns a list because some intents map to
# multiple commands (e.g. VENT_DEFROST_MAX = max_defrost_toggle).

def intent_to_protocol_cmd(intent: dict) -> list[dict]:
    """Map an intent dict → list of PROTOCOL.md command dicts. Delegates to the
    per-domain modules (fsttm.intents). All domains translate here; the grammar
    is what constrains which intents the model can emit."""
    from fsttm import intents
    return intents.translate(intent, enabled=None)


# ── Few-shot TTS translation examples (re-exported from the intents package) ──
from fsttm.intents import TTS_TRANSLATION_EXAMPLES  # noqa: E402,F401
