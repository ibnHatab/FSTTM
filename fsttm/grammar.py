"""
System-intent grammar (attention / sleep classification).

Classifies an AWAKE utterance into a system action. "command" means "not a
system control — pass it on as a normal request". Kept deliberately tiny and
separate from the domain intent schema so future system intents (volume,
repeat, …) can be added here without touching domain intents.

The domain intent schema/grammar/prompt/translation live with the active
domain provider (fsttm.domain / contrib packages).
"""
from __future__ import annotations

from fsttm.domain import compile_grammar

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


def make_system_grammar():
    """Compile the system-intent schema → GBNF (cached)."""
    return compile_grammar(SYSTEM_INTENT_SCHEMA)


_SYSTEM_INTENT_PROMPT_TMPL = (
    "You control a voice assistant named {name}. The user has just addressed "
    "{name} by name. Decide if they are EXPLICITLY turning the assistant off, "
    "or giving a normal request.\n"
    'Answer "mute" or "sleep" ONLY for an explicit, unambiguous request to '
    'disable/dismiss the assistant itself — e.g. "voice off", "{name} off", '
    '"mute", "go to sleep", "stop listening", "that\'s all, goodbye".\n'
    'Answer "command" for EVERYTHING else, including any request to control the '
    'car or stop a car function — e.g. "stop the climate", "stop the music", '
    '"turn it off", "stop", "cancel". When in doubt, answer "command". A request '
    'to stop a CAR FUNCTION is never sleep/mute.\n'
    'Reply with ONLY the JSON {{"action": "..."}}.'
)


def make_system_prompt(name: str = "Nina") -> str:
    """The sleep-classifier prompt with the assistant's configured name."""
    return _SYSTEM_INTENT_PROMPT_TMPL.format(name=name)


# Default-name prompt for callers without config access.
SYSTEM_INTENT_PROMPT = make_system_prompt()
