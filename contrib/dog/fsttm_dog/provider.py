"""
DogProvider — the `dog` entry in the engine's fsttm.domains registry.

Implements the Go2 natural-language intent language (spec.md): a small
constrained representation between the LLM and the robot. The LLM only maps
language → intent JSON; semantic perception resolves targets into spatial
goals; classical navigation executes them. The LLM never emits coordinates,
velocities, ROS messages, or shell commands (§18).

Schema notes vs the spec examples:
  - `intent` is the discriminator (engine convention; the spec's top-level
    categories map 1:1).
  - explicit `null`s become OMITTED fields (cleaner GBNF than union types;
    absent == null downstream).
  - NAVIGATE uses `goal`, QUERY/FIND/FOLLOW use `target`, both the same
    SemanticTarget shape — exactly as the spec examples read.
  - VELOCITY is not in the grammar (§3: planner/controller territory).
"""
from typing import Optional

from fsttm.domain import DomainContext, DomainDispatcher, compile_grammar

# ── schema (§2-§8) ────────────────────────────────────────────────────────────

_TARGET_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string",
                 "enum": ["OBJECT", "PERSON", "POSITION", "REGION", "ROOM",
                          "LANDMARK"]},
        "description": {"type": "string"},
        "attributes": {
            "type": "object",
            "properties": {
                "color": {"type": "string"},
                "size": {"type": "string"},
                "material": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    "required": ["type", "description"],
    "additionalProperties": False,
}

_CONSTRAINT_SCHEMA = {
    "type": "object",
    "properties": {
        "relation": {"type": "string",
                     "enum": ["NEAR", "NEXT_TO", "IN_FRONT_OF", "BEHIND",
                              "ON", "UNDER", "LEFT_OF", "RIGHT_OF",
                              "CLOSEST_TO"]},
        "reference": _TARGET_SCHEMA,
    },
    "required": ["relation", "reference"],
    "additionalProperties": False,
}

# Engine meta intents (TIME/DATE/CHITCHAT/UNKNOWN) ride in the same enum; the
# engine resolves them via meta_intent() below.
_ENGINE_META = ("TIME", "DATE", "CHITCHAT", "UNKNOWN")

INTENTS = ("LOCAL_ACTION", "QUERY", "NAVIGATE", "FIND", "FOLLOW",
           "STOP", "CANCEL") + _ENGINE_META

LOCAL_ACTIONS = ("STAND_UP", "SIT_DOWN", "LIE_DOWN", "STRETCH", "SHAKE",
                 "JUMP", "POUNCE", "TURN", "MOVE")

DOG_SCHEMA = {
    "type": "object",
    "properties": {
        # intent FIRST — the model must commit to the category before filling
        # parameters (grammar generates fields in declaration order).
        "intent": {"type": "string", "enum": list(INTENTS)},
        # LOCAL_ACTION parameters (§3)
        "action": {"type": "string", "enum": list(LOCAL_ACTIONS)},
        "direction": {"type": "string",
                      "enum": ["LEFT", "RIGHT", "AROUND", "FORWARD",
                               "BACKWARD"]},
        "angle_deg": {"type": "number"},
        "duration": {"type": "number"},
        # QUERY / FIND / FOLLOW target (§4-5, §12-13)
        "target": _TARGET_SCHEMA,
        # NAVIGATE goal (§6-8)
        "goal": _TARGET_SCHEMA,
        # spatial relations (§7)
        "constraints": {"type": "array", "items": _CONSTRAINT_SCHEMA},
    },
    "required": ["intent"],
    "additionalProperties": False,
}


# ── prompt ────────────────────────────────────────────────────────────────────

PROMPT_HEADER = """\
You are the voice interface of a quadruped robot dog. Map what the person
says to exactly one intent JSON per utterance. Respond with only the JSON.

You NEVER output coordinates, velocities, or device commands — only the
constrained intent below. The robot's planner and perception do the rest.

Intent categories:
| intent | meaning |
|--------|---------|
| LOCAL_ACTION | immediate body action (stand, sit, stretch, shake, jump, turn, move a step) |
| QUERY | question about what the robot sees/knows; does not move the robot |
| NAVIGATE | go to a goal the robot can resolve (object, region, room, landmark) |
| FIND | search for something (explore if not yet seen) |
| FOLLOW | follow a dynamic target (usually a person) |
| STOP | stop moving immediately |
| CANCEL | cancel the current task |
| TIME / DATE | clock/calendar questions |
| CHITCHAT | greeting or social remark to the robot |
| UNKNOWN | anything the robot genuinely cannot do |

Targets are open vocabulary: put the person's words in `description`
("the black office chair with wheels" stays as said); `attributes` carries
color/size/material when stated. Spatial phrases become `constraints`
(NEAR, NEXT_TO, IN_FRONT_OF, BEHIND, ON, UNDER, LEFT_OF, RIGHT_OF,
CLOSEST_TO) with a `reference` target. NAVIGATE uses `goal`; QUERY, FIND
and FOLLOW use `target`. Omit fields you don't need.
"""

_FEWSHOT = """\
## Examples (utterance → JSON)
"stand up" → {"intent":"LOCAL_ACTION","action":"STAND_UP"}
"sit" → {"intent":"LOCAL_ACTION","action":"SIT_DOWN"}
"turn left ninety degrees" → {"intent":"LOCAL_ACTION","action":"TURN","direction":"LEFT","angle_deg":90}
"turn around" → {"intent":"LOCAL_ACTION","action":"TURN","direction":"AROUND"}
"take a step forward" → {"intent":"LOCAL_ACTION","action":"MOVE","direction":"FORWARD"}
"where is the chair" → {"intent":"QUERY","target":{"type":"OBJECT","description":"chair"}}
"what is next to the table" → {"intent":"QUERY","target":{"type":"OBJECT","description":"table"},"constraints":[{"relation":"NEXT_TO","reference":{"type":"OBJECT","description":"table"}}]}
"go to the door" → {"intent":"NAVIGATE","goal":{"type":"OBJECT","description":"door"}}
"go to the chair next to the window" → {"intent":"NAVIGATE","goal":{"type":"OBJECT","description":"chair"},"constraints":[{"relation":"NEXT_TO","reference":{"type":"OBJECT","description":"window"}}]}
"go to the kitchen" → {"intent":"NAVIGATE","goal":{"type":"REGION","description":"kitchen"}}
"find a fire extinguisher" → {"intent":"FIND","target":{"type":"OBJECT","description":"fire extinguisher"}}
"go find the red chair near the window" → {"intent":"FIND","target":{"type":"OBJECT","description":"chair","attributes":{"color":"red"}},"constraints":[{"relation":"NEAR","reference":{"type":"OBJECT","description":"window"}}]}
"follow me" → {"intent":"FOLLOW","target":{"type":"PERSON","description":"the person"}}
"stop" → {"intent":"STOP"}
"never mind" → {"intent":"CANCEL"}
"what time is it" → {"intent":"TIME"}
"good boy" → {"intent":"CHITCHAT"}
"order me a pizza" → {"intent":"UNKNOWN"}
"""


class DogProvider:
    """fsttm.domains provider for the Go2 robot-dog deployment."""
    name = "dog"
    sub_domains: list = []      # monolithic — one intent language

    def build_schema(self, enabled=None) -> dict:
        return DOG_SCHEMA

    def build_grammar(self, enabled=None):
        return compile_grammar(DOG_SCHEMA)

    def build_prompt(self, enabled=None, variant=None) -> str:
        if variant == "one-shot":
            return PROMPT_HEADER
        return PROMPT_HEADER + "\n" + _FEWSHOT

    def translate(self, intent: dict, enabled=None) -> list:
        """Everything non-meta becomes one dog command for the dispatcher;
        meta intents produce no command (the engine answers them)."""
        name = (intent or {}).get("intent")
        if not name or name in _ENGINE_META:
            return []
        return [{"cmd": "dog", "command": dict(intent)}]

    def meta_intent(self, intent: dict) -> Optional[str]:
        name = (intent or {}).get("intent") if isinstance(intent, dict) else None
        return name if name in _ENGINE_META else None

    def chitchat_system(self, assistant_name: str) -> Optional[str]:
        return (f"You are {assistant_name}, a friendly robot dog's voice. "
                f"Reply to the person's remark in ONE short, warm spoken "
                f"sentence. No lists.")

    def make_dispatcher(self, ctx: DomainContext) -> DomainDispatcher:
        from fsttm_dog.dispatcher import DogDispatcher
        return DogDispatcher(ctx)


PROVIDER = DogProvider()
