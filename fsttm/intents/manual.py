"""
Manual intent domain: questions answered from the vehicle manual via RAG
(how-to, where-is, explain) rather than a device command.

These intents don't map to a PROTOCOL.md command — they signal the server to run
manual retrieval + a grounded LLM answer. translate() returns a single marker
command {"cmd": "manual", ...} so the dispatcher recognises the RAG path; the
server reads it and does the retrieval instead of POSTing to the HVAC backend.
"""
from fsttm.intents.base import IntentModule, register

INTENTS = ["HOWTO", "LOCATE", "EXPLAIN"]

# `topic` carries the thing being asked about, so retrieval has a clean query
# even when the raw utterance is conversational.
EXTRA_PROPS = {
    "topic": {"type": "string"},
}

PROMPT = """\
## Manual / How-To Intents (answered from the vehicle manual)

| Intent | Use for | Examples |
|--------|---------|----------|
| HOWTO   | how to do something | "how do I open the trunk", "how to pair my phone" |
| LOCATE  | where something is  | "where is the hazard button", "where's the actuator" |
| EXPLAIN | what something means | "explain the tyre-pressure indicator", "what is that warning light" |

Set `topic` to the subject in a few words (e.g. "open the trunk", "hazard
button", "tyre-pressure indicator"). These are MANUAL questions, not device
commands — use them only when the user is asking how/where/what, not telling the
car to do something.
"""


def translate(intent):
    name = intent.get("intent")
    if name not in INTENTS:
        return []
    # Marker command — the server runs RAG retrieval for this, not a protocol cmd.
    return [{"cmd": "manual", "intent": name,
             "topic": intent.get("topic", "")}]


MODULE = register(IntentModule("manual", INTENTS, PROMPT, translate,
                               extra_props=EXTRA_PROPS))
