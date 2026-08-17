"""Dog intent language — schema shape, translate mapping, typed parsing,
dispatcher routing (all headless, no models, no ROS2)."""
from fsttm.domain import DispatchOutcome, DomainContext

from fsttm_dog.actions import Relation, SemanticTarget
from fsttm_dog.dispatcher import DogDispatcher
from fsttm_dog.provider import DOG_SCHEMA, PROVIDER


# ── schema (spec.md §§2-8) ───────────────────────────────────────────────────

def test_schema_shape():
    props = DOG_SCHEMA["properties"]
    assert DOG_SCHEMA["required"] == ["intent"]          # no areas, no extras
    assert "area" not in props
    for intent in ("LOCAL_ACTION", "QUERY", "NAVIGATE", "FIND", "FOLLOW",
                   "STOP", "CANCEL", "CHITCHAT", "UNKNOWN"):
        assert intent in props["intent"]["enum"]
    # nested grounded format
    assert props["target"]["properties"]["description"]["type"] == "string"
    assert props["constraints"]["items"]["properties"]["relation"]["enum"]
    assert "reference" in props["constraints"]["items"]["properties"]
    # VELOCITY is planner territory — never in the LLM grammar (§3)
    assert "VELOCITY" not in props["action"]["enum"]
    assert "linear_x" not in props


def test_spec_example_validates_against_schema():
    """The §18 example must be expressible: FIND red chair NEAR window."""
    ij = {"intent": "FIND",
          "target": {"type": "OBJECT", "description": "chair",
                     "attributes": {"color": "red"}},
          "constraints": [{"relation": "NEAR",
                           "reference": {"type": "OBJECT",
                                         "description": "window"}}]}
    t = SemanticTarget.from_json(ij["target"])
    assert t.description == "chair" and t.attributes == {"color": "red"}
    r = Relation.from_json(ij["constraints"][0])
    assert r.type == "NEAR" and r.reference.description == "window"


# ── provider ─────────────────────────────────────────────────────────────────

def test_translate_wraps_dog_command():
    ij = {"intent": "NAVIGATE", "goal": {"type": "REGION",
                                         "description": "kitchen"}}
    assert PROVIDER.translate(ij) == [{"cmd": "dog", "command": ij}]


def test_meta_intents_produce_no_command():
    for m in ("TIME", "DATE", "CHITCHAT", "UNKNOWN"):
        assert PROVIDER.translate({"intent": m}) == []
        assert PROVIDER.meta_intent({"intent": m}) == m
    assert PROVIDER.meta_intent({"intent": "FIND"}) is None


def test_prompt_teaches_the_language():
    p = PROVIDER.build_prompt()
    for kw in ("LOCAL_ACTION", "NAVIGATE", "FIND", "FOLLOW", "constraints",
               "NEXT_TO", "description"):
        assert kw in p
    assert "area" not in p                       # no car-isms
    lean = PROVIDER.build_prompt(variant="one-shot")
    assert "Examples" not in lean and len(lean) < len(p)


# ── dispatcher routing ───────────────────────────────────────────────────────

def _mk(emitted):
    ctx = DomainContext(config={"backend": "logging"},
                        emit=lambda text, level="info": emitted.append(text))
    d = DogDispatcher(ctx)
    return d


def _handle(d, ij):
    return d.handle(ij, PROVIDER.translate(ij))


def test_local_action_routes_to_go2(caplog):
    emitted = []
    d = _mk(emitted)
    out = _handle(d, {"intent": "LOCAL_ACTION", "action": "TURN",
                      "direction": "LEFT", "angle_deg": 90})
    assert out is DispatchOutcome.PASS
    assert any("LOCAL_ACTION" in t and "TURN" in t for t in emitted)


def test_stop_bypasses_planning():
    emitted = []
    d = _mk(emitted)
    _handle(d, {"intent": "STOP"})
    assert any("STOP (immediate)" in t for t in emitted)
    assert any("[nav] CANCEL" in t for t in emitted)


def test_find_unknown_explores():
    emitted = []
    d = _mk(emitted)
    _handle(d, {"intent": "FIND",
                "target": {"type": "OBJECT",
                           "description": "fire extinguisher"}})
    assert any("EXPLORE" in t and "fire extinguisher" in t for t in emitted)


def test_navigate_unresolved_warns_not_moves():
    emitted = []
    d = _mk(emitted)
    _handle(d, {"intent": "NAVIGATE",
                "goal": {"type": "OBJECT", "description": "chair"}})
    assert any("not in" in t for t in emitted)          # stub map is empty
    assert not any("[nav] NAVIGATE" in t for t in emitted)


def test_follow_routes_to_tracker():
    emitted = []
    d = _mk(emitted)
    _handle(d, {"intent": "FOLLOW",
                "target": {"type": "PERSON", "description": "the person"}})
    assert any("FOLLOW" in t for t in emitted)


def test_query_answers_deterministically():
    emitted = []
    d = _mk(emitted)
    ij = {"intent": "QUERY", "target": {"type": "OBJECT",
                                        "description": "chair"}}
    _handle(d, ij)
    ans = d.local_answer(ij)
    assert ans == "I don't see chair in my map yet."
    # non-QUERY intents defer to the LLM ack
    assert d.local_answer({"intent": "STOP"}) is None
