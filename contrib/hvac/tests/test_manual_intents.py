"""Manual (RAG) intent-domain wiring tests — schema, marker translation,
domain gating, prompt teaching. Engine-side VectorStore tests live in the
engine's tests/test_rag.py."""
from fsttm_hvac import provider as intents


def test_manual_domain_registered():
    assert "manual" in intents.INTENT_DOMAINS
    enum = intents.build_schema(None)["properties"]["intent"]["enum"]
    for i in ("HOWTO", "LOCATE", "EXPLAIN"):
        assert i in enum
    assert "topic" in intents.build_schema(None)["properties"]


def test_manual_translate_returns_marker():
    cmd = intents.translate(
        {"intent": "HOWTO", "topic": "open the trunk", "area": 0})
    assert cmd == [{"cmd": "manual", "intent": "HOWTO", "topic": "open the trunk"}]


def test_manual_excluded_when_domain_off():
    enum = intents.build_schema(["climate"])["properties"]["intent"]["enum"]
    assert "HOWTO" not in enum
    # marker not produced when manual domain disabled
    assert intents.translate(
        {"intent": "HOWTO", "topic": "x", "area": 0}, enabled=["climate"]) == []


def test_manual_prompt_teaches_howto():
    p = intents.build_prompt(["manual"])
    assert "HOWTO" in p and "trunk" in p
    # climate-only prompt must NOT teach manual intents
    assert "HOWTO" not in intents.build_prompt(["climate"])
