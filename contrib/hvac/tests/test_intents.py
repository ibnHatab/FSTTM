"""
Intent-module registry tests (fsttm_hvac).

Pure assembly + translation — no model. build_grammar is the only thing needing
llama_cpp, so it's not exercised here; schema/prompt/translate are.
"""
from fsttm_hvac import provider as intents


def test_domains_registered():
    assert intents.INTENT_DOMAINS == ["climate", "lights", "body", "manual"]


def test_schema_all_domains_has_cabin_support():
    sch = intents.build_schema(None)
    props = sch["properties"]
    assert "light_type" in props
    assert "cabin" in props["light_type"]["enum"]
    assert "position" in props          # from body (windows)
    enum = props["intent"]["enum"]
    assert "LIGHTS_ON" in enum and "WARMER" in enum and "SEAT_HEAT_UP" in enum
    assert enum[-5:] == ["TIME", "DATE", "STATUS", "CHITCHAT", "UNKNOWN"]   # meta always last


def test_cabin_light_translates():
    cmd = intents.translate(
        {"intent": "LIGHTS_ON", "area": 0, "light_type": "cabin"})
    assert cmd == [{"cmd": "set", "name": "CABIN_LIGHTS_SWITCH",
                    "area": 0, "value": True}]


def test_climate_only_excludes_lights_and_body():
    sch = intents.build_schema(["climate"])
    props = sch["properties"]
    assert "light_type" not in props
    assert "position" not in props
    enum = props["intent"]["enum"]
    assert "WARMER" in enum
    assert "LIGHTS_ON" not in enum and "SEAT_HEAT_UP" not in enum
    # a disabled-domain intent translates to no command
    assert intents.translate(
        {"intent": "LIGHTS_ON", "light_type": "cabin", "area": 0},
        enabled=["climate"]) == []


def test_prompt_assembly_per_domain():
    p_all = intents.build_prompt(None)
    p_clim = intents.build_prompt(["climate"])
    assert "cabin" in p_all and "Climate Intents" in p_all
    assert "Light Intents" in p_all
    assert "Light Intents" not in p_clim  # lights not taught
    assert "Zone / Area Addressing" in p_clim   # header always present


def test_translate_parity_climate_sample():
    # spot-check several climate intents produce action commands
    assert intents.translate({"intent": "WARMER", "area": 0, "delta": 2}) == \
        [{"cmd": "action", "action": "bump_temperature",
          "args": {"up": True, "area": 5}}] * 2
    assert intents.translate({"intent": "AC_ON", "area": 0}) == \
        [{"cmd": "action", "action": "ac_toggle"}]


def test_meta_and_unknown_produce_no_command():
    assert intents.translate({"intent": "STATUS", "area": 0}) == []
    assert intents.translate({"intent": "UNKNOWN", "area": 0}) == []
    assert intents.translate({"intent": "NOPE", "area": 0}) == []
