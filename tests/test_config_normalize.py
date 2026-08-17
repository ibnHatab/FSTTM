"""Legacy-config shim: pre-0.2 keys (hvac_intent, hvac_backend, system.manual*)
must normalize to the same effective Config as the new schema, so an old
on-device yaml still boots identically."""
from fsttm.config import Config, normalize_config

_BASE = {
    "vad": {"rate": 16000, "device_name": "pulse"},
    "stt": {"model": "models/ggml-base.en-q5_1.bin"},
    "tts": {"model": "voice.onnx"},
    "gpt": {"model": "model.gguf", "n_ctx": 4096},
    "server": {"http": {"host": "0.0.0.0", "port": 8080,
                        "request_max_size": 1048576}},
    "log": {"level": [{"logger": "fsttm", "level": "INFO"}]},
}


def _legacy():
    raw = dict(_BASE)
    raw["hvac_backend"] = {"url": "http://127.0.0.1:8000", "timeout": 2.0}
    raw["system"] = {
        "name": "Nina",
        "hvac_intent": True,
        "hvac_prompt": "prompts/x.txt",
        "intent_domains": ["climate", "manual"],
        "manual": True,
        "manual_store": "store.npz",
        "manual_embed": "embed.gguf",
        "manual_embed_gpu": True,
    }
    return raw


def _modern():
    raw = dict(_BASE)
    raw["domains"] = {"hvac": {
        "backend_url": "http://127.0.0.1:8000", "timeout": 2.0,
        "manual": {"enabled": True, "store": "store.npz",
                   "embed": "embed.gguf", "embed_gpu": True},
    }}
    raw["system"] = {
        "name": "Nina",
        "intent_mode": True,
        "domain": "hvac",
        "intent_prompt": "prompts/x.txt",
        "intent_domains": ["climate", "manual"],
    }
    return raw


def test_legacy_equals_modern():
    old = Config(**normalize_config(_legacy()))
    new = Config(**normalize_config(_modern()))
    assert old.system.intent_mode and new.system.intent_mode
    assert old.system.domain == new.system.domain == "hvac"
    assert old.system.intent_prompt == new.system.intent_prompt
    assert old.domains == new.domains


def test_legacy_implies_hvac_domain():
    raw = dict(_BASE)
    raw["hvac_backend"] = {"url": "http://x:1"}
    cfg = Config(**normalize_config(raw))
    assert cfg.system.domain == "hvac"
    assert cfg.domains["hvac"]["backend_url"] == "http://x:1"


def test_modern_passthrough_untouched():
    cfg = Config(**normalize_config(_modern()))
    assert cfg.domains["hvac"]["manual"]["embed_gpu"] is True
    assert cfg.system.intent_domains == ["climate", "manual"]


def test_no_domain_no_hvac_block():
    cfg = Config(**normalize_config(dict(_BASE)))
    assert cfg.system.domain is None
    assert cfg.domains == {}
