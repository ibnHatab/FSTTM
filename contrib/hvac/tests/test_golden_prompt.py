"""Golden guard for the HVAC intent prompt + schema through the repackaging.

The domain-plugin refactor must keep the assembled prompt and JSON schema
BYTE-IDENTICAL — the Phi-3 few-shot behavior is tuned against exactly this
text (benchmarks in contrib/hvac/benchmarks/). If a change here is
intentional, re-run the intent benchmark before updating the hashes.

The hashes were captured against the pre-plugin fsttm.intents assembly; the
fsttm_hvac provider must keep producing the identical bytes.
"""
import hashlib
import json

GOLDEN_PROMPT_SHA = "1c5fda9aecdff5c23abcb00dfc1025645c4f0839a36fb1d7c2145c969a8fce46"
GOLDEN_SCHEMA_SHA = "3b017fb7a35d8fa777877aff195a6f7a74abeaec6f156b7b64a8398341686277"


def _build():
    from fsttm_hvac import provider as intents
    prompt = intents.build_prompt(None, variant="few-shot-extra")
    schema = intents.build_schema(None)
    return prompt, schema


def test_prompt_golden():
    prompt, _ = _build()
    assert hashlib.sha256(prompt.encode()).hexdigest() == GOLDEN_PROMPT_SHA


def test_schema_golden():
    _, schema = _build()
    digest = hashlib.sha256(
        json.dumps(schema, sort_keys=True).encode()).hexdigest()
    assert digest == GOLDEN_SCHEMA_SHA
