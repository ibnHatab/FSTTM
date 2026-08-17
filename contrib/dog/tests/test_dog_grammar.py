"""Verification gate: the nested grounded schema (objects + arrays of
objects) must compile through LlamaGrammar.from_json_schema — the same call
the two-pass intent path uses at runtime."""
import pytest

llama_cpp = pytest.importorskip("llama_cpp")


def test_nested_schema_compiles_to_grammar():
    from fsttm.domain import compile_grammar
    from fsttm_dog.provider import DOG_SCHEMA
    g = compile_grammar(DOG_SCHEMA)
    assert g is not None
    # cached second call returns the same object
    assert compile_grammar(DOG_SCHEMA) is g


def test_provider_build_grammar():
    from fsttm_dog.provider import PROVIDER
    assert PROVIDER.build_grammar() is not None
