"""
Manual intent domain + RAG store tests.

Embedding/LLM paths need models, so those are not exercised here — this covers
the manual intent wiring and the pure VectorStore cosine search with synthetic
vectors.
"""
import numpy as np

from fsttm import intents
from fsttm.rag.store import VectorStore


# ── manual intent domain ──────────────────────────────────────────────────────

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


# ── vector store cosine search ────────────────────────────────────────────────

def _norm(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def test_vector_store_search_ranks_by_cosine():
    vecs = np.vstack([_norm([1, 0, 0]), _norm([0.9, 0.1, 0]), _norm([0, 1, 0])])
    store = VectorStore(vectors=vecs.astype(np.float32),
                        chunks=["a", "b", "c"],
                        meta=[{"page": 1}, {"page": 2}, {"page": 3}])
    hits = store.search(_norm([1, 0, 0]), k=2)
    assert [h[1] for h in hits] == ["a", "b"]     # closest two, in order
    assert hits[0][0] > hits[1][0]


def test_vector_store_save_load_roundtrip(tmp_path):
    vecs = np.vstack([_norm([1, 0]), _norm([0, 1])]).astype(np.float32)
    store = VectorStore(vectors=vecs, chunks=["x", "y"],
                        meta=[{"page": 5, "source": "m.pdf"}, {"page": 6}])
    p = str(tmp_path / "s.npz")
    store.save(p)
    loaded = VectorStore.load(p)
    assert loaded.chunks == ["x", "y"]
    assert loaded.meta[0]["page"] == 5 and loaded.meta[0]["source"] == "m.pdf"
    assert loaded.search(_norm([1, 0]), k=1)[0][1] == "x"


def test_empty_store_search():
    assert VectorStore().search(_norm([1, 0]), k=3) == []
