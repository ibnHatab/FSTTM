"""
RAG store tests — the pure VectorStore cosine search with synthetic vectors
(embedding/LLM paths need models, so those are not exercised here). The
manual intent-domain wiring tests live in contrib/hvac/tests.
"""
import numpy as np

from fsttm.rag.store import VectorStore


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
