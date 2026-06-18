"""
Minimal local vector store: GGUF embeddings (llama.cpp) + numpy cosine search.

No external vector DB — the corpus (a car manual / paper) is small, so a flat
in-memory matrix with cosine similarity is plenty and stays in one runtime
(llama.cpp), matching the rest of the stack. The index persists to a single
.npz (vectors + chunk text + metadata) so ingest runs once.
"""
import json
import os

import numpy as np


class Embedder:
    """Wraps a GGUF embedding model (e.g. nomic-embed-text) via llama.cpp.

    n_gpu_layers offloads the model to the GPU when llama.cpp was built with
    CUDA (default: all layers if GPU offload is supported, else CPU). The embed
    model is tiny, so this is a big ingest/query speedup on a GPU box and a no-op
    on a CPU-only build."""

    def __init__(self, model_path, n_ctx=512, n_threads=None, n_gpu_layers=None):
        from llama_cpp import Llama
        if n_gpu_layers is None:
            # Default to CPU: on a shared-memory Jetson the embedder loaded on the
            # GPU competes with the Phi-3 LLM + whisper for the VRAM pool and
            # crashes with "CUDA error: out of memory" (cuMemAddressReserve). The
            # embedder is small and runs only on RAG queries, so CPU is fine
            # (~100-200ms). Pass n_gpu_layers=-1 explicitly to offload on a box
            # with spare VRAM.
            n_gpu_layers = 0
        kw = dict(model_path=model_path, embedding=True, n_ctx=n_ctx,
                  n_gpu_layers=n_gpu_layers, verbose=False)
        if n_threads:
            kw["n_threads"] = n_threads
        self._llm = Llama(**kw)

    def embed(self, texts):
        """texts: str or list[str] → float32 array (N, dim), L2-normalised."""
        if isinstance(texts, str):
            texts = [texts]
        vecs = []
        for t in texts:
            e = self._llm.embed(t)
            # some builds return a list-of-token-embeddings for long input;
            # collapse to a single vector by mean-pooling if needed.
            arr = np.asarray(e, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr.mean(axis=0)
            vecs.append(arr)
        m = np.vstack(vecs)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (m / norms).astype(np.float32)


class VectorStore:
    """Flat cosine-similarity store over normalised embeddings."""

    def __init__(self, vectors=None, chunks=None, meta=None):
        self.vectors = vectors if vectors is not None else np.zeros((0, 0), np.float32)
        self.chunks = chunks or []          # list[str]
        self.meta = meta or []              # list[dict] (page, source, …)

    def __len__(self):
        return len(self.chunks)

    def search(self, query_vec, k=4):
        """query_vec: (dim,) normalised → list[(score, chunk, meta)] top-k.

        Flat numpy cosine. Benchmarked on the Jetson (42-chunk corpus): ~50µs/
        query — ~1000x cheaper than the ~50ms embed step, so a flat search is
        plenty for manual-sized corpora (FAISS gave identical results, ~10µs
        faster, only worth it at 10k+ vectors)."""
        if len(self) == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32).ravel()
        sims = self.vectors @ q            # cosine (both normalised)
        idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.chunks[i], self.meta[i]) for i in idx]

    # ── persistence ───────────────────────────────────────────────────────────
    # Portable across numpy 1.x/2.x: the .npz holds ONLY the float vector matrix
    # (numeric arrays are version-stable). Text + metadata go in a sidecar JSON,
    # so a store ingested on one machine loads on another (the previous
    # dtype=object + allow_pickle format embedded numpy._core and broke 2.x→1.x).
    def _sidecar(self, path):
        return (path[:-4] if path.endswith(".npz") else path) + ".chunks.json"

    def save(self, path):
        np.savez_compressed(path, vectors=self.vectors)
        with open(self._sidecar(path), "w") as f:
            json.dump({"chunks": self.chunks, "meta": self.meta}, f)

    @classmethod
    def load(cls, path):
        sidecar = (path[:-4] if path.endswith(".npz") else path) + ".chunks.json"
        if os.path.exists(sidecar):
            with np.load(path) as dz:
                vectors = dz["vectors"].astype(np.float32)
            with open(sidecar) as f:
                meta_obj = json.load(f)
            return cls(vectors=vectors, chunks=meta_obj["chunks"],
                       meta=meta_obj["meta"])
        # ── legacy fallback: old single-file dtype=object .npz ──
        d = np.load(path, allow_pickle=True)
        meta = [json.loads(m) for m in d["meta"].tolist()]
        return cls(vectors=d["vectors"].astype(np.float32),
                   chunks=d["chunks"].tolist(), meta=meta)
