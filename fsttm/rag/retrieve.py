"""
Retrieve grounded context from an ingested manual store, and build a prompt for
the LLM to answer 'how to / where is / explain' questions citing the manual.

Usage (CLI, for testing on the host):
    python -m fsttm.rag.retrieve "how do I open the trunk" \\
        --store models/manual.npz \\
        --embed models/nomic-embed-text-v1.5.Q4_K_M.gguf
"""
import argparse

from fsttm.rag.store import Embedder, VectorStore


class Retriever:
    def __init__(self, store_path, embed_model, k=4, min_score=0.30,
                 embed_gpu=False):
        self.store = VectorStore.load(store_path)
        # embed_gpu=False (default) → CPU embedder, so it doesn't fight the LLM
        # for VRAM on a shared-memory Jetson. -1 offloads all layers on a GPU box.
        self.embed = Embedder(embed_model, n_gpu_layers=(-1 if embed_gpu else 0))
        self.k = k
        self.min_score = min_score

    def retrieve(self, query, k=None):
        """Return top-k (score, chunk, meta) above min_score."""
        qv = self.embed.embed(query)[0]
        hits = self.store.search(qv, k=k or self.k)
        return [h for h in hits if h[0] >= self.min_score]

    def context(self, query, k=None):
        """Format retrieved chunks as a cited context block for the LLM."""
        hits = self.retrieve(query, k=k)
        if not hits:
            return "", []
        blocks = []
        for score, chunk, meta in hits:
            pg = meta.get("page", "?")
            blocks.append(f"[p.{pg}] {chunk}")
        return "\n\n".join(blocks), hits


# Prompt for the grounded answer — concise, SPOKEN-friendly (no page markers /
# brackets, since this is read aloud), refuses when the manual doesn't cover it.
def build_answer_prompt(name, query, context):
    return (
        f"You are {name}, a car assistant. Answer the question using ONLY the "
        f"manual excerpts below. Reply with ONE short spoken sentence (max ~25 "
        f"words), plain prose — no page numbers, no brackets, no lists. If the "
        f"excerpts don't answer it, say: I don't have that in the manual.\n\n"
        f"Manual excerpts:\n{context}\n\n"
        f"Question: {query}\nSpoken answer:"
    )


def main():
    ap = argparse.ArgumentParser("RAG retrieve")
    ap.add_argument("query")
    ap.add_argument("--store", required=True)
    ap.add_argument("--embed", required=True)
    ap.add_argument("-k", type=int, default=4)
    args = ap.parse_args()

    r = Retriever(args.store, args.embed, k=args.k)
    hits = r.retrieve(args.query)
    if not hits:
        print("(no relevant passages above threshold)")
        return
    for score, chunk, meta in hits:
        print(f"\n── score {score:.3f}  p.{meta.get('page','?')} ──")
        print(chunk[:400] + ("…" if len(chunk) > 400 else ""))


if __name__ == "__main__":
    main()
