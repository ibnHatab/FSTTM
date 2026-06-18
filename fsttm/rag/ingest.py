"""
Ingest a manual/PDF into a VectorStore.

Pipeline: pdftotext (poppler, no Python dep) → page-aware text → overlapping
word-window chunks → GGUF embeddings → flat numpy store (.npz).

Usage:
    python -m fsttm.rag.ingest MANUAL.pdf \\
        --embed models/nomic-embed-text-v1.5.Q4_K_M.gguf \\
        --out models/manual.npz
"""
import argparse
import re
import subprocess

from fsttm.rag.store import Embedder, VectorStore


def pdf_to_pages(pdf_path):
    """Return list[(page_no, text)] via pdftotext -layout, page-split on \\f."""
    out = subprocess.run(["pdftotext", "-layout", pdf_path, "-"],
                         capture_output=True, text=True, check=True).stdout
    pages = out.split("\f")
    return [(i + 1, p) for i, p in enumerate(pages) if p.strip()]


def _clean(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_pages(pages, words=180, overlap=40):
    """Overlapping word-window chunks, tagged with their source page."""
    chunks = []
    for page_no, raw in pages:
        toks = _clean(raw).split()
        if not toks:
            continue
        step = max(1, words - overlap)
        for start in range(0, len(toks), step):
            window = toks[start:start + words]
            if len(window) < 20 and chunks:        # merge a tiny tail back
                chunks[-1] = (chunks[-1][0] + " " + " ".join(window),
                              chunks[-1][1])
                break
            chunks.append((" ".join(window), {"page": page_no}))
            if start + words >= len(toks):
                break
    return chunks


def build_store(pdf_path, embed_model, source_name=None):
    pages = pdf_to_pages(pdf_path)
    chunked = chunk_pages(pages)
    texts = [c[0] for c in chunked]
    meta = [{**c[1], "source": source_name or pdf_path} for c in chunked]
    emb = Embedder(embed_model)
    vectors = emb.embed(texts)
    return VectorStore(vectors=vectors, chunks=texts, meta=meta)


def main():
    ap = argparse.ArgumentParser("RAG ingest")
    ap.add_argument("pdf")
    ap.add_argument("--embed", required=True, help="embedding GGUF path")
    ap.add_argument("--out", required=True, help="output .npz store")
    ap.add_argument("--source", default=None, help="display name for the source")
    args = ap.parse_args()

    store = build_store(args.pdf, args.embed, source_name=args.source)
    store.save(args.out)
    print(f"ingested {len(store)} chunks from {args.pdf} → {args.out}")


if __name__ == "__main__":
    main()
