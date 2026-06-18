#!/usr/bin/env python3
"""
End-to-end manual-RAG demo (host, no audio): question → retrieve grounded
context from the ingested manual → LLM answers citing the manual.

This exercises the full 'how-to / where-is / explain' slice the `manual` intent
domain routes to in the live server. Uses the same Retriever + answer prompt.

Usage:
    python scripts/rag_demo.py \\
        --store models/manual.npz \\
        --embed models/nomic-embed-text-v1.5.Q4_K_M.gguf \\
        --llm models/Phi-3-mini-4k-instruct-Q6_K.gguf \\
        "what is the cost of grabbing the floor"
"""
import argparse

from fsttm.rag.retrieve import Retriever, build_answer_prompt


def main():
    ap = argparse.ArgumentParser("manual RAG demo")
    ap.add_argument("question", nargs="+")
    ap.add_argument("--store", required=True)
    ap.add_argument("--embed", required=True)
    ap.add_argument("--llm", required=True, help="chat GGUF for the answer")
    ap.add_argument("--name", default="Nina")
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args()
    question = " ".join(args.question)

    r = Retriever(args.store, args.embed, k=args.k)
    context, hits = r.context(question)
    print(f"\nQ: {question}")
    if not context:
        print("A: (nothing relevant in the manual)")
        return
    pages = sorted({h[2].get("page") for h in hits})
    print(f"   retrieved {len(hits)} passages from pages {pages}")

    from llama_cpp import Llama
    llm = Llama(model_path=args.llm, n_ctx=2048, n_gpu_layers=0, verbose=False)
    prompt = build_answer_prompt(args.name, question, context)
    out = llm.create_completion(prompt, max_tokens=120, temperature=0.2,
                                stop=["\n", "Question:", "Spoken answer:"],
                                stream=False)
    print("A:", out["choices"][0]["text"].strip())


if __name__ == "__main__":
    main()
