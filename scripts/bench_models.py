"""Comparative benchmark: Phi-3-mini vs MiniCPM5-1B on intent classification + RAG.
Run on taycann (GPU). Measures intent accuracy + latency, and RAG answer
quality + latency on a fixed test set."""
import sys, time, json
sys.path.insert(0, ".")
from llama_cpp import Llama
from fsttm.intents.base import build_grammar
from fsttm import intents
from fsttm.rag.retrieve import Retriever, build_answer_prompt

EMBED = "models/nomic-embed-text-v1.5.Q4_K_M.gguf"
STORE = "models/taycan.npz"
MODELS = {
    "Phi-3-mini-Q6":   "models/Phi-3-mini-4k-instruct-Q6_K.gguf",
    "Phi-3-mini-Q4":   "models/Phi-3-mini-4k-instruct-Q4_K_M.gguf",
    "Phi-4-instr-Q6":  "models/Phi-4-mini-instruct-Q6_K.gguf",
    "Phi-4-reason-Q6": "models/Phi-4-mini-reasoning-Q6_K.gguf",
}

# ── intent test set: (utterance, expected_intent) over all domains ──
INTENT_TESTS = [
    ("make it warmer", "WARMER"),
    ("it's too hot in here", "COOLER"),
    ("set temperature to 21 degrees", "SET_TEMPERATURE"),
    ("turn on the air conditioning", "AC_ON"),
    ("defrost the windshield", "VENT_DEFROST"),
    ("turn on the cabin light", "LIGHTS_ON"),
    ("headlights off", "LIGHTS_OFF"),
    ("lock the doors", "DOOR_LOCK"),
    ("open the driver window", "WINDOW_OPEN"),
    ("warm my seat", "SEAT_HEAT_UP"),
    ("how do I open the trunk", "HOWTO"),
    ("where is the charging port", "LOCATE"),
    ("explain the tyre pressure light", "EXPLAIN"),
]
GRAMMAR = build_grammar(None)
SYS = intents.build_prompt(None)

# ── RAG test set: (question, must_be_grounded) ──
RAG_TESTS = [
    "how do I open the trunk",
    "how do I charge the car",
    "explain the tyre pressure warning light",
    "where is the windshield washer fluid",
    "how do I make a sandwich",   # should refuse
]

def run_intent(m):
    correct, lat = 0, []
    for utt, exp in INTENT_TESTS:
        prompt = f"{SYS}\n\nUser: {utt}\nJSON: "
        t=time.monotonic()
        out=m.create_completion(prompt, max_tokens=40, temperature=0, grammar=GRAMMAR, stop=["\n"])
        lat.append((time.monotonic()-t)*1000)
        try:
            got=json.loads(out["choices"][0]["text"]).get("intent")
        except Exception:
            got=None
        if got==exp: correct+=1
        else: print(f"    MISS {utt!r}: got {got} exp {exp}")
    return correct, len(INTENT_TESTS), sum(lat)/len(lat)

def run_rag(m, r):
    lat=[]
    for q in RAG_TESTS:
        ctx,hits=r.context(q)
        t=time.monotonic()
        out=m.create_completion(build_answer_prompt("Nina",q,ctx) if ctx else f"Say: I don't have that.\n",
                                max_tokens=50, temperature=0.2, stop=["\n","Question:"])
        lat.append((time.monotonic()-t)*1000)
        print(f"    {q[:32]:32s} → {out['choices'][0]['text'].strip()[:60]}")
    return sum(lat)/len(lat)

import os
r = Retriever(STORE, EMBED)
print("="*60)
for name, path in MODELS.items():
    if not os.path.exists(path):
        print(f"\n### {name}  — SKIP (not present: {path})")
        continue
    print(f"\n### {name}")
    m = Llama(model_path=path, n_ctx=2048, n_gpu_layers=-1, verbose=False)
    print("  -- INTENT --")
    c,n,il = run_intent(m)
    print("  -- RAG --")
    rl = run_rag(m, r)
    print(f"\n  RESULT {name}: intent {c}/{n} ({100*c/n:.0f}%) @ {il:.0f}ms | rag answer @ {rl:.0f}ms")
    del m
