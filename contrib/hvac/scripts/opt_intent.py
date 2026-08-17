"""
Intent optimisation tool — sweep prompt variants × models through the PRODUCTION
two-pass path (fsttm.two_pass.approach_a) and find the sweet spot for intent
accuracy / field accuracy / latency.

Unlike scripts/bench_models.py (single create_completion, intent-name only), this
exercises the real `approach_a` (eval → grammar JSON pass → rollback → TTS pass)
and scores BOTH the intent label AND the structured fields that production
actually translates (light_type, area, position, delta, temp). That surfaces the
real failures — e.g. "cabin lights" emitting LIGHTS_ON without light_type:"cabin".

Run on a GPU box (taycann):
    .venv/bin/python scripts/opt_intent.py                 # full sweep
    .venv/bin/python scripts/opt_intent.py --model Phi-3-mini-Q6
    .venv/bin/python scripts/opt_intent.py --prompt baseline,fewshot
    .venv/bin/python scripts/opt_intent.py --json-temp 0,0.1,0.3   # sweep pass-1 temp
    .venv/bin/python scripts/opt_intent.py --out results.json

The matrix is (model × prompt-variant × json-temp). Each cell runs the labelled
set through approach_a and reports intent%, field%, p50/p95 latency.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, ".")

# ── models available on taycann (path is skipped if missing) ──────────────────
MODELS = {
    "Phi-3-mini-Q4":   "models/Phi-3-mini-4k-instruct-Q4_K_M.gguf",
    "Phi-3-mini-Q6":   "models/Phi-3-mini-4k-instruct-Q6_K.gguf",
    "Phi-4-instr-Q6":  "models/Phi-4-mini-instruct-Q6_K.gguf",
    "Llama-3.2-3B-Q6": "models/Llama-3.2-3B-Instruct-Q6_K.gguf",
}

# ── labelled test set ─────────────────────────────────────────────────────────
# Each case: utterance, expected intent, and a dict of fields that MUST match
# (only the keys listed are checked — None means "don't care"). These fields are
# what intents.translate() consumes, so getting them right == correct car action.
CASES = [
    # climate
    ("make it warmer",                  "WARMER",          {}),
    ("it's too hot in here",            "COOLER",          {}),
    ("set temperature to 21 degrees",   "SET_TEMPERATURE", {"temp": 21}),
    ("set my temperature to 23",        "SET_TEMPERATURE", {"temp": 23, "area": 1}),
    ("raise the temperature by two",    "WARMER",          {"delta": 2}),
    ("turn on the air conditioning",    "AC_ON",           {}),
    ("defrost the windshield",          "VENT_DEFROST",    {}),
    ("fan speed up",                    "FAN_UP",          {}),
    # lights — the cabin-vs-headlight discrimination that was failing
    ("turn on the cabin lights",        "LIGHTS_ON",       {"light_type": "cabin"}),
    ("headlights on",                   "LIGHTS_ON",        {"light_type": "head"}),
    ("turn off the headlights",         "LIGHTS_OFF",       {"light_type": "head"}),
    ("switch on the interior light",    "LIGHTS_ON",        {"light_type": "cabin"}),
    # body — doors / windows / seats
    ("lock the doors",                  "DOOR_LOCK",       {}),
    ("unlock my door",                  "DOOR_UNLOCK",     {"area": 1}),
    ("open the driver window",          "WINDOW_OPEN",     {"area": 1}),
    ("close all windows",               "WINDOW_CLOSE",    {"area": 0}),
    ("warm my seat",                    "SEAT_HEAT_UP",    {"area": 1}),
    ("cool the passenger seat",         "SEAT_COOL_UP",    {"area": 4}),
    # meta — STATUS is a CAR-state query; TIME/DATE answered from the clock;
    # weather/jokes/chitchat are out-of-domain.
    ("what's my current temperature",   "STATUS",          {}),
    ("what time is it",                 "TIME",            {}),
    ("do you have the time",            "TIME",            {}),
    ("what's the date today",           "DATE",            {}),
    ("what day is it",                  "DATE",            {}),
    ("what's the weather like outside", "UNKNOWN",         {}),
    ("sing me a song",                  "UNKNOWN",         {}),
    ("tell me a joke",                  "UNKNOWN",         {}),
    ("call my wife",                    "UNKNOWN",         {}),
]


# ── prompt variants ───────────────────────────────────────────────────────────
# Each returns the system prompt string. They build on intents.build_prompt() so
# the grammar enum + zone table stay in sync; variants differ in the EXTRA
# guidance appended (few-shot examples, field-discipline reminders).
def _variant_prompt(name):
    # Build the prompt for one of the real config variants (intents.PROMPT_VARIANTS)
    # so the sweep measures exactly what ships.
    return lambda intents, domains: intents.build_prompt(domains, variant=name)


# Keys match config system.prompt_variant values so a winning cell maps straight
# to a config setting. "baseline" kept as an alias for one-shot.
PROMPTS = {
    "one-shot":       _variant_prompt("one-shot"),
    "few-shot":       _variant_prompt("few-shot"),
    "few-shot-extra": _variant_prompt("few-shot-extra"),
    "baseline":       _variant_prompt("one-shot"),
}


def _field_match(got_intent, expected_fields):
    """All expected fields present and equal in the produced JSON."""
    if not expected_fields:
        return True
    if not isinstance(got_intent, dict):
        return False
    for k, v in expected_fields.items():
        gv = got_intent.get(k)
        if isinstance(v, (int, float)) and isinstance(gv, (int, float)):
            if abs(float(gv) - float(v)) > 0.01:
                return False
        elif gv != v:
            return False
    return True


def _pct(n, d):
    return 100.0 * n / d if d else 0.0


def run_cell(model, sys_prompt, grammar, json_temp, two_pass):
    """Run the labelled set once. Returns a result dict."""
    intent_ok = field_ok = 0
    lats = []
    misses = []
    for utt, exp_intent, exp_fields in CASES:
        intent, tts, tj, tt = two_pass.approach_a(
            model, sys_prompt, utt, grammar, json_temp=json_temp)
        lats.append(tj)   # pass-1 latency is what intent accuracy costs
        got = intent.get("intent") if isinstance(intent, dict) else None
        i_ok = (got == exp_intent)
        f_ok = i_ok and _field_match(intent, exp_fields)
        intent_ok += i_ok
        field_ok += f_ok
        if not f_ok:
            misses.append((utt, exp_intent, exp_fields, intent))
    lats.sort()
    n = len(CASES)
    p50 = lats[len(lats) // 2]
    p95 = lats[min(len(lats) - 1, int(len(lats) * 0.95))]
    return {
        "intent_pct": _pct(intent_ok, n),
        "field_pct": _pct(field_ok, n),
        "intent_ok": intent_ok, "field_ok": field_ok, "n": n,
        "p50_ms": round(p50), "p95_ms": round(p95),
        "misses": misses,
    }


def main():
    ap = argparse.ArgumentParser(description="Intent optimisation sweep")
    ap.add_argument("--model", default=None,
                    help="comma list of model keys (default: all present)")
    ap.add_argument("--prompt", default=None,
                    help="comma list of prompt variants (default: all of "
                         "intents.PROMPT_VARIANTS)")
    ap.add_argument("--json-temp", default="0",
                    help="comma list of pass-1 temperatures to sweep")
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--domains", default=None,
                    help="comma list e.g. climate,lights,body (default all)")
    ap.add_argument("--config", default=None,
                    help="read intent_domains + model from a config.yaml "
                         "(so the sweep matches a deployment)")
    ap.add_argument("--out", default=None, help="write full results JSON here")
    args = ap.parse_args()

    from llama_cpp import Llama
    from fsttm_hvac import provider as intents, two_pass
    from fsttm_hvac.provider import build_grammar as make_hvac_grammar

    # --config: pull domains (and the gpt model path) straight from a deployment
    # config so the sweep measures exactly what ships there.
    cfg_domains = None
    if args.config:
        import yaml
        cfg = yaml.safe_load(open(args.config))
        cfg_domains = (cfg.get("system") or {}).get("intent_domains")
        cfg_model = (cfg.get("gpt") or {}).get("model")
        if cfg_model:
            MODELS["config"] = cfg_model
        print(f"=== config {args.config}: domains={cfg_domains} model={cfg_model}")

    domains = (args.domains.split(",") if args.domains else cfg_domains)
    grammar = make_hvac_grammar(domains)
    model_keys = args.model.split(",") if args.model else list(MODELS)
    # Default: sweep every real config variant (intents.PROMPT_VARIANTS).
    prompt_keys = args.prompt.split(",") if args.prompt else list(intents.PROMPT_VARIANTS)
    temps = [float(t) for t in args.json_temp.split(",")]

    rows = []
    for mk in model_keys:
        path = MODELS.get(mk)
        if not path or not os.path.exists(path):
            print(f"### {mk}: SKIP (missing {path})")
            continue
        print(f"\n### loading {mk} ({path})")
        t0 = time.monotonic()
        model = Llama(model_path=path, n_ctx=args.n_ctx, n_gpu_layers=-1,
                      verbose=False)
        print(f"    loaded in {time.monotonic()-t0:.1f}s")
        for pk in prompt_keys:
            sys_prompt = PROMPTS[pk](intents, domains)
            ptoks = len(model.tokenize(sys_prompt.encode(), add_bos=True, special=True))
            for jt in temps:
                r = run_cell(model, sys_prompt, grammar, jt, two_pass)
                r.update(model=mk, prompt=pk, json_temp=jt, prompt_toks=ptoks)
                rows.append(r)
                print(f"  [{mk} · {pk} · t={jt}] intent {r['intent_ok']}/{r['n']} "
                      f"({r['intent_pct']:.0f}%) | field {r['field_ok']}/{r['n']} "
                      f"({r['field_pct']:.0f}%) | p50 {r['p50_ms']}ms p95 {r['p95_ms']}ms "
                      f"| prompt {ptoks}tok")
                for utt, ei, ef, got in r["misses"]:
                    print(f"        MISS {utt!r}: exp {ei}{ef or ''} got {json.dumps(got)}")
        del model

    # ── ranking: field accuracy first (real action correctness), then intent,
    #    then latency. This is the "sweet point". ──
    rows.sort(key=lambda r: (-r["field_pct"], -r["intent_pct"], r["p50_ms"]))
    print("\n" + "=" * 72)
    print("RANKED (by field% → intent% → latency)")
    print("=" * 72)
    print(f"{'model':16s} {'prompt':9s} {'jt':4s} {'field%':>7s} "
          f"{'intent%':>8s} {'p50':>6s} {'p95':>6s} {'ptok':>6s}")
    for r in rows:
        print(f"{r['model']:16s} {r['prompt']:9s} {r['json_temp']:<4} "
              f"{r['field_pct']:6.0f}% {r['intent_pct']:7.0f}% "
              f"{r['p50_ms']:5d}m {r['p95_ms']:5d}m {r['prompt_toks']:6d}")
    if rows:
        best = rows[0]
        print(f"\nSWEET POINT: {best['model']} · prompt={best['prompt']} · "
              f"json_temp={best['json_temp']} → "
              f"field {best['field_pct']:.0f}% / intent {best['intent_pct']:.0f}% "
              f"@ p50 {best['p50_ms']}ms")

    if args.out:
        with open(args.out, "w") as f:
            json.dump([{k: v for k, v in r.items() if k != "misses"} for r in rows],
                      f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
