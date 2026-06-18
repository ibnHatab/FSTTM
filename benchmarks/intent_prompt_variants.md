# Intent prompt-variant comparison

Accuracy/latency of the three `system.prompt_variant` options, measured through
the production two-pass path (`scripts/opt_intent.py`) over a 27-case labelled
set (climate/lights/body + meta TIME/DATE/STATUS/UNKNOWN). Scores both the intent
label **and** the structured fields production translates (area/temp/delta/
light_type/position).

- **Host:** taycann (RTX PRO 2000 Blackwell, llama-cpp 0.3.23)
- **Model:** Phi-3-mini-4k Q4_K_M (the Jetson deployment model)
- **json_temp:** 0 (greedy)
- **Commit:** 7cbb35e
- **Raw data:** [`intent_variants_phi3q4.json`](intent_variants_phi3q4.json),
  [`intent_variants_phi3q4.out`](intent_variants_phi3q4.out)

| variant | intent % | field % | prompt tok | p50 ms | p95 ms |
|---------|---------:|--------:|-----------:|-------:|-------:|
| one-shot       | 96 % | **85 %** | 1382 | 292 | 390 |
| **few-shot** (default) | **100 %** | **100 %** | 1802 | 298 | 379 |
| few-shot-extra | 100 % | 100 % | 2026 | 384 | 468 |

## Points gained

- **few-shot vs one-shot: +15 field points, +4 intent points** for +420 prompt
  tokens and ~+6 ms p50 (negligible). This is the production default.
- **few-shot-extra vs few-shot: +0 points** — the second tier of examples adds
  no accuracy on this set but costs +224 tokens and ~+86 ms p50. Not worth it.

## one-shot misses (all fixed by few-shot)

All four one-shot failures are missing/empty fields, not wrong intents:

- `"set my temperature to 23"` → area 0 (should be 1, possessive "my")
- `"turn on the air conditioning"` → SET_TEMPERATURE (should be AC_ON)
- `"unlock my door"` → area 0 (should be 1)
- `"warm my seat"` → area 0 (should be 1)

The few-shot examples teach possessive→area and the AC mapping, recovering all
four. See `memory: intent-optimization` and `fsttm/intents/__init__.py:_FEWSHOT`.

## Takeaway

**few-shot is the sweet spot** — full accuracy at near-one-shot latency. The
prompt size no longer affects per-turn latency on the Jetson (prompt eval is a
few hundred ms after the save_state/load_state regression was removed in 7cbb35e),
so there's no speed reason to prefer one-shot. Keep `prompt_variant: "few-shot"`.

_Regenerate:_ `scripts/opt_intent.py --model Phi-3-mini-Q4 --domains climate,lights,body --out benchmarks/intent_variants_phi3q4.json`
