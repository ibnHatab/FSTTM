"""
Two-pass constrained intent generation. ONE prompt eval; both passes share the
KV cache by CONTINUING it (no save_state/load_state — that path serialises the
whole KV cache through Python and costs ~5s/call on the Jetson Xavier).

  1. eval(base_prompt_tokens)           → fills KV cache with system+user prompt
  2. generate([], grammar=json_grammar) → JSON tokens (appended to KV)
  3. eval("\\nSpoken response:")         → appended after the JSON in context
  4. generate([], grammar=None)          → TTS text tokens

Returns (intent_dict, tts_text, t_json_ms, t_tts_ms).
"""
from __future__ import annotations  # PEP 563: lazy annotations for Python 3.8 (list[]/tuple[]/set[])
import json
import time
import logging
from typing import Optional

log = logging.getLogger(__name__)

# ── helpers ──────────────────────────────────────────────────────────────────

def _safe_json(text: str) -> Optional[dict]:
    t = text.strip()
    brace = t.rfind("}")
    if brace != -1:
        t = t[: brace + 1]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        log.warning("JSON parse failed: %r", t)
        return None


def _decode(model, token_ids: list[int]) -> str:
    return model.detokenize(token_ids).decode("utf-8", errors="replace")


def _eos_tokens(model) -> set:
    s = set()
    for attr in ("token_eos", "token_bos"):
        try:
            t = getattr(model, attr)()
            if t >= 0:
                s.add(t)
        except Exception:
            pass
    return s


def _phi3_sys_prefix(sys: str) -> str:
    return f"<|system|>\n{sys}<|end|>\n"


def _phi3_user_tail(user_text: str) -> str:
    return f"<|user|>\n{user_text}<|end|>\n<|assistant|>\n"


def _phi3_base_prompt(sys: str, user_text: str) -> str:
    return _phi3_sys_prefix(sys) + _phi3_user_tail(user_text)


# ── KV prefix reuse ──────────────────────────────────────────────────────────
# The intent system prompt (~2000 tokens of domain tables + few-shot) is IDENTICAL
# every utterance, but evaluating it costs ~5s on the Jetson Xavier (~400 tok/s) —
# that 5s of silent dead air IS the intent-mode "lag" (chat's ~30-token prompt
# evals in ~170ms, so chat feels instant). So we eval the system prefix ONCE,
# keep it in the KV cache, and per turn drop only the tokens after it and eval the
# short user tail (~37 tok, ~150ms). 33x faster, byte-identical output (verified).
#
# Requires the low-level KV ops (llama-cpp >=0.2.x): _ctx.kv_cache_seq_rm + a
# settable .n_tokens. _prime_prefix() returns the cached n_prefix or None when the
# ops are unavailable, in which case approach_a falls back to a full eval.
_PREFIX = {"key": None, "n": 0, "ok": True}


def _kv_supported(model) -> bool:
    return (hasattr(getattr(model, "_ctx", None), "kv_cache_seq_rm")
            and hasattr(model, "n_tokens"))


def _prime_prefix(model, system_prompt):
    """Ensure the system prefix is evaluated and cached in the KV. Returns
    n_prefix (token count to keep) or None if reuse isn't possible."""
    if not _PREFIX["ok"] or not _kv_supported(model):
        _PREFIX["ok"] = False
        return None
    key = (id(model), system_prompt)
    if _PREFIX["key"] == key and model.n_tokens >= _PREFIX["n"]:
        return _PREFIX["n"]
    # (Re)prime: eval just the prefix into a fresh context.
    try:
        ptoks = model.tokenize(_phi3_sys_prefix(system_prompt).encode(),
                               add_bos=True, special=True)
        model.reset()
        model.eval(ptoks)
        _PREFIX["key"] = key
        _PREFIX["n"] = len(ptoks)
        return _PREFIX["n"]
    except Exception:
        _PREFIX["ok"] = False
        return None


def _phi3_stop_ids(model) -> set[int]:
    # Only register stops whose FIRST token is a genuine, single-token marker.
    # SentencePiece prefixes many strings with the "▁" space metasymbol (id
    # 29871 on Phi-3), which is its own token — e.g. tokenize("\n\n") = [▁,\n,\n]
    # and tokenize("0") = [▁,0]. Registering ▁ as a stop truncates JSON right
    # after `"area":` (where the integer's leading ▁ comes next). So skip any
    # marker that tokenizes to a leading ▁, and skip plain whitespace markers.
    SPACE_PREFIX = None
    probe = model.tokenize(" x".encode(), add_bos=False, special=True)
    if probe:
        SPACE_PREFIX = probe[0]   # the ▁ id on this tokenizer
    stops = set()
    for s in ["<|end|>", "<|user|>"]:
        toks = model.tokenize(s.encode(), add_bos=False, special=True)
        if toks and toks[0] != SPACE_PREFIX:
            stops.add(toks[0])
    return stops


# ── approach_a: eval once, continue KV cache across both passes ──────────────

def approach_a(model, system_prompt: str, user_text: str,
               grammar, json_temp: float = 0.0) -> tuple[Optional[dict], str, float, float]:
    """
    Two-pass intent generation with KV PREFIX REUSE.
    - The constant system prefix is evaluated ONCE and kept in the KV cache; each
      turn drops only the tokens after it and evals the short user tail (~150ms vs
      ~5s full eval on the Jetson). Both passes then share the KV by continuing it
      (no save_state/load_state — that serialises the cache and is the slow path).

    json_temp: pass-1 (intent JSON) sampling temperature. Default 0.0 = greedy
    (top_k=1), which production uses. The optimisation tool (scripts/opt_intent.py)
    sweeps this; >0 opens top_k so the temperature actually has an effect.
    """
    eos      = _eos_tokens(model)
    stop_ids = _phi3_stop_ids(model)

    _t_eval = time.monotonic()
    # Was the prefix already warm in the KV BEFORE this call? If not, _prime_prefix
    # re-evaluates it now (the slow ~11s path) — meaning the startup pre-warm got
    # wiped by some other model op (chitchat/RAG create_completion resets the KV).
    _warm_before = (_PREFIX["key"] == (id(model), system_prompt)
                    and getattr(model, "n_tokens", 0) >= _PREFIX["n"] > 0)
    _t_prime = time.monotonic()
    n_prefix = _prime_prefix(model, system_prompt)
    t_prime = (time.monotonic() - _t_prime) * 1000
    if n_prefix is not None:
        # Reuse the cached prefix: drop everything after it, rewind position, and
        # eval ONLY the per-turn user tail.
        model._ctx.kv_cache_seq_rm(0, n_prefix, -1)
        model.n_tokens = n_prefix
        tail = model.tokenize(_phi3_user_tail(user_text).encode(),
                              add_bos=False, special=True)
        model.eval(tail)
        _reused = True
    else:
        # Fallback (KV ops unavailable): full eval every turn.
        base_tokens = model.tokenize(
            _phi3_base_prompt(system_prompt, user_text).encode(),
            add_bos=True, special=True)
        model.reset()
        model.eval(base_tokens)
        _reused = False
    t_eval = (time.monotonic() - _t_eval) * 1000

    # Pass 1: grammar-constrained JSON. temp=0 → greedy (top_k=1); a non-zero
    # sweep temperature opens top_k so sampling is actually exercised.
    _jtop_k = 1 if json_temp <= 0.0 else 40
    t0 = time.monotonic()
    json_ids = []
    for tok_id in model.generate([], reset=False, grammar=grammar, temp=json_temp, top_k=_jtop_k):
        json_ids.append(tok_id)
        if tok_id in eos or tok_id in stop_ids or len(json_ids) >= 80:
            break
    t_json = (time.monotonic() - t0) * 1000
    # warm=True → prefix was hot, eval is the cheap tail-only path. warm=False →
    # prime had to re-eval the whole prefix (t_prime is that cost); the startup
    # pre-warm was wiped by an intervening model op.
    log.info("approach_a timing: eval=%.0fms (warm=%s prime=%.0fms tail=%.0fms) "
             "json_gen=%.0fms (%d tok) → %.1fms/tok",
             t_eval, _warm_before, t_prime, t_eval - t_prime, t_json,
             len(json_ids), t_json / max(len(json_ids), 1))
    json_text = _decode(model, json_ids).strip().rstrip("<|end|>").strip()
    intent = _safe_json(json_text)

    # Pass 2: TTS continuation — DON'T roll back the KV cache (load_state is the
    # slow path). The JSON we just generated stays in context; append the cue and
    # keep generating. The model sees "…<json>\nSpoken response:" which is fine
    # for producing the spoken ack.
    cue_tokens = model.tokenize(
        f"\nSpoken response:".encode(), add_bos=False, special=True
    )
    model.eval(cue_tokens)

    t0 = time.monotonic()
    tts_ids = []
    for tok_id in model.generate([], reset=False, grammar=None, temp=0.4, top_k=40):
        tts_ids.append(tok_id)
        text_so_far = _decode(model, tts_ids)
        if tok_id in eos or tok_id in stop_ids or "\n" in text_so_far or len(tts_ids) >= 30:
            break
    t_tts = (time.monotonic() - t0) * 1000
    tts_text = _decode(model, tts_ids).strip().split("\n")[0].rstrip("<|end|>").strip()

    return intent, tts_text, t_json, t_tts


# ── benchmark ─────────────────────────────────────────────────────────────────

def benchmark(model, system_prompt: str, test_cases: list, grammar) -> None:
    print(f"\n{'Input':<40} {'JSON':>7} {'TTS':>7} {'Total':>8}  Intent")
    print("─" * 75)

    for utterance in test_cases:
        intent, tts, tj, tt = approach_a(model, system_prompt, utterance, grammar)
        got = (intent or {}).get("intent", "ERR")
        print(f"{utterance:<40} {tj:>6.0f}ms {tt:>6.0f}ms {tj+tt:>7.0f}ms  {got}")
        print(f"  voice → {tts!r}")
        print()
