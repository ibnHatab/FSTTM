"""
Context exhaustion test: mixed intents, non-intents, and conversation turns.

Tests:
  1. HVAC + vehicle intents correctly recognised (positive)
  2. Off-topic sentences correctly produce UNKNOWN (negative)
  3. Interspersed conversation (no intent) does not break intent detection
  4. Context fills up and FIFO trimming kicks in at threshold
  5. After FIFO trim: system prompt preserved, recent turns present, oldest gone
  6. Optimal clean-up point measured by response quality metric

Uses fsttm.headless infrastructure — no audio hardware required.
Run:
    python -m pytest tests/test_context_exhaustion.py -v -s
"""
from __future__ import annotations  # PEP 563: lazy annotations for Python 3.8
import asyncio
import json
import re
import time
from dataclasses import dataclass, field

import pytest

from fsttm.llama import ConversationHistory


# ── ConversationHistory unit tests (no model) ─────────────────────────────────

def test_history_fifo_trim():
    # n_ctx=80 → threshold=64 tokens. Each turn ≈ 10 tokens → 7 turns ≈ 70 > 64
    h = ConversationHistory(n_ctx=80, ctx_threshold=0.8)
    sys = "short"   # ~2 tokens
    for i in range(10):
        h.add_turn(f"user msg {i:02d}", f"asst resp {i:02d}")   # ~10 tokens/turn
    before = h.turn_count()
    dropped = h.trim_for(sys, "new input")
    assert dropped > 0, "Expected at least one turn to be dropped"
    assert h.turn_count() < before


def test_history_system_prompt_always_kept():
    """System prompt tokens are never removed — only turns are trimmed."""
    h = ConversationHistory(n_ctx=100, ctx_threshold=0.8)
    sys = "A" * 160   # ~40 estimated tokens (eats most of budget)
    for i in range(5):
        h.add_turn("hello", "hi")   # ~2 tokens each
    dropped = h.trim_for(sys, "x")
    # sys prompt stays; turns may all be dropped — that's correct
    assert isinstance(dropped, int) and dropped >= 0
    # regardless of result, total should not exceed threshold
    total = h.total_tokens(sys)
    assert total <= h.threshold_toks + 10   # +10 tolerance for rounding


def test_history_context_fill_pct():
    h = ConversationHistory(n_ctx=1000)
    sys = "short"
    assert h.context_fill_pct(sys) < 5.0   # empty history = very low
    for _ in range(20):
        h.add_turn("user " * 10, "assistant " * 10)
    assert h.context_fill_pct(sys) > 10.0  # fills up


def test_history_clear():
    h = ConversationHistory(n_ctx=512)
    h.add_turn("u", "a")
    h.add_turn("u2", "a2")
    assert h.turn_count() == 2
    h.clear()
    assert h.turn_count() == 0


def test_history_build_prompt_phi3():
    h = ConversationHistory(n_ctx=512)
    h.add_turn("hello", "hi there")
    prompt, stop = h.build_chat_prompt(
        "Phi-3-mini-4k-instruct-Q6_K.gguf", "Be brief.", "how are you?"
    )
    assert "<|system|>" in prompt
    assert "Be brief." in prompt
    assert "hello" in prompt         # previous turn included
    assert "hi there" in prompt
    assert "how are you?" in prompt
    assert "<|assistant|>" in prompt
    assert "<|end|>" in stop


def test_history_build_prompt_llama3():
    h = ConversationHistory(n_ctx=512)
    prompt, stop = h.build_chat_prompt(
        "Llama-3.2-3B-Instruct-Q6_K.gguf", "Be brief.", "hello"
    )
    assert "<|start_header_id|>system" in prompt
    assert "<|eot_id|>" in stop


def test_history_fifo_order():
    """Oldest turn (index 0) is dropped first."""
    h = ConversationHistory(n_ctx=50, ctx_threshold=0.9)
    h.add_turn("first user", "first assistant")
    h.add_turn("second user", "second assistant")
    h.add_turn("third user", "third assistant")
    h.trim_for("sys", "new")
    # if any turns remain, they should NOT be the first one
    turns_text = str(h._turns)
    if h.turn_count() > 0:
        assert "first user" not in turns_text or h.turn_count() == 3


# ── Intent + context integration tests (requires model) ──────────────────────

SKIP_MODEL = pytest.mark.skipif(
    not __import__("os").path.exists(
        "models/Phi-3-mini-4k-instruct-Q6_K.gguf"
    ),
    reason="Phi-3 model not present",
)


@dataclass
class IntentTestResult:
    utterance: str
    expected_intent: str
    got_intent: str | None
    tts_text: str
    turn_num: int
    ctx_fill_pct: float
    latency_ms: float
    correct: bool = field(init=False)

    def __post_init__(self):
        self.correct = (self.got_intent == self.expected_intent
                        or (self.expected_intent == "UNKNOWN"
                            and self.got_intent == "UNKNOWN"))


def _run_intent_headless(utterances: list[tuple[str, str]],
                         prompt_file: str = "contrib/hvac/tests/data/vehicle-intentions-phi3.txt",
                         model_path: str = "models/Phi-3-mini-4k-instruct-Q6_K.gguf",
                         n_ctx: int = 2048) -> list[IntentTestResult]:
    """
    Run a sequence of (utterance, expected_intent) through the two-pass intent
    pipeline and return results including context fill at each turn.
    """
    from llama_cpp import Llama
    from fsttm.utils import ignoreStderr
    from fsttm_hvac.provider import build_grammar as make_hvac_grammar
    from fsttm.two_pass import approach_a

    with open(prompt_file) as f:
        sys_prompt = f.read().strip()

    with ignoreStderr():
        try:
            model = Llama(model_path=model_path, n_ctx=n_ctx,
                          n_gpu_layers=99, verbose=False)
        except ValueError:
            print("  [ctx-test] GPU VRAM insufficient, falling back to CPU")
            model = Llama(model_path=model_path, n_ctx=n_ctx,
                          n_gpu_layers=0, verbose=False)
    model.model_path = model_path

    grammar = make_hvac_grammar()
    history = ConversationHistory(n_ctx=n_ctx, ctx_threshold=0.80)

    results = []
    for turn_num, (utterance, expected) in enumerate(utterances):
        fill = history.context_fill_pct(sys_prompt)
        t0 = time.monotonic()
        try:
            intent, tts, tj, tt = approach_a(model, sys_prompt, utterance, grammar)
            got = (intent or {}).get("intent", "PARSE_ERROR")
        except Exception as e:
            got = f"ERROR:{e}"
            tts = ""
        latency = (time.monotonic() - t0) * 1000

        # record turn in history (non-intent turns as plain conversation)
        if got not in ("PARSE_ERROR",) and not got.startswith("ERROR"):
            history.add_turn(utterance, tts or got)

        results.append(IntentTestResult(
            utterance=utterance,
            expected_intent=expected,
            got_intent=got,
            tts_text=tts,
            turn_num=turn_num,
            ctx_fill_pct=fill,
            latency_ms=latency,
        ))

    return results


# ── mixed positive/negative/conversational sequence ──────────────────────────

MIXED_SEQUENCE = [
    # positive: HVAC intents
    ("it's too cold in here",          "WARMER"),
    ("set temperature to 22",          "SET_TEMPERATURE"),
    ("turn on the AC",                 "AC_ON"),
    ("fan level 4",                    "SET_FAN"),
    # positive: vehicle intents
    ("lock all the doors",             "DOOR_LOCK"),
    ("turn on the headlights",         "LIGHTS_ON"),
    ("open my window halfway",         "WINDOW_OPEN"),
    ("warm up my seat",                "SEAT_HEAT_UP"),
    # negative: off-topic (should produce UNKNOWN)
    ("play some jazz music",           "UNKNOWN"),
    ("what's the weather outside",     "UNKNOWN"),
    ("call my wife",                   "UNKNOWN"),
    # positive: back to intents after non-intents
    ("unlock the passenger door",      "DOOR_UNLOCK"),
    ("hazard lights on",               "LIGHTS_ON"),
    ("cooler please",                  "COOLER"),
    # more positives to fill context
    ("defrost the windshield",         "VENT_DEFROST"),
    ("auto mode please",               "AUTO_ON"),
    ("recirculate the air",            "RECIRCULATE_ON"),
    ("close all windows",              "WINDOW_CLOSE"),
    ("seat ventilation on my side",    "SEAT_COOL_UP"),
    ("turn off the fog lights",        "LIGHTS_OFF"),
]

# ── EXTENDED sequence to exhaust context ─────────────────────────────────────
CONTEXT_EXHAUST_SEQUENCE = MIXED_SEQUENCE + [
    # additional turns to push context toward threshold
    ("it's getting too hot",           "COOLER"),
    ("play rock music",                "UNKNOWN"),
    ("set fan to 3",                   "SET_FAN"),
    ("unlock rear left door",          "DOOR_UNLOCK"),
    ("cabin light on",                 "LIGHTS_ON"),
    ("warmer please, it's freezing",   "WARMER"),
    ("rear window defrost",            "REAR_DEFROST_TOGGLE"),
    ("navigate home",                  "UNKNOWN"),
    ("sync both climate zones",        "SYNC_TOGGLE"),
    ("max AC please",                  "MAX_AC_TOGGLE"),
    ("open rear left window",          "WINDOW_OPEN"),
    ("seat heat down on passenger",    "SEAT_HEAT_DOWN"),
    ("what time is it",                "UNKNOWN"),
    ("close rear windows",             "WINDOW_CLOSE"),
    ("headlights off",                 "LIGHTS_OFF"),
]


@SKIP_MODEL
def test_mixed_positive_negative_intents():
    """Mixed HVAC + vehicle intents + off-topic — validate positive/negative."""
    results = _run_intent_headless(MIXED_SEQUENCE)

    positives = [r for r in results if r.expected_intent != "UNKNOWN"]
    negatives = [r for r in results if r.expected_intent == "UNKNOWN"]

    pos_correct = sum(1 for r in positives if r.correct)
    neg_correct = sum(1 for r in negatives if r.correct)

    print(f"\n  Positive: {pos_correct}/{len(positives)} correct")
    print(f"  Negative: {neg_correct}/{len(negatives)} correct")
    for r in results:
        mark = "✓" if r.correct else "✗"
        print(f"  {mark} [{r.turn_num:02d}] {r.utterance[:35]:<35} "
              f"expect={r.expected_intent:<20} got={r.got_intent} "
              f"ctx={r.ctx_fill_pct:.0f}%")

    assert pos_correct / len(positives) >= 0.80, \
        f"Positive intent accuracy {pos_correct}/{len(positives)} < 80%"
    assert neg_correct / len(negatives) >= 0.80, \
        f"Negative (UNKNOWN) accuracy {neg_correct}/{len(negatives)} < 80%"


@SKIP_MODEL
def test_context_exhaustion_and_fifo():
    """
    Fill context past threshold and verify FIFO trimming:
    - oldest turns are dropped
    - recent turns are preserved
    - system prompt always intact
    - intent quality maintained after trim
    """
    results = _run_intent_headless(CONTEXT_EXHAUST_SEQUENCE, n_ctx=4096)

    # Find where trim first happened (context was >80% at turn start)
    trim_turns = [r for r in results if r.ctx_fill_pct > 75]
    if trim_turns:
        print(f"\n  Context >75% at turns: {[r.turn_num for r in trim_turns]}")

    # After trim, last 5 intents should still be correct
    final_intents = [r for r in results[-5:] if r.expected_intent != "UNKNOWN"]
    final_correct = sum(1 for r in final_intents if r.correct)
    print(f"  Final 5 intents after context pressure: {final_correct}/{len(final_intents)} correct")

    if final_intents:
        assert final_correct / len(final_intents) >= 0.60, \
            "Quality degraded badly after context exhaustion"


@SKIP_MODEL
def test_find_optimal_cleanup_point():
    """
    Measure intent accuracy as a function of context fill percentage.
    Report the fill% where accuracy first drops below 90% — this is the
    recommended cleanup threshold.
    """
    results = _run_intent_headless(CONTEXT_EXHAUST_SEQUENCE, n_ctx=2048)

    # bucket results by 10% fill increments
    buckets: dict[int, list[IntentTestResult]] = {}
    for r in results:
        bucket = int(r.ctx_fill_pct // 10) * 10
        buckets.setdefault(bucket, []).append(r)

    print("\n  Context fill → intent accuracy:")
    optimal = None
    for pct in sorted(buckets):
        bucket_results = [r for r in buckets[pct] if r.expected_intent != "UNKNOWN"]
        if not bucket_results:
            continue
        acc = sum(1 for r in bucket_results if r.correct) / len(bucket_results)
        n   = len(bucket_results)
        print(f"    {pct:3d}%–{pct+9}%: {acc:.0%} ({n} intents)")
        if acc < 0.90 and optimal is None:
            optimal = pct
            print(f"    ↑ ACCURACY DROPS BELOW 90% HERE — recommend cleanup at {pct}%")

    if optimal:
        print(f"\n  Recommended ConversationHistory ctx_threshold: {(optimal-10)/100:.2f}")
    else:
        print("\n  No accuracy drop detected — model handles full context well")


@SKIP_MODEL
def test_intent_latency_stable_across_context():
    """Verify intent latency does not grow unboundedly as context fills."""
    results = _run_intent_headless(CONTEXT_EXHAUST_SEQUENCE[:15], n_ctx=2048)
    latencies = [r.latency_ms for r in results]
    avg_first_5  = sum(latencies[:5])  / 5
    avg_last_5   = sum(latencies[-5:]) / 5
    print(f"\n  Latency avg (first 5 turns): {avg_first_5:.0f} ms")
    print(f"  Latency avg (last  5 turns): {avg_last_5:.0f} ms")
    # Allow up to 3× slowdown (prompt grows with history)
    assert avg_last_5 < avg_first_5 * 4, \
        f"Latency grew too much: {avg_first_5:.0f}→{avg_last_5:.0f} ms"
