"""
Checkpoint narrator headless test.

Two layers, both runnable without audio hardware:

  1. test_split_checkpoints_*  — pure unit tests of the clause splitter, no
     model, no native deps. These are the Stage-2 guard: they run on the host
     (and on the Jetson under Py3.8) to catch regressions in checkpoint
     splitting.

  2. test_checkpoint_dialog    — full scripted dialog through the headless LLM
     pipeline, exercising interrupt + resume (Stage 3). Skipped unless the
     Phi-3 model and llama_cpp are present.

Run:
    pytest tests/test_checkpoint_headless.py -v -s
"""
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import pytest

MODEL_PATH  = "models/Phi-3-mini-4k-instruct-Q6_K.gguf"
PROMPT_FILE = None   # default conversational prompt


# ── sentence splitter (same as server.py _split_checkpoints) ──────────────────

def split_checkpoints(text):
    """Clause-level TTS units — mirrors server.py _split_checkpoints."""
    sentence_re = re.compile(r'(?<=[^0-9][.!?])\s+')
    clause_re   = re.compile(r'(?<=\S),\s+|(?<=\S);\s+|(?<= —)\s+|(?<= –)\s+')
    parts = sentence_re.split(text.strip())
    fine = []
    for part in parts:
        clauses = clause_re.split(part)
        fine.extend(c.strip() for c in clauses if c.strip())
    out, buf = [], ""
    for p in fine:
        buf = (buf + ", " + p).strip() if buf else p
        if len(buf) >= 20:
            out.append(buf); buf = ""
    if buf:
        if out: out[-1] = out[-1].rstrip(",") + " " + buf
        else:   out.append(buf)
    return [s.strip() for s in out if s.strip() and len(s.strip()) > 3]


# ── pure splitter unit tests (no model — the Stage-2 headless guard) ──────────

def test_split_checkpoints_empty():
    assert split_checkpoints("") == []
    assert split_checkpoints("   ") == []


def test_split_checkpoints_single_short_sentence():
    # One short sentence stays as a single unit.
    assert split_checkpoints("The sky is blue today.") == ["The sky is blue today."]


def test_split_checkpoints_multiple_sentences():
    text = ("Rainbows form when sunlight hits water droplets. "
            "The light bends and splits into colours. "
            "Each colour travels at a slightly different angle.")
    ckpts = split_checkpoints(text)
    assert len(ckpts) == 3
    assert ckpts[0].startswith("Rainbows form")
    assert ckpts[-1].endswith("angle.")


def test_split_checkpoints_long_response_splits():
    # A long, clause-heavy response must split into several units.
    text = ("Clouds form when warm air rises and cools, the water vapour "
            "condenses into tiny droplets, and those droplets cluster together "
            "around dust particles until a visible cloud appears.")
    ckpts = split_checkpoints(text)
    assert len(ckpts) >= 2, ckpts


def test_split_checkpoints_no_split_on_numbered_list():
    # "1." and "2." are list markers, not sentence ends — must not split there.
    text = "First do 1. then move to 2. and finish at the end of the line here."
    ckpts = split_checkpoints(text)
    joined = " ".join(ckpts)
    assert "1." in joined and "2." in joined
    # No unit should be a bare "1" / "2" fragment.
    assert all(c.strip() not in {"1", "2", "1.", "2."} for c in ckpts)


def test_split_checkpoints_no_split_on_decimal():
    # Decimal point must not create a checkpoint boundary.
    text = "The speed of light is about 299792.458 kilometres per second exactly."
    ckpts = split_checkpoints(text)
    assert len(ckpts) == 1
    assert "299792.458" in ckpts[0]


def test_split_checkpoints_merges_short_fragments():
    # Short trailing fragment is merged, not emitted on its own.
    text = "The water cycle moves moisture through the atmosphere continuously. Yes."
    ckpts = split_checkpoints(text)
    assert all(len(c) > 3 for c in ckpts)
    # The 4-char "Yes." fragment must not survive as its own unit.
    assert "Yes." not in ckpts


# ── barge-in resume / skip / trim index math (mirrors server.py, no model) ────

REPLAY_THRESHOLD = 0.50


def _barge_in(interrupted, fraction, ckpts_done):
    """Mirror server.py _on_barge_in_narrator's decision math.

    Returns (resume_at, heard_indices). resume_at is stored verbatim in
    _ckpt_interrupted and consumed as-is by _do_resume (no +1)."""
    resume_at = interrupted if fraction < REPLAY_THRESHOLD else interrupted + 1
    # Only units confirmed done strictly before the interrupted one count as heard.
    confirmed = sorted(i for i in ckpts_done if i < interrupted)
    return resume_at, confirmed


def test_barge_in_replay_when_early():
    # Heard <50% of the interrupted unit → replay it (resume AT interrupted).
    resume_at, heard = _barge_in(interrupted=3, fraction=0.30,
                                 ckpts_done={0, 1, 2})
    assert resume_at == 3
    assert heard == [0, 1, 2]


def test_barge_in_skip_when_late():
    # Heard >=50% → skip the interrupted unit (resume AFTER it).
    resume_at, heard = _barge_in(interrupted=3, fraction=0.80,
                                 ckpts_done={0, 1, 2})
    assert resume_at == 4
    assert heard == [0, 1, 2]


def test_barge_in_excludes_interrupted_from_heard():
    # Even if the interrupted unit's done-event slipped into the set, it was
    # cut off and must NOT be counted as heard.
    resume_at, heard = _barge_in(interrupted=2, fraction=0.30,
                                 ckpts_done={0, 1, 2})
    assert heard == [0, 1]           # index 2 excluded
    assert resume_at == 2


def test_do_resume_index_no_off_by_one():
    # _do_resume consumes _ckpt_interrupted verbatim — replaying the stored
    # index, not stored+1. Replay case must re-speak the interrupted unit.
    ckpts = ["unit zero here ok", "unit one here ok", "unit two here ok",
             "unit three here ok"]
    resume_at, _ = _barge_in(interrupted=2, fraction=0.10, ckpts_done={0, 1})
    replayed = ckpts[resume_at:]
    assert replayed[0] == "unit two here ok"   # the interrupted unit replays


def test_split_checkpoints_clause_separators():
    # Each clause here is already >=20 chars, so the merge step keeps them
    # separate instead of folding short fragments together.
    text = ("The warm air rises high into the sky, "
            "the water vapour cools and condenses, "
            "and the heavy droplets finally fall as rain.")
    ckpts = split_checkpoints(text)
    # Comma clauses split into more than one unit.
    assert len(ckpts) >= 2, ckpts


# ── dialog recorder (Stage 3: interrupt + resume; needs the model) ────────────

@dataclass
class Turn:
    role:        str    # 'user' | 'system'
    text:        str
    checkpoints: List[str] = field(default_factory=list)
    interrupted: bool  = False
    resumed:     bool  = False
    resume_from: int   = -1


def _run_dialog(utterances):
    """
    Feed `utterances` into the headless LLM pipeline.
    Returns turns.
    """
    from llama_cpp import Llama
    from fsttm.utils import ignoreStderr
    from fsttm.llama import ConversationHistory

    with ignoreStderr():
        model = Llama(model_path=MODEL_PATH, n_ctx=4096,
                      n_gpu_layers=99, verbose=False)
    model.model_path = MODEL_PATH

    sys_prompt = ("You are a concise but enthusiastic voice assistant. "
                  "Give detailed answers of 3-5 sentences. "
                  "No lists, no markdown.")

    history  = ConversationHistory(n_ctx=4096)
    turns    = []
    RESUME   = {'continue','go on','please continue','keep going',
                'go ahead','carry on','and then','what else'}

    # Checkpoint narrator state
    current_ckpts  = []
    interrupted_at = -1    # -1=none; >=0 = resume-from index (set by barge-in)

    for utterance in utterances:
        u_lower = utterance.lower().strip().rstrip('.!?,')

        # ── resume path ──────────────────────────────────────────────────────
        if u_lower in RESUME and interrupted_at >= 0:
            resume_from = interrupted_at   # already points to correct unit
            remaining   = current_ckpts[resume_from:]
            resumed_text = " ".join(remaining)
            turns.append(Turn(
                role='user', text=utterance,
                resumed=True, resume_from=resume_from,
            ))
            turns.append(Turn(
                role='system', text=resumed_text,
                checkpoints=remaining,
                resumed=True, resume_from=resume_from,
            ))
            # Add to history as if system said the remaining text
            history.add_turn(utterance, resumed_text)
            interrupted_at = -1
            print("\n  USER: {!r}".format(utterance))
            print("  [resume from ckpt {}/{}]".format(resume_from, len(current_ckpts)-1))
            print("  SYS (resumed): {!r}".format(resumed_text[:100]))
            continue

        # ── normal LLM generation ────────────────────────────────────────────
        prompt, stop = history.build_chat_prompt(MODEL_PATH, sys_prompt, utterance)
        t0 = time.monotonic()
        r = model.create_completion(prompt, max_tokens=200, temperature=0.7,
                                    stream=False, stop=stop)
        response = r["choices"][0]["text"].strip()
        latency  = (time.monotonic() - t0) * 1000

        ckpts = split_checkpoints(response)
        history.add_turn(utterance, response)
        current_ckpts = ckpts

        # Simulate barge-in: interrupt utterances ending with "(interrupt)"
        interrupted = utterance.rstrip().endswith('(interrupt)')
        clean_utt   = utterance.replace('(interrupt)', '').strip()

        if interrupted:
            # Simulate: interrupt at checkpoint 2 (third unit) at 30% through it
            interrupted_at   = min(2, len(ckpts) - 1)
            fraction_heard   = 0.30   # early in the unit → replay
            REPLAY_THRESHOLD = 0.50

            # Simulate _ckpts_done: units 0..interrupted_at-1 were PlaybackDone
            ckpts_done = set(range(interrupted_at))
            # Same logic as server.py _on_barge_in_narrator:
            resume_at   = interrupted_at if fraction_heard < REPLAY_THRESHOLD \
                          else interrupted_at + 1
            # Exact history from confirmed-done units only
            confirmed   = sorted(i for i in ckpts_done if i < interrupted_at)
            heard        = " ".join(ckpts[i] for i in confirmed)

            history._turns.pop()
            history.add_turn(clean_utt, heard if heard else "...")
            interrupted_at = resume_at   # store resume-from index
        else:
            interrupted_at = -1

        turns.append(Turn(role='user', text=clean_utt))
        turns.append(Turn(
            role='system', text=response,
            checkpoints=ckpts,
            interrupted=interrupted,
        ))

        print("\n  USER: {!r}".format(clean_utt))
        print("  SYS ({:.0f}ms, {} ckpts):".format(latency, len(ckpts)), end="")
        for i, c in enumerate(ckpts):
            marker = "[{}*]".format(i) if interrupted and i == interrupted_at else "[{}]".format(i)
            print("\n    {} {!r}".format(marker, c[:70]), end="")
        print()
        if interrupted:
            print("  *** INTERRUPTED at checkpoint {} ***".format(interrupted_at))

    return turns


def _assess(turns):
    """Evaluate the dialog quality."""
    total_sys    = [t for t in turns if t.role == 'system']
    resumptions  = [t for t in turns if t.role == 'system' and t.resumed]
    interrupted  = [t for t in turns if t.role == 'system' and t.interrupted]

    issues = []

    # Check: every interrupted system turn is followed by a resumption
    for i, t in enumerate(turns):
        if t.role == 'system' and t.interrupted:
            # next system turn should be a resumption
            next_sys = next(
                (x for x in turns[i+1:] if x.role == 'system'), None
            )
            if next_sys and not next_sys.resumed:
                issues.append(
                    "System interrupted but next response was not a resumption: "
                    "{!r}".format(next_sys.text[:60])
                )

    # Check: resume starts AFTER the interrupted checkpoint
    for t in resumptions:
        if t.resume_from <= 0 and t.resumed:
            issues.append("Resume started at checkpoint 0 (should be >0)")

    # Check: checkpoint splitting produced >1 ckpt for long responses
    for t in total_sys:
        if not t.resumed and len(t.text) > 100 and len(t.checkpoints) < 2:
            issues.append("Long response not split: {!r}".format(t.text[:60]))

    return {
        "total_system_turns": len(total_sys),
        "interrupted": len(interrupted),
        "resumed_correctly": len(resumptions),
        "issues": issues,
        "score": 1.0 - len(issues) / max(1, len(total_sys)),
    }


# ── dialog scripts ────────────────────────────────────────────────────────────

SCRIPT = [
    # Normal turn — no interrupt
    "Tell me about rainbows.",
    # Interrupt mid-response
    "Explain how clouds form. (interrupt)",
    # Resume from interruption
    "continue",
    # New topic — clears interrupt state
    "What is the speed of light?",
    # Interrupt again
    "Describe the water cycle in detail. (interrupt)",
    # Resume
    "go on",
    # Follow-up after resume — should have context consistency
    "And what about groundwater?",
]


@pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="Phi-3 model not present",
)
def test_checkpoint_dialog():
    print("\n" + "=" * 70)
    print("  CHECKPOINT NARRATOR — dialog test")
    print("=" * 70)

    turns = _run_dialog(SCRIPT)

    print("\n" + "=" * 70)
    report = _assess(turns)
    print("\n  Score: {:.0%}".format(report['score']))
    print("  System turns:   {}".format(report['total_system_turns']))
    print("  Interrupted:    {}".format(report['interrupted']))
    print("  Resumed correctly: {}".format(report['resumed_correctly']))
    if report['issues']:
        print("  Issues:")
        for iss in report['issues']:
            print("    ! {}".format(iss))
    else:
        print("  No issues detected.")

    # Save dialog log
    log = []
    for t in turns:
        log.append({
            "role": t.role,
            "text": t.text,
            "checkpoints": t.checkpoints,
            "interrupted": t.interrupted,
            "resumed": t.resumed,
            "resume_from": t.resume_from,
        })
    with open("/tmp/checkpoint_dialog.json", "w") as f:
        json.dump({"turns": log, "report": report}, f, indent=2)
    print("\n  Dialog saved to /tmp/checkpoint_dialog.json")
    print("=" * 70)

    assert report["score"] >= 0.80, "Score {:.0%} < 80%".format(report['score'])
    assert report["resumed_correctly"] == report["interrupted"], \
        "Not all interruptions were followed by correct resumption"
