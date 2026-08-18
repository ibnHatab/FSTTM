# FSM semantic verification against N09-1071

Source: Raux & Eskenazi, *A Finite-State Turn-Taking Model for Spoken Dialog
Systems*, NAACL 2009 (§2.1, §3, Table 1). Audited implementations:
`fsttm/fsttm.py` (Python) and `go/internal/fsm/fsm.go` (Go, a faithful port).
Machine-checked by `go/internal/fsm/paper_test.go` and
`tests/fsttm_paper_test.py`.

## States (§3.1) — CONFORMANT

USER / SYSTEM (one and only one participant claims the floor), FREE_S /
FREE_U (nobody claims it, following resp. a SYSTEM / USER state), BOTH_S /
BOTH_U (both claim it, following resp. a SYSTEM / USER state). States are
defined by *intentions and obligations*, not surface speech vs silence —
both implementations use them exactly this way (e.g. the state stays USER
during pauses inside a user utterance).

## Transitions (§3.1) — CONFORMANT, + 2 documented extensions

Actions: G(rab), R(elease), K(eep), W(ait); pairs are noted
(system action, user action). The paper's canonical set, all present:

| # | Transition | Phenomenon (paper §3.1) |
|---|---|---|
| 1 | SYSTEM —(R,W)→ FREE_S | system releases at prompt end |
| 2 | FREE_S —(W,G)→ USER | turn transition with gap |
| 3 | FREE_S —(G,W)→ SYSTEM | time-out: system re-establishes |
| 4 | USER —(W,R)→ FREE_U | user releases |
| 5 | FREE_U —(G,W)→ SYSTEM | turn transition with gap |
| 6 | FREE_U —(W,G)→ USER | user resumes after pause |
| 7 | SYSTEM —(K,G)→ BOTH_S | user barge-in attempt |
| 8 | BOTH_S —(R,K)→ USER | successful barge-in (system yields) |
| 9 | BOTH_S —(K,R)→ SYSTEM | failed user interruption |
| 10 | USER —(G,K)→ BOTH_U | system cut-in |
| 11 | BOTH_U —(K,R)→ SYSTEM | successful cut-in (user yields) |
| 12 | BOTH_U —(R,K)→ USER | failed system interruption |

Constraints honoured (verified by tests attempting each):

- **No direct transitions between intermediate states** (FREE↔FREE,
  FREE↔BOTH, BOTH↔BOTH): the machine must pass through SYSTEM or USER.
- **Intermediate states are conditioned on the previous floor holder**, so
  not all transitions are bidirectional: no SYSTEM→BOTH_U, no USER→BOTH_S,
  no SYSTEM→FREE_U, no USER→FREE_S.

**Extensions (not in the paper), kept deliberately:**

- **E1 — self-loops** `SYSTEM —(G,W)→ SYSTEM` and `USER —(W,G)→ USER`. The
  event-driven engine re-issues grabs (VAD onsets while USER already holds
  the floor; narrator re-grabs on resume). The paper's action set in these
  states excludes G/W respectively; the self-loops make those re-issues
  idempotent no-ops instead of faults. They never change the state.
- **E2 — initialization action vector**: the machine starts in USER with the
  action pair (W,W), while the paper's USER state implies the user keeps the
  floor, i.e. (W,K). Consequently transition 10 (system cut-in) is
  unreachable until the user has grabbed once — the live engine's first VAD
  onset (a user G self-loop) normalizes the vector immediately, so this is
  unobservable in operation. The paper tests set up states the same way.

## Cost model (§3.2 Table 1, §3.3, §4.1) — CONFORMANT as EXPECTED cost

Table 1 (raw costs): SYSTEM K=0 R=C_S · BOTH_* K=C_O(τ) R=0 ·
USER W=0 G=C_U · FREE_* W=C_G(τ) G=0.

The implementations expose `system_actions_cost()` which computes the §3.3
**expected** cost C(A) = Σ_S P(s=S|O)·C(A,S) with fixed probability
regressors instead of the paper's logistic-regression estimators:

- system side: `K = P_B·C_O(τ)`, `R = (1−P_B)·C_S` with P_B = 0.1 — i.e.
  P(BOTH|claiming) fixed; collapses to Table 1 when P_B ∈ {0,1}.
- user side: `G = (1−P_F)·C_U`, `W = P_F·C_G(τ)` — exactly §4.1's
  `C(G|O) = (1−P(F|O))·C_U`, `C(W|O) = P(F|O)·C_G(τ)`, with
  P_F = 0.38 at pauses (FREE_U), 0.20 in speech (USER).
- constants: C_U = 5000, C_S = 100, C_G^p(τ) = 1·τ (paper live run:
  C_G^p = 1, C_U = 5000 — §5.4), C_O(τ) = exp((τ+100)/1000).

**Deviation D1 (documented):** in FREE_S the implementations use P_F = 0 and
a constant C_G = 1000, making G ≈ C_U and W ≈ 0 — a deliberate "no
time-out re-prompt" policy (the engine never re-grabs a floor it just
released). Paper Table 1 would price G at 0 there.

**Deviation D2 (scope):** action *selection* in the engines is event-driven
(VAD endpointing, LLM/TTS completion events trigger G/R directly); the cost
map is informational (STATE debug, TUI). The paper's decision-theoretic
selection (§3.3) — endpoint when C(G) < C(W) — is not the live decision
rule; VAD `padding_ms` plays the role of the §4.2 pause-duration model.

## Concurrency semantics — the "cut unfinished output" invariants

The paper's model is the ground truth for WHO holds the floor; the engine's
value is cutting unfinished output when the floor changes:

1. **TTS cut**: a confirmed barge-in (transition 8) must stop audible output
   promptly and report the exact fraction heard (drives replay/skip).
   Go: librhvoice `play_speech` callback returns 0 → synthesis aborts
   mid-stream; the malgo playback loop stops within one chunk.
2. **LLM context rollback**: every turn rewinds the KV cache to the constant
   system prefix (`llama_memory_seq_rm(prefix..end)`), so an interrupted /
   discarded generation leaves NO trace — the same utterance must produce
   byte-identical JSON regardless of what ran before it.
3. **Floor release only on completion**: PlaybackDone ≡ audio actually
   finished (or was cut); the FSM releases via transition 1 only then.

Verified by: `go/internal/fsm/paper_test.go` (table + constraints + the six
§3.1 phenomena), `go/internal/llm/rollback_test.go` (byte-identical JSON
across interleaved turns, model-gated), `go/internal/tts/librhvoice_test.go`
(mid-synthesis cut + fraction-heard, lib-gated),
`go/internal/pipeline/e2e_test.go` (full-turn FSM traces, barge-in cut,
echo-drop, output behavior with fake drivers), and
`tests/fsttm_paper_test.py` (Python FSM against the same canonical table).
