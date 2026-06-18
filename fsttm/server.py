from __future__ import annotations  # PEP 563: lazy annotations for Python 3.8 (HvacBridge | None)
import asyncio
import logging as _pylog
import re
from collections import namedtuple
from functools import partial
from typing import List, Optional, Set

_log = _pylog.getLogger("fsttm.server")   # → fsttm.log (see _setup_file_logging)

import reactivex as rx
import reactivex.operators as ops
from reactivex.subject import Subject
from reactivex.scheduler.eventloop import AsyncIOScheduler
from reactivex.scheduler import ImmediateScheduler

from cyclotron import Component
from cyclotron.asyncio.runner import run
import cyclotron_std.logging as logging
import cyclotron_std.sys.stdout as stdout
import cyclotron_std.io.file as file
import cyclotron_std.sys.argv as argv

from fsttm.aec import EchoCancelSession
from fsttm.attention import Attention
from fsttm.config import parse_arguments, parse_config
from fsttm.fsttm import Model as FSM
from fsttm.hvac_bridge import HvacBridge
import fsttm.perception as perception
from fsttm.perception import SpeechDuringPlayback
import fsttm.llama as llama
import fsttm.piper as piper
import fsttm.whisper as whisper

# Spoken when the attention layer goes to sleep / mutes (sleep_intent path).
_SLEEP_CONFIRM_PHRASE = "Voice controls disabled."

# Bracketed placeholders the LLM emits when it lacks a runtime value, e.g.
# "[current temperature]", "<value>". They must never be spoken aloud raw.
_PLACEHOLDER_RE = re.compile(r'[\[<][^\]>]*[\]>]')
# A temperature placeholder specifically — interpolated with the real value when
# the hvac backend cache has one. Matches "[current temperature]", "[temp]", etc.
_TEMP_PLACEHOLDER_RE = re.compile(r'[\[<][^\]>]*temp[^\]>]*[\]>]', re.IGNORECASE)


def _interpolate_placeholders(text, temp_value=None):
    """Replace known placeholders with REAL values before any stripping. Today
    that's the temperature placeholder → the live hvac reading (just the number;
    the sentence usually already says "degrees"). Returns the text (possibly with
    other placeholders left for _strip_placeholders)."""
    if not text or temp_value is None:
        return text
    if '[' not in text and '<' not in text:
        return text
    return _TEMP_PLACEHOLDER_RE.sub("{:g}".format(float(temp_value)), text)


def _strip_placeholders(text):
    """Remove any remaining [..]/<..> placeholder spans from a spoken line and
    tidy the leftover spacing/punctuation. Returns None if nothing meaningful
    remains, so the caller can substitute a clean fallback than speak a fragment."""
    if not text or ('[' not in text and '<' not in text):
        return text
    cleaned = _PLACEHOLDER_RE.sub('', text)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    cleaned = re.sub(r'\s+([.,!?])', r'\1', cleaned).strip()
    return cleaned if len(cleaned) >= 3 else None

FSTTMSink = namedtuple('Sink', [
    'perception', 'stt', 'llm', 'tts', 'logging', 'file', 'stdout'
])
FSTTMSource = namedtuple('Source', [
    'perception', 'stt', 'llm', 'tts', 'file', 'argv'
])
FSTTMDrivers = namedtuple('Drivers', [
    'perception', 'stt', 'llm', 'tts', 'stdout', 'logging', 'file', 'argv'
])


def fsttm_server(aio_scheduler, sources, tui_state=None):
    # ── config ──────────────────────────────────────────────────────────────
    # When tui_state is set, server "log" lines go to the TUI's events panel and
    # chat/intents/state feed the panels; otherwise everything prints to stdout
    # (headless). _emit() is the single switch between the two.
    def _emit(text, level="info"):
        if tui_state is not None:
            tui_state.note(text, level)
        else:
            print(text)

    args = parse_arguments(sources.argv.argv)
    read_request, read_response = args.pipe(
        ops.map(lambda i: file.Read(id='config', path=i.value)),
        file.read(sources.file.response),
    )
    read_request = read_request.pipe(ops.subscribe_on(aio_scheduler))
    config = parse_config(read_response)

    # ── share driver sources (prevent multiple cold subscriptions) ─────────
    voice_src = sources.perception.voice.pipe(ops.share())
    stt_src   = sources.stt.text.pipe(ops.share())
    llm_src   = sources.llm.system.pipe(ops.share())
    tts_src   = sources.tts.audio.pipe(ops.share())

    # ── N09-1071 FSM (turn-taking state machine) ───────────────────────────
    # One instance shared across all reactive branches via Python closure.
    turn = FSM()

    def _log_transition(who, action, had_floor):
        gained = not had_floor
        if tui_state is None:
            # Through the logger → timestamped, so FSM-transition pauses are visible.
            _log.info("[FSM] %s %s floor  (action=%s)  state=%s",
                      who, 'gains' if gained else 'releases', action, turn.state)

    turn.system_cb = lambda a, had: _log_transition('SYSTEM', a, had)
    turn.user_cb   = lambda a, had: _log_transition('USER',   a, had)

    # Single FSM transition hook (always installed): feeds the TUI state diagram
    # AND drives auto-resume after a false barge-in. onchangestate fires for every
    # transition (incl. intra-floor moves like SYSTEM→BOTHs). The auto-resume
    # callbacks are forward-referenced — they exist by the time any event fires.
    def _on_fsm_change(e):
        if tui_state is not None:
            tui_state.set_fsm(turn.state, prev=turn.prev_state, action=e.event)
        _maybe_arm_autoresume()   # defined below (narrator section)

    turn.onchangestate = _on_fsm_change
    if tui_state is not None:
        # seed the initial state so the diagram shows USER before any transition
        tui_state.set_fsm(turn.state, prev="", action="")

    # ── logging ──────────────────────────────────────────────────────────────
    logs_config = config.pipe(
        ops.flat_map(lambda i: rx.from_(i.log.level,
                                        scheduler=ImmediateScheduler())),
        ops.map(lambda i: logging.SetLevel(logger=i.logger, level=i.level)),
    )

    # ── perception control ───────────────────────────────────────────────────
    perception_init = config.pipe(
        ops.flat_map(lambda i: rx.from_([
            perception.Initialize(i.vad.vad_aggressiveness,
                                  i.vad.device,
                                  i.vad.rate,
                                  i.vad.device_name,
                                  i.vad.padding_ms),
            perception.Start(),
        ])),
    )

    # SoftDuck/Unduck are driven by the narrator on RESPONSE boundaries (not per
    # checkpoint) via _duck_subject below: SoftDuck once when a response starts
    # speaking, Unduck once when its last checkpoint finishes (or on barge-in).
    # While soft-ducked, perception routes playback-period speech to a separate
    # SpeechDuringPlayback sentinel instead of STT, so neither AEC residue nor
    # raw-mic speaker bleed is mistaken for a user utterance. Gating by response
    # (rather than per-clause, which re-armed the grace window and left un-ducked
    # gaps between checkpoints) is what makes mid-response barge-in reliable.
    _duck_subject = Subject()
    perception_control = rx.merge(perception_init, _duck_subject)

    # When vad.soft_duck is false, skip the soft-duck/sentinel path entirely and
    # rely on AEC alone — barge-in is then a plain VAD onset while the system has
    # the floor (see barge_in_signal below). Default true.
    _soft_duck = [True]
    config.subscribe(on_next=lambda cfg: _soft_duck.__setitem__(
        0, bool(getattr(cfg.vad, 'soft_duck', True))))

    # ── utterance segmentation ───────────────────────────────────────────────
    utterance_end = voice_src.pipe(ops.filter(lambda i: i is None))
    utterance = voice_src.pipe(
        ops.map(lambda i: bytes(i) if i is not None else b''),
        ops.buffer(utterance_end),
        ops.map(lambda xs: b''.join(xs)),
    )

    # ── FSM: user floor transitions ──────────────────────────────────────────
    # When user starts speaking (first non-None frame after silence)
    speech_started = voice_src.pipe(
        ops.filter(lambda i: i is not None),
        ops.distinct_until_changed(lambda i: i is not None),
        ops.filter(lambda _: True),   # always emit on state change to speech
    )
    # Drive FSM: user grabs on first speech frame, releases at utterance end
    def _user_grab(_):
        try:
            turn.user_action('G')
        except Exception:
            pass  # self-loop in USER state is fine

    def _user_release(_):
        try:
            turn.user_action('R')
        except Exception:
            pass

    speech_started.subscribe(on_next=_user_grab)
    utterance_end.subscribe(on_next=_user_release)

    # ── FSM: system floor transitions ────────────────────────────────────────
    def _system_grab(_):
        try:
            turn.system_action('G')
        except Exception:
            pass

    def _system_release(_):
        try:
            turn.system_action('R')
        except Exception:
            pass

    # The system floor is released once per response by the narrator's
    # _end_narration() (last checkpoint done), NOT on every checkpoint's
    # PlaybackDone — otherwise the floor would flap mid-response.

    # ── barge-in: user speaks while the system is narrating ──────────────────
    # Detected from the SpeechDuringPlayback sentinel emitted by perception while
    # soft-ducked. A grace window suppresses the first 600 ms after a response
    # starts speaking: the TTS onset transient (and any leftover VAD frames from
    # the user's just-finished utterance) would otherwise self-trigger a barge-in
    # immediately. The window is armed ONCE per response (when narration starts),
    # not per checkpoint — see _arm_barge_grace below.
    import time as _time
    _barge_in_open_at = [0.0]   # mutable cell; barge-in ignored until monotonic≥it

    def _arm_barge_grace() -> None:
        _barge_in_open_at[0] = _time.monotonic() + 0.6

    # Soft-duck mode: barge-in from the SpeechDuringPlayback sentinel, gated by
    # the grace window. Used when vad.soft_duck is true (the default).
    barge_in_ducked = voice_src.pipe(
        ops.filter(lambda i: type(i) is SpeechDuringPlayback),
        ops.filter(lambda _: _soft_duck[0]),
        ops.filter(lambda _: _time.monotonic() >= _barge_in_open_at[0]),
    )
    # AEC-only mode: barge-in from a plain VAD onset while the system holds the
    # floor. Used when vad.soft_duck is false (no sentinel is emitted then).
    barge_in_raw = speech_started.pipe(
        ops.filter(lambda _: not _soft_duck[0]),
        ops.filter(lambda _: turn.is_system),
        ops.filter(lambda _: _time.monotonic() >= _barge_in_open_at[0]),
    )
    barge_in_signal = rx.merge(barge_in_ducked, barge_in_raw).pipe(
        ops.throttle_first(1.0, aio_scheduler),  # one barge-in per second max
    )

    # Tentative barge-in: a VAD blip during system speech is only a "maybe". We
    # flip the floor + unduck so STT can evaluate the audio, but DON'T cut TTS or
    # set the resume point yet — the system keeps talking. A real (non-noise)
    # transcript CONFIRMS the barge-in (_confirm_barge_in, cuts TTS); a noise
    # transcript leaves the narration untouched and the floor is restored to
    # SYSTEM by the auto-resume. This stops coughs/sighs from stuttering the
    # speech and thrashing the FSM (which was losing real utterances).
    _barge_tentative = [False]

    def _on_barge_in(_):
        if not turn.is_system:
            _log.debug("barge-in signal ignored (floor=%s, not system)", turn.state)
            return   # system already yielded floor — ignore duplicate signal
        # Listen-only: flip floor to USER so the STT gate opens (is_user), unduck
        # so the audio reaches STT. The system KEEPS speaking — we don't cut TTS
        # or set a resume point until a real transcript confirms (_confirm_barge_in).
        try:
            turn.user_action('G')   # SYSTEM → BOTHs
            turn.system_action('R') # BOTHs  → USER (floor=USER; STT gate opens)
        except Exception:
            pass
        _barge_tentative[0] = True
        if tui_state is not None:
            tui_state.barge_ins += 1
        if _soft_duck[0]:
            _duck_subject.on_next(perception.Unduck())
        _log.debug("barge-in TENTATIVE → floor=%s (listening, TTS still playing)",
                   turn.state)

    barge_in_signal.subscribe(on_next=_on_barge_in)

    # ── STT ──────────────────────────────────────────────────────────────────
    # Config-driven parasite/noise phrases (stt.parasites) dropped before they
    # reach the LLM — extends the []/()/** annotation auto-detection.
    config.subscribe(on_next=lambda cfg: whisper.set_parasites(
        getattr(cfg.stt, 'parasites', None)))

    stt_init = config.pipe(
        ops.flat_map(lambda i: rx.from_([
            whisper.Initialize(i.stt.model, language=i.stt.language)
        ])),
    )
    # Gate: only send to STT when user has the floor (not during system speech)
    stt_request = utterance.pipe(
        ops.filter(lambda i: len(i) > 0),
        ops.filter(lambda _: turn.is_user),   # FSM gate: user must have floor
        ops.map(lambda i: whisper.SpeechToText(data=i, context=None)),
    )
    stt_subjects = rx.merge(stt_init, stt_request)

    # ── HVAC backend bridge ───────────────────────────────────────────────────
    _bridge: list[HvacBridge | None] = [None]

    def _init_bridge(cfg):
        url = getattr(cfg.hvac_backend, 'url', None)
        t   = getattr(cfg.hvac_backend, 'timeout', 2.0)
        if url:
            _bridge[0] = HvacBridge(url=url, timeout=t)
            if tui_state is not None:
                tui_state.hvac_url = url
                tui_state.hvac_ok = True
            _emit(f"[hvac-bridge] connected to {url}", "good")
            # Seed the state cache so the first STATUS query has real values, then
            # refresh periodically to pick up changes made outside the voice path
            # (e.g. the web UI). The /command response also updates it live.
            async def _seed_and_poll():
                while _bridge[0] is not None:
                    try:
                        await _bridge[0].refresh_state()
                    except Exception:
                        pass
                    await asyncio.sleep(10.0)
            asyncio.ensure_future(_seed_and_poll())
        else:
            _bridge[0] = None

    config.subscribe(on_next=_init_bridge)

    # ── Manual RAG retriever (system.manual_store + manual_embed) ─────────────
    _retriever = [None]
    _manual_name = ['Nina']

    def _init_retriever(cfg):
        sysc = getattr(cfg, 'system', None)
        enabled = bool(getattr(sysc, 'manual', False)) if sysc else False
        store = getattr(sysc, 'manual_store', None) if sysc else None
        embed = getattr(sysc, 'manual_embed', None) if sysc else None
        _manual_name[0] = getattr(sysc, 'name', 'Nina') if sysc else 'Nina'
        if not enabled:
            _retriever[0] = None
            return
        if store and embed:
            try:
                from fsttm.rag import Retriever
                embed_gpu = bool(getattr(sysc, 'manual_embed_gpu', False))
                _retriever[0] = Retriever(store, embed, embed_gpu=embed_gpu)
                _emit(f"[manual] RAG ready ({store}, embed={'GPU' if embed_gpu else 'CPU'})",
                      "good")
            except Exception as e:
                _emit(f"[manual] RAG unavailable: {e}", "warn")
                _retriever[0] = None
        else:
            _emit("[manual] enabled but manual_store/manual_embed not set", "warn")

    config.subscribe(on_next=_init_retriever)

    _MANUAL_INTENTS = {'HOWTO', 'LOCATE', 'EXPLAIN'}

    def _is_manual(item):
        ij = item.intent_json or {}
        return isinstance(ij, dict) and ij.get('intent') in _MANUAL_INTENTS

    def _is_chitchat(item):
        ij = item.intent_json or {}
        return isinstance(ij, dict) and ij.get('intent') == 'CHITCHAT'

    def _on_chitchat_intent(item) -> bool:
        """A greeting / social remark → a real conversational reply (like chat
        mode), not the canned UNKNOWN deflection. Reuses the ManualGenerate
        path (one-shot full prompt → Response/ResponseDone narration), so no new
        event type. Returns True if it dispatched. Needs the original utterance."""
        utter = _last_user_text[0].strip()
        if not utter:
            return False
        name = _manual_name[0]
        prompt = (
            f"<|system|>\nYou are {name}, a warm, concise in-car voice assistant. "
            f"Reply to the driver's greeting or remark in ONE short, friendly "
            f"spoken sentence. No lists, no questions about car functions unless "
            f"natural.<|end|>\n"
            f"<|user|>\n{utter}<|end|>\n<|assistant|>\n"
        )
        _llm_subject.on_next(llama.ManualGenerate(prompt=prompt, context='manual'))
        return True

    def _on_manual_intent(item) -> bool:
        """Run RAG retrieval + a grounded answer for a manual question, narrated
        via ManualGenerate→ResponseDone. Returns True if it dispatched the RAG
        answer; False means the caller must speak a fallback (so we never deadlock
        with the floor held and nothing narrated)."""
        from fsttm.rag import build_answer_prompt
        if _retriever[0] is None:
            return False
        ij = item.intent_json or {}
        # Prefer the model's `topic`; fall back to the raw utterance (the model
        # often omits topic — "where is the trunk button" → {HOWTO} with no topic).
        query = (ij.get('topic') or '').strip() or _last_user_text[0].strip()
        if not query:
            return False
        context, hits = _retriever[0].context(query)
        if not context:
            _emit(f"[manual] no passages for {query!r}", "warn")
            return False
        pages = sorted({h[2].get('page') for h in hits})
        _emit(f"[manual] {query!r} → {len(hits)} passages (pp.{pages})", "info")
        prompt = build_answer_prompt(_manual_name[0], query, context)
        # context='manual' tags the ResponseDone so it's narrated even in intent
        # mode (where the normal narrator-response path is gated off).
        _llm_subject.on_next(llama.ManualGenerate(prompt=prompt, context='manual'))
        return True

    def _on_intent_result(item):
        # Manual/RAG and chitchat intents are handled in _on_narrator_intent and
        # don't map to a device command — don't forward them to the hvac backend.
        if _is_manual(item) or _is_chitchat(item):
            return
        if _bridge[0] is not None and item.intent_json:
            asyncio.ensure_future(_bridge[0].post_intent(item.intent_json))

    # Forward IntentResult to hvac-react backend (must be after _on_intent_result)
    llm_src.pipe(
        ops.filter(lambda i: type(i) is llama.IntentResult),
    ).subscribe(on_next=_on_intent_result)

    # ── intent mode: read config at startup ──────────────────────────────────
    # Mutable cells updated when config stream fires.
    _intent_mode = [False]
    _intent_domains = [None]   # None → all registered intent domains

    def _read_intent_cfg(cfg):
        sysc = getattr(cfg, 'system', None)
        _intent_mode[0] = bool(getattr(sysc, 'hvac_intent', False)) if sysc else False
        _intent_domains[0] = getattr(sysc, 'intent_domains', None) if sysc else None
        if tui_state is not None:
            tui_state.intent_mode = _intent_mode[0]
            tui_state.soft_duck = bool(getattr(cfg.vad, 'soft_duck', True))

    config.subscribe(on_next=_read_intent_cfg)

    # ── Attention layer (wake word / sleep) ───────────────────────────────────
    # Sits above the floor FSM: gates whether a transcript is dispatched to the
    # LLM. Disabled (system.attention=false) → always AWAKE, today's behaviour.
    _attn = [Attention(enabled=False)]
    _sleep_intent = [False]

    def _read_system_cfg(cfg):
        sysc = getattr(cfg, 'system', None)
        enabled = bool(getattr(sysc, 'attention', False)) if sysc else False
        _sleep_intent[0] = bool(getattr(sysc, 'sleep_intent', False)) if sysc else False
        _attn[0] = Attention(
            enabled=enabled,
            name=getattr(sysc, 'name', 'Nina') if sysc else 'Nina',
            wake_words=getattr(sysc, 'wake_words', None) if sysc else None,
        )
        if tui_state is not None:
            tui_state.attention_enabled = enabled
            tui_state.attention_state = _attn[0].state if enabled else 'OFF'
        if enabled:
            _emit(f"[attention] enabled — starting {_attn[0].state} "
                  f"(name={_attn[0].name!r}, sleep_intent={_sleep_intent[0]})", "good")

    config.subscribe(on_next=_read_system_cfg)

    def _set_attn_state(state):
        _attn[0].state = state
        if tui_state is not None:
            tui_state.attention_state = state

    # ── LLM ──────────────────────────────────────────────────────────────────
    def _make_llm_init(cfg):
        g = cfg.gpt
        events = [llama.Initialize(
            model_path=g.model,
            n_ctx=getattr(g, 'n_ctx', 2048),
            n_batch=getattr(g, 'n_batch', 512),
            n_threads=getattr(g, 'n_threads', 6),
            n_gpu_layers=getattr(g, 'n_gpu_layers', 99),
        )]
        sysc = getattr(cfg, 'system', None)
        if _intent_mode[0]:
            # Intent mode: assemble the system prompt from the ENABLED intent
            # domains (climate / lights / body), so the model is taught exactly
            # the intents the grammar allows. An optional hvac_prompt file is
            # appended for deployment-specific guidance.
            from fsttm import intents
            variant = getattr(sysc, 'prompt_variant', 'few-shot') if sysc else 'few-shot'
            prompt = intents.build_prompt(_intent_domains[0], variant=variant)
            _emit(f"[intent] prompt variant: {variant} "
                  f"(~{len(prompt)//4} tok)", "info")
            extra_file = getattr(sysc, 'hvac_prompt', None) if sysc else None
            if extra_file:
                try:
                    with open(extra_file) as f:
                        prompt = prompt + "\n\n" + f.read().strip()
                except OSError as e:
                    _emit(f"[intent] WARNING: cannot load prompt file: {e}", "warn")
            events.append(llama.AddSystem(prompt=prompt))
            doms = _intent_domains[0] or intents.INTENT_DOMAINS
            _emit(f"[intent] domains enabled: {', '.join(doms)}", "good")
        else:
            # Plain chat: tell the model its name so "Nina, …" is natural. Only
            # when the attention layer is on (otherwise keep default behaviour).
            if sysc and getattr(sysc, 'attention', False):
                name = getattr(sysc, 'name', 'Nina')
                events.append(llama.AddSystem(
                    prompt=(f"You are {name}, a concise, friendly voice "
                            f"assistant. Users may address you as {name}. "
                            f"Keep replies short and spoken-friendly.")))
        return events

    llm_init = config.pipe(ops.flat_map(lambda i: rx.from_(_make_llm_init(i))))

    # _llm_subject is the single dispatch channel for Generate/IntentGenerate
    # (and narrator-injected TrimHistory / ClassifySystem). All command dispatch
    # goes through the attention gate below, so we can hold/drop utterances while
    # ASLEEP or pending sleep classification — which a declarative pipe can't do.
    _llm_subject = Subject()
    # Barge-in cancels (StopGenerate/CancelPlayback) are pushed through these
    # subjects by _confirm_barge_in once a REAL transcript confirms the barge-in
    # — they no longer fire on the bare VAD sentinel (which is only tentative).
    llm_subjects = rx.merge(llm_init, _llm_subject)

    # Remember the last dispatched utterance so a manual/RAG intent that omits the
    # `topic` field can still retrieve against the raw question.
    _last_user_text = [""]

    def _dispatch_command(text, context):
        """Send a user utterance to the LLM (intent or plain)."""
        _last_user_text[0] = text or ""
        ev = (llama.IntentGenerate(text=text, context=context,
                                   domains=_intent_domains[0])
              if _intent_mode[0] else
              llama.Generate(text=text, context=context))
        _llm_subject.on_next(ev)

    # Double gate (STT time + dispatch time) + attention gate.
    def _on_transcript(i):
        if type(i) is not whisper.TextResult or not i.text:
            return
        # Parasite phrases ("thank you") are likely hallucinations on silence — but
        # a real word if the user actually said it. Keep one ONLY when it confirms a
        # pending barge-in (a real VAD onset occurred while the system spoke); drop
        # it as noise otherwise. Fixes "Thank you" being swallowed mid-narration
        # when the user meant to barge in.
        if getattr(i, 'parasite', False) and not _barge_tentative[0]:
            _log.debug("parasite dropped (no pending barge-in): %r", i.text)
            return
        _log.debug("transcript %r | floor=%s ckpt_int=%s tentative=%s parasite=%s",
                   i.text, turn.state, _ckpt_interrupted[0], _barge_tentative[0],
                   getattr(i, 'parasite', False))
        # A real utterance arrived → the user genuinely took the floor; cancel any
        # pending auto-resume (this was a true barge-in, not a noise blip).
        _cancel_autoresume()
        # CONFIRM a tentative barge-in: the VAD blip flipped the floor to let STT
        # listen, but TTS kept playing. A real (non-noise) transcript proves the
        # user spoke — only now do we cut TTS, set the resume point, cancel the
        # LLM. (A noise transcript never reaches here — whisper drops it — so the
        # auto-resume restores the floor to SYSTEM and the narration plays on.)
        if _barge_tentative[0]:
            _log.debug("real transcript confirms barge-in → cutting TTS")
            _confirm_barge_in()
        if _maybe_resume(i):          # "continue" → narrator resume, not the LLM
            return
        if not turn.is_user:          # floor gate
            _log.debug("transcript dropped — floor not USER (%s)", turn.state)
            return

        decision = _attn[0].on_utterance(i.text, sleep_intent=_sleep_intent[0])
        action = decision["action"]

        if action == "ignore":
            # ASLEEP and no wake word — show it but don't act. _emit goes to the
            # TUI events panel OR stdout (headless), so it's clear in BOTH why a
            # command did nothing: you must say the wake word first.
            if _attn[0].enabled:
                _emit(f"[asleep] ignored (say '{_attn[0].name}' first): {i.text!r}",
                      "info")
            return
        if action == "wake":
            _set_attn_state('AWAKE')
            _emit(f"[attention] wake — {_attn[0].name} is listening", "good")
            if decision["text"]:      # wake word had a trailing command
                _dispatch_command(decision["text"], i.context)
            return

        # action == "command".
        # Sleep/mute is only ever considered when the user ADDRESSED the assistant
        # by name ("Hey Nina, voice off"). A bare command — even one STT garbled
        # into a sleep-like phrase ("stop climate" → "stop, glimar") — goes
        # straight to dispatch and can never disable voice control.
        if _attn[0].enabled and _sleep_intent[0] and decision.get("wake_prefixed"):
            # Real intent: classify {command|sleep|mute} first, act on result.
            _pending_cmd[0] = (i.text, i.context)
            _llm_subject.on_next(llama.ClassifySystem(text=i.text, context=i.context))
        else:
            _dispatch_command(i.text, i.context)

    # Holds the utterance awaiting sleep-intent classification.
    _pending_cmd = [None]

    def _on_system_intent(item):
        pending = _pending_cmd[0]
        _pending_cmd[0] = None
        utter = pending[0] if pending else ""
        if tui_state is not None:
            tui_state.add_system_intent(utter, item.action)
        if item.action in ("sleep", "mute"):
            # Speak a confirmation, then sleep once it finishes (deferred via
            # _sleep_after_speak in _end_narration). Going through the narrator
            # path gives it proper floor/duck handling.
            _emit(f"[attention] intent={item.action} → sleeping", "warn")
            _sleep_after_speak[0] = True
            _speak_confirmation(_SLEEP_CONFIRM_PHRASE)
        elif pending is not None:
            _dispatch_command(pending[0], pending[1])

    stt_src.subscribe(on_next=_on_transcript)
    llm_src.pipe(ops.filter(lambda i: type(i) is llama.SystemIntent)
                 ).subscribe(on_next=_on_system_intent)

    # Surface LLM driver errors — intent-grammar / approach_a / RAG failures emit
    # llama.LlamaError, which was previously unsubscribed (swallowed): a failed
    # intent left the FSM stuck with no feedback. Log it and show it in the TUI.
    def _on_llm_error(i):
        _log.error("LLM driver error (context=%s): %r", i.context, i.error)
        _emit(f"[llm] error: {i.error}", "warn")
    llm_src.pipe(ops.filter(lambda i: type(i) is llama.LlamaError)
                 ).subscribe(on_next=_on_llm_error)

    # ── FSM: system grabs floor when LLM response ready (both modes) ─────────
    # IntentResult replaces ResponseDone in intent mode.
    llm_src.pipe(
        ops.filter(lambda i: (
            (type(i) is llama.IntentResult and bool(i.tts_text)) or
            (type(i) is llama.ResponseDone  and bool(i.full_text))
        )),
    ).subscribe(on_next=_system_grab)

    # Remove the old ResponseDone-only subscription (already replaced above)

    # ── Checkpoint Narrator ────────────────────────────────────────────────────
    # Split each LLM response into clause-level TTS units ("checkpoints") and
    # play them one-by-one through piper. Each unit carries a `ckpt:N` context so
    # PlaybackStarted / AudioPlaybackStarted / PlaybackDone can be tracked back to
    # its index. Stage 3 builds replay/skip-on-barge-in + TrimHistory on top of
    # the per-checkpoint state tracked here.

    RESUME_PHRASES = {
        'continue', 'go on', 'please continue', 'keep going', 'go ahead',
        'please go on', 'carry on', 'and then', 'what else',
    }

    def _split_checkpoints(text: str) -> List[str]:
        """
        Clause-level TTS units — each becomes one piper.Speak event.
        Split on: sentence end (.!?), clause separator (, ; —) with min length.
        Numbered list markers (1. 2.) and decimal points never split.
        Short fragments (<15 chars) merged into previous unit.
        Target: 5-20 words per unit (1-4 seconds of speech).
        """
        # Sentence boundaries (not preceded by digit — avoids "1. item")
        sentence_re = re.compile(r'(?<=[^0-9][.!?])\s+')
        # Clause boundaries: comma/semicolon preceded by >=20 chars of content
        clause_re   = re.compile(r'(?<=\S),\s+|(?<=\S);\s+|(?<= —)\s+|(?<= –)\s+')

        # First split on sentences
        parts = sentence_re.split(text.strip())

        # Then split each sentence on clause markers if long enough
        fine = []
        for part in parts:
            clauses = clause_re.split(part)
            fine.extend(c.strip() for c in clauses if c.strip())

        # Merge fragments shorter than 15 chars into the preceding unit
        out, buf = [], ""
        for p in fine:
            buf = (buf + ", " + p).strip() if buf else p
            if len(buf) >= 20:
                out.append(buf); buf = ""
        if buf:
            if out: out[-1] = out[-1].rstrip(",") + " " + buf
            else:   out.append(buf)

        return [s.strip() for s in out if s.strip() and len(s.strip()) > 3]

    def _is_resume(text: str) -> bool:
        t = text.lower().strip().rstrip('.!?,')
        return t in RESUME_PHRASES

    # ── Narrator state ────────────────────────────────────────────────────────
    _ckpts:           List[str] = []   # all TTS units for current response
    _ckpts_done:      Set[int]  = set()# indices of units PlaybackDone confirmed
    _ckpt_playing     = [-1]           # index currently in piper (PlaybackStarted)
    _ckpt_interrupted = [-1]           # resume point after barge-in; -1=none
    _audio_start_time = [0.0]          # when aplay actually started for current unit
    _audio_duration   = [0.0]          # exact PCM duration of current unit (s)
    REPLAY_THRESHOLD  = 0.50           # < 50% heard → replay; ≥ 50% → skip
    _tts_subject      = Subject()      # narrator → piper channel (Speak/ClearQueue)

    def _ckpt_idx_from(ctx) -> Optional[int]:
        if isinstance(ctx, str) and ctx.startswith('ckpt:'):
            try: return int(ctx.split(':')[1])
            except ValueError: pass
        return None

    # Clean common TTS artefacts before sending a unit to piper
    _artifact_re = re.compile(r'\s*\.,\s*|\s*,\.\s*|\s{2,}')

    def _clean(text: str) -> str:
        return _artifact_re.sub(' ', text).strip().lstrip(',').strip()

    # Narrating-state flag: True while a response's checkpoints are being spoken.
    # Drives one SoftDuck (+grace) at the start and one Unduck/floor-release at
    # the end, regardless of how many checkpoints the response splits into.
    _narrating    = [False]
    _last_emitted = [-1]   # highest checkpoint index actually sent to piper
    _sleep_after_speak = [False]   # flip to ASLEEP once the current phrase ends

    def _begin_narration() -> None:
        """Response starts speaking: soft-duck once and arm the grace window."""
        if not _narrating[0]:
            _narrating[0] = True
            if _soft_duck[0]:
                _duck_subject.on_next(perception.SoftDuck())
        _arm_barge_grace()

    def _end_narration() -> None:
        """Response finished (or was cut): unduck once and release the floor."""
        if _narrating[0]:
            _narrating[0] = False
            if _soft_duck[0]:
                _duck_subject.on_next(perception.Unduck())
            _system_release(None)
        # A pending sleep confirmation just finished playing → sleep now (after
        # the audio, so the transition is clean and not mid-playback).
        if _sleep_after_speak[0]:
            _sleep_after_speak[0] = False
            _set_attn_state('ASLEEP')
            _emit("[attention] → ASLEEP", "warn")

    def _speak_confirmation(phrase) -> None:
        """Speak a short fixed phrase through the narrator path (floor + duck +
        clean release), then act on _sleep_after_speak in _end_narration. Falls
        back to an immediate state flip if there's nothing to play."""
        _ckpts.clear(); _ckpts.append(phrase)
        _ckpts_done.clear()
        _ckpt_playing[0] = 0; _ckpt_interrupted[0] = -1
        if _play_from(0) > 0:
            _system_grab(None)        # system takes the floor to speak
            _begin_narration()
        else:
            _end_narration()          # nothing spoken → just do the pending sleep

    def _play_from(idx: int) -> int:
        # Queue checkpoints [idx, …] to piper and return how many were emitted.
        # A checkpoint that cleans to empty emits no Speak (and thus no
        # PlaybackDone), so the response ends when _last_emitted's PlaybackDone
        # arrives, not len-1.
        emitted = 0
        for i in range(idx, len(_ckpts)):
            text = _clean(_ckpts[i])
            if text:
                _tts_subject.on_next(piper.Speak(text=text, context=f'ckpt:{i}'))
                _last_emitted[0] = i
                emitted += 1
        return emitted

    def _on_narrator_response(item) -> None:
        if not item.full_text: return
        # Interpolate a temperature placeholder with the real reading, then strip
        # any other bracketed placeholder so it's never spoken; whole-placeholder
        # reply → clean fallback.
        _t = _bridge[0].get_value('HVAC_TEMPERATURE_CURRENT') if _bridge[0] else None
        text = _interpolate_placeholders(item.full_text, _t)
        text = _strip_placeholders(text) or "Okay."
        _ckpts.clear(); _ckpts.extend(_split_checkpoints(text))
        _ckpts_done.clear()
        _ckpt_playing[0] = 0; _ckpt_interrupted[0] = -1
        # Begin narration (soft-duck + grace) only if something will be spoken —
        # otherwise no PlaybackDone would ever unduck us. Queueing is synchronous
        # and audio events are async, so ducking right after still precedes the
        # TTS onset.
        if _play_from(0) > 0:
            _begin_narration()

    def _local_answer(item):
        """Spoken answer for intents the system resolves itself (TIME/DATE/STATUS)
        — returns None for everything else so the LLM's tts_text is used.
        Deterministic; never a hallucinated value."""
        ij = item.intent_json or {}
        name = ij.get('intent') if isinstance(ij, dict) else None
        if name == 'TIME' or name == 'DATE':
            import datetime as _dt
            now = _dt.datetime.now()
            if name == 'TIME':
                # 12-hour, spoken-friendly: "It's 3:42 PM." (strip leading zero)
                return "It's {}.".format(now.strftime('%-I:%M %p'))
            # DATE: "It's Monday, June 2nd." with an ordinal day.
            d = now.day
            suffix = 'th' if 11 <= d % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(d % 10, 'th')
            return "It's {}.".format(now.strftime('%A, %B {}{}').format(d, suffix))
        if name == 'STATUS':
            # A car-state query → answer with REAL telemetry from the hvac_react
            # backend cache (set + current), not an LLM-invented value.
            br = _bridge[0]
            area = int(ij.get('area') or 1) or 1
            sset = br.get_value('HVAC_TEMPERATURE_SET', area) if br else None
            cur  = br.get_value('HVAC_TEMPERATURE_CURRENT', area) if br else None
            zone = {1: "driver side", 4: "passenger side"}.get(area, "")
            where = (" on the " + zone) if zone else ""
            if sset is not None and cur is not None:
                return ("Set to {:g} degrees{}, currently {:g}."
                        .format(float(sset), where, float(cur)))
            val = sset if sset is not None else cur
            if val is not None:
                return "Temperature{} is {:g} degrees.".format(where, float(val))
            return "I don't have the current readings right now."
        return None

    def _intent_spoken(item):
        """The SINGLE source of truth for an intent's spoken/displayed text:
        a local answer (TIME/DATE/STATUS) if any, else the LLM ack with a
        temperature placeholder interpolated and any leftover placeholder cleaned.
        Used by BOTH the narrator (TTS) and the chat/display feeds so they never
        diverge (was: TTS spoke the real value, chat showed the raw LLM text)."""
        local = _local_answer(item)
        if local is not None:
            return local
        ack = item.tts_text or ""
        _t = _bridge[0].get_value('HVAC_TEMPERATURE_CURRENT') if _bridge[0] else None
        ack = _interpolate_placeholders(ack, _t)
        return "Okay, done." if ('[' in ack or '<' in ack) else ack

    def _on_narrator_intent(item) -> None:
        # Manual (RAG) intents: try the grounded RAG answer (narrated separately
        # via ManualGenerate→ResponseDone). Only skip narration here if RAG
        # actually fired; if it couldn't (no topic AND no usable query, or no
        # passages), fall through and speak the classifier's tts_text so the floor
        # is always released — never deadlock with SYSTEM held and nothing spoken.
        if _is_chitchat(item):
            if _on_chitchat_intent(item):
                return   # conversational reply streams via Response/ResponseDone
            # couldn't (no utterance) → speak the classifier ack below
        if _is_manual(item):
            if _on_manual_intent(item):
                return
            # RAG didn't fire → speak a fallback below.
        # Single source of truth — same text the chat/display feeds use.
        spoken = _intent_spoken(item)
        if spoken:
            _ckpts.clear(); _ckpts.append(spoken)
            _ckpts_done.clear()
            _ckpt_playing[0] = 0; _ckpt_interrupted[0] = -1
            if _play_from(0) > 0:
                _begin_narration()

    # PlaybackStarted: update which unit is in synthesis/queue
    def _on_ckpt_started(item) -> None:
        idx = _ckpt_idx_from(item.context)
        if idx is not None:
            _ckpt_playing[0] = idx
            if tui_state is not None:
                tui_state.ckpt_cur = idx
                tui_state.ckpt_total = len(_ckpts)

    # AudioPlaybackStarted: unit is NOW audible — record exact timing
    def _on_audio_started(item) -> None:
        _audio_start_time[0] = _time.monotonic()
        _audio_duration[0]   = item.duration_s

    # PlaybackDone: unit fully heard — record it, and if it was the LAST
    # checkpoint of the response, end narration (unduck + release the floor).
    def _on_ckpt_done_narrator(item) -> None:
        idx = _ckpt_idx_from(item.context)
        if idx is None:
            return
        _ckpts_done.add(idx)
        # Last EMITTED checkpoint done with no pending resume → response fully
        # spoken. (A barge-in clears via _on_barge_in_narrator instead.) Using
        # _last_emitted (not len-1) handles trailing units that cleaned to empty.
        if (_narrating[0] and _ckpt_interrupted[0] < 0
                and idx >= _last_emitted[0]):
            _end_narration()

    # Barge-in: flush queued + in-flight checkpoints (ClearQueue), unduck, decide
    # replay vs skip from the audio fraction heard, and roll the LLM history back
    # to the confirmed-heard checkpoints (TrimHistory).
    def _confirm_barge_in() -> None:
        """Confirm a tentative barge-in once a REAL transcript proves the user
        actually spoke (not a cough/sigh). NOW we cut TTS, set the resume point,
        cancel the LLM, and trim history. Called from _on_transcript — never from
        the bare VAD sentinel, so noise can't stutter the narration."""
        _barge_tentative[0] = False
        if not _narrating[0]:
            return   # nothing playing to interrupt (already cut / not narrating)
        interrupted = _ckpt_playing[0]
        _tts_subject.on_next(piper.ClearQueue())
        _llm_subject.on_next(llama.StopGenerate())
        # Narration is cut: floor already flipped + unducked on the tentative
        # signal. The resume path will soft-duck again via _begin_narration.
        _narrating[0] = False

        # Timing for the currently-playing (interrupted) unit
        elapsed  = _time.monotonic() - _audio_start_time[0]
        dur      = _audio_duration[0] or 1.0
        fraction = min(1.0, elapsed / dur)

        if fraction >= REPLAY_THRESHOLD:
            decision  = "skip   ({:.0%} heard)".format(fraction)
            resume_at = interrupted + 1
        else:
            decision  = "replay ({:.0%} heard)".format(fraction)
            resume_at = interrupted

        _ckpt_interrupted[0] = resume_at

        # Exact history: only units confirmed PlaybackDone *before* the
        # interrupted one (the interrupted unit was cut, so it doesn't count
        # even if its done-event slipped in). Roll the LLM's last assistant turn
        # back to just that heard text, so a later "continue" or follow-up keeps
        # the model's memory consistent with what was actually spoken.
        confirmed = sorted(i for i in _ckpts_done if i < interrupted)
        heard_text = " ".join(_ckpts[i] for i in confirmed)
        if heard_text:
            _llm_subject.on_next(llama.TrimHistory(heard_text=heard_text))

        msg = ("  [narrator] ckpt {}/{} ({} confirmed done) "
               "| audio {:.2f}/{:.2f}s={:.0%} | {} | resume→{}".format(
                   interrupted, len(_ckpts) - 1, len(confirmed),
                   elapsed, dur, fraction, decision, resume_at))
        if tui_state is not None:
            tui_state.last_barge = "ckpt {} · {}".format(interrupted, decision.strip())
        _emit(msg, "warn")

    tts_src.pipe(ops.filter(lambda i: type(i) is piper.PlaybackStarted)
    ).subscribe(on_next=_on_ckpt_started)

    tts_src.pipe(ops.filter(lambda i: type(i) is piper.AudioPlaybackStarted)
    ).subscribe(on_next=_on_audio_started)

    tts_src.pipe(ops.filter(lambda i: type(i) is piper.PlaybackDone)
    ).subscribe(on_next=_on_ckpt_done_narrator)


    llm_src.pipe(
        ops.filter(lambda i: type(i) is llama.ResponseDone and bool(i.full_text)
                             and (not _intent_mode[0] or i.context == 'manual')),
    ).subscribe(on_next=_on_narrator_response)

    llm_src.pipe(ops.filter(lambda i: type(i) is llama.IntentResult)
    ).subscribe(on_next=_on_narrator_intent)

    # ── TTS ───────────────────────────────────────────────────────────────────
    # Output device/sink are config-driven (tts.device / tts.sink), resolved by
    # name. The sink default depends on AEC: with module-echo-cancel active, TTS
    # must play into its reference sink (fsttm_ec_sink) so the AEC can subtract
    # it from the mic; in raw-mic mode (FSTTM_NO_AEC) there is no such sink, so
    # fall back to the PulseAudio default. An explicit tts.sink always wins.
    import os as _os
    import subprocess as _sp
    def _sink_exists(name):
        try:
            r = _sp.run(['pactl', 'list', 'short', 'sinks'],
                        capture_output=True, text=True, timeout=2)
            return any(name in line for line in r.stdout.splitlines())
        except Exception:
            return False
    def _tts_init_events(i):
        device = getattr(i.tts, 'device', None) or 'pulse'
        sink   = getattr(i.tts, 'sink', None)
        if sink is None and not _os.environ.get('FSTTM_NO_AEC'):
            # Route TTS into the AEC reference sink so the echo-canceller hears it
            # — BUT only if it actually exists. If the echo-cancel module failed to
            # load (half-loaded AEC), routing to a missing sink silences TTS. Fall
            # back to the PulseAudio default sink so audio always plays.
            if _sink_exists('fsttm_ec_sink'):
                sink = 'fsttm_ec_sink'
            else:
                _emit("[tts] fsttm_ec_sink missing → default sink (AEC not loaded)",
                      "warn")
                sink = None
        return [piper.Initialize(model_path=i.tts.model,
                                 sample_rate=i.tts.sample_rate,
                                 device=device, sink=sink,
                                 cuda=bool(getattr(i.tts, 'cuda', False)))]
    tts_init = config.pipe(ops.flat_map(lambda i: rx.from_(_tts_init_events(i))))
    # All Speak events now flow through _tts_subject, fed by the narrator's
    # _play_from(). Barge-in flushes via ClearQueue (emitted on _tts_subject too).
    tts_subjects = rx.merge(tts_init, _tts_subject)

    # ── Resume ("continue") detection ────────────────────────────────────────
    # When the user says "continue" (etc.) after a barge-in, replay the
    # remaining checkpoints WITHOUT a new LLM call. The llm_request gate above
    # uses _maybe_resume to keep these phrases out of the LLM.
    def _maybe_resume(item) -> bool:
        return _is_resume(item.text) and _ckpt_interrupted[0] >= 0

    resume_stream = stt_src.pipe(
        ops.filter(lambda i: type(i) is whisper.TextResult and _maybe_resume(i)),
        ops.filter(lambda _: turn.is_user),
    )

    def _resume_now(reason="continue") -> None:
        # _ckpt_interrupted holds the exact resume index (replay → the interrupted
        # unit; skip → the next one), set by _on_barge_in_narrator.
        resume_from = _ckpt_interrupted[0]
        _ckpt_interrupted[0] = -1
        if 0 <= resume_from < len(_ckpts):
            _emit("  [narrator] {} — resuming from checkpoint {}/{}".format(
                reason, resume_from, len(_ckpts) - 1))
            _ckpt_playing[0] = resume_from
            try:
                turn.system_action('G')   # system regains floor to speak
            except Exception:
                pass
            if _play_from(resume_from) > 0:
                _begin_narration()        # soft-duck + arm grace for the replay
        else:
            _emit("  [narrator] no remaining checkpoints to resume")

    def _do_resume(_) -> None:
        _resume_now("continue")

    resume_stream.subscribe(on_next=_do_resume)

    # ── Auto-resume after a FALSE barge-in ───────────────────────────────────
    # A spurious VAD blip during system speech can grab the floor (SYSTEM→BOTHs→
    # USER) and, with no real utterance, land in FREEu — abandoning the narration.
    # When we enter FREEu with an interrupted narration pending (_ckpt_interrupted
    # ≥ 0), wait a short delay: if a real STT utterance arrives, the user genuinely
    # took the floor (cancel); otherwise it was a false barge-in, so the system
    # auto-resumes. (Manual "continue" still works any time.)
    _AUTORESUME_DELAY = 1.0            # seconds to wait for a real utterance
    _autoresume_disp = [None]         # scheduled disposable, or None

    def _cancel_autoresume():
        if _autoresume_disp[0] is not None:
            try:
                _autoresume_disp[0].dispose()
            except Exception:
                pass
            _autoresume_disp[0] = None

    def _restore_floor_after_tentative(_sched=None, _state=None):
        # A tentative barge-in (VAD blip) was never confirmed by a real transcript
        # — it was noise. Nothing was cut (TTS kept playing), so just hand the
        # floor back to the still-narrating system and re-duck so STT goes quiet.
        _autoresume_disp[0] = None
        if not _barge_tentative[0] or _ckpt_interrupted[0] >= 0:
            return   # got confirmed in the meantime, or already resolved
        if turn.is_user and _narrating[0]:
            _log.debug("tentative barge-in unconfirmed (noise) → floor→SYSTEM")
            _barge_tentative[0] = False
            try:
                turn.system_action('G') # FREEu → SYSTEM (system reclaims floor)
            except Exception:
                pass
            if _soft_duck[0]:
                _duck_subject.on_next(perception.SoftDuck())

    def _maybe_arm_autoresume():
        # Called on every FSM transition. Arm only when we just entered FREEu and
        # a narration is interrupted-but-unfinished.
        if turn.state == 'FREEu' and _ckpt_interrupted[0] >= 0:
            _cancel_autoresume()
            _log.debug("auto-resume armed (FREEu, ckpt_int=%s, %.1fs)",
                       _ckpt_interrupted[0], _AUTORESUME_DELAY)

            def _fire(_sched=None, _state=None):
                _autoresume_disp[0] = None
                # Only resume if still idle in a user-free state with a pending
                # interruption and the user didn't end up speaking.
                if _ckpt_interrupted[0] >= 0 and turn.is_user:
                    _log.debug("auto-resume FIRE (floor=%s)", turn.state)
                    _resume_now("auto (false barge-in)")
                else:
                    _log.debug("auto-resume skipped (floor=%s, ckpt_int=%s)",
                               turn.state, _ckpt_interrupted[0])
            _autoresume_disp[0] = aio_scheduler.schedule_relative(
                _AUTORESUME_DELAY, _fire)
        elif turn.state == 'FREEu' and _barge_tentative[0] and _ckpt_interrupted[0] < 0:
            # Tentative (unconfirmed) barge-in landed at FREEu — the user's blip
            # closed without a real transcript. Give STT a beat to deliver a
            # late transcript, then restore the floor to the narrating system.
            _cancel_autoresume()
            _log.debug("tentative-restore armed (FREEu, %.1fs)", _AUTORESUME_DELAY)
            _autoresume_disp[0] = aio_scheduler.schedule_relative(
                _AUTORESUME_DELAY, _restore_floor_after_tentative)
        elif turn.is_system:
            # System speaking again (real resume / new response) → drop any timer.
            _cancel_autoresume()

    # ── output: TUI panels OR stdout ──────────────────────────────────────────
    if tui_state is not None:
        # Feed the chat/intent panels and right-panel narrator/hvac fields.
        # The full assistant reply lands once per response (ResponseDone) rather
        # than token-streaming, which the chat panel can't redraw mid-token.
        stt_src.pipe(
            ops.filter(lambda i: type(i) is whisper.TextResult and bool(i.text)),
        ).subscribe(on_next=lambda i: tui_state.add_user(i.text))

        llm_src.pipe(
            ops.filter(lambda i: type(i) is llama.ResponseDone and bool(i.full_text)),
        ).subscribe(on_next=lambda i: tui_state.add_assistant(i.full_text))

        llm_src.pipe(
            ops.filter(lambda i: type(i) is llama.IntentResult),
        ).subscribe(on_next=lambda i: (
            # Display the SAME interpolated text that gets spoken (real temp value,
            # placeholders resolved) — not the raw LLM ack — so chat == TTS.
            # Manual/RAG and chitchat intents stream their reply via ResponseDone,
            # so skip the classifier ack here (else it double-prints).
            None if (_is_manual(i) or _is_chitchat(i))
            else tui_state.add_assistant(_intent_spoken(i) or i.tts_text),
            tui_state.add_intent(i.intent_json,
                                 "(manual → RAG)" if _is_manual(i)
                                 else "(chitchat)" if _is_chitchat(i)
                                 else (_intent_spoken(i) or i.tts_text)),
        ))
        std_out = rx.empty()
    else:
        token_stream = llm_src.pipe(
            ops.filter(lambda i: type(i) is llama.Response),
            ops.map(lambda i: i.text),
        )
        intent_stream = llm_src.pipe(
            ops.filter(lambda i: type(i) is llama.IntentResult),
            # Voice line shows what's actually spoken. For manual/RAG intents the
            # classifier ack is DISCARDED (the grounded answer streams separately
            # via Response→token_stream), so don't print it — it looked like a
            # double answer.
            ops.map(lambda i: f"\n  JSON  → {i.intent_json}\n" + (
                "  (manual → RAG answer follows)\n" if _is_manual(i)
                else "  (chitchat → reply follows)\n" if _is_chitchat(i)
                else f"  Voice → {(_intent_spoken(i) or i.tts_text)!r}\n")),
        )
        transcript_stream = stt_src.pipe(
            ops.filter(lambda i: type(i) is whisper.TextResult),
            ops.map(lambda i: f"\n[user] {i.text}  [FSM:{turn.state}]\n"
                              + ("[intent] " if _intent_mode[0] else "[assistant] ")),
        )
        # Timestamped profiling markers through the logger (token_stream stays raw
        # on stdout so live token streaming isn't broken up). These let you see the
        # PAUSE between a transcript arriving and the intent result, etc.
        stt_src.pipe(
            ops.filter(lambda i: type(i) is whisper.TextResult and bool(i.text)),
        ).subscribe(on_next=lambda i: _log.info(
            "[user] %r  [FSM:%s]", i.text, turn.state))
        llm_src.pipe(
            ops.filter(lambda i: type(i) is llama.IntentResult),
        ).subscribe(on_next=lambda i: _log.info(
            "[intent-result] %s", i.intent_json))
        std_out = rx.merge(transcript_stream, token_stream, intent_stream)

    return FSTTMSink(
        perception=perception.Sink(control=perception_control),
        stt=whisper.Sink(request=stt_subjects),
        llm=llama.Sink(request=llm_subjects),
        tts=piper.Sink(request=tts_subjects),
        logging=logging.Sink(request=logs_config),
        file=file.Sink(request=read_request),
        stdout=stdout.Sink(data=std_out),
    )


def _route_raw_mic():
    """In no-AEC mode, make the Jabra (or first USB input) the PulseAudio
    default source at a usable level, so capture via the `pulse` device gets it.
    Best-effort: warns and continues if pactl/the device is unavailable."""
    import subprocess
    try:
        r = subprocess.run(['pactl', 'list', 'short', 'sources'],
                           capture_output=True, text=True, check=False)
        src = None
        for line in r.stdout.splitlines():
            name = line.split('\t')[1] if '\t' in line else ''
            if name.startswith('alsa_input') and 'Jabra' in name:
                src = name
                break
        if src is None:  # fall back to any real (non-monitor) alsa input
            for line in r.stdout.splitlines():
                name = line.split('\t')[1] if '\t' in line else ''
                if name.startswith('alsa_input') and '.monitor' not in name:
                    src = name
                    break
        if src:
            # Set the default source and VERIFY it stuck — after a restart /
            # device resume PulseAudio sometimes drops the default, which leaves
            # the `pulse` capture device unable to resolve an input (PortAudio
            # then fails with 'channelCount <= maxChans' and the mic never opens,
            # so STT/wake go dead). Retry a couple of times.
            ok = False
            for _ in range(3):
                subprocess.run(['pactl', 'set-default-source', src], check=False)
                subprocess.run(['pactl', 'set-source-mute', src, '0'], check=False)
                subprocess.run(['pactl', 'set-source-volume', src, '90%'], check=False)
                info = subprocess.run(['pactl', 'info'], capture_output=True,
                                      text=True, check=False)
                if src in info.stdout:
                    ok = True
                    break
                import time as _t
                _t.sleep(0.3)
            print(f"raw mic: default source→{src} @90%"
                  + ("" if ok else " (WARNING: default did not verify)"))
        else:
            print("WARNING: no USB input source found for raw-mic capture")
    except Exception as exc:
        print(f"WARNING: could not route raw mic ({exc})")


def _setup_file_logging(path="fsttm.log"):
    """Attach a single FileHandler to the `fsttm` logger root so ALL fsttm.*
    loggers (fsttm, fsttm.whisper, fsttm.server, …) write to one file. The
    per-logger LEVELS come from config (log.level via the cyclotron logging
    driver); the handler itself is at DEBUG so it never filters below that.
    Diagnostics use logging.getLogger("fsttm.<area>").debug(...)."""
    import logging as _logging
    root = _logging.getLogger("fsttm")
    # avoid duplicate handlers on re-init
    if any(getattr(h, '_fsttm_file', False) for h in root.handlers):
        return
    fh = _logging.FileHandler(path, mode="a", encoding="utf-8")
    fh._fsttm_file = True
    fh.setLevel(_logging.DEBUG)
    _fmt = _logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S")
    fh.setFormatter(_fmt)
    root.addHandler(fh)
    # Also mirror to STDOUT with the SAME timestamped format, so the headless
    # console (srv.out) is timestamped to the millisecond — you can see the PAUSES
    # between operations, not just each op's own duration. (Skipped under --tui,
    # which owns the screen.)
    import sys as _sys
    if '--tui' not in _sys.argv and not any(
            getattr(h, '_fsttm_console', False) for h in root.handlers):
        ch = _logging.StreamHandler(_sys.stdout)
        ch._fsttm_console = True
        ch.setLevel(_logging.DEBUG)
        ch.setFormatter(_fmt)
        root.addHandler(ch)
    root.setLevel(_logging.DEBUG)   # let config narrow per-logger; handler keeps all
    root.propagate = False
    root.info("── fsttm logging started → %s ──", path)


def main():
    import os, sys
    # Force UTF-8 stdout/stderr so the → arrows in status lines render even when
    # the (ssh) session locale is not UTF-8 (otherwise they show as mojibake).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8')
        except Exception:
            pass

    _setup_file_logging()   # all fsttm.* logs → fsttm.log

    # --tui: 3-panel Rich interface (chat / intents / state+perf) instead of the
    # scrolling stdout log. Strip the flag before cyclotron's argparse (which
    # only knows --config) sees it. Also accept FSTTM_TUI=1.
    want_tui = ('--tui' in sys.argv) or bool(os.environ.get('FSTTM_TUI'))
    if '--tui' in sys.argv:
        sys.argv.remove('--tui')

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_debug(False)

    # Barge-in cancels TTS by killing the aplay subprocess mid-write. On Python
    # 3.8 (Jetson backport), asyncio.subprocess.communicate()'s internal
    # _feed_stdin task then raises BrokenPipeError on a drain waiter that nobody
    # retrieves, logging a noisy "Future exception was never retrieved"
    # traceback (CPython bpo-39068, fixed in 3.9+). The kill is intentional and
    # the pipe error is expected, so swallow ONLY BrokenPipe/ConnectionReset and
    # defer every other unhandled exception to asyncio's default handler.
    def _ignore_bargein_pipe(loop, context):
        exc = context.get('exception')
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        loop.default_exception_handler(context)
    loop.set_exception_handler(_ignore_bargein_pipe)
    aio_scheduler = AsyncIOScheduler(loop=loop)

    # AEC needs a PulseAudio/PipeWire session + a working module-echo-cancel.
    # Set FSTTM_NO_AEC=1 to skip it (capture the raw mic directly) on hosts
    # where AEC is unavailable or undesired (e.g. a headset with little bleed,
    # or an old PulseAudio whose module-echo-cancel outputs silence).
    no_aec = bool(os.environ.get('FSTTM_NO_AEC'))
    if no_aec:
        from contextlib import nullcontext
        aec_ctx = nullcontext()
        print("AEC disabled (FSTTM_NO_AEC set) — capturing raw mic")
        _route_raw_mic()
    else:
        # Read aec.enabled / aec.rnnoise from the --config file (the reactive
        # config stream isn't available this early in main()).
        _aec_cfg = {}
        try:
            import yaml as _yaml
            _cfg_path = (sys.argv[sys.argv.index('--config') + 1]
                         if '--config' in sys.argv else 'config.sample.yaml')
            with open(_cfg_path) as _f:
                _aec_cfg = (_yaml.safe_load(_f) or {}).get('aec', {}) or {}
        except Exception:
            pass
        aec_ctx = EchoCancelSession(
            enabled=_aec_cfg.get('enabled', True),
            rnnoise=_aec_cfg.get('rnnoise', False),
            method=_aec_cfg.get('method', 'auto'))

    # Build the TUI (if requested and rich is installed). tui_state is threaded
    # into fsttm_server so the reactive streams feed the panels; tui owns the
    # Live render loop. If rich is missing, fall back to headless stdout.
    tui_state = None
    tui = None
    if want_tui:
        from fsttm import tui as _tui_mod
        if _tui_mod.FsttmTUI.available:
            tui_state = _tui_mod.TUIState()
            tui_state.aec = "off (raw mic)" if no_aec else "on"
            tui = _tui_mod.FsttmTUI(loop, tui_state)
        else:
            print("WARNING: --tui requested but 'rich' is not installed; "
                  "running headless. (pip install rich)")

    with aec_ctx:
        if tui is not None:
            tui.start()
        try:
            run(
                Component(
                    call=partial(fsttm_server, aio_scheduler, tui_state=tui_state),
                    input=FSTTMSource,
                ),
                FSTTMDrivers(
                    perception=perception.make_driver(loop),
                    stt=whisper.make_driver(loop),
                    llm=llama.make_driver(loop),
                    tts=piper.make_driver(loop),
                    stdout=stdout.make_driver(),
                    logging=logging.make_driver(),
                    file=file.make_driver(),
                    argv=argv.make_driver(),
                ),
                loop=loop,
            )
        finally:
            if tui is not None:
                tui.stop()


if __name__ == '__main__':
    import sys
    # Only inject the default config when the user didn't pass --config,
    # otherwise argparse sees two --config values and the default wins.
    if '--config' not in sys.argv:
        sys.argv += ['--config', 'config.sample.yaml']
    main()
