"""
Shared intent flow — the single place an IntentResult is interpreted.

Used by both the server pipeline and the headless harness so they can never
diverge. The engine dispatches on the DOMAIN PROVIDER's translate() output
(command dicts with a "cmd" discriminator) and its meta_intent() mapping —
never on hardcoded intent names.

Flow per IntentResult:
  side_effects()   — dispatcher.handle(intent, commands): backend POSTs etc.
  try_defer()      — CHITCHAT → engine streams a persona reply;
                     dispatcher.defer_narration() → e.g. RAG-grounded answer.
                     True means "someone else narrates" (skip the ack).
  spoken()         — TIME/DATE engine clock answers, else the dispatcher's
                     local_answer (live telemetry), else the interpolated LLM
                     ack with a clean fallback.
"""
from __future__ import annotations

import datetime as _dt
import re
from typing import Optional

from fsttm.domain import (
    DomainDispatcher, DomainProvider, META_CHITCHAT, META_DATE, META_TIME,
)

# Bracketed placeholders the LLM emits when it lacks a runtime value, e.g.
# "[current temperature]", "<value>". They must never be spoken aloud raw.
_PLACEHOLDER_RE = re.compile(r'[\[<][^\]>]*[\]>]')


def strip_placeholders(text: str) -> Optional[str]:
    """Remove any remaining [..]/<..> placeholder spans from a spoken line and
    tidy the leftover spacing/punctuation. Returns None if nothing meaningful
    remains, so the caller can substitute a clean fallback than speak a
    fragment."""
    if not text or ('[' not in text and '<' not in text):
        return text
    cleaned = _PLACEHOLDER_RE.sub('', text)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    cleaned = re.sub(r'\s+([.,!?])', r'\1', cleaned).strip()
    return cleaned if len(cleaned) >= 3 else None


def _clock_answer(meta: str) -> str:
    """Deterministic TIME/DATE answers — never a hallucinated value."""
    now = _dt.datetime.now()
    if meta == META_TIME:
        # 12-hour, spoken-friendly: "It's 3:42 PM." (strip leading zero)
        return "It's {}.".format(now.strftime('%-I:%M %p'))
    d = now.day
    suffix = ('th' if 11 <= d % 100 <= 13
              else {1: 'st', 2: 'nd', 3: 'rd'}.get(d % 10, 'th'))
    return "It's {}.".format(now.strftime('%A, %B {}{}').format(d, suffix))


class IntentFlow:
    """Provider + dispatcher pair driving one deployment's intent handling."""

    def __init__(self, provider: DomainProvider, dispatcher: DomainDispatcher,
                 *, assistant_name: str = "Nina",
                 last_user_text=lambda: "", narrate_prompt=lambda prompt: None,
                 enabled=None):
        self.provider = provider
        self.dispatcher = dispatcher
        self.assistant_name = assistant_name
        self._last_user_text = last_user_text
        self._narrate_prompt = narrate_prompt
        self.enabled = enabled          # sub-domain subset (None → all)

    # ── classification helpers (display tags + gating) ────────────────────
    def commands(self, intent: dict) -> list:
        try:
            return self.provider.translate(intent or {}, self.enabled) or []
        except Exception:
            return []

    def is_chitchat(self, intent: dict) -> bool:
        return self.provider.meta_intent(intent or {}) == META_CHITCHAT

    def is_deferred_marker(self, intent: dict) -> bool:
        """True when translate() yields a narration marker (e.g. the manual/RAG
        {"cmd": "manual"} command) rather than backend commands only."""
        return any(isinstance(c, dict) and c.get("cmd") == "manual"
                   for c in self.commands(intent))

    # ── side effects (backend commands) ───────────────────────────────────
    def side_effects(self, intent: dict) -> None:
        if not intent:
            return
        try:
            self.dispatcher.handle(intent, self.commands(intent))
        except Exception:
            pass

    # ── narration ─────────────────────────────────────────────────────────
    def try_defer(self, intent: dict) -> bool:
        """CHITCHAT → engine persona reply; else offer the dispatcher the
        chance to narrate (RAG). True = narration handled elsewhere."""
        if self.is_chitchat(intent):
            if self._chitchat_reply():
                return True
            return False   # no utterance to reply to → speak the ack
        try:
            return bool(self.dispatcher.defer_narration(
                intent, self.commands(intent)))
        except Exception:
            return False

    def _chitchat_reply(self) -> bool:
        """A greeting / social remark → a real conversational reply (like chat
        mode), not the canned UNKNOWN deflection. Streams via the narrate
        prompt (ManualGenerate → Response/ResponseDone). Needs the original
        utterance."""
        utter = (self._last_user_text() or "").strip()
        if not utter:
            return False
        persona = (self.provider.chitchat_system(self.assistant_name)
                   or (f"You are {self.assistant_name}, a warm, concise voice "
                       f"assistant. Reply to the user's greeting or remark in "
                       f"ONE short, friendly spoken sentence."))
        prompt = (f"<|system|>\n{persona}<|end|>\n"
                  f"<|user|>\n{utter}<|end|>\n<|assistant|>\n")
        self._narrate_prompt(prompt)
        return True

    def spoken(self, intent: dict, tts_text: str) -> str:
        """The SINGLE source of truth for an intent's spoken/displayed text:
        an engine clock answer (TIME/DATE), else the dispatcher's local answer
        (live telemetry), else the LLM ack with domain placeholders
        interpolated and any leftover placeholder cleaned. Used by BOTH the
        narrator (TTS) and the chat/display feeds so they never diverge."""
        meta = self.provider.meta_intent(intent or {})
        if meta in (META_TIME, META_DATE):
            return _clock_answer(meta)
        try:
            local = self.dispatcher.local_answer(intent or {})
        except Exception:
            local = None
        if local is not None:
            return local
        ack = tts_text or ""
        try:
            ack = self.dispatcher.interpolate(ack)
        except Exception:
            pass
        return "Okay, done." if ('[' in ack or '<' in ack) else ack

    def interpolate_response(self, text: str) -> str:
        """Domain placeholder interpolation + generic cleanup for a full chat
        response (the narrator's ResponseDone path)."""
        try:
            text = self.dispatcher.interpolate(text)
        except Exception:
            pass
        return strip_placeholders(text) or "Okay."
