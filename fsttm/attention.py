"""
Attention layer — wake-word / sleep state on top of the FSTTM turn-taking model.

Humans separate *getting attention* from *giving instructions* from *ending the
conversation*. This module models that as a small state machine that sits ABOVE
the N09-1071 floor FSM and gates whether a transcribed utterance is dispatched to
the LLM.

States
------
  ASLEEP  — only the wake word is acted on; everything else is ignored.
  AWAKE   — utterances are dispatched to the LLM as commands.

(A future MUTED state — "hard off", re-armed only by an explicit phrase — can be
added; for now sleep/wake covers the requirement.)

Detection
---------
- Wake: cheap text match on the STT transcript (the existing VAD→whisper pipeline
  keeps running while ASLEEP; only the LLM is gated). "hey nina", "nina …".
- Sleep: ONLY when ``system.sleep_intent`` is on — a real LLM intent classifier
  decides (handled in server.py via the system-intent grammar). With sleep_intent
  off the system, once woken, stays AWAKE (there is no phrase-based sleep).

Everything is config-driven (``config.system``): ``attention`` toggles the whole
layer (off → always-AWAKE, today's behaviour); ``sleep_intent`` enables
LLM-classified sleep. Transitions are silent (no TTS); the TUI shows the state.
"""
import re

ASLEEP = "ASLEEP"
AWAKE = "AWAKE"


def _norm(text):
    """Lowercase, strip punctuation/extra space for robust phrase matching."""
    return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())


class Attention:
    def __init__(self, enabled=True, wake_words=None, name="nina",
                 start_asleep=True):
        # When disabled, the layer is a no-op: always AWAKE, wake/sleep ignored.
        self.enabled = enabled
        self.name = (name or "nina").lower()
        self.wake_words = [w.lower() for w in (wake_words or
                                               ["nina", "hey nina", "hi nina"])]
        self.state = (ASLEEP if (enabled and start_asleep) else AWAKE)

    # ── predicates ────────────────────────────────────────────────────────────
    @property
    def awake(self):
        return (not self.enabled) or self.state == AWAKE

    @property
    def asleep(self):
        return self.enabled and self.state == ASLEEP

    # ── wake detection (text match) ─────────────────────────────────────────────
    def match_wake(self, text):
        """Return (matched, remainder). remainder is the text AFTER the wake word
        (so "nina what's the weather" → command "what's the weather"); empty if
        the utterance was only the wake word."""
        t = _norm(text)
        for w in sorted((_norm(w) for w in self.wake_words),
                        key=len, reverse=True):  # longest first
            # wake word as a whole-word prefix or anywhere as a standalone token
            m = re.search(r"\b" + re.escape(w) + r"\b", t)
            if m:
                remainder = (t[:m.start()] + " " + t[m.end():]).strip()
                return True, remainder
        return False, ""

    def wake(self):
        self.state = AWAKE

    def sleep(self):
        self.state = ASLEEP

    # ── one-call gate used by the server ────────────────────────────────────────
    def on_utterance(self, text, sleep_intent=False):
        """Decide what to do with a transcribed utterance.

        Returns a dict:
          {'action': 'wake'|'command'|'ignore', 'text': <text for the LLM>,
           'wake_prefixed': <bool — utterance addressed the assistant by name>}

        - ASLEEP: wake word → 'wake' (+ the utterance as first command if it had
                  text after the wake word); else 'ignore'.
        - AWAKE : always 'command'. Going back to sleep is the LLM classifier's
                  job (server.py, when sleep_intent is true) — but ONLY when the
                  utterance is wake-prefixed ('wake_prefixed'). A bare command
                  like "stop climate" (no "Nina") must NEVER be able to disable
                  voice control just because STT garbled it into a sleep phrase.
        When the layer is disabled, everything is a 'command'.
        """
        if not self.enabled:
            return {"action": "command", "text": text, "wake_prefixed": False}

        if self.state == ASLEEP:
            matched, remainder = self.match_wake(text)
            if matched:
                self.wake()
                # Keep the wake word in the text we hand the LLM so the model
                # (told its name is Nina) has the full utterance; if there was a
                # command after the wake word, that's the first command.
                return {"action": "wake", "text": text if remainder else "",
                        "wake_prefixed": True}
            return {"action": "ignore", "text": "", "wake_prefixed": False}

        # AWAKE → always a command. Surface whether the user addressed the
        # assistant by name so the server only considers sleep/mute then.
        matched, _ = self.match_wake(text)
        return {"action": "command", "text": text, "wake_prefixed": matched}
