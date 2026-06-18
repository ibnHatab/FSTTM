"""
Attention state-machine tests (fsttm/attention.py).

Pure logic — no model, no audio. Covers wake-word matching, sleep transitions,
the disabled (always-awake) path, and the on_utterance gate the server uses.
"""
from fsttm.attention import Attention, ASLEEP, AWAKE


def _a(**kw):
    kw.setdefault("enabled", True)
    return Attention(**kw)


# ── disabled = always awake, no gating ────────────────────────────────────────

def test_disabled_is_always_awake():
    a = Attention(enabled=False)
    assert a.awake and not a.asleep
    d = a.on_utterance("anything at all")
    assert d["action"] == "command" and d["text"] == "anything at all"


# ── start state ───────────────────────────────────────────────────────────────

def test_enabled_starts_asleep():
    a = _a()
    assert a.state == ASLEEP and a.asleep


# ── wake detection ────────────────────────────────────────────────────────────

def test_wake_bare_word():
    a = _a()
    d = a.on_utterance("Nina")
    assert d["action"] == "wake" and a.state == AWAKE
    assert d["text"] == ""          # bare wake word → no trailing command


def test_wake_with_trailing_command():
    a = _a()
    d = a.on_utterance("Nina, what's the weather tomorrow?")
    assert d["action"] == "wake" and a.state == AWAKE
    # full utterance handed to the LLM (model knows its name is Nina)
    assert "weather" in d["text"]


def test_wake_phrase_hey_nina():
    a = _a()
    d = a.on_utterance("hey nina turn on the lights")
    assert d["action"] == "wake" and a.state == AWAKE
    assert "lights" in d["text"]


def test_asleep_ignores_non_wake():
    a = _a()
    d = a.on_utterance("what's the weather")
    assert d["action"] == "ignore" and a.state == ASLEEP


def test_wake_is_fuzzy_to_punctuation_and_case():
    a = _a()
    assert a.on_utterance("NINA!!!")["action"] == "wake"


def test_wake_not_triggered_by_substring():
    # "ninang" should not match the whole word "nina"
    a = _a()
    assert a.on_utterance("the ninang festival")["action"] == "ignore"


# ── awake: always a command; sleep is NOT phrase-based ────────────────────────

def test_awake_always_command_no_phrase_sleep():
    a = _a()
    a.wake()
    # Even an obvious "sleep phrase" stays a command — only the LLM classifier
    # (server side) may sleep, and only when sleep_intent is on.
    for txt in ["that's all", "go to sleep", "set a timer for ten minutes"]:
        d = a.on_utterance(txt, sleep_intent=False)
        assert d["action"] == "command", txt
        assert a.state == AWAKE


def test_command_text_passthrough():
    a = _a()
    a.wake()
    d = a.on_utterance("set a timer for ten minutes")
    assert d["text"].startswith("set a timer")


def test_sleep_intent_on_still_returns_command_here():
    # With sleep_intent on, on_utterance returns 'command' (the server then runs
    # the LLM classifier to decide command vs sleep). State is unchanged here.
    a = _a()
    a.wake()
    d = a.on_utterance("that's all for now", sleep_intent=True)
    assert d["action"] == "command"
    assert a.state == AWAKE


# ── explicit sleep()/wake() round-trip (driven by the server's classifier) ────

def test_sleep_then_requires_wake_again():
    a = _a()
    a.on_utterance("nina")                 # wake
    assert a.state == AWAKE
    a.sleep()                              # server slept us (intent classifier)
    assert a.state == ASLEEP
    # now ignored until wake again
    assert a.on_utterance("are you there")["action"] == "ignore"
    assert a.on_utterance("hey nina")["action"] == "wake"


def test_custom_name_and_wake_words():
    a = Attention(enabled=True, name="Jarvis", wake_words=["jarvis"])
    assert a.on_utterance("jarvis")["action"] == "wake"
    assert a.state == AWAKE
    assert a.on_utterance("anything at all")["action"] == "command"


# ── wake_prefixed gate: sleep/mute only when addressed by name ────────────────
# A bare command (no wake word) must NEVER be eligible for sleep/mute, even if
# STT garbles it into a sleep-like phrase. "Hey Nina, voice off" is the contract.

def test_awake_bare_command_not_wake_prefixed():
    a = _a()
    a.wake()
    d = a.on_utterance("stop the climate")        # user meant "stop climate"
    assert d["action"] == "command"
    assert d["wake_prefixed"] is False            # → server skips sleep classifier

def test_awake_named_command_is_wake_prefixed():
    a = _a()
    a.wake()
    d = a.on_utterance("hey nina voice off")
    assert d["action"] == "command"
    assert d["wake_prefixed"] is True             # → server runs sleep classifier

def test_garbled_sleepish_bare_phrase_not_wake_prefixed():
    a = _a()
    a.wake()
    # whisper turned "stop, climate" into "stop, glimar" — still no wake word
    assert a.on_utterance("stop glimar")["wake_prefixed"] is False
    assert a.on_utterance("that's all")["wake_prefixed"] is False

def test_asleep_and_disabled_report_wake_prefixed():
    a = _a()                                       # ASLEEP
    assert a.on_utterance("hey nina")["wake_prefixed"] is True
    assert a.on_utterance("random talk")["wake_prefixed"] is False
    off = Attention(enabled=False)
    assert off.on_utterance("anything")["wake_prefixed"] is False
