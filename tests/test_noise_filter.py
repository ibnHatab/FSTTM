"""
whisper._is_noise — reject sound-annotation / hallucination transcripts so a
spurious barge-in (keyboard click, sigh, breath → whisper emits "(…)"/"[…]")
produces NO TextResult, and the narrator auto-resumes.
"""
from fsttm.whisper import _is_noise


def test_blank_and_short_are_noise():
    assert _is_noise("")
    assert _is_noise("   ")
    assert _is_noise("a")          # < 2 chars


def test_bracketed_annotations_are_noise():
    for t in ["[BLANK_AUDIO]", "[Music]", "[Applause]", "[ Silence ]"]:
        assert _is_noise(t), t


def test_parenthesized_annotations_are_noise():
    for t in ["(sighs)", "(keyboard clicking)", "(coughs)", "(breathes)",
              "(SIGHS)", "( keyboard clicking )"]:
        assert _is_noise(t), t


def test_asterisk_annotations_are_noise():
    # *cough* style slipped into the LLM before — must now be dropped.
    for t in ["*cough*", "*sighs*", "*clears throat*", "*COUGH*"]:
        assert _is_noise(t), t


def test_multiple_annotations_only_are_noise():
    assert _is_noise("(sighs) (keyboard clicking)")
    assert _is_noise("[Music] (applause)")
    assert _is_noise("*cough* (sighs)")


def test_config_parasites_overridable():
    from fsttm.whisper import set_parasites, _is_noise as isn
    set_parasites(["okay computer", "uh huh"])
    assert isn("okay computer") and isn("Uh huh.")
    assert not isn("thank you")          # replaced the defaults
    set_parasites(None)                  # keep custom (None = no change)
    assert isn("okay computer")
    set_parasites(["thank you", "thanks"])   # restore for other tests


def test_thank_you_hallucination_is_noise():
    assert _is_noise("thank you")
    assert _is_noise("Thanks.")


def test_real_speech_is_not_noise():
    for t in ["how do I open the trunk", "make it warmer", "Nina",
              "what is two plus two", "yes please"]:
        assert not _is_noise(t), t


def test_annotation_plus_real_speech_is_not_noise():
    # If there's real content alongside an annotation, keep it.
    assert not _is_noise("(sighs) turn on the lights")
    assert not _is_noise("open the window [door slams]")


# ── hard-noise vs parasite split (barge-in confirmation fix) ──────────────────
# Annotations are ALWAYS noise; parasites are emitted-but-flagged so the server
# can keep them when they confirm a real barge-in.

def test_hard_noise_vs_parasite_split():
    from fsttm.whisper import _is_hard_noise, _is_parasite, set_parasites
    set_parasites(["thank you", "thanks"])
    # annotations / short → hard noise, NOT parasite
    assert _is_hard_noise("(sighs)") and not _is_parasite("(sighs)")
    assert _is_hard_noise("[BLANK_AUDIO]")
    assert _is_hard_noise("a")
    # parasite phrases → parasite, NOT hard noise (they're real words)
    assert _is_parasite("thank you") and not _is_hard_noise("thank you")
    assert _is_parasite("Thanks.")
    # real speech → neither
    assert not _is_hard_noise("turn on the lights")
    assert not _is_parasite("turn on the lights")
    set_parasites(["thank you", "thanks"])   # restore
