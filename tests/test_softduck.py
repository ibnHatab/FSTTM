"""
SoftDuckRouter tests — the soft-duck / pre-roll / utterance-delimiter logic that
sits between the VAD and STT. Pure (no audio, no reactivex), so the barge-in
frame routing is deterministically testable.

Frame convention here: a non-None "frame" is any speech payload (we use small
byte strings as stand-ins); None is the VAD utterance delimiter.
"""
from fsttm.perception import SoftDuckRouter, SpeechDuringPlayback

F = lambda n: bytes([n])          # a stand-in speech frame


def _route(router, frames):
    """Feed a list of frames (bytes or None) → flat list of emitted events."""
    out = []
    for f in frames:
        out.extend(router.on_frame(f))
    return out


def test_not_ducked_passes_frames_and_delimiters():
    r = SoftDuckRouter()
    out = _route(r, [F(1), F(2), None, F(3), None])
    assert out == [F(1), F(2), None, F(3), None]      # forwarded verbatim


def test_ducked_speech_emits_sentinel_not_frame():
    r = SoftDuckRouter()
    r.soft_duck()
    out = _route(r, [F(1), F(2)])
    # only barge-in sentinels reach STT-side; the frames are withheld (pre-roll)
    assert all(isinstance(e, SpeechDuringPlayback) for e in out)
    assert len(out) == 2


def test_ducked_silence_emits_nothing():
    r = SoftDuckRouter()
    r.soft_duck()
    assert _route(r, [None, None]) == []


def test_unduck_flushes_preroll_onset():
    r = SoftDuckRouter()
    r.soft_duck()
    _route(r, [F(1), F(2), F(3)])     # speech during ducking → pre-roll
    flushed = r.unduck()
    assert flushed == [F(1), F(2), F(3)]   # onset recovered, in order
    assert r.ducked is False


def test_ducked_utterances_do_not_merge():
    # THE BUG: utterance 1 (during ducking) must not glue to utterance 2.
    # A None while ducked with buffered speech flushes utt-1 + a delimiter, so
    # downstream sees two separate utterances.
    r = SoftDuckRouter()
    r.soft_duck()
    out = _route(r, [F(1), F(2), None,      # utterance 1 ends
                     F(3), F(4)])           # utterance 2 starts (still ducked)
    # strip the sentinels — look at the frame/None stream that reaches STT
    stt = [e for e in out if not isinstance(e, SpeechDuringPlayback)]
    assert stt == [F(1), F(2), None]        # utt-1 flushed WITH its delimiter
    # utt-2 still buffered; flush on unduck
    assert r.unduck() == [F(3), F(4)]


def test_preroll_is_bounded():
    r = SoftDuckRouter(preroll_frames=3)
    r.soft_duck()
    _route(r, [F(i) for i in range(10)])    # 10 frames, keep last 3
    assert r.unduck() == [F(7), F(8), F(9)]


def test_soft_duck_clears_stale_preroll():
    r = SoftDuckRouter()
    r.soft_duck()
    _route(r, [F(1)])
    r.soft_duck()                            # new response → fresh window
    assert r.unduck() == []


def test_full_barge_in_cycle():
    # live → system speaks (soft_duck) → user barges in → unduck → live again
    r = SoftDuckRouter()
    assert _route(r, [F(1), None]) == [F(1), None]     # user utterance pre-TTS
    r.soft_duck()                                       # TTS starts
    barge = _route(r, [F(2), F(3)])                     # user interrupts
    assert all(isinstance(e, SpeechDuringPlayback) for e in barge)
    assert r.unduck() == [F(2), F(3)]                   # onset recovered
    assert _route(r, [F(4), None]) == [F(4), None]      # rest flows live
