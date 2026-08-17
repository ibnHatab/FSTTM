"""Voice-filter tests: disabled passthrough, enforce/shadow semantics, and
verifier math with a monkeypatched extractor (no model download needed).
A live sherpa-onnx path runs only when a model file is present locally."""
import asyncio
import os

import numpy as np
import pytest
from reactivex.subject import Subject

from fsttm.voicefilter import driver as vf
from fsttm.voicefilter.speaker import FilterResult, SpeakerVerifier


def _run(events, timeout=2.0):
    out = []
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        comp = vf.make_driver(loop)
        subject = Subject()
        src = comp.call(vf.Sink(request=subject))
        src.result.subscribe(on_next=out.append)

        async def scenario():
            for ev in events:
                subject.on_next(ev)
            for _ in range(20):
                await asyncio.sleep(0.01)

        loop.run_until_complete(asyncio.wait_for(scenario(), timeout))
    finally:
        loop.close()
    return out


def test_disabled_is_synchronous_passthrough():
    out = _run([vf.Initialize(cfg={'enabled': False}),
                vf.Filter(data=b'\x00\x01' * 100, context='c1')])
    assert [type(e).__name__ for e in out] == ["Accepted"]
    assert out[0].data == b'\x00\x01' * 100 and out[0].context == 'c1'


def test_no_initialize_still_passes():
    out = _run([vf.Filter(data=b'ab', context=None)])
    assert [type(e).__name__ for e in out] == ["Accepted"]


def test_failed_load_degrades_to_passthrough():
    out = _run([vf.Initialize(cfg={'enabled': True, 'provider': 'speaker',
                                   'model': '/nonexistent.onnx',
                                   'profiles': '/nonexistent.npz'}),
                vf.Filter(data=b'ab', context=None)])
    assert [type(e).__name__ for e in out] == ["Accepted"]


class _StubVerifier:
    """Accepts iff the utterance starts with the byte 0x01."""
    def load(self, cfg):
        pass

    def check(self, pcm, rate):
        ok = pcm[:1] == b'\x01'
        return FilterResult(accepted=ok, score=0.9 if ok else 0.1,
                            speaker='axadmin')


def test_enforce_drops_rejected(monkeypatch):
    monkeypatch.setattr(vf, '_load_provider', lambda name: _StubVerifier())
    out = _run([vf.Initialize(cfg={'enabled': True}),
                vf.Filter(data=b'\x01match', context='a'),
                vf.Filter(data=b'\x00nomatch', context='b')])
    kinds = {e.context if hasattr(e, 'context') else None: type(e).__name__
             for e in out}
    assert kinds['a'] == 'Accepted'
    assert kinds['b'] == 'Rejected'
    rej = [e for e in out if type(e).__name__ == 'Rejected'][0]
    assert rej.score == pytest.approx(0.1) and rej.speaker == 'axadmin'


def test_shadow_never_drops(monkeypatch):
    monkeypatch.setattr(vf, '_load_provider', lambda name: _StubVerifier())
    out = _run([vf.Initialize(cfg={'enabled': True, 'mode': 'shadow'}),
                vf.Filter(data=b'\x00nomatch', context='b')])
    assert [type(e).__name__ for e in out] == ["Accepted"]


# ── verifier math (extractor monkeypatched) ──────────────────────────────────

def _mk_verifier(tmp_path, monkeypatch, emb_fn, threshold=0.45):
    profiles = {'alice': np.array([1.0, 0.0], np.float32),
                'bob': np.array([0.0, 1.0], np.float32)}
    p = str(tmp_path / 'profiles.npz')
    np.savez(p, **profiles)
    monkeypatch.setattr('fsttm.voicefilter.speaker.make_extractor',
                        lambda model: object())
    monkeypatch.setattr('fsttm.voicefilter.speaker.embed',
                        lambda extractor, pcm, rate: emb_fn(pcm))
    v = SpeakerVerifier()
    v.load({'model': 'x.onnx', 'profiles': p, 'threshold': threshold,
            'min_utterance_s': 0.5})
    return v


def test_verifier_picks_best_profile(tmp_path, monkeypatch):
    v = _mk_verifier(tmp_path, monkeypatch,
                     lambda pcm: np.array([0.9, 0.1], np.float32))
    res = v.check(b'\x00' * 32000, 16000)     # 1 s ≥ min_utterance
    assert res.accepted and res.speaker == 'alice'
    assert res.score > 0.9


def test_verifier_rejects_below_threshold(tmp_path, monkeypatch):
    v = _mk_verifier(tmp_path, monkeypatch,
                     lambda pcm: np.array([0.5, 0.5], np.float32),
                     threshold=0.9)
    res = v.check(b'\x00' * 32000, 16000)
    assert not res.accepted


def test_short_utterance_bypasses(tmp_path, monkeypatch):
    v = _mk_verifier(tmp_path, monkeypatch,
                     lambda pcm: np.array([0.0, 0.0], np.float32))
    res = v.check(b'\x00' * 2000, 16000)      # 62 ms < 0.5 s
    assert res.accepted and res.score != res.score   # NaN score


# ── live sherpa-onnx (gated on a local model file) ───────────────────────────

_MODEL = os.path.join(os.path.dirname(__file__), '..', 'models', 'speaker')


@pytest.mark.skipif(
    not (os.path.isdir(_MODEL) and any(f.endswith('.onnx')
                                       for f in os.listdir(_MODEL))),
    reason="no speaker-embedding model under models/speaker/")
def test_live_embedding_shape():
    import glob
    from fsttm.voicefilter.speaker import embed, make_extractor
    model = sorted(glob.glob(os.path.join(_MODEL, '*.onnx')))[0]
    ex = make_extractor(model)
    emb = embed(ex, b'\x00\x01' * 16000, 16000)
    assert emb.ndim == 1 and emb.size >= 128
