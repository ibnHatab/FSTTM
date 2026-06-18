import asyncio
import logging as _pylog
import re
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import reactivex as rx
from cyclotron import Component

_log = _pylog.getLogger("fsttm.whisper")   # → fsttm.log

Sink = namedtuple('Sink', ['request'])
Source = namedtuple('Source', ['text'])

Initialize  = namedtuple('Initialize',  ['model', 'with_probs', 'language'])
Initialize.__new__.__defaults__ = (None, False, 'en')

SpeechToText = namedtuple('SpeechToText', ['data', 'context'])

# parasite=True marks a transcript that matched a parasite phrase ("thank you")
# — a likely hallucination on silence, but a real word a user might also speak.
# Whisper still EMITS it (doesn't swallow it) so the server can drop it on idle
# but honour it when it confirms a pending barge-in (real speech energy occurred).
TextResult = namedtuple('TextResult', ['text', 'context', 'parasite'])
TextResult.__new__.__defaults__ = (False,)
TextError  = namedtuple('TextError',  ['error', 'context'])

# Whisper emits sound annotations / hallucinations on noise (keyboard clicks,
# sighs, coughs) or silence, NOT real speech. Dropping them means no TextResult
# fires, so a spurious barge-in is treated as false and the narrator auto-resumes
# (and the annotation never reaches the LLM, e.g. "*cough*").

# Any whole-block annotation: [..], (..), or *..* — covers [Music], (sighs),
# *cough*. Used both to reject annotation-only text and to strip annotations.
_ANNOTATION = re.compile(r'[\[(][^\])]*[\])]|\*[^*]*\*')
_MIN_CHARS = 2   # discard single-character "transcriptions"

# Parasite phrases — literal hallucinations to drop (config-overridable via
# stt.parasites). These are matched as whole-string (case-insensitive), ignoring
# trailing punctuation.
_PARASITES = ["thank you", "thanks"]


def set_parasites(phrases):
    """Replace the parasite-phrase list (from config stt.parasites). None/empty
    keeps the defaults."""
    global _PARASITES
    if phrases:
        _PARASITES = [p.strip().lower() for p in phrases if p.strip()]


def _is_hard_noise(text: str) -> bool:
    """Annotation-only / too-short transcripts that are NEVER real speech —
    "(sighs)", "*cough*", "[BLANK_AUDIO]", single chars. Always dropped."""
    t = text.strip()
    if len(t) < _MIN_CHARS:
        return True
    stripped = _ANNOTATION.sub('', t).strip()
    return len(stripped) < _MIN_CHARS


def _is_parasite(text: str) -> bool:
    """A parasite phrase ("thank you", "thanks") — a likely whisper hallucination
    on silence, but also a real thing a user might say. Dropped CONDITIONALLY by
    the server (idle → drop; confirming a barge-in → keep)."""
    low = text.strip().lower().rstrip('.!?,').strip()
    return low in _PARASITES


def _is_noise(text: str) -> bool:
    """Back-compat: hard noise OR parasite (used by tests / callers that want the
    old unconditional behaviour)."""
    return _is_hard_noise(text) or _is_parasite(text)


def make_driver(loop=None):
    def driver(sink):
        whisper_model = None
        language = 'en'

        # whisper.cpp's CUDA backend is NOT thread-safe: its GPU memory pool
        # asserts on out-of-LIFO frees (ggml-cuda.cu GGML_ASSERT). Two utterances
        # arriving close together would otherwise transcribe concurrently on the
        # default multi-thread executor and corrupt the pool → SIGABRT. Pin all
        # transcribes to a single dedicated worker so they queue and run serially.
        _stt_executor = ThreadPoolExecutor(max_workers=1,
                                           thread_name_prefix='whisper')

        def _resolve_model_path(model_name):
            """Map a whisper model name (e.g. 'base') to a whisper.cpp GGML .bin.

            Accepts a direct path to a .bin, or a short name resolved against
            ~/repo/vox/whisper.cpp/models. Prefers the q5_1 quant, then fp16.
            """
            import os
            if model_name and (model_name.endswith('.bin') or os.path.sep in model_name):
                return os.path.expanduser(model_name)
            base = os.path.expanduser('~/repo/vox/whisper.cpp/models')
            name = model_name or 'base.en'
            if not name.endswith('.en') and name in ('tiny', 'base', 'small', 'medium'):
                name += '.en'   # force English models (config language is 'en')
            for cand in (f'ggml-{name}-q5_1.bin', f'ggml-{name}.bin'):
                p = os.path.join(base, cand)
                if os.path.exists(p):
                    return p
            return os.path.join(base, f'ggml-{name}.bin')

        def setup_model(model_name, lang):
            nonlocal whisper_model, language
            # pywhispercpp's bundled .so libs (libwhisper.so.1, libggml*.so,
            # libggml-cuda.so) land in site-packages, off the default loader path.
            # Preload them RTLD_GLOBAL in dependency order (ggml first, then
            # whisper) so the _pywhispercpp C extension resolves its NEEDED libs.
            import os, sys, ctypes, glob
            sitedirs = [p for p in sys.path if p.endswith('site-packages')]
            def _preload(pattern):
                for d in sitedirs:
                    for so in sorted(glob.glob(os.path.join(d, pattern))):
                        try:
                            ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                        except OSError:
                            pass
            for pat in ('libggml-base.so*', 'libggml-cpu.so*', 'libggml-cuda.so*',
                        'libggml.so*', 'libwhisper.so*'):
                _preload(pat)
            from pywhispercpp.model import Model
            path = _resolve_model_path(model_name)
            print(f"Loading whisper.cpp (CUDA) model: {path} (lang={lang})")
            # greedy (beam 1), single processor, force language — matches the
            # old faster-whisper settings; whisper.cpp runs the encode on GPU.
            whisper_model = Model(path, language=(lang or 'en'),
                                  n_threads=6, print_progress=False,
                                  print_realtime=False, redirect_whispercpp_logs_to=None)
            language = lang or 'en'
            # Warm up CUDA kernels (first encode pays ~1.2s JIT cost); do it now
            # so the first real utterance is fast.
            try:
                whisper_model.transcribe(np.zeros(16000, dtype=np.float32))
            except Exception:
                pass
            print("Whisper ready (whisper.cpp CUDA, warmed)")

        def transcribe_sync(pcm_bytes):
            import time as _t
            audio = np.frombuffer(pcm_bytes, np.int16).astype(np.float32) / 32768.0
            if len(audio) < 16000:
                audio = np.pad(audio, (0, 16000 - len(audio)))
            audio_s = len(audio) / 16000.0
            _t0 = _t.monotonic()
            segs = whisper_model.transcribe(audio)
            text = " ".join(s.text for s in segs).strip()
            stt_ms = (_t.monotonic() - _t0) * 1000
            # RTF = compute time / audio duration; <1 is faster-than-realtime.
            rtf = stt_ms / 1000 / max(audio_s, 0.01)
            # Surface perf to the TUI right panel (no-op import if rich absent);
            # also print so headless runs keep the [stt] line.
            _tui_active = False
            try:
                from fsttm.tui import record_stt_perf, active as _tui_active_fn
                record_stt_perf(stt_ms, audio_s, rtf)
                _tui_active = _tui_active_fn()
            except Exception:
                pass
            if not _tui_active:
                print(f"  [stt] {stt_ms:.0f}ms for {audio_s:.1f}s audio (RTF={rtf:.2f})")
            return text

        def on_subscribe(observer, scheduler):
            def on_whisper_request(item):
                if type(item) is Initialize:
                    setup_model(item.model, item.language)
                elif type(item) is SpeechToText:
                    if whisper_model is not None:
                        async def _transcribe():
                            try:
                                # Serial executor (max_workers=1) → never two
                                # concurrent whisper.cpp CUDA calls.
                                text = await loop.run_in_executor(
                                    _stt_executor, transcribe_sync, item.data
                                )
                                if _is_hard_noise(text):
                                    _log.debug("noise dropped: %r", text)
                                    return   # annotation/short → never real speech
                                parasite = _is_parasite(text)
                                if parasite:
                                    # Emit it flagged — the server keeps it only
                                    # when it confirms a pending barge-in.
                                    _log.debug("parasite (conditional): %r", text)
                                else:
                                    _log.debug("transcript: %r", text)
                                loop.call_soon(observer.on_next,
                                               TextResult(text=text, context=item.context,
                                                          parasite=parasite))
                            except Exception as exc:
                                loop.call_soon(observer.on_next,
                                               TextError(error=exc, context=item.context))

                        asyncio.ensure_future(_transcribe())
                else:
                    observer.on_error(f"Unknown item type: {type(item)}")

            sink.request.subscribe(on_next=on_whisper_request,
                                   on_error=lambda e: observer.on_error(e))

        return Source(text=rx.create(on_subscribe))

    return Component(call=driver, input=Sink)
