"""
LLM driver using llama-cpp-python bindings (direct, no subprocess).

Thread safety: llama-cpp-python / CUDA are NOT thread-safe. A single
serialised inference thread (_worker) processes one Generate at a time.
Concurrent requests are dropped (queue depth 1) to keep latency low.
StopGenerate sets _stop_event; the worker checks it between tokens so
C++ inference exits at the next stop-token or max_tokens boundary.
"""
from __future__ import annotations  # PEP 563: lazy annotations for Python 3.8 (list[]/tuple[])
import logging as _pylog
import queue
import threading
from collections import namedtuple

import reactivex as rx
from cyclotron import Component

from fsttm.utils import ignoreStderr

_log = _pylog.getLogger("fsttm.llama")   # → fsttm.log (propagates to fsttm root)

# Bind llama_cpp's OWN bundled libggml/libllama at import time, BEFORE
# fsttm.whisper preloads pywhispercpp's ggml RTLD_GLOBAL at Initialize.
# Both ship distinct ggml builds; if whisper's ggml claims the global
# symbol namespace first, libllama resolves against it and aborts
# (undefined symbol: ggml_backend_cuda_log_set_callback). Importing here
# makes llama deterministically win, regardless of model warm order.
import llama_cpp  # noqa: F401  (import for side-effect: dlopen libllama first)

Sink = namedtuple('Sink', ['request'])
Source = namedtuple('Source', ['system'])

Initialize      = namedtuple('Initialize',
                             ['model_path', 'n_ctx', 'n_batch',
                              'n_threads', 'n_gpu_layers'])
# n_ctx/n_batch from config — the intent base prompt (system + domain prompt +
# few-shot) easily exceeds the old hardcoded 2048; too-small n_ctx made the very
# first model.eval() fail with `llama_decode returned 1`.
Initialize.__new__.__defaults__ = (None, 2048, 512, 6, 99)
Generate        = namedtuple('Generate',        ['text', 'context'])
Generate.__new__.__defaults__ = (None, None)
IntentGenerate  = namedtuple('IntentGenerate',  ['text', 'context', 'domains'])
IntentGenerate.__new__.__defaults__ = (None, None, None)  # domains None → all
# ClassifySystem: grammar-constrained classification of an utterance into a
# system action {command, sleep, mute} — used by the attention layer's
# sleep_intent path. Does NOT touch conversation history.
ClassifySystem  = namedtuple('ClassifySystem',  ['text', 'context'])
ClassifySystem.__new__.__defaults__ = (None, None)
# ManualGenerate: one-shot grounded answer from a fully-formed prompt (RAG
# manual context already baked in by the server). Bypasses conversation history;
# emits ResponseDone so the narrator speaks it like any reply.
ManualGenerate  = namedtuple('ManualGenerate',  ['prompt', 'context'])
ManualGenerate.__new__.__defaults__ = (None, None)
StopGenerate    = namedtuple('StopGenerate', [])
AddSystem       = namedtuple('AddSystem',    ['prompt'])
AddSystem.__new__.__defaults__ = (None,)
# TrimHistory: after a barge-in, rewrite the last assistant turn to only the
# text the user actually heard (the confirmed-done checkpoints), so the model's
# memory matches what was spoken rather than the full generated reply.
TrimHistory     = namedtuple('TrimHistory',  ['heard_text'])
TrimHistory.__new__.__defaults__ = (None,)

Response        = namedtuple('Response',     ['text',        'context'])
Response.__new__.__defaults__ = (None, None)
ResponseDone    = namedtuple('ResponseDone', ['full_text',   'context'])
ResponseDone.__new__.__defaults__ = (None, None)
IntentResult    = namedtuple('IntentResult', ['intent_json', 'tts_text', 'context'])
IntentResult.__new__.__defaults__ = (None, None, None)
SystemIntent    = namedtuple('SystemIntent', ['action', 'context'])  # command|sleep|mute
SystemIntent.__new__.__defaults__ = (None, None)
LlamaError      = namedtuple('LlamaError',   ['error',       'context'])

# Stop tokens that cover plain-text role markers the model sometimes emits
_EXTRA_STOP = ["\nUser:", "\nAssistant:", "User:", "\n\n\n"]


def _build_prompt(model_path: str, sys: str, user_text: str) -> tuple[str, list[str]]:
    mp = model_path.lower()
    if "phi-3" in mp or "phi3" in mp:
        prompt = (
            f"<|system|>\n{sys}<|end|>\n"
            f"<|user|>\n{user_text}<|end|>\n"
            f"<|assistant|>\n"
        )
        stop = ["<|end|>", "<|user|>"] + _EXTRA_STOP
    elif "llama-3" in mp or "llama3" in mp:
        prompt = (
            f"<|start_header_id|>system<|end_header_id|>\n{sys}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n{user_text}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )
        stop = ["<|eot_id|>", "<|end_of_text|>"] + _EXTRA_STOP
    else:
        # Phi-4-mini reasoning
        prompt = (
            f"<|system|>{sys}<|end|>"
            f"<|user|>{user_text}<|end|>"
            f"<|assistant|>"
        )
        stop = ["<|end|>", "<|user|>", "<|system|>"] + _EXTRA_STOP
    return prompt, stop


class ConversationHistory:
    """
    Application-level FIFO context window.

    Maintains a list of (user, assistant) turn pairs. When the estimated token
    count of system_prompt + history + new_user would exceed `ctx_threshold`
    (fraction of n_ctx), the oldest turns are dropped — FIFO — until it fits.
    The system prompt is always preserved (n_keep behaviour).

    Token estimate: 1 token ≈ 4 characters (rough but fast; no tokenizer call).
    """
    def __init__(self, n_ctx: int = 2048, ctx_threshold: float = 0.80):
        self.n_ctx          = n_ctx
        self.ctx_threshold  = ctx_threshold
        self.threshold_toks = int(n_ctx * ctx_threshold)
        self._turns: list[tuple[str, str]] = []   # [(user, assistant), ...]

    def _est_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def total_tokens(self, sys_prompt: str) -> int:
        t = self._est_tokens(sys_prompt)
        for u, a in self._turns:
            t += self._est_tokens(u) + self._est_tokens(a)
        return t

    def add_turn(self, user: str, assistant: str) -> None:
        self._turns.append((user, assistant))

    def trim_for(self, sys_prompt: str, new_user: str) -> int:
        """Drop oldest turns until sys+history+new_user fits. Returns turns dropped."""
        dropped = 0
        budget  = self.threshold_toks - self._est_tokens(sys_prompt) \
                                       - self._est_tokens(new_user) \
                                       - 64   # generation headroom
        while self._turns:
            used = sum(self._est_tokens(u) + self._est_tokens(a) for u, a in self._turns)
            if used <= budget:
                break
            self._turns.pop(0)
            dropped += 1
        return dropped

    def build_chat_prompt(self, model_path: str, sys_prompt: str,
                          new_user: str) -> tuple[str, list[str]]:
        """Build full multi-turn prompt, trimming context if needed."""
        dropped = self.trim_for(sys_prompt, new_user)
        if dropped:
            print(f"  [ctx-fifo] dropped {dropped} oldest turn(s) to fit context")

        mp = model_path.lower()
        turns_text = ""
        if "phi-3" in mp or "phi3" in mp:
            for u, a in self._turns:
                turns_text += f"<|user|>\n{u}<|end|>\n<|assistant|>\n{a}<|end|>\n"
            prompt = (
                f"<|system|>\n{sys_prompt}<|end|>\n"
                + turns_text
                + f"<|user|>\n{new_user}<|end|>\n<|assistant|>\n"
            )
            stop = ["<|end|>", "<|user|>"] + _EXTRA_STOP
        elif "llama-3" in mp or "llama3" in mp:
            for u, a in self._turns:
                turns_text += (
                    f"<|start_header_id|>user<|end_header_id|>\n{u}<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n{a}<|eot_id|>"
                )
            prompt = (
                f"<|start_header_id|>system<|end_header_id|>\n{sys_prompt}<|eot_id|>"
                + turns_text
                + f"<|start_header_id|>user<|end_header_id|>\n{new_user}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n"
            )
            stop = ["<|eot_id|>", "<|end_of_text|>"] + _EXTRA_STOP
        else:
            for u, a in self._turns:
                turns_text += f"<|user|>{u}<|end|><|assistant|>{a}<|end|>"
            prompt = (
                f"<|system|>{sys_prompt}<|end|>"
                + turns_text
                + f"<|user|>{new_user}<|end|><|assistant|>"
            )
            stop = ["<|end|>", "<|user|>", "<|system|>"] + _EXTRA_STOP

        return prompt, stop

    def context_fill_pct(self, sys_prompt: str) -> float:
        return self.total_tokens(sys_prompt) / self.n_ctx * 100

    def turn_count(self) -> int:
        return len(self._turns)

    def clear(self) -> None:
        self._turns.clear()


def make_driver(loop=None):
    # grammar and two_pass imported lazily so the module loads without GPU
    def driver(sink):
        model       = None
        sys_prompt  = ""
        _stop_event = threading.Event()
        history     = ConversationHistory(n_ctx=2048)
        # Depth-1 queue: a new Generate drops any pending unstarted request
        _req_queue: queue.Queue = queue.Queue(maxsize=1)

        # ── intent two-pass handler ───────────────────────────────────────
        def _handle_intent(item: IntentGenerate):
            from fsttm.grammar import make_hvac_grammar, TTS_TRANSLATION_EXAMPLES
            from fsttm.two_pass import approach_a
            _log.debug("intent dispatch: text=%r domains=%s", item.text, item.domains)
            try:
                grammar = make_hvac_grammar(item.domains)
            except Exception as exc:
                _log.exception("make_hvac_grammar failed (domains=%s)", item.domains)
                loop.call_soon_threadsafe(observer.on_next, LlamaError(error=exc, context=item.context))
                return
            try:
                intent, tts, tj, tt = approach_a(
                    model, sys_prompt, item.text, grammar
                )
                _log.info("intent OK: JSON=%.0fms TTS=%.0fms intent=%r",
                          tj, tt, intent)
                try:   # surface the split timing to the TUI (regression watch)
                    from fsttm.tui import record_intent_perf
                    record_intent_perf(tj, tt)
                except Exception:
                    pass
            except Exception as exc:
                intent, tts = None, ""
                _log.exception("approach_a failed for %r", item.text)
                loop.call_soon_threadsafe(observer.on_next, LlamaError(error=exc, context=item.context))
                return
            loop.call_soon_threadsafe(
                observer.on_next,
                IntentResult(intent_json=intent, tts_text=tts, context=item.context),
            )

        # ── system-intent classifier (attention sleep_intent) ─────────────
        def _handle_classify(item):
            """Grammar-constrained one-shot classification into a system action.
            Stateless: does not read or mutate conversation history."""
            import json as _json
            from fsttm.grammar import make_system_grammar, SYSTEM_INTENT_PROMPT
            action = "command"
            try:
                grammar = make_system_grammar()
                prompt = (f"{SYSTEM_INTENT_PROMPT}\n\nUser: {item.text}\nJSON: ")
                out = model.create_completion(
                    prompt, max_tokens=16, temperature=0.0,
                    grammar=grammar, stream=False,
                )
                txt = out["choices"][0]["text"].strip()
                action = _json.loads(txt).get("action", "command")
            except Exception as exc:
                loop.call_soon_threadsafe(
                    observer.on_next, LlamaError(error=exc, context=item.context))
            loop.call_soon_threadsafe(
                observer.on_next,
                SystemIntent(action=action, context=item.context))

        # ── manual RAG answer (one-shot, history-free) ────────────────────
        def _handle_manual(item):
            """Generate a grounded answer from a fully-formed prompt (RAG context
            already in item.prompt). Streams Response tokens + ResponseDone so the
            narrator speaks it; does NOT touch conversation history."""
            _stop_event.clear()
            ctx = item.context
            acc = []
            try:
                for chunk in model.create_completion(
                        item.prompt, max_tokens=120, temperature=0.2,
                        top_k=40, top_p=0.9, stream=True,
                        stop=["\n", "Question:", "Manual excerpts:",
                              "Spoken answer:"]):
                    if _stop_event.is_set():
                        break
                    tok = chunk["choices"][0]["text"]
                    if not tok:
                        continue
                    acc.append(tok)
                    loop.call_soon_threadsafe(
                        observer.on_next, Response(text=tok, context=ctx))
            except Exception as exc:
                loop.call_soon_threadsafe(
                    observer.on_next, LlamaError(error=exc, context=ctx))
            finally:
                loop.call_soon_threadsafe(
                    observer.on_next,
                    ResponseDone(full_text="".join(acc), context=ctx))

        def _reprime_after_completion():
            """A create_completion (classify/manual/chat) resets the KV cache and
            wipes the intent prefix → the next intent command pays the ~11s
            re-prime. Re-warm it NOW, but ONLY if no request is queued (so we use
            the idle gap while the user listens to the answer, never delay a
            pending command). Intent mode only (long prefix)."""
            if model is None or not sys_prompt or not _req_queue.empty():
                return
            try:
                from fsttm.two_pass import _prime_prefix
                _prime_prefix(model, sys_prompt)
            except Exception:
                pass

        # ── single serialised inference worker ────────────────────────────
        def _worker():
            nonlocal model
            while True:
                item = _req_queue.get()
                if item is None:
                    break          # sentinel: shut down worker
                if model is None:
                    continue
                if isinstance(item, IntentGenerate):
                    _handle_intent(item)
                    continue
                if isinstance(item, ClassifySystem):
                    _handle_classify(item)
                    _reprime_after_completion()
                    continue
                if isinstance(item, ManualGenerate):
                    _handle_manual(item)
                    _reprime_after_completion()
                    continue
                if not isinstance(item, Generate):
                    continue

                _stop_event.clear()
                ctx   = item.context
                mp    = getattr(model, 'model_path', '')
                # FIFO context: use conversation history, trim if needed
                prompt, stop = history.build_chat_prompt(mp, sys_prompt, item.text)
                fill = history.context_fill_pct(sys_prompt)
                if fill > 50:
                    print(f"  [ctx] {fill:.0f}% full, {history.turn_count()} turns")

                accumulated = []
                in_think    = False
                try:
                    for chunk in model.create_completion(
                        prompt,
                        max_tokens=80,       # short answers for voice
                        temperature=0.7,
                        top_k=40,
                        top_p=0.95,
                        repeat_penalty=1.1,
                        stream=True,
                        stop=stop,
                    ):
                        if _stop_event.is_set():
                            break
                        tok = chunk["choices"][0]["text"]
                        if not tok:
                            continue
                        # strip Phi-4 <think> blocks
                        if "<think>" in tok:
                            in_think = True
                        if "</think>" in tok:
                            in_think = False
                            tok = tok.split("</think>", 1)[-1]
                            if not tok:
                                continue
                        if in_think:
                            continue

                        accumulated.append(tok)
                        loop.call_soon_threadsafe(
                            observer.on_next,
                            Response(text=tok, context=ctx),
                        )
                except Exception as exc:
                    loop.call_soon_threadsafe(
                        observer.on_next,
                        LlamaError(error=exc, context=ctx),
                    )
                finally:
                    full_reply = "".join(accumulated)
                    # Record turn in FIFO history (even partial/interrupted)
                    if full_reply.strip():
                        history.add_turn(item.text, full_reply.strip())
                    loop.call_soon_threadsafe(
                        observer.on_next,
                        ResponseDone(full_text=full_reply, context=ctx),
                    )
                    _reprime_after_completion()   # chat reset the KV → re-warm

        worker_thread = threading.Thread(target=_worker, daemon=True, name="llama-worker")
        worker_thread.start()

        def on_subscribe(obs, scheduler):
            nonlocal model, sys_prompt, observer
            observer = obs   # capture for worker closure

            def on_request(item):
                nonlocal model, sys_prompt
                if type(item) is Initialize:
                    try:
                        from llama_cpp import Llama
                        print(f"Loading Llama model: {item.model_path} "
                              f"(n_ctx={item.n_ctx}, n_batch={item.n_batch})")
                        _log.info("loading Llama: %s n_ctx=%s n_batch=%s "
                                  "n_threads=%s n_gpu_layers=%s", item.model_path,
                                  item.n_ctx, item.n_batch, item.n_threads,
                                  item.n_gpu_layers)
                        with ignoreStderr():
                            model = Llama(
                                model_path=item.model_path,
                                n_ctx=item.n_ctx,
                                n_batch=item.n_batch,
                                n_threads=item.n_threads,
                                n_gpu_layers=item.n_gpu_layers,
                                verbose=False,
                            )
                        model.model_path = item.model_path
                        history.n_ctx = model.n_ctx()
                        history.threshold_toks = int(model.n_ctx() * 0.80)
                        print("Llama model ready")
                    except Exception as exc:
                        loop.call_soon_threadsafe(
                            obs.on_next, LlamaError(error=exc, context=None)
                        )
                elif type(item) is AddSystem:
                    sys_prompt = item.prompt or ""
                    # Pre-warm: eval the intent system prefix into the KV cache NOW
                    # (at startup, before any request) so the FIRST command hits the
                    # warm cache (~150ms) instead of paying the ~5s prime. Only
                    # meaningful in intent mode (long prompt); harmless otherwise.
                    if model is not None and sys_prompt:
                        try:
                            from fsttm.two_pass import _prime_prefix
                            import time as _t
                            _t0 = _t.monotonic()
                            n = _prime_prefix(model, sys_prompt)
                            if n:
                                print(f"Intent prefix pre-warmed: {n} tok in "
                                      f"{(_t.monotonic()-_t0)*1000:.0f}ms")
                                _log.info("intent prefix pre-warmed: %d tok in %.0fms",
                                          n, (_t.monotonic()-_t0)*1000)
                        except Exception as exc:
                            _log.debug("prefix pre-warm skipped: %s", exc)
                elif type(item) is TrimHistory:
                    # Barge-in rollback: replace the last assistant turn with
                    # only the checkpoints the user actually heard. Empty heard
                    # text → "..." so the turn structure stays intact.
                    if history._turns and item.heard_text is not None:
                        user, _ = history._turns[-1]
                        history._turns[-1] = (user, item.heard_text.strip() or "...")
                elif type(item) is StopGenerate:
                    _stop_event.set()
                    # drain any pending unstarted request
                    try:
                        _req_queue.get_nowait()
                    except queue.Empty:
                        pass
                elif type(item) in (Generate, IntentGenerate, ClassifySystem,
                                    ManualGenerate):
                    _stop_event.set()   # cancel running inference first
                    # replace any pending request (drop old, queue new)
                    try:
                        _req_queue.get_nowait()
                    except queue.Empty:
                        pass
                    _req_queue.put(item)
                else:
                    obs.on_error(f"Unknown item type: {type(item)}")

            sink.request.subscribe(on_next=on_request,
                                   on_error=lambda e: obs.on_error(e))

        # observer must be set before worker uses it; placeholder prevents NameError
        observer = None

        return Source(system=rx.create(on_subscribe))

    return Component(call=driver, input=Sink)
