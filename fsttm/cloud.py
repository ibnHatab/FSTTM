"""
LLM driver over an OpenAI-compatible chat API (cloud or remote), the same
message protocol as fsttm.llama: Initialize / AddSystem / Generate /
IntentGenerate / ClassifySystem / ManualGenerate / StopGenerate / TrimHistory
in, Response / ResponseDone / IntentResult / SystemIntent / LlamaError out.

One endpoint shape covers every provider we care about:
  Gemini      base_url https://generativelanguage.googleapis.com/v1beta/openai/
              model gemini-3.1-flash-lite, key in $GEMINI_API_KEY
  OpenRouter  base_url https://openrouter.ai/api/v1, any of its models,
              key in $OPENROUTER_API_KEY
  llama-server (local or on another box) base_url http://host:8080/v1

What differs from the local driver:
  * No KV cache, so nothing to pre-warm, re-prime or roll back. The
    conversation is the message list of ConversationHistory, sent whole on
    every turn.
  * Barge-in: the local driver rewrites the last assistant turn to the heard
    text (TrimHistory), because its KV holds what was never said. Here that
    rollback is OFF by default (gpt.trim_on_barge_in: false): the reply the
    model wrote stays in its history, cut or not. Set it true to keep the
    heard-text behaviour.
  * Intents: no GBNF. The provider's JSON schema goes in response_format
    (json_schema), and the spoken acknowledgement comes in the same JSON as a
    "spoken" field, so one round trip gives both passes' results.
  * Cancel: StopGenerate closes the HTTP stream at the next chunk.

The key is read from the environment variable named by gpt.api_key_env (or a
file named by gpt.api_key_file); it is never logged.
"""
from __future__ import annotations

import collections
import json
import logging as _pylog
import os
import queue
import threading
import time
import urllib.error
import urllib.request

import reactivex as rx
from cyclotron import Component

from fsttm.llama import (Sink, Source, Initialize, AddSystem, Generate, IntentGenerate,
                         ClassifySystem, ManualGenerate, StopGenerate, TrimHistory, StreamEvent,
                         Response, ResponseDone, IntentResult, SystemIntent, LlamaError,
                         ConversationHistory)

_log = _pylog.getLogger("fsttm.cloud")

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_KEY_ENV = "GEMINI_API_KEY"
VOICE_STYLE = ("Your words are spoken aloud by a speech synthesizer: answer in one to three short "
               "sentences of plain text, no markdown, no lists, no emoji.")


# The model works with three streams, and is told so. Only the conversation is addressed to
# it; the other two run in parallel with it and with each other, and reach it as context.
STREAMS_PROMPT = """\
You work with three separate streams. Keep them apart.

1. CONVERSATION: the person's words (speech recognition, so it can mishear) and your own
   replies. This is the only stream addressed to you, and the only one you answer.
2. ENVIRONMENT: the state of the home as Home Assistant reports it (devices, sensors),
   updated continuously and independently of the conversation. It is the truth about the
   devices, more reliable than what you assumed or what the person remembers.
3. REASONING: conclusions of the home's rule engine, a reasoning process that runs in
   parallel with you and with the conversation: whether a request was done or refused and
   why, what it inferred, what it noticed. It is not the person speaking and not your own
   thought. About what the home did and can do, it is authoritative: when it contradicts
   what you expected or said, follow it and correct yourself.

ENVIRONMENT and REASONING are given below with their times; a newer entry supersedes an
older one. They are not messages to you: never answer them as if the person had said them,
never read them out unasked. Use them when they bear on what the person says, and mention a
REASONING entry on your own only when the person should know it (a refusal, a problem).
You do not switch anything yourself: a request to change something in the home goes to the
rule engine, which decides it while you speak. Until its outcome is in REASONING, say only
that you are on it ("Turning off the TV.", "One moment.") — never that it is done, and never
invent a refusal. When the outcome arrives it tells you what happened; say so if asked, or
on your own when it was refused."""


def render_streams(env, reasoning, now=None):
    """The ENVIRONMENT and REASONING sections of the system prompt, or "" when both are empty."""
    hm = lambda ts: time.strftime("%H:%M:%S", time.localtime(ts)) if ts else "--:--:--"
    out = []
    if env:
        out.append("ENVIRONMENT (as of %s):" % hm(now or time.time()))
        out += ["- %s: %s  (%s)" % (k, text, hm(ts)) for k, (text, ts) in sorted(env.items())]
    if reasoning:
        out.append("REASONING (newest last):")
        out += ["- %s %s" % (hm(ts), text) for text, ts in reasoning]
    return "\n".join(out)


class CloudError(RuntimeError):
    pass


THINKING_HINT = ("the model spent its whole reply on thinking (finish_reason length, no text): "
                 "turn thinking off with gpt.request_extra, e.g. {reasoning_effort: none} "
                 "(OpenAI-style APIs; Gemini: the lowest effort the model accepts) or, for a "
                 "llama-server, {chat_template_kwargs: {enable_thinking: false}}; or raise gpt.max_tokens")


class Chat:
    """One OpenAI-compatible endpoint: chat/completions, streamed or not."""

    def __init__(self, base_url, model, api_key, timeout=30.0, extra=None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.key, self.timeout = model, api_key, timeout
        self.extra = dict(extra or {})

    def _open(self, body):
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json",
                     **({"Authorization": "Bearer " + self.key} if self.key else {})})
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:400]
            raise CloudError("HTTP %d from %s: %s" % (e.code, self.url, detail)) from None
        except (urllib.error.URLError, OSError) as e:
            raise CloudError("%s: %s" % (self.url, e)) from None

    def body(self, messages, **kw):
        b = {"model": self.model, "messages": messages}
        b.update(self.extra)
        b.update({k: v for k, v in kw.items() if v is not None})
        return b

    def complete(self, messages, **kw):
        """The whole reply's text (one request, no stream)."""
        with self._open(self.body(messages, stream=False, **kw)) as r:
            out = json.load(r)
        try:
            ch = out["choices"][0]
            text = ch["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            raise CloudError("unexpected answer: %s" % json.dumps(out)[:300]) from None
        if not text and ch.get("finish_reason") == "length":
            raise CloudError(THINKING_HINT)
        return text

    def stream(self, messages, stop_event, **kw):
        """The reply's text pieces as they arrive (server-sent events); ends early,
        closing the connection, once stop_event is set."""
        r = self._open(self.body(messages, stream=True, **kw))
        said, thought, finish = False, False, None
        try:
            for raw in r:
                if stop_event.is_set():
                    return
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except ValueError:
                    continue
                if ev.get("error"):
                    raise CloudError(str(ev["error"])[:300])
                for ch in ev.get("choices") or ():
                    delta = ch.get("delta") or {}
                    thought = thought or bool(delta.get("reasoning_content") or delta.get("reasoning"))
                    finish = ch.get("finish_reason") or finish
                    piece = delta.get("content")
                    if piece:
                        said = True
                        yield piece
        finally:
            r.close()
        if not said and (thought or finish == "length"):
            raise CloudError(THINKING_HINT)


def _api_key(item):
    path = getattr(item, "api_key_file", None)
    if path:
        with open(os.path.expanduser(path)) as f:
            return f.read().strip()
    env = getattr(item, "api_key_env", None) or DEFAULT_KEY_ENV
    return os.environ.get(env, "")


def _messages(sys_prompt, history, new_user):
    """System prompt, the kept turns, the new utterance, as chat messages."""
    dropped = history.trim_for(sys_prompt, new_user)
    if dropped:
        _log.info("ctx-fifo: dropped %d oldest turn(s)", dropped)
    msgs = [{"role": "system", "content": sys_prompt}] if sys_prompt else []
    for u, a in history._turns:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": new_user})
    return msgs


def _intent_schema(domains):
    """The active provider's intent schema with a "spoken" field beside it (the
    second pass of the local driver, in the same answer)."""
    from fsttm.domain import active_provider
    schema = json.loads(json.dumps(active_provider().build_schema(domains)))
    props = schema.setdefault("properties", {})
    props["spoken"] = {"type": "string",
                       "description": "your own short spoken reply to the person (one sentence, "
                                      "first person; never a repeat of their words)"}
    req = schema.setdefault("required", [])
    if "spoken" not in req:
        req.append("spoken")
    return schema


def make_driver(loop=None):
    def driver(sink):
        chat = [None]                 # Chat, once Initialize has come
        cfg = {"temp": 0.7, "max_tokens": 200, "trim": False}
        sys_prompt = [""]
        history = ConversationHistory(n_ctx=32768)
        env = {}                                             # key -> (text, ts): the ENVIRONMENT stream
        reasoning = collections.deque(maxlen=12)              # (text, ts): the REASONING stream
        stop_event = threading.Event()
        req_queue: queue.Queue = queue.Queue(maxsize=1)
        observer = [None]

        def emit(ev):
            loop.call_soon_threadsafe(observer[0].on_next, ev)

        def with_streams(base):
            """The base prompt with the streams' explanation and their current entries."""
            parts = [base] if base else []
            parts.append(STREAMS_PROMPT)
            st = render_streams(env, list(reasoning))
            if st:
                parts.append(st)
            return "\n\n".join(parts)

        def chat_system():
            s = sys_prompt[0]
            return with_streams((s + "\n\n" + VOICE_STYLE) if s else VOICE_STYLE)

        # ── plain chat, streamed ──────────────────────────────────────────
        def handle_generate(item):
            stop_event.clear()
            acc, t0 = [], time.monotonic()
            try:
                for piece in chat[0].stream(_messages(chat_system(), history, item.text), stop_event,
                                            temperature=cfg["temp"], max_tokens=cfg["max_tokens"]):
                    acc.append(piece)
                    emit(Response(text=piece, context=item.context))
            except Exception as exc:
                _log.warning("chat failed: %s", exc)
                emit(LlamaError(error=exc, context=item.context))
            finally:
                full = "".join(acc)
                if full.strip():
                    history.add_turn(item.text, full.strip())
                _log.info("chat %.0f ms%s: %r", (time.monotonic() - t0) * 1000,
                          " (cut)" if stop_event.is_set() else "", full[:120])
                emit(ResponseDone(full_text=full, context=item.context))

        # ── intent: the provider's schema as the response format ──────────
        def handle_intent(item):
            t0 = time.monotonic()
            try:
                schema = _intent_schema(item.domains)
                txt = chat[0].complete(
                    _messages(with_streams(sys_prompt[0]), history, item.text), temperature=0.0,
                    max_tokens=cfg["max_tokens"],
                    response_format={"type": "json_schema",
                                     "json_schema": {"name": "intent", "schema": schema}})
                intent = json.loads(txt)
                tts = str(intent.pop("spoken", "") or "")
            except Exception as exc:
                _log.warning("intent failed for %r: %s", item.text, exc)
                emit(LlamaError(error=exc, context=item.context))
                return
            if tts:
                history.add_turn(item.text, tts)
            _log.info("intent %.0f ms: %r / %r", (time.monotonic() - t0) * 1000, intent, tts)
            emit(IntentResult(intent_json=intent, tts_text=tts, context=item.context))

        # ── the attention layer's system classifier (history-free) ────────
        def handle_classify(item):
            from fsttm.grammar import SYSTEM_INTENT_PROMPT
            action = "command"
            try:
                txt = chat[0].complete(
                    [{"role": "system", "content": SYSTEM_INTENT_PROMPT},
                     {"role": "user", "content": item.text}],
                    temperature=0.0, max_tokens=20,
                    response_format={"type": "json_schema", "json_schema": {"name": "system_action", "schema": {
                        "type": "object", "required": ["action"],
                        "properties": {"action": {"type": "string", "enum": ["command", "sleep", "mute"]}}}}})
                action = json.loads(txt).get("action", "command")
            except Exception as exc:
                emit(LlamaError(error=exc, context=item.context))
            emit(SystemIntent(action=action, context=item.context))

        # ── a grounded one-shot answer (RAG prompt already built) ─────────
        def handle_manual(item):
            stop_event.clear()
            acc = []
            try:
                for piece in chat[0].stream([{"role": "user", "content": item.prompt}], stop_event,
                                            temperature=0.2, max_tokens=cfg["max_tokens"]):
                    acc.append(piece)
                    emit(Response(text=piece, context=item.context))
            except Exception as exc:
                emit(LlamaError(error=exc, context=item.context))
            finally:
                emit(ResponseDone(full_text="".join(acc), context=item.context))

        HANDLERS = {Generate: handle_generate, IntentGenerate: handle_intent,
                    ClassifySystem: handle_classify, ManualGenerate: handle_manual}

        def worker():
            while True:
                item = req_queue.get()
                if item is None:
                    break
                if chat[0] is None:
                    emit(LlamaError(error=CloudError("no endpoint: Initialize did not succeed"),
                                    context=getattr(item, "context", None)))
                    continue
                HANDLERS[type(item)](item)

        threading.Thread(target=worker, daemon=True, name="cloud-llm-worker").start()

        def on_subscribe(obs, scheduler):
            observer[0] = obs

            def on_request(item):
                t = type(item)
                if t is Initialize:
                    key = _api_key(item)
                    base = getattr(item, "base_url", None) or DEFAULT_BASE_URL
                    if not key and ("googleapis" in base or "openrouter" in base):
                        obs.on_next(LlamaError(error=CloudError(
                            "no API key: set $%s (gpt.api_key_env) or gpt.api_key_file"
                            % (getattr(item, "api_key_env", None) or DEFAULT_KEY_ENV)), context=None))
                        return
                    chat[0] = Chat(base, item.model_path, key,
                                   timeout=float(getattr(item, "timeout", None) or 30.0),
                                   extra=getattr(item, "request_extra", None))
                    cfg["temp"] = float(getattr(item, "temperature", None) or cfg["temp"])
                    cfg["max_tokens"] = int(getattr(item, "max_tokens", None) or cfg["max_tokens"])
                    cfg["trim"] = bool(getattr(item, "trim_on_barge_in", False))
                    print("Cloud LLM: %s at %s (barge-in trim %s)"
                          % (item.model_path, base, "on" if cfg["trim"] else "off"))
                    _log.info("cloud llm: model=%s base=%s trim=%s", item.model_path, base, cfg["trim"])
                elif t is AddSystem:
                    sys_prompt[0] = item.prompt or ""
                elif t is TrimHistory:
                    # the context rollback after a barge-in: off unless asked for
                    if cfg["trim"] and history._turns and item.heard_text is not None:
                        user, _ = history._turns[-1]
                        history._turns[-1] = (user, item.heard_text.strip() or "...")
                elif t is StreamEvent:
                    ts = item.ts or time.time()
                    if item.stream == "env" and item.key:
                        if item.text is None:
                            env.pop(item.key, None)           # the entity is gone
                        else:
                            env[item.key] = (str(item.text), ts)
                    elif item.stream == "reasoning" and item.text:
                        reasoning.append((str(item.text), ts))
                    else:
                        _log.warning("stream event ignored: %r", item)
                elif t is StopGenerate:
                    stop_event.set()
                    try:
                        req_queue.get_nowait()
                    except queue.Empty:
                        pass
                elif t in HANDLERS:
                    stop_event.set()            # a new request cancels the running one
                    try:
                        req_queue.get_nowait()
                    except queue.Empty:
                        pass
                    req_queue.put(item)
                else:
                    obs.on_error("Unknown item type: %s" % t)

            sink.request.subscribe(on_next=on_request, on_error=lambda e: obs.on_error(e))

        return Source(system=rx.create(on_subscribe))

    return Component(call=driver, input=Sink)
