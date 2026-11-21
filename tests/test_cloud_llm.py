"""fsttm.cloud against a fake OpenAI-compatible endpoint (stdlib HTTP server,
no network): streaming chat, cancel, the barge-in trim off and on, intents in
the provider's schema, the system classifier, a missing key, and the backend
chosen by Initialize."""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from reactivex.subject import Subject

from fsttm import llama
from fsttm.llama import (Initialize, AddSystem, Generate, IntentGenerate, ClassifySystem,
                         StopGenerate, TrimHistory, Response, ResponseDone, IntentResult,
                         SystemIntent, LlamaError, Sink)


class Fake:
    """The endpoint's behaviour, set per test; every request body is kept."""
    def __init__(self):
        self.bodies = []
        self.reply = ["Hello", " there", "."]
        self.json = {"intent": "CHITCHAT", "spoken": "Hi."}
        self.delay = 0.0


def serve(fake):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body["_auth"] = self.headers.get("Authorization")
            fake.bodies.append(body)
            if not body.get("stream"):
                out = json.dumps({"choices": [{"message": {"content": json.dumps(fake.json)}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            import time
            try:
                for piece in fake.reply:
                    ev = {"choices": [{"delta": {"content": piece}}]}
                    self.wfile.write(("data: %s\n\n" % json.dumps(ev)).encode())
                    self.wfile.flush()
                    time.sleep(fake.delay)
                if not fake.reply:        # thinking ate the budget: reasoning, then length
                    ev = {"choices": [{"delta": {"reasoning_content": "Let me think..."},
                                       "finish_reason": "length"}]}
                    self.wfile.write(("data: %s\n\n" % json.dumps(ev)).encode())
                self.wfile.write(b"data: [DONE]\n\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def endpoint(monkeypatch):
    fake = Fake()
    srv = serve(fake)
    monkeypatch.setenv("TEST_LLM_KEY", "k-123")
    yield fake, "http://127.0.0.1:%d/v1" % srv.server_address[1]
    srv.shutdown()


async def run(requests, until, timeout=5.0):
    """Feed the requests to the driver; collect its events until `until(events)`."""
    loop = asyncio.get_running_loop()
    sub, events, done = Subject(), [], asyncio.Event()
    src = llama.make_driver(loop).call(Sink(request=sub))

    def on(ev):
        events.append(ev)
        if until(events):
            done.set()
    src.system.subscribe(on_next=on)
    for r in requests:
        if callable(r):
            await r(sub, events)
        else:
            sub.on_next(r)
    await asyncio.wait_for(done.wait(), timeout)
    return sub, events


def init(base, **kw):
    return Initialize(model_path="test-model", backend="openai", base_url=base,
                      api_key_env="TEST_LLM_KEY", **kw)


def dones(evs):
    return [e for e in evs if type(e) is ResponseDone]


@pytest.mark.asyncio
async def test_chat_streams_and_keeps_history(endpoint):
    fake, base = endpoint

    async def second(sub, events):
        while not dones(events):
            await asyncio.sleep(0.01)
        sub.on_next(Generate(text="and again", context=2))
    _, evs = await run([init(base), AddSystem(prompt="You are Nina."), Generate(text="hi", context=1), second],
                       lambda e: len(dones(e)) == 2)
    assert "".join(e.text for e in evs if type(e) is Response and e.context == 1) == "Hello there."
    b1, b2 = fake.bodies
    assert b1["model"] == "test-model" and b1["stream"] is True and b1["_auth"] == "Bearer k-123"
    assert b1["messages"][0]["role"] == "system" and b1["messages"][0]["content"].startswith("You are Nina.")
    assert [m["content"] for m in b2["messages"][1:]] == ["hi", "Hello there.", "and again"]


@pytest.mark.asyncio
async def test_stop_cuts_the_stream(endpoint):
    fake, base = endpoint
    fake.reply, fake.delay = ["w%d " % i for i in range(50)], 0.05

    async def stop(sub, events):
        while not any(type(e) is Response for e in events):
            await asyncio.sleep(0.01)
        sub.on_next(StopGenerate())
    _, evs = await run([init(base), Generate(text="talk", context=1), stop], lambda e: bool(dones(e)))
    assert len(dones(evs)[0].full_text.split()) < 50


@pytest.mark.asyncio
@pytest.mark.parametrize("trim,expect", [(False, "Hello there."), (True, "Hello")])
async def test_barge_in_trim_is_off_unless_asked(endpoint, trim, expect):
    fake, base = endpoint

    async def cut_then_ask(sub, events):
        while not dones(events):
            await asyncio.sleep(0.01)
        sub.on_next(TrimHistory(heard_text="Hello"))
        sub.on_next(Generate(text="next", context=2))
    await run([init(base, trim_on_barge_in=trim), Generate(text="hi", context=1), cut_then_ask],
              lambda e: len(dones(e)) == 2)
    assert fake.bodies[1]["messages"][-2] == {"role": "assistant", "content": expect}


@pytest.mark.asyncio
async def test_intent_uses_the_schema_and_splits_spoken(endpoint):
    from fsttm.domain import NullProvider
    import fsttm.domain as dom
    fake, base = endpoint
    dom._ACTIVE[0] = NullProvider()
    _, evs = await run([init(base), IntentGenerate(text="hello", context=7)],
                       lambda e: any(type(x) in (IntentResult, LlamaError) for x in e))
    res = [e for e in evs if type(e) is IntentResult]
    assert res, evs
    assert res[0].intent_json == {"intent": "CHITCHAT"} and res[0].tts_text == "Hi."
    rf = fake.bodies[0]["response_format"]
    assert rf["type"] == "json_schema" and "spoken" in rf["json_schema"]["schema"]["properties"]


@pytest.mark.asyncio
async def test_classify(endpoint):
    fake, base = endpoint
    fake.json = {"action": "sleep"}
    _, evs = await run([init(base), ClassifySystem(text="go to sleep", context=3)],
                       lambda e: any(type(x) is SystemIntent for x in e))
    assert [e.action for e in evs if type(e) is SystemIntent] == ["sleep"]


@pytest.mark.asyncio
async def test_missing_key_is_an_error_not_a_crash(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    _, evs = await run([Initialize(model_path="gemini-3.1-flash-lite", backend="openai")],
                       lambda e: any(type(x) is LlamaError for x in e))
    assert "GEMINI_API_KEY" in str(evs[0].error)


@pytest.mark.asyncio
async def test_thinking_only_reply_says_how_to_turn_it_off(endpoint):
    fake, base = endpoint
    fake.reply = []                       # no text at all, as when thinking ate max_tokens
    _, evs = await run([init(base), Generate(text="hi", context=1)], lambda e: bool(dones(e)))
    errs = [e for e in evs if type(e) is LlamaError]
    assert errs and "thinking" in str(errs[0].error)


def test_config_fields():
    from fsttm.config import llama_cloud_fields
    f = llama_cloud_fields({"backend": "openai", "temp": 0.3, "request_extra": {"reasoning_effort": "none"},
                            "model": "x"})
    assert f == {"backend": "openai", "temperature": 0.3, "request_extra": {"reasoning_effort": "none"}}
    assert Initialize(model_path="m").backend == "llama"
