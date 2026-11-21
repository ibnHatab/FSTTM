"""
The streams that run beside the conversation, and the door they come in by.

Two streams reach the LLM as context, never as turns (fsttm.cloud renders them
into the system prompt, with what they are):
  env        the home's state as Home Assistant reports it: {"stream": "env",
             "key": "light.living_room", "text": "on"}; a newer entry for a key
             replaces the older; "text": null removes the key
  reasoning  what the rule engine (Metis) concluded: {"stream": "reasoning",
             "text": "refused on(heating): it is not a device I know"}; appended

Producers are separate processes (the Home Assistant bridge, the Metis
sidecar): they connect to a local TCP port and write one JSON object per line.
Config:  streams: {port: 8766}   (absent or null: no door)

    python -m fsttm.streams send env light.living_room on      # by hand
    python -m fsttm.streams send reasoning "the TV is not answering"
"""
from __future__ import annotations

import json
import logging
import socket
import socketserver
import sys
import threading
import time

from fsttm.llama import StreamEvent

_log = logging.getLogger("fsttm.streams")
STREAMS = ("env", "reasoning")
DEFAULT_PORT = 8766


def parse(line: str) -> StreamEvent:
    """One JSON line → a StreamEvent, or ValueError naming what is wrong."""
    d = json.loads(line)
    if not isinstance(d, dict) or d.get("stream") not in STREAMS:
        raise ValueError('a line is {"stream": "env"|"reasoning", "key": ..., "text": ...}')
    if d["stream"] == "env" and not d.get("key"):
        raise ValueError("an env entry needs a key (the entity)")
    if d["stream"] == "reasoning" and not d.get("text"):
        raise ValueError("a reasoning entry needs a text")
    return StreamEvent(stream=d["stream"], key=d.get("key"), text=d.get("text"),
                       ts=float(d.get("ts") or time.time()))


class StreamDoor:
    """A local TCP listener: each JSON line becomes deliver(StreamEvent). One
    thread per connection; a producer may keep its connection open."""

    def __init__(self, deliver, port: int = DEFAULT_PORT, host: str = "127.0.0.1"):
        door = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                for raw in self.rfile:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        ev = parse(line)
                    except ValueError as e:
                        self.wfile.write(("error: %s\n" % e).encode())
                        continue
                    door.count += 1
                    deliver(ev)

        socketserver.ThreadingTCPServer.allow_reuse_address = True
        self.server = socketserver.ThreadingTCPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.count = 0
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True,
                                       name="stream-door")

    def start(self):
        self.thread.start()
        _log.info("stream door on 127.0.0.1:%d", self.port)
        return self

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def send(events, port: int = DEFAULT_PORT, host: str = "127.0.0.1"):
    """Write events (dicts) to a running door."""
    with socket.create_connection((host, port), timeout=5) as s:
        s.sendall("".join(json.dumps(e) + "\n" for e in events).encode())


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    port = DEFAULT_PORT
    if a[:1] == ["--port"]:
        port, a = int(a[1]), a[2:]
    if len(a) >= 3 and a[0] == "send" and a[1] == "env":
        ev = {"stream": "env", "key": a[2], "text": " ".join(a[3:]) or None}
    elif len(a) >= 3 and a[0] == "send" and a[1] == "reasoning":
        ev = {"stream": "reasoning", "text": " ".join(a[2:])}
    else:
        print(__doc__.strip().splitlines()[-2].strip())
        print(__doc__.strip().splitlines()[-1].strip())
        return 2
    send([ev], port)
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ── the panel: the streams as they run, for a screen beside the dashboard ──────────────────
#
# A page and a server-sent-events feed on a second local port (streams.panel_port, default
# the door's + 1): the ENVIRONMENT as a table that is replaced in place, the REASONING as a
# log, and the CONVERSATION beside them, each in its own column. Read-only.

import collections
import http.server
import queue as _queue

PANEL_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Nina · streams</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --bg:#0b0e13; --panel:#11161e; --line:#222a35; --fg:#c9d1d9; --dim:#7d8590; --env:#39c5cf; --rsn:#d2a8ff; --you:#e6edf3; --nina:#7ee787; --warn:#f0b72f; }
  * { box-sizing: border-box; }
  html, body { margin:0; height:100%; background:var(--bg); color:var(--fg); font:13px/1.45 ui-monospace, "DejaVu Sans Mono", Menlo, monospace; }
  body { display:grid; grid-template-rows: auto 1fr 1fr 1fr; height:100vh; }
  header { padding:6px 10px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; color:var(--dim); }
  header b { color:var(--fg); letter-spacing:.15em; }
  section { min-height:0; display:flex; flex-direction:column; border-bottom:1px solid var(--line); }
  h2 { margin:0; padding:5px 10px; font-size:11px; letter-spacing:.12em; font-weight:600; background:var(--panel); border-bottom:1px solid var(--line); }
  #env h2 { color:var(--env); } #rsn h2 { color:var(--rsn); } #conv h2 { color:var(--you); }
  .body { overflow:auto; padding:4px 10px; flex:1; }
  .row { padding:2px 0; border-bottom:1px solid #161c25; display:flex; gap:8px; }
  .t { color:var(--dim); flex:none; }
  .k { color:var(--dim); flex:none; min-width:0; }
  .v { overflow-wrap:anywhere; }
  .fresh { animation: fresh 2.5s ease-out; }
  @keyframes fresh { from { background:#1d2b3a; } to { background:transparent; } }
  .refused .v { color:var(--warn); }
  .you .v { color:var(--you); } .nina .v { color:var(--nina); }
  .empty { color:var(--dim); padding:6px 0; }
</style></head>
<body>
<header><b>NINA · STREAMS</b><span id="conn">connecting…</span></header>
<section id="env"><h2>ENVIRONMENT · the home as Home Assistant reports it</h2><div class="body" id="envb"><div class="empty">no state yet</div></div></section>
<section id="rsn"><h2>REASONING · the rule engine's conclusions</h2><div class="body" id="rsnb"><div class="empty">nothing concluded yet</div></div></section>
<section id="conv"><h2>CONVERSATION</h2><div class="body" id="convb"><div class="empty">nobody has spoken yet</div></div></section>
<script>
const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'})[c]);
const hm = ts => new Date(ts * 1000).toTimeString().slice(0, 8);
const env = new Map();
function drawEnv(fresh) {
  const b = $('#envb');
  if (!env.size) { b.innerHTML = '<div class="empty">no state yet</div>'; return; }
  b.innerHTML = [...env.entries()].sort(([a], [c]) => a.localeCompare(c)).map(([k, e]) =>
    `<div class="row${k === fresh ? ' fresh' : ''}"><span class="t">${hm(e.ts)}</span><span class="v">${esc(e.text)}</span></div>`).join('');
}
function add(box, cls, ts, text, who) {
  const b = $(box);
  b.querySelector('.empty')?.remove();
  const d = document.createElement('div');
  d.className = `row fresh ${cls}`;
  d.innerHTML = `<span class="t">${hm(ts)}</span>${who ? `<span class="k">${esc(who)}</span>` : ''}<span class="v">${esc(text)}</span>`;
  b.prepend(d);
  while (b.children.length > 200) b.lastChild.remove();
}
function on(ev, first) {
  if (ev.stream === 'env') { if (ev.text == null) env.delete(ev.key); else env.set(ev.key, ev); if (!first) drawEnv(ev.key); }
  else if (ev.stream === 'reasoning') add('#rsnb', /refused|cannot|can't|not /i.test(ev.text) ? 'refused' : '', ev.ts, ev.text);
  else if (ev.stream === 'conversation') add('#convb', ev.who === 'user' ? 'you' : 'nina', ev.ts, ev.text, ev.who === 'user' ? 'you' : 'nina');
}
function connect() {
  const es = new EventSource('events');
  es.addEventListener('snapshot', m => {
    const s = JSON.parse(m.data);
    env.clear(); $('#rsnb').innerHTML = ''; $('#convb').innerHTML = '';
    for (const e of s.env) on(e, true);
    for (const e of s.reasoning) on(e, true);
    for (const e of s.conversation) on(e, true);
    drawEnv();
    if (!s.reasoning.length) $('#rsnb').innerHTML = '<div class="empty">nothing concluded yet</div>';
    if (!s.conversation.length) $('#convb').innerHTML = '<div class="empty">nobody has spoken yet</div>';
  });
  es.onmessage = m => on(JSON.parse(m.data));
  es.onopen = () => $('#conn').textContent = 'live';
  es.onerror = () => { $('#conn').textContent = 'reconnecting…'; };
}
connect();
</script></body></html>
"""


class StreamPanel:
    """The panel's state (the env table, the last reasoning and conversation entries) and
    its HTTP server: GET / the page, GET /events a server-sent-events feed that starts with
    a snapshot. publish(dict) from any thread."""

    def __init__(self, port: int, host: str = "127.0.0.1", keep: int = 100):
        self.env = {}
        self.reasoning = collections.deque(maxlen=keep)
        self.conversation = collections.deque(maxlen=keep)
        self.lock = threading.Lock()
        self.clients = []
        panel = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = PANEL_HTML.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path != "/events":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                q = _queue.Queue(maxsize=1000)
                with panel.lock:
                    snap = {"env": list(panel.env.values()), "reasoning": list(panel.reasoning),
                            "conversation": list(panel.conversation)}
                    panel.clients.append(q)
                try:
                    self.wfile.write(("event: snapshot\ndata: %s\n\n" % json.dumps(snap)).encode())
                    self.wfile.flush()
                    while True:
                        try:
                            ev = q.get(timeout=15)
                            self.wfile.write(("data: %s\n\n" % json.dumps(ev)).encode())
                        except _queue.Empty:
                            self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    with panel.lock:
                        if q in panel.clients:
                            panel.clients.remove(q)

        http.server.ThreadingHTTPServer.allow_reuse_address = True
        self.server = http.server.ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True, name="stream-panel")

    def start(self):
        self.thread.start()
        _log.info("stream panel on http://127.0.0.1:%d/", self.port)
        return self

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def publish(self, ev: dict):
        ev = dict(ev, ts=ev.get("ts") or time.time())
        with self.lock:
            if ev.get("stream") == "env":
                if ev.get("text") is None:
                    self.env.pop(ev.get("key"), None)
                else:
                    self.env[ev["key"]] = ev
            elif ev.get("stream") == "reasoning":
                self.reasoning.append(ev)
            elif ev.get("stream") == "conversation":
                self.conversation.append(ev)
            for q in list(self.clients):
                try:
                    q.put_nowait(ev)
                except _queue.Full:
                    pass

    def publish_event(self, se: StreamEvent):
        self.publish({"stream": se.stream, "key": se.key, "text": se.text, "ts": se.ts})
