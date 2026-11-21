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
