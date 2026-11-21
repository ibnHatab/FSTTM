"""
The Metis sidecar: the home's rule engine beside the conversation.

It runs metis-agent's domain agent (one process serving the home world's packages:
metis-worlds `home`, branch home) on a wall-clock beat, and joins it to FSTTM two ways:

  REASONING   every outcome and every unasked conclusion of the rules (a refusal and why, a
              suggestion, a surprise) is written to the stream door as a reasoning entry
              (fsttm.streams), so the LLM has it as context.
  tools       a local port answers the LLM's tool calls: {"op": "tools"} lists one tool per
              package (its intent schema from `python -m metis.intents schema`), {"op":
              "call", "domain", "args"} hands the intent to the rules and waits for their
              outcome (done or refused, with the rule's sentence). fsttm.cloud calls it.

The LLM never switches a device itself: it asks a package, the package's rules decide, the
host (the world's, Home Assistant through its token) actuates, and the outcome comes back.

    python -m fsttm.feeds.metis --config config.nuc.yaml      # fsttm-metis

Config:
    metis:
        python: ~/metis/metis-agent/py/.venv/bin/python
        cwd: ~/metis/metis-agent/py
        catalog: ~/metis/metis-worlds/home
        domains: [lights, air, media, scenes, comfort]
        env: {METISC: ~/metis/metis-lang/src/metisc, METIS_HOME_HA: "http://127.0.0.1:8123"}
        tool_port: 8768
        beat_s: 1.0
        call_timeout_s: 6
    streams: {port: 8766}
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import queue
import socket
import socketserver
import subprocess
import threading
import time

import yaml

log = logging.getLogger("fsttm.feeds.metis")

DEFAULTS = {
    "python": "~/metis/metis-agent/py/.venv/bin/python",
    "cwd": "~/metis/metis-agent/py",
    "catalog": "~/metis/metis-worlds/home",
    "domains": ["lights", "air", "media", "scenes", "comfort"],
    "env": {},
    "tool_port": 8768,
    "beat_s": 1.0,
    "call_timeout_s": 6.0,
}
FINAL = ("done", "refused", "expired", "cancelled")
ex = os.path.expanduser


def reasoning_line(r: dict) -> str | None:
    """A domain agent's reply as one REASONING entry, or None when it is not one (an
    acceptance, a clock ack, a read)."""
    d, kind = r.get("domain") or "?", r.get("kind")
    if kind == "result" and r.get("status") in FINAL + ("progress",):
        return "%s %s: %s" % (d, r["status"], r.get("text") or "")
    if kind == "event" and r.get("text"):
        return "%s noticed: %s" % (d, r["text"])
    if kind == "surprise":
        return "%s surprise: %s" % (d, r.get("difference") or r.get("record") or "the world differs from the rules")
    if kind == "error" and r.get("id") not in (None, "p0"):
        return "%s error: %s" % (d, r.get("error"))
    return None


class Door:
    """The stream door (fsttm.streams), reopened when the voice process restarts."""

    def __init__(self, port):
        self.port, self.sock = port, None

    def send(self, events):
        data = "".join(json.dumps(e) + "\n" for e in events).encode()
        for _ in range(2):
            try:
                if self.sock is None:
                    self.sock = socket.create_connection(("127.0.0.1", self.port), timeout=5)
                self.sock.sendall(data)
                return True
            except OSError:
                if self.sock is not None:
                    self.sock.close()
                self.sock = None
        return False


class Sidecar:
    def __init__(self, cfg: dict, stream_port: int):
        self.cfg = cfg
        self.door = Door(stream_port)
        self.ids = itertools.count(1)
        self.waiting = {}                 # id -> Queue of the final reply
        self.lock = threading.Lock()
        self.proc = None
        self.tools = self.load_tools()

    # ── the domain agent ──────────────────────────────────────────────────
    def env(self):
        e = dict(os.environ, METIS_CATALOG=ex(self.cfg["catalog"]))
        e.update({k: ex(str(v)) for k, v in (self.cfg.get("env") or {}).items()})
        return e

    def load_tools(self):
        """One tool per package: its name, what it is (the manifest's line), its intent schema."""
        tools = []
        for d in self.cfg["domains"]:
            out = subprocess.run([ex(self.cfg["python"]), "-m", "metis.intents", "schema", d],
                                 cwd=ex(self.cfg["catalog"]), env=self.env(), capture_output=True,
                                 text=True, timeout=60)
            if out.returncode:
                log.warning("no intent schema for %s: %s", d, out.stderr.strip()[-200:])
                continue
            schema = json.loads(out.stdout)
            line = ""
            try:
                man = yaml.safe_load(open(os.path.join(ex(self.cfg["catalog"]), "domains", d, "domain.yaml")))
                line = man.get("line") or man.get("display") or ""
            except (OSError, ValueError):
                pass
            tools.append({"type": "function", "function": {
                "name": "home_" + d,
                "description": ("Ask the home's rule engine (package %s) to act or answer: %s. "
                                "The rules decide; the result is their outcome." % (d, line)),
                "parameters": schema}})
        log.info("tools: %s", ", ".join(t["function"]["name"] for t in tools))
        return tools

    def start_agent(self):
        argv = [ex(self.cfg["python"]), "-m", "metis_agent.domain_agent", "--domain",
                ",".join(self.cfg["domains"]), "--case", "live", "--consult", "template", "--records"]
        self.proc = subprocess.Popen(argv, cwd=ex(self.cfg["cwd"]), env=self.env(), text=True,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(target=self.read_out, daemon=True, name="metis-out").start()
        threading.Thread(target=self.read_err, daemon=True, name="metis-err").start()

    def write(self, obj):
        with self.lock:
            self.proc.stdin.write(json.dumps(obj) + "\n")
            self.proc.stdin.flush()

    def read_err(self):
        for line in self.proc.stderr:
            line = line.rstrip()
            if line and ("Error" in line or "Traceback" in line or "error" in line):
                log.warning("domain agent: %s", line[:300])

    def read_out(self):
        for line in self.proc.stdout:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("clock") == "ready":
                log.info("domain agent ready: %s", r.get("domains"))
                continue
            text = reasoning_line(r)
            if text:
                self.door.send([{"stream": "reasoning", "text": text}])
                log.info("reasoning: %s", text)
            rid = r.get("id")
            final = (r.get("kind") == "result" and r.get("status") in FINAL) or r.get("kind") == "error"
            if rid in self.waiting and final:
                self.waiting.pop(rid).put(r)
        log.warning("the domain agent ended (%s)", self.proc.poll())

    def beat(self):
        while True:
            time.sleep(float(self.cfg["beat_s"]))
            if self.proc is None or self.proc.poll() is not None:
                return
            try:
                self.write({"clock": "beat"})
            except (BrokenPipeError, OSError):
                return

    def call(self, domain, args):
        """An intent to a package; the rules' outcome {status, text} (or a timeout)."""
        if domain.startswith("home_"):
            domain = domain[5:]
        if domain not in self.cfg["domains"]:
            return {"status": "error", "text": "no package %s (one of %s)" % (domain, ", ".join(self.cfg["domains"]))}
        args = dict(args or {})
        verb = args.pop("intent", None)
        if not verb:
            return {"status": "error", "text": "the call needs an intent"}
        rid = "t%d" % next(self.ids)
        q = queue.Queue(maxsize=1)
        self.waiting[rid] = q
        self.write({"id": rid, "domain": domain, "verb": verb, "args": args})
        try:
            r = q.get(timeout=float(self.cfg["call_timeout_s"]))
        except queue.Empty:
            self.waiting.pop(rid, None)
            return {"status": "pending", "text": "the rules have not decided yet; the outcome will come in REASONING"}
        if r.get("kind") == "error":
            return {"status": "error", "text": r.get("error")}
        return {"status": r["status"], "text": r.get("text") or ""}

    # ── the tool port ─────────────────────────────────────────────────────
    def serve_tools(self):
        side = self

        class H(socketserver.StreamRequestHandler):
            def handle(self):
                for raw in self.rfile:
                    try:
                        req = json.loads(raw)
                        if req.get("op") == "tools":
                            out = {"tools": side.tools}
                        elif req.get("op") == "call":
                            out = side.call(req.get("domain", ""), req.get("args"))
                        else:
                            out = {"status": "error", "text": "op is tools or call"}
                    except Exception as e:  # noqa: BLE001 — a bad request never ends the port
                        out = {"status": "error", "text": "%s: %s" % (type(e).__name__, e)}
                    self.wfile.write((json.dumps(out) + "\n").encode())
                    self.wfile.flush()

        socketserver.ThreadingTCPServer.allow_reuse_address = True
        srv = socketserver.ThreadingTCPServer(("127.0.0.1", int(self.cfg["tool_port"])), H)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True, name="metis-tools").start()
        log.info("tools on 127.0.0.1:%d", srv.server_address[1])
        return srv

    def run(self):
        self.serve_tools()
        while True:
            self.start_agent()
            self.write({"ports": {}, "domain": self.cfg["domains"][0], "id": "p0"})
            self.beat()
            for q in list(self.waiting.values()):
                q.put({"kind": "error", "error": "the rule engine restarted"})
            self.waiting.clear()
            log.warning("restarting the domain agent in 3 s")
            time.sleep(3)


def call_tool(port: int, domain: str, args: dict, timeout: float = 10.0) -> dict:
    """A tool call over the sidecar's port (fsttm.cloud uses it)."""
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as s:
        s.sendall((json.dumps({"op": "call", "domain": domain, "args": args}) + "\n").encode())
        return json.loads(s.makefile().readline())


def list_tools(port: int, timeout: float = 5.0) -> list:
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as s:
        s.sendall(b'{"op": "tools"}\n')
        return json.loads(s.makefile().readline()).get("tools") or []


def main(argv=None):
    ap = argparse.ArgumentParser("fsttm-metis", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    raw = yaml.safe_load(open(a.config)) or {}
    cfg = dict(DEFAULTS, **(raw.get("metis") or {}))
    port = int(((raw.get("streams") or {}).get("port")) or 8766)
    Sidecar(cfg, port).run()


if __name__ == "__main__":
    main()
