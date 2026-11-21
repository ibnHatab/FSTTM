"""
Home Assistant → the ENVIRONMENT stream (fsttm.streams).

Connects to Home Assistant's websocket API, reads every state once, then
follows state_changed; each entity that passes the filter is written to the
stream door as {"stream": "env", "key": <entity_id>, "text": <its state in
words>}. A removed entity is written with text null. Reconnects forever: Home
Assistant restarts, the NUC boots in any order.

    python -m fsttm.feeds.hass --config config.nuc.yaml

Config (the `hass` block; the token is a file, never the config):
    hass:
        url: ws://127.0.0.1:8123/api/websocket
        token_file: ~/.config/fsttm/hass.token
        domains: [fan, light, switch, media_player, climate, cover, lock, sensor, binary_sensor]
        exclude: [sensor.sun_*, "*_display_brightness"]   # fnmatch patterns on entity_id
        attributes: [percentage, preset_mode, brightness, temperature, current_temperature,
                     hvac_action, media_title, source, volume_level, unit_of_measurement]
    streams: {port: 8766}

Needs the `websockets` package (pip install fsttm[hass]).
"""
from __future__ import annotations

import argparse
import asyncio
import fnmatch
import json
import logging
import os
import socket
import time

import yaml

log = logging.getLogger("fsttm.feeds.hass")

DEFAULTS = {
    "url": "ws://127.0.0.1:8123/api/websocket",
    "token_file": "~/.config/fsttm/hass.token",
    "domains": ["fan", "light", "switch", "media_player", "climate", "cover", "lock",
                "sensor", "binary_sensor", "vacuum", "water_heater"],
    "exclude": ["sensor.sun_*", "*_display_brightness", "sensor.backup_*", "event.*"],
    "attributes": ["percentage", "preset_mode", "brightness", "temperature", "current_temperature",
                   "hvac_action", "media_title", "source", "volume_level", "unit_of_measurement"],
}


def wanted(entity_id: str, cfg: dict) -> bool:
    domain = entity_id.split(".", 1)[0]
    if domain not in cfg["domains"]:
        return False
    return not any(fnmatch.fnmatch(entity_id, p) for p in cfg["exclude"])


def words(state: dict, cfg: dict) -> str:
    """An entity's state as one short line: its name, its state, the attributes that matter."""
    a = state.get("attributes") or {}
    name = a.get("friendly_name") or state["entity_id"]
    value = state.get("state")
    unit = a.get("unit_of_measurement")
    head = "%s: %s%s" % (name, value, (" " + unit) if unit and value not in ("unavailable", "unknown") else "")
    extra = []
    for k in cfg["attributes"]:
        if k == "unit_of_measurement" or a.get(k) in (None, "", []):
            continue
        v = a[k]
        if k == "brightness" and isinstance(v, (int, float)):
            v = "%d%%" % round(v * 100 / 255)
        elif k == "percentage":
            v = "%s%%" % v
        extra.append("%s %s" % (k.replace("_", " "), v))
    return head + (" (" + ", ".join(extra) + ")" if extra else "")


class Door:
    """A connection to fsttm's stream door, reopened when the server restarts. A new
    connection first carries the whole current state (`known`: entity -> words), so a
    voice process that started after this one, or restarted, is current at once."""

    def __init__(self, port: int, known: dict):
        self.port, self.sock, self.known = port, None, known

    def send(self, events):
        for _ in range(2):
            try:
                fresh = self.sock is None
                if fresh:
                    self.sock = socket.create_connection(("127.0.0.1", self.port), timeout=5)
                    events = [{"stream": "env", "key": k, "text": t} for k, t in self.known.items()
                              if t is not None]
                    log.info("door connected: %d entities", len(events))
                data = "".join(json.dumps(e) + "\n" for e in events).encode()
                self.sock.sendall(data)
                return True
            except OSError:
                if self.sock is not None:
                    self.sock.close()
                self.sock = None
        return False


async def follow(cfg: dict, door: Door, last: dict):
    import websockets
    token = open(os.path.expanduser(cfg["token_file"])).read().strip()
    async with websockets.connect(cfg["url"], max_size=2 ** 24) as ws:
        hello = json.loads(await ws.recv())
        if hello.get("type") != "auth_required":
            raise RuntimeError("unexpected greeting: %s" % hello)
        await ws.send(json.dumps({"type": "auth", "access_token": token}))
        ok = json.loads(await ws.recv())
        if ok.get("type") != "auth_ok":
            raise PermissionError("Home Assistant refused the token: %s" % ok.get("message"))
        await ws.send(json.dumps({"id": 1, "type": "get_states"}))
        await ws.send(json.dumps({"id": 2, "type": "subscribe_events", "event_type": "state_changed"}))
        log.info("connected to %s (Home Assistant %s)", cfg["url"], ok.get("ha_version"))
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("id") == 1 and msg.get("type") == "result":
                batch = []
                for st in msg.get("result") or ():
                    if wanted(st["entity_id"], cfg):
                        text = words(st, cfg)
                        last[st["entity_id"]] = text
                        batch.append({"stream": "env", "key": st["entity_id"], "text": text})
                door.send(batch)
                log.info("sent %d entities", len(batch))
            elif msg.get("type") == "event":
                d = msg["event"]["data"]
                eid = d["entity_id"]
                if not wanted(eid, cfg):
                    continue
                new = d.get("new_state")
                text = words(new, cfg) if new else None
                if last.get(eid) == text:
                    continue                       # attribute churn that does not change the words
                last[eid] = text
                door.send([{"stream": "env", "key": eid, "text": text}])
                log.info("%s -> %s", eid, text)


async def keep_door(door: Door, every: float = 5.0):
    """While the door is closed (the voice process not up yet, or restarting), knock every
    few seconds; the first connection carries the whole state."""
    while True:
        await asyncio.sleep(every)
        if door.sock is None and door.known:
            door.send([])


async def run(cfg: dict, port: int):
    last, wait = {}, 1.0
    door = Door(port, last)
    asyncio.get_running_loop().create_task(keep_door(door))
    while True:
        try:
            await follow(cfg, door, last)
        except PermissionError:
            raise
        except Exception as e:  # noqa: BLE001 — Home Assistant away: try again
            log.warning("Home Assistant: %s; again in %.0f s", e, wait)
            await asyncio.sleep(wait)
            wait = min(wait * 2, 30.0)
            continue
        wait = 1.0


def main(argv=None):
    ap = argparse.ArgumentParser("fsttm-hass", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    raw = yaml.safe_load(open(a.config)) or {}
    cfg = dict(DEFAULTS, **(raw.get("hass") or {}))
    port = int(((raw.get("streams") or {}).get("port")) or 8766)
    asyncio.run(run(cfg, port))


if __name__ == "__main__":
    main()
