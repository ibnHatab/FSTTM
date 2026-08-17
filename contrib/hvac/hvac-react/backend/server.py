"""
HVAC backend server.

Exposes the VHAL property store over:
  - REST (JSON)        : for external control (e.g. the Python client) and the UI
  - WebSocket (JSON)   : live state push to the React UI and any other listener

Protocol (JSON):
  State snapshot message (server -> client):
    {"type": "state", "props": {<NAME>: {<area>: <value>, ...}, ...}}
  Delta message (server -> client), sent after any change:
    {"type": "delta", "changes": [{"name": NAME, "area": INT, "value": VAL}, ...]}

  Commands (client -> server, over WS or POST /command):
    {"cmd": "set",  "name": NAME, "area": INT, "value": VAL}
    {"cmd": "action", "action": ACTION, "args": {...}}     # high-level ops
    {"cmd": "get_state"}                                    # request snapshot

  High-level actions mirror ClimateControlViewModel:
    bump_temperature {up: bool, area: int}
    update_temperature {value: float, area: int}
    bump_fan_speed {up: bool, area: int}
    update_fan_speed {value: int, area: int}
    fan_direction_toggle {direction: int, area: int}
    auto_toggle {area: int}
    dual_toggle | ac_toggle | recirc_toggle | ac_max_toggle |
    max_defrost_toggle | window_defrost_toggle | power_toggle
    set_expert_mode {on: bool}
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from controller import HvacController
from vhal import CONFIG_BY_NAME, PROPERTY_CONFIGS, VhalStore

store = VhalStore()
controller = HvacController(store)

app = FastAPI(title="HVAC VHAL Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- connection manager ------------------------------------------------------
class Hub:
    def __init__(self) -> None:
        self.clients: Set[WebSocket] = set()
        self.lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self.lock:
            self.clients.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self.lock:
            self.clients.discard(ws)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        data = json.dumps(message)
        dead = []
        async with self.lock:
            targets = list(self.clients)
        for ws in targets:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)


hub = Hub()


def _changes_to_json(changes: List) -> List[Dict[str, Any]]:
    return [{"name": n, "area": a, "value": v} for (n, a, v) in changes]


async def _broadcast_changes(changes: List) -> None:
    if changes:
        await hub.broadcast({"type": "delta", "changes": _changes_to_json(changes)})


# --- command dispatch (shared by REST + WS) ----------------------------------
ACTION_HANDLERS = {
    "bump_temperature": lambda a: controller.bump_temperature(bool(a["up"]), int(a["area"])),
    "update_temperature": lambda a: controller.update_temperature(float(a["value"]), int(a["area"])),
    "bump_fan_speed": lambda a: controller.bump_fan_speed(bool(a["up"]), int(a["area"])),
    "update_fan_speed": lambda a: controller.update_fan_speed(int(a["value"]), int(a["area"])),
    "fan_direction_toggle": lambda a: controller.fan_direction_toggle(int(a["direction"]), int(a["area"])),
    "auto_toggle": lambda a: controller.auto_toggle(int(a["area"])),
    "dual_toggle": lambda a: controller.dual_toggle(),
    "ac_toggle": lambda a: controller.ac_toggle(),
    "recirc_toggle": lambda a: controller.recirc_toggle(),
    "ac_max_toggle": lambda a: controller.ac_max_toggle(),
    "max_defrost_toggle": lambda a: controller.max_defrost_toggle(),
    "window_defrost_toggle": lambda a: controller.window_defrost_toggle(),
    "power_toggle": lambda a: controller.power_toggle(),
    "set_expert_mode": lambda a: controller.set_expert_mode(bool(a["on"])),
    # ── Doors ─────────────────────────────────────────────────────────────────
    "door_lock":   lambda a: controller.door_lock(int(a.get("area", 0)), bool(a["locked"])),
    "door_move":   lambda a: controller.door_move(int(a.get("area", 0)), bool(a["open"])),
    # ── Lights ────────────────────────────────────────────────────────────────
    "headlights_toggle":    lambda a: controller.headlights_toggle(),
    "hazard_toggle":        lambda a: controller.hazard_toggle(),
    "fog_toggle":           lambda a: controller.fog_toggle(),
    "cabin_lights_toggle":  lambda a: controller.cabin_lights_toggle(),
    # ── Seat comfort ──────────────────────────────────────────────────────────
    "bump_seat_temp": lambda a: controller.bump_seat_temp(bool(a["up"]), int(a.get("area", 0))),
    "bump_seat_vent": lambda a: controller.bump_seat_vent(bool(a["up"]), int(a.get("area", 0))),
    # ── Windows ───────────────────────────────────────────────────────────────
    "window_move": lambda a: controller.window_move(int(a.get("area", 0)), int(a["position"])),
    "window_lock": lambda a: controller.window_lock(int(a.get("area", 0)), bool(a["locked"])),
}


def dispatch(cmd: Dict[str, Any]) -> List:
    """Run a command dict; return list of changes. Raises on bad input."""
    kind = cmd.get("cmd")
    if kind == "set":
        return store.set(cmd["name"], int(cmd.get("area", 0)), cmd["value"])
    if kind == "action":
        action = cmd["action"]
        handler = ACTION_HANDLERS.get(action)
        if handler is None:
            raise ValueError(f"unknown action {action}")
        return handler(cmd.get("args", {}) or {})
    if kind == "get_state":
        return []
    raise ValueError(f"unknown command {kind}")


# --- REST --------------------------------------------------------------------
@app.get("/state")
def get_state() -> Dict[str, Any]:
    return {"type": "state", "props": store.snapshot()}


@app.get("/config")
def get_config() -> Dict[str, Any]:
    return {
        "properties": [
            {
                "name": c.name,
                "type": c.prop_type,
                "areas": c.area_ids,
                "writable": c.writable,
                "min": c.min_value,
                "max": c.max_value,
                "enum": c.enum_values,
            }
            for c in PROPERTY_CONFIGS
        ]
    }


class CommandIn(BaseModel):
    cmd: str
    name: str | None = None
    area: int | None = 0
    value: Any | None = None
    action: str | None = None
    args: Dict[str, Any] | None = None


@app.post("/command")
async def post_command(command: CommandIn) -> JSONResponse:
    try:
        changes = dispatch(command.model_dump(exclude_none=False))
    except (KeyError, ValueError, PermissionError) as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    await _broadcast_changes(changes)
    return JSONResponse(content={
        "ok": True,
        "changes": _changes_to_json(changes),
        "props": store.snapshot(),
    })


# Convenience: directly set one property via REST
@app.post("/set/{name}")
async def set_prop(name: str, body: Dict[str, Any]) -> JSONResponse:
    try:
        changes = store.set(name, int(body.get("area", 0)), body["value"])
    except (KeyError, ValueError, PermissionError) as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    await _broadcast_changes(changes)
    return JSONResponse(content={"ok": True, "changes": _changes_to_json(changes)})


# --- WebSocket ---------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await hub.connect(ws)
    # send initial snapshot
    await ws.send_text(json.dumps({"type": "state", "props": store.snapshot()}))
    try:
        while True:
            raw = await ws.receive_text()
            try:
                cmd = json.loads(raw)
                changes = dispatch(cmd)
                if cmd.get("cmd") == "get_state":
                    await ws.send_text(json.dumps({"type": "state", "props": store.snapshot()}))
                else:
                    await _broadcast_changes(changes)
            except (KeyError, ValueError, PermissionError, json.JSONDecodeError) as exc:
                await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
    except WebSocketDisconnect:
        await hub.disconnect(ws)


@app.get("/")
def root() -> Dict[str, str]:
    return {"service": "HVAC VHAL backend", "ws": "/ws", "state": "/state"}
