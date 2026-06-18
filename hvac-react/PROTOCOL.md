# HVAC Control Protocol & State Management

This document describes the JSON control protocol exposed by the backend and
how state flows through the whole system (external Python client ↔ backend ↔
React UI). It is the contract any external controller must follow.

---

## 1. Architecture overview

```
                    ┌──────────────────────────────────────────┐
                    │              backend (FastAPI)             │
  Python client ───►│  REST  /command /set /state /config       │
  (or any HTTP)     │                                            │
                    │  ┌──────────────┐   ┌──────────────────┐   │
   React UI  ◄─────►│  │  VhalStore   │◄──│  HvacController   │   │  WebSocket /ws
   (browser)  WS    │  │ (truth: the  │   │ (high-level ops,  │   │  push deltas
                    │  │  prop map)   │   │  mirrors VM)      │   │
                    │  └──────────────┘   └──────────────────┘   │
                    │         │  every change → Hub.broadcast()   │
                    └─────────┼──────────────────────────────────┘
                              ▼
                    all connected WebSocket clients
```

**Single source of truth:** the backend `VhalStore`. There is *no* independent
client-side truth. The React UI holds only a mirror that is overwritten by
snapshots/deltas from the server. Any mutation — whether it originates from the
UI or from an external Python client — goes to the store, and the resulting
change is broadcast to *every* connected client, so all views stay consistent.

---

## 2. The property model (VHAL)

State is a flat map keyed by **property name** and **area** (zone), mirroring
Android's `VehiclePropertyIds` + `VehicleAreaSeat`.

```
props[<PROPERTY_NAME>][<area>] = <value>
```

### Properties (← `CarHvacManager.ID_*`)

| Name                       | Type  | Areas        | Writable | Range / values |
|----------------------------|-------|--------------|----------|----------------|
| `HVAC_TEMPERATURE_SET`     | float | 1, 4         | yes      | 16.0–28.0, snaps to 0.5 |
| `HVAC_TEMPERATURE_CURRENT` | float | 1, 4         | yes      | 16.0–28.0 |
| `HVAC_FAN_SPEED`           | int   | 1, 4         | yes      | 1–7 |
| `HVAC_FAN_DIRECTION`       | int   | 1, 4         | yes      | bitmask: FACE=1, FLOOR=2, DEFROST=4 |
| `HVAC_AUTO_ON`             | bool  | 1, 4         | yes      | — |
| `HVAC_AC_ON`               | bool  | 0 (global)   | yes      | — |
| `HVAC_RECIRC_ON`           | bool  | 0            | yes      | — |
| `HVAC_MAX_AC_ON`           | bool  | 0            | yes      | — |
| `HVAC_DUAL_ON`             | bool  | 0            | yes      | — |
| `HVAC_MAX_DEFROST_ON`      | bool  | 0            | yes      | — |
| `HVAC_DEFROSTER`           | bool  | 1            | yes      | rear-window defrost |
| `HVAC_POWER_ON`            | bool  | 0            | yes      | — |
| `HVAC_EXPERT_MODE`         | bool  | 0            | yes      | drives UI screen |
| `ENV_OUTSIDE_TEMPERATURE`  | float | 0            | yes      | — |

### Areas (VehicleAreaSeat bitmask)

| Constant       | Value |
|----------------|-------|
| `ROW_1_LEFT`   | `0x1` |
| `ROW_1_RIGHT`  | `0x4` |
| both front     | `0x5` (`0x1 | 0x4`) |
| **broadcast**  | `0`   |

**Area `0` in a command means "apply to all supported zones of this property."**
For a per-zone property like `HVAC_FAN_SPEED`, `area:0` writes both `1` and `4`.
For a global property like `HVAC_AC_ON`, the only area is `0` anyway.

When a command targets a multi-zone area mask (e.g. `0x5`), the store applies
the value to every supported area whose bit intersects the mask.

The catalogue is discoverable at runtime via `GET /config` — clients should not
hard-code ranges if they can query them.

---

## 3. Transports

Two transports expose the **same** command vocabulary:

| Transport | Use | Endpoint |
|-----------|-----|----------|
| REST      | external control, scripting, one-shot reads | `POST /command`, `POST /set/{name}`, `GET /state`, `GET /config` |
| WebSocket | live UI, bidirectional, push updates | `ws://<host>/ws` |

Both feed the same `dispatch()` function, so anything you can do over REST you
can do over WS and vice-versa.

---

## 4. Commands (client → server)

A command is a JSON object with a `cmd` discriminator. There are three kinds.

### 4.1 `set` — write a raw property

```json
{ "cmd": "set", "name": "HVAC_FAN_SPEED", "area": 1, "value": 5 }
```

- `area` defaults to `0` (broadcast) if omitted.
- The value is **coerced and clipped** to the property's type/range
  (e.g. temperature snaps to the nearest 0.5 and is clamped to 16–28).
- Non-writable properties are rejected (`400` / WS error).

### 4.2 `action` — high-level operation

These mirror `ClimateControlViewModel` semantics (they encapsulate the bumping,
toggling, and coupling logic rather than writing raw values).

```json
{ "cmd": "action", "action": "bump_temperature", "args": { "up": true, "area": 1 } }
```

| Action | Args | Behaviour |
|--------|------|-----------|
| `bump_temperature`      | `up:bool, area:int` | ±0.5 °C, clipped |
| `update_temperature`    | `value:float, area:int` | set + clip |
| `bump_fan_speed`        | `up:bool, area:int` | ±1, clamped 1–7 |
| `update_fan_speed`      | `value:int, area:int` | set + clamp |
| `fan_direction_toggle`  | `direction:int, area:int` | XOR the direction bit; if it would clear all bits, toggle AUTO for that side instead |
| `auto_toggle`           | `area:int` | flip `HVAC_AUTO_ON` for that side |
| `dual_toggle`           | — | flip `HVAC_DUAL_ON` (UI "SYNC" = `!dual`) |
| `ac_toggle`             | — | flip `HVAC_AC_ON` |
| `recirc_toggle`         | — | flip `HVAC_RECIRC_ON` |
| `ac_max_toggle`         | — | flip `HVAC_MAX_AC_ON` |
| `max_defrost_toggle`    | — | flip `HVAC_MAX_DEFROST_ON` |
| `window_defrost_toggle` | — | flip `HVAC_DEFROSTER` (rear) |
| `power_toggle`          | — | flip `HVAC_POWER_ON` |
| `set_expert_mode`       | `on:bool` | set `HVAC_EXPERT_MODE` (switches the UI screen) |

Missing required args raise an error (`KeyError` → `400` over REST, error frame
over WS). Unknown action names are rejected.

### 4.3 `get_state` — request a fresh snapshot (WS only, meaningful)

```json
{ "cmd": "get_state" }
```

Over WS the server replies with a full `state` message. Over REST, use
`GET /state` instead.

---

## 5. Server → client messages

### 5.1 `state` — full snapshot

Sent once immediately on WebSocket connect, and in response to `get_state`.

```json
{
  "type": "state",
  "props": {
    "HVAC_TEMPERATURE_SET": { "1": 22.0, "4": 22.0 },
    "HVAC_FAN_SPEED":       { "1": 1,    "4": 1 },
    "HVAC_AC_ON":           { "0": false },
    "...": {}
  }
}
```

> Note: area keys are **strings** in JSON (`"1"`, `"4"`, `"0"`).

### 5.2 `delta` — incremental change

Broadcast to **all** connected clients after any successful mutation (from any
source). This is the mechanism that keeps the UI and external controllers in
sync.

```json
{
  "type": "delta",
  "changes": [
    { "name": "HVAC_FAN_SPEED", "area": 1, "value": 5 },
    { "name": "HVAC_FAN_SPEED", "area": 4, "value": 5 }
  ]
}
```

- A single command can produce multiple changes (e.g. a broadcast write, or
  `bump_temperature` on both zones).
- If a command results in **no actual change** (value already equal), the
  `changes` list is empty and **nothing is broadcast** — the store de-dupes.

### 5.3 `error` — command rejected (WS only)

```json
{ "type": "error", "message": "property HVAC_FAN_SPEED is not writable" }
```

Over REST the equivalent is an HTTP `400` with `{ "error": "..." }`.

---

## 6. REST endpoint reference

| Method | Path | Body | Response |
|--------|------|------|----------|
| GET  | `/state`  | — | `{type:"state", props:{...}}` |
| GET  | `/config` | — | `{properties:[{name,type,areas,writable,min,max,enum}]}` |
| POST | `/command`| a command object (§4) | `{ok:true, changes:[...], props:{...}}` or `400 {error}` |
| POST | `/set/{name}` | `{area, value}` | `{ok:true, changes:[...]}` or `400 {error}` |
| GET  | `/` | — | service banner |

`POST /command` returns the full post-change snapshot in `props`, convenient for
fire-and-forget scripting without opening a WebSocket. It **also** broadcasts the
delta to WS clients.

---

## 7. State management — detailed flow

### Backend (authoritative)

- **`VhalStore`** (`backend/vhal.py`) holds `values[name][area]`. It is the only
  authoritative state. `set()` coerces/clips, writes, and returns the list of
  `(name, area, value)` tuples that actually changed (empty if no-op).
- **`HvacController`** (`backend/controller.py`) implements the high-level
  actions on top of the store (bump/clip, fan-direction XOR with the auto-on
  fallback, sync logic). It never holds state itself — it reads and writes the
  store.
- **`Hub`** (`backend/server.py`) tracks connected WebSockets and broadcasts
  `delta` frames. After *every* mutation (`_broadcast_changes`) the deltas go to
  all clients. Dead sockets are pruned on send failure.

There is no persistence — state is in-memory and resets to the property
`default`s on server restart.

### Frontend (mirror only)

`frontend/src/App.jsx` holds the entire client state layer inline:

- Opens `/ws` (through the Vite dev proxy), auto-reconnects ~5 s after a close.
- On `state` → replaces the whole local `vhalProps` map.
- On `delta` → merges each change into `vhalProps` immutably:
  `next[name][area] = value`.
- `sendCommand(payload)` sends a command frame (over WS, or POST `/command` as a
  fallback when the socket is down) — the UI never optimistically mutates local
  state. It waits for the echoed `delta`. This guarantees the UI reflects the
  server's coerced value (e.g. a temperature clipped to 0.5), not the raw input.
- `getPropVal(name, area, fallback)` is the read selector, with a fallback to the
  first stored area when an area isn't present.

Because the UI is a pure mirror with no optimistic writes, an external Python
client and the browser are always showing the same state, and either can drive
it.

---

## 8. Examples

### curl

```bash
# read everything
curl -s localhost:8000/state | jq

# set fan speed on both front zones (broadcast)
curl -s localhost:8000/command -H 'Content-Type: application/json' \
  -d '{"cmd":"set","name":"HVAC_FAN_SPEED","area":0,"value":4}'

# bump driver temperature up half a degree
curl -s localhost:8000/command -H 'Content-Type: application/json' \
  -d '{"cmd":"action","action":"bump_temperature","args":{"up":true,"area":1}}'

# turn on A/C
curl -s localhost:8000/command -H 'Content-Type: application/json' \
  -d '{"cmd":"action","action":"ac_toggle"}'
```

### Python (example client)

```python
from hvac_client import HvacClient, ROW_1_LEFT, FACE
c = HvacClient()
c.power()                       # action power_toggle
c.set_temperature(21.5)         # both front zones
c.set_fan_speed(4)
c.toggle_fan_direction(FACE, ROW_1_LEFT)
c.expert_mode(True)             # UI switches to Expert screen
print(c.state()["props"])
```

### JavaScript / WebSocket

```js
const ws = new WebSocket("ws://localhost:8000/ws")
ws.onmessage = (e) => {
  const m = JSON.parse(e.data)
  if (m.type === "state") console.log("snapshot", m.props)
  if (m.type === "delta") console.log("changes", m.changes)
}
ws.onopen = () =>
  ws.send(JSON.stringify({ cmd: "action", action: "power_toggle" }))
```
