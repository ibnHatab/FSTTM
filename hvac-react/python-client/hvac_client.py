"""
Example external Python interface to the HVAC backend.

Demonstrates controlling the HVAC over the JSON/REST protocol exactly as an
external process would. Run the backend first (see backend/README equivalent
in the project README), then:

    python hvac_client.py demo
    python hvac_client.py set HVAC_FAN_SPEED 1 5      # name area value
    python hvac_client.py action power_toggle
    python hvac_client.py state
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict
from urllib import request

BASE = "http://127.0.0.1:8000"

# Vehicle area seat bitmask (mirrors android.car.VehicleAreaSeat)
ROW_1_LEFT = 0x0001
ROW_1_RIGHT = 0x0004

# Fan direction bits
FACE = 0x1
FLOOR = 0x2
DEFROST = 0x4


def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    data = json.dumps(payload).encode()
    req = request.Request(BASE + path, data=data,
                          headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def _get(path: str) -> Dict[str, Any]:
    with request.urlopen(BASE + path) as resp:
        return json.loads(resp.read().decode())


class HvacClient:
    """Thin wrapper around the backend command protocol."""

    def state(self) -> Dict[str, Any]:
        return _get("/state")

    def config(self) -> Dict[str, Any]:
        return _get("/config")

    def set_prop(self, name: str, area: int, value: Any) -> Dict[str, Any]:
        return _post("/command", {"cmd": "set", "name": name, "area": area, "value": value})

    def action(self, action: str, **args: Any) -> Dict[str, Any]:
        return _post("/command", {"cmd": "action", "action": action, "args": args})

    # convenience helpers ----------------------------------------------------
    def set_temperature(self, value: float, area: int = ROW_1_LEFT | ROW_1_RIGHT):
        return self.action("update_temperature", value=value, area=area)

    def set_fan_speed(self, value: int, area: int = ROW_1_LEFT | ROW_1_RIGHT):
        return self.action("update_fan_speed", value=value, area=area)

    def toggle_fan_direction(self, direction: int, area: int = ROW_1_LEFT):
        return self.action("fan_direction_toggle", direction=direction, area=area)

    def power(self):
        return self.action("power_toggle")

    def ac(self):
        return self.action("ac_toggle")

    def auto(self, area: int = ROW_1_LEFT):
        return self.action("auto_toggle", area=area)

    def expert_mode(self, on: bool):
        return self.action("set_expert_mode", on=on)


def _demo() -> None:
    c = HvacClient()
    print("Turning power on...")
    c.power()
    print("Setting temperature to 21.5 (both zones)...")
    c.set_temperature(21.5)
    print("Setting fan speed to 4...")
    c.set_fan_speed(4)
    print("Enabling A/C...")
    c.ac()
    print("Toggling FACE airflow on left seat...")
    c.toggle_fan_direction(FACE, ROW_1_LEFT)
    print("Switching to expert mode...")
    c.expert_mode(True)
    print("\nCurrent state:")
    print(json.dumps(c.state()["props"], indent=2))


def main(argv: list[str]) -> None:
    c = HvacClient()
    if not argv or argv[0] == "demo":
        _demo()
    elif argv[0] == "state":
        print(json.dumps(c.state()["props"], indent=2))
    elif argv[0] == "config":
        print(json.dumps(c.config(), indent=2))
    elif argv[0] == "set" and len(argv) == 4:
        name, area, value = argv[1], int(argv[2], 0), argv[3]
        # try to parse value as number/bool
        parsed: Any
        if value.lower() in ("true", "false"):
            parsed = value.lower() == "true"
        else:
            try:
                parsed = float(value) if "." in value else int(value)
            except ValueError:
                parsed = value
        print(json.dumps(c.set_prop(name, area, parsed), indent=2))
    elif argv[0] == "action" and len(argv) >= 2:
        kwargs = {}
        for kv in argv[2:]:
            k, _, v = kv.partition("=")
            if v.lower() in ("true", "false"):
                kwargs[k] = v.lower() == "true"
            else:
                try:
                    kwargs[k] = int(v, 0)
                except ValueError:
                    kwargs[k] = v
        print(json.dumps(c.action(argv[1], **kwargs), indent=2))
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
