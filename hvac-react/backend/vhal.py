"""
VHAL property model for the HVAC backend.

This mirrors the semantics of the Android app's CarHvacManager
(de.porscheengineering.climate.hvac.CarHvacManager) and the
ClimateControlViewModel. Properties are addressed by a string name
(stable JSON key) and a vehicle area (zone) bitmask, exactly like the
Android VehiclePropertyIds / VehicleAreaSeat model.

The external Python interface and the React UI both speak in terms of
these property names + areas, so the protocol stays close to VHAL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# --- VehicleAreaSeat bitmask (matches android.car.VehicleAreaSeat) ----------
class Seat:
    ROW_1_LEFT = 0x0001
    ROW_1_RIGHT = 0x0004
    ROW_2_LEFT = 0x0010
    ROW_2_CENTER = 0x0020
    ROW_2_RIGHT = 0x0040

    LEFT_SEATS = ROW_1_LEFT | ROW_2_LEFT | ROW_2_CENTER
    RIGHT_SEATS = ROW_1_RIGHT | ROW_2_RIGHT


# --- VehicleAreaDoor bitmask (matches android.car.VehicleAreaDoor) -----------
class Door:
    ROW_1_LEFT  = 0x0001   # front-left  (driver)
    ROW_1_RIGHT = 0x0004   # front-right (passenger)
    ROW_2_LEFT  = 0x0010   # rear-left
    ROW_2_RIGHT = 0x0040   # rear-right
    REAR        = 0x0200   # trunk / liftgate


# --- VehicleAreaWindow bitmask -----------------------------------------------
class Window:
    ROW_1_LEFT  = 0x0001
    ROW_1_RIGHT = 0x0004
    ROW_2_LEFT  = 0x0010
    ROW_2_RIGHT = 0x0040
    # "global" / all-front zone used by the app for some single-zone props
    GLOBAL = ROW_1_LEFT | ROW_1_RIGHT


# --- Fan direction bitmask (matches VehicleHvacFanDirection) -----------------
class FanDirection:
    FACE = 0x1
    FLOOR = 0x2
    DEFROST = 0x4


# --- Value type tags ---------------------------------------------------------
TYPE_BOOL = "bool"
TYPE_INT = "int"
TYPE_FLOAT = "float"


@dataclass
class PropertyConfig:
    """Describes a single VHAL property, like a CarPropertyConfig."""

    name: str
    prop_type: str
    area_ids: List[int]
    writable: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    # supported enum values (used for fan direction etc.); None = free value
    enum_values: Optional[List[int]] = None
    # default value applied to every supported area at startup
    default: Any = None


# --- Tuning constants (match ClimateControlViewModel companion) --------------
MIN_FAN_SPEED = 1
MAX_FAN_SPEED = 7
FAN_SPEED_STEP = 1

MIN_TEMPERATURE = 16.0
MAX_TEMPERATURE = 28.0
TEMPERATURE_STEP = 0.5


def clip_temperature(value: float) -> float:
    clipped = max(MIN_TEMPERATURE, min(MAX_TEMPERATURE, value))
    # round to nearest 0.5, like clippedTemperature()
    return round(clipped * 2) / 2.0


def clip_fan_speed(value: int) -> int:
    return max(MIN_FAN_SPEED, min(MAX_FAN_SPEED, int(value)))


# --- Property catalogue ------------------------------------------------------
# Names are the JSON-stable identifiers used by the protocol. They map 1:1 to
# the CarHvacManager.ID_* constants.
FRONT_ZONES  = [Seat.ROW_1_LEFT, Seat.ROW_1_RIGHT]
GLOBAL_ZONE  = [0]
ALL_DOORS    = [Door.ROW_1_LEFT, Door.ROW_1_RIGHT,
                Door.ROW_2_LEFT, Door.ROW_2_RIGHT, Door.REAR]
CABIN_DOORS  = [Door.ROW_1_LEFT, Door.ROW_1_RIGHT,
                Door.ROW_2_LEFT, Door.ROW_2_RIGHT]
ALL_WINDOWS  = [Window.ROW_1_LEFT, Window.ROW_1_RIGHT,
                Window.ROW_2_LEFT, Window.ROW_2_RIGHT]

# Seat temperature/ventilation levels
MIN_SEAT_TEMP = 0   # 0 = off
MAX_SEAT_TEMP = 3
MIN_SEAT_VENT = 0
MAX_SEAT_VENT = 3

PROPERTY_CONFIGS: List[PropertyConfig] = [
    PropertyConfig("HVAC_TEMPERATURE_SET", TYPE_FLOAT, FRONT_ZONES,
                   min_value=MIN_TEMPERATURE, max_value=MAX_TEMPERATURE, default=22.0),
    PropertyConfig("HVAC_TEMPERATURE_CURRENT", TYPE_FLOAT, FRONT_ZONES,
                   min_value=MIN_TEMPERATURE, max_value=MAX_TEMPERATURE, default=22.0),
    PropertyConfig("HVAC_FAN_SPEED", TYPE_INT, FRONT_ZONES,
                   min_value=MIN_FAN_SPEED, max_value=MAX_FAN_SPEED, default=1),
    PropertyConfig("HVAC_FAN_DIRECTION", TYPE_INT, FRONT_ZONES,
                   enum_values=[FanDirection.FACE, FanDirection.FLOOR, FanDirection.DEFROST],
                   default=FanDirection.FACE),
    PropertyConfig("HVAC_AUTO_ON", TYPE_BOOL, FRONT_ZONES, default=False),
    PropertyConfig("HVAC_AC_ON", TYPE_BOOL, GLOBAL_ZONE, default=False),
    PropertyConfig("HVAC_RECIRC_ON", TYPE_BOOL, GLOBAL_ZONE, default=False),
    PropertyConfig("HVAC_MAX_AC_ON", TYPE_BOOL, GLOBAL_ZONE, default=False),
    PropertyConfig("HVAC_DUAL_ON", TYPE_BOOL, GLOBAL_ZONE, default=False),
    PropertyConfig("HVAC_MAX_DEFROST_ON", TYPE_BOOL, GLOBAL_ZONE, default=False),
    PropertyConfig("HVAC_DEFROSTER", TYPE_BOOL, [Seat.ROW_1_LEFT], default=False),
    PropertyConfig("HVAC_POWER_ON", TYPE_BOOL, GLOBAL_ZONE, default=False),
    PropertyConfig("HVAC_EXPERT_MODE", TYPE_BOOL, GLOBAL_ZONE, writable=True, default=False),
    PropertyConfig("ENV_OUTSIDE_TEMPERATURE", TYPE_FLOAT, GLOBAL_ZONE, default=18.0),

    # ── Doors (VehiclePropertyIds.DOOR_LOCK / DOOR_MOVE) ─────────────────────
    PropertyConfig("DOOR_LOCK", TYPE_BOOL, ALL_DOORS,  default=True),
    PropertyConfig("DOOR_MOVE", TYPE_INT,  CABIN_DOORS,
                   min_value=0, max_value=1, default=0),  # 0=closed, 1=open

    # ── Lights (global, area 0) ───────────────────────────────────────────────
    PropertyConfig("HEADLIGHTS_SWITCH",    TYPE_BOOL, GLOBAL_ZONE, default=False),
    PropertyConfig("HAZARD_LIGHTS_SWITCH", TYPE_BOOL, GLOBAL_ZONE, default=False),
    PropertyConfig("FOG_LIGHTS_SWITCH",    TYPE_BOOL, GLOBAL_ZONE, default=False),
    PropertyConfig("CABIN_LIGHTS_SWITCH",  TYPE_BOOL, GLOBAL_ZONE, default=False),

    # ── Seat comfort (per front seat) ─────────────────────────────────────────
    PropertyConfig("HVAC_SEAT_TEMPERATURE",  TYPE_INT, FRONT_ZONES,
                   min_value=MIN_SEAT_TEMP, max_value=MAX_SEAT_TEMP, default=0),
    PropertyConfig("HVAC_SEAT_VENTILATION",  TYPE_INT, FRONT_ZONES,
                   min_value=MIN_SEAT_VENT, max_value=MAX_SEAT_VENT, default=0),

    # ── Windows ──────────────────────────────────────────────────────────────
    PropertyConfig("WINDOW_LOCK", TYPE_BOOL, ALL_WINDOWS, default=False),
    PropertyConfig("WINDOW_MOVE", TYPE_INT,  ALL_WINDOWS,
                   min_value=0, max_value=100, default=0),  # 0=closed, 100=fully open
]

CONFIG_BY_NAME: Dict[str, PropertyConfig] = {c.name: c for c in PROPERTY_CONFIGS}


class VhalStore:
    """In-memory property store keyed by (name, area), like a VHAL backend."""

    def __init__(self) -> None:
        # values[name][area] = value
        self._values: Dict[str, Dict[int, Any]] = {}
        for cfg in PROPERTY_CONFIGS:
            self._values[cfg.name] = {area: cfg.default for area in cfg.area_ids}

    # -- raw access ----------------------------------------------------------
    def get(self, name: str, area: int) -> Any:
        cfg = CONFIG_BY_NAME.get(name)
        if cfg is None:
            raise KeyError(f"unknown property {name}")
        areas = self._values[name]
        if area in areas:
            return areas[area]
        # fall back to first supported area (mirrors zones.toList()[0] usage)
        return next(iter(areas.values()))

    def first_area(self, name: str) -> int:
        return next(iter(self._values[name].keys()))

    def supported_areas(self, name: str) -> List[int]:
        return list(self._values[name].keys())

    def _coerce(self, cfg: PropertyConfig, value: Any) -> Any:
        if cfg.prop_type == TYPE_BOOL:
            return bool(value)
        if cfg.prop_type == TYPE_INT:
            v = int(value)
            if cfg.min_value is not None:
                v = max(int(cfg.min_value), v)
            if cfg.max_value is not None:
                v = min(int(cfg.max_value), v)
            return v
        if cfg.prop_type == TYPE_FLOAT:
            v = float(value)
            if cfg.name == "HVAC_TEMPERATURE_SET":
                v = clip_temperature(v)
            elif cfg.min_value is not None and cfg.max_value is not None:
                v = max(cfg.min_value, min(cfg.max_value, v))
            return v
        return value

    def set(self, name: str, area: int, value: Any) -> List[Tuple[str, int, Any]]:
        """Set a property; returns the list of (name, area, value) changes made."""
        cfg = CONFIG_BY_NAME.get(name)
        if cfg is None:
            raise KeyError(f"unknown property {name}")
        if not cfg.writable:
            raise PermissionError(f"property {name} is not writable")

        coerced = self._coerce(cfg, value)
        changes: List[Tuple[str, int, Any]] = []
        # area==0 means "apply to all supported areas" (global/broadcast)
        targets = cfg.area_ids if area == 0 else [a for a in cfg.area_ids if a & area]
        if not targets:
            # area not in catalogue but explicitly addressed: store under it anyway
            targets = [area]
        for a in targets:
            if self._values[name].get(a) != coerced:
                self._values[name][a] = coerced
                changes.append((name, a, coerced))
        return changes

    # -- snapshot ------------------------------------------------------------
    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Full state as {name: {str(area): value}} for JSON transport."""
        out: Dict[str, Dict[str, Any]] = {}
        for name, areas in self._values.items():
            out[name] = {str(area): val for area, val in areas.items()}
        return out
