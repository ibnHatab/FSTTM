"""
High-level HVAC operations, mirroring ClimateControlViewModel.

These wrap the raw VhalStore set/get calls with the same behaviour the
Android ViewModel implements: temperature/fan bumping with clipping,
fan-direction bit toggling (with the auto-on coupling), AUTO/SYNC logic, etc.

Every mutating method returns the flat list of (name, area, value) changes so
the server can broadcast deltas to connected clients.
"""

from __future__ import annotations

from typing import Any, List, Tuple

from vhal import (
    FanDirection,
    Seat, Door, Window,
    VhalStore,
    FAN_SPEED_STEP,
    TEMPERATURE_STEP,
    MIN_SEAT_TEMP, MAX_SEAT_TEMP,
    MIN_SEAT_VENT, MAX_SEAT_VENT,
    CABIN_DOORS, ALL_WINDOWS,
    clip_fan_speed,
    clip_temperature,
)

Change = Tuple[str, int, Any]


class HvacController:
    def __init__(self, store: VhalStore) -> None:
        self.store = store

    # -- temperature ---------------------------------------------------------
    def update_temperature(self, value: float, area_seat: int) -> List[Change]:
        return self.store.set("HVAC_TEMPERATURE_SET", area_seat, clip_temperature(value))

    def bump_temperature(self, up: bool, area_seat: int) -> List[Change]:
        changes: List[Change] = []
        step = TEMPERATURE_STEP if up else -TEMPERATURE_STEP
        if area_seat & Seat.ROW_1_LEFT:
            cur = self.store.get("HVAC_TEMPERATURE_SET", Seat.ROW_1_LEFT)
            changes += self.update_temperature(cur + step, Seat.ROW_1_LEFT)
        if area_seat & Seat.ROW_1_RIGHT:
            cur = self.store.get("HVAC_TEMPERATURE_SET", Seat.ROW_1_RIGHT)
            changes += self.update_temperature(cur + step, Seat.ROW_1_RIGHT)
        return changes

    # -- fan speed -----------------------------------------------------------
    def update_fan_speed(self, value: int, area_seat: int) -> List[Change]:
        return self.store.set("HVAC_FAN_SPEED", area_seat, clip_fan_speed(value))

    def bump_fan_speed(self, up: bool, area_seat: int) -> List[Change]:
        changes: List[Change] = []
        step = FAN_SPEED_STEP if up else -FAN_SPEED_STEP
        if area_seat & Seat.ROW_1_LEFT:
            cur = self.store.get("HVAC_FAN_SPEED", Seat.ROW_1_LEFT)
            changes += self.update_fan_speed(cur + step, Seat.ROW_1_LEFT)
        if area_seat & Seat.ROW_1_RIGHT:
            cur = self.store.get("HVAC_FAN_SPEED", Seat.ROW_1_RIGHT)
            changes += self.update_fan_speed(cur + step, Seat.ROW_1_RIGHT)
        return changes

    # -- fan direction (bit toggle) ------------------------------------------
    def fan_direction_toggle(self, direction: int, area_seat: int) -> List[Change]:
        """XOR-toggle a direction bit. If toggling would clear all bits, instead
        toggle AUTO for that side (matches funDirectionToggle)."""
        if area_seat & Seat.ROW_1_LEFT:
            cur = self.store.get("HVAC_FAN_DIRECTION", Seat.ROW_1_LEFT)
            if (cur ^ direction) == 0:
                auto = not self.store.get("HVAC_AUTO_ON", Seat.ROW_1_LEFT)
                return self._set_auto(Seat.ROW_1_LEFT, auto)
            return self.store.set("HVAC_FAN_DIRECTION", Seat.ROW_1_LEFT, cur ^ direction)
        if area_seat & Seat.ROW_1_RIGHT:
            cur = self.store.get("HVAC_FAN_DIRECTION", Seat.ROW_1_RIGHT)
            if (cur ^ direction) == 0:
                auto = not self.store.get("HVAC_AUTO_ON", Seat.ROW_1_RIGHT)
                return self._set_auto(Seat.ROW_1_RIGHT, auto)
            return self.store.set("HVAC_FAN_DIRECTION", Seat.ROW_1_RIGHT, cur ^ direction)
        return []

    # -- auto / sync ---------------------------------------------------------
    def _set_auto(self, area_seat: int, on: bool) -> List[Change]:
        return self.store.set("HVAC_AUTO_ON", area_seat, on)

    def auto_toggle(self, area_seat: int) -> List[Change]:
        cur = self.store.get("HVAC_AUTO_ON", area_seat)
        return self._set_auto(area_seat, not cur)

    def dual_toggle(self) -> List[Change]:
        cur = self.store.get("HVAC_DUAL_ON", self.store.first_area("HVAC_DUAL_ON"))
        return self.store.set("HVAC_DUAL_ON", 0, not cur)

    # -- simple boolean toggles ----------------------------------------------
    def _toggle_bool(self, name: str) -> List[Change]:
        cur = self.store.get(name, self.store.first_area(name))
        return self.store.set(name, 0, not cur)

    def ac_toggle(self) -> List[Change]:
        return self._toggle_bool("HVAC_AC_ON")

    def recirc_toggle(self) -> List[Change]:
        return self._toggle_bool("HVAC_RECIRC_ON")

    def ac_max_toggle(self) -> List[Change]:
        return self._toggle_bool("HVAC_MAX_AC_ON")

    def max_defrost_toggle(self) -> List[Change]:
        return self._toggle_bool("HVAC_MAX_DEFROST_ON")

    def window_defrost_toggle(self) -> List[Change]:
        cur = self.store.get("HVAC_DEFROSTER", Seat.ROW_1_LEFT)
        return self.store.set("HVAC_DEFROSTER", Seat.ROW_1_LEFT, not cur)

    def power_toggle(self) -> List[Change]:
        return self._toggle_bool("HVAC_POWER_ON")

    # -- expert / simple screen ---------------------------------------------
    def set_expert_mode(self, on: bool) -> List[Change]:
        return self.store.set("HVAC_EXPERT_MODE", 0, on)

    # ── Doors ────────────────────────────────────────────────────────────────
    def door_lock(self, area: int, locked: bool) -> List[Change]:
        return self.store.set("DOOR_LOCK", area, locked)

    def door_move(self, area: int, open_: bool) -> List[Change]:
        return self.store.set("DOOR_MOVE", area, 1 if open_ else 0)

    # ── Lights ───────────────────────────────────────────────────────────────
    def headlights_toggle(self) -> List[Change]:
        return self._toggle_bool("HEADLIGHTS_SWITCH")

    def hazard_toggle(self) -> List[Change]:
        return self._toggle_bool("HAZARD_LIGHTS_SWITCH")

    def fog_toggle(self) -> List[Change]:
        return self._toggle_bool("FOG_LIGHTS_SWITCH")

    def cabin_lights_toggle(self) -> List[Change]:
        return self._toggle_bool("CABIN_LIGHTS_SWITCH")

    # ── Seat comfort ─────────────────────────────────────────────────────────
    def bump_seat_temp(self, up: bool, area: int) -> List[Change]:
        changes: List[Change] = []
        step = 1 if up else -1
        targets = [a for a in [Seat.ROW_1_LEFT, Seat.ROW_1_RIGHT] if area == 0 or (area & a)]
        for a in targets:
            cur = self.store.get("HVAC_SEAT_TEMPERATURE", a)
            new_val = max(MIN_SEAT_TEMP, min(MAX_SEAT_TEMP, cur + step))
            changes += self.store.set("HVAC_SEAT_TEMPERATURE", a, new_val)
        return changes

    def bump_seat_vent(self, up: bool, area: int) -> List[Change]:
        changes: List[Change] = []
        step = 1 if up else -1
        targets = [a for a in [Seat.ROW_1_LEFT, Seat.ROW_1_RIGHT] if area == 0 or (area & a)]
        for a in targets:
            cur = self.store.get("HVAC_SEAT_VENTILATION", a)
            new_val = max(MIN_SEAT_VENT, min(MAX_SEAT_VENT, cur + step))
            changes += self.store.set("HVAC_SEAT_VENTILATION", a, new_val)
        return changes

    # ── Windows ──────────────────────────────────────────────────────────────
    def window_move(self, area: int, position: int) -> List[Change]:
        """Set window position 0 (closed) to 100 (fully open)."""
        pos = max(0, min(100, int(position)))
        return self.store.set("WINDOW_MOVE", area, pos)

    def window_lock(self, area: int, locked: bool) -> List[Change]:
        return self.store.set("WINDOW_LOCK", area, locked)
