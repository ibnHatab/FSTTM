"""
DogDispatcher — routes parsed intents to the typed robot seams (spec.md §19).

    intent JSON ──┬── LOCAL_ACTION ────────► ActionBackend (Go2 actions)
                  ├── STOP / CANCEL ───────► ActionBackend + NavigationBackend
                  │                          (immediate; bypasses planning §14)
                  ├── QUERY ───────────────► SemanticMemory.query
                  ├── NAVIGATE ────────────► SemanticMemory → NavigationBackend
                  ├── FIND ────────────────► known instance? go : explore (§12)
                  └── FOLLOW ──────────────► NavigationBackend.follow (§13)

This scaffold wires the logging stubs (config `domains.dog.backend: logging`)
so the whole voice → intent → typed-action path runs headless on a dev box.
On the robot, replace the stubs with nav2 action clients + the DINOv3
semantic map behind the same Protocols.
"""
from __future__ import annotations

import logging
from typing import Optional

from fsttm.domain import DispatchOutcome, DomainContext, DomainDispatcher

from fsttm_dog.actions import (
    Candidate, FindGoal, FollowTarget, LocalAction, LoggingActionBackend,
    LoggingNavigationBackend, LoggingSemanticMemory, NavigateGoal, PoseGoal,
    Relation, SemanticQuery, SemanticTarget,
)

_log = logging.getLogger("fsttm_dog.dispatcher")


def _parse_target(ij: dict, key: str) -> SemanticTarget:
    return SemanticTarget.from_json(ij.get(key) or ij.get(
        "goal" if key == "target" else "target") or {})


def _parse_constraints(ij: dict) -> list:
    return [Relation.from_json(c) for c in (ij.get("constraints") or [])
            if isinstance(c, dict)]


class DogDispatcher(DomainDispatcher):
    def __init__(self, ctx: DomainContext):
        self.ctx = ctx
        cfg = ctx.config or {}
        backend = cfg.get("backend", "logging")
        if backend != "logging":
            ctx.emit(f"[dog] backend {backend!r} not available in the "
                     f"scaffold — using logging stubs", "warn")
        emit = lambda text: ctx.emit(text, "info")   # noqa: E731
        self.actions = LoggingActionBackend(emit)
        self.memory = LoggingSemanticMemory(emit)
        self.nav = LoggingNavigationBackend(emit)
        self._last_query: Optional[list] = None

    def start(self) -> None:
        self.ctx.tui_status("go2:logging", True)
        self.ctx.emit("[dog] dispatcher ready (logging backends)", "good")

    # ── side effects ──────────────────────────────────────────────────────
    def handle(self, intent: dict, commands: list) -> DispatchOutcome:
        for c in commands:
            if not (isinstance(c, dict) and c.get("cmd") == "dog"):
                continue
            try:
                self._execute(c.get("command") or {})
            except Exception as exc:
                _log.exception("dog command failed")
                self.ctx.emit(f"[dog] command failed: {exc}", "warn")
        return DispatchOutcome.PASS

    def _execute(self, ij: dict) -> None:
        name = ij.get("intent")
        if name == "STOP":              # §14: immediate, no semantic reasoning
            self.actions.stop()
            self.nav.cancel()
        elif name == "CANCEL":
            self.actions.cancel()
            self.nav.cancel()
        elif name == "LOCAL_ACTION":
            self.actions.execute(LocalAction(
                action=ij.get("action") or "STAND_UP",
                direction=ij.get("direction"),
                angle_deg=ij.get("angle_deg"),
                duration=ij.get("duration")))
        elif name == "QUERY":
            q = SemanticQuery(target=_parse_target(ij, "target"),
                              constraints=_parse_constraints(ij))
            self._last_query = self.memory.query(q)
        elif name == "NAVIGATE":
            goal = NavigateGoal(goal=_parse_target(ij, "goal"),
                                constraints=_parse_constraints(ij))
            self._navigate(goal)
        elif name == "FIND":
            goal = FindGoal(target=_parse_target(ij, "target"),
                            constraints=_parse_constraints(ij))
            self._find(goal)
        elif name == "FOLLOW":
            self.nav.follow(FollowTarget(target=_parse_target(ij, "target")))

    def _resolve(self, target: SemanticTarget, constraints) -> Optional[Candidate]:
        """Language goal → semantic query → best instance (§6). The stub map
        returns nothing; on the robot this is the object-instance memory."""
        hits = self.memory.query(SemanticQuery(target=target,
                                               constraints=list(constraints)))
        return hits[0] if hits else None

    def _navigate(self, goal: NavigateGoal) -> None:
        best = self._resolve(goal.goal, goal.constraints)
        if best is None:
            self.ctx.emit(f"[dog] NAVIGATE: {goal.goal.description!r} not in "
                          f"the semantic map — planner would reject or ask",
                          "warn")
            return
        self.nav.navigate(PoseGoal(position=best.position,
                                   source="semantic_object",
                                   instance_id=best.instance_id))

    def _find(self, goal: FindGoal) -> None:
        """known instance? → go there : explore (§12, §15)."""
        best = self._resolve(goal.target, goal.constraints)
        if best is not None:
            self.nav.navigate(PoseGoal(position=best.position,
                                       source="semantic_object",
                                       instance_id=best.instance_id))
        else:
            self.nav.explore(SemanticQuery(target=goal.target,
                                           constraints=list(goal.constraints)))

    # ── narration ─────────────────────────────────────────────────────────
    def local_answer(self, intent: dict) -> Optional[str]:
        """QUERY answers deterministically from the semantic map — never an
        LLM-invented observation."""
        ij = intent or {}
        if ij.get("intent") != "QUERY":
            return None
        desc = ((ij.get("target") or {}).get("description")
                or "that").strip() or "that"
        hits = self._last_query or []
        if not hits:
            return f"I don't see {desc} in my map yet."
        n = len(hits)
        return (f"I know {n} {'place' if n == 1 else 'places'} for {desc}; "
                f"the best match is instance {hits[0].instance_id}.")

    def status_line(self):
        return ("go2:logging", True)
