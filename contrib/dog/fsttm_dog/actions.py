"""Typed action/query layer — the stable seam between the intent language and
the robot (spec.md §§9,16-18).

The LLM never sees these types; it emits constrained intent JSON, the
dispatcher parses that into the dataclasses here, and the backends (nav2
action clients, DINOv3 semantic map, Go2 action interface) implement the
Protocols. This package ships logging stubs only — no ROS2 imports anywhere;
the real backends plug in on the robot.

Design rules from the spec:
  - navigation is expressed as a GOAL, never velocity sequences (§6)
  - the semantic layer resolves language targets to 3D poses; the planner
    receives deterministic machine commands only (§17-18)
  - VELOCITY exists as a type for the planner/controller — it is NOT in the
    LLM grammar (§3 safety note)
  - STOP/CANCEL bypass semantic reasoning (§14)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence

# ── semantic target (§5) ──────────────────────────────────────────────────────

TARGET_TYPES = ("OBJECT", "PERSON", "POSITION", "REGION", "ROOM", "LANDMARK")
RELATION_TYPES = ("NEAR", "NEXT_TO", "IN_FRONT_OF", "BEHIND", "ON", "UNDER",
                  "LEFT_OF", "RIGHT_OF", "CLOSEST_TO")


@dataclass
class SemanticTarget:
    """Open-vocabulary target: `description` drives grounding (detector
    classes, DINOv3 features, future open-vocab embeddings); `cls` is an
    optional closed-class hint."""
    type: str = "OBJECT"                  # TARGET_TYPES
    description: str = ""
    cls: Optional[str] = None             # spec `class` (reserved word)
    attributes: dict = field(default_factory=dict)   # color / size / material

    @classmethod
    def from_json(cls_, d: dict) -> "SemanticTarget":
        d = d or {}
        return cls_(type=d.get("type", "OBJECT"),
                    description=d.get("description", ""),
                    cls=d.get("class"),
                    attributes={k: v for k, v in
                                (d.get("attributes") or {}).items()
                                if v is not None})


@dataclass
class Relation:
    """Spatial constraint resolved by the semantic-map layer (§7)."""
    type: str                              # RELATION_TYPES
    reference: SemanticTarget

    @classmethod
    def from_json(cls, d: dict) -> "Relation":
        return cls(type=d.get("relation") or d.get("type", "NEAR"),
                   reference=SemanticTarget.from_json(d.get("reference")))


# ── commands out of the intent layer ─────────────────────────────────────────

LOCAL_ACTIONS = ("STAND_UP", "SIT_DOWN", "LIE_DOWN", "STRETCH", "SHAKE",
                 "JUMP", "POUNCE", "TURN", "MOVE")


@dataclass
class LocalAction:
    """Immediate Go2 action, no semantic navigation (§3). Maps to the
    wireless-controller/action interface (lx/ly/rx + named actions)."""
    action: str                            # LOCAL_ACTIONS
    direction: Optional[str] = None        # LEFT/RIGHT/AROUND (TURN),
                                           # FORWARD/BACKWARD (MOVE)
    angle_deg: Optional[float] = None
    duration: Optional[float] = None


@dataclass
class Velocity:
    """Raw velocity — planner/controller territory. NEVER emitted by the LLM
    (excluded from the grammar); exists so the planner has a typed channel to
    the same executor."""
    linear_x: float = 0.0
    linear_y: float = 0.0
    angular_z: float = 0.0
    duration: Optional[float] = None


@dataclass
class SemanticQuery:
    """query(target, constraints) → candidates[] (§4, §9). Does not
    necessarily move the robot."""
    target: SemanticTarget
    constraints: List[Relation] = field(default_factory=list)


@dataclass
class Candidate:
    """One resolved instance from the semantic map / object memory (§9-11)."""
    instance_id: int
    score: float
    position: Sequence[float]              # [x, y, z] map frame
    last_seen: float = 0.0
    confidence: float = 0.0


@dataclass
class NavigateGoal:
    """Language-level navigation goal (§6-8): resolved by the semantic layer
    into a PoseGoal, then executed by the classical stack."""
    goal: SemanticTarget
    constraints: List[Relation] = field(default_factory=list)


@dataclass
class FindGoal:
    """FIND ≠ NAVIGATE (§12): known instance → go there; unknown → the
    planner produces an exploration goal (frontier-based fallback, §15)."""
    target: SemanticTarget
    constraints: List[Relation] = field(default_factory=list)


@dataclass
class FollowTarget:
    """Dynamic target (§13): tracker → target pose → local controller; never
    converted into a permanent map goal."""
    target: SemanticTarget


@dataclass
class PoseGoal:
    """What the semantic/planning layer ultimately hands to Nav2 (§17)."""
    position: Sequence[float]              # [x, y, z]
    yaw: float = 0.0
    source: str = "semantic_object"
    instance_id: Optional[int] = None


# ── backend protocols (implemented on the robot; stubs below) ────────────────

class ActionBackend(Protocol):
    """Go2 local-action interface (bottom of the stack, §19)."""
    def execute(self, action: LocalAction) -> None: ...
    def velocity(self, vel: Velocity) -> None: ...
    def stop(self) -> None: ...            # §14: immediate, bypasses planning
    def cancel(self) -> None: ...


class SemanticMemory(Protocol):
    """Semantic map + object instance memory (§9-11)."""
    def query(self, q: SemanticQuery) -> List[Candidate]: ...


class NavigationBackend(Protocol):
    """Classical navigation stack (Nav2) behind the semantic layer (§17)."""
    def navigate(self, goal: PoseGoal) -> None: ...
    def explore(self, q: SemanticQuery) -> None: ...   # §15 planner operation
    def follow(self, target: FollowTarget) -> None: ...
    def cancel(self) -> None: ...


# ── logging stubs (headless-testable; replaced on the robot) ─────────────────

class LoggingActionBackend:
    def __init__(self, emit=print):
        self._emit = emit

    def execute(self, action: LocalAction) -> None:
        self._emit(f"[go2] LOCAL_ACTION {action}")

    def velocity(self, vel: Velocity) -> None:
        self._emit(f"[go2] VELOCITY {vel}")

    def stop(self) -> None:
        self._emit("[go2] STOP (immediate)")

    def cancel(self) -> None:
        self._emit("[go2] CANCEL")


class LoggingSemanticMemory:
    def __init__(self, emit=print):
        self._emit = emit

    def query(self, q: SemanticQuery) -> List[Candidate]:
        self._emit(f"[semantic] QUERY {q}")
        return []                          # nothing enrolled in the stub map


class LoggingNavigationBackend:
    def __init__(self, emit=print):
        self._emit = emit

    def navigate(self, goal: PoseGoal) -> None:
        self._emit(f"[nav] NAVIGATE {goal}")

    def explore(self, q: SemanticQuery) -> None:
        self._emit(f"[nav] EXPLORE for {q.target.description!r}")

    def follow(self, target: FollowTarget) -> None:
        self._emit(f"[nav] FOLLOW {target.target.description!r}")

    def cancel(self) -> None:
        self._emit("[nav] CANCEL")
