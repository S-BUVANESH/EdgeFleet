"""Core shared types for EdgeFleet.

All positions are in metres on a 2D plane (x forward, y right). Heading is
radians. Time is deterministic simulation seconds (fixed timestep multiples).
Energy is Watt-hours.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def angle_diff(a: float, b: float) -> float:
    """Smallest signed difference between angles a and b."""
    d = math.fmod(a - b + math.pi, 2 * math.pi)
    if d < 0:
        d += 2 * math.pi
    return d - math.pi


@dataclass(frozen=True)
class Pose:
    x: float = 0.0
    y: float = 0.0
    th: float = 0.0

    def shifted(self, dx: float, dy: float) -> "Pose":
        c, s = math.cos(self.th), math.sin(self.th)
        return Pose(self.x + dx * c - dy * s, self.y + dx * s + dy * c, self.th)

    def dist(self, other: "Pose") -> float:
        return math.hypot(other.x - self.x, other.y - self.y)


@dataclass(frozen=True)
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def add(self, o: "Vec2") -> "Vec2":
        return Vec2(self.x + o.x, self.y + o.y)

    def scale(self, k: float) -> "Vec2":
        return Vec2(self.x * k, self.y * k)

    def norm(self) -> float:
        return math.hypot(self.x, self.y)


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle: centre + half extents."""
    cx: float
    cy: float
    hx: float
    hy: float

    @staticmethod
    def from_bounds(x0: float, y0: float, x1: float, y1: float) -> "Rect":
        return Rect((x0 + x1) / 2.0, (y0 + y1) / 2.0, abs(x1 - x0) / 2.0, abs(y1 - y0) / 2.0)

    @property
    def x0(self): return self.cx - self.hx
    @property
    def y0(self): return self.cy - self.hy
    @property
    def x1(self): return self.cx + self.hx
    @property
    def y1(self): return self.cy + self.hy

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (self.x0 - margin <= x <= self.x1 + margin and
                self.y0 - margin <= y <= self.y1 + margin)

    def inflated(self, m: float) -> "Rect":
        return Rect(self.cx, self.cy, self.hx + m, self.hy + m)


# ---------------------------------------------------------------------------
# Robot configuration (heterogeneous fleet)
# ---------------------------------------------------------------------------

@dataclass
class RobotSpec:
    id: str
    length: float = 1.2       # m, along heading
    width: float = 0.8        # m, across heading
    max_speed: float = 1.5    # m/s
    cruise_speed: float = 1.2 # m/s preferred speed
    accel: float = 0.8        # m/s^2
    decel: float = 1.0        # m/s^2 comfortable braking
    max_decel: float = 1.6    # m/s^2 hard braking capability
    max_omega: float = 1.5    # rad/s
    payload_kg: float = 300.0
    capacity_wh: float = 800.0
    reserve_wh: float = 120.0     # must retain after task completion
    dock_reserve_wh: float = 40.0 # extra to reach charging bay
    power_base_w: float = 60.0    # electronics + rolling resistance baseline
    power_kws_per_m2s2: float = 40.0  # ~ drag/mass term per (m/s)^2 * area proxy
    sensor_range: float = 18.0    # m peer detection range
    sensor_fov: float = math.radians(360.0)
    loc_sigma_xy: float = 0.03    # localisation std-dev (m)
    color: str = "#4f7cac"

    @property
    def diag(self) -> float:
        return math.hypot(self.length, self.width)

    def footprint_radius(self) -> float:
        return self.diag / 2.0

    def braking_distance(self, v: float, reaction: float = 0.25) -> float:
        """Distance travelled from current speed v until standstill incl. reaction."""
        return max(0.0, v) * reaction + (max(0.0, v) ** 2) / (2.0 * self.max_decel)

    def swept_radius(self, ahead: float) -> float:
        """Radius of the swept volume when travelling `ahead` metres forward."""
        return math.hypot(self.length / 2.0 + max(0.0, ahead), self.width / 2.0)

    def energy_rate_w(self, speed: float, loaded: bool = False) -> float:
        load_factor = 1.25 if loaded else 1.0
        return self.power_base_w * load_factor + \
            self.power_kws_per_m2s2 * load_factor * speed * speed

    def to_dict(self) -> dict:
        return {
            "id": self.id, "length": self.length, "width": self.width,
            "max_speed": self.max_speed, "cruise_speed": self.cruise_speed,
            "accel": self.accel, "decel": self.decel, "max_decel": self.max_decel,
            "max_omega": self.max_omega, "payload_kg": self.payload_kg,
            "capacity_wh": self.capacity_wh, "reserve_wh": self.reserve_wh,
            "dock_reserve_wh": self.dock_reserve_wh,
            "sensor_range": self.sensor_range, "loc_sigma_xy": self.loc_sigma_xy,
            "color": self.color,
        }


# ---------------------------------------------------------------------------
# Task model
# ---------------------------------------------------------------------------

TASK_STATES = ("created", "advertised", "evaluated", "assigned", "executing",
               "transit_pickup", "carry", "completed", "released", "rebid",
               "reassigned", "failed")


@dataclass
class TaskDef:
    id: str
    pickup_zone: str = ""
    drop_zone: str = ""
    pickup_cell: Optional[Tuple[int, int]] = None
    drop_cell: Optional[Tuple[int, int]] = None
    payload_kg: float = 50.0
    priority: int = 5          # 1 highest .. 9 lowest
    deadline_s: float = 300.0  # seconds after creation
    type: str = "transport"    # transport | charge
    created_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id, "pickup_zone": self.pickup_zone, "drop_zone": self.drop_zone,
            "pickup_cell": list(self.pickup_cell) if self.pickup_cell else None,
            "drop_cell": list(self.drop_cell) if self.drop_cell else None,
            "payload_kg": self.payload_kg, "priority": self.priority,
            "deadline_s": self.deadline_s, "type": self.type, "created_at": self.created_at,
        }


@dataclass
class TaskState:
    definition: TaskDef
    state: str = "created"
    epoch: int = 0                 # incremented on every release/rebid
    version: int = 0               # incremented on every mutation
    owner: Optional[str] = None
    assigned_at: float = 0.0
    completed_at: Optional[float] = None
    released_reason: Optional[str] = None
    bids: list = field(default_factory=list)   # recent bid records (for UI)
    award_note: Optional[str] = None

    def to_dict(self) -> dict:
        d = self.definition.to_dict()
        d.update({
            "task_state": self.state, "epoch": self.epoch, "version": self.version,
            "owner": self.owner, "assigned_at": self.assigned_at,
            "completed_at": self.completed_at, "released_reason": self.released_reason,
            "award_note": self.award_note,
            "bids": [dict(b) for b in self.bids[-8:]],
        })
        return d
