"""Authoritative warehouse model.

The SAME object feeds the simulation physics, every AMR brain (through a
legitimate local-knowledge view) and the frontend (via JSON).  The frontend
never invents geometry: it renders exactly what `Warehouse.to_dict()` returns.

Layout is built from a declarative description so that dimensions, racks,
aisles, intersections, choke points, stations and charging bays are all
user-configurable (see scenarios / API PUT /config/warehouse).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .types import Rect


@dataclass
class Zone:
    id: str
    kind: str          # storage | aisle | intersection | choke | station | charge | staging | corridor
    rect: Rect
    label: str = ""
    capacity: int = 99   # how many robots may legitimately be inside at once (1 => single-lane choke)

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "label": self.label or self.id,
                "capacity": self.capacity,
                "rect": [round(self.rect.x0, 3), round(self.rect.y0, 3),
                         round(self.rect.x1, 3), round(self.rect.y1, 3)]}


@dataclass
class Station:
    id: str
    cell: Tuple[int, int]
    kind: str          # pickup | drop | charge | home
    zone_id: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "cell": list(self.cell), "kind": self.kind, "zone_id": self.zone_id}


DEFAULT_WAREHOUSE = {
    "name": "SIH26123 Demo Warehouse",
    "width": 48.0, "height": 30.0, "cell": 1.5,
    "grid_cols": 32, "grid_rows": 20,
    "racks": [
        {"x0": 6.0, "y0": 4.0, "x1": 16.0, "y1": 7.0},
        {"x0": 6.0, "y0": 10.0, "x1": 16.0, "y1": 13.0},
        {"x0": 6.0, "y0": 16.0, "x1": 16.0, "y1": 19.0},
        {"x0": 6.0, "y0": 22.0, "x1": 16.0, "y1": 25.0},
        {"x0": 24.0, "y0": 4.0, "x1": 34.0, "y1": 7.0},
        {"x0": 24.0, "y0": 10.0, "x1": 34.0, "y1": 13.0},
        {"x0": 24.0, "y0": 16.0, "x1": 34.0, "y1": 19.0},
        {"x0": 24.0, "y0": 22.0, "x1": 34.0, "y1": 25.0},
    ],
    "static_obstacles": [],
    "corridors": [
        {"x0": 0.0, "y0": 0.0, "x1": 48.0, "y1": 2.4},
        {"x0": 0.0, "y0": 27.6, "x1": 48.0, "y1": 30.0},
        {"x0": 0.0, "y0": 0.0, "x1": 2.4, "y1": 30.0},
        {"x0": 45.6, "y0": 0.0, "x1": 48.0, "y1": 30.0},
        {"x0": 18.4, "y0": 0.0, "x1": 21.6, "y1": 30.0},
        {"x0": 0.0, "y0": 14.0, "x1": 48.0, "y1": 15.8},
    ],
    "chokes": [
        {"id": "CH-N", "x0": 18.4, "y0": 7.5, "x1": 21.6, "y1": 9.5},
        {"id": "CH-S", "x0": 18.4, "y0": 20.5, "x1": 21.6, "y1": 22.5},
        {"id": "CH-E", "x0": 36.5, "y0": 13.6, "x1": 39.0, "y1": 16.2},
    ],
    "stations": [
        {"id": "P01", "cx": 1.2, "cy": 3.0, "kind": "pickup"},
        {"id": "P02", "cx": 1.2, "cy": 27.0, "kind": "pickup"},
        {"id": "D01", "cx": 46.8, "cy": 3.0, "kind": "drop"},
        {"id": "D02", "cx": 46.8, "cy": 27.0, "kind": "drop"},
        {"id": "C01", "cx": 20.0, "cy": 1.2, "kind": "charge"},
        {"id": "C02", "cx": 24.0, "cy": 1.2, "kind": "charge"},
        {"id": "HOME-A", "cx": 2.5, "cy": 14.9, "kind": "home"},
        {"id": "HOME-B", "cx": 4.5, "cy": 14.9, "kind": "home"},
        {"id": "HOME-C", "cx": 6.5, "cy": 14.9, "kind": "home"},
        {"id": "HOME-D", "cx": 8.5, "cy": 14.9, "kind": "home"},
    ],
    "dead_zones": [],
}


class Warehouse:
    """Occupancy grid + semantic zones. Single source of truth."""

    def __init__(self, spec: Optional[dict] = None):
        s = spec or DEFAULT_WAREHOUSE
        self.spec = s
        self.name: str = s["name"]
        self.width: float = float(s["width"])
        self.height: float = float(s["height"])
        self.cell: float = float(s.get("cell", 1.5))
        self.cols: int = int(math.ceil(self.width / self.cell))
        self.rows: int = int(math.ceil(self.height / self.cell))
        # blocked[c][r] True if static obstacle covers cell centre
        self.blocked: List[List[bool]] = [[False] * self.rows for _ in range(self.cols)]
        self.zones: Dict[str, Zone] = {}
        self.stations: Dict[str, Station] = {}
        self.racks: List[Rect] = []
        self.static_obstacles: List[Rect] = []
        self.dead_zones: List[Rect] = []

        for r in s.get("racks", []):
            self.racks.append(Rect.from_bounds(r["x0"], r["y0"], r["x1"], r["y1"]))
        for r in s.get("static_obstacles", []):
            self.static_obstacles.append(Rect.from_bounds(r["x0"], r["y0"], r["x1"], r["y1"]))
        for r in s.get("dead_zones", []):
            self.dead_zones.append(Rect.from_bounds(r["x0"], r["y0"], r["x1"], r["y1"]))

        for c in s.get("corridors", []):
            z = Zone(f"Z_{c['x0']}_{c['y0']}_{c['x1']}_{c['y1']}", "corridor",
                     Rect.from_bounds(c["x0"], c["y0"], c["x1"], c["y1"]))
            self.zones[z.id] = z
        for ch in s.get("chokes", []):
            z = Zone(ch["id"], "choke", Rect.from_bounds(ch["x0"], ch["y0"], ch["x1"], ch["y1"]),
                     label=ch["id"], capacity=int(ch.get("capacity", 1)))
            self.zones[z.id] = z
        for st in s.get("stations", []):
            zone_kind = {"pickup": "station", "drop": "station", "charge": "charge", "home": "staging"}[st["kind"]]
            rad = 1.6
            zr = Rect(st["cx"], st["cy"], rad, rad)
            z = Zone(f"ZS_{st['id']}", zone_kind, zr, label=st["id"], capacity=int(st.get("capacity", 2)))
            self.zones[z.id] = z
            station = Station(st["id"], self.world_to_cell(st["cx"], st["cy"]), st["kind"], z.id)
            z2 = self.zones[z.id]
            z2.rect = zr
            self.stations[st["id"]] = station

        # rasterise obstacles into grid
        for rect in self.racks + self.static_obstacles:
            c0, r0 = self.world_to_cell(rect.x0 + 0.01, rect.y0 + 0.01)
            c1, r1 = self.world_to_cell(rect.x1 - 0.01, rect.y1 - 0.01)
            for c in range(c0, c1 + 1):
                for r in range(r0, r1 + 1):
                    wx, wy = self.cell_to_world(c, r)
                    if rect.contains(wx, wy):
                        self.blocked[c][r] = True

    # ---------------- grid helpers ----------------
    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        c = max(0, min(self.cols - 1, int(x / self.cell)))
        r = max(0, min(self.rows - 1, int(y / self.cell)))
        return (c, r)

    def cell_to_world(self, c: int, r: int) -> Tuple[float, float]:
        return ((c + 0.5) * self.cell, (r + 0.5) * self.cell)

    def in_bounds(self, c: int, r: int) -> bool:
        return 0 <= c < self.cols and 0 <= r < self.rows

    def cell_blocked(self, c: int, r: int) -> bool:
        if not self.in_bounds(c, r):
            return True
        return self.blocked[c][r]

    def point_blocked(self, x: float, y: float) -> bool:
        if x < 0 or y < 0 or x > self.width or y > self.height:
            return True
        for rect in self.racks + self.static_obstacles:
            if rect.contains(x, y):
                return True
        return False

    def free_disk(self, cx: float, cy: float, radius: float) -> bool:
        """True if a disk of `radius` centred at world point (cx,cy) fits in free space."""
        if cx < radius or cy < radius or cx > self.width - radius or cy > self.height - radius:
            return False
        for rect in self.racks + self.static_obstacles:
            nx = max(rect.x0, min(cx, rect.x1))
            ny = max(rect.y0, min(cy, rect.y1))
            if (nx - cx) ** 2 + (ny - cy) ** 2 < radius * radius:
                return False
        return True

    # ---------------- semantic queries ----------------
    def zone_at(self, x: float, y: float) -> Optional[Zone]:
        # most specific (smallest area) containing zone wins; chokes beat corridors
        best = None
        for z in self.zones.values():
            if z.rect.contains(x, y):
                if best is None or (z.rect.hx * z.rect.hy < best.rect.hx * best.rect.hy):
                    best = z
        return best

    def choke_zones(self) -> List[Zone]:
        return [z for z in self.zones.values() if z.kind == "choke"]

    def station_pos(self, sid: str) -> Tuple[float, float]:
        st = self.stations[sid]
        return self.cell_to_world(*st.cell)

    def nearest_free_cell(self, x: float, y: float, radius: float) -> Tuple[int, int]:
        cc, cr = self.world_to_cell(x, y)
        if self.free_disk(x, y, radius):
            return (cc, cr)
        for d in range(1, max(self.cols, self.rows)):
            for dc in range(-d, d + 1):
                for dr in (-d, d):
                    c, r = cc + dc, cr + dr
                    if self.in_bounds(c, r) and not self.blocked[c][r]:
                        wx, wy = self.cell_to_world(c, r)
                        if self.free_disk(wx, wy, radius):
                            return (c, r)
                for dr in range(-d + 1, d):
                    for dc in (-d, d):
                        c, r = cc + dc, cr + dr
                        if self.in_bounds(c, r) and not self.blocked[c][r]:
                            wx, wy = self.cell_to_world(c, r)
                            if self.free_disk(wx, wy, radius):
                                return (c, r)
        return (cc, cr)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "width": self.width, "height": self.height,
            "cell": self.cell, "cols": self.cols, "rows": self.rows,
            "zones": [z.to_dict() for z in self.zones.values()],
            "racks": [[round(r.x0, 3), round(r.y0, 3), round(r.x1, 3), round(r.y1, 3)] for r in self.racks],
            "static_obstacles": [[round(r.x0, 3), round(r.y0, 3), round(r.x1, 3), round(r.y1, 3)]
                                 for r in self.static_obstacles],
            "dead_zones": [[round(r.x0, 3), round(r.y0, 3), round(r.x1, 3), round(r.y1, 3)]
                           for r in self.dead_zones],
            "stations": [st.to_dict() for st in self.stations.values()],
            "chokes": [z.to_dict() for z in self.choke_zones()],
        }

    def local_view(self, x: float, y: float, radius: float) -> dict:
        """What an AMR can legitimately know about the static map around a point
        (from its onboard stored map + live perception)."""
        out = {"zones": [], "obstacles": []}
        for z in self.zones.values():
            zx, zy = z.rect.cx, z.rect.cy
            if math.hypot(zx - x, zy - y) <= radius + max(z.rect.hx, z.rect.hy):
                out["zones"].append(z.to_dict())
        for r in self.racks:
            if math.hypot(r.cx - x, r.cy - y) <= radius + max(r.hx, r.hy):
                out["obstacles"].append([r.x0, r.y0, r.x1, r.y1])
        return out
