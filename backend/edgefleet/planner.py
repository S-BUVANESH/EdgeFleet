"""Deterministic path planning on the warehouse grid.

A* with 8-connectivity, footprint-aware clearance and a zone-penalty map so
that brains can (legitimately, from their own stored map) prefer or avoid
particular zones — e.g. route around a congested choke after negotiation.

Everything is deterministic: identical inputs produce identical paths, tie
breaks are by fixed neighbour order then cell index.
"""
from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Set, Tuple

from .warehouse import Warehouse

Cell = Tuple[int, int]

# fixed neighbour order for determinism (E, NE, N, NW, W, SW, S, SE)
NEIGHBOURS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def _octile(a: Cell, b: Cell) -> float:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)


def astar(wh: Warehouse, start: Cell, goal: Cell, radius: float,
          zone_penalty: Optional[Dict[str, float]] = None,
          blocked_cells: Optional[Set[Cell]] = None,
          max_expansions: int = 60000) -> Optional[List[Cell]]:
    """Return list of cells from start..goal inclusive, or None if unreachable.

    `radius` is the robot's circumscribed footprint radius; a cell is passable
    only if its centre admits a free disk of that radius and no obstacle cell
    centre lies within 2*radius along a diagonal step (corner cutting check).
    """
    blocked_cells = blocked_cells or set()
    if wh.cell_blocked(*start) or wh.cell_blocked(*goal):
        return None

    def passable(c: Cell) -> bool:
        if not wh.in_bounds(*c) or wh.blocked[c[0]][c[1]] or c in blocked_cells:
            return False
        wx, wy = wh.cell_to_world(*c)
        return wh.free_disk(wx, wy, radius)

    def corner_ok(a: Cell, b: Cell) -> bool:
        # forbid cutting corners between two blocked cells on diagonal moves
        if a[0] != b[0] and a[1] != b[1]:
            if wh.cell_blocked(b[0], a[1]) or wh.cell_blocked(a[0], b[1]):
                return False
        return True

    open_heap: List[Tuple[float, int, int, Cell]] = []
    counter = 0
    g: Dict[Cell, float] = {start: 0.0}
    came: Dict[Cell, Cell] = {}
    closed: Set[Cell] = set()
    heapq.heappush(open_heap, (_octile(start, goal), counter, 0, start))
    expansions = 0
    while open_heap:
        f, _, _, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)
        expansions += 1
        if expansions > max_expansions:
            return None
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return path
        cx, cy = wh.cell_to_world(*cur)
        for dc, dr in NEIGHBOURS:
            nxt = (cur[0] + dc, cur[1] + dr)
            if nxt in closed or not passable(nxt) or not corner_ok(cur, nxt):
                continue
            nx, ny = wh.cell_to_world(*nxt)
            step = math.hypot(nx - cx, ny - cy)
            pen = 0.0
            if zone_penalty:
                z = wh.zone_at(nx, ny)
                if z is not None:
                    pen = zone_penalty.get(z.id, 0.0)
            ng = g[cur] + step + pen
            if ng < g.get(nxt, float("inf")) - 1e-9:
                g[nxt] = ng
                came[nxt] = cur
                counter += 1
                heapq.heappush(open_heap, (ng + _octile(nxt, goal), counter, 0, nxt))
    return None


def smooth_path(wh: Warehouse, pts: List[Tuple[float, float]], radius: float) -> List[Tuple[float, float]]:
    """Simple line-of-sight smoothing over world points."""
    if len(pts) <= 2:
        return pts
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1:
            if _los_clear(wh, pts[i], pts[j], radius):
                break
            j -= 1
        out.append(pts[j])
        i = j
    out.append(pts[-1])
    return out


def _los_clear(wh: Warehouse, a: Tuple[float, float], b: Tuple[float, float], radius: float,
               steps: int = 40) -> bool:
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = max(2, min(steps, int(math.hypot(dx, dy) / (wh.cell * 0.5))))
    for k in range(n + 1):
        t = k / n
        if not wh.free_disk(a[0] + dx * t, a[1] + dy * t, radius):
            return False
    return True


def path_length(pts: List[Tuple[float, float]]) -> float:
    return sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
               for i in range(len(pts) - 1))
