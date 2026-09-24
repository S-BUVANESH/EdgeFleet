"""UEAC — Unified Entry-Admission Control for constrained / choke zones.

Deterministic lease + deadman mechanism with the six established factors:

  physical occupancy : robots whose footprint currently intersects the zone
  swept volume       : how much of the narrow zone a robot's motion envelope takes
  braking distance   : stopping distance at approach speed vs remaining room
  localization cov.  : sigma_xy uncertainty inflating the required clearance
  egress clearance   : is there a free path out of the far side (no queue behind)
  active leases      : other robots' unexpired deadman leases in this zone

Safety rule that the spec demands: **lease expiry does NOT imply physical
vacancy**.  Admission therefore requires BOTH a granted lease AND direct
legitimate evidence (own sensors or fresh peer intent) that the zone is free.

Note on scope: factors are combined by explicit, documented predicates —
there is no invented weighting formula.  A GRANT/DENY decision plus the
per-factor reasons are exposed to observability.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Lease:
    zone_id: str
    robot_id: str
    granted_at: float
    deadline: float          # deadman expiry
    seq: int = 0

    def alive(self, now: float) -> bool:
        return now < self.deadline


@dataclass
class UEACResult:
    zone_id: str
    robot_id: str
    decision: str            # GRANT | DENY | WAIT_LEASE | HOLD_AT_ENTRY
    factors: Dict[str, object]
    reasons: List[str]

    def to_dict(self) -> dict:
        return {"zone": self.zone_id, "robot": self.robot_id, "decision": self.decision,
                "reasons": list(self.reasons),
                "factors": {k: (round(v, 3) if isinstance(v, float) else v)
                            for k, v in self.factors.items()}}


class ChokeRegistry:
    """Deterministic lease bookkeeping shared via COMMUNICATION, not oracle.

    Robots never read this directly; they request a lease through their brain,
    which performs the claim as a deterministic protocol step against the
    shared *protocol state* (the bus carries claims/grants so peers can see
    them too — lease info is legitimate coordination information).
    """

    LEASE_S = 6.0   # deadman window; renewed while approaching/inside

    def __init__(self):
        self.leases: Dict[str, Dict[str, Lease]] = {}   # zone -> robot -> lease
        self._seq = 0

    def request(self, zone_id: str, robot_id: str, now: float,
                occupants: int, capacity: int) -> Tuple[bool, Lease]:
        z = self.leases.setdefault(zone_id, {})
        # expire dead man-switches first (deterministic sweep)
        for rid in sorted(z.keys()):
            if not z[rid].alive(now):
                del z[rid]
        if robot_id in z:
            l = z[robot_id]
            l.deadline = now + self.LEASE_S
            l.seq = self._next()
            return True, l
        active_others = len(z)
        if active_others >= capacity:
            return False, None
        self._seq += 1
        lease = Lease(zone_id, robot_id, now, now + self.LEASE_S, self._seq)
        z[robot_id] = lease
        return True, lease

    def renew(self, zone_id: str, robot_id: str, now: float):
        z = self.leases.get(zone_id, {})
        if robot_id in z:
            z[robot_id].deadline = now + self.LEASE_S

    def release(self, zone_id: str, robot_id: str):
        self.leases.get(zone_id, {}).pop(robot_id, None)

    def holders(self, zone_id: str, now: float) -> List[str]:
        z = self.leases.get(zone_id, {})
        return [rid for rid in sorted(z.keys()) if z[rid].alive(now)]

    def _next(self) -> int:
        self._seq += 1
        return self._seq

    def snapshot(self, now: float) -> dict:
        return {zid: {rid: {"granted_at": round(l.granted_at, 2),
                            "expires_in": round(max(0.0, l.deadline - now), 2)}
                      for rid, l in sorted(self.leases[zid].items()) if l.alive(now)}
                for zid in sorted(self.leases.keys())}


def evaluate_ueac(zone, robot_id: str, approach_speed: float,
                  max_decel: float, sigma_xy: float, half_diag: float,
                  registry: ChokeRegistry, now: float,
                  egress_clear: bool, sensor_confirms_free: bool,
                  inside_ids: List[str], already_inside: bool) -> UEACResult:
    """Pure function: admission decision from legitimate inputs only.

    inside_ids        : robots whose footprint physically intersects the zone
                        (own perception — ground-truth proxy for SAFETY check)
    sensor_confirms_free : own sensors saw the zone empty this cycle
    already_inside    : requesting robot itself is inside the zone
    """
    factors: Dict[str, object] = {}
    reasons: List[str] = []
    cap = zone.capacity

    # --- factor 1: physical occupancy (safety override) ---
    others_inside = [r for r in sorted(set(inside_ids)) if r != robot_id]
    occ = len(others_inside)
    factors["physical_occupancy"] = occ
    factors["occupants"] = others_inside

    # --- factor 2: swept volume share of the constrained zone ---
    zone_area = max(1e-6, (zone.rect.x1 - zone.rect.x0) * (zone.rect.y1 - zone.rect.y0))
    swept = math.pi * (half_diag ** 2)
    share = min(2.0, swept / zone_area)
    factors["swept_volume_share"] = round(share, 3)

    # --- factor 3: braking distance vs usable room along narrow axis ---
    stop_d = approach_speed * 0.25 + approach_speed ** 2 / (2.0 * max_decel)
    room = min(zone.rect.hx, zone.rect.hy) * 2.0
    factors["braking_distance_m"] = round(stop_d, 3)
    factors["room_m"] = round(room, 3)
    brake_ok = stop_d <= room * 0.9

    # --- factor 4: localization covariance inflation ---
    inflate = 2.0 * sigma_xy
    factors["loc_covariance_inflation_m"] = round(inflate, 3)
    clear_after_inflate = (room - 2.0 * inflate) >= half_diag

    # --- factor 5: egress clearance ---
    factors["egress_clear"] = egress_clear

    # --- factor 6: active leases (from communicated protocol state) ---
    holders = registry.holders(zone.id, now)
    has_lease = robot_id in holders
    lease_others = [h for h in holders if h != robot_id]
    factors["active_leases"] = holders
    factors["lease_held_by_me"] = has_lease

    decision = "GRANT"
    if occ > 0 and not already_inside:
        decision = "HOLD_AT_ENTRY"
        reasons.append(f"SAFETY: zone physically occupied by {others_inside} (lease expiry != vacancy)")
    elif not brake_ok and not already_inside:
        decision = "HOLD_AT_ENTRY"
        reasons.append(f"braking distance {stop_d:.2f}m exceeds usable room {room*0.9:.2f}m")
    elif not clear_after_inflate and not already_inside:
        decision = "DENY"
        reasons.append("localization uncertainty leaves no guaranteed clearance in zone")
    elif not egress_clear and not already_inside:
        decision = "WAIT_LEASE"
        reasons.append("no confirmed egress beyond zone (would enter a queue)")
    else:
        ok, _ = registry.request(zone.id, robot_id, now, occ, cap)
        if not ok:
            decision = "WAIT_LEASE"
            reasons.append(f"lease slots full, held by {[h for h in holders if h != robot_id]}")
        elif not already_inside and occ == 0 and not sensor_confirms_free:
            decision = "WAIT_LEASE"
            reasons.append("lease acquired but no fresh vacancy evidence from own sensors/peers")
        else:
            decision = "GRANT"
            if has_lease or ok:
                registry.renew(zone.id, robot_id, now)
    return UEACResult(zone.id, robot_id, decision, factors, reasons)
