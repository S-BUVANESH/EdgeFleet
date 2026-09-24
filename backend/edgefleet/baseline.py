"""STOP-AND-WAIT baseline brain.

Same physical execution surface as the EdgeFleet brain, but coordination is
the classic reactive rule: if any peer is closer than a fixed geometric
threshold in front of me -> stop and wait until it clears.  No intent
sharing, no negotiation, no UEAC leases, no predictive conflicts.  Used for
paired experiments against EdgeFleet under identical tasks/seed.
"""
from __future__ import annotations

import math
from typing import Optional

from .brain import Brain
from .types import Pose, clamp, angle_diff


class BaselineBrain(Brain):
    """Inherits planning/task/battery handling; overrides coordination."""

    STOP_DIST = 2.0     # m — simplistic geometric rule (the thing we compare against)

    def _predict_conflicts(self):
        self.conflict_last = None
        return None

    def _negotiate(self, conflict):
        return

    def _choke_admission(self) -> bool:
        # baseline has no admission control at all — just drive in
        self.ueac_last = None
        return False

    def _motion_command(self, conflict, choke_hold: bool):
        if self.phase == "charging":
            self.v_cmd = self.w_cmd = 0.0
            return
        # stop & wait: nearest forward peer within threshold
        blocked_by = None
        best = None
        for s in getattr(self, "_last_sensed", []):
            dx, dy = s["x"] - self.est.x, s["y"] - self.est.y
            d = math.hypot(dx, dy)
            ahead = dx * math.cos(self.est.th) + dy * math.sin(self.est.th)
            if d < self.STOP_DIST and ahead > -0.3:
                if best is None or d < best:
                    best, blocked_by = d, s["id"]
        super()._motion_command(None, False)
        if blocked_by:
            self.v_cmd = 0.0
            self.last_decision = "STOP_AND_WAIT"
            self.decision_detail = f"peer {blocked_by} within {self.STOP_DIST}m — waiting"
            self.waiting_for = blocked_by
            self.wait_reason = "baseline stop-and-wait"
            if self._wf_prev != blocked_by:
                self.wait_started = self.now
                self._wf_prev = blocked_by
        else:
            if self.v_cmd > 0.05:
                self.waiting_for = None
                self._wf_prev = None

    def _deadlock_watch(self):
        # baseline cannot reason about cycles; starvation simply persists
        return

    def _publish_intent(self):
        # baseline broadcasts only its pose (beacon), no trajectory intent
        if self.now - self.t_last_intent < 1.0:
            return
        self.t_last_intent = self.now
        self.bus.send("intent", self.id,
                      {"x": round(self.est.x, 2), "y": round(self.est.y, 2),
                       "th": round(self.est.th, 2), "v": round(self.est_v, 2),
                       "traj_id": 0, "traj": [], "lease": None,
                       "dest": "-", "carrying": self.carrying,
                       "dims": [self.spec.length, self.spec.width],
                       "prio": [9999, 9, 9999, 9999, self.id]},
                      self.now, ttl=4.0, scope_xy=(self.est.x, self.est.y))
