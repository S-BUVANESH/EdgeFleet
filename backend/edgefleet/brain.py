"""The AMR Brain — independent decision-making entity.

CRITICAL RULE enforced by construction: the brain's constructor receives ONLY

    * its own RobotSpec
    * a static map snapshot (Warehouse.to_dict() — legitimate onboard map)
    * a ChokeRegistry handle (deterministic SHARED PROTOCOL STATE for leases;
      claims are broadcast on the bus so peers legitimately observe them)
    * the CommBus + its node id
    * a `physics` callback exposing ONLY this robot's own actuation surface
      (pose, speed, energy, set_velocity, charge status, emergency power-off)
    * a `world_query` callback that is the SENSOR: it returns only other
      robots' poses within `sensor_range`, and only if the caller pays the
      sensing cost (it is invoked at most once per cycle).

There is no reference to the fleet dict, task table or ground truth anywhere
in this module.  Everything else arrives through message handlers.

Workflow executed every control cycle (10 Hz):

 SENSE → LOCALIZE → UPDATE LOCAL WORLD MODEL → RECEIVE PEER INFORMATION →
 UPDATE PEER KNOWLEDGE → EVALUATE CURRENT TASK → PLAN/REPLAN →
 PREDICT FUTURE SPACE-TIME CONFLICTS → NEGOTIATE WITH PEERS →
 CHECK SAFETY/UEAC → CHECK CHOKE ADMISSION → GENERATE MOTION COMMAND →
 EXECUTE PHYSICALLY → PUBLISH INTENT → repeat
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .types import Pose, Rect, RobotSpec, TaskDef, clamp, angle_diff
from .comms import SimBus, Message
from .ueac import ChokeRegistry, evaluate_ueac
from .planner import astar, smooth_path, path_length
from .warehouse import Warehouse


# ---------------------------------------------------------------------------
# Local knowledge structures (built ONLY from sensors + comms)
# ---------------------------------------------------------------------------

@dataclass
class PeerKnowledge:
    rid: str
    pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # sensed or communicated
    pose_source: str = "none"       # sensor | intent | none
    pose_t: float = -999.0
    speed: float = 0.0
    traj: List[Tuple[float, float, float]] = field(default_factory=list)  # (t,x,y)
    traj_id: int = -1
    traj_t: float = -999.0
    waiting_for: Optional[str] = None
    wait_reason: str = ""
    lease_zone: Optional[str] = None
    stale: bool = False

    def occupancy_at(self, t: float, now: float) -> Optional[Tuple[float, float]]:
        """Where peer believes itself at absolute sim time t (per its intent).
        Intent trajectory times are relative to the publish instant."""
        if not self.traj:
            return None
        rel = t - self.traj_t   # convert absolute query time to peer-relative frame
        if rel < self.traj[0][0] - 0.25 or rel > self.traj[-1][0] + 0.25:
            return None
        # interpolate
        pts = self.traj
        lo, hi = 0, len(pts) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if pts[mid][0] <= rel:
                lo = mid
            else:
                hi = mid
        t0, x0, y0 = pts[lo]
        t1, x1, y1 = pts[hi]
        f = 0.0 if t1 == t0 else clamp((rel - t0) / (t1 - t0), 0.0, 1.0)
        return (x0 + (x1 - x0) * f, y0 + (y1 - y0) * f)


@dataclass
class TrajPoint:
    t: float     # absolute sim time
    d: float     # distance along path
    s: float     # arc position
    v: float


class Brain:
    CONTROL_DT = 0.1          # 10 Hz decision loop
    PLAN_HZ_DT = 0.25         # trajectory sample interval
    HORIZON_S = 14.0
    STALE_S = 3.0             # peer info older than this is rejected as stale

    def __init__(self, spec: RobotSpec, wh: Warehouse, registry: ChokeRegistry,
                 bus: SimBus, physics: dict):
        self.spec = spec
        self.id = spec.id
        self.wh = wh                      # local stored map copy (static geometry)
        self.registry = registry          # deterministic protocol state (leases)
        self.bus = bus
        self.phys = physics               # {get_pose,get_speed,get_energy,set_velocity,is_charging,emergency_off}
        self.rng = physics["rng"]         # seeded per-robot noise generator

        # ---- estimated state (LOCALIZE) ----
        true_pose: Pose = physics["get_pose"]()
        self.est: Pose = true_pose
        self.est_v: float = physics["get_speed"]()
        self.energy_wh: float = physics["get_energy"]()
        self.charging: bool = physics["is_charging"]()
        self.net_ok: bool = True

        # ---- task state ----
        self.task: Optional[TaskDef] = None
        self.task_epoch: int = 0
        self.phase: str = "idle"          # idle|to_pickup|at_pickup|to_drop|done|to_charge|charging
        self.carrying: bool = False

        # ---- plan ----
        self.waypoints: List[Tuple[float, float]] = []
        self.wp_index: int = 0
        self.traj: List[TrajectorySample] = []
        self.traj_id: int = 0
        self.dest_label: str = "-"
        self.plan_replans: int = 0
        self.zone_penalty: Dict[str, float] = {}

        # ---- motion command ----
        self.v_cmd: float = 0.0
        self.w_cmd: float = 0.0
        self.last_decision: str = "IDLE"
        self.decision_detail: str = ""
        self.yield_to: Optional[str] = None
        self.waiting_for: Optional[str] = None
        self.wait_reason: str = ""
        self.wait_since: float = 0.0
        self.stuck_escape_until: float = 0.0

        # ---- negotiation state ----
        self.negot_seq: int = 0
        self.pending_proposals: Dict[int, dict] = {}     # seq -> outgoing proposal
        self.incoming_resolved: set = set()
        self.accepted_offset_until: float = 0.0          # hold position until
        self.temporal_offset_s: float = 0.0
        self.recent_negotiations: List[dict] = []

        # ---- peer knowledge ----
        self.peers: Dict[str, PeerKnowledge] = {}
        self.seen_msg_ids: set = set()
        self._peer_priority: Dict[str, tuple] = {}
        self._peer_specs: Dict[str, Tuple[float, float]] = {spec.id: (spec.length, spec.width)}
        self._nego_cooldown: Dict[str, float] = {}
        self._slow_until: float = 0.0
        self._in_choke: Optional[str] = None
        self.charge_target: Optional[Tuple[float, float]] = None
        self.wait_started: float = 0.0
        self._wf_prev: Optional[str] = None
        self._escape_wp: Optional[Tuple[float, float]] = None
        self._award_seen: set = set()
        self.yield_until: float = 0.0

        # ---- observability ring buffers ----
        self.log: List[dict] = []
        self.ueac_last: Optional[dict] = None
        self.conflict_last: Optional[dict] = None
        self.cycle_count: int = 0
        self.t_last_plan: float = -99.0
        self.t_last_intent: float = -99.0
        self.t_last_dep: float = -99.0
        self.deadlock_participant: Optional[List[str]] = None
        self.escape_role: str = ""

        # ---- subscriptions ----
        for topic in ("intent", "proposal", "response", "lease", "dependency",
                      "task_ad", "task_award", "net_status"):
            bus.subscribe(self.id, topic, self._on_msg)

    # =====================================================================
    # helpers
    # =====================================================================
    def _log(self, kind: str, msg: str, **extra):
        entry = {"t": round(self.now, 2), "robot": self.id, "kind": kind, "msg": msg}
        entry.update(extra)
        self.log.append(entry)
        if len(self.log) > 400:
            del self.log[:200]

    @property
    def now(self) -> float:
        return self.phys["time"]()

    # =====================================================================
    # MAIN WORKFLOW (called by simulation at CONTROL_DT)
    # =====================================================================
    def cycle(self):
        # 1. SENSE -------------------------------------------------------
        sensed = self._sense()
        # 2. LOCALIZE ----------------------------------------------------
        self._localize()
        # 3. UPDATE LOCAL WORLD MODEL ------------------------------------
        self._update_world_model(sensed)
        # 4./5. peer information already streamed via _on_msg into self.peers
        self._mark_stale()
        # 6. EVALUATE CURRENT TASK (+ battery feasibility / release) -----
        self._evaluate_task()
        # 7. PLAN / REPLAN -----------------------------------------------
        self._plan_if_needed()
        base = getattr(self, "baseline", False)
        # 8. PREDICT FUTURE SPACE-TIME CONFLICTS -------------------------
        conflict = None if base else self._predict_conflicts()
        # 9. NEGOTIATE ---------------------------------------------------
        if not base:
            self._negotiate(conflict)
        # 10./11. SAFETY / UEAC / CHOKE ADMISSION ------------------------
        choke_hold = False if base else self._choke_admission()
        # deadlock bookkeeping
        if not base:
            self._deadlock_watch()
        # 12. GENERATE MOTION COMMAND ------------------------------------
        self._motion_command(conflict, choke_hold)
        # 13. EXECUTE PHYSICALLY ------------------------------------------
        self.phys["set_velocity"](max(0.0, self.v_cmd), self.w_cmd)
        # 14. PUBLISH INTENT ----------------------------------------------
        self._publish_intent()
        self._publish_dependency()
        self.cycle_count += 1

    # ---------------------------------------------------------------- SENSE
    def _sense(self) -> List[dict]:
        """One sensor sweep: peers within range (legitimate proximity sensing)."""
        out = []
        for p in self.phys["sensors"]():        # [{'id','x','y','th','v'}] within sensor_range
            out.append(p)
        self._last_sensed = out
        return out

    # ------------------------------------------------------------- LOCALIZE
    def _localize(self):
        tp = self.phys["get_pose"]()
        sigma = self.spec.loc_sigma_xy * (1.0 + 0.35 * (self.est_v / max(0.1, self.spec.cruise_speed)))
        ex = self.rng.gauss(0.0, sigma)
        ey = self.rng.gauss(0.0, sigma)
        eth = self.rng.gauss(0.0, sigma * 0.4)
        self.est = Pose(tp.x + ex, tp.y + ey, tp.th + eth)
        self.est_v = self.phys["get_speed"]()
        self.energy_wh = self.phys["get_energy"]()
        self.charging = self.phys["is_charging"]()
        self.net_ok = self.phys["net_ok"]()

    # ------------------------------------------------- LOCAL WORLD MODEL UPD
    def _update_world_model(self, sensed: List[dict]):
        for s in sensed:
            pk = self.peers.setdefault(s["id"], PeerKnowledge(s["id"]))
            pk.pose = (s["x"], s["y"], s["th"])
            pk.pose_source = "sensor"
            pk.pose_t = self.now
            pk.speed = s["v"]
            pk.stale = False

    def _mark_stale(self):
        for rid, pk in list(self.peers.items()):
            fresh = max(pk.pose_t, pk.traj_t)
            pk.stale = (self.now - fresh) > self.STALE_S

    # ------------------------------------------------------ MESSAGE HANDLING
    def _on_msg(self, m: Message):
        if m.id in self.seen_msg_ids:
            return                              # duplicate rejection
        self.seen_msg_ids.add(m.id)
        if len(self.seen_msg_ids) > 4000:
            self.seen_msg_ids = set(list(self.seen_msg_ids)[-2000:])
        if self.now - m.t_sent > m.ttl:
            return                              # stale-message rejection
        h = getattr(self, "_rx_" + m.topic, None)
        if h:
            h(m)

    def _rx_intent(self, m: Message):
        rid = m.sender
        pk = self.peers.setdefault(rid, PeerKnowledge(rid))
        p = m.payload
        if p.get("prio"):
            self._peer_priority[rid] = tuple(p["prio"])
        if p.get("dims"):
            self._peer_specs[rid] = tuple(p["dims"])
        if pk.traj_id >= p["traj_id"] and self.now - pk.traj_t < 1.0:
            pass  # keep newer trajectory unless pose is fresher below
        elif p.get("traj"):
            pk.traj = [tuple(pt) for pt in p["traj"]]
            pk.traj_id = p["traj_id"]
            pk.traj_t = self.now
        if pk.pose_source != "sensor" or (self.now - pk.pose_t) > 0.5:
            pk.pose = (p["x"], p["y"], p["th"])
            pk.pose_source = "intent"
            pk.pose_t = self.now
        pk.speed = p.get("v", 0.0)
        pk.lease_zone = p.get("lease")
        pk.stale = False

    def _rx_proposal(self, m: Message):
        if m.target != self.id:
            return
        p = m.payload
        # Deterministic compatible outcome: proposer has priority iff
        # (deadline urgency, task priority, remaining time, remaining dist, id)
        # favours them. Safety always overrides: we never accept an action that
        # would force us into an unsafe state; rejecting is always safe.
        mine = self._priority_tuple()
        theirs = tuple(p["priority"])
        # smaller tuple = higher priority; proposer claims `theirs`
        if theirs < mine:
            # they have priority -> we yield with the requested action
            self._accept_proposal(p)
            self.bus.send("response", self.id,
                          {"seq": p["seq"], "proposer": m.sender, "result": "ACCEPT",
                           "action": p["action"]}, self.now, target=m.sender,
                          scope_xy=(self.est.x, self.est.y))
        else:
            self.bus.send("response", self.id,
                          {"seq": p["seq"], "proposer": m.sender, "result": "REJECT"},
                          self.now, target=m.sender, scope_xy=(self.est.x, self.est.y))

    def _rx_response(self, m: Message):
        p = m.payload
        seq = p["seq"]
        pend = self.pending_proposals.get(seq)
        if not pend:
            return
        if p["result"] == "ACCEPT":
            pend["accepted_by"].append(m.sender)
            if len(pend["accepted_by"]) >= pend["need"]:
                # all affected peers accepted our plan adjustment
                self._apply_own_adjustment(pend)
            del self.pending_proposals[seq]
        else:
            # rejected: WE yield instead (peer has priority) — safety-preserving fallback
            del self.pending_proposals[seq]
            self._yield_to(m.sender, reason="proposal rejected")

    def _rx_lease(self, m: Message):
        p = m.payload
        for rid, zid in p.get("held", {}).items():
            pk = self.peers.setdefault(rid, PeerKnowledge(rid))
            pk.lease_zone = zid

    def _rx_dependency(self, m: Message):
        p = m.payload
        pk = self.peers.setdefault(m.sender, PeerKnowledge(m.sender))
        pk.waiting_for = p.get("waiting_for")
        pk.wait_reason = p.get("reason", "")

    def _rx_net_status(self, m: Message):
        self._log("NET", f"network state -> {m.payload['state']}")

    def _rx_task_ad(self, m: Message):
        """Advertise handling: independent bid decision."""
        td = TaskDef(**{k: v for k, v in m.payload["task"].items()
                        if k in TaskDef.__dataclass_fields__})
        td.pickup_cell = tuple(td.pickup_cell) if td.pickup_cell else None
        td.drop_cell = tuple(td.drop_cell) if td.drop_cell else None
        if self.task is not None or self.phase != "idle":
            return
        est = self._task_cost_estimate(td)
        if est is None:
            self._log("TASK", f"T{td.id}: declined advertisement (not feasible)", task=td.id)
            return
        dist, etime, eenergy = est
        score = etime + max(0.0, (td.deadline_s - etime)) * 0.0 + \
            (td.priority * 0.5) + (eenergy / self.spec.capacity_wh) * 30.0
        free = (self.task is None and self.phase == "idle")
        if not free:
            score += 10000.0   # never outbid a free robot -> no ambiguous ownership
        self.bus.send("task_bid", self.id,
                      {"task": td.id, "epoch": m.payload["epoch"], "score": round(score, 3),
                       "eta_s": round(etime, 2), "energy_wh": round(eenergy, 1),
                       "dist_m": round(dist, 1), "free": free},
                      self.now, epoch=m.payload["epoch"], scope_xy=(self.est.x, self.est.y))
        self._log("TASK", f"bid T{td.id} score={score:.1f} eta={etime:.0f}s need={eenergy:.0f}Wh",
                  task=td.id)

    def _rx_task_award(self, m: Message):
        p = m.payload
        if p.get("winner") != self.id:
            return
        key = (p["task"], p.get("epoch", 0))
        if key in self._award_seen:
            return                       # duplicate award message
        self._award_seen.add(key)
        if len(self._award_seen) > 200:
            self._award_seen = set(list(self._award_seen)[-100:])
        if self.task is not None or self.phase != "idle":
            self._log("TASK", f"award for {p['task']} declined — I am busy; FMS will re-award")
            return
        obj = p.get("task_obj") or p["task"]
        if isinstance(obj, str):
            return
        td = TaskDef(**{k: v for k, v in obj.items() if k in TaskDef.__dataclass_fields__})
        td.pickup_cell = tuple(td.pickup_cell) if td.pickup_cell else None
        td.drop_cell = tuple(td.drop_cell) if td.drop_cell else None
        self.task = td
        self.task_epoch = p["epoch"]
        self.phase = "to_pickup"
        self.waypoints = []
        self.wp_index = 0
        self.dest_label = f"{td.pickup_zone or 'PICKUP'}"
        self._log("TASK", f"AWARD accepted: {td.id} pickup {td.pickup_zone} -> drop {td.drop_zone}",
                  task=td.id)

    # ------------------------------------------------------------- BIDDING
    def _task_cost_estimate(self, td: TaskDef) -> Optional[Tuple[float, float, float]]:
        """(distance, time, energy) estimate for this robot to do task td, or None
        if physically impossible for THIS robot (payload, battery reserve...)."""
        if td.payload_kg > self.spec.payload_kg:
            return None
        start = (self.est.x, self.est.y)
        px, py = self.wh.station_pos(td.pickup_zone) if td.pickup_zone in self.wh.stations else (start[0], start[1])
        dx_, dy_ = self.wh.station_pos(td.drop_zone) if td.drop_zone in self.wh.stations else (start[0], start[1])
        r = self.spec.footprint_radius()
        c1 = astar(self.wh, self.wh.world_to_cell(*start), self.wh.world_to_cell(px, py), r)
        if c1 is None:
            return None
        c2 = astar(self.wh, self.wh.world_to_cell(px, py), self.wh.world_to_cell(dx_, dy_), r)
        if c2 is None:
            return None
        p1 = smooth_path(self.wh, [self.wh.cell_to_world(*c) for c in c1], r)
        p2 = smooth_path(self.wh, [self.wh.cell_to_world(*c) for c in c2], r)
        d1, d2 = path_length(p1), path_length(p2)
        v = self.spec.cruise_speed
        t1 = d1 / v + 3.0                    # pickup operation
        t2 = d2 / (v * 0.85) + 3.0           # loaded slower + drop op
        e1 = self.spec.energy_rate_w(v, False) * t1 / 3600.0
        e2 = self.spec.energy_rate_w(v * 0.85, True) * t2 / 3600.0
        total_e = e1 + e2
        if self.energy_wh - total_e < self.spec.reserve_wh:
            return None                      # cannot preserve reserve after task
        return (d1 + d2, t1 + t2, total_e)

    # ------------------------------------------------------------ TASK EVAL
    def _evaluate_task(self):
        if self.phase == "staging":
            self._end_staging()
            return
        if self.task is None:
            return
        # Battery feasibility re-check: can I still finish + keep reserve?
        if self.phase in ("to_pickup", "at_pickup", "to_drop"):
            rem = self._remaining_estimate()
            if rem is not None:
                _, etime, eenergy = rem
                margin = self.energy_wh - eenergy - self.spec.reserve_wh
                if margin < 0 and not self.charging:
                    self._release_task(f"battery insufficient: have {self.energy_wh:.0f}Wh, "
                                       f"need {eenergy + self.spec.reserve_wh:.0f}Wh")
                    return
                self._battery_margin = margin
        # completion detection handled by physics arrival -> phase machine below
        self._phase_machine()

    def _remaining_estimate(self):
        if self.task is None:
            return None
        tgt = self._current_goal()
        if tgt is None:
            return None
        r = self.spec.footprint_radius()
        cells = astar(self.wh, self.wh.world_to_cell(self.est.x, self.est.y),
                      self.wh.world_to_cell(*tgt), r, self.zone_penalty)
        if cells is None:
            return None
        pts = smooth_path(self.wh, [self.wh.cell_to_world(*c) for c in cells], r)
        d = path_length(pts)
        v = self.spec.cruise_speed * (0.85 if self.carrying else 1.0)
        t = d / max(0.3, v) + (3.0 if self.phase != "to_drop" else 3.0)
        e = self.spec.energy_rate_w(v, self.carrying) * t / 3600.0
        return (d, t, e)

    def _release_task(self, reason: str):
        if self.task is None:
            return
        tid, ep = self.task.id, self.task_epoch
        self.bus.send("task_release", self.id, {"task": tid, "epoch": ep, "reason": reason},
                      self.now, epoch=ep, scope_xy=(self.est.x, self.est.y))
        self._log("TASK", f"RELEASE {tid}: {reason}", task=tid)
        self._award_seen.pop((tid, ep), None)
        self.task = None
        self.phase = "idle"
        self.carrying = False
        self.waypoints = []
        self.dest_label = "-"
        # low battery -> head to charger autonomously
        if self.energy_wh < self.spec.reserve_wh * 1.5:
            self._goto_charge()

    def _begin_staging(self, spot, reason: str):
        """Decentralised congestion/cycle relief: drive to a nearby free spot
        out of the traffic lane, then resume normal evaluation."""
        self.phase = "staging"
        self._stage_goal = spot
        self.waypoints = []
        self.t_last_plan = -99
        self.dest_label = "STAGING"
        self.staged_until = self.now + 8.0
        self._log("STAGE", f"pulling aside ({reason})")

    def _end_staging(self):
        gx, gy = getattr(self, "_stage_goal", (None, None))
        arrived = gx is not None and math.hypot(gx - self.est.x, gy - self.est.y) < 0.7 and self.est_v < 0.12
        timeout = self.now > getattr(self, "staged_until", 0) + 15.0
        if not (arrived or timeout):
            return False
        tid = self.task.id if self.task else None
        if self.task:
            self.phase = "to_drop" if self.carrying else "to_pickup"
            self.dest_label = self.task.drop_zone if self.phase == "to_drop" else self.task.pickup_zone
        elif self.energy_wh < self.spec.reserve_wh * 1.5:
            self._goto_charge()
            return True
        else:
            self.phase = "idle"
            self.dest_label = "-"
        self.waypoints = []
        self.t_last_plan = -99
        self.waiting_for = None
        self._wf_prev = None
        self.escape_role = ""
        self.stuck_escape_until = 0.0
        self._log("STAGE", f"staged clear; resuming as {self.phase} ({tid or 'no task'})")
        return True

    def _goto_charge(self):
        charges = [st for st in self.wh.stations.values() if st.kind == "charge"]
        if not charges:
            return
        best = min(charges, key=lambda s: math.hypot(*(lambda p: (p[0] - self.est.x, p[1] - self.est.y))(self.wh.cell_to_world(*s.cell))))
        cx, cy = self.wh.cell_to_world(*best.cell)
        self.charge_target = (cx, cy)
        self.phase = "to_charge"
        self.dest_label = best.id
        self.waypoints = [(cx, cy)]
        self.wp_index = 0
        self._log("BATTERY", f"heading to charging bay {best.id}")

    def _current_goal(self) -> Optional[Tuple[float, float]]:
        if self.task is None:
            return None
        if self.phase in ("to_pickup", "at_pickup"):
            st = self.wh.stations.get(self.task.pickup_zone)
            return self.wh.cell_to_world(*st.cell) if st else None
        if self.phase == "to_drop":
            st = self.wh.stations.get(self.task.drop_zone)
            return self.wh.cell_to_world(*st.cell) if st else None
        return None

    def _phase_machine(self):
        goal = self._current_goal()
        if goal is None:
            return
        d = math.hypot(goal[0] - self.est.x, goal[1] - self.est.y)
        if self.phase == "to_pickup" and d < 0.8:
            self.phase = "at_pickup"
            self.pickup_at = self.now
            self.waypoints = []
            self._log("TASK", f"arrived at pickup {self.task.pickup_zone}; grabbing pallet")
        elif self.phase == "at_pickup" and self.now - getattr(self, "pickup_at", 0) > 1.5:
            self.carrying = True
            self.phase = "to_drop"
            self.waypoints = []
            self.t_last_plan = -99
            self.dest_label = self.task.drop_zone or "DROP"
            self._log("TASK", f"payload secured; en route to {self.task.drop_zone}")
        elif self.phase == "to_drop" and d < 0.8:
            self.drop_at = self.now
            self.phase = "at_drop"
            self.waypoints = []
            self._log("TASK", f"arrived at drop {self.task.drop_zone}; placing pallet")
        elif self.phase == "at_drop" and self.now - getattr(self, "drop_at", 0) > 1.5:
            self.bus.send("task_done", self.id, {"task": self.task.id, "epoch": self.task_epoch},
                          self.now, epoch=self.task_epoch, scope_xy=(self.est.x, self.est.y))
            self._log("TASK", f"COMPLETED {self.task.id} at {self.task.drop_zone}")
            self.task = None
            self.carrying = False
            self.phase = "idle"
            self.waypoints = []
            self.dest_label = "-"
        elif self.phase == "to_charge":
            gx, gy = getattr(self, "charge_target", (None, None))
            if gx is not None and math.hypot(gx - self.est.x, gy - self.est.y) < 0.5:
                self.phase = "charging"
                self.dest_label = "CHARGING"
        elif self.phase == "charging":
            if self.energy_wh >= 0.95 * self.spec.capacity_wh:
                self.phase = "idle"
                self.dest_label = "-"
                self._log("BATTERY", "recharged to 95%, back to idle")

    # --------------------------------------------------------------- PLANNING
    def _plan_if_needed(self):
        if self.phase in ("idle", "at_pickup", "at_drop", "charging"):
            self.waypoints = []
            return
        goal = self._current_goal()
        if goal is None and self.phase in ("to_charge", "staging"):
            goal = self.charge_target or getattr(self, "_stage_goal", None)
        if goal is None:
            return
        stuck = (self.est_v < 0.05 and self.now - self.t_last_plan > 4.0)
        need = (not self.waypoints) or (self.now - self.t_last_plan > 6.0) or stuck
        if need:
            r = self.spec.footprint_radius()
            sc = self.wh.world_to_cell(self.est.x, self.est.y)
            gc = self.wh.world_to_cell(*goal)
            cells = astar(self.wh, sc, gc, r, self.zone_penalty)
            if cells is None:
                cells = astar(self.wh, sc, gc, r)  # retry ignoring soft penalties
            if cells:
                pts = smooth_path(self.wh, [self.wh.cell_to_world(*c) for c in cells], r)
                if pts and abs(pts[0][0] - self.est.x) + abs(pts[0][1] - self.est.y) > 2.5:
                    pts = [(self.est.x, self.est.y)] + pts
                self.waypoints = pts
                self.wp_index = 0
                # skip waypoints already behind us
                while self.wp_index < len(pts) - 1 and \
                        math.hypot(pts[self.wp_index][0] - self.est.x,
                                   pts[self.wp_index][1] - self.est.y) < 0.7:
                    self.wp_index += 1
                self.t_last_plan = self.now
                if self.cycle_count % 1 == 0 and need and self.traj:
                    self.plan_replans += 1
        self._build_traj(goal)

    def _build_traj(self, goal):
        """Forward-simulate velocity profile along waypoints -> space-time trajectory."""
        pts = list(self.waypoints[self.wp_index:]) if self.waypoints else [goal]
        if not pts:
            pts = [goal]
        if not pts or abs(pts[-1][0] - goal[0]) + abs(pts[-1][1] - goal[1]) > 0.4:
            pts = pts + [goal]
        samples: List[TrajectorySample] = []
        s = 0.0
        segl = []
        for i in range(len(pts) - 1):
            segl.append(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]))
        total = sum(segl)
        if total < 1e-6:
            self.traj = []
            return
        v = 0.0
        t = self.now
        pos = 0.0
        dt = self.PLAN_HZ_DT
        cruise = self.spec.cruise_speed * (0.85 if self.carrying else 1.0)
        # simple accel/decel profile toward end of remaining path
        last_emit = -1.0
        while pos < total and t - self.now < self.HORIZON_S:
            remaining = total - pos
            v_brake = math.sqrt(max(0.0, 2.0 * self.spec.decel * max(0.0, remaining - 0.15)))
            v_target = min(cruise, v_brake)
            if v < v_target:
                v = min(v_target, v + self.spec.accel * dt)
            else:
                v = max(v_target, v - self.spec.max_decel * dt)
            v = max(0.0, v)
            pos += v * dt
            t += dt
            x, y, _ = self._point_at_distance(pts, segl, pos)
            if t - last_emit >= dt - 1e-9:
                samples.append(TrajectorySample(t, pos, x, y, v))
                last_emit = t
            if v < 0.02 and remaining < 0.2:
                break
        self.traj = samples

    @staticmethod
    def _point_at_distance(pts, segl, d):
        acc = 0.0
        for i, L in enumerate(segl):
            if acc + L >= d or i == len(segl) - 1:
                f = 0.0 if L < 1e-9 else clamp((d - acc) / L, 0.0, 1.0)
                x = pts[i][0] + (pts[i + 1][0] - pts[i][0]) * f
                y = pts[i][1] + (pts[i + 1][1] - pts[i][1]) * f
                th = math.atan2(pts[i + 1][1] - pts[i][1], pts[i + 1][0] - pts[i][0])
                return x, y, th
            acc += L
        return pts[-1][0], pts[-1][1], 0.0

    # ----------------------------------------------------- CONFLICT PREDICTION
    def _predict_conflicts(self) -> Optional[dict]:
        """Compare my planned space-time occupancy against peer trajectories.

        Conflict when swept disks overlap within a temporal risk window derived
        from BOTH robots' braking capability (predictive, not reactive).
        """
        if not self.traj or len(self.traj) < 2:
            self.conflict_last = None
            return None
        worst = None
        for rid, pk in sorted(self.peers.items()):
            if pk.stale or not pk.traj or rid == self.id:
                continue
            for mine in self.traj[::2]:
                their = pk.occupancy_at(mine.t, self.now)
                if their is None:
                    continue
                gap = math.hypot(their[0] - mine.x, their[1] - mine.y)
                # required separation: both footprints + combined braking allowance
                req = ((self.spec.diag + self._peer_diag(rid)) / 2.0 +
                       0.35 * (mine.v + pk.speed) +
                       2.0 * self.spec.loc_sigma_xy)
                if gap < req:
                    risk = 1.0 - gap / max(req, 1e-6)
                    if worst is None or risk > worst["risk"]:
                        worst = {"with": rid, "t": mine.t, "gap": gap, "req": req,
                                 "risk": risk, "mx": mine.x, "my": mine.y}
        self.conflict_last = ({k: (round(v, 2) if isinstance(v, float) else v)
                               for k, v in worst.items()} if worst else None)
        return worst

    def _peer_diag(self, rid: str) -> float:
        # peer dimension knowledge is legitimate: robot specs are published on
        # the network at join time; fall back to conservative default.
        spec = self._peer_specs.get(rid)
        if spec:
            return math.hypot(spec[0], spec[1])
        return 1.44

    # ----------------------------------------------------------- NEGOTIATION
    def _priority_tuple(self):
        """Deterministic priority: deadline urgency, task prio, remaining time,
        remaining distance, robot id.  Smaller tuple = higher priority."""
        rem = self._remaining_estimate() if self.task else None
        slack = 9999.0
        prio = 9
        rt = rd = 9999.0
        if self.task:
            deadline_abs = self.task.created_at + self.task.deadline_s
            slack = deadline_abs - (self.now + (rem[1] if rem else 9999.0))
            prio = self.task.priority
            rt = rem[1] if rem else 9999.0
            rd = rem[0] if rem else 9999.0
        return (round(slack, 2), prio, round(rt, 2), round(rd, 2), self.id)

    def _negotiate(self, conflict: Optional[dict]):
        # expire old holds
        if self.yield_to and self.now > getattr(self, "yield_until", 0):
            self.yield_to = None
        if not conflict:
            return
        rid = conflict["with"]
        pk = self.peers.get(rid)
        mine = self._priority_tuple()
        theirs = self._peer_priority.get(rid)
        # If we already negotiated with this peer recently, don't ping-pong
        cool = self._nego_cooldown.get(rid, -99)
        if self.now - cool < 4.0:
            return
        self._nego_cooldown[rid] = self.now
        if theirs is not None and tuple(theirs) < mine:
            # peer has priority: yield deterministically (temporal offset)
            self._yield_to(rid, reason=f"predicted conflict @{conflict['t']:.1f}s, peer higher priority")
        else:
            # propose that THEY adjust; if they reject we will yield anyway
            self.negot_seq += 1
            action = "TEMPORAL_OFFSET"
            if conflict["risk"] > 0.75:
                action = "YIELD"
            self.pending_proposals[self.negot_seq] = {
                "seq": self.negot_seq, "target": rid, "action": action,
                "conflict": conflict, "accepted_by": [], "need": 1, "t": self.now}
            self.bus.send("proposal", self.id,
                          {"seq": self.negot_seq, "action": action,
                           "priority": list(mine), "conflict_t": round(conflict["t"], 2),
                           "offset_s": 2.0},
                          self.now, target=rid, ttl=3.0, scope_xy=(self.est.x, self.est.y))
            self._log("NEGO", f"PROPOSE {action} to {rid} (conflict t={conflict['t']:.1f}s gap={conflict['gap']:.2f}<{conflict['req']:.2f}m)")
            self.recent_negotiations.append({"t": round(self.now, 1), "dir": "->", "peer": rid,
                                             "action": action, "result": "pending"})

    def _yield_to(self, rid: str, reason: str):
        self.yield_to = rid
        self.temporal_offset_s = 2.0
        self.accepted_offset_until = self.now + 2.5
        self.yield_until = self.now + 4.0
        self.last_decision = "TEMPORAL_OFFSET"
        self.decision_detail = f"hold {self.temporal_offset_s:.1f}s for {rid}: {reason}"
        self._log("NEGO", f"YIELD/offset {self.temporal_offset_s:.1f}s -> {rid}: {reason}")
        self.recent_negotiations.append({"t": round(self.now, 1), "dir": "<-", "peer": rid,
                                         "action": "YIELD", "result": "accepted"})

    def _accept_proposal(self, p: dict):
        self.yield_to = p.get("proposer") or p.get("sender")
        self.temporal_offset_s = float(p.get("offset_s", 2.0))
        self.accepted_offset_until = self.now + self.temporal_offset_s + 0.5
        self.yield_until = self.now + self.temporal_offset_s + 2.0
        self.last_decision = "TEMPORAL_OFFSET"
        self.decision_detail = f"accepted proposal from {self.yield_to}"
        self._log("NEGO", f"ACCEPT proposal #{p['seq']} from {self.yield_to}: {p['action']}")

    def _apply_own_adjustment(self, pend: dict):
        # All peers accepted OUR plan: we keep trajectory but slow slightly to
        # guarantee the negotiated temporal separation.
        self.last_decision = "SLOW"
        self.decision_detail = f"peers accepted my {pend['action']}; trimming speed"
        self._slow_until = self.now + 3.0
        self._log("NEGO", f"peers ACCEPTED my {pend['action']} (#{pend['seq']})")

    # -------------------------------------------------- UEAC / CHOKE ADMIT
    def _next_choke(self) -> Optional[Tuple[object, float]]:
        """First choke zone on my planned path ahead, plus distance to entry."""
        if not self.waypoints:
            return None
        travelled = 0.0
        pts = self.waypoints[self.wp_index:]
        if not pts:
            return None
        prev = (self.est.x, self.est.y)
        for q in pts[1:]:
            seg = math.hypot(q[0] - prev[0], q[1] - prev[1])
            steps = max(1, int(seg / 0.3))
            for k in range(1, steps + 1):
                f = k / steps
                x = prev[0] + (q[0] - prev[0]) * f
                y = prev[1] + (q[1] - prev[1]) * f
                z = self.wh.zone_at(x, y)
                if z is not None and z.kind == "choke":
                    return (z, travelled + seg * f)
            travelled += seg
            prev = q
        return None

    def _egress_clear(self, zone) -> bool:
        """Beyond-far-side check using fused knowledge (own sensors + intents)."""
        # exit cell: furthest point of zone rect along travel direction
        for rid, pk in self.peers.items():
            if rid == self.id or pk.stale:
                continue
            x, y = pk.pose[0], pk.pose[1]
            near = (zone.rect.x0 - 2.0 <= x <= zone.rect.x1 + 2.0 and
                    zone.rect.y0 - 2.0 <= y <= zone.rect.y1 + 2.0)
            inside_exit = (zone.rect.x0 - 2.5 <= x <= zone.rect.x1 + 2.5 and
                           zone.rect.y0 - 2.5 <= y <= zone.rect.y1 + 2.5)
            if near and inside_exit and pk.speed < 0.05:
                return False
        return True

    def _choke_admission(self) -> bool:
        """Returns True => must stop before entering the choke."""
        nc = self._next_choke()
        self._in_choke = None
        cz = self.wh.zone_at(self.est.x, self.est.y)
        if cz is not None and cz.kind == "choke":
            self._in_choke = cz.id
            self.registry.renew(cz.id, self.id, self.now)
            self.bus.send("lease", self.id, {"held": {self.id: cz.id}}, self.now,
                          ttl=2.0, scope_xy=(self.est.x, self.est.y))
            return False
        if nc is None:
            # no choke ahead: release any leases we still hold
            for zid in list(self.registry.leases.keys()):
                if self.id in self.registry.leases.get(zid, {}):
                    self.registry.release(zid, self.id)
            return False
        zone, dist = nc
        sensor_free = all(
            not (zone.rect.contains(s["x"], s["y"]) and s["id"] != self.id)
            for s in getattr(self, "_last_sensed", []))
        inside_ids = [s["id"] for s in getattr(self, "_last_sensed", [])
                      if zone.rect.contains(s["x"], s["y"]) and s["id"] != self.id]
        # predictive: also treat peers whose intent says they'll be inside soon
        predicted_ids = list(inside_ids)
        for rid, pk in self.peers.items():
            if rid == self.id or pk.stale or not pk.traj:
                continue
            entry_t = self.now + dist / max(0.3, self.est_v + 0.001)
            occ = pk.occupancy_at(entry_t + 1.0, self.now)
            if occ and zone.rect.contains(occ[0], occ[1], margin=0.6):
                predicted_ids.append(rid)
        res = evaluate_ueac(zone, self.id, self.est_v, self.spec.max_decel,
                            self.spec.loc_sigma_xy, self.spec.diag / 2.0,
                            self.registry, self.now,
                            egress_clear=self._egress_clear(zone),
                            sensor_confirms_free=sensor_free,
                            inside_ids=predicted_ids, already_inside=False)
        self.ueac_last = res.to_dict()
        if res.decision == "GRANT":
            self.bus.send("lease", self.id, {"held": {self.id: zone.id}}, self.now,
                          ttl=2.0, scope_xy=(self.est.x, self.est.y))
            return False
        if res.decision in ("WAIT_LEASE", "HOLD_AT_ENTRY"):
            self.waiting_for = next(iter([h for h in res.factors["active_leases"] if h != self.id]),
                                     res.factors.get("occupants", ["unknown"])[0]
                                     if res.factors.get("occupants") else None)
            self.wait_reason = f"UEAC {res.decision} @ {zone.id}: {res.reasons[-1] if res.reasons else ''}"
            self.last_decision = "CHOKE_HOLD"
            self.decision_detail = self.wait_reason
            return dist < max(1.2, self.spec.braking_distance(self.est_v) + 0.6)
        return True

    # --------------------------------------------------------- DEADLOCK WATCH
    def _deadlock_watch(self):
        """Build dependency graph from communicated waits; detect cycles locally.

        Edge semantics: A -> B means 'A waits for B' (from A's published
        dependency messages).  A cycle among robots that are all waiting long
        enough = deadlock.  Escape role chosen deterministically: the robot
        with the LOWEST priority tuple (i.e. most urgent) stays; the LAST robot
        in the cycle's canonical rotation backs off to a nearby free staging
        spot.  No central manager: each member runs the same detection and
        independently decides its own role.
        """
        eff_wait = self.waiting_for or self.yield_to
        if eff_wait != self._wf_prev:
            self.wait_started = self.now
            self._wf_prev = eff_wait
        self.waiting_for = eff_wait
        # gather graph: self edge + edges learned from dependency msgs
        edges = {}
        if self.waiting_for:
            edges[self.id] = self.waiting_for
        for rid, pk in self.peers.items():
            if pk.waiting_for:
                edges[rid] = pk.waiting_for
        # find cycle containing self (deterministic walk)
        cycle = []
        cur = self.id
        seen = set()
        while cur in edges and cur not in seen:
            seen.add(cur)
            cycle.append(cur)
            cur = edges[cur]
            if cur == self.id and len(cycle) >= 2:
                break
        else:
            self.deadlock_participant = None
            self.escape_role = ""
            return
        if cur != self.id or len(cycle) < 2:
            self.deadlock_participant = None
            return
        if self.now - self.wait_started < 3.0:
            return
        canon = sorted(cycle)
        self.deadlock_participant = canon
        # escape robot = last in canonical order of the cycle rotation starting at min id
        rot = canon[canon.index(min(canon)):] + canon[:canon.index(min(canon))]
        escaper = rot[-1]
        if escaper == self.id:
            if self.escape_role != "escaper":
                self.escape_role = "escaper"
                self._begin_escape()
                self._log("DEADLOCK", f"cycle {canon} detected; I am ESCAPER (deterministic rotation)")
        else:
            if self.escape_role != "holder":
                self.escape_role = "holder"
                self._log("DEADLOCK", f"cycle {canon} detected; {escaper} escapes, I hold")

    def _begin_escape(self):
        """Deterministic escape role: release leases, pull into the nearest free
        off-lane spot. The retreat plan is published as intent — that IS the
        escape negotiation outcome communicated to the cycle members."""
        self.last_decision = "DEADLOCK_ESCAPE"
        self.decision_detail = f"escaping cycle {self.deadlock_participant} to staging spot"
        self._log("ESCAPE", "selected as escaper; retreating via staging spot (intent published)")
        r = self.spec.footprint_radius() + 0.15
        best = None
        cc, cr = self.wh.world_to_cell(self.est.x, self.est.y)
        for d in range(2, 10):
            cands = []
            for dc in range(-d, d + 1):
                cands += [(cc + dc, cr - d), (cc + dc, cr + d)]
            for dr in range(-d + 1, d):
                cands += [(cc - d, cr + dr), (cc + d, cr + dr)]
            for (c, rr) in sorted(cands):
                if not self.wh.in_bounds(c, rr) or self.wh.blocked[c][rr]:
                    continue
                wx, wy = self.wh.cell_to_world(c, rr)
                z = self.wh.zone_at(wx, wy)
                if z is not None and z.kind in ("choke",):
                    continue
                if not self.wh.free_disk(wx, wy, r):
                    continue
                dd = math.hypot(wx - self.est.x, wy - self.est.y)
                if best is None or dd < best[0] - 1e-9:
                    best = (dd, (wx, wy))
            if best:
                break
        for zid in list(self.registry.leases.keys()):
            if self.id in self.registry.leases.get(zid, {}):
                self.registry.release(zid, self.id)
        if best:
            self._begin_staging(best[1], "deadlock cycle escape")
            self.stuck_escape_until = self.now + 30.0
        else:
            self.stuck_escape_until = self.now + 3.0

    # ------------------------------------------------------- MOTION COMMAND
    def _motion_command(self, conflict: Optional[dict], choke_hold: bool):
        if getattr(self, "baseline", False):
            conflict = None
            choke_hold = False
        if self.phase == "charging":
            self.v_cmd = 0.0
            self.w_cmd = 0.0
            return

        pts = self.waypoints[self.wp_index:] if self.waypoints else []
        goal = self._current_goal() or self.charge_target
        lookahead = pts[0] if pts else goal
        if lookahead is None:
            self.v_cmd = 0.0
            self.w_cmd = 0.0
            return
        # advance waypoint index
        while self.wp_index < len(self.waypoints) - 1 and \
                math.hypot(self.waypoints[self.wp_index][0] - self.est.x,
                           self.waypoints[self.wp_index][1] - self.est.y) < 0.9:
            self.wp_index += 1
        tx, ty = self.waypoints[self.wp_index] if self.waypoints else lookahead
        ang = math.atan2(ty - self.est.y, tx - self.est.x)
        diff = angle_diff(ang, self.est.th)
        dist = math.hypot(tx - self.est.x, ty - self.est.y)

        cruise = self.spec.cruise_speed * (0.85 if self.carrying else 1.0)
        v_allow = cruise
        # turn slowdown
        v_allow = min(v_allow, max(0.25, cruise * (1.0 - abs(diff) / math.pi)))
        # negotiated temporal hold (brake-decayed by the kinematic limiter below)
        if self.now < self.accepted_offset_until and self.yield_to:
            v_allow = 0.0
            self.last_decision = "TEMPORAL_OFFSET"
        if self._slow_until > self.now:
            v_allow = min(v_allow, cruise * 0.45)
        # BASELINE: simplistic reactive rule — distance < threshold -> stop
        if getattr(self, "baseline", False):
            for s in sorted(getattr(self, "_last_sensed", []), key=lambda z: z["id"]):
                d = math.hypot(s["x"] - self.est.x, s["y"] - self.est.y)
                ahead = (s["x"] - self.est.x) * math.cos(self.est.th) + \
                        (s["y"] - self.est.y) * math.sin(self.est.th)
                if d < 2.0 and ahead > -0.3:
                    v_allow = 0.0
                    self.last_decision = "STOP_AND_WAIT"
                    self.decision_detail = f"peer {s['id']} within 2.0 m — waiting"
                    self.waiting_for = s["id"]
                    self.wait_reason = "baseline stop-and-wait"
                    break
        # predictive conflict safety margin: brake curve vs conflict point
        if conflict:
            t_conf = conflict["t"] - self.now
            safe_v = max(0.0, (conflict["gap"] - 0.35) / max(0.3, t_conf))
            v_allow = min(v_allow, safe_v)
            if self.yield_to is None and conflict["risk"] > 0.6 and t_conf < 2.0:
                v_allow = 0.0
                self.last_decision = "SAFETY_STOP"
                self.decision_detail = f"reactive safety stop (predicted {conflict['with']} @{t_conf:.1f}s)"
        # choke admission hold
        if choke_hold:
            v_allow = 0.0
        # final kinematic brake limit toward commanded stop
        if v_allow < self.v_cmd:
            dv = self.spec.max_decel * self.CONTROL_DT
            self.v_cmd = max(v_allow, self.v_cmd - dv)
        else:
            self.v_cmd = min(v_allow, self.v_cmd + self.spec.accel * self.CONTROL_DT)
        self.v_cmd = clamp(self.v_cmd, 0.0, self.spec.max_speed)
        # angular: pure-pivot when nearly stopped and badly misaligned
        if self.v_cmd < 0.05 and abs(diff) > 0.6:
            self.w_cmd = clamp(diff * 1.8, -self.spec.max_omega, self.spec.max_omega)
        else:
            self.w_cmd = clamp(angle_diff(ang, self.est.th) * 1.6, -self.spec.max_omega, self.spec.max_omega)
        if self.v_cmd > 0.05 and self.last_decision not in ("SAFETY_STOP", "CHOKE_HOLD",
                                                            "TEMPORAL_OFFSET", "DEADLOCK_ESCAPE", "SLOW"):
            self.last_decision = "TRACK"
            self.decision_detail = f"follow plan -> {self.dest_label}"
        # waiting-for attribution for dependency graph
        if self.v_cmd < 0.02 and self.last_decision in ("SAFETY_STOP", "TEMPORAL_OFFSET") and conflict:
            if self.waiting_for is None:
                self.waiting_for = conflict["with"]
                self.wait_reason = "yielding to peer"
        elif self.v_cmd > 0.05:
            self.waiting_for = None
            self.wait_reason = ""

    # ---------------------------------------------------------- PUBLISHING
    def _publish_intent(self):
        if self.now - self.t_last_intent < 0.5:
            return
        self.t_last_intent = self.now
        self.traj_id += 1
        traj = [[round(s.t - self.now, 2), round(s.x, 2), round(s.y, 2)] for s in self.traj[::2][:40]]
        self.bus.send("intent", self.id,
                      {"x": round(self.est.x, 2), "y": round(self.est.y, 2),
                       "th": round(self.est.th, 2), "v": round(self.est_v, 2),
                       "traj_id": self.traj_id, "traj": traj,
                       "lease": getattr(self, "_in_choke", None),
                       "dest": self.dest_label, "carrying": self.carrying,
                       "dims": [self.spec.length, self.spec.width],
                       "prio": [round(x, 2) if isinstance(x, float) else x
                                for x in self._priority_tuple()]},
                      self.now, ttl=4.0, scope_xy=(self.est.x, self.est.y))

    def _publish_dependency(self):
        if self.now - self.t_last_dep < 0.7:
            return
        self.t_last_dep = self.now
        self.bus.send("dependency", self.id,
                      {"waiting_for": self.waiting_for, "reason": self.wait_reason[:120]},
                      self.now, ttl=3.0, scope_xy=(self.est.x, self.est.y))

    # ------------------------------------------------------------- OBSERVABILITY
    def snapshot(self) -> dict:
        return {
            "id": self.id, "spec": self.spec.to_dict(),
            "est": {"x": round(self.est.x, 2), "y": round(self.est.y, 2), "th": round(self.est.th, 2),
                    "v": round(self.est_v, 2)},
            "energy_wh": round(self.energy_wh, 1), "capacity_wh": self.spec.capacity_wh,
            "phase": self.phase, "task": self.task.id if self.task else None,
            "task_epoch": self.task_epoch if self.task else None,
            "carrying": self.carrying, "dest": self.dest_label,
            "decision": self.last_decision, "detail": self.decision_detail,
            "waiting_for": self.waiting_for, "wait_reason": self.wait_reason,
            "yield_to": self.yield_to,
            "conflict": self.conflict_last, "ueac": self.ueac_last,
            "traj": [[round(s.t, 2), round(s.x, 2), round(s.y, 2), round(s.v, 2)] for s in self.traj[::2]],
            "waypoints": [[round(x, 2), round(y, 2)] for x, y in self.waypoints],
            "peers": {rid: {"pose": [round(pk.pose[0], 2), round(pk.pose[1], 2)],
                            "src": pk.pose_source, "stale": pk.stale,
                            "age": round(self.now - max(pk.pose_t, pk.traj_t), 1),
                            "waiting_for": pk.waiting_for,
                            "traj_pts": len(pk.traj)}
                      for rid, pk in sorted(self.peers.items())},
            "log": self.log[-14:],
            "negotiations": self.recent_negotiations[-8:],
            "deadlock": {"cycle": self.deadlock_participant, "role": self.escape_role},
            "replans": self.plan_replans,
            "net_state": "LINK_LOST" if not self.net_ok else "OK",
        }


@dataclass
class TrajectorySample:
    t: float
    d: float
    x: float
    y: float
    v: float
