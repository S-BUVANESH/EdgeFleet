"""Deterministic simulation engine.

Owns: physics, kinematics, sensors, battery, collisions, robot state, task
state, FMS (task advertisement / bidding window / deterministic award),
metrics and the replay log.  The browser never computes anything; it renders
snapshots produced here.

Fixed timestep 0.1 s, seeded RNG, headless-capable (`run_headless`).
"""
from __future__ import annotations

import json
import math
import random
import time as _wall
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .types import Pose, RobotSpec, TaskDef, clamp
from .warehouse import Warehouse, DEFAULT_WAREHOUSE
from .comms import SimBus, Message
from .ueac import ChokeRegistry
from .brain import Brain
from .planner import astar, smooth_path, path_length

DT = 0.1            # physics & control timestep (10 Hz)


@dataclass
class RobotPhys:
    spec: RobotSpec
    pose: Pose
    v: float = 0.0        # achieved linear speed
    w: float = 0.0
    v_cmd: float = 0.0
    w_cmd: float = 0.0
    energy_wh: float = 0.0
    charging: bool = False
    net_ok: bool = True   # in dead zone / partition side => brain loses link
    rng: random.Random = None
    loaded_kg: float = 0.0
    total_energy_used_wh: float = 0.0


class FMS:
    """Fleet Management System: publishes tasks, collects bids, awards
    deterministically.  NOT a motion controller — after award it is out of
    the loop entirely."""

    BID_WINDOW_S = 2.5

    def __init__(self, sim: "Simulation"):
        self.sim = sim
        self.tasks: Dict[str, "TaskRuntime"] = {}
        self._counter = 0

    def publish(self, td: TaskDef):
        tr = TaskRuntime(td)
        tr._sim = self.sim
        self.tasks[td.id] = tr
        return tr

    def step(self):
        now = self.sim.now
        for tr in sorted(self.tasks.values(), key=lambda t: t.defn.id):
            if tr.state == "created" and now - tr.defn.created_at >= 0.3:
                tr.state = "advertised"
                tr.advertised_at = now
                self.sim.bus.send("task_ad", "FMS",
                                  {"task": tr.defn.to_dict(), "epoch": tr.epoch},
                                  now, ttl=self.BID_WINDOW_S + 2.0)
                self.sim.event("TASK", f"FMS advertised {tr.defn.id} (epoch {tr.epoch})")
            elif tr.state == "advertised":
                if now - tr.advertised_at >= self.BID_WINDOW_S:
                    # FMS never creates ambiguous ownership: award only when a
                    # bidder is actually free to take the job.
                    bids = [b for b in tr.bids if b["epoch"] == tr.epoch]
                    busy = {rid for rid in self.sim.brains
                            if self.sim.brains[rid].task is not None or self.sim.brains[rid].phase != "idle"}
                    if any(b["robot"] not in busy for b in bids):
                        self._award(tr)
                    elif now - tr.advertised_at >= self.BID_WINDOW_S + 60:
                        self._award(tr)   # starvation guard: force-award anyway
                    else:
                        tr.advertised_at = now - self.BID_WINDOW_S + 1.0  # retry soon
            else:
                owner = tr.state_owner()
                if owner and owner.phase in ("to_pickup", "at_pickup"):
                    tr.state = "transit_pickup"
                elif owner and owner.carrying:
                    tr.state = "carry"
                elif owner and owner.phase == "to_drop":
                    tr.state = "carry" if owner.carrying else "executing"

    def _award(self, tr: "TaskRuntime"):
        busy = {rid for rid in self.sim.brains
                if self.sim.brains[rid].task is not None or self.sim.brains[rid].phase != "idle"}
        bids = [b for b in tr.bids if b["epoch"] == tr.epoch and b["robot"] not in busy]
        if not bids:
            # no feasible bidder -> re-advertise later (deterministic backoff)
            tr.state = "created"
            tr.defn.created_at = self.sim.now
            tr.retry = getattr(tr, "retry", 0) + 1
            self.sim.event("TASK", f"{tr.defn.id}: no valid bids, re-advertising")
            return
        # deterministic: lowest score wins; tie-break by robot id
        bids.sort(key=lambda b: (b["score"], b["robot"]))
        winner = None
        for b in bids:                      # first still-free bidder wins (deterministic)
            br = self.sim.brains.get(b["robot"])
            if br is not None and br.task is None and br.phase == "idle":
                winner = b
                break
        if winner is None:                  # every bidder went busy -> retry soon
            tr.state = "advertised"
            tr.advertised_at = self.sim.now - self.BID_WINDOW_S + 2.0
            return
        best = winner
        tr.owner = best["robot"]
        tr.state = "assigned"
        tr.assigned_at = self.sim.now
        note = "lowest composite bid (eta+priority+energy/cap)"
        tr.award_note = f"{best['robot']} score={best['score']:.1f} eta={best['eta_s']}s :: {note}"
        self.sim.bus.send("task_award", "FMS",
                          {"task": tr.defn.id, "epoch": tr.epoch, "winner": best["robot"],
                           "task_obj": tr.defn.to_dict()},
                          self.sim.now, epoch=tr.epoch)
        self.sim.event("TASK", f"{tr.defn.id} AWARDED to {best['robot']} ({tr.award_note})")
        tr.state = "assigned"   # single unambiguous owner until completion or release

    def on_bid(self, m: Message):
        tr = self.tasks.get(m.payload["task"])
        if tr is None or tr.epoch != m.payload.get("epoch", tr.epoch):
            return  # stale-epoch bid rejected
        if tr.state not in ("advertised", "rebid") or tr.owner is not None:
            return
        b = dict(m.payload)
        b["robot"] = m.sender
        br = self.sim.brains.get(m.sender)
        b["free"] = bool(br and br.task is None and br.phase == "idle")
        tr.bids.append(b)

    def on_done(self, m: Message):
        tr = self.tasks.get(m.payload["task"])
        if tr is None or tr.epoch != m.payload.get("epoch"):
            return
        if tr.owner != m.sender:
            return
        tr.state = "completed"
        tr.completed_at = self.sim.now
        self.sim.event("TASK", f"{tr.defn.id} COMPLETED by {m.sender}")

    def release_robot_leases(self, rid: str):
        for zid in list(self.sim.registry.leases.keys()):
            self.sim.registry.leases[zid].pop(rid, None)

    def on_release(self, m: Message):
        tr = self.tasks.get(m.payload["task"])
        if tr is None or tr.epoch != m.payload.get("epoch"):
            return
        if tr.owner != m.sender:
            return
        tr.owner = None
        tr.epoch += 1
        tr.state = "created"
        tr.bids = []
        tr.defn.created_at = self.sim.now
        tr.released_reason = m.payload["reason"]
        self.sim.event("TASK", f"{tr.defn.id} RELEASED by {m.sender}: {m.payload['reason']} -> rebid (epoch {tr.epoch})")


class TaskRuntime:
    def __init__(self, defn: TaskDef):
        self.defn = defn
        self._sim = None
        self.state = "created"
        self.epoch = 0
        self.version = 0
        self.owner: Optional[str] = None
        self.bids: List[dict] = []
        self.advertised_at = 0.0
        self.assigned_at = 0.0
        self.completed_at: Optional[float] = None
        self.released_reason: Optional[str] = None
        self.award_note: Optional[str] = None

    def state_owner(self):
        sim = self._sim
        if sim and self.owner:
            b = sim.brains.get(self.owner)
            if b and b.task and b.task.id == self.defn.id and b.task_epoch == self.epoch:
                return b
        return None

    def to_dict(self, sim: "Simulation") -> dict:
        d = self.defn.to_dict()
        owner = self.state_owner()
        owner_id = self.owner if owner else None
        phase = owner.phase if owner else None
        st = self.state
        if owner:
            if phase in ("to_pickup", "at_pickup"):
                st = "executing"
            elif phase in ("to_drop", "at_drop"):
                st = "carrying"
        d.update({"task_state": st, "epoch": self.epoch, "owner": owner_id,
                  "phase": phase, "assigned_at": round(self.assigned_at, 1),
                  "completed_at": round(self.completed_at, 1) if self.completed_at else None,
                  "released_reason": self.released_reason, "award_note": self.award_note,
                  "bids": [dict(b) for b in self.bids[-8:]]})
        return d


class Simulation:
    def __init__(self, seed: int = 42, warehouse_spec: Optional[dict] = None,
                 fleet_specs: Optional[List[dict]] = None, coordination: str = "edgefleet"):
        self.seed = seed
        self.rng = random.Random(seed)
        self.wh = Warehouse(warehouse_spec or DEFAULT_WAREHOUSE)
        self.registry = ChokeRegistry()
        self.bus = SimBus(random.Random(seed ^ 0x5EED))
        self.coordination = coordination      # edgefleet | baseline
        self.now = 0.0
        self.step_index = 0
        self.robots: Dict[str, RobotPhys] = {}
        self.brains: Dict[str, Brain] = {}
        self.fms = FMS(self)
        self.events: List[dict] = []
        self.msg_log: List[dict] = []
        self.metrics = Metrics()
        self.paused = True
        self.finished = False
        self.scenario_name = "-"
        self.replay_frames: List[dict] = []
        self.replay_meta = {"seed": seed, "coordination": coordination,
                            "warehouse": self.wh.to_dict()}
        specs = fleet_specs or self.default_fleet()
        homes = sorted((st for st in self.wh.stations.values() if st.kind == "home"),
                       key=lambda st: st.id)
        for i, sd in enumerate(specs):
            sp = RobotSpec(**sd)
            if i < len(homes):
                hx, hy = self.wh.cell_to_world(*homes[i].cell)
                th = 0.0
            else:
                hx, hy = 1.5, 3.0 + i * 2.0
                th = 0.0
            self._spawn(sp, Pose(hx, hy, th))
        self.bus.location_cb = self._node_pos

    @staticmethod
    def default_fleet() -> List[dict]:
        return [
            dict(id="AMR-01", length=1.2, width=0.8, max_speed=1.6, cruise_speed=1.3,
                 accel=0.9, decel=1.1, max_decel=1.7, max_omega=1.6, payload_kg=350,
                 capacity_wh=900, reserve_wh=120, loc_sigma_xy=0.03, color="#2f6fb2"),
            dict(id="AMR-02", length=1.5, width=1.0, max_speed=1.2, cruise_speed=1.0,
                 accel=0.7, decel=0.9, max_decel=1.2, max_omega=1.1, payload_kg=600,
                 capacity_wh=1200, reserve_wh=160, loc_sigma_xy=0.05, color="#c26d3b"),
            dict(id="AMR-03", length=0.9, width=0.6, max_speed=2.0, cruise_speed=1.6,
                 accel=1.2, decel=1.4, max_decel=2.2, max_omega=2.2, payload_kg=150,
                 capacity_wh=600, reserve_wh=90, loc_sigma_xy=0.02, color="#3fa06a"),
            dict(id="AMR-04", length=1.3, width=0.9, max_speed=1.4, cruise_speed=1.15,
                 accel=0.8, decel=1.0, max_decel=1.5, max_omega=1.3, payload_kg=450,
                 capacity_wh=1000, reserve_wh=140, loc_sigma_xy=0.04, color="#8a5bb8"),
        ]

    def _spawn(self, spec: RobotSpec, spawn_pose: Pose):
        rp = RobotPhys(spec=spec, pose=spawn_pose,
                       energy_wh=spec.capacity_wh * self.rng.uniform(0.55, 1.0),
                       rng=random.Random(hash(spec.id) % (2 ** 31)))
        self.robots[spec.id] = rp
        phys_api = {
            "time": lambda: self.now,
            "get_pose": lambda: rp.pose,
            "get_speed": lambda: rp.v,
            "get_energy": lambda: rp.energy_wh,
            "is_charging": lambda: rp.charging,
            "net_ok": lambda: rp.net_ok,
            "set_velocity": (lambda v, w: self._set_vel(spec.id, v, w)),
            "sensors": lambda rid=spec.id: self._sense(rid),
            "rng": rp.rng,
        }
        if self.coordination == "baseline":
            from .baseline import BaselineBrain
            brain = BaselineBrain(spec, self.wh, self.registry, self.bus, phys_api)
        else:
            brain = Brain(spec, self.wh, self.registry, self.bus, phys_api)
        self.brains[spec.id] = brain
        # bus subscription for FMS topics
        self.bus.subscribe("FMS", "task_bid", self.fms.on_bid)
        self.bus.subscribe("FMS", "task_release", self.fms.on_release)
        self.bus.subscribe("FMS", "task_done", self.fms.on_done)
        # record message traffic for observability
        self._hook_msg_logging()

    def _hook_msg_logging(self):
        orig_send = self.bus.send
        sim = self

        def send(topic, sender, payload, now, target=None, epoch=0, ttl=10.0, scope_xy=None):
            m = orig_send(topic, sender, payload, now, target=target, epoch=epoch, ttl=ttl, scope_xy=scope_xy)
            if m is not None and topic not in ("lease",):
                sim.msg_log.append({"t": round(now, 2), "topic": topic, "from": sender,
                                    "to": target or "*", "brief": _brief(topic, payload)})
                if len(sim.msg_log) > 500:
                    del sim.msg_log[:250]
            return m
        self.bus.send = send

    def _set_vel(self, rid: str, v: float, w: float):
        rp = self.robots[rid]
        if rp.energy_wh <= 0:
            v = w = 0.0
        rp.v_cmd, rp.w_cmd = v, w

    def _sense(self, rid: str) -> List[dict]:
        me = self.robots[rid]
        out = []
        for oid, other in sorted(self.robots.items()):
            if oid == rid:
                continue
            d = me.pose.dist(other.pose)
            if d <= me.spec.sensor_range:
                out.append({"id": oid, "x": round(other.pose.x, 2), "y": round(other.pose.y, 2),
                            "th": round(other.pose.th, 2), "v": round(other.v, 2)})
        return out

    def _node_pos(self, node: str):
        rp = self.robots.get(node)
        return (rp.pose.x, rp.pose.y) if rp else None

    # ------------------------------------------------------------- events
    def event(self, kind: str, msg: str, robot: str = ""):
        e = {"t": round(self.now, 2), "kind": kind, "msg": msg, "robot": robot}
        self.events.append(e)
        if len(self.events) > 800:
            del self.events[:400]

    # --------------------------------------------------------------- step
    def step_once(self):
        if self.finished:
            return
        self._post_step_checks()
        self.now = round(self.now + DT, 6)
        self.step_index += 1
        # network model per-robot link status
        for rid, rp in self.robots.items():
            blocked = False
            for (x0, y0, x1, y1) in self.bus.dead_zones:
                if x0 <= rp.pose.x <= x1 and y0 <= rp.pose.y <= y1:
                    blocked = True
            if self.bus.partition_x is not None:
                pass  # handled at delivery; keep link flag for dead zones only
            rp.net_ok = not blocked
        # brains decide (deterministic order by id)
        for rid in sorted(self.brains.keys()):
            try:
                self.brains[rid].cycle()
            except Exception as ex:  # noqa: BLE001 — surface but don't kill sim
                if self.step_index % 50 == 1:
                    self.event("ERROR", f"{rid} brain exception: {ex}")
        # physics integrate
        self._physics_step()
        # comms deliver
        self.bus.step(self.now)
        # FMS
        self.fms.step()
        # metrics
        self._metrics_step()
        # scenario script hooks
        if self.scenario:
            self.scenario.tick(self)
        # replay frame every 10 steps (1 Hz)
        if self.step_index % 10 == 0:
            self._capture_frame()

    def _physics_step(self):
        ids = sorted(self.robots.keys())
        for rid in ids:
            rp = self.robots[rid]
            spec = rp.spec
            # differential-drive approximation with accel limits
            dv = clamp(rp.v_cmd - rp.v, -spec.max_decel * DT, spec.accel * DT)
            rp.v = clamp(max(0.0, rp.v + dv), 0.0, spec.max_speed)
            dw = clamp(rp.w_cmd - rp.w, -spec.max_omega * 4 * DT, spec.max_omega * 4 * DT)
            rp.w = clamp(dw + rp.w, -spec.max_omega, spec.max_omega)
            if rp.v < 1e-4:
                rp.w *= 0.9
            nx = rp.pose.x + rp.v * math.cos(rp.pose.th) * DT
            ny = rp.pose.y + rp.v * math.sin(rp.pose.th) * DT
            nth = rp.pose.th + rp.w * DT
            # static collision: reject motion that drives footprint into rack
            rad = min(spec.width / 2.0, spec.length / 2.0) * 0.95
            half = spec.width / 2.0
            ok_full = (self.wh.free_disk(nx, ny, half))
            ok_x = self.wh.free_disk(nx, rp.pose.y, half)
            ok_y = self.wh.free_disk(rp.pose.x, ny, half)
            if ok_full:
                rp.pose = Pose(nx, ny, nth)
            elif ok_x and abs(math.sin(rp.pose.th)) < 0.6:
                rp.pose = Pose(nx, rp.pose.y, nth)
            elif ok_y and abs(math.cos(rp.pose.th)) < 0.6:
                rp.pose = Pose(rp.pose.x, ny, nth)
            else:
                rp.v = 0.0
        # robot-robot interaction: physical safety envelope (last-resort)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = self.robots[ids[i]], self.robots[ids[j]]
                gap = math.hypot(a.pose.x - b.pose.x, a.pose.y - b.pose.y)
                minsep = (a.spec.length + b.spec.length) / 2.0 * 0.55 + \
                         max(a.spec.width, b.spec.width) / 2.0 * 0.55
                if gap < minsep:
                    if not self.metrics._collision_episode:
                        self.metrics.collisions += 1
                        self.metrics._collision_episode = True
                    self.event("COLLISION", f"physical contact {a.spec.id} <-> {b.spec.id} "
                                            f"(gap {gap:.2f}m < {minsep:.2f}m)")
                    # push apart deterministically along connecting line
                    ang = math.atan2(b.pose.y - a.pose.y, b.pose.x - a.pose.x)
                    push = (minsep - gap) / 2.0 + 0.01
                    ac = self.wh.nearest_free_cell(a.pose.x - math.cos(ang) * push,
                                                   a.pose.y - math.sin(ang) * push,
                                                   a.spec.width / 2.0)
                    bc = self.wh.nearest_free_cell(b.pose.x + math.cos(ang) * push,
                                                   b.pose.y + math.sin(ang) * push,
                                                   b.spec.width / 2.0)
                    awx, awy = self.wh.cell_to_world(*ac)
                    bwx, bwy = self.wh.cell_to_world(*bc)
                    a.pose = Pose(awx, awy, a.pose.th)
                    b.pose = Pose(bwx, bwy, b.pose.th)
                    a.v = b.v = 0.0
                else:
                    pass
        if all(math.hypot(self.robots[ids[i]].pose.x - self.robots[ids[j]].pose.x,
                          self.robots[ids[i]].pose.y - self.robots[ids[j]].pose.y) >= 
               (self.robots[ids[i]].spec.length + self.robots[ids[j]].spec.length)/2.0*0.55 + 
               max(self.robots[ids[i]].spec.width, self.robots[ids[j]].spec.width)/2.0*0.55
               for i in range(len(ids)) for j in range(i+1, len(ids))):
            self.metrics._collision_episode = False
        # battery + charging
        for rid, rp in self.robots.items():
            near_charge = any(st.kind == "charge" and
                              math.hypot(self.wh.cell_to_world(*st.cell)[0] - rp.pose.x,
                                         self.wh.cell_to_world(*st.cell)[1] - rp.pose.y) < 1.0
                              for st in self.wh.stations.values())
            rp.charging = near_charge and rp.v < 0.05 and rp.energy_wh < rp.spec.capacity_wh
            if rp.charging:
                CHARGE_POWER_W = 900.0   # fast-charging dock model (documented sim assumption)
                rp.energy_wh = min(rp.spec.capacity_wh, rp.energy_wh + CHARGE_POWER_W * DT / 3600.0)
            else:
                p = rp.spec.energy_rate_w(rp.v, rp.loaded_kg > 0 or
                                          (rid in self.brains and self.brains[rid].carrying))
                de = p * DT / 3600.0
                rp.energy_wh = max(0.0, rp.energy_wh - de)
                rp.total_energy_used_wh += de

    def _metrics_step(self):
        m = self.metrics
        m.sim_time = self.now
        moving = waiting = 0
        for rid, rp in self.robots.items():
            b = self.brains[rid]
            if rp.v > 0.05:
                moving += 1
                m.moving_robot_seconds += DT
            elif b.phase not in ("idle", "charging", "at_pickup", "at_drop"):
                waiting += 1
                m.waiting_robot_seconds += DT
                if b.last_decision in ("CHOKE_HOLD",):
                    m.choke_wait_seconds += DT
                if b.last_decision in ("SAFETY_STOP", "TEMPORAL_OFFSET"):
                    m.yield_seconds += DT
            if b.conflict_last:
                m.conflict_predictions += 1
            if b.last_decision == "DEADLOCK_ESCAPE":
                pass
        # near-conflict detection ground-truth check (predictive validation)
        ids = sorted(self.robots.keys())
        now_near = set()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = self.robots[ids[i]], self.robots[ids[j]]
                gap = math.hypot(a.pose.x - b.pose.x, a.pose.y - b.pose.y)
                if gap < 1.1:
                    now_near.add((ids[i], ids[j]))
        for pr in now_near - m._near_pairs:
            m.near_episodes += 1
        m._near_pairs = now_near
        # deadlock detection (global observer for METRICS ONLY — robots do not
        # use this; they detect cycles in their own brains)
        edges = {rid: self.brains[rid].waiting_for for rid in sorted(self.brains)
                 if self.brains[rid].waiting_for}
        cyc = _find_cycle(edges)
        if cyc:
            key = tuple(sorted(cyc))
            if key != m._last_deadlock_key:
                m.deadlocks_detected += 1
                m._last_deadlock_key = key
                self.event("DEADLOCK", f"dependency cycle {list(cyc)} detected (observer)")
        else:
            m._last_deadlock_key = None
        # choke occupancy
        occ = {}
        for z in self.wh.choke_zones():
            occ[z.id] = sum(1 for rp in self.robots.values()
                            if z.rect.contains(rp.pose.x, rp.pose.y))
            if occ[z.id] > z.capacity:
                m.choke_overcapacity_seconds += DT
        m.choke_occupancy = occ
        m.tasks_completed = sum(1 for tr in self.fms.tasks.values() if tr.state == "completed")
        m.tasks_total = len(self.fms.tasks)
        m.takeovers = sum(1 for tr in self.fms.tasks.values() if tr.epoch > 0)
        m.replans = sum(b.plan_replans for b in self.brains.values())
        m.unnecessary_stops = m.near_events  # conservative proxy
        m.messages_delivered = self.bus.delivered_count
        m.messages_dropped = self.bus.dropped_count

    # ------------------------------------------------------------ snapshot
    def snapshot(self) -> dict:
        robots = []
        for rid in sorted(self.robots):
            rp = self.robots[rid]
            b = self.brains[rid]
            snap = b.snapshot()
            snap["true"] = {"x": round(rp.pose.x, 2), "y": round(rp.pose.y, 2),
                            "th": round(rp.pose.th, 2), "v": round(rp.v, 2),
                            "v_cmd": round(rp.v_cmd, 2)}
            snap["energy_pct"] = round(100.0 * rp.energy_wh / rp.spec.capacity_wh, 1)
            robots.append(snap)
        return {
            "t": round(self.now, 2), "step": self.step_index, "paused": self.paused,
            "finished": self.finished, "scenario": self.scenario_name,
            "coordination": self.coordination,
            "network": {"state": self.bus.net_state, "delivered": self.bus.delivered_count,
                        "dropped": self.bus.dropped_count, "pending": self.bus.pending_count()},
            "robots": robots,
            "tasks": [tr.to_dict(self) for tr in sorted(self.fms.tasks.values(), key=lambda t: t.defn.id)],
            "leases": self.registry.snapshot(self.now),
            "events": self.events[-40:],
            "messages": self.msg_log[-60:],
            "metrics": self.metrics.to_dict(),
        }

    def _capture_frame(self):
        s = self.snapshot()
        slim = {
            "t": s["t"],
            "robots": [{k: r[k] for k in ("id", "true", "est", "phase", "decision", "task",
                                          "energy_wh", "conflict", "ueac", "traj")}
                       for r in s["robots"]],
            "tasks": s["tasks"], "leases": s["leases"], "network": s["network"],
            "metrics": s["metrics"],
        }
        self.replay_frames.append(slim)
        if len(self.replay_frames) > 7200:
            self.replay_frames = self.replay_frames[-6000:]

    # -------------------------------------------------------------- control
    def run_steps(self, n: int):
        for _ in range(n):
            self.step_once()

    def run_headless(self, seconds: float, progress_cb=None):
        n = int(seconds / DT)
        for i in range(n):
            self.step_once()
            if progress_cb and i % 100 == 0:
                progress_cb(i / n)
        return self.metrics.to_dict()

    scenario: Optional[object] = None

    def _post_step_checks(self):
        pass

    def attach_scenario(self, sc):
        self.scenario = sc
        self.scenario_name = sc.id

    def set_network(self, **kw):
        self.bus.set_network(kw.pop("state", "NORMAL"), **kw)


def _find_cycle(edges: Dict[str, str]):
    """Deterministic cycle search over wait-for graph (observer/metrics use)."""
    color = {}
    stack = []

    def dfs(u):
        color[u] = 1
        stack.append(u)
        v = edges.get(u)
        if v is not None:
            c = color.get(v, 0)
            if c == 0:
                r = dfs(v)
                if r:
                    return r
            elif c == 1:
                idx = stack.index(v)
                return list(stack[idx:])
        stack.pop()
        color[u] = 2
        return None

    for u in sorted(edges.keys()):
        if color.get(u, 0) == 0:
            c = dfs(u)
            if c:
                return c
    return None


def _brief(topic: str, p: dict) -> str:
    try:
        if topic == "intent":
            return f"pos({p['x']},{p['y']}) traj:{len(p.get('traj', []))}pts dest={p.get('dest')}"
        if topic == "proposal":
            return f"{p['action']} prio={p['priority'][:2]} conflict_t={p.get('conflict_t')}"
        if topic == "response":
            return f"{p['result']} seq={p['seq']}"
        if topic == "task_ad":
            return f"{p['task']['id']} {p['task']['pickup_zone']}->{p['task']['drop_zone']}"
        if topic == "task_bid":
            return f"{p['task']} score={p['score']} eta={p['eta_s']}s"
        if topic == "task_award":
            return f"{p['task']} -> {p['winner']}"
        if topic == "task_release":
            return f"{p['task']}: {p['reason'][:60]}"
        if topic == "task_done":
            return f"{p['task']}"
        if topic == "dependency":
            wf = p.get("waiting_for")
            return f"waits_for={wf}" if wf else "clear"
        if topic == "net_status":
            return p.get("state", "?")
    except Exception:
        pass
    return ""


class Metrics:
    def __init__(self):
        self.sim_time = 0.0
        self.collisions = 0
        self._collision_episode = False
        self.near_events = 0
        self._near_pairs = set()
        self.near_episodes = 0
        self.conflict_predictions = 0
        self.moving_robot_seconds = 0.0
        self.waiting_robot_seconds = 0.0
        self.choke_wait_seconds = 0.0
        self.yield_seconds = 0.0
        self.deadlocks_detected = 0
        self.choke_overcapacity_seconds = 0.0
        self.choke_occupancy: Dict[str, int] = {}
        self.tasks_completed = 0
        self.tasks_total = 0
        self.takeovers = 0
        self.replans = 0
        self.unnecessary_stops = 0
        self.messages_delivered = 0
        self.messages_dropped = 0
        self._last_deadlock_key = None

    def to_dict(self) -> dict:
        tot = self.moving_robot_seconds + self.waiting_robot_seconds
        return {
            "sim_time": round(self.sim_time, 1),
            "collisions": self.collisions,
            "near_conflicts_groundtruth": self.near_episodes,
            "conflict_predictions": self.conflict_predictions // 10,  # ticks -> ~seconds
            "moving_s": round(self.moving_robot_seconds, 1),
            "waiting_s": round(self.waiting_robot_seconds, 1),
            "choke_wait_s": round(self.choke_wait_seconds, 1),
            "yield_s": round(self.yield_seconds, 1),
            "wait_fraction": round(self.waiting_robot_seconds / tot, 3) if tot > 1 else 0.0,
            "deadlocks_detected": self.deadlocks_detected,
            "choke_overcapacity_s": round(self.choke_overcapacity_seconds, 1),
            "choke_occupancy": self.choke_occupancy,
            "tasks_completed": self.tasks_completed,
            "tasks_total": self.tasks_total,
            "takeovers": self.takeovers,
            "replans": self.replans,
            "messages_delivered": self.messages_delivered,
            "messages_dropped": self.messages_dropped,
        }
