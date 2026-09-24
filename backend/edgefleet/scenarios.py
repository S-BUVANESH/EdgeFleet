"""Configurable experiment scenarios.

A scenario is a deterministic script attached to the simulation: it injects
tasks at given sim times, forces network states, drains batteries, and can
place robots into conflict-prone positions.  Every scenario works with either
coordination mode (edgefleet / baseline) so experiments can be PAIRED with an
identical seed.

Scenarios never fabricate outcomes — they only create the *conditions*; the
behaviour (conflicts, deadlocks, releases, takeovers) must emerge from the
brains and is then measured by the metrics module.
"""
from __future__ import annotations

import math
from typing import List, Optional

from .types import TaskDef


class Scenario:
    id = "custom"
    label = "Custom"
    duration_s = 240.0
    coordination_default = "edgefleet"

    def __init__(self, **overrides):
        self.injected = []
        self.done = set()
        for k, v in overrides.items():
            setattr(self, k, v)

    def tasks(self, sim) -> List[TaskDef]:
        return []

    def events_at(self, sim) -> None:
        return None

    # called every sim step
    def tick(self, sim):
        if not self.injected:
            base = self.tasks(sim)
            for td in base:
                sim.fms.publish(td)
                sim.event("SCENARIO", f"publish {td.id}: {td.pickup_zone}->{td.drop_zone} "
                                      f"prio{td.priority} deadline {td.deadline_s:.0f}s")
            self.injected = True
        self.events_at(sim)


def _t(i, pu, dr, prio=5, deadline=300, payload=60, created=0.0, tid=None):
    td = TaskDef(id=tid or f"T{i:02d}", pickup_zone=pu, drop_zone=dr,
                 priority=prio, deadline_s=deadline, payload_kg=payload,
                 created_at=created)
    return td


class NormalScenario(Scenario):
    id = "normal"
    label = "NORMAL OPERATION — steady mixed workload"
    duration_s = 300.0

    def tasks(self, sim):
        return [
            _t(1, "P01", "D01", prio=5, deadline=240),
            _t(2, "P02", "D02", prio=4, deadline=240),
            _t(3, "P01", "D02", prio=6, deadline=300),
            _t(4, "P02", "D01", prio=3, deadline=280),
        ]


class PredictiveConflictScenario(Scenario):
    id = "predictive_conflict"
    label = "PREDICTIVE CONFLICT — mirrored cross-aisle missions"
    duration_s = 200.0

    def tasks(self, sim):
        # crossing flows through the central corridor & CH-E choke
        return [
            _t(1, "P01", "D02", prio=4, deadline=200),
            _t(2, "P02", "D01", prio=5, deadline=200),
            _t(3, "P01", "D01", prio=7, deadline=260),
            _t(4, "P02", "D02", prio=6, deadline=260),
        ]


class ChokeNegotiationScenario(Scenario):
    id = "choke_negotiation"
    label = "CHOKE NEGOTIATION — four robots, one single-lane choke (CH-N)"
    duration_s = 260.0

    def tasks(self, sim):
        # all routes funnel west->east through CH-N region via central corridor
        return [
            _t(1, "P01", "D01", prio=3, deadline=260),
            _t(2, "P02", "D02", prio=4, deadline=260),
            _t(3, "P01", "D02", prio=5, deadline=300),
            _t(4, "P02", "D01", prio=6, deadline=300),
        ]


class DeadlockScenario(Scenario):
    id = "deadlock"
    label = "DEADLOCK — head-on single-lane confrontation (no escape here)"
    duration_s = 220.0
    coordination_default = "baseline"   # shows deadlock FORMING

    def tasks(self, sim):
        return [
            _t(1, "P01", "D01", prio=2, deadline=200),
            _t(2, "P02", "D02", prio=2, deadline=200),
        ]

    def events_at(self, sim):
        # place two robots nose-to-nose inside the narrow north choke lane
        if self.step_marker(sim) == 1:
            r1 = sim.robots.get("AMR-01")
            r2 = sim.robots.get("AMR-03")
            if r1 and r2:
                z = sim.wh.zones.get("CH-N")
                if z:
                    y = z.rect.cy
                    r1.pose = r1.pose.__class__(z.rect.x0 - 0.4, y, 0.0)
                    r2.pose = r2.pose.__class__(z.rect.x1 + 0.4, y, math.pi)
                    sim.event("SCENARIO", "AMR-01 & AMR-03 positioned head-on at CH-N")

    @staticmethod
    def step_marker(sim):
        return 1 if sim.step_index == 2 else 0


class DeadlockEscapeScenario(Scenario):
    id = "deadlock_escape"
    label = "DEADLOCK ESCAPE — same confrontation, EdgeFleet brains resolve it"
    duration_s = 220.0
    coordination_default = "edgefleet"

    def tasks(self, sim):
        return [
            _t(1, "P01", "D01", prio=2, deadline=200),
            _t(2, "P02", "D02", prio=2, deadline=200),
        ]

    def events_at(self, sim):
        if sim.step_index == 2:
            r1 = sim.robots.get("AMR-01")
            r2 = sim.robots.get("AMR-03")
            if r1 and r2:
                z = sim.wh.zones.get("CH-N")
                if z:
                    y = z.rect.cy
                    r1.pose = type(r1.pose)(z.rect.x0 - 0.4, y, 0.0)
                    r2.pose = type(r2.pose)(z.rect.x1 + 0.4, y, math.pi)
                    sim.event("SCENARIO", "head-on placement at CH-N; watching for cycle + escape")


class LowBatteryScenario(Scenario):
    id = "low_battery"
    label = "LOW BATTERY — owner releases task mid-flight, peers rebid"
    duration_s = 300.0

    def tasks(self, sim):
        return [
            _t(1, "P01", "D02", prio=3, deadline=280, payload=80),
            _t(2, "P02", "D01", prio=5, deadline=280, payload=80),
        ]

    def events_at(self, sim):
        # drain AMR-02 (big hauler) once it holds a task, forcing release
        if sim.step_index == 60:
            rp = sim.robots.get("AMR-02")
            if rp:
                rp.energy_wh = rp.spec.reserve_wh + 40.0
                sim.event("SCENARIO", "AMR-02 battery drained below completion reserve")


class TaskTakeoverScenario(Scenario):
    id = "task_takeover"
    label = "TASK TAKEOVER — released work is re-bid and physically executed by a peer"
    duration_s = 320.0

    def tasks(self, sim):
        return [
            _t(1, "P01", "D01", prio=2, deadline=300, payload=100),
            _t(2, "P02", "D02", prio=4, deadline=300, payload=100),
        ]

    def events_at(self, sim):
        if sim.step_index == 120:
            # kill battery margin of whoever currently owns T01
            tr = sim.fms.tasks.get("T01")
            if tr and tr.owner:
                rp = sim.robots.get(tr.owner)
                if rp:
                    rp.energy_wh = rp.spec.reserve_wh + 25.0
                    sim.event("SCENARIO", f"{tr.owner} (owner of T01) drained -> release expected")


class NetworkDegradationScenario(Scenario):
    id = "network_degradation"
    label = "NETWORK DEGRADATION — packet loss, latency, dead zone, recovery"
    duration_s = 300.0

    def tasks(self, sim):
        return [
            _t(1, "P01", "D01", prio=4, deadline=280),
            _t(2, "P02", "D02", prio=4, deadline=280),
            _t(3, "P01", "D02", prio=6, deadline=320),
        ]

    def events_at(self, sim):
        t = sim.now
        stage = getattr(self, "_stage", set())

        def once(key):
            if key in stage:
                return False
            stage.add(key)
            return True

        if t > 30 and once("deg"):
            sim.bus.set_network("DEGRADED", drop_prob=0.25, latency=0.6)
            sim.bus.send("net_status", "NET", {"state": "DEGRADED"}, t)
            sim.event("NET", "radio degraded: 25% loss, +0.6s latency")
        if t > 90 and once("dz"):
            sim.bus.set_network("DEGRADED", dead_zones=[[22.0, 13.0, 28.0, 17.0]],
                                drop_prob=0.25, latency=0.6)
            sim.event("NET", "dead zone active around Z08/Z13 area")
        if t > 150 and once("part"):
            sim.bus.set_network("PARTITION", partition_x=20.0, dead_zones=[])
            sim.event("NET", "network partition along x=20m")
        if t > 210 and once("rec"):
            sim.bus.set_network("NORMAL")
            sim.bus.send("net_status", "NET", {"state": "NORMAL"}, t)
            sim.event("NET", "network recovered")
        self._stage = stage


class BaselineStopAndWaitScenario(NormalScenario):
    id = "baseline_stop_wait"
    label = "BASELINE STOP-AND-WAIT — same workload, reactive control"
    coordination_default = "baseline"


SCENARIOS = {s.id: s for s in [
    NormalScenario(), PredictiveConflictScenario(), ChokeNegotiationScenario(),
    DeadlockScenario(), DeadlockEscapeScenario(), LowBatteryScenario(),
    TaskTakeoverScenario(), NetworkDegradationScenario(), BaselineStopAndWaitScenario(),
]}


def make_scenario(sid: str) -> Scenario:
    proto = SCENARIOS[sid]
    return type(proto)()
