"""Wireless communication abstraction (publish / subscribe).

Design goals
------------
* The AMR brain talks to a `CommBus` interface only.  Today the transport is
  the deterministic in-process `SimBus`; swapping in MQTT later means writing
  one new class that implements `send()`/`deliver()` — no brain changes.
* Every message carries: id, sender, topic, epoch, sim timestamp, ttl and a
  spatial scope.  Receivers reject stale/duplicate/out-of-epoch messages.
* Network model is simulated: NORMAL / DEGRADED (drop + latency) /
  DEAD ZONE (robots inside a rectangle are isolated) / PARTITION (two halves
  cannot hear each other).  Recovery restores normal delivery.
* Delivery order is deterministic: per-receiver queues are sorted by
  (delivery_time, msg_seq).

Topics used by the system:
    intent          robot -> peers            planned trajectory summary
    proposal        robot -> robot            negotiation offer (yield/slow/offset/reroute)
    response        robot -> robot            accept/reject of proposal
    lease           robot -> zone-subscribers choke lease claim/release
    dependency      robot -> peers            "I am waiting for X" (deadlock graph feed)
    task_ad         FMS   -> fleet            advertised task
    task_bid        robot -> FMS              bid with cost estimate
    task_award      FMS   -> fleet            deterministic award
    task_release    robot -> FMS+fleet        owner can no longer complete
    net_status      bus   -> all              network state change notice
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class Message:
    id: int
    seq: int
    topic: str
    sender: str
    payload: dict
    t_sent: float
    epoch: int = 0
    ttl: float = 10.0
    target: Optional[str] = None          # None => broadcast to subscribers
    scope_xy: Optional[Tuple[float, float]] = None  # spatial scoping centre

    def to_dict(self) -> dict:
        return {"id": self.id, "topic": self.topic, "sender": self.sender,
                "target": self.target, "t": round(self.t_sent, 2), "epoch": self.epoch,
                "payload": self.payload}


class SimBus:
    """Deterministic pub/sub bus with a configurable impairment model."""

    def __init__(self, rng):
        self._rng = rng                    # seeded random.Random for determinism
        self._subscribers: Dict[str, Dict[str, Callable[[Message], None]]] = {}
        self._pending: List[Tuple[float, int, str, Message]] = []
        self._msg_id = 0
        self.net_state = "NORMAL"          # NORMAL|DEGRADED|PARTITION|RECOVERY
        self.drop_prob = 0.0
        self.latency_s = 0.0               # base latency (sim seconds)
        self.dead_zones: List[Tuple[float, float, float, float]] = []  # x0,y0,x1,y1
        self.partition_x: Optional[float] = None
        self.delivered_count = 0
        self.dropped_count = 0

    # ---------------- subscription ----------------
    def subscribe(self, node: str, topic: str, cb: Callable[[Message], None]):
        self._subscribers.setdefault(topic, {})[node] = cb

    def unsubscribe(self, node: str, topic: str):
        if topic in self._subscribers:
            self._subscribers[topic].pop(node, None)

    # ---------------- network model control ----------------
    def set_network(self, state: str, dead_zones=None, partition_x=None, drop_prob=0.0, latency=0.0):
        self.net_state = state
        self.dead_zones = [tuple(z) for z in (dead_zones or [])]
        self.partition_x = partition_x
        self.drop_prob = drop_prob if state == "DEGRADED" else 0.0
        self.latency_s = latency if state == "DEGRADED" else 0.0

    def _link_ok(self, sx: float, sy: float, rx: float, ry: float) -> bool:
        for (x0, y0, x1, y1) in self.dead_zones:
            if x0 <= sx <= x1 and y0 <= sy <= y1:
                return False
            if x0 <= rx <= x1 and y0 <= ry <= y1:
                return False
        if self.partition_x is not None:
            if (sx < self.partition_x) != (rx < self.partition_x):
                return False
        return True

    # ---------------- send / step ----------------
    def send(self, topic: str, sender: str, payload: dict, now: float,
             target: Optional[str] = None, epoch: int = 0, ttl: float = 10.0,
             scope_xy: Optional[Tuple[float, float]] = None) -> Optional[Message]:
        self._msg_id += 1
        m = Message(id=self._msg_id, seq=self._msg_id, topic=topic, sender=sender,
                    payload=payload, t_sent=now, epoch=epoch, ttl=ttl, target=target,
                    scope_xy=scope_xy)
        return m if self._dispatch(m, now) else None

    def _pos_of(self, node: str, m: Message) -> Optional[Tuple[float, float]]:
        # senders always attach their pose; receivers' poses come from location_cb
        return m.scope_xy

    def _dispatch(self, m: Message, now: float) -> bool:
        nodes = ([m.target] if m.target else list(self._subscribers.get(m.topic, {}).keys()))
        any_link = False
        for node in nodes:
            if node == m.sender:
                continue
            cb = self._subscribers.get(m.topic, {}).get(node)
            if cb is None:
                continue
            rpos = self.location_cb(node) if self.location_cb else None
            spos = m.scope_xy
            if spos is None or rpos is None:
                lat = 0.05
                link = True
            else:
                link = self._link_ok(spos[0], spos[1], rpos[0], rpos[1])
                dist = math.hypot(spos[0] - rpos[0], spos[1] - rpos[1])
                lat = max(0.02, 0.02 + dist / 3e8 * 1000)  # radio propagation, scaled for sim
                if self.net_state == "DEGRADED":
                    lat += self.latency_s
                    if self._rng.random() < self.drop_prob:
                        self.dropped_count += 1
                        continue
            if not link:
                self.dropped_count += 1
                continue
            any_link = True
            self.delivered_count += 1
            self._pending.append((now + lat, m.seq, node, m))
        self._pending.sort(key=lambda p: (p[0], p[1]))
        return any_link

    location_cb: Optional[Callable[[str], Optional[Tuple[float, float]]]] = None

    def step(self, now: float):
        ready = [p for p in self._pending if p[0] <= now]
        self._pending = [p for p in self._pending if p[0] > now]
        for _, _, node, m in ready:
            cb = self._subscribers.get(m.topic, {}).get(node)
            if cb:
                cb(m)

    def pending_count(self) -> int:
        return len(self._pending)
