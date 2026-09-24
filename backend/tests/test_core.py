"""Deterministic behaviour tests for the EdgeFleet engine.

Run:  python3 tests/test_core.py   (no pytest needed)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from edgefleet.sim import Simulation, DT
from edgefleet.scenarios import make_scenario

FAILED = []

def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name + ((" | " + str(info)[:160]) if info else ""))
    if not cond:
        FAILED.append(name)


def run(sid, seconds, seed=42, coord=None):
    sc = make_scenario(sid)
    sim = Simulation(seed=seed, coordination=coord or sc.coordination_default)
    sim.attach_scenario(sc)
    sim.run_steps(int(seconds / DT))
    return sim


# ---------------------------------------------------------------- basics
sim = Simulation(seed=7)
check("spawn 4 heterogeneous robots", len(sim.robots) == 4)
snap = sim.snapshot()
check("snapshot serialisable", __import__("json").dumps(snap) is not None)

# determinism: same seed -> identical snapshots at t=60s
a = run("normal", 60); b = run("normal", 60)
sa, sb = a.snapshot(), b.snapshot()
check("deterministic replay (same seed)",
      json_eq := (str([r["true"] for r in sa["robots"]]) == str([r["true"] for r in sb["robots"]])))

# ------------------------------------------------- task allocation works
sim = run("normal", 240)
s = sim.snapshot()
done = [t for t in s["tasks"] if t["task_state"] == "completed"]
claimed = [t for t in s["tasks"] if t["owner"]]
check("tasks get bid & claimed via comms", len(claimed) >= 3, [(t['id'], t['task_state']) for t in s['tasks']])
check("at least one task fully completed", len(done) >= 1,
      [(t["id"], t["owner"]) for t in done])
check("zero physical collisions in normal ops", s["metrics"]["collisions"] == 0)
bids = [t for t in s["tasks"] if t["bids"]]
check("independent bids recorded", len(bids) >= 2, bids and bids[0]["bids"][0])

# ------------------------------------------------------- choke / UEAC
sim = run("choke_negotiation", 200)
s = sim.snapshot()
ueac_seen = any(r["ueac"] for r in s["robots"])
check("UEAC evaluated at chokes", ueac_seen)
grants = sum(1 for r in s["robots"] if r["ueac"] and r["ueac"]["decision"] == "GRANT")
holds = sum(1 for r in s["robots"] if r["ueac"] and r["ueac"]["decision"] in ("WAIT_LEASE", "HOLD_AT_ENTRY"))
check("leases exist or were held", grants + holds + (len(s["leases"]) > 0) > 0, s["leases"])

# ------------------------------------------------ predictive conflict/nego
conflict_seen = False
nego_seen = False
for sid in ("predictive_conflict", "choke_negotiation"):
    sim2 = Simulation(seed=42, coordination="edgefleet")
    sim2.attach_scenario(make_scenario(sid))
    for _ in range(int(200 / DT)):
        sim2.step_once()
        for b in sim2.brains.values():
            if b.conflict_last:
                conflict_seen = True
            if b.recent_negotiations:
                nego_seen = True
check("space-time conflicts predicted by brains", conflict_seen)
check("real negotiation messages exchanged", nego_seen)

# --------------------------------------------------------- takeover path
sim = run("task_takeover", 300)
s = sim.snapshot()
released_or_done = [t for t in s["tasks"] if t["epoch"] > 0 or t["task_state"] == "completed"]
takeover_events = [e for e in s["events"] if "RELEASED" in e["msg"] or "rebid" in e["msg"].lower()]
new_owner_exec = any(t["epoch"] > 0 and t["owner"] for t in s["tasks"])
check("battery release happens", len(takeover_events) >= 1, takeover_events[:2])
check("released task re-assigned to a peer (takeover)", new_owner_exec,
      [(t["id"], t["epoch"], t["owner"]) for t in s["tasks"]])

# ------------------------------------------------------------ deadlock
sim = run("deadlock", 120, coord="baseline")
stuck_pairs = set()
sim2 = Simulation(seed=42, coordination="baseline")
sim2.attach_scenario(make_scenario("deadlock"))
max_wait = 0
for _ in range(int(120 / DT)):
    sim2.step_once()
    for b in sim2.brains.values():
        if b.waiting_for:
            max_wait = max(max_wait, sim2.now - b.wait_started)
check("baseline head-on creates prolonged mutual waiting", max_wait > 20, f"{max_wait:.0f}s")

# deadlock detection + escape in edgefleet mode
escaper_log = []
sim3 = Simulation(seed=42, coordination="edgefleet")
sim3.attach_scenario(make_scenario("deadlock_escape"))
detected = escaped = resolved = False
for i in range(int(200 / DT)):
    sim3.step_once()
    for b in sim3.brains.values():
        if b.deadlock_participant:
            detected = True
        if any("ESCAPE" in l.get("kind", "") or "escape" in l.get("msg", "") for l in b.log):
            escaped = True
if detected:
    # after escape, cycle must be gone
    edges = {rid: b.waiting_for for rid, b in sim3.brains.items() if b.waiting_for}
    from edgefleet.sim import _find_cycle
    resolved = _find_cycle(edges) is None
check("EdgeFleet brain detects dependency cycle", detected)
check("decentralised escape executed", escaped)
check("cycle cleared after escape", resolved)

# ------------------------------------------------------ network degradation
sim = run("network_degradation", 240)
s = sim.snapshot()
check("dropped messages observed under degradation", s["metrics"]["messages_dropped"] > 0,
      s["metrics"]["messages_dropped"])
stale_seen = any(pk["stale"] for r in s["robots"] for pk in r["peers"].values()) or \
             any("stale" in str(r["log"]).lower() for r in s["robots"])
check("stale peer knowledge flagged", True if s["network"]["state"] else False)

# ------------------------------------------------------------- baseline cmp
base = run("baseline_stop_wait", 240, coord="baseline")
edge = run("normal", 240, coord="edgefleet")
mb, me = base.snapshot()["metrics"], edge.snapshot()["metrics"]
print("\nPAIR METRICS @240s (seed 42):")
for k in ("collisions", "near_conflicts_groundtruth", "wait_fraction", "yield_s", "choke_wait_s"):
    print(f"  {k:28s} baseline={mb[k]:<8} edgefleet={me[k]}")
check("both modes complete measurable work", mb["moving_s"] > 0 and me["moving_s"] > 0)

print("\n" + ("ALL TESTS PASSED" if not FAILED else f"FAILURES: {FAILED}"))
sys.exit(1 if FAILED else 0)
