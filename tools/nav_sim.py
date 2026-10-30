"""Drive the navigation in a 2D house, many times over, and count what happens.

Laptop only. The world, the truck, the lidar and the map are in
tools/nav_world.py; this runs episodes on them and writes up the results.

  # one episode, with a picture
  python tools/nav_sim.py one --map <q5_hires_result.npz> --scenario goto --png out.png

  # a batch: random start/goal pairs and rove runs, every response level
  python tools/nav_sim.py batch --maps <dir or npz...> --episodes 400 --rove 12 \\
      --rove-minutes 5 --jobs 4 --out <dir>

Scenarios:

  goto        the house already mapped (as if driven round once), a goal
              somewhere else in it: a click on a map I can see
  goto_blank  the same with no map at all: the click is somewhere the truck
              has not been, and the walls appear as it drives
  rove        no map and no goal, for --rove-minutes: frontier exploration

The safety layer here is a stand-in for the real one (crawler/nav/driver.py
uses the real one): the forward floor is any solid cell in a box from the
bumper out to FORWARD_FLOOR_M off the lens, the probes' width, the rear stop
is the simulated lidar's rear sector, STUCK is a forward commanded for a second
that went nowhere, and the back-out reverses at lock at ESCAPE_SPEED until
the front opens, the rear stop, or its cap. The same order and the same
numbers as the car; the camera itself is not simulated.

What it reports, per episode (JSON lines) and summed per scenario and level:
success, time to goal, collisions (the truck's footprint entering a solid
cell: the move is refused and counted once per contact), back-outs and why,
the least clearance between the footprint and anything, plans and their
time, and for rove the floor area mapped by minute.

The house maps are read from wherever they are at run time and never copied
anywhere in the repo.
"""

import argparse
import glob
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy import ndimage

from crawler.autopilot import driver as dv
from crawler.nav import costmap as cmod
from crawler.nav import driver as nd
from crawler.nav import mapview as mv
from crawler.nav import navigator as nv
from crawler.nav import planner as pl
from crawler.nav import truck as tk

import nav_world as nw

DT = 0.02
DECIDE_EVERY = 5                       # 10 Hz, AUTO_TICK_INTERVAL
FORWARD_SPEED = dv.FORWARD_SPEED
FORWARD_FLOOR_M = dv.FORWARD_FLOOR_MM / 1000.0
PROBE_HALF_M = tk.FLOOR_PROBE_HALF_M    # the outermost probe at 400 mm
STUCK_S = 1.0
RECOVERY_COOLDOWN_S = 5.0
BACKOUT_MIN_S = dv.BACKOUT_MIN_S
BACKOUT_CAP_S = dv.REVERSE_SECONDS
BACKOUT_OPEN_M = dv.BACKOUT_OPEN_MM / 1000.0
SETTLE_S = dv.SETTLE_SECONDS

# How much slower the Pi plans than this laptop: a plan's answer reaches
# the simulated truck this many times its laptop time later (it stands still
# meanwhile, unless it has a path already), and the planning budget is cut
# by the same factor. ESTIMATED: 3-5x is the usual Pi 4 against an M-series
# core for numpy-heavy Python; 3 is the kind end.
PI_SLOWDOWN = 3.0

LEVELS = ("ideal", "measured", "harsh", "axle60")
SCENARIOS = ("goto", "goto_blank", "goto_wall", "rove")
# Scenarios on a house already mapped. goto_wall: the goal is a click inside
# something solid near the floor (a wall, furniture), and the right answer
# is the reachable point nearest it.
MAPPED = ("goto", "goto_wall")
# A moving decision whose front command is this far from the last moving
# one's: a jerk of the wheels, half the lock or more in a tenth of a second.
HARSH_STEER = 0.3
# The ESC's drag brake stops the truck in about 0.3 s; what moves after this
# much of a hold is the hold not holding.
HOLD_SETTLE_S = 0.5
# A closest hold is right when it ends no more than this further from the
# goal than the true closest reachable point is: the arrival tolerance, the
# AIM_CLEAR_M the aim keeps off a wall beyond the footprint's own, and 0.1 m
# for the measured pose error and a cost-map cell.
CLOSEST_OK_M = nv.ARRIVE_M + nv.AIM_CLEAR_M - (cmod.INSCRIBED_M + 0.02) + 0.1


def _probe_box(lens_m):
    """The box a camera distance `lens_m` reaches, from the bumper out. The
    probes measure off the lens (tk.LENS_M ahead of the pose), not the
    bumper."""
    xs = np.arange(tk.FRONT_M + 0.02, tk.LENS_M + lens_m, 0.025)
    ys = np.arange(-PROBE_HALF_M, PROBE_HALF_M + 1e-9, 0.025)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return gx.ravel(), gy.ravel()


_FLOOR_BOX = _probe_box(FORWARD_FLOOR_M)
_OPEN_BOX = _probe_box(BACKOUT_OPEN_M)


def _box_solid(world, pose, box):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    bx, by = box
    return bool(world.solid_at(x + c * bx - s * by, y + s * bx + c * by).any())


class Safety:
    """The stand-in safety layer and back-out. See the module docstring."""

    def __init__(self):
        self.backout_until = None
        self.backout_since = None
        self.settle_until = None
        self.side = -1.0
        self.open_count = 0
        self.decisions = 0
        self.fwd_since = None
        self.still_since = None
        self.last_recovery = -math.inf
        self.standoff_since = None
        self.last_front = 0.0
        self.causes = {"stuck": 0, "standoff": 0, "nav": 0}
        self.floor_holds = 0
        self.rear_stops = 0
        self.last_end = -math.inf
        self.level = 0

    def begin(self, now, cause):
        self.causes[cause] += 1
        self.level = self.level + 1 if now - self.last_end <= dv.ESCALATE_WINDOW_S else 0
        side = math.copysign(1.0, self.last_front) if abs(self.last_front) > 0.15 else -1.0
        if self.level >= 2:
            side = -self.side
        self.side = side
        cap = dv.ESCALATED_REVERSE_SECONDS if self.level >= 1 else BACKOUT_CAP_S
        self.backout_since = now
        self.backout_until = now + cap
        self.open_count = 0
        self.decisions = 0
        self.fwd_since = None
        self.last_recovery = now

    def recovering(self, now, world, pose, rear):
        """(throttle, front) while a back-out runs, else None. Sets
        self.handed_back on the decision it ends."""
        self.handed_back = False
        if self.settle_until is not None:
            if now < self.settle_until:
                return (0.0, 0.0)
            self.settle_until = None
            self.handed_back = True
            return None
        if self.backout_until is None:
            return None
        self.decisions += 1
        open_ = not _box_solid(world, pose, _OPEN_BOX)
        self.open_count = self.open_count + 1 if open_ else 0
        done = (now >= self.backout_until
                or (now - self.backout_since >= BACKOUT_MIN_S
                    and self.open_count >= dv.BACKOUT_OPEN_DECISIONS))
        if rear is not None and rear < dv.REAR_STOP_CLEAR_MM:
            self.rear_stops += 1
            done = True
        if done:
            self.backout_until = None
            self.settle_until = now + SETTLE_S
            self.last_end = now
            return (0.0, 0.0)
        thr = dv.ESCAPE_SPEED * min(1.0, self.decisions / dv.ESCAPE_RAMP_DECISIONS)
        return (thr, self.side * tk.MAX_TURN)


def run_episode(world, scenario, start, goal=None, level="measured", seed=0,
                max_s=None, rove_s=300.0, params=nv.Params(), trace=False,
                response=None, extra_goals=(), lost=None, hold_s=0.0,
                backout=False, expect=None, slowdown=None):
    """One episode. A goto ends once the navigator has held (arrived, at
    the closest point, or given up) for `hold_s`; `extra_goals` are clicked
    one at a time after each hold, the next only once the last is held.
    `lost` (t0, t1) is a SLAM that is lost over that window: no view and
    nothing written. `backout` is the old safety layer (the reverse at lock
    on STUCK and a floor standoff); off, as on the car, they mark the spot
    and plan again. `expect` is the point a closest hold should be at, for
    closest_err_m."""
    rng = np.random.default_rng(seed)
    resp = response if response is not None else nw.Response.level(level)
    truck = nw.Truck(world, start, resp, rng, DT)
    lidar = nw.Lidar(world, rng)
    smap = nw.SimMap(world.size_m)
    if scenario in MAPPED:
        smap.fill_from(world)
    perr = nw.PoseError(rng)
    local = mv.LocalMap()
    # The Pi's plan latency: each plan lands `slowdown` times what it took
    # here later, against a budget cut by as much. 0 is every plan at once
    # with no budget, which is the same drive however loaded this machine
    # is: the tests' setting.
    slow = PI_SLOWDOWN if slowdown is None else float(slowdown)
    budget = params.plan_budget_s
    nav = nv.Navigator(params._replace(
        plan_budget_s=None if budget is None or slow <= 0 else budget / slow),
        seed=seed, planner=nv.InlinePlanner(slow))
    if scenario == "rove":
        nav.rove()
        max_s = rove_s if max_s is None else max_s
    else:
        nav.set_goal(goal[0], goal[1])
        if max_s is None:
            legs = [start[:2], goal[:2]] + [g[:2] for g in extra_goals]
            d = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                    for a, b in zip(legs, legs[1:]))
            max_s = (max(60.0, 12.0 * d) + (len(legs) - 1) * hold_s
                     + (0.0 if lost is None else lost[1] - lost[0]))
    safety = Safety()
    throttle, front, rear_cmd = 0.0, 0.0, 0.0
    steps = int(max_s / DT)
    min_clear = truck.clearance()
    track = [truck.pose] if trace else None
    est_track = [] if trace else None
    paths = [] if trace else None
    coverage = []
    reach = None
    if scenario == "rove":
        reach = nw.reachable_mask(world, start[0], start[1], cmod.INSCRIBED_M)
    plan_ms = []
    done_t = None
    outcome = "timeout"
    events = []
    t0 = time.perf_counter()
    pending = [tuple(g) for g in extra_goals]
    held_at = None              # when the current goal was first held
    holds = []                  # one per goal: (status, time held, metres off)
    hold_travel = 0.0           # metres moved more than HOLD_SETTLE_S into a hold
    hold_drive = 0              # decisions with a throttle while holding
    lost_drive = 0              # decisions with a throttle while the map was lost
    lost_ticks = 0
    steer_moves = harsh_steer = 0
    last_front = None
    for step in range(steps):
        now = step * DT
        if step % DECIDE_EVERY == 0:
            map_lost = lost is not None and lost[0] <= now < lost[1]
            ranges = lidar.scan(truck.pose)
            est = perr.apply(truck.pose)
            if not map_lost:
                # The live SLAM writes nothing while it is lost.
                pts = smap.integrate(est, ranges, lidar.angles)
                local.add(now, est[:2], pts)
            rear = lidar.rear_clearance(ranges)
            min_clear = min(min_clear, truck.clearance())
            backing = safety.recovering(now, world, truck.pose, rear)
            if safety.handed_back:
                nav.recovered(now)
            if backing is not None:
                throttle, front = backing
                safety.fwd_since = None
            else:
                view = None if map_lost else smap.view(est, 1.0)
                plans_before = nav.plans
                out = nav.tick(now, view, local)
                if nav.plans != plans_before:
                    plan_ms.append(nav.last_plan_ms)
                    if trace and nav.path is not None:
                        paths.append(nav.path.poses.copy())
                if out.recover:
                    safety.begin(now, "nav")
                    throttle, front = 0.0, 0.0
                elif nav.holding:
                    throttle, front = 0.0, 0.0
                    last_front = None
                    if held_at is None:
                        held_at = now
                        holds.append((nav.status, round(now, 2), round(math.hypot(
                            truck.x - goal[0], truck.y - goal[1]), 3)))
                    if now - held_at >= hold_s - 1e-9:
                        if not pending:
                            done_t = now
                            outcome = {nv.ST_ARRIVED: "arrived",
                                       nv.ST_CLOSEST: "closest"}.get(nav.status, "gave up")
                            break
                        goal = pending.pop(0)
                        nav.set_goal(goal[0], goal[1])
                        held_at = None
                else:
                    held_at = None
                    throttle, front, _ = nd.request_for(out, FORWARD_SPEED)
                    if throttle > 0 and _box_solid(world, truck.pose, _FLOOR_BOX):
                        throttle = 0.0
                        safety.floor_holds += 1
                        if safety.standoff_since is None:
                            safety.standoff_since = now
                        elif now - safety.standoff_since >= nd.STANDOFF_S - 1e-6:
                            safety.standoff_since = None
                            if backout:
                                safety.begin(now, "standoff")
                            else:
                                safety.causes["standoff"] += 1
                                nav.blocked_ahead(now, est, nd.FLOOR_AHEAD_M)
                    else:
                        safety.standoff_since = None
                    if throttle < 0 and rear is not None and rear < dv.REAR_STOP_CLEAR_MM:
                        throttle, front = 0.0, 0.0
                        safety.rear_stops += 1
                    # STUCK: forward commanded a second, the truck not moving.
                    if throttle > 0:
                        if safety.fwd_since is None:
                            safety.fwd_since = now
                        if abs(truck.v) < 0.03 and abs(truck.yaw_rate) < 0.05:
                            if safety.still_since is None:
                                safety.still_since = now
                        else:
                            safety.still_since = None
                        if (now - safety.fwd_since >= STUCK_S
                                and safety.still_since is not None
                                and now - safety.still_since >= STUCK_S
                                and now - safety.last_recovery >= RECOVERY_COOLDOWN_S):
                            if backout:
                                safety.begin(now, "stuck")
                            else:
                                # NavDriver: neutral, the spot marked, a fresh plan.
                                safety.causes["stuck"] += 1
                                safety.last_recovery = now
                                nav.blocked_ahead(now, est, tk.FRONT_M)
                                throttle, front = 0.0, 0.0
                            nav.recovered(now)
                    else:
                        safety.fwd_since = None
                        safety.still_since = None
                    if throttle > 0:
                        safety.last_front = front
                    if throttle != 0.0:
                        if last_front is not None:
                            steer_moves += 1
                            harsh_steer += abs(front - last_front) > HARSH_STEER
                        last_front = front
                    else:
                        last_front = None
                if nav.holding and throttle != 0.0:
                    hold_drive += 1
                if map_lost:
                    lost_ticks += 1
                    lost_drive += throttle != 0.0
            if scenario == "rove" and step % 500 == 0:
                coverage.append((now, _covered_m2(smap, world, reach)))
            if trace:
                est_track.append(est)
        x0, y0 = truck.x, truck.y
        truck.step(throttle, front)
        if held_at is not None and now - held_at >= HOLD_SETTLE_S:
            hold_travel += math.hypot(truck.x - x0, truck.y - y0)
        if trace and step % 5 == 0:
            track.append(truck.pose)
    else:
        done_t = max_s
        if scenario == "rove":
            outcome = "rove"
        elif held_at is not None:
            outcome = {nv.ST_ARRIVED: "arrived",
                       nv.ST_CLOSEST: "closest"}.get(nav.status, "gave up")
    if scenario == "rove":
        coverage.append((done_t, _covered_m2(smap, world, reach)))
    wall_s = time.perf_counter() - t0
    res = {
        "scenario": scenario, "level": level, "seed": int(seed),
        "world": os.path.basename(os.path.dirname(world.name)) or world.name,
        "start": [round(float(v), 3) for v in start],
        "goal": None if goal is None else [round(float(v), 3) for v in goal[:2]],
        "outcome": outcome, "time_s": round(float(done_t), 2),
        "collisions": int(truck.collisions), "contact_s": round(truck.contact_s, 2),
        "backouts": dict(safety.causes), "floor_holds": safety.floor_holds,
        "rear_stops": safety.rear_stops,
        "min_clearance_m": round(float(min_clear), 3),
        "travel_m": round(truck.travel, 2), "plans": nav.plans,
        "forecast_stops": nav.forecast_stops,
        "why": next((e for _, e in reversed(nav.events) if e.startswith("gave up")), ""),
        "plan_ms_mean": round(float(np.mean(plan_ms)), 1) if plan_ms else 0.0,
        "plan_ms_max": round(float(np.max(plan_ms)), 1) if plan_ms else 0.0,
        "plan_ms": [round(float(v), 1) for v in plan_ms],
        "final_err_m": None if goal is None else round(math.hypot(
            truck.x - goal[0], truck.y - goal[1]), 3),
        "closest_err_m": None if expect is None or goal is None else round(
            math.hypot(truck.x - goal[0], truck.y - goal[1])
            - math.hypot(expect[0] - goal[0], expect[1] - goal[1]), 3),
        "holds": holds, "hold_travel_m": round(hold_travel, 3),
        "hold_drive": hold_drive, "lost_ticks": lost_ticks, "lost_drive": lost_drive,
        "steer_moves": steer_moves, "harsh_steer": harsh_steer,
        "wall_s": round(wall_s, 1),
    }
    if scenario == "rove":
        res["coverage"] = [(round(t, 1), round(a, 2)) for t, a in coverage]
        res["reachable_m2"] = round(float(reach.sum()) * world.res ** 2, 2)
    if trace:
        res["_trace"] = {"track": np.array(track), "paths": paths,
                         "smap": smap, "events": nav.events}
    return res


def _covered_m2(smap, world, reach):
    """Floor the map knows is free that really is reachable floor, m^2."""
    free = smap.known_free()
    k = int(round(smap.res / world.res))
    r = reach
    px, py = (-r.shape[0]) % k, (-r.shape[1]) % k
    r = np.pad(r, ((0, px), (0, py)))
    r = r.reshape(r.shape[0] // k, k, r.shape[1] // k, k).any(axis=(1, 3))
    ox = int(round(-smap.origin[0] / smap.res))
    oy = int(round(-smap.origin[1] / smap.res))
    w, h = min(r.shape[0], free.shape[0] - ox), min(r.shape[1], free.shape[1] - oy)
    return float((free[ox:ox + w, oy:oy + h] & r[:w, :h]).sum()) * smap.res ** 2


# ---------------------------------------------------------------------------
# Picking episodes
# ---------------------------------------------------------------------------

def pick_pair(world, rng, min_d=1.5, max_d=12.0, tries=200):
    """A start and goal on the same reachable floor, a real drive apart."""
    need = cmod.INSCRIBED_M + 0.04
    for _ in range(tries):
        s = nw.sample_poses(world, rng, 1, clearance=0.4)
        if not s:
            return None
        s = s[0]
        mask = nw.reachable_mask(world, s[0], s[1], need)
        g = nw.sample_poses(world, rng, 1, clearance=0.35, mask=mask)
        if not g:
            continue
        g = g[0]
        d = math.hypot(g[0] - s[0], g[1] - s[1])
        if min_d <= d <= max_d:
            return s, g
    return None


def closest_reachable(world, start, goal):
    """The reachable floor point nearest `goal`, by the truck's own rule (a
    disc of the cost map's inscribed radius from `start`), or None."""
    mask = nw.reachable_mask(world, start[0], start[1], cmod.INSCRIBED_M + 0.02)
    ii, jj = np.nonzero(mask)
    if not len(ii):
        return None
    x = (ii + 0.5) * world.res
    y = (jj + 0.5) * world.res
    k = int(np.argmin(np.hypot(x - goal[0], y - goal[1])))
    return float(x[k]), float(y[k])


def pick_wall_goal(world, rng, start, min_off=0.15, max_off=0.8, min_d=1.5,
                   max_d=12.0, tries=400):
    """(goal, expect): a goal inside something solid, min_off-max_off from
    the nearest point the truck can reach from `start`, and that point."""
    mask = nw.reachable_mask(world, start[0], start[1], cmod.INSCRIBED_M + 0.02)
    if not mask.any():
        return None
    reach_d = ndimage.distance_transform_edt(~mask) * world.res
    ok = world.solid & (reach_d >= min_off) & (reach_d <= max_off)
    ii, jj = np.nonzero(ok)
    for _ in range(tries if len(ii) else 0):
        k = int(rng.integers(len(ii)))
        g = ((ii[k] + 0.5) * world.res, (jj[k] + 0.5) * world.res)
        if min_d <= math.hypot(g[0] - start[0], g[1] - start[1]) <= max_d:
            return g, closest_reachable(world, start, g)
    return None


def jobs_for(maps, episodes, rove, levels, seed, scenarios):
    """(map, scenario, start, goal, level, seed, expect) per episode. The
    goto and goto_blank pairs are the same with or without goto_wall: its
    goals come off a stream of their own."""
    rng = np.random.default_rng(seed)
    wall_rng = np.random.default_rng(seed + 7919)
    jobs = []
    worlds = {m: nw.load_map(m) for m in maps}
    for i in range(episodes):
        m = maps[i % len(maps)]
        pair = pick_pair(worlds[m], rng)
        if pair is None:
            continue
        s, g = pair
        for sc in scenarios:
            if sc == "rove":
                continue
            goal, expect = g, None
            if sc == "goto_wall":
                found = pick_wall_goal(worlds[m], wall_rng, s)
                if found is None:
                    continue
                goal, expect = found
            for lv in levels:
                jobs.append((m, sc, s, goal, lv, int(rng.integers(1 << 30)), expect))
    if "rove" in scenarios:
        for i in range(rove):
            m = maps[i % len(maps)]
            s = nw.sample_poses(worlds[m], rng, 1, clearance=0.45)[0]
            for lv in levels:
                jobs.append((m, "rove", s, None, lv, int(rng.integers(1 << 30)), None))
    return jobs


_WORLDS = {}
_PARAMS = nv.Params()
_ROVE_S = 300.0
_BACKOUT = False
_HOLD_S = 3.0


def _init(params, rove_s, backout=False, hold_s=3.0):
    global _PARAMS, _ROVE_S, _BACKOUT, _HOLD_S
    _PARAMS, _ROVE_S, _BACKOUT, _HOLD_S = params, rove_s, backout, hold_s


def _run(job):
    m, sc, s, g, lv, seed, expect = job
    if m not in _WORLDS:
        _WORLDS[m] = nw.load_map(m)
    try:
        return run_episode(_WORLDS[m], sc, s, g, lv, seed, params=_PARAMS,
                           rove_s=_ROVE_S, expect=expect, backout=_BACKOUT,
                           hold_s=_HOLD_S)
    except Exception as e:
        return {"scenario": sc, "level": lv, "seed": seed, "outcome": "error",
                "error": repr(e)}


def summarise(rows):
    """{(scenario, level): stats}."""
    out = {}
    keys = sorted({(r["scenario"], r["level"]) for r in rows})
    for k in keys:
        rs = [r for r in rows if (r["scenario"], r["level"]) == k]
        n = len(rs)
        if k[0] == "rove":
            cov = {}
            for minute in (1, 2, 5):
                vals = [_cov_at(r, minute * 60) for r in rs if "coverage" in r]
                frac = [_cov_at(r, minute * 60) / r["reachable_m2"]
                        for r in rs if "coverage" in r and r.get("reachable_m2")]
                cov[minute] = (float(np.mean(vals)) if vals else 0.0,
                               float(np.mean(frac)) if frac else 0.0)
            out[k] = {"n": n, "coverage": cov}
        else:
            if k[0] == "goto_wall":
                # Right is holding at the closest point, within the arrival
                # tolerance and a cell of it.
                ok = [r for r in rs if r["outcome"] in ("closest", "arrived")
                      and r.get("closest_err_m") is not None
                      and r["closest_err_m"] <= CLOSEST_OK_M]
                errs = [r["closest_err_m"] for r in ok]
            else:
                ok = [r for r in rs if r["outcome"] == "arrived"]
                errs = [r["final_err_m"] for r in ok]
            out[k] = {"n": n, "success": len(ok) / n if n else 0.0,
                      "time_med": float(np.median([r["time_s"] for r in ok])) if ok else math.nan,
                      "gave_up": sum(r["outcome"] == "gave up" for r in rs) / max(n, 1),
                      "closest": sum(r["outcome"] == "closest" for r in rs) / max(n, 1),
                      "timeout": sum(r["outcome"] == "timeout" for r in rs) / max(n, 1),
                      "errors": sum(r["outcome"] == "error" for r in rs),
                      "err_med": float(np.median(errs)) if errs else math.nan,
                      "err_p90": float(np.percentile(errs, 90)) if errs else math.nan}
        good = [r for r in rs if r["outcome"] != "error"]
        moves = sum(r.get("steer_moves", 0) for r in good)
        out[k]["harsh_steer"] = (sum(r.get("harsh_steer", 0) for r in good) / moves
                                 if moves else math.nan)
        out[k]["hold_travel_max"] = max([r.get("hold_travel_m", 0.0) for r in good] or [0.0])
        out[k]["hold_drive"] = sum(r.get("hold_drive", 0) for r in good)
        out[k].update({
            "collisions_per_ep": float(np.mean([r["collisions"] for r in good])) if good else 0,
            "eps_with_collision": float(np.mean([r["collisions"] > 0 for r in good])) if good else 0,
            "backouts_per_ep": float(np.mean([sum(r["backouts"].values()) for r in good])) if good else 0,
            "stuck_per_ep": float(np.mean([r["backouts"]["stuck"] for r in good])) if good else 0,
            "min_clear_p05": float(np.percentile([r["min_clearance_m"] for r in good], 5)) if good else 0,
            "min_clear_med": float(np.median([r["min_clearance_m"] for r in good])) if good else 0,
            "plan_ms_mean": float(np.mean([r["plan_ms_mean"] for r in good])) if good else 0,
            "plan_ms_max": float(np.max([r["plan_ms_max"] for r in good])) if good else 0,
        })
        pooled = [v for r in good for v in r.get("plan_ms", [])]
        out[k]["plan_ms_p50"] = float(np.percentile(pooled, 50)) if pooled else math.nan
        out[k]["plan_ms_p90"] = float(np.percentile(pooled, 90)) if pooled else math.nan
    return out


def _cov_at(r, t):
    best = 0.0
    for tt, a in r["coverage"]:
        if tt <= t + 1e-6:
            best = a
    return best


def table(summary):
    lines = []
    lines.append("| scenario | level | n | success | time med s | err med / p90 cm | "
                 "gave up | closest | timeout | harsh steer | hold moved max cm | "
                 "collision eps | collisions/ep | STUCK+standoff/ep | STUCK/ep | "
                 "min clear p05 / med cm | plan ms p50 / p90 / max |")
    lines.append("|" + "---|" * 18)
    for (sc, lv), s in summary.items():
        if sc == "rove":
            continue
        lines.append(
            f"| {sc} | {lv} | {s['n']} | {100 * s['success']:.1f}% | {s['time_med']:.1f} | "
            f"{100 * s['err_med']:.0f} / {100 * s['err_p90']:.0f} | "
            f"{100 * s['gave_up']:.1f}% | {100 * s['closest']:.1f}% | "
            f"{100 * s['timeout']:.1f}% | {100 * s['harsh_steer']:.1f}% | "
            f"{100 * s['hold_travel_max']:.1f} | "
            f"{100 * s['eps_with_collision']:.1f}% | {s['collisions_per_ep']:.2f} | "
            f"{s['backouts_per_ep']:.2f} | {s['stuck_per_ep']:.2f} | "
            f"{100 * s['min_clear_p05']:.1f} / {100 * s['min_clear_med']:.1f} | "
            f"{s['plan_ms_p50']:.0f} / {s['plan_ms_p90']:.0f} / {s['plan_ms_max']:.0f} |")
    rove = [(k, s) for k, s in summary.items() if k[0] == "rove"]
    if rove:
        lines.append("")
        lines.append("| rove | level | n | m^2 at 1 / 2 / 5 min | of reachable at 1 / 2 / 5 min | "
                     "collision eps | collisions/ep | back-outs/ep | min clear p05 cm |")
        lines.append("|" + "---|" * 9)
        for (sc, lv), s in rove:
            c = s["coverage"]
            lines.append(
                f"| rove | {lv} | {s['n']} | {c[1][0]:.1f} / {c[2][0]:.1f} / {c[5][0]:.1f} | "
                f"{100 * c[1][1]:.0f}% / {100 * c[2][1]:.0f}% / {100 * c[5][1]:.0f}% | "
                f"{100 * s['eps_with_collision']:.1f}% | {s['collisions_per_ep']:.2f} | "
                f"{s['backouts_per_ep']:.2f} | {100 * s['min_clear_p05']:.1f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pictures
# ---------------------------------------------------------------------------

def draw(world, res, path_png, title=""):
    """Two panels side by side, drawn with OpenCV (the car's own dependency;
    matplotlib is not in the venv): the world with the driven track, and the
    map the navigator built with the same track. Blue the track, orange the
    last few plans, green the start, red the goal, the truck's outline where
    it ended."""
    import cv2
    tr = res["_trace"]
    smap = tr["smap"]
    scale = 40.0                       # px per metre
    w_m, h_m = world.size_m
    W, H = int(w_m * scale), int(h_m * scale)

    def px(x, y):
        return int(round(x * scale)), int(round(H - 1 - y * scale))

    def panel(base):
        img = cv2.cvtColor((base * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for p in tr["paths"][-6:]:
            pts = np.array([px(x, y) for x, y, _ in p], np.int32)
            cv2.polylines(img, [pts], False, (0, 140, 255), 1, cv2.LINE_AA)
        pts = np.array([px(x, y) for x, y, _ in tr["track"]], np.int32)
        cv2.polylines(img, [pts], False, (200, 90, 20), 2, cv2.LINE_AA)
        corners = np.array([px(x, y) for x, y in tk.footprint_corners(*tr["track"][-1])], np.int32)
        cv2.fillPoly(img, [corners], (200, 90, 20))
        cv2.drawMarker(img, px(*res["start"][:2]), (0, 160, 0), cv2.MARKER_TRIANGLE_UP, 14, 2)
        if res["goal"]:
            cv2.drawMarker(img, px(*res["goal"]), (0, 0, 220), cv2.MARKER_STAR, 16, 2)
        return img

    # The world: solid black, floor white. Resampled to the panel.
    sol = cv2.resize(world.solid.T[::-1].astype(np.uint8), (W, H),
                     interpolation=cv2.INTER_NEAREST)
    left = panel(1.0 - 0.85 * sol)
    lo = smap.lo
    m = np.full(lo.shape, 0.6)
    m[lo < 0] = 1.0
    m[lo > nw.L_OCC] = 0.0
    # Map cells into world pixels.
    ox, oy = smap.origin
    xs = (np.arange(W) + 0.5) / scale
    ys = (H - 1 - np.arange(H) + 0.5) / scale
    ix = np.clip(((xs - ox) / smap.res).astype(int), 0, lo.shape[0] - 1)
    iy = np.clip(((ys - oy) / smap.res).astype(int), 0, lo.shape[1] - 1)
    right = panel(m[ix[None, :], iy[:, None]])
    img = np.concatenate([left, np.full((H, 8, 3), 255, np.uint8), right], 1)
    head = np.full((44, img.shape[1], 3), 255, np.uint8)
    text = title or (f"{res['scenario']} {res['level']}: {res['outcome']} {res['time_s']} s, "
                     f"collisions {res['collisions']}, back-outs {sum(res['backouts'].values())}, "
                     f"min clearance {100 * res['min_clearance_m']:.0f} cm")
    cv2.putText(head, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(head, "left: world and track   right: the map it built (white free, black solid, grey unknown)",
                (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.imwrite(path_png, np.concatenate([head, img], 0))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def params_from(args):
    p = nv.Params()
    over = json.loads(args.params) if getattr(args, "params", None) else {}
    planner = over.pop("planner", {})
    if planner:
        p = p._replace(planner=p.planner._replace(**planner))
    return p._replace(**over)


def _maps(spec):
    out = []
    for s in spec:
        if os.path.isdir(s):
            out += sorted(glob.glob(os.path.join(s, "*", "q5_hires_result.npz")))
        else:
            out.append(s)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    one = sub.add_parser("one")
    one.add_argument("--map", required=True)
    one.add_argument("--scenario", default="goto", choices=SCENARIOS)
    one.add_argument("--level", default="measured", choices=LEVELS)
    one.add_argument("--seed", type=int, default=0)
    one.add_argument("--rove-minutes", type=float, default=3.0)
    one.add_argument("--png")
    one.add_argument("--params")
    b = sub.add_parser("batch")
    for p in (one, b):
        p.add_argument("--backout", action="store_true",
                       help="the old safety layer: a reverse at lock on STUCK "
                            "and on a floor standoff (off on the car)")
        p.add_argument("--hold-s", type=float, default=3.0,
                       help="seconds an episode goes on holding at its goal")
    b.add_argument("--maps", nargs="+", required=True)
    b.add_argument("--episodes", type=int, default=100)
    b.add_argument("--rove", type=int, default=8)
    b.add_argument("--rove-minutes", type=float, default=5.0)
    b.add_argument("--levels", nargs="+", default=list(LEVELS))
    b.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    b.add_argument("--jobs", type=int, default=4)
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--out", required=True)
    b.add_argument("--tag", default="run")
    b.add_argument("--params")
    b.add_argument("--pngs", type=int, default=6)
    args = ap.parse_args(argv)
    params = params_from(args)

    if args.cmd == "one":
        world = nw.load_map(args.map)
        rng = np.random.default_rng(args.seed)
        if args.scenario == "rove":
            s = nw.sample_poses(world, rng, 1, clearance=0.45)[0]
            g = None
        else:
            s, g = pick_pair(world, rng)
        expect = None
        if args.scenario == "goto_wall":
            g, expect = pick_wall_goal(world, rng, s)
        r = run_episode(world, args.scenario, s, g, args.level, args.seed,
                        rove_s=args.rove_minutes * 60, params=params,
                        trace=bool(args.png), expect=expect,
                        backout=args.backout, hold_s=args.hold_s)
        if args.png:
            draw(world, r, args.png)
        r.pop("_trace", None)
        print(json.dumps(r))
        return 0

    maps = _maps(args.maps)
    if args.jobs > 4:
        print("at most 4 jobs", file=sys.stderr)
        args.jobs = 4
    os.makedirs(args.out, exist_ok=True)
    jobs = jobs_for(maps, args.episodes, args.rove, args.levels, args.seed,
                    args.scenarios)
    print(f"{len(jobs)} episodes on {len(maps)} maps, {args.jobs} jobs", file=sys.stderr)
    t0 = time.time()
    rows = []
    out_path = os.path.join(args.out, f"{args.tag}.jsonl")
    with open(out_path, "w") as f, Pool(args.jobs, initializer=_init,
                                        initargs=(params, args.rove_minutes * 60,
                                                  args.backout, args.hold_s)) as pool:
        for i, r in enumerate(pool.imap_unordered(_run, jobs, chunksize=1)):
            rows.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)} {time.time() - t0:.0f} s", file=sys.stderr)
    summary = summarise(rows)
    text = table(summary)
    with open(os.path.join(args.out, f"{args.tag}.md"), "w") as f:
        f.write(f"params: {json.dumps(params._asdict(), default=str)}\n\n{text}\n")
    print(text)
    # Sample pictures: a few failures and a few successes, re-run with a trace.
    worlds = {}
    picks = ([r for r in rows if r["outcome"] in ("gave up", "timeout")][:args.pngs // 2]
             + [r for r in rows if r.get("collisions", 0) > 0][:args.pngs // 3]
             + [r for r in rows if r["outcome"] in ("arrived", "rove")][:args.pngs // 2])
    for k, r in enumerate(picks):
        job = next((j for j in jobs if j[5] == r["seed"]), None)
        if job is None:
            continue
        m = job[0]
        worlds.setdefault(m, nw.load_map(m))
        rr = run_episode(worlds[m], job[1], job[2], job[3], job[4], job[5],
                         params=params, rove_s=args.rove_minutes * 60, trace=True,
                         expect=job[6], backout=args.backout, hold_s=args.hold_s)
        draw(worlds[m], rr, os.path.join(args.out, f"{args.tag}_{k:02d}_{rr['scenario']}_"
                                                    f"{rr['level']}_{rr['outcome'].replace(' ', '_')}.png"))
    print(f"done in {time.time() - t0:.0f} s -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
