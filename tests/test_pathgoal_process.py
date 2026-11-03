"""The path planner's process (autopilot/pathgoal.py, MODE_PROCESS): the plan
made there is the one made in the caller, it comes back with its cost map
for the line of sight here, the SLAM pump hands it each map, and when the
process dies the last plan is used until it is back."""

import os
import queue
import time

import numpy as np
import pytest

pytest.importorskip("scipy")

from crawler.autopilot import pathgoal as pgo  # noqa: E402
from crawler.mapping import runner as rn  # noqa: E402

from test_goal_reseed import crop_at  # noqa: E402
from test_pathgoal import WALL, Clock, MapSlam, Source, planner, world  # noqa: E402


def wait_for(cond, timeout=30.0):
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def in_process(view, clock):
    return pgo.PathPlanner(MapSlam(view), clock=clock, stage_deg=None,
                           aim_rate_dps=None)


def test_a_plan_made_in_the_process_is_the_one_made_in_the_caller():
    view = world([WALL])
    src = Source((4.0, 0.0, 0))
    crop = crop_at(0.0, 0.0, 0.0)
    here, _ = planner(view)
    want = here.encode(crop, src)
    clock = Clock()
    p = in_process(view, clock)
    try:
        p.update(crop, src)
        assert p.settle(60.0), "the process answered"
        assert wait_for(lambda: p.plans >= 1)
        assert p._proc is not None and p._proc.pid != os.getpid()
        assert p._thread.name == "path-plans"
        got = p.encode(crop, src)
        assert p.fed == "subgoal" and p.why == ""
        np.testing.assert_array_equal(got, want)
        plan, ref = p._plan, here._plan
        np.testing.assert_array_equal(plan.points, ref.points)
        assert plan.route == ref.route and plan.vias == ref.vias
        assert plan.cm.inscribed == ref.cm.inscribed
        assert plan.made == clock.t, "made on the planner's clock, not the process's"
        assert plan.cpu_ms > 0.0
    finally:
        p.stop()
    assert p._proc is None


def test_when_the_process_dies_the_last_plan_is_used_until_it_is_back():
    view = world([WALL])
    src = Source((4.0, 0.0, 0))
    crop = crop_at(0.0, 0.0, 0.0)
    clock = Clock()
    p = in_process(view, clock)
    said = []
    p._say = lambda *a: said.append(a)
    try:
        p.update(crop, src)
        assert p.settle(60.0) and wait_for(lambda: p.plans >= 1)
        plan, first = p._plan, p._proc
        first.kill()
        first.join(10.0)
        # Meanwhile: the plan in hand.
        p.encode(crop, src)
        assert p._plan is plan and p.fed == "subgoal"
        assert wait_for(lambda: p.restarts == 1 and p._proc is not None
                        and p._proc is not first and p._proc.is_alive())
        assert any("died" in a[2] for a in said)
        # Gone stale before the new one plans: the goal itself, as with no path.
        clock.t += pgo.STALE_S + 0.1
        p._req_at = clock.t            # nothing sent this decision
        g = p.encode(crop, src)
        assert p.why == "plan stale"
        np.testing.assert_array_equal(g, src.encode(crop))
        # And planned again on the new one: the map and the goal were resent.
        p._req_at = None
        p.update(crop, src)
        assert p.settle(60.0) and wait_for(lambda: p._plan is not plan)
        p.encode(crop, src)
        assert p.fed == "subgoal"
    finally:
        p.stop()


def test_the_pump_hands_each_map_to_its_listeners_once():
    slam = rn.SlamRunner()
    got = []
    slam.map_listeners.append(got.append)
    slam.map_listeners.append(lambda v: 1 / 0)       # skipped, never in the way
    view = world([WALL])
    slam._take(("map", view))
    assert got == [view] and slam.map() is view


def test_on_map_sends_the_process_each_map_once():
    p = pgo.PathPlanner(MapSlam(None), clock=Clock())
    p._inbox = queue.Queue(maxsize=pgo.INBOX)
    a, b = world([WALL]), world([WALL], t=1.0)
    for v in (a, a, None, b, b):
        p.on_map(v)
    sent = []
    while not p._inbox.empty():
        sent.append(p._inbox.get_nowait())
    assert [m[1] for m in sent] == [a, b] and all(m[0] == "map" for m in sent)
    # In the caller or on a thread nothing is sent anywhere.
    for kw in ({"background": False}, {"mode": pgo.MODE_THREAD}):
        q = pgo.PathPlanner(MapSlam(None), clock=Clock(), **kw)
        q._inbox = queue.Queue()
        q.on_map(a)
        assert q._inbox.empty()


def test_requests_go_over_no_more_often_than_req_every_s_but_a_new_goal_at_once():
    clock = Clock()
    p = pgo.PathPlanner(MapSlam(world([WALL])), clock=clock)
    p._ensure_thread = lambda: None
    p._inbox = queue.Queue(maxsize=100)
    src = Source((4.0, 0.0, 0))

    def reqs():
        out = []
        while not p._inbox.empty():
            m = p._inbox.get_nowait()
            if m[0] == "req":
                out.append(m)
        return out

    p.update(crop_at(0.0, 0.0), src)
    p.update(crop_at(0.01, 0.0), src)
    assert len(reqs()) == 1
    clock.t += pgo.REQ_EVERY_S
    p.update(crop_at(0.02, 0.0), src)
    (m,) = reqs()
    assert m[2].pose[0] == 0.02 and m[3] == clock.t
    src.goal = (4.0, 1.0, 0)
    p.update(crop_at(0.03, 0.0), src)
    (m,) = reqs()
    assert m[2].goal[:2] == (4.0, 1.0)


def test_a_mode_that_is_not_one_is_refused():
    with pytest.raises(ValueError):
        pgo.PathPlanner(MapSlam(None), mode="fork")
