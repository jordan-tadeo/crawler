"""A clicked goal across a SLAM reseed: carried onto the fresh map, marked
approximate, and put back exactly when a merge brings its map back.

slam.py says how each fresh map sits on the one before it (Result.link,
off the gyro and the commanded speed while lost), egomap.py keeps those
links in every crop, and goal.ClickGoal moves the goal along them. These
hold each part, and a synthetic room the truck is lost in for eleven
seconds while it drives near three metres, then finds again.
"""

import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from crawler.autopilot import goal as gl  # noqa: E402
from crawler.autopilot import policy_goal as pg  # noqa: E402
from crawler.mapping import egomap as em  # noqa: E402
from crawler.mapping import grid as gr  # noqa: E402
from crawler.mapping import runner as rn  # noqa: E402
from crawler.mapping import scan as sc  # noqa: E402
from crawler.mapping import slam as sl  # noqa: E402
from crawler.mapping.imu import ImuTrack  # noqa: E402
from crawler.nav import click as ck  # noqa: E402
from crawler.telemetry import nav_view as nv  # noqa: E402
from crawler.telemetry import telemetry as tm  # noqa: E402

from test_mapping import LEVEL, PERIOD, ROOM, path_pose, scan_at  # noqa: E402


def crop_at(x=0.0, y=0.0, yaw=0.0, frame=0, present=True, links=(), t=0.0):
    return em.Crop(t, present, np.full((em.SIZE, em.SIZE), gr.FREE, np.uint8),
                   np.zeros((em.SIZE, em.SIZE), np.uint8), x, y, yaw, frame,
                   tuple(links))


def link(seq, frm, to, T=(0.0, 0.0, 0.0), err_m=0.0, err_yaw=0.0, at=(0.0, 0.0),
         why="lost"):
    return em.Link(seq, why, frm, to, T[0], T[1], T[2], err_m, err_yaw, at[0], at[1])


def click(x, y):
    return ck.ClickGoal(x, y, None, "map", "/move_base_simple/goal", 0.0)


class Clicks:
    def __init__(self, *goals):
        self.goals = list(goals)

    def take(self):
        return self.goals.pop(0) if self.goals else None


def apply(T, xy):
    c, s = math.cos(T[2]), math.sin(T[2])
    return (c * xy[0] - s * xy[1] + T[0], s * xy[0] + c * xy[1] + T[1])


# --- ClickGoal along the links ----------------------------------------------------------

def test_a_reseed_carries_the_goal_onto_the_fresh_map_marked_approximate():
    said = []
    g = gl.ClickGoal(say=lambda *a: said.append(a))
    g.poll(Clicks(click(3.0, 1.0)), crop_at(frame=0))
    assert not g.status()["approx"] and g.status()["approx_m"] == 0.0
    T = (0.5, -0.2, math.radians(30))
    L = link(1, 0, 1, T, err_m=0.3, err_yaw=math.radians(5), at=(1.0, 1.0))
    on_new = crop_at(0.0, 0.0, frame=1, links=[L])
    g.observe(on_new)
    want = apply(T, (3.0, 1.0))
    assert g.goal[2] == 1
    assert g.goal[0] == pytest.approx(want[0]) and g.goal[1] == pytest.approx(want[1])
    st = g.status()
    assert st["approx"] and st["on_map"] and st["frame"] == 1 == st["frame_now"]
    # Its own error plus the heading's, swung over the 2 m from the truck.
    assert st["approx_m"] == pytest.approx(0.3 + 2.0 * math.radians(5))
    assert g.encode(on_new)[0, pg.PRESENT] == 1.0, "the model is fed it"
    assert any("carried it onto" in a[2] for a in said)
    # Seen again on the next crop: not carried twice.
    g.observe(on_new)
    assert g.goal[0] == pytest.approx(want[0])


def test_a_merge_snaps_the_goal_back_to_where_it_was_clicked():
    said = []
    g = gl.ClickGoal(say=lambda *a: said.append(a))
    g.poll(Clicks(click(3.0, 1.0)), crop_at(frame=0))
    lost = link(1, 0, 1, (0.5, -0.2, 0.1), err_m=0.3)
    g.observe(crop_at(frame=1, links=[lost]))
    assert g.status()["approx"]
    # The merge's transform is not quite the inverse: the carry was off.
    back = link(2, 1, 0, (-0.6, 0.2, -0.1), why="merged")
    g.observe(crop_at(frame=0, links=[lost, back]))
    assert g.goal == (3.0, 1.0, 0)
    st = g.status()
    assert not st["approx"] and st["approx_m"] == 0.0
    said_snap = [a[2] for a in said if "exactly" in a[2]]
    assert said_snap and " m off" in said_snap[0]
    off = math.dist(apply(back[4:7], apply(lost[4:7], (3.0, 1.0))), (3.0, 1.0))
    assert f"{off:.2f} m off" in said_snap[0]
    assert g.transfer_errors == [pytest.approx(off)]


def test_reseeds_chain_and_a_merge_through_them_comes_back_exact():
    g = gl.ClickGoal()
    g.poll(Clicks(click(2.0, 0.0)), crop_at(frame=0))
    a = link(1, 0, 1, (1.0, 0.0, 0.0), err_m=0.2)
    b = link(2, 1, 2, (0.0, 1.0, 0.0), err_m=0.1, why="young")
    # Two reseeds the control loop never saw a crop between.
    g.observe(crop_at(frame=2, links=[a, b]))
    assert g.goal[2] == 2
    assert g.goal[:2] == pytest.approx((3.0, 1.0))
    assert g.status()["approx_m"] == pytest.approx(0.3)
    m = link(3, 2, 0, (-1.0, -1.0, 0.0), why="merged")
    g.observe(crop_at(frame=0, links=[a, b, m]))
    assert g.goal == (2.0, 0.0, 0) and not g.status()["approx"]


def test_a_goal_clicked_on_the_fresh_map_moves_onto_the_old_one_by_the_merge():
    g = gl.ClickGoal()
    lost = link(1, 0, 1, (0.5, 0.0, 0.0), err_m=0.3)
    g.observe(crop_at(frame=1, links=[lost]))
    g.poll(Clicks(click(1.0, 1.0)), crop_at(frame=1, links=[lost]))
    T = (2.0, -1.0, math.pi / 2)
    m = link(2, 1, 0, T, why="merged")
    g.observe(crop_at(frame=0, links=[lost, m]))
    assert g.goal[2] == 0 and g.goal[:2] == pytest.approx(apply(T, (1.0, 1.0)))
    assert not g.status()["approx"], "a merge is a fit, not a guess"


def test_links_from_before_the_click_never_move_it():
    g = gl.ClickGoal()
    old = link(1, 0, 1, (5.0, 0.0, 0.0), err_m=0.3)
    g.observe(crop_at(frame=1, links=[old]))
    g.poll(Clicks(click(1.0, 0.0)), crop_at(frame=1, links=[old]))
    back = link(2, 1, 0, (-5.0, 0.0, 0.0), why="merged")
    g.observe(crop_at(frame=0, links=[old, back]))
    assert g.goal[:2] == pytest.approx((-4.0, 0.0)) and g.goal[2] == 0


def test_a_click_on_the_crop_a_merge_came_in_on_is_not_carried_by_it():
    # The loss and the merge both arrive on the crop the click is placed off:
    # they happened before the click, and the goal is exact where clicked.
    said = []
    g = gl.ClickGoal(say=lambda *a: said.append(a))
    lost = link(1, 0, 1, (5.0, 0.0, 0.0), err_m=0.3)
    back = link(2, 1, 0, (-4.0, 0.0, 0.0), why="merged")
    g.poll(Clicks(click(1.0, 0.0)), crop_at(frame=0, links=[lost, back]))
    assert g.goal == (1.0, 0.0, 0) and not g.status()["approx"]
    assert g.transfer_errors == [] and not any("exactly" in a[2] for a in said)


def test_an_approximate_goal_still_holds_on_arrival():
    g = gl.ClickGoal()
    g.poll(Clicks(click(2.0, 0.0)), crop_at(frame=0))
    g.observe(crop_at(frame=1, links=[link(1, 0, 1, (0.0, 0.0, 0.0), err_m=0.5)]))
    assert g.status()["approx"]
    g.observe(crop_at(2.0 - pg.ARRIVE_M + 0.02, frame=1,
                      links=[link(1, 0, 1, (0.0, 0.0, 0.0), err_m=0.5)]))
    assert g.reached()


def test_the_marker_draws_an_approximate_goal_as_wide_as_its_doubt():
    from types import SimpleNamespace
    from crawler import __main__ as cm
    g = gl.ClickGoal()
    pilot = SimpleNamespace(goal=g)
    g.poll(Clicks(click(2.0, 0.0)), crop_at(frame=0))
    exact = cm.AutoPilot.goal_marker(pilot)
    assert exact == {"goal": (2.0, 0.0)}
    g.observe(crop_at(frame=1, links=[link(1, 0, 1, err_m=0.6)]))
    approx = cm.AutoPilot.goal_marker(pilot)
    assert approx["goal"] == pytest.approx((2.0, 0.0)) and approx["approx_m"] >= 0.6
    pytest.importorskip("foxglove")
    from test_lidar_view import decode
    sharp = decode(nv.path_scene(exact, 0)).entities[0].cylinders[0]
    soft = decode(nv.path_scene(approx, 0)).entities[0].cylinders[0]
    assert soft.size.x == pytest.approx(2 * approx["approx_m"])
    assert sharp.size.x == pytest.approx(nv.DISC_M)
    assert soft.color.a < sharp.color.a


def test_the_status_fits_its_schema_with_approx():
    props = tm.JSON_SCHEMAS["goal_status"]["schema"]["properties"]
    g = gl.ClickGoal()
    g.poll(Clicks(click(1.0, 0.0)), crop_at())
    assert set(g.status()) == set(props)
    assert {"approx", "approx_m"} <= set(props)


# --- egomap keeps the links -------------------------------------------------------------

def _res(**kw):
    base = dict(t=0.0, x=0.0, y=0.0, yaw=0.0, state="tracking", matched=True,
                written=True, why="", inliers=1.0, rms_m=0.0, constraint=0.4,
                returns=700, points=600, ground_dropped=0, tilt_deg=0.0, iters=3, ms=1.0)
    base.update(kw)
    return sl.Result(**base)


def test_every_crop_carries_the_links_between_its_map_frames():
    ego, g = em.EgoMap(), gr.Grid(size=256)
    assert ego.after_step(g, _res()).links == ()
    T = (0.1, 0.2, 0.3, 0.4, 0.05, 1.0, 2.0)
    c = ego.after_step(g, _res(reseed="lost", link=T))
    assert c.frame == 1 and len(c.links) == 1
    L = c.links[0]
    assert (L.seq, L.why, L.frm, L.to) == (1, "lost", 0, 1)
    assert (L.x, L.y, L.yaw, L.err_m, L.err_yaw, L.px, L.py) == T
    assert ego.after_step(g, _res()).links == c.links, "kept on every crop"
    c = ego.after_step(g, _res(reseed="merged", link=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
    assert c.frame == 0 and [(k.frm, k.to) for k in c.links] == [(0, 1), (1, 0)]


def test_a_lost_reseed_that_kept_the_bigger_shelf_merges_back_onto_it():
    # slam._start_over keeps whichever map is bigger on the shelf: a second
    # loss on a small fresh map drops the fresh map, not the old one.
    ego, g = em.EgoMap(), gr.Grid(size=256)
    ego.after_step(g, _res(reseed="lost"))
    assert ego.after_step(g, _res(reseed="lost", kept_shelf=True)).frame == 2
    assert ego.after_step(g, _res(reseed="merged")).frame == 0
    ego.after_step(g, _res(reseed="lost"))
    assert ego.after_step(g, _res(reseed="lost", kept_shelf=False)).frame == 4
    assert ego.after_step(g, _res(reseed="merged")).frame == 3


def test_the_links_kept_are_bounded():
    ego, g = em.EgoMap(), gr.Grid(size=256)
    for k in range(em.LINKS_KEPT + 10):
        c = ego.after_step(g, _res(reseed="young", link=(0.0,) * 7))
    assert len(c.links) == em.LINKS_KEPT and c.links[-1].seq == em.LINKS_KEPT + 10


# --- the runner hands SLAM the commanded speed ------------------------------------------

class FakeScan:
    def __init__(self, t, ranges):
        self.t, self.ranges, self.stamp_ms = t, ranges, None


def test_the_runner_hands_each_scan_the_commanded_speed(monkeypatch):
    monkeypatch.setattr(rn, "ranges_of", lambda scan, mount: scan.ranges)
    r = rn.SlamRunner(mode=rn.MODE_THREAD, speed=lambda: 0.42)
    r._worker = object()
    import queue
    r._inbox = queue.Queue(maxsize=4)
    r.on_scan(FakeScan(1.0, np.zeros(sc.SLOTS, np.float32)))
    assert r._inbox.get_nowait()[4] == 0.42
    r.speed = lambda: 1 / 0
    r.on_scan(FakeScan(1.1, np.zeros(sc.SLOTS, np.float32)))
    assert r._inbox.get_nowait()[4] is None, "a speed that fails is no speed"


def test_the_worker_steps_slam_with_the_speed(monkeypatch):
    seen = []
    real = sl.Slam.step

    def step(self, t, ranges, period=None, speed=None):
        seen.append(speed)
        return real(self, t, ranges, period, speed=speed)
    monkeypatch.setattr(sl.Slam, "step", step)
    import queue
    inbox, outbox = queue.Queue(), queue.Queue()
    inbox.put((0.1, None, scan_at(0.1), [], 0.3))
    inbox.put((0.2, None, scan_at(0.2), []))
    inbox.put(None)
    rn._work(inbox, outbox, None, 1e9)
    assert seen == [0.3, None]


def test_main_feeds_the_runner_the_speed_the_throttle_asks_for():
    import inspect
    from crawler import __main__ as cm
    src = inspect.getsource(cm.main)
    assert "speed=" in src and "ground_speed(vecon.last_throttle" in src


# --- the whole thing: lost for eleven seconds in a synthetic room -----------------------

V_BLIND = 0.3          # m/s while it is lost
W_BLIND = -0.15        # rad/s
BLIND_S = 11.0
T_LOST = 6.0


def blind_pose(t):
    """The room path to T_LOST, then a steady arc at V_BLIND, W_BLIND for
    BLIND_S, then standing still where that left it."""
    if t <= T_LOST:
        return path_pose(t)
    p0 = path_pose(T_LOST)
    s = min(t, T_LOST + BLIND_S) - T_LOST
    yaw = p0[2] + W_BLIND * s
    x = p0[0] + V_BLIND / W_BLIND * (math.sin(yaw) - math.sin(p0[2]))
    y = p0[1] - V_BLIND / W_BLIND * (math.cos(yaw) - math.cos(p0[2]))
    return np.array([x, y, yaw])


def clutter(pose, rng):
    """Short walls strewn round the truck, a new set every scan: nothing
    that fits the map, or the scan before."""
    segs = []
    for _ in range(16):
        cx = pose[0] + rng.uniform(-4.0, 4.0)
        cy = pose[1] + rng.uniform(-4.0, 4.0)
        a, L = rng.uniform(0.0, math.pi), rng.uniform(0.4, 1.2)
        segs.append(((cx, cy), (cx + L * math.cos(a), cy + L * math.sin(a))))
    return segs


def run_lost_drive(feed_speed):
    """Results, true poses and the speed fed, scan by scan."""
    rng = np.random.default_rng(7)
    imu = ImuTrack()
    slam = sl.Slam(imu=imu)
    out, truth = [], []
    t_end = T_LOST + BLIND_S + 40.0
    k = 0
    while True:
        k += 1
        t = k * PERIOD
        for j in range(10):
            tj = t - PERIOD + (j + 1) * 0.01
            rate = (blind_pose(tj + 1e-3)[2] - blind_pose(tj - 1e-3)[2]) / 2e-3
            imu.add(tj, (-rate, 0.0, 0.0), LEVEL)
        blind = T_LOST < t <= T_LOST + BLIND_S
        segs = clutter(blind_pose(t), rng) if blind else ROOM
        v = V_BLIND if blind else 0.0
        res = slam.step(t, scan_at(t, blind_pose, segs),
                        speed=v if feed_speed else None)
        out.append(res)
        truth.append(blind_pose(t))
        if res.reseed == "merged" or t > t_end:
            return out, truth


@pytest.fixture(scope="module")
def lost_drive():
    return run_lost_drive(True)


GOALS = ((0.0, 0.0), (2.0, 1.0), (-1.5, -1.0), (3.0, -1.5))


def carried(out, goal):
    """`goal` on the map before the first reseed, carried link by link to
    the frame the drive ended on, and the error estimate it came with."""
    xy, err = goal, 0.0
    for r in out:
        if r.reseed:
            x, y, yaw, e_m, e_yaw, px, py = r.link
            err += e_m + math.dist(xy, (px, py)) * e_yaw
            xy = apply((x, y, yaw), xy)
    return xy, err


def test_a_lost_truck_says_how_each_fresh_map_sits_on_the_one_before(lost_drive):
    out, truth = lost_drive
    why = [r.reseed for r in out if r.reseed]
    assert why[0] == "lost" and why[-1] == "merged", why
    assert all(len(r.link) == 7 for r in out if r.reseed)
    assert not any(r.link for r in out if not r.reseed)
    # Each goal there and back again: the carry, then the merge's fit.
    moved = math.dist(truth[-1][:2], path_pose(T_LOST)[:2])
    assert moved > 2.5, "it drove while it was lost"
    for gxy in GOALS:
        xy, err = carried(out, gxy)
        e = math.dist(xy, gxy)
        assert e < 0.25, (gxy, e)
        assert e <= err, "the estimate owns up to the error"


def test_without_the_speed_the_carry_is_the_gyro_only_and_says_so():
    out, truth = run_lost_drive(False)
    assert [r.reseed for r in out if r.reseed][-1] == "merged"
    xy, err = carried(out, GOALS[0])
    e = math.dist(xy, GOALS[0])
    assert e > 1.0, "it drove metres it could not count"
    assert err >= e


def test_the_goal_rides_the_whole_drive_and_comes_back_exact(lost_drive):
    out, truth = lost_drive
    ego, said = em.EgoMap(), []
    g = gl.ClickGoal(say=lambda *a: said.append(a))
    grid = gr.Grid(size=64)
    first = True
    approx_seen = False
    for r in out:
        c = ego.after_step(grid, r)
        if first:
            g.poll(Clicks(click(2.0, 1.0)), c)
            first = False
        g.observe(c)
        if c.frame != 0 and g.goal is not None and g.goal[2] == c.frame:
            approx_seen = approx_seen or g.status()["approx"]
    assert approx_seen, "on the fresh maps it was there, approximate"
    assert g.goal == (2.0, 1.0, 0) and not g.status()["approx"]
    assert any("carried it onto" in a[2] for a in said)
    assert any("exactly" in a[2] for a in said)
    assert len(g.transfer_errors) == 1 and g.transfer_errors[0] < 0.25
