"""The goal the car was actually fed, per status message, off a drive's own recording.

    python tools/fed_goals.py OUT_DIR RUN.mcap [RUN.mcap ...]

A hindsight goal (crawler/autopilot/policy_goal.py) is where my hand took the
car anyway, so it says nothing the frame and the map do not. On a patrol or a
clicked goal the car was told where to go, and that point is in the
recording: /path/status's aim (the planner's subgoal, or the goal itself
inside the lookahead) when the planner ran, else /goal/status's clicked
point. Each is put in the truck's frame off the newest tracking
/slam/status pose -- the same live map frame both are on -- so the vector is
the one the model was fed, give or take the crop's own pose.

Writes OUT_DIR/<run>.npz, one row per status message:

  log_t        the message's log time, seconds: what a dataset tick pairs on
  ahead, left  metres, truck frame (left > 0)
  ok           a goal was fed and a fresh tracking pose was there to put it
               in the truck's frame
  source       "path" or "goal"
  layout       LAYOUT

policy_train.py --fed-goals OUT_DIR pairs each tick with the newest row at or
before it, no more than STALE_S old, and feeds that goal in place of the
drawn one where ok.

Needs the `mcap` package.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler.autopilot import policy_goal as pg  # noqa: E402

LAYOUT = "fed1"
# A status message is paired with a pose no older than this, and a tick with
# a status no older than STALE_S: both topics run at a few Hz.
POSE_STALE_S = 0.3
STALE_S = 0.5
TOPICS = ("/slam/status", "/path/status", "/goal/status")


def fed_points(slam, path, goal):
    """slam (N, 5): t, x, y, yaw (radians), tracking (0/1).
    path (M, 4): t, fed (0/1), aim_x, aim_y -- empty when the planner did not run.
    goal (K, 4): t, set and on the frame the truck is on (0/1), x, y.
    -> dict log_t, ahead, left, ok, source over the path rows when there are
    any (the planner's aim is what the model got), else the goal rows."""
    slam = np.asarray(slam, float).reshape(-1, 5)
    path = np.asarray(path, float).reshape(-1, 4)
    goal = np.asarray(goal, float).reshape(-1, 4)
    src, rows = ("path", path) if len(path) else ("goal", goal)
    t = rows[:, 0]
    n = len(t)
    ok = rows[:, 1] > 0
    i = np.searchsorted(slam[:, 0], t, side="right") - 1 if len(slam) else np.full(n, -1)
    has = i >= 0
    k = np.where(has, i, 0)
    if len(slam):
        has &= (t - slam[k, 0] <= POSE_STALE_S) & (slam[k, 4] > 0)
    ok &= has
    ahead = np.zeros(n)
    left = np.zeros(n)
    if ok.any():
        a, l = pg.relative((slam[k, 1], slam[k, 2], slam[k, 3]), rows[:, 2], rows[:, 3])
        ahead, left = np.where(ok, a, 0.0), np.where(ok, l, 0.0)
    return {"log_t": t, "ahead": ahead, "left": left, "ok": ok, "source": np.array(src),
            "layout": np.array(LAYOUT)}


def pair(fed_t, tick_t, stale_s=STALE_S):
    """For each tick time, the index of the newest fed row at or before it,
    no more than stale_s old, else -1."""
    fed_t = np.asarray(fed_t, float)
    tick_t = np.asarray(tick_t, float)
    j = np.searchsorted(fed_t, tick_t, side="right") - 1
    good = j >= 0
    jj = np.where(good, j, 0)
    good &= (tick_t - fed_t[jj]) <= stale_s if len(fed_t) else False
    return np.where(good, j, -1)


def read(path):
    """One MCAP -> fed_points' three arrays."""
    from mcap.reader import make_reader
    slam, pth, goal = [], [], []
    with open(path, "rb") as f:
        try:
            for _s, ch, m in make_reader(f).iter_messages(topics=list(TOPICS)):
                try:
                    d = json.loads(m.data)
                except Exception:
                    continue
                t = m.log_time / 1e9
                if ch.topic == "/slam/status":
                    slam.append((t, d.get("x") or 0.0, d.get("y") or 0.0,
                                 math.radians(d.get("yaw_deg") or 0.0),
                                 float(d.get("state") == "tracking")))
                elif ch.topic == "/path/status":
                    pth.append((t, float(d.get("fed", "none") != "none"),
                                d.get("aim_x") or 0.0, d.get("aim_y") or 0.0))
                else:
                    on = bool(d.get("goal_set")) and d.get("frame") == d.get("frame_now")
                    goal.append((t, float(on), d.get("x") or 0.0, d.get("y") or 0.0))
        except Exception:
            pass  # a file cut off mid-chunk: what was read stands
    return slam, pth, goal


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    for path in args.runs:
        name = os.path.basename(path).rsplit(".", 1)[0]
        out = fed_points(*read(path))
        if not out["ok"].any():
            print(f"{name}: no goal fed")
            continue
        np.savez_compressed(os.path.join(args.out, name + ".npz"), **out)
        print(f"{name}: {int(out['ok'].sum())} of {len(out['ok'])} {out['source']} statuses fed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
