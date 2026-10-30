"""Grade models on the scorecard: the same drives, never trained on, every time.

    python tools/policy_scorecard.py DATA.npz --map CROPS_DIR MODEL.onnx [MODEL.onnx ...] \
        [--json OUT.json]

The sessions are tools/scorecard.txt, names only. policy_train.py keeps every
model off them unless told otherwise (--train-on-scorecard), so any two models
trained since can be put side by side on ticks neither has seen. A model
trained before the list existed may have seen them; its sidecar's
train_sessions says, and the table marks it "seen".

Each model is run as the car runs a map model: policy.PolicyModel, one decision
tick at a time in time order through its own ChunkEnsembler, then read_output.
A map model has no stop mask, so its stop is graded as it is. Graded are the
decision ticks my hand labelled outside a BAD mark (takeovers, on these
drives). What is measured:

  dir_acc / dir_bal   gear against my gear, and the mean of the three
                      classes' recalls
  stop_rec / stop_prec, rev_rec / rev_prec   per gear
  steer_mae, side     the steer the car would command (a per-gear model's,
                      the head of the gear it picked) against my stick, and
                      how often it is on my side (|label| > SIDE_BAND)
  stop_side           side on the ticks I steered standing still
  lock                across each of my forward<->reverse swaps where I
                      changed lock (both sides past LOCK_MIN): the share of
                      the first AFTER moving ticks the model steers on my new
                      side. lock_swap says how often I swapped at all
  mirror_fail         every graded tick run again on its mirror image (the
                      frame's columns, the IMU's lateral signs, the scan's
                      bins, the crop's columns, the goal's bearing), raw
                      offset 0: a fail is another gear, or a steer side that
                      did not swap. No goal; mirror_fail_goal with one
  goal_*              the same ticks with a hindsight goal HIND_S on (the
                      truck's own pose then), where there is one: gear,
                      side, and the no-goal decode on the same ticks beside
                      it (goal_gain_dir, goal_gain_side)
  goal_turn_any / goal_turn / goal_gap / goal_back_rev / goal_use   adherence: the same
                      ticks with a made-up goal GOAL_PROBE_M off left, then
                      right, then behind (goal_adherence). A hindsight goal
                      is where I went anyway, so goal_dir can be passed
                      without ever reading the goal; these cannot
  behind_*            the goal ticks whose goal is behind the truck
                      (cos < 0): gear and reverse recall
  onset_dir / onset_side   the first ONSET ticks of each takeover, where the
                      model was wrong enough that I took over, mean per
                      takeover
  map_use             how far the steer moves (mean |change|, raw offset 0)
                      with the crop's planes blanked and present left on.
                      0 is a map branch that does nothing: every map model
                      up to goal2 had one (policy_train.MapBranch)
  lowband_use         the same for the lower band (column_profile, "the
                      lower band"), every column of it blind; and
                      lowband_use_dir, the share of ticks whose gear changes

With --tips TIPS.npz, a dataset of drives that went over (built like any
other, never trained on), each model is also run through those whole drives
and graded on the second before each tip (tip_scores: tip_go, tip_thr,
calm_go). Those are the car's own ticks, not my labels.

What this cannot say: whether the car drives well. These are my own labels on
moments I took over; the car's feel is mine to judge on a drive.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crawler.autopilot import column_profile as cp  # noqa: E402
from crawler.autopilot import policy as pl  # noqa: E402
from crawler.autopilot import policy_goal as pg  # noqa: E402
from crawler.autopilot import policy_imu as pimu  # noqa: E402
from crawler.autopilot import policy_lidar as pli  # noqa: E402
from crawler.mapping import egomap as em  # noqa: E402
import policy_eval as pe  # noqa: E402
import policy_train as pt  # noqa: E402

FWD, REV, STOP = (pl.ACTIONS.index(a) for a in (pl.FORWARD, pl.REVERSE, pl.STOP))
SIDE_BAND = 0.1
LOCK_MIN = 0.3
AFTER = 5
ONSET = 10
HIND_S = 8.0
TAKEOVER = "takeover"
# goal_adherence's made-up goal: as far along as the planner puts the point
# it feeds the model (pathgoal.LOOKAHEAD_M), and this far off either side.
GOAL_PROBE_M = 1.5
GOAL_PROBE_DEG = 60.0


def side(s, band=SIDE_BAND):
    s = np.asarray(s, float)
    return np.where(s > band, 1, np.where(s < -band, -1, 0))


def scorecard_ticks(data, names):
    """Decision ticks of the scorecard's sessions, session then time."""
    on = pt.scorecard_mask(data, names) & np.asarray(data["decision"], bool)
    ticks = np.flatnonzero(on)
    return ticks[np.lexsort((data["t_ns"][ticks], data["session"][ticks]))]


def inputs(model, data, i, goal=None, mirrored=False, blank_map=False, blank_lowband=False):
    """Tick i's inputs as the car feeds them, the mirror image if asked, or
    with the crop's planes all zero and its present flag as it was, or with
    the lower band every column blind (column_profile.lowband_blank)."""
    x = pe.x_input(model, data, i)
    if blank_lowband and getattr(model, "lowband", None) is not None:
        x[0, cp.PROFILE_FEATURES:] = cp.lowband_blank(model.lowband)
    prev = np.zeros((1, pl.PREV_SIZE), np.float32)
    mode = None
    if model.has_mode:
        mode = pl.encode_mode(int(data["mode_in"][i]) == pl.STEER_MODES.index(pl.SPLIT))
    extra = {}
    if model.uses_imu:
        extra["imu"] = pe.imu_input(model, data, i)
    if model.uses_lidar:
        extra["lidar"] = pe.lidar_input(data, i)
    if model.uses_map:
        extra["map"] = pe.map_input(data, i)
        if blank_map:
            extra["map"][:, :em.PRESENT] = 0.0
    if model.uses_goal:
        extra["goal"] = (np.asarray(goal, np.float32)[None] if goal is not None
                         else pg.missing())
    if mirrored:
        x = x[..., ::-1].copy()
        if "imu" in extra:
            extra["imu"] = (extra["imu"] * pimu.mirror_of(model.imu_yaw)).astype(np.float32)
        if "lidar" in extra:
            extra["lidar"] = pli.mirror(extra["lidar"]).astype(np.float32)
        if "map" in extra:
            extra["map"] = em.mirror(extra["map"]).copy()
        if "goal" in extra:
            extra["goal"] = (extra["goal"] * pg.MIRROR).astype(np.float32)
    return x, prev, mode, extra


def decode(model, data, ticks, goals=None, mirrored=False, blend=True, blank_map=False,
           blank_lowband=False, with_throttle=False):
    """(action, steer) per tick: the car's read_output, after the chunk blend
    (blend) or off the raw answer's offset 0. A chain starts over on a new
    session and wherever the ensembler's own link rule says. with_throttle:
    (action, steer, throttle), the throttle the fraction of the cap it asks
    for, 0 on a stop or no answer."""
    act = np.full(len(ticks), STOP, np.int64)
    steer = np.zeros(len(ticks))
    thr = np.zeros(len(ticks))
    sess, ens = None, None
    for j, i in enumerate(ticks):
        if data["session"][i] != sess:
            sess = data["session"][i]
            ens = pl.ChunkEnsembler(model.chunk, model.ensemble_m, model.ensemble_link_s)
        x, prev, mode, extra = inputs(model, data, i,
                                      None if goals is None else goals[j], mirrored,
                                      blank_map, blank_lowband)
        out = model.run(x, prev, mode, **extra)
        if blend:
            out = ens.apply(out, data["t_ns"][i] / 1e9,
                            int(data["mode_in"][i]) if model.has_mode else None)
        decided, _ = pl.read_output(out)
        if decided is None:
            continue
        act[j] = pl.ACTIONS.index(decided[0])
        steer[j] = decided[2]
        thr[j] = 0.0 if act[j] == STOP else decided[1]
    return (act, steer, thr) if with_throttle else (act, steer)


def tip_scores(model, data, tip_deg=pt.TIP_DEG, lead_s=pt.TIP_LEAD_S):
    """The model run through drives that tipped (--tips), every decision
    tick but the tipped ones, in time order as the car runs it:

      tip_crossings  how many times a session went past tip_deg
      tip_ticks      the decision ticks in their run-ups (policy_train.tip_marks)
      tip_go         the share of those it keeps a moving gear on
      tip_thr        the mean throttle it asks for there, 0 on a stop
      calm_go        the share it drives on the rest, where the car was
                     moving: a model that only ever stops is not safe, it
                     is parked, and this says so

    These are the car's drives, not my labels: nothing says what the right
    gear in a run-up was, only that the one taken went over."""
    tipped, runup = pt.tip_marks(data, tip_deg, lead_s)
    dec = np.asarray(data["decision"], bool) & ~tipped
    ticks = np.flatnonzero(dec)
    ticks = ticks[np.lexsort((data["t_ns"][ticks], data["session"][ticks]))]
    act, _, thr = decode(model, data, ticks, with_throttle=True)
    run = runup[ticks]
    calm = ~run & (np.asarray(data["action"])[ticks] != STOP)
    go = act != STOP
    sess, t = np.asarray(data["session"]), np.asarray(data["t_ns"])
    tips = np.flatnonzero(tipped)
    starts = sum(1 for a, b in zip(tips[:-1], tips[1:]) if sess[a] != sess[b] or b != a + 1)
    return {"tip_crossings": int(len(tips) > 0) + starts,
            "tip_ticks": int(run.sum()),
            "tip_go": _rate(go, run),
            "tip_thr": float(thr[run].mean()) if run.any() else float("nan"),
            "calm_go": _rate(go, calm),
            "calm_ticks": int(calm.sum()),
            "tip_first_s": float(t[tips[0]] / 1e9) if len(tips) else float("nan")}


def probe_goals(n, bearing_deg, dist_m=None):
    """n copies of a goal dist_m off at bearing_deg (CCW, + is left)."""
    dist_m = GOAL_PROBE_M if dist_m is None else dist_m
    b = np.radians(bearing_deg)
    return pg.encode_many(np.full(n, dist_m * np.cos(b)), np.full(n, dist_m * np.sin(b)),
                          np.ones(n, bool))


def nose_left(action, steer):
    """How hard the command turns the truck's nose left: -steer going
    forward, +steer reversing, 0 stopped (negative steer is left)."""
    gear = np.where(action == FWD, 1.0, np.where(action == REV, -1.0, 0.0))
    return -gear * np.asarray(steer, float)


def goal_adherence(model, data, ticks):
    """Whether the model goes where it is told, off the same ticks with a
    made-up goal GOAL_PROBE_M away, raw offset 0 (no chunk blend):

      goal_turn_any  on every tick, the share where the nose turns more
                     than SIDE_BAND further left for the left goal than for
                     the right one. The nose turns with the gear: forward,
                     negative steer turns it left; reverse, positive steer
                     does; a stop does not turn it. So backing round toward
                     a goal counts, and driving away from it does not
      goal_turn      on the ticks it drives forward with the goal
                     GOAL_PROBE_DEG left and with it as far right, the share
                     whose steer is more than SIDE_BAND further right for the
                     right one (negative steer is left)
      goal_gap       the mean of that difference, right minus left; 0 is a
                     model that never looks at the goal
      goal_back_rev  how much more often it reverses with the goal straight
                     behind than straight ahead
      goal_use       the mean |steer change| the left and right goals make
                     against no goal at all, as map_use is for the map

    No label in it: a hindsight goal is where I went anyway, and the rest of
    the inputs already say that, so this is the only grade of the goal
    that cannot be passed without it."""
    n = len(ticks)
    out = {"goal_probe_ticks": int(n)}
    if n == 0:
        return out
    run = lambda g: decode(model, data, ticks, g, blend=False)  # noqa: E731
    la, ls = run(probe_goals(n, GOAL_PROBE_DEG))
    ra, rs = run(probe_goals(n, -GOAL_PROBE_DEG))
    fa, _ = run(probe_goals(n, 0.0))
    ba, _ = run(probe_goals(n, 180.0))
    na, ns = run(np.zeros((n, pg.SIZE), np.float32))
    both = (la == FWD) & (ra == FWD)
    gap = rs - ls
    out["goal_turn_ticks"] = int(both.sum())
    out["goal_turn"] = _rate(gap > SIDE_BAND, both)
    out["goal_turn_any"] = float(np.mean(nose_left(la, ls) - nose_left(ra, rs) > SIDE_BAND))
    out["goal_gap"] = float(gap[both].mean()) if both.any() else float("nan")
    out["goal_back_rev"] = float(np.mean(ba == REV) - np.mean(fa == REV))
    out["goal_use"] = float((np.abs(ls - ns).mean() + np.abs(rs - ns).mean()) / 2)
    return out


def hind_goals(data, ticks, k_s=HIND_S):
    """The hindsight goal k_s on for each tick, none where there is none."""
    if "goal_poses" not in data:
        return np.zeros((len(ticks), pg.SIZE), np.float32)
    on = np.asarray(data["map_on"], bool)[ticks]
    return pt.goal_inputs(data, ticks, on, np.random.default_rng(0), no_goal_share=0.0,
                          k_s=k_s)


def _rate(hit, mask):
    n = int(np.sum(mask))
    return float(np.sum(hit & mask) / n) if n else float("nan")


def gear_scores(pred, label, ok):
    out = {"dir_acc": _rate(pred == label, ok)}
    recalls = [_rate(pred == c, ok & (label == c)) for c in (FWD, REV, STOP)]
    out["dir_bal"] = float(np.nanmean(recalls))
    for name, c in (("stop", STOP), ("rev", REV)):
        out[f"{name}_rec"] = _rate(pred == c, ok & (label == c))
        out[f"{name}_prec"] = _rate(label == c, ok & (pred == c))
    return out


def reversals(data, ticks, ok):
    """[(before, after)] index arrays into ticks: my forward<->reverse swaps,
    through at most 5 stop ticks, AFTER moving ticks each side, one session."""
    a = data["action"][ticks]
    s = data["session"][ticks]
    mv = np.flatnonzero(ok & (a != STOP))
    out = []
    for x, y in zip(mv, mv[1:]):
        if a[x] == a[y] or s[x] != s[y] or y - x > 6:
            continue
        before = mv[(mv <= x) & (mv > x - 2 * AFTER - 2)][-AFTER:]
        after = mv[mv >= y][:AFTER]
        if (len(before) < 3 or len(after) < 3 or (a[before] != a[x]).any()
                or (a[after] != a[y]).any() or (s[after] != s[x]).any()):
            continue
        out.append((before, after))
    return out


def takeover_onsets(data, ticks, ok):
    """Index arrays into ticks: the first ONSET labelled ticks of each run of
    takeover ticks (gap under policy_dataset.PREV_GAP_S)."""
    src = pt.sources_of(data)[ticks] == TAKEOVER
    t = data["t_ns"][ticks] / 1e9
    s = data["session"][ticks]
    out, cur = [], []
    for j in range(len(ticks)):
        if src[j] and cur and s[j] == s[cur[-1]] and t[j] - t[cur[-1]] <= 0.25:
            cur.append(j)
            continue
        if cur:
            out.append(np.array(cur))
        cur = [j] if src[j] else []
    if cur:
        out.append(np.array(cur))
    return [o[ok[o]][:ONSET] for o in out if ok[o].any()]


def score(model, data, ticks):
    s_ok, t_ok = pe.label_masks(data, "pad")
    bad = pe.bad_marked(data)[ticks]
    sk, tk = s_ok[ticks] & ~bad, t_ok[ticks] & ~bad
    label, lsteer = data["action"][ticks].astype(np.int64), data["steer"][ticks]
    act, steer = decode(model, data, ticks)
    out = gear_scores(act, label, tk)
    out["steer_mae"] = float(np.abs(steer - lsteer)[sk].mean())
    turn = sk & (np.abs(lsteer) > SIDE_BAND)
    out["side"] = _rate(side(steer) == side(lsteer), turn)
    out["stop_side"] = _rate(side(steer) == side(lsteer), turn & (label == STOP))
    # Across my reversals.
    both = sk & tk
    swaps, held, hits, n = 0, 0, 0, 0
    for before, after in reversals(data, ticks, both):
        lb, la = lsteer[before].mean(), lsteer[after].mean()
        if abs(lb) < LOCK_MIN or abs(la) < LOCK_MIN:
            continue
        held += 1
        if np.sign(lb) == np.sign(la):
            continue
        swaps += 1
        hits += int((np.sign(steer[after]) == np.sign(la)).sum())
        n += len(after)
    out["lock_swap"] = swaps / held if held else float("nan")
    out["lock"] = hits / n if n else float("nan")
    out["reversals"] = swaps
    # The mirror, raw offset 0.
    graded = sk | tk
    g = np.flatnonzero(graded)
    ra, rs = decode(model, data, ticks[g], blend=False)
    ma, ms = decode(model, data, ticks[g], mirrored=True, blend=False)
    out["mirror_fail"] = float(np.mean((ra != ma) | (side(rs) != -side(ms))))
    if model.uses_map:
        mapped = np.asarray(data["map_on"], bool)[ticks[g]]
        _, bs = decode(model, data, ticks[g], blend=False, blank_map=True)
        out["map_use"] = float(np.abs(rs - bs)[mapped].mean()) if mapped.any() else float("nan")
    if getattr(model, "lowband", None) is not None:
        la, ls = decode(model, data, ticks[g], blend=False, blank_lowband=True)
        out["lowband_use"] = float(np.abs(rs - ls).mean())
        out["lowband_use_dir"] = float(np.mean(ra != la))
    # Goals.
    if model.uses_goal:
        goals = hind_goals(data, ticks)
        has = goals[:, pg.PRESENT] > 0
        ga, gs = decode(model, data, ticks, goals)
        gk, gt = has & sk, has & tk
        out["goal_ticks"] = int((has & graded).sum())
        out["goal_dir"] = _rate(ga == label, gt)
        out["goal_side"] = _rate(side(gs) == side(lsteer), gk & turn)
        out["goal_gain_dir"] = out["goal_dir"] - _rate(act == label, gt)
        out["goal_gain_side"] = out["goal_side"] - _rate(side(steer) == side(lsteer), gk & turn)
        behind = has & (goals[:, pg.COS] < 0)
        out["behind_ticks"] = int((behind & graded).sum())
        out["behind_dir"] = _rate(ga == label, behind & tk)
        out["behind_rev_rec"] = _rate(ga == REV, behind & tk & (label == REV))
        gg = np.flatnonzero(graded & has)
        ra, rs = decode(model, data, ticks[gg], goals[gg], blend=False)
        ma, ms = decode(model, data, ticks[gg], goals[gg], mirrored=True, blend=False)
        out["mirror_fail_goal"] = float(np.mean((ra != ma) | (side(rs) != -side(ms))))
        # Adherence, on the graded ticks the car would feed a goal on at
        # all: a map model's only where its crop is there.
        shown = (np.asarray(data["map_on"], bool)[ticks[g]] if model.uses_map
                 else np.ones(len(g), bool))
        out.update(goal_adherence(model, data, ticks[g][shown]))
    # Takeover onsets.
    on = takeover_onsets(data, ticks, graded)
    out["takeovers"] = len(on)
    out["onset_dir"] = float(np.nanmean([_rate(act[o] == label[o], tk[o]) for o in on]))
    out["onset_side"] = float(np.nanmean([_rate(side(steer[o]) == side(lsteer[o]),
                                                turn[o]) for o in on]))
    out["graded"] = int(graded.sum())
    return out


KEYS = ("dir_acc", "dir_bal", "stop_rec", "stop_prec", "rev_rec", "rev_prec", "steer_mae",
        "side", "stop_side", "lock", "mirror_fail", "mirror_fail_goal", "goal_dir",
        "goal_side", "goal_gain_side", "goal_turn_any", "goal_turn", "goal_gap", "goal_back_rev", "goal_use",
        "behind_dir", "behind_rev_rec", "onset_dir",
        "onset_side", "map_use", "lowband_use", "lowband_use_dir")


def table(rows):
    """[(name, scores)] -> a markdown table, one model a row."""
    head = "| model | " + " | ".join(KEYS) + " |"
    lines = [head, "|" + "---|" * (len(KEYS) + 1)]
    for name, s in rows:
        lines.append(f"| {name} | " + " | ".join(
            f"{s[k]:.3f}" if isinstance(s.get(k), float) and np.isfinite(s[k]) else "-"
            for k in KEYS) + " |")
    return "\n".join(lines)


def load(path, crops, x_cache=None):
    data = pe.load(path)
    pt.load_maps(data, crops)
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--map", required=True, metavar="CROPS_DIR",
                    help="the crops tools/map_crops.py replayed, with their poses")
    ap.add_argument("--json", default=None, help="write every number here too")
    ap.add_argument("--tips", default=None, metavar="TIPS.npz",
                    help="also run each model through these drives that tipped "
                         "(tip_scores); never trained on, as the scorecard's")
    ap.add_argument("--tips-map", default=None, metavar="CROPS_DIR",
                    help="the crops for --tips, if not in --map")
    args = ap.parse_args(argv)
    names = pt.scorecard_sessions()
    if not names:
        print(f"no scorecard at {pt.SCORECARD_FILE}")
        return 1
    data = load(args.data, args.map)
    missing = [n for n in names if n not in data["meta"]["sessions"]]
    if missing:
        print(f"{args.data} has no {missing}: build the dataset with them in")
        return 1
    ticks = scorecard_ticks(data, names)
    tips = load(args.tips, args.tips_map or args.map) if args.tips else None
    rows = []
    for path in args.models:
        model = pl.PolicyModel(path)
        seen = sorted(set(names) & set(model.meta.get("train_sessions", [])))
        s = score(model, data, ticks)
        if tips is not None:
            s.update(tip_scores(model, tips))
            s["tips_seen"] = sorted(set(tips["meta"]["sessions"])
                                    & set(model.meta.get("train_sessions", [])))
        s["seen"] = seen
        label = os.path.splitext(os.path.basename(path))[0] + (" (seen)" if seen else "")
        rows.append((label, s))
        print(f"{label}: {json.dumps({k: s[k] for k in s if not k.endswith('seen')})}",
              flush=True)
    print(table(rows))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"sessions": names, "models": dict(rows)}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
