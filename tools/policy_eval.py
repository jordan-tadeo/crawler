"""Grade an exported policy against a dataset, and write a report a person can read.

    python tools/policy_eval.py DATA.npz MODEL.onnx [--out REPORT.md] [--labels any]

Runs the .onnx through policy.PolicyModel -- the class the car loads -- one
tick at a time with policy.encode_input(), so what is graded is what would
run. Needs numpy and onnxruntime only, no torch.

Per session: the direction confusion matrix, throttle and steering error,
and the baselines that make those numbers mean something:

  direction   always-forward, and the gradient rule's own verdict
  throttle    the median training magnitude, on every moving tick
  steering    the gradient rule's steer (read_the_gradient(), as recorded on
              /footprint or re-run on the legacy frame), and dead straight

Then every /marks window: what was commanded across it, what the model would
have said on the same frames, and what the rule said. That includes ticks
inside a recover, which the model is never asked about on the car -- the
window is about the moment, not the decision count.

What this cannot grade: the safety veto (the floor and the stuck detector
run on the raw frame and the IMU, which a dataset tick does not keep), and
the model's own previous action -- a model trained with --prev is fed the
recorded previous action here, which is kinder than the car will be.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import policy as pl  # noqa: E402
import policy_imu as pimu  # noqa: E402

FWD = pl.ACTIONS.index(pl.FORWARD)
STOP = pl.ACTIONS.index(pl.STOP)


def load(path):
    z = np.load(path, allow_pickle=False)
    data = {k: z[k] for k in z.files if k != "meta"}
    data["meta"] = json.loads(str(z["meta"]))
    return data


def label_masks(data, label_mode):
    """(steer_ok, throttle_ok) per tick: which groups are a label.

    pad: only a group my hand had. any: every group, the autopilot's too --
    except a group a learned policy drove, which is never a label: cloning a
    model's own output teaches the next model nothing but its mistakes.
    """
    steer = np.asarray(data["steer_label"], dtype=bool)
    throttle = np.asarray(data["throttle_label"], dtype=bool)
    if label_mode != "any":
        return steer, throttle
    policy = np.asarray(data.get("policy_drove", np.zeros(len(steer), bool)), dtype=bool)
    return steer | ~policy, throttle | ~policy


def imu_input(model, data, i):
    """Tick i's IMU input for a model that takes one: the dataset array for
    its list (policy_imu.DATASET_KEY by imu_yaw), tilt zeroed unless the
    model was trained with it -- as policy_imu.ImuFeatures(tilt=..., yaw=...)
    makes it on the car."""
    feats = data[pimu.DATASET_KEY[bool(getattr(model, "imu_yaw", True))]][i]
    if not getattr(model, "imu_tilt", False):
        feats = pimu.without_tilt(feats)
    return pl.encode_imu(feats)


def _run(model, data, i, split=None):
    """One tick through the model. A mode head is fed `split`, or the
    recorded mode going into the tick when that is None."""
    x = pl.encode_input(model.representation, data["frames"][data["frame"][i]])
    prev = (data["prev"][i:i + 1] if model.uses_prev
            else np.zeros((1, pl.PREV_SIZE), dtype=np.float32))
    mode = None
    if getattr(model, "has_mode", False):
        if split is None:
            split = int(data["mode_in"][i]) == pl.STEER_MODES.index(pl.SPLIT)
        mode = pl.encode_mode(split)
    if getattr(model, "uses_imu", False):
        return model.run(x, prev, mode, imu=imu_input(model, data, i))
    if mode is None:
        return model.run(x, prev)
    return model.run(x, prev, mode)


def predict(model, data, ticks):
    """(actions, magnitudes, steers) for these tick indices."""
    actions, mags, steers = [], [], []
    for i in ticks:
        decided, _ = pl.read_output(_run(model, data, i))
        if decided is None:
            actions.append(STOP)
            mags.append(0.0)
            steers.append(0.0)
            continue
        action, mag, steer, _ = decided
        actions.append(pl.ACTIONS.index(action))
        mags.append(mag)
        steers.append(steer)
    return np.array(actions, dtype=np.int64), np.array(mags), np.array(steers)


SPLIT_I = pl.STEER_MODES.index(pl.SPLIT)
# A replayed latch switch matches one of my toggles if it lands this close to
# it. The label leads a toggle by MODE_LEAD_S and the latch needs its dwell,
# so a head that learned exactly the label switches on the toggle; a second
# either side is "it switched for the same reason I did".
LATCH_MATCH_S = 1.0


def predict_mode(model, data, ticks):
    """p(split) per tick, fed the recorded mode going in. nan on a bad answer."""
    out = []
    for i in ticks:
        p, _ = pl.read_mode(_run(model, data, i))
        out.append(np.nan if p is None else p)
    return np.array(out, dtype=np.float64)


def latch_replay(model, data, ticks):
    """The runtime's ModeLatch over these ticks, fed its own mode, as the car
    runs it. Starts each contiguous run (PREV_GAP_S) at the recorded mode,
    as auto engaging does. [(t_ns, new split)] for every switch it made."""
    switch = getattr(model, "mode_switch", None) or {}
    latch = pl.ModeLatch(switch.get("threshold", pl.MODE_THRESHOLD),
                         switch.get("dwell_s", pl.MODE_DWELL_S),
                         switch.get("min_interval_s", pl.MODE_MIN_INTERVAL_S))
    made, last_t = [], None
    for i in ticks:
        t_ns = int(data["t_ns"][i])
        now = t_ns / 1e9
        if last_t is None or (t_ns - last_t) / 1e9 > 0.25:
            latch.set(int(data["mode_in"][i]) == SPLIT_I, now)
        last_t = t_ns
        p, _ = pl.read_mode(_run(model, data, i, latch.split))
        if p is not None and latch.update(p, now):
            made.append((t_ns, latch.split))
    return made


def mode_scores(model, data, ticks, s_ok):
    """The mode head against always-stay, open loop and through the latch."""
    if not getattr(model, "has_mode", False) or "steer_mode" not in data:
        return {}
    graded = ticks[s_ok]
    truth = data["steer_mode"][graded].astype(np.int64)
    stay = data["mode_in"][graded].astype(np.int64)
    p = predict_mode(model, data, graded)
    pred = np.where(np.isfinite(p) & (p > 0.5), SPLIT_I, 1 - SPLIT_I)
    switch = truth != stay                 # the lead ticks: I wanted the other mode
    window = data["mode_switch"][graded]

    names = data["meta"]["sessions"]
    t_lo, t_hi = (int(data["t_ns"][ticks].min()), int(data["t_ns"][ticks].max())) \
        if len(ticks) else (0, -1)
    sessions = set(int(s) for s in data["session"][ticks])
    toggles = [t for sid in sessions
               for t in data["meta"].get("toggles", {}).get(names[sid], [])
               if t_lo <= t <= t_hi]
    made = latch_replay(model, data, ticks)
    matched = sum(1 for t in toggles
                  if any(abs(m - t) / 1e9 <= LATCH_MATCH_S for m, _ in made))
    false = sum(1 for m, _ in made
                if not any(abs(m - t) / 1e9 <= LATCH_MATCH_S for t in toggles))

    def acc(a, b):
        return float(np.mean(a == b)) if len(a) else float("nan")

    return {
        "mode_ticks": int(len(graded)),
        "mode_accuracy": acc(pred, truth),
        "mode_stay_accuracy": acc(stay, truth),
        "mode_split_ticks": int((truth == SPLIT_I).sum()),
        "mode_switch_ticks": int(switch.sum()),
        "mode_switch_recall": acc(pred[switch], truth[switch]),
        "mode_window_ticks": int(window.sum()),
        "mode_window_accuracy": acc(pred[window], truth[window]),
        "mode_window_stay_accuracy": acc(stay[window], truth[window]),
        "mode_toggles": len(toggles),
        "mode_latch_switches": len(made),
        "mode_latch_toggles_matched": matched,
        "mode_latch_false_switches": false,
    }


def confusion(truth, pred):
    m = np.zeros((len(pl.ACTIONS), len(pl.ACTIONS)), dtype=np.int64)
    for t, p in zip(truth, pred):
        m[t, p] += 1
    return m


def balanced_accuracy(m):
    rows = m.sum(1)
    present = rows > 0
    return float(np.mean(np.diag(m)[present] / rows[present])) if present.any() else float("nan")


def _mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b)))) if len(a) else float("nan")


def score_ticks(model, data, ticks, label_mode, median_mag):
    """The numbers for one set of decision ticks."""
    truth = data["action"][ticks].astype(np.int64)
    pred, mag, steer = predict(model, data, ticks)
    steer_all, throttle_all = label_masks(data, label_mode)
    t_ok = throttle_all[ticks]
    s_ok = steer_all[ticks]

    m = confusion(truth[t_ok], pred[t_ok])
    base = confusion(truth[t_ok], np.full(int(t_ok.sum()), FWD))
    rule_known = data["rule_action"][ticks] >= 0
    rule = confusion(truth[t_ok & rule_known], data["rule_action"][ticks][t_ok & rule_known])

    moving = t_ok & (truth != STOP)
    true_mag = data["magnitude"][ticks]
    rule_steer = data["rule_steer"][ticks]
    s_rule = s_ok & np.isfinite(rule_steer)
    true_steer = data["steer"][ticks]
    n = int(t_ok.sum())
    return {
        "ticks": int(len(ticks)),
        "direction_ticks": n,
        "direction_accuracy": float(np.trace(m) / n) if n else float("nan"),
        "direction_balanced_accuracy": balanced_accuracy(m),
        "always_forward_accuracy": float(np.trace(base) / n) if n else float("nan"),
        "always_forward_balanced_accuracy": balanced_accuracy(base),
        "rule_accuracy": (float(np.trace(rule) / rule.sum()) if rule.sum() else float("nan")),
        "rule_balanced_accuracy": balanced_accuracy(rule),
        "confusion": m.tolist(),
        "throttle_ticks": int(moving.sum()),
        "throttle_mae": _mae(mag[moving], true_mag[moving]),
        "throttle_median_baseline_mae": _mae(np.full(int(moving.sum()), median_mag),
                                             true_mag[moving]),
        "steer_ticks": int(s_ok.sum()),
        "steer_mae": _mae(steer[s_ok], true_steer[s_ok]),
        "steer_rule_mae": _mae(rule_steer[s_rule], true_steer[s_rule]),
        "steer_model_mae_where_rule_known": _mae(steer[s_rule], true_steer[s_rule]),
        "steer_straight_mae": _mae(np.zeros(int(s_ok.sum())), true_steer[s_ok]),
        **mode_scores(model, data, np.asarray(ticks), s_ok),
    }


MARK_GOOD, MARK_BAD = 1, -1  # policy_dataset.MARK_GOOD / MARK_BAD


def marks_of(data):
    """The per-tick /marks flags, zeros for a dataset built before there were any."""
    return data.get("mark", np.zeros(len(data["t_ns"]), dtype=np.int8))


def bad_marked(data):
    """Ticks a BAD mark says were bad driving: out of training and out of val."""
    return marks_of(data) == MARK_BAD


def median_training_magnitude(data, train_mask):
    moving = train_mask & (data["action"] != STOP)
    return float(np.median(data["magnitude"][moving])) if moving.any() else 1.0


def score(model, data, session_ids, label_mode, train_mask):
    """Pooled val numbers over these sessions, plus one block per session."""
    median_mag = median_training_magnitude(data, train_mask)
    graded = data["decision"] & ~bad_marked(data)
    pooled = np.flatnonzero(np.isin(data["session"], session_ids) & graded)
    out = score_ticks(model, data, pooled, label_mode, median_mag)
    out["median_magnitude"] = median_mag
    out["bad_marked_left_out"] = int((np.isin(data["session"], session_ids)
                                      & data["decision"] & bad_marked(data)).sum())
    out["per_session"] = {}
    names = data["meta"]["sessions"]
    for sid in session_ids:
        ticks = np.flatnonzero((data["session"] == sid) & graded)
        out["per_session"][names[sid]] = score_ticks(model, data, ticks, label_mode, median_mag)
    return out


def format_scores(s):
    lines = [
        f"direction ({s['direction_ticks']} ticks): accuracy {s['direction_accuracy']:.3f} "
        f"(always-forward {s['always_forward_accuracy']:.3f}, rule {s['rule_accuracy']:.3f}); "
        f"balanced {s['direction_balanced_accuracy']:.3f} "
        f"(always-forward {s['always_forward_balanced_accuracy']:.3f}, "
        f"rule {s['rule_balanced_accuracy']:.3f})",
        f"throttle magnitude MAE ({s['throttle_ticks']} moving ticks): {s['throttle_mae']:.3f} "
        f"(median {s['throttle_median_baseline_mae']:.3f})",
        f"steering MAE ({s['steer_ticks']} ticks, fraction of full lock): {s['steer_mae']:.3f} "
        f"(rule {s['steer_rule_mae']:.3f}, straight {s['steer_straight_mae']:.3f})",
        "confusion, rows truth / columns model, " + " ".join(pl.ACTIONS) + ": "
        + str(s["confusion"]),
    ]
    if "mode_ticks" in s:
        lines += [
            f"steering mode ({s['mode_ticks']} ticks, {s['mode_split_ticks']} split): accuracy "
            f"{s['mode_accuracy']:.3f} (always-stay {s['mode_stay_accuracy']:.3f}); "
            f"switch recall {s['mode_switch_recall']:.3f} over {s['mode_switch_ticks']} lead "
            f"ticks (always-stay 0); switch windows {s['mode_window_accuracy']:.3f} "
            f"(always-stay {s['mode_window_stay_accuracy']:.3f}, {s['mode_window_ticks']} ticks)",
            f"mode latch replay: {s['mode_latch_switches']} switches, "
            f"{s['mode_latch_toggles_matched']} of {s['mode_toggles']} toggles matched "
            f"within {LATCH_MATCH_S:.0f} s, {s['mode_latch_false_switches']} false",
        ]
    return "\n".join(lines)


def _counts(actions):
    c = np.bincount(np.asarray(actions, dtype=np.int64), minlength=len(pl.ACTIONS))
    return ", ".join(f"{a} {n}" for a, n in zip(pl.ACTIONS, c) if n) or "none"


def marks_section(model, data):
    lines = []
    names = data["meta"]["sessions"]
    for sid, name in enumerate(names):
        for t_ns, mark in data["meta"]["marks"].get(name, []):
            try:
                offset = t_ns / 1e9 - float(mark["mono_s"])
                lo = (float(mark["window_start_s"]) + offset) * 1e9
                hi = (float(mark["window_end_s"]) + offset) * 1e9
            except (KeyError, TypeError, ValueError):
                continue
            ticks = np.flatnonzero((data["session"] == sid)
                                   & (data["t_ns"] >= lo) & (data["t_ns"] <= hi))
            head = (f"- **{mark.get('kind', '?')}** ({mark.get('button', '')}) in {name} "
                    f"at {float(mark['mono_s']):.1f} s, mode {mark.get('mode', '?')}")
            if not len(ticks):
                lines.append(head + ": no kept ticks in the window")
                continue
            pred, mag, steer = predict(model, data, ticks)
            rule = data["rule_action"][ticks]
            lines.append(head + f", {len(ticks)} ticks")
            lines.append(f"  - commanded: {_counts(data['action'][ticks])}; "
                         f"mean steer {np.mean(data['steer'][ticks]):+.2f}")
            lines.append(f"  - model: {_counts(pred)}; mean steer {np.mean(steer):+.2f}, "
                         f"mean throttle {np.mean(mag):.2f} of cap")
            lines.append(f"  - rule: {_counts(rule[rule >= 0])}; "
                         f"mean steer {np.nanmean(data['rule_steer'][ticks]):+.2f}")
            seq = "".join(pl.ACTIONS[a][0].upper() for a in data["action"][ticks])
            mseq = "".join(pl.ACTIONS[a][0].upper() for a in pred)
            lines.append(f"  - tick by tick (F/R/S): commanded `{seq}`, model `{mseq}`")
    return lines or ["- no marks in these sessions"]


def report(model, data, label_mode, session_ids=None, train_mask=None):
    names = data["meta"]["sessions"]
    if session_ids is None:
        session_ids = list(range(len(names)))
    if train_mask is None:
        train_mask = np.zeros(len(data["t_ns"]), dtype=bool)
    s = score(model, data, session_ids, label_mode, train_mask)
    meta = model.meta
    out = [f"# Policy eval: {os.path.basename(model.path)}", "",
           f"Representation **{meta.get('representation')}**, {meta.get('parameters')} "
           f"parameters, input contract `{meta.get('input_contract')}`, labels "
           f"`{label_mode}`. Trained on {meta.get('train_sessions')}, "
           f"held out {meta.get('val_sessions')}.", "",
           "Direction is graded on decision ticks only (not the inside of a recover). "
           "Throttle is the fraction of that direction's cap, on moving ticks. Steering "
           "is a fraction of full lock (STEER_SCALE 0.7), so 0.10 is 0.07 of servo travel.", "",
           "## All graded sessions", "", "```", format_scores(s), "```", ""]
    for name, block in s["per_session"].items():
        out += [f"## {name}", "", "```", format_scores(block), "```", ""]
    out += ["## Marks", ""] + marks_section(model, data) + [""]
    return "\n".join(out), s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data")
    ap.add_argument("model")
    ap.add_argument("--out", default=None, help="write the markdown report here")
    ap.add_argument("--labels", choices=("pad", "any"), default=None,
                    help="default: whatever the model was trained with")
    ap.add_argument("--session", action="append", default=None,
                    help="grade only these sessions (default: all of them)")
    args = ap.parse_args(argv)

    data = load(args.data)
    model = pl.PolicyModel(args.model)
    label_mode = args.labels or model.meta.get("labels", "pad")
    names = data["meta"]["sessions"]
    ids = [names.index(n) for n in args.session] if args.session else None
    train = np.isin(data["session"], [names.index(n) for n in
                                      model.meta.get("train_sessions", []) if n in names])
    text, _ = report(model, data, label_mode, ids, train & data["decision"])
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
