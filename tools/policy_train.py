"""Train a driving policy off a policy_dataset.py .npz, and export it to ONNX.

    python tools/policy_train.py DATA.npz OUT.onnx --representation profile --val SESSION
    python tools/policy_train.py DATA.npz OUT.onnx --blocks 5

Writes OUT.onnx and OUT.json (the sidecar policy.PolicyModel reads), then
grades the exported file -- not the torch module -- with policy_eval, so the
numbers printed are the numbers the car's runtime would produce.

Laptop only: needs torch, which is not and must never be a dependency of
the car. The car runs the .onnx through onnxruntime.

The heads
---------
  logits    (1, 3)  direction over policy.ACTIONS, cross-entropy weighted by
                    inverse class frequency (reverse and stop are rare)
  throttle  (1, 1)  sigmoid, the fraction of that direction's speed cap;
                    trained only on ticks that are moving
  steer     (1, 1)  tanh, front steering as a fraction of STEER_SCALE (0.7)
  mode_logits (1, 2) logits over policy.STEER_MODES, the mode I want, given
                    the third input `mode` (the one the car is in). The rear
                    is not a head: unified is steering_pair(front) and split
                    is 0, which is every split row the dataset keeps.

Each group is trained only where it is a label: steering and the mode where
steer_by is "pad", direction and throttle where throttle_by is "pad".
--labels any drops that mask, which is what cloning the autopilot as a smoke
test needs.

The mode is sticky, so almost every tick's answer is "the one I am in", and
a head that only copies its input scores 99%. What it is for is the few
ticks around an L3 press, so those (policy_dataset's mode_switch, within
MODE_WINDOW_S of a toggle) count MODE_SWITCH_WEIGHT times in the mode loss.
The input also goes straight to the mode logits as well as through the
trunk, so "stay" is cheap to learn and the trunk's job is the exception.

/marks on pad-driven ticks (policy_dataset.mark_flags): a tick in a BAD
window is out of training (and out of the val numbers); a tick in a GOOD
window counts --good-weight times in every head's loss. 2 by default: enough
that a stretch I said was right pulls, not so much that a few seconds of it
outvote the rest of the session.

The val split
-------------
--val holds out whole sessions, which is the honest split when there are
several. With one, --blocks N cuts the decision ticks into N contiguous
stretches of time, holds each out in turn, trains on the rest less
BLOCK_GUARD_S either side of the held-out stretch (neighbouring ticks are
nearly the same picture), and reports every fold and the mean. Never a
random split by tick. After the folds it trains once more on everything and
exports that; its sidecar carries the fold numbers under "cv".

Three inputs are always in the graph, so the runtime has one code path:
x (the frame, as policy.encode_input() builds it), prev (the last decision,
zeros when the sidecar says uses_prev_action is false) and mode. --imu adds
a fourth, `imu`: the dataset's policy_imu features, which the car rebuilds
live with the same ImuFeatures. Its sidecar says uses_imu, and imu_tilt when
the dataset had an up vector; without one the tilt columns are zeroed here
and on the car (policy_imu.without_tilt). --imu and --prev combine. The
flip augmentation mirrors the IMU too (policy_imu.MIRROR). A graph without
--imu has no `imu` input at all, so every model before it loads unchanged.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import column_profile as cp  # noqa: E402
import policy as pl  # noqa: E402
import policy_imu as pimu  # noqa: E402
import policy_eval as pe  # noqa: E402

GOOD_WEIGHT = 2.0
# How much a tick inside a switch window counts in the mode loss. My one
# session has 4 toggles and 36 window ticks against 1900; at 10 the window is
# about a sixth of the mode loss, enough that the head cannot ignore it,
# not so much that it learns to switch everywhere it saw full lock.
MODE_SWITCH_WEIGHT = 10.0
# Training ticks this close to a held-out block are left out of that fold.
BLOCK_GUARD_S = 1.0


def build_model(representation, uses_imu=False):
    import torch
    from torch import nn

    imu_width = pimu.IMU_SIZE if uses_imu else 0

    class Heads(nn.Module):
        def __init__(self, width):
            super().__init__()
            # prev is always wired in, so every exported graph has the same
            # inputs; a model trained without it is fed zeros on both sides
            # and its weights there never see a gradient. The IMU is only
            # wired in when it is used, so an old graph stays an old graph.
            self.trunk = nn.Sequential(
                nn.Linear(width + pl.PREV_SIZE + pl.MODE_SIZE + imu_width, 64), nn.ReLU())
            self.logits = nn.Linear(64, len(pl.ACTIONS))
            self.throttle = nn.Linear(64, 1)
            self.steer = nn.Linear(64, 1)
            self.mode = nn.Linear(64, pl.MODE_SIZE)
            # The mode going in, straight to the mode logits: "stay" is one
            # weight, and the trunk only has to learn when not to.
            self.mode_skip = nn.Linear(pl.MODE_SIZE, pl.MODE_SIZE, bias=False)

        def forward(self, features, prev, mode, imu=None):
            parts = [features, prev, mode] + ([imu] if imu_width else [])
            h = self.trunk(torch.cat(parts, dim=1))
            return (self.logits(h), torch.sigmoid(self.throttle(h)),
                    torch.tanh(self.steer(h)), self.mode(h) + self.mode_skip(mode))

    class ImageNet(nn.Module):
        """(1, 2, 50, 80) -> four strided convs, 25x40, 13x20, 7x10, 4x5."""

        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(2, 8, 5, stride=2, padding=2), nn.ReLU(),
                nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Flatten())
            self.heads = Heads(16 * 4 * 5)

        def forward(self, x, prev, mode, imu=None):
            return self.heads(self.features(x), prev, mode, imu)

    class ProfileNet(nn.Module):
        """(1, 3, 32) -> two 1-D convs along the columns -> the heads."""

        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(cp.PROFILE_FEATURES, 16, 5, padding=2), nn.ReLU(),
                nn.Conv1d(16, 16, 5, stride=2, padding=2), nn.ReLU(),
                nn.Flatten())
            self.heads = Heads(16 * (cp.PROFILE_COLUMNS // 2))

        def forward(self, x, prev, mode, imu=None):
            return self.heads(self.features(x), prev, mode, imu)

    model = ImageNet() if representation == pl.IMAGE else ProfileNet()
    model.uses_imu = bool(uses_imu)
    return model


def encode_frames(frames, representation):
    return np.concatenate([pl.encode_input(representation, f) for f in frames])


def encode_modes(mode_in):
    """(N,) STEER_MODES indices -> (N, MODE_SIZE) one-hots, as encode_mode()."""
    out = np.zeros((len(mode_in), pl.MODE_SIZE), dtype=np.float32)
    out[np.arange(len(mode_in)), np.asarray(mode_in, dtype=np.int64)] = 1.0
    return out


def flip(x, prev, imu=None):
    """Mirror left-right: the frame's last axis, and every steering sign.
    The steering mode has no side, so it goes through as it is. The IMU's
    lateral parts flip sign (policy_imu.MIRROR)."""
    import torch

    prev = prev.clone()
    prev[:, -1] = -prev[:, -1]
    if imu is not None:
        imu = imu * torch.from_numpy(pimu.MIRROR)
    return torch.flip(x, dims=[-1]), prev, imu


def imu_inputs(data, tilt):
    """The dataset's IMU features as the model is fed them: with the tilt
    columns zeroed unless it is trained with tilt."""
    feats = np.asarray(data["imu"], dtype=np.float32)
    return feats if tilt else pimu.without_tilt(feats)


def dataset_has_tilt(data):
    return bool(data["meta"].get("imu_tilt", False))


def tick_weights(data, idx, good_weight):
    """Per training tick loss weight: good_weight inside a GOOD mark, else 1."""
    return np.where(pe.marks_of(data)[idx] == pe.MARK_GOOD,
                    float(good_weight), 1.0).astype(np.float32)


def mode_weights(data, idx, switch_weight=MODE_SWITCH_WEIGHT):
    """Per training tick mode-loss factor: switch_weight inside a switch window."""
    window = np.asarray(data.get("mode_switch", np.zeros(len(data["t_ns"]), bool)))[idx]
    return np.where(window, float(switch_weight), 1.0).astype(np.float32)


def _weighted(values, w):
    return (values * w).sum() / w.sum()


def training_mask(data, in_val, labelled):
    """Decision ticks outside val, labelled, and not inside a BAD mark."""
    return ~in_val & data["decision"] & labelled & ~pe.bad_marked(data)


def time_blocks(data, n, guard_s=BLOCK_GUARD_S):
    """[(held_out, guard)] boolean masks: n contiguous stretches of decision
    ticks, in session-then-time order, equal by count. `guard` is every tick
    of the held-out block's sessions within guard_s of it, the block too:
    what a fold must not train on."""
    order = np.lexsort((data["t_ns"], data["session"]))
    order = order[data["decision"][order]]
    out = []
    for chunk in np.array_split(order, n):
        held = np.zeros(len(data["t_ns"]), dtype=bool)
        held[chunk] = True
        guard = held.copy()
        for sid in np.unique(data["session"][chunk]):
            in_s = chunk[data["session"][chunk] == sid]
            lo = data["t_ns"][in_s].min() - guard_s * 1e9
            hi = data["t_ns"][in_s].max() + guard_s * 1e9
            guard |= (data["session"] == sid) & (data["t_ns"] >= lo) & (data["t_ns"] <= hi)
        out.append((held, guard))
    return out


def train(data, representation, train_mask, label_mode, uses_prev, epochs, seed,
          good_weight=GOOD_WEIGHT, switch_weight=MODE_SWITCH_WEIGHT, uses_imu=False):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(representation, uses_imu)

    idx = np.flatnonzero(train_mask)
    x_all = torch.from_numpy(data["x"])
    x = x_all[data["frame"][idx]]
    prev = torch.from_numpy(data["prev"][idx] if uses_prev
                            else np.zeros_like(data["prev"][idx]))
    mode_in = torch.from_numpy(encode_modes(data["mode_in"][idx]))
    imu = (torch.from_numpy(imu_inputs(data, dataset_has_tilt(data))[idx])
           if uses_imu else None)
    mode = torch.from_numpy(data["steer_mode"][idx].astype(np.int64))
    action = torch.from_numpy(data["action"][idx].astype(np.int64))
    mag = torch.from_numpy(data["magnitude"][idx])
    steer = torch.from_numpy(data["steer"][idx])
    w_all = torch.from_numpy(tick_weights(data, idx, good_weight))
    w_mode = w_all * torch.from_numpy(mode_weights(data, idx, switch_weight))
    steer_all, throttle_all = pe.label_masks(data, label_mode)
    steer_ok = torch.from_numpy(steer_all[idx])
    throttle_ok = torch.from_numpy(throttle_all[idx])

    counts = np.bincount(data["action"][idx][throttle_ok.numpy()],
                         minlength=len(pl.ACTIONS)).astype(np.float64)
    # Inverse frequency, square-rooted. Plain inverse frequency on the smoke
    # set (901 forward to 25 reverse) weighted a reverse 36x and the model
    # answered by reversing at a third of open floor; the square root is 6x.
    weights = np.where(counts > 0,
                       np.sqrt(counts.sum() / np.maximum(counts, 1) / len(pl.ACTIONS)), 0.0)
    weights = weights / max(weights[counts > 0].min(), 1e-9)
    weights = np.minimum(weights, 10.0)  # one example of a class must not own the loss
    ce = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32),
                             reduction="none")
    ce_mode = nn.CrossEntropyLoss(reduction="none")

    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    batch = 64
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(idx))
        for start in range(0, len(idx), batch):
            b = order[start:start + batch]
            xb, pb = x[b], prev[b]
            ib = imu[b] if imu is not None else None
            sb = steer[b]
            if torch.rand(1).item() < 0.5:
                xb, pb, ib = flip(xb, pb, ib)
                sb = -sb
            logits, t_out, s_out, m_out = model(xb, pb, mode_in[b], ib)
            tok, sok = throttle_ok[b], steer_ok[b]
            wb = w_all[b]
            moving = tok & (action[b] != pl.ACTIONS.index(pl.STOP))
            loss = torch.zeros(())
            if tok.any():
                loss = loss + _weighted(ce(logits[tok], action[b][tok]), wb[tok])
            if moving.any():
                loss = loss + _weighted((t_out[moving, 0] - mag[b][moving]).abs(), wb[moving])
            if sok.any():
                loss = loss + _weighted((s_out[sok, 0] - sb[sok]).abs(), wb[sok])
                loss = loss + _weighted(ce_mode(m_out[sok], mode[b][sok]), w_mode[b][sok])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model, weights.tolist()


OUTPUTS = ["logits", "throttle", "steer", pl.MODE_OUTPUT]


def export(model, representation, out_path, meta):
    import torch

    shape = ((1, 2, pl.INPUT_HEIGHT, pl.INPUT_WIDTH) if representation == pl.IMAGE
             else (1, cp.PROFILE_FEATURES, cp.PROFILE_COLUMNS))
    args = (torch.zeros(shape), torch.zeros((1, pl.PREV_SIZE)),
            torch.zeros((1, pl.MODE_SIZE)))
    names = ["x", "prev", pl.MODE_INPUT]
    if getattr(model, "uses_imu", False):
        args += (torch.zeros((1, pimu.IMU_SIZE)),)
        names.append(pl.IMU_INPUT)
    torch.onnx.export(
        model, args, out_path,
        input_names=names, output_names=OUTPUTS,
        opset_version=17, dynamo=False)
    write_sidecar(out_path, meta)


def imu_meta(uses_imu, tilt):
    """The IMU half of the sidecar: what policy.check_imu reads."""
    if not uses_imu:
        return {}
    return {"uses_imu": True, "imu_features": list(pimu.FEATURES),
            "imu_window": pimu.WINDOW, "imu_tilt": bool(tilt),
            "imu_scales": {"gyro_rad_s": pimu.GYRO_SCALE,
                           "accel_m_s2": pimu.ACCEL_SCALE, "clip": pimu.CLIP,
                           "max_age_ms": pimu.MAX_AGE_MS, "gap_s": pimu.GAP_S}}


def write_sidecar(out_path, meta):
    with open(os.path.splitext(out_path)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)


def mode_meta(switch_weight=MODE_SWITCH_WEIGHT):
    """The mode head's half of the sidecar: what policy.check_sidecar reads."""
    return {
        "steer_modes": list(pl.STEER_MODES),
        "mode_switch": {"threshold": pl.MODE_THRESHOLD, "dwell_s": pl.MODE_DWELL_S,
                        "min_interval_s": pl.MODE_MIN_INTERVAL_S},
        "mode_rear": {pl.UNIFIED: "steering_pair(front)", pl.SPLIT: 0.0},
        "mode_not_modelled": "split with the right stick on the rear (|rear| > "
                             "split_rear_rest); the dataset drops those rows",
        "mode_switch_weight": float(switch_weight),
    }


def _mean_scores(folds):
    """Fold scores -> the mean of every number, and the switch counts summed."""
    keys = [k for k, v in folds[0].items() if isinstance(v, (int, float))]
    out = {}
    for k in keys:
        vals = [f[k] for f in folds if isinstance(f.get(k), (int, float))
                and np.isfinite(f[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    out["confusion"] = np.sum([f["confusion"] for f in folds], axis=0).tolist()
    # Pooled where a fold mean would hide it: one toggle per fold at best.
    for k in ("mode_switch_ticks", "mode_toggles", "mode_latch_switches",
              "mode_latch_toggles_matched", "mode_latch_false_switches",
              "mode_window_ticks"):
        if k in folds[0]:
            out[k + "_total"] = int(sum(f[k] for f in folds))
    hits = sum(f["mode_switch_recall"] * f["mode_switch_ticks"] for f in folds
               if f.get("mode_switch_ticks"))
    if "mode_switch_ticks" in folds[0]:
        n = sum(f["mode_switch_ticks"] for f in folds)
        out["mode_switch_recall_pooled"] = float(hits / n) if n else float("nan")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data")
    ap.add_argument("out", help="OUT.onnx; OUT.json is written beside it")
    ap.add_argument("--representation", choices=pl.REPRESENTATIONS, default=pl.PROFILE)
    ap.add_argument("--val", action="append", default=None,
                    help="session name to hold out (repeatable); default the last one")
    ap.add_argument("--blocks", type=int, default=0,
                    help="instead of --val: N contiguous time blocks, each held out "
                         "in turn, then a final model on everything")
    ap.add_argument("--labels", choices=("pad", "any"), default="pad",
                    help="pad: only hand-driven groups are labels. any: clone "
                         "whatever was commanded (the autopilot smoke test)")
    ap.add_argument("--prev", action="store_true",
                    help="feed the previous decision in as a second input")
    ap.add_argument("--imu", action="store_true",
                    help="feed the policy_imu features in as a fourth input "
                         "(tilt too, when the dataset has it)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--good-weight", type=float, default=GOOD_WEIGHT,
                    help="loss weight of a pad tick inside a GOOD /marks window "
                         f"(default {GOOD_WEIGHT}; BAD-window ticks are always left out)")
    ap.add_argument("--switch-weight", type=float, default=MODE_SWITCH_WEIGHT,
                    help="mode-loss weight of a tick inside a switch window "
                         f"(default {MODE_SWITCH_WEIGHT})")
    args = ap.parse_args(argv)

    import torch

    data = pe.load(args.data)
    meta_in = data["meta"]
    sessions = meta_in["sessions"]
    if "steer_mode" not in data:
        print(f"{args.data} has no steering-mode labels: rebuild it with policy_dataset.py")
        return 1
    if abs(float(meta_in.get("steer_scale", -1.0)) - pl.STEER_SCALE) > 1e-6:
        # Its steering labels are a fraction of a different lock; a model
        # trained on them would steer wrong, and the car refuses it anyway.
        print(f"{args.data} steers at {meta_in.get('steer_scale')}, the car at "
              f"{pl.STEER_SCALE}: rebuild it with policy_dataset.py")
        return 1
    if args.imu and "imu" not in data:
        print(f"{args.data} has no IMU features: rebuild it with policy_dataset.py")
        return 1
    steer_ok, throttle_ok = pe.label_masks(data, args.labels)
    labelled = steer_ok | throttle_ok
    data["x"] = encode_frames(data["frames"], args.representation)
    fit = dict(representation=args.representation, label_mode=args.labels,
               uses_prev=args.prev, epochs=args.epochs, seed=args.seed,
               good_weight=args.good_weight, switch_weight=args.switch_weight,
               uses_imu=args.imu)
    tmp = os.path.splitext(args.out)[0] + ".fold.onnx"

    cv = None
    if args.blocks:
        folds = []
        for k, (held, guard) in enumerate(time_blocks(data, args.blocks)):
            mask = training_mask(data, guard, labelled)
            model, _ = train(data, train_mask=mask, **fit)
            with torch.no_grad():
                export(model, args.representation, tmp,
                       {**_base_meta(meta_in, args, [], []), **mode_meta(args.switch_weight)})
            graded = held & ~pe.bad_marked(data)
            s = pe.score_ticks(pl.PolicyModel(tmp), data, np.flatnonzero(graded),
                               args.labels, pe.median_training_magnitude(data, mask))
            folds.append(s)
            print(f"--- block {k + 1} of {args.blocks}: {int(graded.sum())} ticks held out, "
                  f"{int(mask.sum())} trained")
            print(pe.format_scores(s))
        for ext in (".onnx", ".json"):
            try:
                os.remove(os.path.splitext(tmp)[0] + ext)
            except OSError:
                pass
        cv = {"blocks": args.blocks, "guard_s": BLOCK_GUARD_S,
              "mean": _mean_scores(folds), "folds": folds}
        print(f"--- mean over {args.blocks} blocks")
        print(pe.format_scores(cv["mean"]))
        val_names, val_ids = [], []
        in_val = np.zeros(len(data["t_ns"]), dtype=bool)
    else:
        val_names = args.val or [sessions[-1]]
        unknown = [v for v in val_names if v not in sessions]
        if unknown:
            print(f"no such session {unknown}; have {sessions}")
            return 1
        val_ids = [sessions.index(v) for v in val_names]
        in_val = np.isin(data["session"], val_ids)

    train_mask = training_mask(data, in_val, labelled)
    if not train_mask.any():
        print("no training ticks: every labelled decision is in --val, or nothing is "
              "pad-labelled (legacy files are all autopilot: try --labels any)")
        return 1

    model, weights = train(data, train_mask=train_mask, **fit)
    params = int(sum(p.numel() for p in model.parameters()))

    meta = {**_base_meta(meta_in, args, val_ids, val_names), **mode_meta(args.switch_weight),
            "train_decisions": int(train_mask.sum()),
            "class_weights": weights,
            "good_marked_train_ticks": int((train_mask & (pe.marks_of(data) == pe.MARK_GOOD)).sum()),
            "bad_marked_left_out": int((data["decision"] & pe.bad_marked(data)).sum()),
            "parameters": params}
    with torch.no_grad():
        export(model, args.representation, args.out, meta)

    model_rt = pl.PolicyModel(args.out)
    if cv is not None:
        meta["cv"] = cv
        # Graded on what it was trained on: a check the export works, not a score.
        meta["train_fit"] = pe.score_ticks(
            model_rt, data, np.flatnonzero(train_mask), args.labels,
            pe.median_training_magnitude(data, train_mask))
    else:
        meta["val"] = pe.score(model_rt, data, val_ids, args.labels, train_mask)
    write_sidecar(args.out, meta)
    print(f"{args.representation}: {params} parameters, "
          f"{os.path.getsize(args.out)} bytes, {int(train_mask.sum())} training decisions")
    print(pe.format_scores(meta["val"] if cv is None else meta["train_fit"]))
    return 0


def _base_meta(meta_in, args, val_ids, val_names):
    sessions = meta_in["sessions"]
    return {
        "input_contract": meta_in["input_contract"],
        "representation": args.representation,
        "actions": list(pl.ACTIONS),
        "uses_prev_action": bool(args.prev),
        **imu_meta(getattr(args, "imu", False), meta_in.get("imu_tilt", False)),
        "outputs": OUTPUTS,
        "steer_scale": pl.STEER_SCALE,
        "labels": args.labels,
        "train_sessions": [s for i, s in enumerate(sessions) if i not in val_ids],
        "val_sessions": val_names,
        "good_weight": args.good_weight,
        "epochs": args.epochs,
        "seed": args.seed,
    }


if __name__ == "__main__":
    sys.exit(main())
