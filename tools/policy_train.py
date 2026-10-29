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
  steer     (1, 1)  tanh, front steering as a fraction of STEER_SCALE (0.7);
                    the regression head, and the default (--steer-head)
  steer_dir (1, 3)  --steer-head direction instead of steer: logits over
                    policy.STEER_DIRS, left / straight / right, cross-entropy
                    on every steering label, the class off STRAIGHT_BAND
  steer_mag (1, 1)  and sigmoid, how far as a fraction of STEER_SCALE,
                    trained only where the label is a turn. The car steers
                    the class's side times this, 0 for straight
                    (policy.decode_steer). One regression averages a hard
                    left and a hard right into straight; this cannot
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

Interventions (policy_dataset's source array): a takeover tick is my hand
correcting auto, and a manual-after-auto tick is the few seconds of driving
right after I left it with START. Each is a mistake the model made on its
own drive and what I did instead -- data a record session can only stumble
into -- so each counts --takeover-weight times (TAKEOVER_WEIGHT) in every
head's loss, on top of any GOOD mark. The per-group masks still hold: a
takeover where I only steered is a steering label and nothing else. The
trainer prints what share of each head's loss weight that comes to, since a
weight on its own says nothing without the count it multiplies. --blocks
holds out record ticks only, so the folds are the same folds as a dataset
without drive logs, and every intervention trains in every fold.

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
exports that; its sidecar carries the fold numbers under "cv". --fold runs one
piece of that alone (a block, or the final model), which is how
tools/retrain.py trains them all at once, one process each.

Three inputs are always in the graph, so the runtime has one code path:
x (the frame, as policy.encode_input() builds it), prev (the last decision,
zeros when the sidecar says uses_prev_action is false) and mode. --imu adds
a fourth, `imu`: the dataset's policy_imu features, which the car rebuilds
live with the same ImuFeatures. Its sidecar says uses_imu. The tilt columns
(the up vector and tilt_valid) are zeroed here and on the car
(policy_imu.without_tilt) unless --imu-tilt asks for them, and then the
sidecar says imu_tilt. Tilt is asked for, never taken because the dataset
has it: most sessions predate the up vector, so whether a model learns off
the few that have it is a choice of recipe, and an A/B needs both arms off
the same dataset. --imu-tilt on a dataset with no up vector anywhere is
refused. A session without one is in the dataset as zeros and tilt_valid 0,
never a made-up reading. --imu and --prev combine. The
flip augmentation mirrors the IMU too (policy_imu.MIRROR), and swaps the
left and right steering classes (flip_steer_labels). A graph without
--imu has no `imu` input at all, so every model before it loads unchanged.
--imu-no-yaw is --imu without the yaw rate: the dataset's imu_no_yaw array
(policy_imu.NO_YAW_FEATURES, built by the same ImuFeatures with yaw=False,
as the car builds it), and imu_yaw false in the sidecar.

--lidar adds a fifth input, `lidar` (after `imu` when both are used): the
roof lidar scan each tick read, policy_lidar.encode_many over the dataset's
lidar_bins and lidar_present, as policy_lidar.encode() makes it on the car.
A small branch of its own reads it (build_model's LidarBranch), gated by the
present flag. Most sessions predate the lidar and go in as no scan; on the
ones that have it, LIDAR_DROPOUT of the ticks are fed no scan instead, fresh
every batch, so that "a scan is there" cannot become a tell for which
session this is, and the model learns to drive with the lidar gone. Only in
training: the car never blanks a scan it has. The flip mirrors the scan
(policy_lidar.MIRROR). The sidecar says uses_lidar and lidar_layout.
"""

import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys

import numpy as np

# The training venv (~/crawler-train) does not install this project, so the
# checkout root goes on the path for `import crawler`. Under `uv run` it is
# installed already and this is a no-op.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler.autopilot import column_profile as cp  # noqa: E402
from crawler.autopilot import policy as pl  # noqa: E402
from crawler.autopilot import policy_imu as pimu  # noqa: E402
from crawler.autopilot import policy_lidar as pli  # noqa: E402
from crawler.mapping import egomap as em  # noqa: E402
import policy_eval as pe  # noqa: E402

GOOD_WEIGHT = 2.0
# How much an intervention tick counts against a record tick. 77 takeover
# ticks across the first three policy drives against about 7500 record
# labels: at 3 they are about 3% of the loss, enough to be heard and not
# enough to drown the sessions they correct.
TAKEOVER_WEIGHT = 3.0
# policy_dataset.SOURCES, by name: what the dataset's "source" array indexes.
SOURCE_RECORD = "record"
INTERVENTION_SOURCES = ("takeover", "manual_after_auto")
# How much a tick inside a switch window counts in the mode loss. My one
# session has 4 toggles and 36 window ticks against 1900; at 10 the window is
# about a sixth of the mode loss, enough that the head cannot ignore it,
# not so much that it learns to switch everywhere it saw full lock.
MODE_SWITCH_WEIGHT = 10.0
# Training ticks this close to a held-out block are left out of that fold.
BLOCK_GUARD_S = 1.0


# The direction head's straight class: a label inside this, either side of
# 0, is straight; past it, left or right by its sign. A fraction of full lock.
# Off my 19 record sessions' 7416 steering labels: 20% are exactly 0 (the
# pad's 0.12 stick deadzone gives a literal 0), and the rest spread flat
# from 0 up, with no shoulder to put an edge on, so the band only has to
# be wider than noise: 0.1 takes 24% in, 0.05 would take 22%. 0.1 is 0.07
# of servo travel, the rest drift a recording from before the deadzone
# carries (up to about 0.07 raw, docs/TRAINING_DATA.md). Reversing, only 8%
# of my labels are inside it, and the rest are 46% left and 46% right.
STRAIGHT_BAND = 0.1


def steer_classes(steer, band=STRAIGHT_BAND):
    """Steering labels -> pl.STEER_DIRS indices: left past -band, right
    past +band, straight between."""
    s = np.asarray(steer)
    return np.where(s < -band, pl.STEER_DIRS.index(pl.LEFT),
                    np.where(s > band, pl.STEER_DIRS.index(pl.RIGHT),
                             pl.STEER_DIRS.index(pl.STRAIGHT))).astype(np.int64)


def flip_steer_labels(steer, steer_dir):
    """The mirror's steering labels: every steer's sign, and left and right
    swapped (straight stays straight). numpy or torch."""
    return -steer, (len(pl.STEER_DIRS) - 1) - steer_dir


def output_names(steer_head=pl.STEER_REGRESSION):
    """The graph's outputs, in order, for a steering head."""
    steer = ([pl.STEER_OUTPUT] if steer_head == pl.STEER_REGRESSION
             else [pl.STEER_DIR_OUTPUT, pl.STEER_MAG_OUTPUT])
    return ["logits", "throttle", *steer, pl.MODE_OUTPUT]


# The lidar branch's output width, before the present flag joins it.
LIDAR_FEATURES = 16
# Sensor dropout: the share of training ticks with a real scan that are fed
# no scan instead, fresh every batch. Training only; the car never blanks.
LIDAR_DROPOUT = 0.3


# The map branch's output width, before the present flag joins it.
MAP_FEATURES = 32
# Map dropout, as the lidar's: the share of training ticks with a map that
# are fed none, fresh every batch. Training only.
MAP_DROPOUT = 0.3


def build_model(representation, uses_imu=False, imu_yaw=True, steer_head=pl.STEER_REGRESSION,
                uses_lidar=False, uses_map=False):
    import torch
    from torch import nn

    if steer_head not in pl.STEER_HEADS:
        raise ValueError(f"steer head {steer_head!r}: one of {pl.STEER_HEADS}")
    direction = steer_head == pl.STEER_DIRECTION
    imu_width = pimu.size(imu_yaw) if uses_imu else 0
    lidar_width = LIDAR_FEATURES + 1 if uses_lidar else 0
    map_width = MAP_FEATURES + 1 if uses_map else 0

    class MapBranch(nn.Module):
        """(N, em.CHANNELS, 64, 64) -> (N, MAP_FEATURES + 1).

        The egocentric SLAM crop (crawler/mapping/egomap.py): four strided
        convs down to 4 x 4, a small linear layer, multiplied by present
        (the last plane) as the lidar branch is, and present on to the
        trunk: no map is exactly no map features.
        """

        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(em.CHANNELS - 1, 8, 5, stride=2, padding=2), nn.ReLU(),
                nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Flatten())
            self.fc = nn.Sequential(
                nn.Linear(16 * (em.SIZE // 16) ** 2, MAP_FEATURES), nn.ReLU())

        def forward(self, m):
            present = m[:, em.PRESENT, 0, 0:1]
            h = self.fc(self.conv(m[:, :em.PRESENT])) * present
            return torch.cat([h, present], dim=1)

    class LidarBranch(nn.Module):
        """(N, pli.SIZE) -> (N, LIDAR_FEATURES + 1).

        The ranges and the returns as two channels round the car, padded
        circularly (bin 71 is next to bin 0), one strided conv, a max over
        each 15 degrees, then a small linear layer. Its output is multiplied
        by present, so no scan is exactly no lidar features, and present
        itself goes on to the trunk: a model with the lidar missing is the
        camera model it was trained to be on every session without one.
        """

        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv1d(2, 8, 5, stride=2), nn.ReLU(),
                                      nn.MaxPool1d(3))
            self.fc = nn.Sequential(nn.Linear(8 * (pli.BINS // 6), LIDAR_FEATURES), nn.ReLU())

        def forward(self, lidar):
            present = lidar[:, pli.PRESENT:pli.PRESENT + 1]
            x = torch.stack([lidar[:, pli.RANGES], lidar[:, pli.RETURNS]], dim=1)
            x = torch.cat([x[..., -2:], x, x[..., :2]], dim=-1)
            h = self.fc(self.conv(x).flatten(1)) * present
            return torch.cat([h, present], dim=1)

    class Heads(nn.Module):
        def __init__(self, width):
            super().__init__()
            # prev is always wired in, so every exported graph has the same
            # inputs; a model trained without it is fed zeros on both sides
            # and its weights there never see a gradient. The IMU is only
            # wired in when it is used, so an old graph stays an old graph.
            self.trunk = nn.Sequential(
                nn.Linear(width + pl.PREV_SIZE + pl.MODE_SIZE + imu_width + lidar_width + map_width, 64),
                nn.ReLU())
            self.lidar = LidarBranch() if lidar_width else None
            self.map = MapBranch() if map_width else None
            self.logits = nn.Linear(64, len(pl.ACTIONS))
            self.throttle = nn.Linear(64, 1)
            if direction:
                self.steer_dir = nn.Linear(64, len(pl.STEER_DIRS))
                self.steer_mag = nn.Linear(64, 1)
            else:
                self.steer = nn.Linear(64, 1)
            self.mode = nn.Linear(64, pl.MODE_SIZE)
            # The mode going in, straight to the mode logits: "stay" is one
            # weight, and the trunk only has to learn when not to.
            self.mode_skip = nn.Linear(pl.MODE_SIZE, pl.MODE_SIZE, bias=False)

        def forward(self, features, prev, mode, *extra):
            # extra: imu if the model takes it, then lidar if it takes that,
            # then the map, in the graph's input order.
            extra = list(extra)
            imu = extra.pop(0) if imu_width else None
            lidar = extra.pop(0) if lidar_width else None
            ego = extra.pop(0) if map_width else None
            parts = [features, prev, mode] + ([imu] if imu_width else [])
            if lidar_width:
                parts.append(self.lidar(lidar))
            if map_width:
                parts.append(self.map(ego))
            h = self.trunk(torch.cat(parts, dim=1))
            steer = ((self.steer_dir(h), torch.sigmoid(self.steer_mag(h))) if direction
                     else (torch.tanh(self.steer(h)),))
            return (self.logits(h), torch.sigmoid(self.throttle(h)), *steer,
                    self.mode(h) + self.mode_skip(mode))

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

        def forward(self, x, prev, mode, *extra):
            return self.heads(self.features(x), prev, mode, *extra)

    class ProfileNet(nn.Module):
        """(1, 3, 32) -> two 1-D convs along the columns -> the heads."""

        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(cp.PROFILE_FEATURES, 16, 5, padding=2), nn.ReLU(),
                nn.Conv1d(16, 16, 5, stride=2, padding=2), nn.ReLU(),
                nn.Flatten())
            self.heads = Heads(16 * (cp.PROFILE_COLUMNS // 2))

        def forward(self, x, prev, mode, *extra):
            return self.heads(self.features(x), prev, mode, *extra)

    model = ImageNet() if representation == pl.IMAGE else ProfileNet()
    model.uses_imu = bool(uses_imu)
    model.uses_lidar = bool(uses_lidar)
    model.uses_map = bool(uses_map)
    model.imu_yaw = bool(imu_yaw)
    model.steer_head = steer_head
    return model


def encode_frames(frames, representation):
    return np.concatenate([pl.encode_input(representation, f) for f in frames])


def encode_modes(mode_in):
    """(N,) STEER_MODES indices -> (N, MODE_SIZE) one-hots, as encode_mode()."""
    out = np.zeros((len(mode_in), pl.MODE_SIZE), dtype=np.float32)
    out[np.arange(len(mode_in)), np.asarray(mode_in, dtype=np.int64)] = 1.0
    return out


def flip(x, prev, imu=None, imu_yaw=True, lidar=None):
    """Mirror left-right: the frame's last axis, and every steering sign.
    The steering mode has no side, so it goes through as it is. The IMU's
    lateral parts flip sign (policy_imu.mirror_of: MIRROR, or its no-yaw
    columns). The lidar's bins swap sides (policy_lidar.MIRROR). Returns
    (x, prev, imu) without a lidar, (x, prev, imu, lidar) with one."""
    import torch

    prev = prev.clone()
    prev[:, -1] = -prev[:, -1]
    if imu is not None:
        imu = imu * torch.from_numpy(pimu.mirror_of(imu_yaw))
    if lidar is not None:
        return torch.flip(x, dims=[-1]), prev, imu, pli.mirror(lidar)
    return torch.flip(x, dims=[-1]), prev, imu


def map_inputs(data, rows):
    """Dataset ticks -> their map input, egomap.encode_many over the crops
    load_maps paired them with; a tick with none is all zeros."""
    rows = np.asarray(rows)
    ok = data["map_row"][rows] >= 0
    r = np.where(ok, data["map_row"][rows], 0)
    return em.encode_many(data["map_codes"][r], data["map_trail"][r],
                          ok & data["map_present"][r])


def load_maps(data, crops_dir):
    """Pair every tick with egomap crops from tools/map_crops.py (one npz per
    session, by name): data gains map_row (-1 with none), map_codes,
    map_trail, map_present. A session with no file has no map, like a
    session recorded before the lidar. Returns how many ticks have one."""
    names = data["meta"]["sessions"]
    n = len(data["t_ns"])
    row = np.full(n, -1, np.int64)
    codes, trail, present = [], [], []
    base = 0
    for sid, name in enumerate(names):
        path = os.path.join(crops_dir, name + ".npz")
        if not os.path.exists(path):
            continue
        z = np.load(path)
        if str(z["layout"]) != em.LAYOUT:
            raise ValueError(f"{path}: map layout {z['layout']}, this trainer draws {em.LAYOUT}")
        ticks = np.flatnonzero(data["session"] == sid)
        j = em.pair_ticks(z["log_t"], data["t_ns"][ticks] / 1e9)
        row[ticks[j >= 0]] = base + j[j >= 0]
        codes.append(z["codes"])
        trail.append(z["trail"])
        present.append(z["present"])
        base += len(z["log_t"])
    size = (em.SIZE, em.SIZE)
    data["map_row"] = row
    data["map_codes"] = np.concatenate(codes) if codes else np.zeros((1,) + size, np.uint8)
    data["map_trail"] = np.concatenate(trail) if trail else np.zeros((1,) + size, np.uint8)
    data["map_present"] = np.concatenate(present) if present else np.zeros(1, bool)
    has = row >= 0
    has[has] &= data["map_present"][row[has]]
    data["map_on"] = has
    return int(has.sum())


def imu_inputs(data, tilt, yaw=True):
    """The dataset's IMU features as the model is fed them: the array for
    its list (policy_imu.DATASET_KEY), with the tilt columns zeroed unless it
    is trained with tilt."""
    key = pimu.DATASET_KEY[bool(yaw)]
    if key not in data:
        raise ValueError(f"the dataset has no {key!r} array: rebuild it with policy_dataset.py")
    feats = np.asarray(data[key], dtype=np.float32)
    if feats.shape[-1] != pimu.size(yaw):
        raise ValueError(f"the dataset's {key!r} is {feats.shape[-1]} wide, "
                         f"not {pimu.size(yaw)}: rebuild it with policy_dataset.py")
    return feats if tilt else pimu.without_tilt(feats)


def lidar_inputs(data):
    """The dataset's lidar scans as the model is fed them: policy_lidar's
    encode_many, the function the car's encode() runs, over every tick."""
    if "lidar_bins" not in data or "lidar_present" not in data:
        raise ValueError("the dataset has no lidar arrays: rebuild it with policy_dataset.py")
    return pli.encode_many(data["lidar_bins"], data["lidar_present"])


def lidar_blanks(present, rate, rng):
    """Which ticks of a batch the sensor dropout feeds no scan: a share
    `rate` of the ones with a scan, drawn fresh. A tick with none already
    has none."""
    return present & (rng.random(len(present)) < rate)


def dataset_has_tilt(data):
    return bool(data["meta"].get("imu_tilt", False))


def tilt_of(want_tilt, meta_in):
    """Whether a model is trained with tilt: only when asked (--imu-tilt),
    and ValueError when asked of a dataset with no up vector in it."""
    if not want_tilt:
        return False
    if not meta_in.get("imu_tilt", False):
        raise ValueError("--imu-tilt, but no session in the dataset has an up vector")
    return True


def sources_of(data):
    """Per tick its source name, every tick "record" in a dataset from
    before there was a source array."""
    names = data.get("meta", {}).get("sources")
    if "source" not in data or not names:
        return np.full(len(data["t_ns"]), SOURCE_RECORD, dtype=object)
    return np.asarray(names, dtype=object)[np.asarray(data["source"], dtype=np.int64)]


def intervention_ticks(data):
    """Ticks that are my hand correcting auto: a takeover, or manual after it."""
    return np.isin(sources_of(data), INTERVENTION_SOURCES)


def tick_weights(data, idx, good_weight, takeover_weight=TAKEOVER_WEIGHT):
    """Per training tick loss weight: good_weight inside a GOOD mark, times
    takeover_weight on an intervention, else 1."""
    good = np.where(pe.marks_of(data)[idx] == pe.MARK_GOOD, float(good_weight), 1.0)
    fix = np.where(intervention_ticks(data)[idx], float(takeover_weight), 1.0)
    return (good * fix).astype(np.float32)


def loss_shares(data, train_mask, label_mode, good_weight, takeover_weight,
                class_weights=None):
    """What share of each head's loss weight comes off intervention ticks.

    {head: (intervention ticks, all ticks, share)} for direction, throttle
    and steering, over the training ticks where that head has a label. The
    share is of the summed per-tick weight (and, for direction, the class
    weight too), which is what the loss is an average over; it is a share
    of the weight, not of the error, which depends on what the model gets
    wrong.
    """
    idx = np.flatnonzero(train_mask)
    w = tick_weights(data, idx, good_weight, takeover_weight).astype(np.float64)
    fix = intervention_ticks(data)[idx]
    steer_ok, throttle_ok = (m[idx] for m in pe.label_masks(data, label_mode))
    action = np.asarray(data["action"])[idx]
    cw = (np.asarray(class_weights, dtype=np.float64)[action] if class_weights is not None
          else np.ones(len(idx)))
    moving = throttle_ok & (action != pl.ACTIONS.index(pl.STOP))
    out = {}
    for head, ok, extra in (("direction", throttle_ok, cw), ("throttle", moving, 1.0),
                            ("steering", steer_ok, 1.0)):
        ww = (w * extra)[ok]
        total = float(ww.sum())
        out[head] = (int((fix & ok).sum()), int(ok.sum()),
                     float(ww[fix[ok]].sum() / total) if total else 0.0)
    return out


def format_shares(shares, takeover_weight):
    return (f"interventions at weight {takeover_weight:g}: " + "; ".join(
        f"{head} {n} of {total} ticks, {share:.1%} of its loss weight"
        for head, (n, total, share) in shares.items()))


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
    what a fold must not train on.

    Record ticks only, when the dataset has any: a drive log's ticks are
    never held out, so the folds are the ones the record sessions alone
    make, and an intervention trains in every fold."""
    order = np.lexsort((data["t_ns"], data["session"]))
    keep = np.asarray(data["decision"], dtype=bool).copy()
    record = sources_of(data) == SOURCE_RECORD
    if (keep & record).any():
        keep &= record
    order = order[keep[order]]
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
          good_weight=GOOD_WEIGHT, switch_weight=MODE_SWITCH_WEIGHT, uses_imu=False,
          imu_yaw=True, imu_tilt=False, steer_head=pl.STEER_REGRESSION,
          straight_band=STRAIGHT_BAND, takeover_weight=TAKEOVER_WEIGHT,
          uses_lidar=False, lidar_dropout=LIDAR_DROPOUT, uses_map=False,
          map_dropout=MAP_DROPOUT):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(representation, uses_imu, imu_yaw, steer_head, uses_lidar, uses_map)
    direction = steer_head == pl.STEER_DIRECTION

    idx = np.flatnonzero(train_mask)
    x_all = torch.from_numpy(data["x"])
    x = x_all[data["frame"][idx]]
    prev = torch.from_numpy(data["prev"][idx] if uses_prev
                            else np.zeros_like(data["prev"][idx]))
    mode_in = torch.from_numpy(encode_modes(data["mode_in"][idx]))
    imu = (torch.from_numpy(imu_inputs(data, imu_tilt, imu_yaw)[idx])
           if uses_imu else None)
    lidar = torch.from_numpy(lidar_inputs(data)[idx]) if uses_lidar else None
    lidar_on = (np.asarray(data["lidar_present"], dtype=bool)[idx] if uses_lidar
                else np.zeros(len(idx), dtype=bool))
    # The map is encoded a batch at a time: every tick at once is 80 KB each.
    map_on = (np.asarray(data["map_on"], dtype=bool)[idx] if uses_map
              else np.zeros(len(idx), dtype=bool))
    mode = torch.from_numpy(data["steer_mode"][idx].astype(np.int64))
    action = torch.from_numpy(data["action"][idx].astype(np.int64))
    mag = torch.from_numpy(data["magnitude"][idx])
    steer = torch.from_numpy(data["steer"][idx])
    steer_dir = torch.from_numpy(steer_classes(data["steer"][idx], straight_band))
    straight = pl.STEER_DIRS.index(pl.STRAIGHT)
    w_all = torch.from_numpy(tick_weights(data, idx, good_weight, takeover_weight))
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
    ce_steer = nn.CrossEntropyLoss(reduction="none")

    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    batch = 64
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(idx))
        for start in range(0, len(idx), batch):
            b = order[start:start + batch]
            xb, pb = x[b], prev[b]
            ib = imu[b] if imu is not None else None
            lb = None
            if lidar is not None:
                blank = lidar_blanks(lidar_on[b.numpy()], lidar_dropout, rng)
                lb = torch.where(torch.from_numpy(blank)[:, None],
                                 torch.zeros(()), lidar[b])
            mb = None
            if uses_map:
                mb = map_inputs(data, idx[b.numpy()])
                blank = lidar_blanks(map_on[b.numpy()], map_dropout, rng)
                mb[blank] = 0.0
                mb = torch.from_numpy(mb)
            sb, db = steer[b], steer_dir[b]
            if torch.rand(1).item() < 0.5:
                if lb is not None:
                    xb, pb, ib, lb = flip(xb, pb, ib, imu_yaw, lb)
                else:
                    xb, pb, ib = flip(xb, pb, ib, imu_yaw)
                if mb is not None:
                    mb = torch.flip(mb, dims=[-1])   # egomap.mirror
                sb, db = flip_steer_labels(sb, db)
            extra = [v for v in (ib, lb, mb) if v is not None]
            out = model(xb, pb, mode_in[b], *extra)
            logits, t_out, m_out = out[0], out[1], out[-1]
            tok, sok = throttle_ok[b], steer_ok[b]
            wb = w_all[b]
            moving = tok & (action[b] != pl.ACTIONS.index(pl.STOP))
            loss = torch.zeros(())
            if tok.any():
                loss = loss + _weighted(ce(logits[tok], action[b][tok]), wb[tok])
            if moving.any():
                loss = loss + _weighted((t_out[moving, 0] - mag[b][moving]).abs(), wb[moving])
            if sok.any() and direction:
                # Which way, on every steering label; how far, only where it
                # is a turn: a straight label says nothing about how hard.
                d_out, g_out = out[2], out[3]
                loss = loss + _weighted(ce_steer(d_out[sok], db[sok]), wb[sok])
                turn = sok & (db != straight)
                if turn.any():
                    loss = loss + _weighted((g_out[turn, 0] - sb[turn].abs()).abs(), wb[turn])
            elif sok.any():
                loss = loss + _weighted((out[2][sok, 0] - sb[sok]).abs(), wb[sok])
            if sok.any():
                loss = loss + _weighted(ce_mode(m_out[sok], mode[b][sok]), w_mode[b][sok])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model, weights.tolist()


OUTPUTS = output_names(pl.STEER_REGRESSION)


def export(model, representation, out_path, meta):
    import torch

    shape = ((1, 2, pl.INPUT_HEIGHT, pl.INPUT_WIDTH) if representation == pl.IMAGE
             else (1, cp.PROFILE_FEATURES, cp.PROFILE_COLUMNS))
    args = (torch.zeros(shape), torch.zeros((1, pl.PREV_SIZE)),
            torch.zeros((1, pl.MODE_SIZE)))
    names = ["x", "prev", pl.MODE_INPUT]
    if getattr(model, "uses_imu", False):
        args += (torch.zeros((1, pimu.size(getattr(model, "imu_yaw", True)))),)
        names.append(pl.IMU_INPUT)
    if getattr(model, "uses_lidar", False):
        args += (torch.zeros((1, pli.SIZE)),)
        names.append(pl.LIDAR_INPUT)
    if getattr(model, "uses_map", False):
        args += (torch.zeros((1, em.CHANNELS, em.SIZE, em.SIZE)),)
        names.append(pl.MAP_INPUT)
    torch.onnx.export(
        model, args, out_path,
        input_names=names,
        output_names=output_names(getattr(model, "steer_head", pl.STEER_REGRESSION)),
        opset_version=17, dynamo=False)
    write_sidecar(out_path, meta)


def imu_meta(uses_imu, tilt, yaw=True):
    """The IMU half of the sidecar: what policy.check_imu reads."""
    if not uses_imu:
        return {}
    return {"uses_imu": True, "imu_features": list(pimu.features(yaw)),
            "imu_window": pimu.WINDOW, "imu_tilt": bool(tilt), "imu_yaw": bool(yaw),
            "imu_scales": {"gyro_rad_s": pimu.GYRO_SCALE,
                           "accel_m_s2": pimu.ACCEL_SCALE, "clip": pimu.CLIP,
                           "max_age_ms": pimu.MAX_AGE_MS, "gap_s": pimu.GAP_S}}


def map_meta(uses_map, dropout=MAP_DROPOUT):
    """The map half of the sidecar: what policy.check_map reads. A uses_map
    model drives with the hand-written driving rules off (policy.py,
    PolicyDriver) and is refused on a car not running --slam."""
    if not uses_map:
        return {}
    return {"uses_map": True, **em.meta(), "map_dropout": float(dropout),
            "map_features": MAP_FEATURES}


def lidar_meta(uses_lidar, dropout=LIDAR_DROPOUT):
    """The lidar half of the sidecar: what policy.check_lidar reads, and the
    dropout it was trained with (a note; the car never blanks)."""
    if not uses_lidar:
        return {}
    return {**pli.meta(), "lidar_dropout": float(dropout),
            "lidar_features": LIDAR_FEATURES}


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git_commit(repo=REPO):
    """(commit, dirty) of the training code, or (None, None) outside git.

    dirty is whether the tree had uncommitted changes, which means the commit
    alone does not reproduce this model.
    """
    try:
        head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10)
        if head.returncode != 0:
            return None, None
        status = subprocess.run(["git", "-C", repo, "status", "--porcelain",
                                 "--untracked-files=no"],
                                capture_output=True, text=True, timeout=10)
        return head.stdout.strip(), bool(status.stdout.strip())
    except Exception:
        return None, None


def provenance_meta(argv, model_name=None, commit=git_commit):
    """Where the model came from: when, off which code, by which command, and
    the family it is to be registered under if I gave one. Never a version:
    a number means better than the one before, and only tools/models.py
    promote can say that, so tools/models.py register files every model as a
    candidate."""
    sha, dirty = commit()
    out = {
        "created": datetime.datetime.now(datetime.timezone.utc)
                   .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "train_commit": sha,
        "train_commit_dirty": dirty,
        "train_command": shlex.join(["tools/policy_train.py", *argv]),
    }
    if model_name:
        out["model_name"] = model_name
    return out


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


def steer_meta(steer_head=pl.STEER_REGRESSION, band=STRAIGHT_BAND):
    """The steering head's half of the sidecar: what policy.check_steer_head
    reads, and the outputs the graph has."""
    out = {"steer_head": steer_head, "outputs": output_names(steer_head)}
    if steer_head == pl.STEER_DIRECTION:
        out["steer_dirs"] = list(pl.STEER_DIRS)
        out["steer_straight_band"] = float(band)
    return out


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
                         "(tilt zeroed unless --imu-tilt)")
    ap.add_argument("--imu-no-yaw", action="store_true",
                    help="--imu without the yaw rate (gyro_x): pitch, roll, "
                         "accel and tilt only (policy_imu.NO_YAW_FEATURES)")
    ap.add_argument("--imu-tilt", action="store_true",
                    help="--imu with the up vector fed in (zeros and tilt_valid 0 "
                         "on sessions without one); refused on a dataset with none")
    ap.add_argument("--lidar", action="store_true",
                    help="feed the roof lidar scan in as an input of its own "
                         "(policy_lidar), with sensor dropout in training")
    ap.add_argument("--lidar-dropout", type=float, default=LIDAR_DROPOUT,
                    help="with --lidar: the share of training ticks with a scan fed "
                         f"none instead, fresh every batch (default {LIDAR_DROPOUT})")
    ap.add_argument("--map", default=None, metavar="CROPS_DIR",
                    help="feed the SLAM map in as an input of its own (crawler/mapping/"
                         "egomap.py): the crops tools/map_crops.py replayed over each "
                         "lidar session, with dropout in training. Needs --lidar")
    ap.add_argument("--map-dropout", type=float, default=MAP_DROPOUT,
                    help=f"with --map: the share of ticks with a map fed none (default {MAP_DROPOUT})")
    ap.add_argument("--steer-head", choices=pl.STEER_HEADS, default=pl.STEER_REGRESSION,
                    help="regression: one tanh steer. direction: left/straight/right "
                         "plus how far, which cannot average two hard turns into straight")
    ap.add_argument("--straight-band", type=float, default=STRAIGHT_BAND,
                    help="with --steer-head direction: |steer| under this is the "
                         f"straight class (default {STRAIGHT_BAND} of full lock)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--good-weight", type=float, default=GOOD_WEIGHT,
                    help="loss weight of a pad tick inside a GOOD /marks window "
                         f"(default {GOOD_WEIGHT}; BAD-window ticks are always left out)")
    ap.add_argument("--takeover-weight", type=float, default=TAKEOVER_WEIGHT,
                    help="loss weight of an intervention tick: a takeover, or manual "
                         f"right after auto (default {TAKEOVER_WEIGHT}; build the dataset "
                         "with policy_dataset.py --drive-logs DIR to have any)")
    ap.add_argument("--switch-weight", type=float, default=MODE_SWITCH_WEIGHT,
                    help="mode-loss weight of a tick inside a switch window "
                         f"(default {MODE_SWITCH_WEIGHT})")
    ap.add_argument("--model-name", default=None,
                    help="family name for the sidecar, e.g. kblock")
    ap.add_argument("--fold", default=None, metavar="K|final",
                    help="with --blocks N, one piece of it: K (1..N) trains and grades "
                         "only block K and writes that fold model, its scores in the "
                         "sidecar's \"fold\"; final trains only the model on everything, "
                         "with no train fit. "
                         "tools/retrain.py runs each piece in its own process")
    # Kept only to say why it is gone: a release number is earned through
    # tools/models.py promote, never stamped at training time.
    ap.add_argument("--model-version", default=None, help=argparse.SUPPRESS)
    argv = list(sys.argv[1:] if argv is None else argv)
    args = ap.parse_args(argv)
    if args.imu_no_yaw or args.imu_tilt:
        args.imu = True
    if args.model_version is not None:
        print("--model-version is gone: a model gets a number only from "
              "tools/models.py promote, once it beats the latest release")
        return 2
    if args.fold is not None and not (
            args.blocks and (args.fold == "final"
                             or (args.fold.isdigit() and 1 <= int(args.fold) <= args.blocks))):
        print(f"--fold is final or a block 1..N of --blocks N, not {args.fold!r}")
        return 2

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
    if args.imu and pimu.DATASET_KEY[not args.imu_no_yaw] not in data:
        print(f"{args.data} has no {pimu.DATASET_KEY[not args.imu_no_yaw]} IMU features: "
              f"rebuild it with policy_dataset.py")
        return 1
    if args.lidar and "lidar_bins" not in data:
        print(f"{args.data} has no lidar arrays: rebuild it with policy_dataset.py")
        return 1
    if args.lidar and not np.asarray(data["lidar_present"]).any():
        print(f"{args.data}: --lidar, but no tick in it has a lidar scan")
        return 1
    try:
        tilt_of(args.imu_tilt, meta_in)
    except ValueError as e:
        print(f"{args.data}: {e}")
        return 1
    if args.map and not args.lidar:
        print("--map needs --lidar: the map is built off the lidar, and a car "
              "with a map always has the scan")
        return 2
    if args.map:
        try:
            with_map = load_maps(data, args.map)
        except ValueError as e:
            print(e)
            return 1
        if not with_map:
            print(f"--map {args.map}: no tick of {args.data} paired with a crop")
            return 1

    import torch

    steer_ok, throttle_ok = pe.label_masks(data, args.labels)
    labelled = steer_ok | throttle_ok
    data["x"] = encode_frames(data["frames"], args.representation)
    fit = dict(representation=args.representation, label_mode=args.labels,
               uses_prev=args.prev, epochs=args.epochs, seed=args.seed,
               good_weight=args.good_weight, switch_weight=args.switch_weight,
               uses_imu=args.imu, imu_yaw=not args.imu_no_yaw,
               imu_tilt=args.imu_tilt, steer_head=args.steer_head,
               straight_band=args.straight_band, takeover_weight=args.takeover_weight,
               uses_lidar=args.lidar, lidar_dropout=args.lidar_dropout,
               uses_map=bool(args.map), map_dropout=args.map_dropout)
    tmp = args.out if args.fold else os.path.splitext(args.out)[0] + ".fold.onnx"

    cv = None
    if args.blocks and args.fold != "final":
        folds = []
        for k, (held, guard) in enumerate(time_blocks(data, args.blocks)):
            if args.fold is not None and k + 1 != int(args.fold):
                continue
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
            if args.fold is not None:
                write_sidecar(args.out, {**_base_meta(meta_in, args, [], []),
                                         **mode_meta(args.switch_weight),
                                         "fold": {"block": k + 1, "blocks": args.blocks,
                                                  "guard_s": BLOCK_GUARD_S,
                                                  "trained": int(mask.sum()), "scores": s}})
                return 0
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
    elif args.blocks:
        # --fold final: the folds are other processes, and the caller puts
        # their numbers under "cv" once they are in.
        val_names, val_ids = [], []
        in_val = np.zeros(len(data["t_ns"]), dtype=bool)
    else:
        # The last record session: a drive log's few labelled ticks are no
        # val set, and there is no hand driving in the rest of it to grade.
        src = sources_of(data)
        record_ids = sorted({int(i) for i in np.asarray(data["session"])[src == SOURCE_RECORD]})
        val_names = args.val or [sessions[record_ids[-1] if record_ids else -1]]
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

    if args.imu_tilt:
        # A fold can hold out every session with an up vector, and then the
        # model learns tilt off zeros alone: say so, it is not a tilt model.
        tilt_col = pimu.FEATURES.index("tilt_valid")
        tilt_ticks = int((train_mask & (data["imu"][:, tilt_col] > 0)).sum())
        print(f"tilt: an up vector on {tilt_ticks} of {int(train_mask.sum())} training ticks")
    if args.lidar:
        print(f"lidar: a scan on {int((train_mask & data['lidar_present']).sum())} of "
              f"{int(train_mask.sum())} training ticks, {args.lidar_dropout:.0%} of them "
              f"blanked each batch")
    if args.map:
        print(f"map: a crop on {int((train_mask & data['map_on']).sum())} of "
              f"{int(train_mask.sum())} training ticks, {args.map_dropout:.0%} of them "
              f"blanked each batch")
    model, weights = train(data, train_mask=train_mask, **fit)
    params = int(sum(p.numel() for p in model.parameters()))
    shares = loss_shares(data, train_mask, args.labels, args.good_weight,
                         args.takeover_weight, weights)
    print(format_shares(shares, args.takeover_weight))

    meta = {**_base_meta(meta_in, args, val_ids, val_names), **mode_meta(args.switch_weight),
            "train_decisions": int(train_mask.sum()),
            "class_weights": weights,
            "good_marked_train_ticks": int((train_mask & (pe.marks_of(data) == pe.MARK_GOOD)).sum()),
            "bad_marked_left_out": int((data["decision"] & pe.bad_marked(data)).sum()),
            "intervention_train_ticks": int((train_mask & intervention_ticks(data)).sum()),
            "intervention_loss_share": {head: share for head, (_, _, share) in shares.items()},
            "parameters": params,
            **({"imu_tilt_train_ticks": tilt_ticks} if args.imu_tilt else {}),
            **({"lidar_train_ticks": int((train_mask & data["lidar_present"]).sum())}
               if args.lidar else {}),
            **({"map_train_ticks": int((train_mask & data["map_on"]).sum())}
               if args.map else {}),
            **provenance_meta(argv, args.model_name)}
    with torch.no_grad():
        export(model, args.representation, args.out, meta)

    model_rt = pl.PolicyModel(args.out)
    if cv is not None:
        meta["cv"] = cv
    # --fold final has no train fit: the caller has the folds, and grading
    # every training tick one at a time takes longer than the training did.
    if args.blocks and args.fold != "final":
        # Graded on what it was trained on: a check the export works, not a score.
        meta["train_fit"] = pe.score_ticks(
            model_rt, data, np.flatnonzero(train_mask), args.labels,
            pe.median_training_magnitude(data, train_mask))
    elif not args.blocks:
        meta["val"] = pe.score(model_rt, data, val_ids, args.labels, train_mask)
    write_sidecar(args.out, meta)
    print(f"{args.representation}: {params} parameters, "
          f"{os.path.getsize(args.out)} bytes, {int(train_mask.sum())} training decisions")
    if args.fold != "final":
        print(pe.format_scores(meta["train_fit"] if args.blocks else meta["val"]))
    return 0


def _base_meta(meta_in, args, val_ids, val_names):
    sessions = meta_in["sessions"]
    return {
        "input_contract": meta_in["input_contract"],
        "representation": args.representation,
        "actions": list(pl.ACTIONS),
        "uses_prev_action": bool(args.prev),
        **imu_meta(getattr(args, "imu", False),
                   tilt_of(getattr(args, "imu_tilt", False), meta_in),
                   not getattr(args, "imu_no_yaw", False)),
        **lidar_meta(getattr(args, "lidar", False),
                     getattr(args, "lidar_dropout", LIDAR_DROPOUT)),
        **map_meta(bool(getattr(args, "map", None)),
                   getattr(args, "map_dropout", MAP_DROPOUT)),
        **steer_meta(getattr(args, "steer_head", pl.STEER_REGRESSION),
                     getattr(args, "straight_band", STRAIGHT_BAND)),
        "steer_scale": pl.STEER_SCALE,
        "labels": args.labels,
        "train_sessions": [s for i, s in enumerate(sessions) if i not in val_ids],
        "val_sessions": val_names,
        "good_weight": args.good_weight,
        "takeover_weight": getattr(args, "takeover_weight", TAKEOVER_WEIGHT),
        "epochs": args.epochs,
        "seed": args.seed,
    }


if __name__ == "__main__":
    sys.exit(main())
