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

Accepted auto (--accepted-weight W): the car's own decisions that I let
stand, as weak labels. A tick the model drove (policy_drove, source auto)
counts when there is no takeover in the ACCEPT_AFTER_S after it or the
ACCEPT_BEFORE_S before it, the session goes on for the whole ACCEPT_AFTER_S
with no hole in its ticks (an excluded window is a hole: whatever happened
in there is not "let stand"), and it is not in a tip's run-up. Its label is
what the car sent, at W a tick. 0, the default, is no auto tick a label at
all, as every model before it. It is there because takeovers are 77% of the
gear loss, and they are the moments the car was wrong: a model that only
ever sees those learns that the floor is a place to reverse on.

--class-weight-power P raises the gear's class weights (square-rooted
inverse frequency, normalised to forward 1) to P: 1 is the weights as they
always were, 0.5 half their log, 0 none. --gear-weights F,R,S sets them
outright instead (in pl.ACTIONS order), so one class can keep its weight
while another's is softened.

--accepted-own DIR (tools/auto_overrides.py) keeps an accepted-auto tick
only where the command was the model's own: not the goal's "goal reached",
the floor's hold, the rear stop, the tilt veto or a recover. Those are the
car's rules, and a stop they made is no stop the model chose.

Keyframes (--keyframe-weight W, policy_keyframes): a copycat of my last
--keyframe-history decisions alone predicts each one, out of fold by
session, and the --keyframe-share of decisions it gets most wrong (its
action prediction error) count W times on top of every other weight -- the
moments I changed my mind, which a policy can most easily coast through
(Wen et al. 2021, W 5 on the top 10%). 0, the default, is no keyframes.

--intervention-share P sets the takeover and accepted weights off the
training ticks' own counts, as Sirius (Liu et al., RSS 2023) does: the
interventions P of the gear's ticks by weight, the record ticks the share
they have, the accepted auto the rest. --accept-after-s is the window before
a takeover that is not "let stand" (their pre-intervention, weight 0).

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

--chunk K widens the gear head to 3K logits and the steer head to K: the
answer for this decision and the next K-1 (ACT's action chunks). Offset i's
label is the i-th next decision tick of the same session, every step of the
chain under policy.CHUNK_LINK_S (chunk_index), with that tick's own masks and
weight, and the gear and steer losses are averaged over the offsets. Throttle
and mode stay one step. The car blends the last K answers
(policy.ChunkEnsembler) with the weight --ensemble-m sets, so the sidecar says
chunk, ensemble_m and ensemble_link_s. --chunk 1 is the one-step model, the
same training step for step and a sidecar with none of the three. The eval
here grades offset 0 alone.
"""

import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys

import numpy as np

# The training venv (<folder>/train, tools/data_paths.py) does not install this project, so the
# checkout root goes on the path for `import crawler`. Under `uv run` it is
# installed already and this is a no-op.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler.autopilot import column_profile as cp  # noqa: E402
from crawler.autopilot import policy as pl  # noqa: E402
from crawler.autopilot import policy_goal as pg  # noqa: E402
from crawler.autopilot import policy_imu as pimu  # noqa: E402
from crawler.autopilot import policy_lidar as pli  # noqa: E402
from crawler.mapping import egomap as em  # noqa: E402
import policy_eval as pe  # noqa: E402
import policy_keyframes as pk  # noqa: E402

GOOD_WEIGHT = 2.0
# How much an intervention tick counts against a record tick. 77 takeover
# ticks across the first three policy drives against about 7500 record
# labels: at 3 they are about 3% of the loss, enough to be heard and not
# enough to drown the sessions they correct.
TAKEOVER_WEIGHT = 3.0
# Accepted auto (--accepted-weight): how long after and before a takeover an
# auto tick is not one I let stand, and the longest hole in a session's
# ticks that still counts as the car driving on.
ACCEPT_AFTER_S = 3.0
ACCEPT_BEFORE_S = 2.0
ACCEPT_HOLE_S = 1.0
# policy_dataset.SOURCES, by name: what the dataset's "source" array indexes.
SOURCE_RECORD = "record"
SOURCE_AUTO = "auto"
SOURCE_TAKEOVER = "takeover"
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
             else list(pl.GEAR_STEER_OUTPUTS) if steer_head == pl.STEER_PER_GEAR
             else [pl.STEER_DIR_OUTPUT, pl.STEER_MAG_OUTPUT])
    return ["logits", "throttle", *steer, pl.MODE_OUTPUT]


# The per-gear head (policy.STEER_PER_GEAR): which gear's steer a tick's
# steering label trains. A forward or a reverse is its own. A stop is the
# gear the car moves in next, if it moves within STOP_GEAR_S, else the one it
# last moved in within STOP_GEAR_S, else forward: the lock I put on standing
# still between gears is for the gear that comes next, and the car steers a
# model stop with the head of the better of forward and reverse
# (policy.steer_gear), the gear its logits say comes next.
STOP_GEAR_S = 2.0
GEAR_FORWARD, GEAR_REVERSE = 0, 1


def steer_gears(data, stop_gear_s=STOP_GEAR_S):
    """(N,) per tick GEAR_FORWARD or GEAR_REVERSE: the head its steering
    label trains (see STOP_GEAR_S). Off every tick's action, labelled or
    not: what the car did next is what came next, whoever drove it."""
    action = np.asarray(data["action"], dtype=np.int64)
    t = np.asarray(data["t_ns"], dtype=np.int64)
    sess = np.asarray(data["session"], dtype=np.int64)
    fwd, rev = pl.ACTIONS.index(pl.FORWARD), pl.ACTIONS.index(pl.REVERSE)
    out = np.where(action == rev, GEAR_REVERSE, GEAR_FORWARD).astype(np.int64)
    order = np.lexsort((t, sess))
    a, ts, ss = action[order], t[order], sess[order]
    moving = a != pl.ACTIONS.index(pl.STOP)
    n = len(order)
    nxt = np.full(n, -1, np.int64)
    last = -1
    for j in range(n - 1, -1, -1):
        if j + 1 < n and ss[j + 1] != ss[j]:
            last = -1
        if moving[j]:
            last = j
        nxt[j] = last
    prv = np.full(n, -1, np.int64)
    last = -1
    for j in range(n):
        if j and ss[j - 1] != ss[j]:
            last = -1
        if moving[j]:
            last = j
        prv[j] = last
    for j in np.flatnonzero(~moving):
        pick = -1
        if nxt[j] >= 0 and (ts[nxt[j]] - ts[j]) / 1e9 <= stop_gear_s:
            pick = nxt[j]
        elif prv[j] >= 0 and (ts[j] - ts[prv[j]]) / 1e9 <= stop_gear_s:
            pick = prv[j]
        out[order[j]] = GEAR_REVERSE if pick >= 0 and a[pick] == rev else GEAR_FORWARD
    return out


# The scorecard: drives no model is trained on, so every model can be graded
# on the same ticks it has never seen (tools/policy_scorecard.py). Names only,
# one a line, # for a comment; the recordings themselves never come here.
SCORECARD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scorecard.txt")


def scorecard_sessions(path=SCORECARD_FILE):
    """The scorecard's session names, [] with no file."""
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        return [line.split("#", 1)[0].strip() for line in f
                if line.split("#", 1)[0].strip()]


# Drives the owner sent back off the car: tools/policy_scorecard.py replays
# them (drive_open_rev), so no model trains on them either. Same format as
# the scorecard; add a drive, never take one off.
REJECTED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rejected_drives.txt")


def rejected_sessions(path=REJECTED_FILE):
    """The rejected drives' session names, [] with no file."""
    return scorecard_sessions(path)


def held_out_sessions():
    """Every session no model trains on unless told: the scorecard's, then
    the rejected drives'."""
    names = scorecard_sessions()
    return names + [n for n in rejected_sessions() if n not in names]


def scorecard_mask(data, names):
    """Per tick, whether its session is one of `names` (the scorecard's)."""
    sessions = list(data["meta"]["sessions"])
    ids = [sessions.index(n) for n in names if n in sessions]
    return np.isin(np.asarray(data["session"]), ids)


# The mirror augmentation (--mirror). coin: each batch goes in as it is or
# mirrored, a coin toss a batch, which is every model up to goal2. both: each
# batch goes in as it is and mirrored, side by side, so every tick is seen
# both ways every epoch and no batch leans to one side. none: never.
MIRROR_MODES = ("coin", "both", "none")

# The map branch's activation (--map-act): relu is every map model up to
# goal2, whose map branch trained to all zeros (MapBranch); leaky keeps
# MAP_LEAK of the gradient through a unit that is off.
MAP_ACTS = ("relu", "leaky")
MAP_LEAK = 0.05

# The optimizer (--optimizer): adam, the weight decay folded into the
# gradient, is every model up to goal2; adamw decays the weights apart.
OPTIMIZERS = ("adam", "adamw")


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


# The goal branch's output width, before the present flag joins it.
GOAL_FEATURES = 16


def build_model(representation, uses_imu=False, imu_yaw=True, steer_head=pl.STEER_REGRESSION,
                uses_lidar=False, uses_map=False, chunk=1, uses_goal=False, map_act="relu",
                lowband=None):
    import torch
    from torch import nn

    if lowband is not None and (lowband not in cp.LOWBAND_LAYOUTS
                                or representation != pl.PROFILE):
        raise ValueError(f"lowband {lowband!r}: one of {cp.LOWBAND_LAYOUTS}, "
                         f"on a profile model")
    # The lower band is three more rows of x under the footprint band's,
    # through the same convs: one picture of the columns, six numbers deep.
    x_rows = profile_rows_of(lowband)

    if map_act not in MAP_ACTS:
        raise ValueError(f"map activation {map_act!r}: one of {MAP_ACTS}")
    if steer_head not in pl.STEER_HEADS:
        raise ValueError(f"steer head {steer_head!r}: one of {pl.STEER_HEADS}")
    if not 1 <= int(chunk) <= pl.CHUNK_MAX:
        raise ValueError(f"chunk {chunk}: 1..{pl.CHUNK_MAX}")
    if chunk > 1 and steer_head == pl.STEER_DIRECTION:
        raise ValueError(f"chunk {chunk} needs a regression steer head")
    direction = steer_head == pl.STEER_DIRECTION
    per_gear = steer_head == pl.STEER_PER_GEAR
    imu_width = pimu.size(imu_yaw) if uses_imu else 0
    lidar_width = LIDAR_FEATURES + 1 if uses_lidar else 0
    map_width = MAP_FEATURES + 1 if uses_map else 0
    if uses_goal and not uses_map:
        raise ValueError("a goal model needs the map: the goal is a point on it")
    goal_width = GOAL_FEATURES + 1 if uses_goal else 0

    class GoalBranch(nn.Module):
        """(N, pg.SIZE) -> (N, GOAL_FEATURES + 1): one layer over the goal
        vector (policy_goal.py), gated by its present flag as the map's is,
        and present on to the trunk: no goal is exactly no goal features."""

        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(pg.SIZE, GOAL_FEATURES), nn.ReLU())

        def forward(self, g):
            present = g[:, pg.PRESENT:pg.PRESENT + 1]
            return torch.cat([self.fc(g) * present, present], dim=1)

    class MapBranch(nn.Module):
        """(N, em.CHANNELS, 64, 64) -> (N, MAP_FEATURES + 1).

        The egocentric SLAM crop (crawler/mapping/egomap.py): four strided
        convs down to 4 x 4, a small linear layer, multiplied by present
        (the last plane) as the lidar branch is, and present on to the
        trunk: no map is exactly no map features.

        With plain ReLUs this branch died in every map model up to goal2:
        its gradient starts near the size of the weight decay, the units go
        dead within a few hundred steps, and Adam then walks the weights to
        zero, so the map changed nothing. map_act="leaky" (MAP_ACTS) keeps
        a gradient through every unit.
        """

        def __init__(self):
            super().__init__()
            act = (lambda: nn.LeakyReLU(MAP_LEAK)) if map_act == "leaky" else nn.ReLU
            self.conv = nn.Sequential(
                nn.Conv2d(em.CHANNELS - 1, 8, 5, stride=2, padding=2), act(),
                nn.Conv2d(8, 16, 3, stride=2, padding=1), act(),
                nn.Conv2d(16, 16, 3, stride=2, padding=1), act(),
                nn.Conv2d(16, 16, 3, stride=2, padding=1), act(),
                nn.Flatten())
            self.fc = nn.Sequential(
                nn.Linear(16 * (em.SIZE // 16) ** 2, MAP_FEATURES), act())

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
                nn.Linear(width + pl.PREV_SIZE + pl.MODE_SIZE + imu_width + lidar_width
                          + map_width + goal_width, 64),
                nn.ReLU())
            self.lidar = LidarBranch() if lidar_width else None
            self.map = MapBranch() if map_width else None
            self.goal = GoalBranch() if goal_width else None
            # A chunk model answers for this decision and the next chunk - 1:
            # offset i's gear logits at [3i, 3i + 3), its steer at i.
            self.logits = nn.Linear(64, len(pl.ACTIONS) * chunk)
            self.throttle = nn.Linear(64, 1)
            if direction:
                self.steer_dir = nn.Linear(64, len(pl.STEER_DIRS))
                self.steer_mag = nn.Linear(64, 1)
            elif per_gear:
                # policy.GEAR_STEER_OUTPUTS: forward's steer, then reverse's.
                self.steer_forward = nn.Linear(64, chunk)
                self.steer_reverse = nn.Linear(64, chunk)
            else:
                self.steer = nn.Linear(64, chunk)
            self.mode = nn.Linear(64, pl.MODE_SIZE)
            # The mode going in, straight to the mode logits: "stay" is one
            # weight, and the trunk only has to learn when not to.
            self.mode_skip = nn.Linear(pl.MODE_SIZE, pl.MODE_SIZE, bias=False)

        def forward(self, features, prev, mode, *extra):
            # extra: imu if the model takes it, then lidar if it takes that,
            # then the map, then the goal, in the graph's input order.
            extra = list(extra)
            imu = extra.pop(0) if imu_width else None
            lidar = extra.pop(0) if lidar_width else None
            ego = extra.pop(0) if map_width else None
            goal = extra.pop(0) if goal_width else None
            parts = [features, prev, mode] + ([imu] if imu_width else [])
            if lidar_width:
                parts.append(self.lidar(lidar))
            if map_width:
                parts.append(self.map(ego))
            if goal_width:
                parts.append(self.goal(goal))
            h = self.trunk(torch.cat(parts, dim=1))
            steer = ((self.steer_dir(h), torch.sigmoid(self.steer_mag(h))) if direction
                     else (torch.tanh(self.steer_forward(h)), torch.tanh(self.steer_reverse(h)))
                     if per_gear else (torch.tanh(self.steer(h)),))
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
        """(1, 3, 32), or (1, 6, 32) with the lower band -> two 1-D convs
        along the columns -> the heads."""

        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(x_rows, 16, 5, padding=2), nn.ReLU(),
                nn.Conv1d(16, 16, 5, stride=2, padding=2), nn.ReLU(),
                nn.Flatten())
            self.heads = Heads(16 * (cp.PROFILE_COLUMNS // 2))

        def forward(self, x, prev, mode, *extra):
            return self.heads(self.features(x), prev, mode, *extra)

    model = ImageNet() if representation == pl.IMAGE else ProfileNet()
    model.uses_imu = bool(uses_imu)
    model.uses_lidar = bool(uses_lidar)
    model.uses_map = bool(uses_map)
    model.uses_goal = bool(uses_goal)
    model.imu_yaw = bool(imu_yaw)
    model.steer_head = steer_head
    model.chunk = int(chunk)
    model.lowband = lowband
    return model


def profile_rows_of(lowband):
    """x's rows for a profile model: the footprint band's three, and three
    more with a lower band."""
    return cp.PROFILE_FEATURES * (2 if lowband is not None else 1)


def encode_frames(frames, representation, lowband=None, pitch=None):
    """Every frame's x, as policy.encode_input draws it on the car. pitch is
    (len(frames),) radians, a LOWBAND_FLOOR model's (frame_pitch)."""
    if lowband is None:
        return np.concatenate([pl.encode_input(representation, f) for f in frames])
    if pitch is None:
        pitch = np.zeros(len(frames))
    return np.concatenate([pl.encode_input(representation, f, lowband, p)
                           for f, p in zip(frames, pitch)])


def frame_pitch(data, lowband):
    """(len(frames),) the pitch each frame's lower band is drawn at: the
    pitch of the tick that points at it (policy.lowband_pitch off the tick's
    up vector, policy_eval.tick_up), 0 for a frame no tick points at and
    for every layout but LOWBAND_FLOOR."""
    out = np.zeros(len(data["frames"]))
    if lowband != cp.LOWBAND_FLOOR:
        return out
    for i in range(len(data["frame"])):
        out[data["frame"][i]] = pl.lowband_pitch(lowband, pe.tick_up(data, i))
    return out


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


def mirror_inputs(x, prev, imu, lidar, ego, goal, imu_yaw=True):
    """Every input of a batch, mirrored left to right, any of them None: the
    frame and prev (flip), the IMU's lateral signs, the scan's bins, the
    crop's columns (egomap.mirror) and the goal's bearing (policy_goal.MIRROR).
    The mode has no side. Labels are the caller's: every steer changes sign."""
    import torch

    if lidar is not None:
        x, prev, imu, lidar = flip(x, prev, imu, imu_yaw, lidar)
    else:
        x, prev, imu = flip(x, prev, imu, imu_yaw)
    if ego is not None:
        ego = torch.flip(ego, dims=[-1])   # egomap.mirror
    if goal is not None:
        goal = goal * torch.from_numpy(pg.MIRROR)   # policy_goal.mirror
    return x, prev, imu, lidar, ego, goal


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
    # The poses the crops are drawn at, for hindsight goals (policy_goal.py):
    # a file from before they were written has none, and its rows never
    # get a goal.
    pose = {"log_t": [], "x": [], "y": [], "yaw": [], "frame": [], "posed": [], "sess": []}
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
        m = len(z["log_t"])
        posed = all(k in z.files for k in ("x", "y", "yaw", "frame"))
        pose["log_t"].append(np.asarray(z["log_t"], float))
        for k in ("x", "y", "yaw"):
            pose[k].append(np.asarray(z[k], float) if posed else np.zeros(m))
        pose["frame"].append(np.asarray(z["frame"], np.int64) if posed
                             else np.zeros(m, np.int64))
        pose["posed"].append(np.full(m, posed))
        pose["sess"].append(np.full(m, sid, np.int64))
        base += len(z["log_t"])
    size = (em.SIZE, em.SIZE)
    data["map_row"] = row
    data["map_codes"] = np.concatenate(codes) if codes else np.zeros((1,) + size, np.uint8)
    data["map_trail"] = np.concatenate(trail) if trail else np.zeros((1,) + size, np.uint8)
    data["map_present"] = np.concatenate(present) if present else np.zeros(1, bool)
    has = row >= 0
    has[has] &= data["map_present"][row[has]]
    data["map_on"] = has
    if codes:
        g = {k: np.concatenate(v) for k, v in pose.items()}
        g["present"] = data["map_present"] & g.pop("posed")
        g["key"] = pg.sort_key(g["sess"], g["log_t"])
        g["progress"] = pg.path_progress(g)
        data["goal_poses"] = g
    else:
        data.pop("goal_poses", None)
    return int(has.sum())


# Path goals (--goal-path-m): the share drawn nearer than the lookahead.
PATH_NEAR_SHARE = 0.25


def goal_draw_meta(path_m):
    """The sidecar's goal_draw: path goals path_m along, or "hindsight"
    (None). The car reads its lookahead back off it (policy_goal.lookahead_of)."""
    if path_m is None:
        return "hindsight"
    return {"path_m": float(path_m), "near_share": PATH_NEAR_SHARE}


def load_fed_goals(data, fed_dir):
    """Pair every tick with the goal the car was fed (tools/fed_goals.py, one
    npz per session, by name): data gains fed_ok, fed_ahead, fed_left.
    Returns how many ticks have one."""
    import fed_goals as fg
    names = data["meta"]["sessions"]
    n = len(data["t_ns"])
    ok = np.zeros(n, bool)
    ahead = np.zeros(n)
    left = np.zeros(n)
    for sid, name in enumerate(names):
        path = os.path.join(fed_dir, name + ".npz")
        if not os.path.exists(path):
            continue
        z = np.load(path)
        if str(z["layout"]) != fg.LAYOUT:
            raise ValueError(f"{path}: fed-goal layout {z['layout']}, this trainer reads {fg.LAYOUT}")
        ticks = np.flatnonzero(data["session"] == sid)
        j = fg.pair(z["log_t"], data["t_ns"][ticks] / 1e9)
        has = j >= 0
        jj = np.where(has, j, 0)
        has &= z["ok"][jj]
        ok[ticks[has]] = True
        ahead[ticks[has]] = z["ahead"][jj][has]
        left[ticks[has]] = z["left"][jj][has]
    data["fed_ok"], data["fed_ahead"], data["fed_left"] = ok, ahead, left
    return int(ok.sum())


def goal_inputs(data, rows, map_shown, rng, no_goal_share=pg.NO_GOAL_SHARE,
                k_s=None, path_m=None):
    """Goals for dataset ticks `rows`, as the model is fed them.

    Drawn ones first: hindsight goals (policy_goal.hindsight_goals), k drawn
    from [NEAR_S, FAR_S] unless given, or with path_m path goals
    (policy_goal.path_goals) path_m along the truck's own path, a share
    PATH_NEAR_SHARE of them nearer, down to ARRIVE_M, as the car feeds the
    goal itself once it is inside the planner's lookahead. Then the goal the
    car was really fed (load_fed_goals) wherever there was one, in place of
    the drawn one. No goal where the tick's map is not shown to the model
    this time (map_shown: no map, or blanked by the map dropout) -- the car
    has no goal without the map -- and on no_goal_share of the rest."""
    rows = np.asarray(rows)
    n = len(rows)
    g = data.get("goal_poses")
    if g is None or n == 0:
        return np.zeros((n, pg.SIZE), np.float32)
    if path_m is not None:
        d = np.where(rng.random(n) < PATH_NEAR_SHARE,
                     rng.uniform(pg.ARRIVE_M, path_m, n), path_m)
        ahead, left, ok = pg.path_goals(data["map_row"][rows], d, g)
    else:
        if k_s is None:
            k_s = rng.uniform(pg.NEAR_S, pg.FAR_S, n)
        ahead, left, ok = pg.hindsight_goals(data["map_row"][rows], k_s, g)
    if "fed_ok" in data:
        fed = data["fed_ok"][rows]
        ahead = np.where(fed, data["fed_ahead"][rows], ahead)
        left = np.where(fed, data["fed_left"][rows], left)
        ok = ok | fed
    ok &= np.asarray(map_shown, bool)
    if no_goal_share > 0:
        ok &= rng.random(n) >= no_goal_share
    return pg.encode_many(ahead, left, ok)


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


def takeover_near(data, after_s, before_s=0.0, hole_s=None):
    """Per tick, whether an intervention tick (a takeover, or manual after
    auto) of the same session falls in the
    after_s after it (strictly later) or the before_s before it (strictly
    earlier). With hole_s, also true where the session's ticks stop, or
    skip more than hole_s, inside the after_s: nothing says what came next."""
    src = intervention_ticks(data)
    sess = np.asarray(data["session"])
    t = np.asarray(data["t_ns"], np.int64)
    out = np.zeros(len(t), bool)
    after, before = int(after_s * 1e9), int(before_s * 1e9)
    for sid in np.unique(sess):
        rows = np.flatnonzero(sess == sid)
        rows = rows[np.argsort(t[rows], kind="stable")]
        tr = t[rows]
        tk = tr[src[rows]]
        if len(tk):
            j = np.searchsorted(tk, tr, side="right")
            nxt = np.where(j < len(tk), tk[np.minimum(j, len(tk) - 1)], np.iinfo(np.int64).max)
            out[rows] |= nxt - tr <= after
            if before > 0:
                k = np.searchsorted(tk, tr, side="left") - 1
                prv = np.where(k >= 0, tk[np.maximum(k, 0)], np.iinfo(np.int64).min // 2)
                out[rows] |= tr - prv <= before
        if hole_s is not None:
            # Where each hole starts (the last tick before it) and the end.
            gap = np.flatnonzero(np.diff(tr) > hole_s * 1e9)
            stops = np.append(tr[gap], tr[-1])
            j = np.searchsorted(stops, tr, side="left")
            out[rows] |= stops[np.minimum(j, len(stops) - 1)] - tr < after
    return out


def accepted_auto(data, after_s=ACCEPT_AFTER_S, before_s=ACCEPT_BEFORE_S,
                  hole_s=ACCEPT_HOLE_S):
    """The auto decisions the model drove that I let stand (--accepted-weight):
    no takeover in the after_s after or the before_s before, the session on
    for all of after_s without a hole over hole_s, outside a BAD mark and,
    where there is an up vector, outside a tip and its run-up."""
    ok = ((sources_of(data) == SOURCE_AUTO)
          & np.asarray(data.get("policy_drove", np.zeros(len(data["t_ns"]), bool)), bool)
          & np.asarray(data["decision"], bool) & ~pe.bad_marked(data)
          & ~takeover_near(data, after_s, before_s, hole_s))
    if "imu" in data:
        tipped, runup = tip_marks(data)
        ok &= ~tipped & ~runup
    return ok


def load_accepted_own(data, own_dir):
    """Per tick, whether the command was the model's own (tools/auto_overrides.py,
    one npz per session, by name): the newest /footprint row at or before
    the tick, no more than STALE_S old. A session with no file, or a tick
    with no row, is own: there is nothing to say otherwise. Returns (own,
    sessions with a file)."""
    import auto_overrides as ao
    names = data["meta"]["sessions"]
    own = np.ones(len(data["t_ns"]), bool)
    found = 0
    for sid, name in enumerate(names):
        path = os.path.join(own_dir, name + ".npz")
        if not os.path.exists(path):
            continue
        z = np.load(path)
        if str(z["layout"]) != ao.LAYOUT:
            raise ValueError(f"{path}: layout {z['layout']}, this trainer reads {ao.LAYOUT}")
        found += 1
        ticks = np.flatnonzero(data["session"] == sid)
        j = ao.pair(z["log_t"], data["t_ns"][ticks] / 1e9)
        has = j >= 0
        own[ticks[has]] = z["own"][j[has]]
    return own, found


def intervention_share_weights(data, mask, share):
    """(takeover_weight, accepted_weight) that make the intervention ticks
    `share` of the summed tick weight over `mask`, the record ticks the
    share they have unweighted, and the accepted auto (data["accepted"]) the
    rest: Sirius's P*(intv), P*(demo) = P(demo), P*(robot) the remainder,
    each class's weight P*(c) / P(c). Raises when there is no room left for
    the accepted ticks, or a class is empty."""
    mask = np.asarray(mask, bool)
    intv = mask & intervention_ticks(data)
    robot = mask & np.asarray(data.get("accepted", np.zeros(len(mask), bool)), bool) & ~intv
    demo = mask & ~intv & ~robot
    n_i, n_r, n_d = int(intv.sum()), int(robot.sum()), int(demo.sum())
    n = n_i + n_r + n_d
    rest = 1.0 - float(share) - n_d / max(n, 1)
    if not n_i or not n_r or rest <= 0:
        raise ValueError(f"no intervention share {share:g}: {n_i} interventions, {n_r} accepted, "
                         f"{n_d} record of {n}")
    return float(share) * n / n_i, rest * n / n_r


def tick_weights(data, idx, good_weight, takeover_weight=TAKEOVER_WEIGHT):
    """Per training tick loss weight: good_weight inside a GOOD mark, times
    takeover_weight on an intervention, times the accepted-auto weight on
    one of those (data["accepted_w"], set by --accepted-weight), times the
    keyframe weight (data["keyframe_w"], set by --keyframe-weight), else 1."""
    good = np.where(pe.marks_of(data)[idx] == pe.MARK_GOOD, float(good_weight), 1.0)
    fix = np.where(intervention_ticks(data)[idx], float(takeover_weight), 1.0)
    weak = np.asarray(data["accepted_w"])[idx] if "accepted_w" in data else 1.0
    w = good * fix * weak
    if "keyframe_w" in data:
        w = w * np.asarray(data["keyframe_w"])[idx]
    return w.astype(np.float32)


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
    steer_ok, throttle_ok = (m[idx] for m in label_masks(data, label_mode))
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


def chunk_index(data, k, link_s=pl.CHUNK_LINK_S):
    """(N, k) dataset rows: column i is tick j's i-th next decision tick in
    the same session, every step of the chain under link_s, the gap the car's
    ChunkEnsembler starts over at; -1 past a gap or the session's end.
    Column 0 is the tick itself, on decision ticks; -1 on the rest."""
    n = len(data["t_ns"])
    fut = np.full((n, k), -1, np.int64)
    t = np.asarray(data["t_ns"]) / 1e9
    o = np.lexsort((t, data["session"]))
    o = o[np.asarray(data["decision"], dtype=bool)[o]]
    ok = (np.diff(data["session"][o]) == 0) & (np.diff(t[o]) <= link_s)   # link j -> j + 1
    fut[o, 0] = o
    for i in range(1, k):
        good = np.ones(max(len(o) - i, 0), dtype=bool)
        for s in range(i):
            good &= ok[s:len(ok) - i + 1 + s]
        fut[o[:len(o) - i][good], i] = o[i:][good]
    return fut


def label_masks(data, label_mode):
    """policy_eval.label_masks, and the accepted-auto ticks (data["accepted"],
    set by --accepted-weight) as both labels too."""
    steer, throttle = pe.label_masks(data, label_mode)
    if "accepted" in data:
        weak = np.asarray(data["accepted"], bool)
        steer, throttle = steer | weak, throttle | weak
    return steer, throttle


def soften_weights(weights, power=1.0):
    """The gear's class weights to the power `power` (--class-weight-power):
    1 as they are, 0.5 half their log, 0 all 1. A class with no example
    keeps its 0."""
    w = np.asarray(weights, np.float64)
    return np.where(w > 0, np.power(np.where(w > 0, w, 1.0), float(power)), 0.0)


def gear_class_weights(counts, power=1.0, override=None):
    """The gear's class weights off its (weighted) per-class label counts:
    inverse frequency, square-rooted, the commonest class at 1, capped at 10,
    then to `power` (soften_weights). `override` (--gear-weights), when
    given, is used as it is instead."""
    if override is not None:
        return np.asarray(override, np.float64)
    counts = np.asarray(counts, np.float64)
    # Inverse frequency, square-rooted. Plain inverse frequency on the smoke
    # set (901 forward to 25 reverse) weighted a reverse 36x and the model
    # answered by reversing at a third of open floor; the square root is 6x.
    weights = np.where(counts > 0,
                       np.sqrt(counts.sum() / np.maximum(counts, 1) / len(pl.ACTIONS)), 0.0)
    weights = weights / max(weights[counts > 0].min(), 1e-9)
    weights = np.minimum(weights, 10.0)  # one example of a class must not own the loss
    return soften_weights(weights, power)


def training_mask(data, in_val, labelled):
    """Decision ticks outside val, labelled, and not inside a BAD mark."""
    return ~in_val & data["decision"] & labelled & ~pe.bad_marked(data)


# Training against a tip (--tip-weight). A tip is a tick whose up vector is
# TIP_DEG or more off the session's own upright, the median up of its valid
# readings: a session driven all along on a lean is not on its side. 45 is
# past every near-tip the car came back from but one (59), and the car went
# from under 10 to 45 in 0.5-0.7 s on every tip it had. The run-up is the
# TIP_LEAD_S before the first tipped tick of each crossing: on the tips it
# had, that is the stall or the impact and the start of the roll.
TIP_DEG = 45.0
TIP_LEAD_S = 1.0
# The unlikelihood's floor, so a gear the model is certain of costs a large
# number and not infinity.
TIP_EPS = 1e-3


def tip_marks(data, tip_deg=TIP_DEG, lead_s=TIP_LEAD_S):
    """(tipped, runup) per tick off the dataset's up vector (data["imu"]):
    tipped, tip_deg or more off the session's median up on a valid reading;
    runup, the lead_s before each tipped tick of the same session, not
    tipped itself, whatever its own reading."""
    imu = np.asarray(data["imu"])
    c = pimu.FEATURES.index("up_x")
    up = imu[:, c:c + 3].astype(np.float64)
    ok = imu[:, pimu.FEATURES.index("tilt_valid")] > 0.5
    sess = np.asarray(data["session"])
    t = np.asarray(data["t_ns"], np.int64)
    tipped = np.zeros(len(t), bool)
    runup = np.zeros(len(t), bool)
    lead = int(lead_s * 1e9)
    for sid in np.unique(sess):
        rows = np.flatnonzero(sess == sid)
        good = rows[ok[rows]]
        if not len(good):
            continue
        ref = np.median(up[good], axis=0)
        norm = np.linalg.norm(ref)
        if norm < 1e-6:
            continue
        cos = up[good] @ (ref / norm)
        over = good[np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))) >= tip_deg]
        if not len(over):
            continue
        tipped[over] = True
        # Each tick's next tipped time in its own session: inside lead of
        # it is the run-up. A tipped tick's next is itself, so it is never
        # its own run-up.
        times = np.sort(t[over])
        k = np.searchsorted(times, t[rows], side="left")
        nxt = np.where(k < len(times), times[np.minimum(k, len(times) - 1)],
                       np.iinfo(np.int64).max)
        runup[rows] = (nxt > t[rows]) & (nxt - t[rows] <= lead)
    return tipped, runup


def tip_labelled(labelled, tipped, runup):
    """The labels left once a tip's are dropped: a car on its side is not
    driving, and my hand in the second before it went over is not driving
    to copy."""
    return np.asarray(labelled, bool) & ~np.asarray(tipped, bool) & ~np.asarray(runup, bool)


def tip_rows(data, in_val, runup):
    """The ticks the tip penalty goes on: run-up decisions in a moving gear,
    outside val and BAD marks, whoever drove them. The car's own drives have
    no label, and these are what it did before it went over."""
    return (~np.asarray(in_val, bool) & np.asarray(data["decision"], bool)
            & np.asarray(runup, bool) & ~pe.bad_marked(data)
            & (np.asarray(data["action"]) != pl.ACTIONS.index(pl.STOP)))


def tip_penalty(p_taken, throttle, xp=np):
    """Per row: the unlikelihood of the gear the car took into a tip,
    -log(1 - p), plus the throttle it would ask for. xp is numpy or torch."""
    return -xp.log(1.0 - p_taken + TIP_EPS) + throttle


# The goal contrast (--goal-contrast). A hindsight goal is where my hand went
# anyway, and the frame and the map mostly say that already, so a model
# can fit every goal tick and never read the goal (goal6: the hindsight goal
# moves its gear and side on the scorecard by under half a percent). The
# contrast runs each goal tick again with its goal turned a quarter to a
# half turn away, and costs whatever the turned goal explains my driving
# there as well as the real one did: only the goal can tell the two apart.
# No action is made up: the label is mine both times.
GOAL_CONTRAST_MARGIN = 0.2


def rotate_goals(goals, angles, xp=np):
    """Goal inputs (N, SIZE) with each bearing turned angles (N,) radians
    CCW; distance and presence kept, a row with no goal left all zeros."""
    g = goals
    c, s = xp.cos(angles), xp.sin(angles)
    sin = g[:, pg.SIN] * c + g[:, pg.COS] * s
    cos = g[:, pg.COS] * c - g[:, pg.SIN] * s
    p = g[:, pg.PRESENT]
    # PRESENT, DIST, SIN, COS are columns 0..3 (policy_goal).
    return xp.stack([p, g[:, pg.DIST], sin * p, cos * p], 1)


def contrast_angles(rng, n):
    """n turns for the contrast's goals: 90 to 180 degrees, either way."""
    a = rng.uniform(np.pi / 2, np.pi, n)
    return np.where(rng.random(n) < 0.5, a, -a)


def contrast_rows(present, tok, sok, steer_only=False):
    """The rows the goal contrast costs: a goal in, and a label to explain --
    the steer's alone when steer_only."""
    return present & (sok if steer_only else (tok | sok))


def goal_contrast(loss_true, loss_fake, margin=GOAL_CONTRAST_MARGIN, xp=np):
    """Per row: what it costs that the turned goal explains my action within
    margin of the real one, relu(margin + true - fake)."""
    return xp.clip(margin + loss_true - loss_fake, 0.0, None)


def tip_meta(weight, deg=TIP_DEG, lead_s=TIP_LEAD_S, hygiene=None):
    """The sidecar's word on it: None when nothing was done about tips.
    Hygiene, the tip's labels dropped, comes with any weight."""
    hygiene = weight > 0 if hygiene is None else hygiene
    if not hygiene and weight <= 0:
        return {"tip": None}
    return {"tip": {"weight": float(weight), "deg": float(deg), "lead_s": float(lead_s),
                    "hygiene": bool(hygiene)}}


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
          map_dropout=MAP_DROPOUT, chunk=1, uses_goal=False, no_goal_share=pg.NO_GOAL_SHARE,
          mirror="coin", stop_gear_s=STOP_GEAR_S, map_act="relu", optimizer="adam",
          lowband=None, tip_mask=None, tip_weight=0.0, label_off=None, contrast_weight=0.0,
          goal_path_m=None, contrast_steer_only=False, class_weight_power=1.0,
          gear_weights=None, steer_tick_w=None):
    """tip_mask: the rows the tip penalty goes on (tip_rows), trained beside
    train_mask with no label of their own. label_off: ticks that are never a
    label, at any chunk offset either (tip_marks' tipped and run-up).
    steer_tick_w: a per-tick factor on the steer loss alone, every chunk
    offset at its own tick's (--keyframe-steer-only)."""
    import torch
    from torch import nn

    if optimizer not in OPTIMIZERS:
        raise ValueError(f"optimizer {optimizer!r}: one of {OPTIMIZERS}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(representation, uses_imu, imu_yaw, steer_head, uses_lidar, uses_map,
                        chunk, uses_goal, map_act, lowband)
    # The goals' draws on a generator of their own, so a model without the
    # goal is trained on exactly the draws it always was.
    goal_rng = np.random.default_rng([seed, 1])
    direction = steer_head == pl.STEER_DIRECTION
    per_gear = steer_head == pl.STEER_PER_GEAR
    if mirror not in MIRROR_MODES:
        raise ValueError(f"mirror {mirror!r}: one of {MIRROR_MODES}")

    use_tip = tip_mask is not None and tip_weight > 0 and bool(np.any(tip_mask))
    idx = np.flatnonzero(train_mask | tip_mask) if use_tip else np.flatnonzero(train_mask)
    tip_on = (torch.from_numpy(np.asarray(tip_mask, bool)[idx] & ~np.asarray(train_mask)[idx])
              if use_tip else None)
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
    gears_all = steer_gears(data, stop_gear_s) if per_gear else None
    gear_of = torch.from_numpy(gears_all[idx]) if per_gear else None
    w_all = torch.from_numpy(tick_weights(data, idx, good_weight, takeover_weight))
    w_steer = (w_all * torch.from_numpy(np.asarray(steer_tick_w, np.float32)[idx])
               if steer_tick_w is not None else w_all)
    w_mode = w_all * torch.from_numpy(mode_weights(data, idx, switch_weight))
    steer_all, throttle_all = label_masks(data, label_mode)
    if label_off is not None:
        keep = ~np.asarray(label_off, bool)
        steer_all, throttle_all = steer_all & keep, throttle_all & keep
    if use_tip:
        # A penalty row is no label at offset 0, whatever my hand did there.
        steer_all, throttle_all = steer_all.copy(), throttle_all.copy()
        pen = idx[tip_on.numpy()]
        steer_all[pen] = throttle_all[pen] = False
    steer_ok = torch.from_numpy(steer_all[idx])
    throttle_ok = torch.from_numpy(throttle_all[idx])

    # An accepted-auto tick counts its weight here, so a pile of weak forward
    # labels does not read as a class the strong ones are rarer against.
    weak_w = (np.asarray(data["accepted_w"], np.float64)[idx] if "accepted_w" in data
              else np.ones(len(idx)))
    tok = throttle_ok.numpy()
    counts = np.bincount(data["action"][idx][tok], weights=weak_w[tok],
                         minlength=len(pl.ACTIONS)).astype(np.float64)
    weights = gear_class_weights(counts, class_weight_power, gear_weights)
    ce = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32),
                             reduction="none")
    ce_mode = nn.CrossEntropyLoss(reduction="none")
    ce_steer = nn.CrossEntropyLoss(reduction="none")
    if chunk > 1:
        # Offset i's labels: the i-th next decision tick's gear and steer,
        # with that tick's own masks and weight. Offset 0 is the stock loss.
        fut = chunk_index(data, chunk)[idx]
        fut0 = np.where(fut >= 0, fut, 0)
        bad = pe.bad_marked(data)
        c_action = torch.from_numpy(data["action"][fut0].astype(np.int64))
        c_steer = torch.from_numpy(data["steer"][fut0].astype(np.float32))
        c_gear = torch.from_numpy(gears_all[fut0]) if per_gear else None
        c_tok = torch.from_numpy((fut >= 0) & throttle_all[fut0] & ~bad[fut0])
        c_sok = torch.from_numpy((fut >= 0) & steer_all[fut0] & ~bad[fut0])
        c_w = torch.from_numpy(tick_weights(data, fut0.ravel(), good_weight, takeover_weight)
                               .reshape(fut0.shape))
        c_ws = (c_w * torch.from_numpy(np.asarray(steer_tick_w, np.float32)[fut0])
                if steer_tick_w is not None else c_w)
        # Training ticks are decision ticks outside BAD marks, so offset 0 is
        # exactly the one-step labels.
        assert bool((c_tok[:, 0] == throttle_ok).all() and (c_sok[:, 0] == steer_ok).all())

    # OPTIMIZERS: "adam" folds the weight decay into the gradient, which on
    # a branch with a small gradient (the map's) is most of what it gets;
    # "adamw" decays the weights apart from it.
    def tip_term(logits0, throttle, bb):
        # The tip penalty (tip_penalty) on the batch's penalty rows, summed
        # over the batch's length: tip_weight is what one such row counts
        # against a labelled row's one.
        on = tip_on[bb]
        if not bool(on.any()):
            return torch.zeros(())
        p = torch.softmax(logits0[on], dim=1).gather(1, action[bb][on][:, None])[:, 0]
        return tip_weight * tip_penalty(p, throttle[on], torch).sum() / len(bb)

    # The contrast's turns on a generator of their own, so every other draw
    # is the one it always was.
    contrast_rng = np.random.default_rng([seed, 2])

    def row_loss(o, bb, tok, sok, sb, steer_only=False):
        # Per row, unweighted: the gear's cross entropy where it is a label
        # plus the steer's L1 where that is, offset 0. steer_only drops the
        # gear: the steer is read off the labelled gear's head either way.
        n = len(bb)
        lg = o[0].reshape(n, chunk, len(pl.ACTIONS))[:, 0]
        st = (torch.where(gear_of[bb] == GEAR_REVERSE, o[3][:, 0], o[2][:, 0]) if per_gear
              else o[2][:, 0])
        g = nn.functional.cross_entropy(lg, action[bb], reduction="none")
        gear = torch.zeros(()) if steer_only else torch.where(tok, g, torch.zeros(()))
        return gear + torch.where(sok, (st - sb).abs(), torch.zeros(()))

    def contrast_term(out, inputs, gb, bb, tok, sok, sb):
        # The goal contrast: the batch again with each goal turned away
        # (contrast_angles), costed where it explains my driving as well.
        # Steer only (contrast_steer_only): the wrong goal is costed on the
        # steer alone, read off the gear my hand chose, so changing gear for
        # it -- the way out D and E took -- buys nothing.
        rows = contrast_rows(gb[:, pg.PRESENT] > 0, tok, sok, contrast_steer_only)
        if not bool(rows.any()):
            return torch.zeros(())
        ang = torch.from_numpy(contrast_angles(contrast_rng, len(gb)).astype(np.float32))
        fake = torch.where(rows[:, None], rotate_goals(gb, ang, torch), gb)
        xb_, pb_, mo_, ib_, lb_, mb_ = inputs
        extra = [v for v in (ib_, lb_, mb_, fake) if v is not None]
        o_fake = model(xb_, pb_, mo_, *extra)
        lt = row_loss(out, bb, tok, sok, sb, contrast_steer_only)
        lf = row_loss(o_fake, bb, tok, sok, sb, contrast_steer_only)
        return contrast_weight * goal_contrast(lt[rows], lf[rows], xp=torch).sum() / len(bb)

    opt = (torch.optim.AdamW if optimizer == "adamw" else torch.optim.Adam)(
        model.parameters(), lr=3e-3, weight_decay=1e-4)
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
            mb = gb = None
            if uses_map:
                mb = map_inputs(data, idx[b.numpy()])
                blank = lidar_blanks(map_on[b.numpy()], map_dropout, rng)
                mb[blank] = 0.0
                gb = None
                if uses_goal:
                    shown = map_on[b.numpy()] & ~blank
                    gb = torch.from_numpy(goal_inputs(data, idx[b.numpy()], shown, goal_rng,
                                                      no_goal_share, path_m=goal_path_m))
                mb = torch.from_numpy(mb)
            # The mirror (MIRROR_MODES). Every input of a mirrored row is the
            # same moment seen in a mirror, and its steering labels change
            # sign; bb is which training tick each row of the batch is, and
            # sign is +1 as it is, -1 mirrored.
            if mirror == "both":
                mx, mp, mi, ml, mm, mg = mirror_inputs(xb, pb, ib, lb, mb, gb, imu_yaw)
                cat = lambda a, c: None if a is None else torch.cat([a, c])  # noqa: E731
                xb, pb, ib, lb, mb, gb = (cat(xb, mx), cat(pb, mp), cat(ib, mi),
                                          cat(lb, ml), cat(mb, mm), cat(gb, mg))
                bb = torch.cat([b, b])
                sign = torch.cat([torch.ones(len(b)), -torch.ones(len(b))])
            else:
                bb = b
                flipped = mirror == "coin" and torch.rand(1).item() < 0.5
                if flipped:
                    xb, pb, ib, lb, mb, gb = mirror_inputs(xb, pb, ib, lb, mb, gb, imu_yaw)
                sign = torch.full((len(b),), -1.0 if flipped else 1.0)
            sb = steer[bb] * sign
            db = torch.where(sign < 0, (len(pl.STEER_DIRS) - 1) - steer_dir[bb], steer_dir[bb])
            extra = [v for v in (ib, lb, mb, gb) if v is not None]
            out = model(xb, pb, mode_in[bb], *extra)
            logits, t_out, m_out = out[0], out[1], out[-1]
            tok, sok = throttle_ok[bb], steer_ok[bb]
            wb = w_all[bb]
            moving = tok & (action[bb] != pl.ACTIONS.index(pl.STOP))
            loss = torch.zeros(())
            if contrast_weight > 0 and gb is not None:
                loss = loss + contrast_term(out, (xb, pb, mode_in[bb], ib, lb, mb), gb, bb,
                                            tok, sok, sb)
            if chunk > 1:
                # The gear and steer losses averaged over the offsets, as ACT
                # averages its L1; throttle and mode stay one step. A per-gear
                # head's offset i is trained on the head of that offset's
                # own gear (steer_gears), the other head left alone.
                n = len(bb)
                lc = logits.reshape(n, chunk, len(pl.ACTIONS))
                sc = c_steer[bb] * sign[:, None]
                ta, sa, wc, ac = c_tok[bb], c_sok[bb], c_w[bb], c_action[bb]
                so = (torch.where(c_gear[bb] == GEAR_REVERSE, out[3], out[2]) if per_gear
                      else out[2])
                gear, turn = torch.zeros(()), torch.zeros(())
                for i in range(chunk):
                    if ta[:, i].any():
                        gear = gear + _weighted(ce(lc[ta[:, i], i], ac[ta[:, i], i]),
                                                wc[ta[:, i], i])
                    if sa[:, i].any():
                        turn = turn + _weighted((so[sa[:, i], i] - sc[sa[:, i], i]).abs(),
                                                c_ws[bb][sa[:, i], i])
                loss = loss + gear / chunk + turn / chunk
                if moving.any():
                    loss = loss + _weighted((t_out[moving, 0] - mag[bb][moving]).abs(), wb[moving])
                if sok.any():
                    loss = loss + _weighted(ce_mode(m_out[sok], mode[bb][sok]), w_mode[bb][sok])
                if use_tip:
                    loss = loss + tip_term(lc[:, 0], t_out[:, 0], bb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                continue
            if tok.any():
                loss = loss + _weighted(ce(logits[tok], action[bb][tok]), wb[tok])
            if moving.any():
                loss = loss + _weighted((t_out[moving, 0] - mag[bb][moving]).abs(), wb[moving])
            if sok.any() and direction:
                # Which way, on every steering label; how far, only where it
                # is a turn: a straight label says nothing about how hard.
                d_out, g_out = out[2], out[3]
                ws = w_steer[bb]
                loss = loss + _weighted(ce_steer(d_out[sok], db[sok]), ws[sok])
                turn = sok & (db != straight)
                if turn.any():
                    loss = loss + _weighted((g_out[turn, 0] - sb[turn].abs()).abs(), ws[turn])
            elif sok.any() and per_gear:
                so = torch.where(gear_of[bb] == GEAR_REVERSE, out[3][:, 0], out[2][:, 0])
                loss = loss + _weighted((so[sok] - sb[sok]).abs(), w_steer[bb][sok])
            elif sok.any():
                loss = loss + _weighted((out[2][sok, 0] - sb[sok]).abs(), w_steer[bb][sok])
            if sok.any():
                loss = loss + _weighted(ce_mode(m_out[sok], mode[bb][sok]), w_mode[bb][sok])
            if use_tip:
                loss = loss + tip_term(logits, t_out[:, 0], bb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model, weights.tolist()


OUTPUTS = output_names(pl.STEER_REGRESSION)


def export(model, representation, out_path, meta):
    import torch

    shape = ((1, 2, pl.INPUT_HEIGHT, pl.INPUT_WIDTH) if representation == pl.IMAGE
             else (1, profile_rows_of(getattr(model, "lowband", None)), cp.PROFILE_COLUMNS))
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
    if getattr(model, "uses_goal", False):
        args += (torch.zeros((1, pg.SIZE)),)
        names.append(pl.GOAL_INPUT)
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


def map_meta(uses_map, dropout=MAP_DROPOUT, act="relu"):
    """The map half of the sidecar: what policy.check_map reads. A uses_map
    model drives with the hand-written driving rules off (policy.py,
    PolicyDriver) and is refused on a car not running --slam."""
    if not uses_map:
        return {}
    return {"uses_map": True, **em.meta(), "map_dropout": float(dropout),
            "map_features": MAP_FEATURES, "map_act": act}


def goal_meta(uses_goal, no_goal_share=pg.NO_GOAL_SHARE):
    """The goal half of the sidecar: what policy.check_goal reads."""
    if not uses_goal:
        return {}
    return {"uses_goal": True, **pg.meta(), "goal_no_goal_share": float(no_goal_share),
            "goal_features": GOAL_FEATURES}


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


def chunk_meta(chunk=1, ensemble_m=0.0):
    """The chunk half of the sidecar: what policy.check_chunk reads. Nothing
    for a one-step model, as before the chunk heads."""
    if chunk <= 1:
        return {}
    return {"chunk": int(chunk), "ensemble_m": float(ensemble_m),
            "ensemble_link_s": pl.CHUNK_LINK_S}


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


def steer_meta(steer_head=pl.STEER_REGRESSION, band=STRAIGHT_BAND, stop_gear_s=STOP_GEAR_S):
    """The steering head's half of the sidecar: what policy.check_steer_head
    reads, and the outputs the graph has."""
    out = {"steer_head": steer_head, "outputs": output_names(steer_head)}
    if steer_head == pl.STEER_PER_GEAR:
        out["steer_gears"] = [pl.FORWARD, pl.REVERSE]
        out["steer_stop_gear_s"] = float(stop_gear_s)
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
    ap.add_argument("--goal", action="store_true",
                    help="feed a goal on the map in as an input of its own "
                         "(crawler/autopilot/policy_goal.py), trained on hindsight goals: "
                         "the truck's own pose a few seconds on. Needs --map, with crops "
                         "that carry their poses (tools/map_crops.py)")
    ap.add_argument("--no-goal-share", type=float, default=pg.NO_GOAL_SHARE,
                    help="with --goal: the share of goal-able ticks fed no goal "
                         f"(default {pg.NO_GOAL_SHARE})")
    ap.add_argument("--steer-head", choices=pl.STEER_HEADS, default=pl.STEER_REGRESSION,
                    help="regression: one tanh steer. direction: left/straight/right "
                         "plus how far, which cannot average two hard turns into straight")
    ap.add_argument("--straight-band", type=float, default=STRAIGHT_BAND,
                    help="with --steer-head direction: |steer| under this is the "
                         f"straight class (default {STRAIGHT_BAND} of full lock)")
    ap.add_argument("--mirror", choices=MIRROR_MODES, default="coin",
                    help="the left-right mirror augmentation: coin, each batch as it is or "
                         "mirrored; both, every batch both ways side by side; none")
    ap.add_argument("--map-act", choices=MAP_ACTS, default="relu",
                    help="with --map: the map branch's activation; relu trained to a dead "
                         "branch in every map model up to goal2, leaky does not")
    ap.add_argument("--optimizer", choices=OPTIMIZERS, default="adam",
                    help="adam folds the weight decay into the gradient (every model up to "
                         "goal2); adamw decays the weights apart from it")
    ap.add_argument("--stop-gear-s", type=float, default=STOP_GEAR_S,
                    help="with --steer-head per_gear: a stop's steering trains the head of "
                         "the gear the car moves in next within this many seconds, else the "
                         f"last one (default {STOP_GEAR_S})")
    ap.add_argument("--train-on-scorecard", action="store_true",
                    help="train on the scorecard's sessions too (tools/scorecard.txt), "
                         "which every model is otherwise kept off so they stay a fair test")
    ap.add_argument("--lowband", choices=cp.LOWBAND_LAYOUTS, default=None,
                    help="add the bottom third of the frame as a second column profile "
                         "under the footprint band's (column_profile.py): lower_third, "
                         "nearest/median/valid; lower_third_floor, against a flat floor "
                         "at the IMU's pitch")
    ap.add_argument("--tip-weight", type=float, default=0.0, metavar="W",
                    help="train against tips: drop every label on a tip and its run-up "
                         "(tip_marks) and put the tip penalty, the unlikelihood of the "
                         "gear the car took plus its throttle, on the run-up's moving "
                         "decisions at W a row. 0 is off")
    ap.add_argument("--goal-contrast", type=float, default=0.0, metavar="W",
                    help="the goal contrast at W: each goal tick again with its goal turned "
                         "90-180 degrees away, costed where that explains my driving within "
                         f"{GOAL_CONTRAST_MARGIN} of the real goal. Needs --goal; 0 is off")
    ap.add_argument("--contrast-steer-only", action="store_true",
                    help="the goal contrast on the steer alone, off the labelled gear's head")
    ap.add_argument("--goal-path-m", type=float, default=None, metavar="M",
                    help="draw goals M metres along the truck's own path instead of "
                         "hindsight goals k s ahead; the sidecar says M, and the car's "
                         "planner feeds the point M along its path to this model "
                         f"(policy_goal.lookahead_of; {pg.PATH_M:g} with none)")
    ap.add_argument("--fed-goals", default=None, metavar="DIR",
                    help="tools/fed_goals.py output: the goal the car was fed, in place of "
                         "the drawn one on every tick that had one")
    ap.add_argument("--tip-hygiene", action="store_true",
                    help="only drop the tip's labels, no penalty")
    ap.add_argument("--tip-deg", type=float, default=TIP_DEG,
                    help="degrees off the session's upright that is a tip")
    ap.add_argument("--tip-lead-s", type=float, default=TIP_LEAD_S,
                    help="seconds before a tip that are its run-up")
    ap.add_argument("--chunk", type=int, default=1, metavar="K",
                    help="action chunks: the gear and steer heads answer for this decision "
                         "and the next K-1, which the car blends (policy.ChunkEnsembler). "
                         "1 is the one-step model")
    ap.add_argument("--ensemble-m", type=float, default=0.0, metavar="M",
                    help="with --chunk: the car's blend weight exp(M * age), age 0 the "
                         "newest answer; under 0 trusts the newest more, 0 a plain mean")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--good-weight", type=float, default=GOOD_WEIGHT,
                    help="loss weight of a pad tick inside a GOOD /marks window "
                         f"(default {GOOD_WEIGHT}; BAD-window ticks are always left out)")
    ap.add_argument("--takeover-weight", type=float, default=TAKEOVER_WEIGHT,
                    help="loss weight of an intervention tick: a takeover, or manual "
                         f"right after auto (default {TAKEOVER_WEIGHT}; build the dataset "
                         "with policy_dataset.py --drive-logs DIR to have any)")
    ap.add_argument("--accepted-weight", type=float, default=0.0,
                    help="loss weight of an auto tick the model drove and I let stand "
                         f"(accepted_auto: no takeover {ACCEPT_AFTER_S:g} s after or "
                         f"{ACCEPT_BEFORE_S:g} s before); 0, the default, is none a label")
    ap.add_argument("--gear-weights", default=None, metavar="F,R,S",
                    help="the gear's class weights outright, in the order "
                         f"{','.join(pl.ACTIONS)}, in place of the computed ones")
    ap.add_argument("--accepted-own", default=None, metavar="DIR",
                    help="tools/auto_overrides.py output: with --accepted-weight, only the "
                         "ticks whose command was the model's own")
    ap.add_argument("--accept-after-s", type=float, default=ACCEPT_AFTER_S,
                    help="with --accepted-weight: an auto tick this long or less before a "
                         f"takeover is not let stand (default {ACCEPT_AFTER_S:g})")
    ap.add_argument("--intervention-share", type=float, default=None, metavar="P",
                    help="with --accepted-weight: set the takeover and accepted weights so "
                         "interventions are P of the training ticks by weight, record ticks "
                         "their own share, accepted auto the rest (Sirius)")
    ap.add_argument("--keyframe-weight", type=float, default=0.0, metavar="W",
                    help="the decisions a copycat of my past actions gets most wrong count "
                         "W times (policy_keyframes); 0, the default, none")
    ap.add_argument("--keyframe-share", type=float, default=pk.KEYFRAME_SHARE, metavar="P",
                    help="with --keyframe-weight: the top P of decisions by the copycat's error")
    ap.add_argument("--keyframe-history", type=int, default=pk.HISTORY, metavar="K",
                    help="with --keyframe-weight: past decisions the copycat sees")
    ap.add_argument("--keyframe-steer-only", action="store_true",
                    help="with --keyframe-weight: rank by the steer's error alone and weight "
                         "only the steer loss")
    ap.add_argument("--class-weight-power", type=float, default=1.0,
                    help="raise the gear's class weights to this power: 1 as they are "
                         "(the default), 0.5 half their log, 0 none")
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
    if not 1 <= args.chunk <= pl.CHUNK_MAX:
        print(f"--chunk is 1..{pl.CHUNK_MAX}, not {args.chunk}")
        return 2
    if args.chunk > 1 and args.steer_head == pl.STEER_DIRECTION:
        print("--chunk needs a regression steer head (regression or per_gear)")
        return 2
    if args.lowband is not None and args.representation != pl.PROFILE:
        print("--lowband is a second column profile: it needs --representation profile")
        return 2
    if args.chunk > 1 and not -10.0 <= args.ensemble_m <= 10.0:
        print(f"--ensemble-m is -10..10, not {args.ensemble_m}")
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
    if args.goal and not args.map:
        print("--goal needs --map: the goal is a point on the map")
        return 2
    if args.goal_contrast > 0 and (not args.goal or args.steer_head == pl.STEER_DIRECTION):
        print("--goal-contrast needs --goal and a regression steer head")
        return 2
    if (args.goal_path_m is not None or args.fed_goals or args.contrast_steer_only) and not args.goal:
        print("--goal-path-m, --fed-goals and --contrast-steer-only need --goal")
        return 2
    if args.fed_goals:
        try:
            fed = load_fed_goals(data, args.fed_goals)
        except ValueError as e:
            print(e)
            return 1
        print(f"fed goals: the car's own goal on {fed} of {len(data['t_ns'])} ticks")
    if args.goal and not bool(np.any(data["goal_poses"]["present"])):
        print(f"--goal: no crop in {args.map} carries its pose; replay them again "
              "with tools/map_crops.py")
        return 1

    import torch

    gear_weights = None
    if args.gear_weights:
        try:
            gear_weights = [float(v) for v in args.gear_weights.split(",")]
        except ValueError:
            gear_weights = []
        if len(gear_weights) != len(pl.ACTIONS) or min(gear_weights) <= 0:
            print(f"--gear-weights: {len(pl.ACTIONS)} positive numbers, "
                  f"{','.join(pl.ACTIONS)}")
            return 2
    if (args.accepted_own or args.intervention_share is not None) and args.accepted_weight <= 0:
        print("--accepted-own and --intervention-share need --accepted-weight")
        return 2
    if args.accepted_weight > 0:
        weak = accepted_auto(data, after_s=args.accept_after_s)
        if args.accepted_own:
            try:
                own, found = load_accepted_own(data, args.accepted_own)
            except ValueError as e:
                print(e)
                return 1
            print(f"accepted own: {int((weak & ~own).sum())} of {int(weak.sum())} accepted "
                  f"ticks were the car's rules, not the model ({found} sessions with a file)")
            weak = weak & own
        data["accepted"] = weak
        data["accepted_w"] = np.where(weak, args.accepted_weight, 1.0).astype(np.float32)
        print(f"accepted auto: {int(weak.sum())} decisions the model drove and I let stand, "
              f"at {args.accepted_weight:g} a tick")
    steer_ok, throttle_ok = label_masks(data, args.labels)
    labelled = steer_ok | throttle_ok
    # The scorecard's sessions are never trained on (unless asked): every
    # mask below is off `labelled`, so this one line keeps them out of the
    # folds, the final model and --val alike.
    held_card = [] if args.train_on_scorecard else [n for n in held_out_sessions()
                                                     if n in sessions]
    if held_card:
        labelled = labelled & ~scorecard_mask(data, held_card)
        print(f"scorecard: {len(held_card)} sessions kept out of training "
              f"({SCORECARD_FILE}, {REJECTED_FILE})")
    tip_on = args.tip_weight > 0 or args.tip_hygiene
    runup = label_off = None
    if tip_on:
        tipped, runup = tip_marks(data, args.tip_deg, args.tip_lead_s)
        label_off = tipped | runup
        before = int((labelled & data["decision"]).sum())
        labelled = tip_labelled(labelled, tipped, runup)
        print(f"tips: {int(tipped.sum())} ticks past {args.tip_deg:g} deg, {int(runup.sum())} "
              f"in the {args.tip_lead_s:g} s before; "
              f"{before - int((labelled & data['decision']).sum())} labelled decisions dropped")
    card_mask = (scorecard_mask(data, held_card) if held_card
                 else np.zeros(len(data["t_ns"]), dtype=bool))

    def tip_mask_of(out_of):
        if args.tip_weight <= 0:
            return None
        return tip_rows(data, out_of | card_mask, runup)

    if args.intervention_share is not None:
        # Off every labelled decision, the folds' and the final model's alike.
        every = training_mask(data, np.zeros(len(data["t_ns"]), bool), labelled)
        try:
            tw, aw = intervention_share_weights(
                data, every & label_masks(data, args.labels)[1], args.intervention_share)
        except ValueError as e:
            print(e)
            return 1
        args.takeover_weight = tw
        args.accepted_weight = aw
        data["accepted_w"] = np.where(data["accepted"], aw, 1.0).astype(np.float32)
        print(f"intervention share {args.intervention_share:g}: takeover weight {tw:.3f}, "
              f"accepted weight {aw:.3f}")

    keyframe = steer_tick_w = None
    if args.keyframe_steer_only and args.keyframe_weight <= 0:
        print("--keyframe-steer-only needs --keyframe-weight")
        return 2
    if args.keyframe_weight > 0:
        if not 0 < args.keyframe_share < 1 or args.keyframe_history < 1:
            print("--keyframe-share is between 0 and 1, --keyframe-history 1 or more")
            return 2
        # Off every labelled decision, the folds' and the final model's
        # alike, and the same for every --seed: the copycat's own seed is 0.
        every = training_mask(data, np.zeros(len(data["t_ns"]), bool), labelled)
        k_steer, k_throttle = label_masks(data, args.labels)
        ape, parts = pk.action_prediction_error(data, every, k_steer, k_throttle,
                                                k=args.keyframe_history)
        if args.keyframe_steer_only:
            # The steer's own surprises, on the steer's loss alone: the gear
            # keeps the weights it had, so a stop is no likelier for it.
            ape = parts["steer"]
        kf_w, keyframe = pk.keyframe_weights(ape, every, args.keyframe_weight,
                                             args.keyframe_share)
        if args.keyframe_steer_only:
            steer_tick_w = kf_w
        else:
            data["keyframe_w"] = kf_w
        past = pk.history_index(data, args.keyframe_history)
        print(f"keyframes: {int(keyframe.sum())} of {int(every.sum())} decisions at "
              f"{args.keyframe_weight:g}, {pk.gear_changes(data, keyframe, past)} of them a "
              f"gear change; copycat mean Brier {parts['gear'][every & k_throttle].mean():.3f}, "
              f"steer squared error {parts['steer'][every & k_steer].mean():.4f}")
    data["x"] = encode_frames(data["frames"], args.representation, args.lowband,
                              frame_pitch(data, args.lowband))
    fit = dict(representation=args.representation, label_mode=args.labels,
               uses_prev=args.prev, epochs=args.epochs, seed=args.seed,
               good_weight=args.good_weight, switch_weight=args.switch_weight,
               uses_imu=args.imu, imu_yaw=not args.imu_no_yaw,
               imu_tilt=args.imu_tilt, steer_head=args.steer_head,
               straight_band=args.straight_band, takeover_weight=args.takeover_weight,
               uses_lidar=args.lidar, lidar_dropout=args.lidar_dropout,
               uses_map=bool(args.map), map_dropout=args.map_dropout, chunk=args.chunk,
               uses_goal=args.goal, no_goal_share=args.no_goal_share,
               mirror=args.mirror, stop_gear_s=args.stop_gear_s,
               map_act=args.map_act, optimizer=args.optimizer, lowband=args.lowband,
               tip_weight=args.tip_weight, label_off=label_off,
               contrast_weight=args.goal_contrast, goal_path_m=args.goal_path_m,
               contrast_steer_only=args.contrast_steer_only,
               class_weight_power=args.class_weight_power, gear_weights=gear_weights,
               steer_tick_w=steer_tick_w)
    tmp = args.out if args.fold else os.path.splitext(args.out)[0] + ".fold.onnx"

    cv = None
    if args.blocks and args.fold != "final":
        folds = []
        for k, (held, guard) in enumerate(time_blocks(data, args.blocks)):
            if args.fold is not None and k + 1 != int(args.fold):
                continue
            mask = training_mask(data, guard, labelled)
            model, _ = train(data, train_mask=mask, tip_mask=tip_mask_of(guard), **fit)
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
    if args.goal:
        # A tick that can have a hindsight goal at all: one at NEAR_S on
        # (the shortest k) is there. Fewer at longer k, near a file's end.
        rows = np.flatnonzero(train_mask & data["map_on"])
        _, _, ok = pg.hindsight_goals(data["map_row"][rows], pg.NEAR_S, data["goal_poses"])
        print(f"goal: a hindsight goal possible on {int(ok.sum())} of {int(train_mask.sum())} "
              f"training ticks, none on {args.no_goal_share:.0%} of them and on every "
              f"map-blanked one, k drawn from {pg.NEAR_S:g}-{pg.FAR_S:g} s each batch")
    tip_mask = tip_mask_of(in_val)
    if tip_mask is not None:
        print(f"tips: the penalty on {int(tip_mask.sum())} run-up decisions in a moving gear, "
              f"at {args.tip_weight:g} a row")
    model, weights = train(data, train_mask=train_mask, tip_mask=tip_mask, **fit)
    params = int(sum(p.numel() for p in model.parameters()))
    shares = loss_shares(data, train_mask, args.labels, args.good_weight,
                         args.takeover_weight, weights)
    print(format_shares(shares, args.takeover_weight))
    keyframe_meta = None
    if keyframe is not None:
        idx = np.flatnonzero(train_mask)
        base = tick_weights({k: v for k, v in data.items() if k != "keyframe_w"}, idx,
                            args.good_weight, args.takeover_weight)
        if args.keyframe_steer_only:
            # Of the steer loss's weight, over the ticks with a steer label.
            on = label_masks(data, args.labels)[0][idx]
            kshare = pk.weight_share(base[on], kf_w[idx][on], keyframe[idx][on])
        else:
            kshare = pk.weight_share(base, kf_w[idx], keyframe[idx])
        keyframe_meta = {"weight": args.keyframe_weight, "share": args.keyframe_share,
                         "history": args.keyframe_history,
                         "steer_only": bool(args.keyframe_steer_only),
                         "train_ticks": int((train_mask & keyframe).sum()),
                         "loss_weight_share": kshare}
        print(f"keyframes: {keyframe_meta['train_ticks']} of {int(train_mask.sum())} training "
              f"decisions, {kshare:.1%} of the summed tick weight")

    meta = {**_base_meta(meta_in, args, val_ids, val_names), **mode_meta(args.switch_weight),
            "train_decisions": int(train_mask.sum()),
            "class_weights": weights,
            "good_marked_train_ticks": int((train_mask & (pe.marks_of(data) == pe.MARK_GOOD)).sum()),
            "bad_marked_left_out": int((data["decision"] & pe.bad_marked(data)).sum()),
            "intervention_train_ticks": int((train_mask & intervention_ticks(data)).sum()),
            "intervention_loss_share": {head: share for head, (_, _, share) in shares.items()},
            "parameters": params,
            **tip_meta(args.tip_weight, args.tip_deg, args.tip_lead_s,
                       hygiene=tip_on),
            **({"tip_penalty_rows": int(tip_mask.sum())} if tip_mask is not None else {}),
            "goal_contrast": ({"weight": args.goal_contrast, "margin": GOAL_CONTRAST_MARGIN,
                               "steer_only": bool(args.contrast_steer_only)}
                              if args.goal_contrast > 0 else None),
            "goal_draw": goal_draw_meta(args.goal_path_m),
            "goal_fed": bool(args.fed_goals),
            "accepted_auto": ({"weight": args.accepted_weight, "after_s": args.accept_after_s,
                               "before_s": ACCEPT_BEFORE_S, "hole_s": ACCEPT_HOLE_S,
                               "train_ticks": int((train_mask & data["accepted"]).sum()),
                               "own_only": bool(args.accepted_own),
                               "intervention_share": args.intervention_share}
                              if args.accepted_weight > 0 else None),
            "class_weight_power": args.class_weight_power,
            "gear_weights": gear_weights,
            **({"keyframes": keyframe_meta} if keyframe_meta is not None else {}),
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
                   getattr(args, "map_dropout", MAP_DROPOUT),
                   getattr(args, "map_act", "relu")),
        **goal_meta(bool(getattr(args, "goal", False)),
                    getattr(args, "no_goal_share", pg.NO_GOAL_SHARE)),
        **steer_meta(getattr(args, "steer_head", pl.STEER_REGRESSION),
                     getattr(args, "straight_band", STRAIGHT_BAND),
                     getattr(args, "stop_gear_s", STOP_GEAR_S)),
        **chunk_meta(getattr(args, "chunk", 1), getattr(args, "ensemble_m", 0.0)),
        **pl.lowband_meta(getattr(args, "lowband", None)),
        "steer_scale": pl.STEER_SCALE,
        "labels": args.labels,
        "train_sessions": [s for i, s in enumerate(sessions) if i not in val_ids
                           and (getattr(args, "train_on_scorecard", False)
                                or s not in held_out_sessions())],
        "val_sessions": val_names,
        "good_weight": args.good_weight,
        "takeover_weight": getattr(args, "takeover_weight", TAKEOVER_WEIGHT),
        "epochs": args.epochs,
        "seed": args.seed,
        "mirror": getattr(args, "mirror", "coin"),
        "optimizer": getattr(args, "optimizer", "adam"),
        "scorecard_sessions": ([] if getattr(args, "train_on_scorecard", False)
                               else [n for n in scorecard_sessions() if n in sessions]),
        "rejected_sessions": ([] if getattr(args, "train_on_scorecard", False)
                              else [n for n in rejected_sessions() if n in sessions]),
    }


if __name__ == "__main__":
    sys.exit(main())
