"""Train a driving policy off a policy_dataset.py .npz, and export it to ONNX.

    python tools/policy_train.py DATA.npz OUT.onnx --representation profile --val SESSION

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
  steer     (1, 1)  tanh, front steering as a fraction of max_turn

Each group is trained only where it is a label: steering where steer_by is
"pad", direction and throttle where throttle_by is "pad". --labels any
drops that mask, which is what cloning the autopilot as a smoke test needs.

/marks on pad-driven ticks (policy_dataset.mark_flags): a tick in a BAD
window is out of training (and out of the val numbers); a tick in a GOOD
window counts --good-weight times in every head's loss. 2 by default: enough
that a stretch I said was right pulls, not so much that a few seconds of it
outvote the rest of the session.

Both representations take the same two inputs, so the runtime has one code
path: x (the frame, as policy.encode_input() builds it) and prev (the last
decision, zeros when the sidecar says uses_prev_action is false).
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
import policy_eval as pe  # noqa: E402

GOOD_WEIGHT = 2.0


def build_model(representation):
    import torch
    from torch import nn

    class Heads(nn.Module):
        def __init__(self, width):
            super().__init__()
            # prev is always wired in, so every exported graph has the same
            # two inputs; a model trained without it is fed zeros on both
            # sides and its weights there never see a gradient.
            self.trunk = nn.Sequential(nn.Linear(width + pl.PREV_SIZE, 64), nn.ReLU())
            self.logits = nn.Linear(64, len(pl.ACTIONS))
            self.throttle = nn.Linear(64, 1)
            self.steer = nn.Linear(64, 1)

        def forward(self, features, prev):
            h = self.trunk(torch.cat([features, prev], dim=1))
            return (self.logits(h), torch.sigmoid(self.throttle(h)),
                    torch.tanh(self.steer(h)))

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

        def forward(self, x, prev):
            return self.heads(self.features(x), prev)

    class ProfileNet(nn.Module):
        """(1, 3, 32) -> two 1-D convs along the columns -> the heads."""

        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(cp.PROFILE_FEATURES, 16, 5, padding=2), nn.ReLU(),
                nn.Conv1d(16, 16, 5, stride=2, padding=2), nn.ReLU(),
                nn.Flatten())
            self.heads = Heads(16 * (cp.PROFILE_COLUMNS // 2))

        def forward(self, x, prev):
            return self.heads(self.features(x), prev)

    return ImageNet() if representation == pl.IMAGE else ProfileNet()


def encode_frames(frames, representation):
    return np.concatenate([pl.encode_input(representation, f) for f in frames])


def flip(x, prev):
    """Mirror left-right: the frame's last axis, and every steering sign."""
    import torch

    prev = prev.clone()
    prev[:, -1] = -prev[:, -1]
    return torch.flip(x, dims=[-1]), prev


def tick_weights(data, idx, good_weight):
    """Per training tick loss weight: good_weight inside a GOOD mark, else 1."""
    return np.where(pe.marks_of(data)[idx] == pe.MARK_GOOD,
                    float(good_weight), 1.0).astype(np.float32)


def _weighted(values, w):
    return (values * w).sum() / w.sum()


def training_mask(data, in_val, labelled):
    """Decision ticks outside val, labelled, and not inside a BAD mark."""
    return ~in_val & data["decision"] & labelled & ~pe.bad_marked(data)


def train(data, representation, train_mask, label_mode, uses_prev, epochs, seed,
          good_weight=GOOD_WEIGHT):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(representation)

    idx = np.flatnonzero(train_mask)
    x_all = torch.from_numpy(data["x"])
    x = x_all[data["frame"][idx]]
    prev = torch.from_numpy(data["prev"][idx] if uses_prev
                            else np.zeros_like(data["prev"][idx]))
    action = torch.from_numpy(data["action"][idx].astype(np.int64))
    mag = torch.from_numpy(data["magnitude"][idx])
    steer = torch.from_numpy(data["steer"][idx])
    w_all = torch.from_numpy(tick_weights(data, idx, good_weight))
    if label_mode == "any":
        steer_ok = torch.ones(len(idx), dtype=torch.bool)
        throttle_ok = torch.ones(len(idx), dtype=torch.bool)
    else:
        steer_ok = torch.from_numpy(data["steer_label"][idx])
        throttle_ok = torch.from_numpy(data["throttle_label"][idx])

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

    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    batch = 64
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(idx))
        for start in range(0, len(idx), batch):
            b = order[start:start + batch]
            xb, pb = x[b], prev[b]
            sb = steer[b]
            if torch.rand(1).item() < 0.5:
                xb, pb = flip(xb, pb)
                sb = -sb
            logits, t_out, s_out = model(xb, pb)
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
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model, weights.tolist()


def export(model, representation, out_path, meta):
    import torch

    shape = ((1, 2, pl.INPUT_HEIGHT, pl.INPUT_WIDTH) if representation == pl.IMAGE
             else (1, cp.PROFILE_FEATURES, cp.PROFILE_COLUMNS))
    torch.onnx.export(
        model, (torch.zeros(shape), torch.zeros((1, pl.PREV_SIZE))), out_path,
        input_names=["x", "prev"], output_names=["logits", "throttle", "steer"],
        opset_version=17, dynamo=False)
    with open(os.path.splitext(out_path)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data")
    ap.add_argument("out", help="OUT.onnx; OUT.json is written beside it")
    ap.add_argument("--representation", choices=pl.REPRESENTATIONS, default=pl.PROFILE)
    ap.add_argument("--val", action="append", default=None,
                    help="session name to hold out (repeatable); default the last one")
    ap.add_argument("--labels", choices=("pad", "any"), default="pad",
                    help="pad: only hand-driven groups are labels. any: clone "
                         "whatever was commanded (the autopilot smoke test)")
    ap.add_argument("--prev", action="store_true",
                    help="feed the previous decision in as a second input")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--good-weight", type=float, default=GOOD_WEIGHT,
                    help="loss weight of a pad tick inside a GOOD /marks window "
                         f"(default {GOOD_WEIGHT}; BAD-window ticks are always left out)")
    args = ap.parse_args(argv)

    import torch

    data = pe.load(args.data)
    meta_in = data["meta"]
    sessions = meta_in["sessions"]
    val_names = args.val or [sessions[-1]]
    unknown = [v for v in val_names if v not in sessions]
    if unknown:
        print(f"no such session {unknown}; have {sessions}")
        return 1
    val_ids = [sessions.index(v) for v in val_names]
    in_val = np.isin(data["session"], val_ids)
    if args.labels == "pad":
        labelled = data["steer_label"] | data["throttle_label"]
    else:
        labelled = np.ones(len(in_val), dtype=bool)
    train_mask = training_mask(data, in_val, labelled)
    if not train_mask.any():
        print("no training ticks: every labelled decision is in --val, or nothing is "
              "pad-labelled (legacy files are all autopilot: try --labels any)")
        return 1

    data["x"] = encode_frames(data["frames"], args.representation)
    model, weights = train(data, args.representation, train_mask, args.labels,
                           args.prev, args.epochs, args.seed, args.good_weight)
    params = int(sum(p.numel() for p in model.parameters()))

    meta = {
        "input_contract": meta_in["input_contract"],
        "representation": args.representation,
        "actions": list(pl.ACTIONS),
        "uses_prev_action": bool(args.prev),
        "outputs": ["logits", "throttle", "steer"],
        "steer_scale": meta_in["steer_scale"],
        "labels": args.labels,
        "train_sessions": [s for i, s in enumerate(sessions) if i not in val_ids],
        "val_sessions": val_names,
        "train_decisions": int(train_mask.sum()),
        "class_weights": weights,
        "good_weight": args.good_weight,
        "good_marked_train_ticks": int((train_mask & (pe.marks_of(data) == pe.MARK_GOOD)).sum()),
        "bad_marked_left_out": int((data["decision"] & pe.bad_marked(data)).sum()),
        "parameters": params,
        "epochs": args.epochs,
        "seed": args.seed,
    }
    with torch.no_grad():
        export(model, args.representation, args.out, meta)

    model_rt = pl.PolicyModel(args.out)
    scores = pe.score(model_rt, data, val_ids, args.labels, train_mask)
    meta["val"] = scores
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{args.representation}: {params} parameters, "
          f"{os.path.getsize(args.out)} bytes, {int(train_mask.sum())} training decisions")
    print(pe.format_scores(scores))
    return 0


if __name__ == "__main__":
    sys.exit(main())
