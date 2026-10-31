"""Keyframes: the decisions a copycat of my own last few actions gets wrong.

Wen et al. 2021, "Keyframe-Focused Visual Imitation Learning" (arXiv
2106.06452): a policy that can see its own past actions learns to repeat
them, and is wrong exactly where I changed my mind -- the moment I turned
toward the waypoint, or dropped into reverse. Their fix is a weight, worked
out once before training:

  1. a small MLP copycat predicts each decision's action from the k
     decisions before it and nothing else (copycat_inputs),
  2. its action prediction error (APE) on that decision is how much of the
     decision the past did not explain (action_prediction_error),
  3. the top `share` of decisions by APE count `weight` times in the loss,
     the rest 1 (keyframe_weights; their step function, W 5 on the top 10%).

The car's action is a gear (forward / reverse / stop) and a steer in -1..1,
so the APE is two parts, each scaled to its mean over the decisions it is
fitted on so neither owns the ranking:

  gear   the Brier score of the copycat's gear softmax, sum (p - onehot)^2,
         where the gear is a label
  steer  (copycat steer - steer)^2, where the steer is a label

APE = gear / mean(gear) + steer / mean(steer). The copycat is out of fold:
each session's APE comes off a copycat that never saw that session, so it
cannot memorise a drive's changes and call them predictable. Arrays in,
arrays out; the copycat is plain numpy.
"""

import numpy as np

from crawler.autopilot import policy as pl

HISTORY = 6           # past decisions the copycat sees; their H = 6 on CARLA
KEYFRAME_SHARE = 0.1  # their THR: the top 10% of APE
FOLDS = 5
HIDDEN = 64
EPOCHS = 30
STEP_FEATURES = len(pl.ACTIONS) + 2   # gear one-hot, steer, present


def history_index(data, k=HISTORY, link_s=pl.CHUNK_LINK_S):
    """(N, k) dataset rows: column i is tick j's (i + 1)-th previous decision
    tick in the same session, every step of the chain under link_s (the gap
    the chunk labels break at too); -1 past a gap, the session's start, or on
    a tick that is no decision."""
    n = len(data["t_ns"])
    past = np.full((n, k), -1, np.int64)
    t = np.asarray(data["t_ns"]) / 1e9
    session = np.asarray(data["session"])
    o = np.lexsort((t, session))
    o = o[np.asarray(data["decision"], dtype=bool)[o]]
    ok = (np.diff(session[o]) == 0) & (np.diff(t[o]) <= link_s)   # o[j] -> o[j + 1]
    for i in range(1, k + 1):
        good = np.ones(max(len(o) - i, 0), dtype=bool)
        for s in range(i):
            good &= ok[s:len(ok) - i + 1 + s]
        past[o[i:][good], i - 1] = o[:len(o) - i][good]
    return past


def copycat_inputs(data, rows, past):
    """(len(rows), k * STEP_FEATURES) float32: each past decision's gear
    one-hot, steer and a 1 that it is there; all zeros where it is not."""
    p = past[rows]
    have = p >= 0
    p0 = np.where(have, p, 0)
    gear = np.eye(len(pl.ACTIONS), dtype=np.float32)[np.asarray(data["action"])[p0]]
    steer = np.asarray(data["steer"], np.float32)[p0][..., None]
    feats = np.concatenate([gear, steer, np.ones_like(steer)], axis=-1)
    feats = np.where(have[..., None], feats, 0.0)
    return feats.reshape(len(rows), -1).astype(np.float32)


def fit_copycat(inputs, gear, steer, gear_ok, steer_ok, seed=0, epochs=EPOCHS,
                hidden=HIDDEN, lr=3e-3, batch=512):
    """A two-layer MLP from the past actions alone to (gear logits, steer),
    trained with Adam on cross entropy where the gear is a label plus squared
    error (of a tanh steer) where the steer is. Plain numpy on a generator
    of its own, so it never moves torch's draws. Returns
    predict(inputs) -> (gear probs, steer)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(inputs, np.float64)
    n_g = len(pl.ACTIONS)
    params = []
    for a, b in ((x.shape[1], hidden), (hidden, hidden), (hidden, n_g + 1)):
        params += [rng.normal(0.0, np.sqrt(2.0 / a), (a, b)), np.zeros(b)]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    g = np.asarray(gear, np.int64)
    s = np.asarray(steer, np.float64)
    gok = np.asarray(gear_ok, bool)
    sok = np.asarray(steer_ok, bool)
    onehot = np.eye(n_g)

    def forward(xb):
        h1 = np.maximum(xb @ params[0] + params[1], 0.0)
        h2 = np.maximum(h1 @ params[2] + params[3], 0.0)
        return h1, h2, h2 @ params[4] + params[5]

    def softmax(z):
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    step = 0
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for start in range(0, len(x), batch):
            b = order[start:start + batch]
            xb = x[b]
            h1, h2, out = forward(xb)
            d_out = np.zeros_like(out)
            gb, sb = gok[b], sok[b]
            if gb.any():
                d_out[gb, :-1] = (softmax(out[gb, :-1]) - onehot[g[b][gb]]) / gb.sum()
            if sb.any():
                th = np.tanh(out[sb, -1])
                d_out[sb, -1] = 2.0 * (th - s[b][sb]) * (1.0 - th ** 2) / sb.sum()
            d_h2 = (d_out @ params[4].T) * (h2 > 0)
            d_h1 = (d_h2 @ params[2].T) * (h1 > 0)
            grads = (xb.T @ d_h1, d_h1.sum(0), h1.T @ d_h2, d_h2.sum(0),
                     h2.T @ d_out, d_out.sum(0))
            step += 1
            for i, gr in enumerate(grads):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr ** 2
                params[i] -= (lr * (m[i] / (1 - 0.9 ** step))
                              / (np.sqrt(v[i] / (1 - 0.999 ** step)) + 1e-8))

    def predict(new):
        out = forward(np.asarray(new, np.float64))[2]
        return softmax(out[:, :-1]), np.tanh(out[:, -1])
    return predict


def session_folds(sessions, n_folds=FOLDS, seed=0):
    """{session id: fold} with the sessions dealt round robin in a fixed
    shuffle; as many folds as sessions when there are fewer."""
    ids = np.unique(np.asarray(sessions))
    ids = ids[np.random.default_rng(seed).permutation(len(ids))]
    n = max(1, min(n_folds, len(ids)))
    return {int(s): i % n for i, s in enumerate(ids)}, n


def action_prediction_error(data, fit_mask, steer_ok, throttle_ok, k=HISTORY,
                            n_folds=FOLDS, seed=0, epochs=EPOCHS):
    """(ape, parts) on every tick: the APE (module doc) of each fit_mask tick
    off a copycat trained on the other folds' fit_mask ticks, 0 elsewhere.
    parts: {"gear": raw Brier, "steer": raw squared error} the same way."""
    n = len(data["t_ns"])
    fit_mask = np.asarray(fit_mask, bool)
    steer_ok = np.asarray(steer_ok, bool) & fit_mask
    throttle_ok = np.asarray(throttle_ok, bool) & fit_mask
    past = history_index(data, k)
    rows = np.flatnonzero(fit_mask)
    session = np.asarray(data["session"])
    fold_of, folds = session_folds(session[rows], n_folds, seed)
    fold = np.array([fold_of[int(s)] for s in session[rows]], np.int64)
    inputs = copycat_inputs(data, rows, past)
    action = np.asarray(data["action"], np.int64)[rows]
    steer = np.asarray(data["steer"], np.float64)[rows]
    gok, sok = throttle_ok[rows], steer_ok[rows]
    brier = np.zeros(len(rows))
    sq = np.zeros(len(rows))
    for f in range(folds):
        test = fold == f
        train = ~test if folds > 1 else np.ones(len(rows), bool)
        predict = fit_copycat(inputs[train], action[train], steer[train], gok[train],
                              sok[train], seed=seed + f, epochs=epochs)
        probs, s_hat = predict(inputs[test])
        onehot = np.eye(len(pl.ACTIONS))[action[test]]
        brier[test] = ((probs - onehot) ** 2).sum(axis=1)
        sq[test] = (s_hat - steer[test]) ** 2
    brier = np.where(gok, brier, 0.0)
    sq = np.where(sok, sq, 0.0)
    gear_mean = brier[gok].mean() if gok.any() else 1.0
    steer_mean = sq[sok].mean() if sok.any() else 1.0
    ape_rows = brier / max(gear_mean, 1e-12) + sq / max(steer_mean, 1e-12)
    ape, parts = np.zeros(n), {"gear": np.zeros(n), "steer": np.zeros(n)}
    ape[rows], parts["gear"][rows], parts["steer"][rows] = ape_rows, brier, sq
    return ape, parts


def keyframe_weights(ape, fit_mask, weight, share=KEYFRAME_SHARE):
    """(weights, keyframe): `weight` on the top `share` of fit_mask ticks
    by APE, 1 on every other tick; keyframe marks the upweighted ones."""
    fit_mask = np.asarray(fit_mask, bool)
    keyframe = np.zeros(len(ape), bool)
    if fit_mask.any() and share > 0:
        cut = np.quantile(ape[fit_mask], 1.0 - share)
        keyframe = fit_mask & (ape >= cut) & (ape > 0)
    return np.where(keyframe, float(weight), 1.0).astype(np.float32), keyframe


def weight_share(base_w, keyframe_w, keyframe):
    """The share of the summed loss weight (the other weights times the
    keyframe one) that sits on keyframes."""
    w = np.asarray(base_w, np.float64) * np.asarray(keyframe_w, np.float64)
    total = w.sum()
    return float(w[np.asarray(keyframe, bool)].sum() / total) if total else 0.0


def gear_changes(data, keyframe, past):
    """How many keyframes changed gear from the decision before them."""
    rows = np.flatnonzero(keyframe & (past[:, 0] >= 0))
    action = np.asarray(data["action"])
    return int((action[rows] != action[past[rows, 0]]).sum())
