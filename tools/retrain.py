#!/usr/bin/env python3
"""Retrain the driving policy in one command: recordings in, a candidate registered.

    scripts/retrain.sh                       # pull, build, train, grade, register
    scripts/retrain.sh --no-pull             # what is on this Mac already
    scripts/retrain.sh -- --imu-no-yaw       # anything after -- goes to policy_train.py
    scripts/retrain.sh --dry-run             # everything but the register

What it does, in order, and the time each stage took is printed at the end:

  pull      scripts/pull-recordings.sh, unless --no-pull. A car that cannot be
            reached is a warning, and the run goes on with what is here.
  scan      every crawler-*.mcap in ~/crawler-data/mcap (never a .orig, the
            copy mcap_check kept before a repair), each one decoded once into
            the cache, the decodes spread over --jobs processes.
  filter    the junk rules below, off numbers the decode wrote down.
  dataset   policy_dataset.py's own main() over the sessions that are left,
            with its MCAP reader swapped for one that reads the cache. Every
            tick, label and feature is the code a plain policy_dataset.py run
            uses, so training and the car cannot drift apart through here.
  train     policy_train.py once per held-out block (--fold K) and once for
            the final model on everything (--fold final), all at once, one
            process each at TRAIN_THREADS (one: measured fastest), set outright.
            The frames go through encode_frames once, here, not once a process.
  grade     while those train: the model the car runs (deployed_entry: the
            newest drive log row's model, else the registry's "deployed",
            else the newest "Deployed" note; never an ssh) on the same
            held-out ticks. Then one table, the candidate against it.
  register  the final model as a candidate, the fold table in its note. Never
            a deploy: that stays scripts/models.sh deploy, by hand.

The cache. ~/crawler-data/cache/<session>.<key>.pkl is that file's decoded
policy_adapters.Session, and <session>.<key>.json the numbers the junk rules
read. The key hashes the file's size and mtime and the source of every module
of this repo the decode can reach from policy_dataset (policy_adapters, the
policy, the IMU features, ...): edit any of it and every session decodes again;
add a recording and only that one does. A pickle and not an .npz, because a
Session is lists of NamedTuples, and an .npz would need a schema here that the
next input (the lidar) would have to edit. The dataset itself is cached under
the hash of everything that went into it, so a run with nothing new skips
straight to training.

The junk rules. A session trains unless one of these says it is junk:
  stub        the file is under STUB_BYTES: a run that never got going.
  no /footprint  nothing in it is on the policy's input contract: a drive in
              manual, a legacy recording, or a record session from before.
  old caps    policy_dataset.caps_verdict leaves it out: recorded at speed
              caps the car no longer drives at.
  no labels   fewer than MIN_USABLE decision ticks my hand labelled outside a
              BAD mark. 1.5 s: the shortest real manoeuvre in the set is a
              2.1 s back-up, and every session under the line has none at all.
  excluded    its name starts with a prefix in ~/crawler-data/exclude.txt
              (policy_dataset.EXCLUDE_FILE, --exclude-file): a drive I said
              never to learn from, the takeovers I made because the car kept
              stopping on 10-29 the first of them. A rule with a window
              (prefix@FROM..TO) keeps the session and policy_dataset drops
              only its ticks inside the window. The windows are in the
              dataset's cache key, so adding one builds the set again.

The folds are policy_train.py's --blocks N, so the numbers are the ones a plain
`policy_train.py --blocks 5` prints. The deployed model is graded as it is, a
file, so on a block it trained on its score is a fit and not a test; the
"seen" column says how much of each block that is. The last row is the fair
one: the record sessions it never trained on, against a candidate trained (in
one more process, beside the folds) on every session but those, as the
deployed model was. A fold model there would be missing the rest of its
block too, and lose for that alone.

The recipe: EPOCHS and TAKEOVER_WEIGHT below, each with the runs that chose
it. Anything after -- goes to policy_train.py after them and wins.
"""

import argparse
import contextlib
import datetime
import glob
import hashlib
import json
import os
import pickle
import shlex
import subprocess
import sys
import tempfile
import time
import types

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS)
sys.path[:0] = [p for p in (REPO, TOOLS) if p not in sys.path]

import models as md  # noqa: E402
import policy_adapters as pa  # noqa: E402
import policy_dataset as pd  # noqa: E402
import policy_eval as pe  # noqa: E402
import policy_train as pt  # noqa: E402
import pull_mcaps as pm  # noqa: E402
from crawler.autopilot import policy as pl  # noqa: E402

DATA_ROOT = os.path.expanduser("~/crawler-data")
MCAP_DIR = md.MCAP_DIR
MODELS_DIR = md.MODELS_DIR
CACHE_DIR = os.path.expanduser(os.environ.get("CRAWLER_CACHE", os.path.join(DATA_ROOT, "cache")))
RUNS_DIR = os.path.expanduser(os.environ.get("CRAWLER_RUNS", os.path.join(DATA_ROOT, "retrain")))

STUB_BYTES = 64 * 1024
MIN_USABLE = 15
BLOCKS = 5
# 20 graded better held out than 60 on the DAgger-3 folds: 0.650 against 0.628.
# Again on the 33-session set (2026-10-29), five folds, two seeds each, pooled:
# direction 0.856 against 0.832 at takeover weight 1, 0.847 against 0.815 at 3,
# and reverse recall 0.611 against 0.544 and 0.581 against 0.498.
EPOCHS = 20
# policy_train.py's --takeover-weight, before anything after --, so one given
# there wins. Same set, 20 epochs, four seeds, pooled over five folds: 1
# against 3, steering MAE 0.521 against 0.537 (lower on every seed), direction
# 0.852 against 0.844, reverse precision 0.398 against 0.380, balanced 0.517
# both. On the one session dagger2-1029 never saw (702 ticks, 53 of them
# reverse) 3 read better on reverse recall, 0.731 against 0.580, too few
# reverses to outweigh the folds.
TAKEOVER_WEIGHT = 1.0
JOBS = 12
# Torch threads per training process, the folds and the final model alike.
# One fold, 5 epochs, a loaded Mac: 1 thread 0.47 s an epoch, 2 1.63, 4 2.41,
# 8 4.29; MPS 0.42 with a 3.5 s warm-up, so the CPU.
TRAIN_THREADS = 1
# policy_train.py flags this script sets itself, and so refuses after --.
OWN_FLAGS = ("--blocks", "--fold", "--epochs", "--val")
# Long enough for any real run; a stuck child is killed, never waited on forever.
TRAIN_TIMEOUT_S = 3600
WORKER = "--train-worker"
REVERSE = pl.ACTIONS.index(pl.REVERSE)


# ---------------------------------------------------------------------------
# The cache key. Pure; the tests drive these.
# ---------------------------------------------------------------------------

def source_files(module, roots=(os.path.join(REPO, "crawler"), TOOLS)):
    """Every source file under `roots` that `module` can reach: the modules in
    its globals, and the modules the functions and classes in them came from,
    followed all the way down. A module a later change imports is picked up
    without this list changing."""
    roots = tuple(os.path.abspath(r) + os.sep for r in roots)
    seen, todo, files = set(), [module], set()
    while todo:
        m = todo.pop()
        if id(m) in seen:
            continue
        seen.add(id(m))
        f = getattr(m, "__file__", None)
        if not f or not os.path.abspath(f).startswith(roots):
            continue
        files.add(os.path.abspath(f))
        for v in list(vars(m).values()):
            if isinstance(v, types.ModuleType):
                todo.append(v)
            else:
                owner = sys.modules.get(getattr(v, "__module__", None) or "")
                if owner is not None:
                    todo.append(owner)
    return sorted(files)


def code_hash(files, root=REPO):
    """sha256 over each file's path (relative to root) and bytes."""
    h = hashlib.sha256()
    for f in sorted(files):
        h.update(os.path.relpath(f, root).encode() + b"\0")
        with open(f, "rb") as fh:
            h.update(fh.read())
        h.update(b"\0")
    return h.hexdigest()


def cache_key(path, code, drive):
    """16 hex: the decode code, the file's size and mtime, and how it is read."""
    st = os.stat(path)
    text = f"{code}|{os.path.basename(path)}|{st.st_size}|{st.st_mtime_ns}|{bool(drive)}"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def cache_paths(cache_dir, name, key):
    stem = os.path.join(cache_dir, f"{name}.{key}")
    return stem + ".pkl", stem + ".json"


def decode_code_hash():
    """This file too: the numbers it writes beside each decode are cached."""
    return code_hash(source_files(pd) + [os.path.abspath(__file__)])


# ---------------------------------------------------------------------------
# Which recordings, and which of them are junk.
# ---------------------------------------------------------------------------

class Candidate:
    """One recording, its cache entry under the current code, and its verdict."""

    def __init__(self, path, code, cache_dir):
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.drive = not self.name.startswith(pd.RECORD_PREFIX)
        self.size = os.path.getsize(path)
        self.key = cache_key(path, code, self.drive)
        self.pkl, self.json = cache_paths(cache_dir, self.name, self.key)
        self.stats = None
        self.why = []

    @property
    def adapter_arg(self):
        """What policy_dataset.main reads this file with: a drive log is read
        as footprint whatever it announces first."""
        return pa.FootprintAdapter.name if self.drive else None


def candidates(mcap_dir=MCAP_DIR):
    """Every crawler-*.mcap directly in mcap_dir, oldest name first. Never a
    .orig (the file before a repair) and nothing in a subfolder: drives/ holds
    copies of drive logs that are here too."""
    found = glob.glob(os.path.join(os.path.expanduser(mcap_dir), "crawler-*.mcap"))
    return sorted(p for p in found
                  if p.endswith(".mcap") and not p.endswith(".orig") and os.path.isfile(p))


def junk_reasons(size, stats, min_usable=MIN_USABLE):
    """Why a session does not train, [] when it does. stats is what
    session_stats wrote down, None when the file was too small to decode."""
    why = []
    if size < STUB_BYTES:
        why.append(f"stub: {size} bytes")
    if stats is None:
        return why
    if not stats["footprint"]:
        why.append(f"no /footprint rows (read as {stats['adapter']}, {stats['commands']} commands)")
    if stats["caps"].startswith(pd.CAPS_EXCLUDED):
        why.append("old caps: " + stats["caps"].split(": ", 1)[-1])
    if stats["usable"] < min_usable:
        why.append(f"no labels: {stats['usable']} usable ticks (< {min_usable})")
    return why


def session_stats(session, truncated):
    """The numbers junk_reasons reads, off policy_dataset's own functions."""
    footprint = session.adapter == pa.FootprintAdapter.name and bool(session.commands)
    caps, _ = pd.caps_verdict(session)
    try:
        ticks, _, _ = pd.align(session, None, pd.IdleRules(), None)
        flags = pd.mark_flags(ticks, session.marks)
        usable = int(sum(1 for t, f in zip(ticks, flags)
                         if t.decision and (t.steer_label or t.throttle_label)
                         and f != pd.MARK_BAD))
        error = ""
    except Exception as e:  # a file this broken is junk, said so, never a crash
        ticks, usable, error = [], 0, f"{type(e).__name__}: {e}"
    return {"name": session.name, "adapter": session.adapter, "footprint": footprint,
            "commands": len(session.commands), "frames": len(session.frames),
            "caps": caps, "ticks": len(ticks), "usable": usable,
            "truncated": truncated or "", "error": error}


def _write_atomic(path, write):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


def driven_models(read):
    """Run read() and note, for each policy_model_sha256 a /footprint row
    carries, the log time of its last row: which model drove, and when.
    -> (read()'s result, {sha: last t_ns})."""
    seen = {}
    real = pa.FootprintAdapter.feed

    def feed(self, topic, t_ns, payload):
        if topic == "/footprint" and isinstance(payload, dict):
            sha = payload.get("policy_model_sha256")
            if isinstance(sha, str) and len(sha) == 64:
                seen[sha] = max(seen.get(sha, 0), int(t_ns))
        return real(self, topic, t_ns, payload)

    pa.FootprintAdapter.feed = feed
    try:
        out = read()
    finally:
        pa.FootprintAdapter.feed = real
    return out, seen


def decode_into_cache(path, adapter_arg, pkl, js):
    """Decode one MCAP and write its cache entry, the .json last: a .json
    there means the .pkl beside it is whole. -> stats."""
    t = time.perf_counter()
    (session, truncated), models = driven_models(lambda: pd.read_session(path, adapter_arg))
    stats = session_stats(session, truncated)
    stats["policy_models"] = models
    stats["decode_s"] = round(time.perf_counter() - t, 3)
    _write_atomic(pkl, lambda f: pickle.dump({"session": session, "truncated": truncated},
                                             f, protocol=pickle.HIGHEST_PROTOCOL))
    _write_atomic(js, lambda f: f.write(json.dumps(stats).encode()))
    return stats


def _prune(cache_dir, cand):
    """Drop that session's entries under any other key."""
    for p in glob.glob(os.path.join(cache_dir, glob.escape(cand.name) + ".*")):
        if os.path.basename(p).split(".")[1] != cand.key and p.endswith((".pkl", ".json")):
            os.remove(p)


def ensure_cache(cands, cache_dir, jobs=JOBS, decode=decode_into_cache):
    """Fill in every candidate's stats, decoding only the ones with no cache
    entry under their key. -> the candidates that were decoded."""
    todo = []
    for c in cands:
        if c.size < STUB_BYTES:
            continue
        if os.path.exists(c.json) and os.path.exists(c.pkl):
            with open(c.json) as f:
                c.stats = json.load(f)
        else:
            todo.append(c)
    if todo and (jobs <= 1 or len(todo) == 1):
        for c in todo:
            c.stats = decode(c.path, c.adapter_arg, c.pkl, c.json)
    elif todo:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(min(jobs, len(todo)), mp_context=mp.get_context("spawn")) as ex:
            futures = [(c, ex.submit(decode, c.path, c.adapter_arg, c.pkl, c.json))
                       for c in sorted(todo, key=lambda c: -c.size)]
            for c, fut in futures:
                c.stats = fut.result()
    for c in todo:
        _prune(cache_dir, c)
    return todo


def apply_filter(cands, excludes=()):
    """The junk rules, and the exclude file's (policy_dataset.read_excludes):
    a session I said never to learn from is out whatever is in it."""
    for c in cands:
        c.why = junk_reasons(c.size, c.stats)
        why = pd.excluded(c.path, excludes)
        if why is not None:
            c.why.append(f"excluded: {why}")
    return [c for c in cands if not c.why]


# ---------------------------------------------------------------------------
# The dataset: policy_dataset.main, reading the cache.
# ---------------------------------------------------------------------------

def dataset_key(code, kept, excludes=()):
    """The cache key of a dataset: the decode code, the sessions, and every
    exclude window on one of them (policy_dataset drops those ticks)."""
    windows = sorted(f"{prefix} {why}" for prefix, why in excludes
                     if pd.WINDOW_MARK in prefix
                     and any(c.name.startswith(prefix.partition(pd.WINDOW_MARK)[0])
                             for c in kept))
    text = "|".join([code, str(MIN_USABLE), *sorted(f"{c.name}.{c.key}" for c in kept),
                     *windows])
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def build_dataset(kept, out, log_path, scratch, exclude_file=pd.EXCLUDE_FILE):
    """policy_dataset.main over `kept`, with each Session out of the cache,
    reading the same exclude file the filter did (its windows)."""
    by_path = {os.path.realpath(c.path): c for c in kept}
    real = pd.read_session

    def cached(path, adapter_name=None):
        c = by_path.get(os.path.realpath(path))
        if c is None or c.adapter_arg != adapter_name:
            return real(path, adapter_name)
        with open(c.pkl, "rb") as f:
            d = pickle.load(f)
        return d["session"], d["truncated"]

    records = [c.path for c in kept if not c.drive]
    if not records:
        raise SystemExit("no record session survived the junk rules: nothing to grade on")
    with tempfile.TemporaryDirectory(dir=scratch) as links:
        drives = [c for c in kept if c.drive]
        for c in drives:
            os.symlink(c.path, os.path.join(links, os.path.basename(c.path)))
        argv = ([out, *records, "--uncompressed", "--exclude-file", exclude_file]
                + (["--drive-logs", links] if drives else []))
        pd.read_session = cached
        try:
            with open(log_path, "w") as log, contextlib.redirect_stdout(log):
                rc = pd.main(argv)
        finally:
            pd.read_session = real
    if rc != 0:
        raise SystemExit(f"policy_dataset.py failed ({rc}); see {log_path}")


def _encode_chunk(frames, representation):
    return pt.encode_frames(frames, representation)


def encode_frames_parallel(frames, representation, jobs=JOBS):
    """pt.encode_frames over every frame, in --jobs pieces."""
    if jobs <= 1 or len(frames) < 4096:
        return pt.encode_frames(frames, representation)
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    pieces = np.array_split(np.arange(len(frames)), jobs)
    with ProcessPoolExecutor(jobs, mp_context=mp.get_context("spawn")) as ex:
        parts = list(ex.map(_encode_chunk, [frames[p[0]:p[-1] + 1] for p in pieces if len(p)],
                            [representation] * jobs))
    return np.concatenate(parts)


def train_worker(argv):
    """In a child: policy_train.main, with the frames already encoded."""
    xpath, representation, rest = argv[0], argv[1], argv[2:]
    if rest and rest[0] == "--":
        rest = rest[1:]
    x = np.load(xpath)
    real = pt.encode_frames

    def encode(frames, rep):
        if rep == representation and len(frames) == len(x):
            return x
        return real(frames, rep)

    pt.encode_frames = encode
    import torch
    # Said here as well as in OMP_NUM_THREADS: torch left to itself takes
    # every core, and twelve of those at once run several times slower.
    torch.set_num_threads(TRAIN_THREADS)
    torch.set_num_interop_threads(1)
    return pt.main(rest)


# ---------------------------------------------------------------------------
# The car's model, and the table.
# ---------------------------------------------------------------------------

def last_driven(registry, stats):
    """(entry, its last row as models.stamp text) for the registered model
    with the newest /footprint row across these sessions' stats, or None."""
    best = None
    for s in stats:
        for sha, t_ns in ((s or {}).get("policy_models") or {}).items():
            if best is None or t_ns > best[0]:
                e = md.find_by_sha(registry, sha)
                if e is not None:
                    best = (t_ns, e)
    if best is None:
        return None
    when = datetime.datetime.fromtimestamp(best[0] / 1e9, datetime.timezone.utc)
    return best[1], when.strftime("%Y-%m-%dT%H:%M:%SZ")


def deployed_entry(registry, stats=()):
    """(entry or None, how it was found), with no ssh. What the car runs is
    what models.py list stars, and that asks the car (active_entry: its
    symlink, by sha256 over ssh). Here, in this order:

      1. the model in the newest /footprint row of the pulled drives (each
         row carries policy_model_sha256), unless a deploy marker is newer
         than that row: then the car was re-pointed after it last drove.
      2. the registry's "deployed", written by models.py deploy.
      3. the entry with the newest note that starts "Deployed", which is
         how a deploy was written down before the marker.
    """
    mark = registry.get("deployed")
    marked = md.find_by_sha(registry, mark.get("onnx_sha256"), mark.get("onnx")) if mark else None
    driven = last_driven(registry, stats)
    if driven and not (marked and str(mark.get("date", "")) > driven[1]):
        return driven[0], f"the model in the newest drive log row, {driven[1]}"
    if marked is not None:
        return marked, f"the registry's deploy marker, {mark.get('date')}"
    best = None
    for e in registry["releases"] + registry["candidates"]:
        for n in e.get("notes_log") or []:
            if str(n.get("text", "")).lstrip().lower().startswith("deployed"):
                if best is None or str(n.get("date", "")) > best[0]:
                    best = (str(n.get("date", "")), e)
    if best:
        return best[1], f"the newest \"Deployed\" note, {best[0]} (no deploy marker yet)"
    return None, "nothing in the registry says what the car runs"


def trained_on(entry):
    """The session names a registry entry trained on, empty for None."""
    return {s["name"] if isinstance(s, dict) else s
            for s in (entry or {}).get("train_sessions") or []}


def unseen_sessions(names, session, graded, seen):
    """The sessions with graded ticks that a model trained on `seen` never
    saw, in dataset order. The head-to-head holds these out whole: a fold
    model graded on them would be missing the rest of its block as well,
    which the deployed model had, and that alone cost it 0.13 of balanced
    accuracy on dagger2-1029's one such session (0.427 against 0.561)."""
    with_ticks = set(np.unique(np.asarray(session)[np.asarray(graded, dtype=bool)]).tolist())
    return [n for i, n in enumerate(names) if i in with_ticks and n not in seen]


def pooled(scores):
    """Pool score_ticks results: the confusions summed, the steering MAE
    weighted by its ticks. None when there is nothing."""
    scores = [s for s in scores if s and s.get("ticks")]
    if not scores:
        return None
    m = np.sum([np.asarray(s["confusion"]) for s in scores], axis=0)
    n_steer = sum(s.get("steer_ticks", 0) for s in scores)
    steer = (sum(s["steer_mae"] * s["steer_ticks"] for s in scores if s.get("steer_ticks"))
             / n_steer) if n_steer else float("nan")
    return {"ticks": int(sum(s["ticks"] for s in scores)), "confusion": m.tolist(),
            "steer_mae": float(steer)}


def headline(s):
    """(ticks, direction accuracy, balanced, reverse recall, reverse precision,
    steering MAE) off a score_ticks result or a pooled one."""
    if not s:
        return None
    m = np.asarray(s["confusion"], dtype=float)
    rows, cols = m.sum(1), m.sum(0)
    n = m.sum()
    bal = [m[k, k] / rows[k] for k in range(len(rows)) if rows[k]]
    nan = float("nan")
    return (int(s["ticks"]), m.trace() / n if n else nan, float(np.mean(bal)) if bal else nan,
            m[REVERSE, REVERSE] / rows[REVERSE] if rows[REVERSE] else nan,
            m[REVERSE, REVERSE] / cols[REVERSE] if cols[REVERSE] else nan,
            float(s.get("steer_mae", nan)))


def _cells(h):
    if h is None:
        return "      -     -     -     -     -"
    return " ".join(f"{v:5.3f}" if np.isfinite(v) else "  nan" for v in h[1:])


def comparison_table(rows, deployed_label):
    """rows: [(name, ticks, seen share or None, candidate score, deployed score)]."""
    head = "dir   bal   rev_r rev_p steer"
    out = [f"{'':24} {'':>6} {'':>5}  {'candidate':^29} | {deployed_label or 'deployed':^29}",
           f"{'':24} {'ticks':>6} {'seen':>5}  {head} | {head}"]
    for name, ticks, seen, cand, dep in rows:
        seen_txt = f"{seen:5.0%}" if seen is not None else "    -"
        out.append(f"{name:24} {ticks:6d} {seen_txt}  {_cells(headline(cand))} | "
                   f"{_cells(headline(dep))}")
    return "\n".join(out)


def note_text(n_sessions, n_drives, args, extra, fold_scores, mean, dep_label, unseen):
    """The registry note: what trained, and the held-out table in one line each."""
    parts = [f"retrain.py: {n_sessions} sessions ({n_drives} drive logs), "
             f"{args.blocks} blocks, {args.epochs} epochs, seed {args.seed}"
             + (f", flags {shlex.join(extra)}" if extra else "")
             + ". Held out (ticks dir bal rev_r rev_p steer):"]
    for k, s in enumerate(fold_scores):
        parts.append(f"b{k + 1} {s['ticks']} {_cells(headline(s))}")
    parts.append(f"mean {_cells(headline(mean))}")
    if dep_label and unseen:
        parts.append(f"vs {dep_label} on ticks it never trained on ({unseen[0]['ticks']}): "
                     f"cand {_cells(headline(unseen[0]))}, dep {_cells(headline(unseen[1]))}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# The run.
# ---------------------------------------------------------------------------

class Clock:
    def __init__(self):
        self.stages, self.t0 = [], time.perf_counter()
        self._t = self.t0

    def lap(self, name):
        now = time.perf_counter()
        self.stages.append((name, now - self._t))
        self._t = now

    def text(self):
        lines = [f"  {name:10} {s:7.1f} s" for name, s in self.stages]
        lines.append(f"  {'total':10} {time.perf_counter() - self.t0:7.1f} s")
        return "\n".join(lines)


def parse_args(argv):
    argv = list(argv)
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-pull", action="store_true", help="skip scripts/pull-recordings.sh")
    ap.add_argument("--mcap-dir", default=MCAP_DIR)
    ap.add_argument("--models-dir", default=MODELS_DIR)
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--runs-dir", default=RUNS_DIR)
    ap.add_argument("--exclude-file", default=pd.EXCLUDE_FILE,
                    help="sessions never to learn from, one name prefix a line")
    ap.add_argument("--blocks", type=int, default=BLOCKS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=JOBS,
                    help=f"processes for the decodes and the training (default {JOBS})")
    ap.add_argument("--tag", default=None, help="candidate tag (default retrain-<stamp>)")
    ap.add_argument("--baseline", default=None, metavar="TARGET",
                    help="grade against this (1.2, candidate-<tag>) instead of the deployed model")
    ap.add_argument("--dry-run", action="store_true", help="everything but the register")
    args = ap.parse_args(argv)
    clash = [f for f in extra if f.split("=")[0] in OWN_FLAGS]
    if clash:
        ap.error(f"{clash} after --: retrain.py sets those itself (--blocks, --epochs here)")
    return args, extra


def recipe_flags(args, extra):
    """policy_train.py flags every training process gets: the defaults here
    first, so the same flag after -- comes later and wins."""
    return ["--epochs", str(args.epochs), "--seed", str(args.seed), "--model-name", md.FAMILY,
            "--takeover-weight", f"{TAKEOVER_WEIGHT:g}", *extra]


def _train_flags(extra):
    """The two policy_train.py settings this script needs to know itself."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--representation", default=pl.PROFILE)
    p.add_argument("--labels", default="pad")
    known, _ = p.parse_known_args(extra)
    return known.representation, known.labels


def thread_env():
    """One thread count for every BLAS and OpenMP pool, said outright: the
    decodes and the frame encodes run --jobs at once as well."""
    n = str(TRAIN_THREADS)
    return {"OMP_NUM_THREADS": n, "MKL_NUM_THREADS": n, "VECLIB_MAXIMUM_THREADS": n,
            "OPENBLAS_NUM_THREADS": n}


def run_children(commands, logs, jobs, while_waiting=None, timeout_s=TRAIN_TIMEOUT_S):
    """Run every command, at most `jobs` at once; while_waiting runs once
    they are going. -> [returncode]."""
    env = {**os.environ, **thread_env(), "PYTHONWARNINGS": "ignore"}
    pending = list(range(len(commands)))
    running, codes, files = {}, [None] * len(commands), []
    deadline = time.monotonic() + timeout_s
    waited = while_waiting is None
    try:
        while pending or running:
            while pending and len(running) < max(1, jobs):
                i = pending.pop(0)
                f = open(logs[i], "w")
                files.append(f)
                running[i] = subprocess.Popen(commands[i], stdout=f, stderr=subprocess.STDOUT,
                                              env=env, cwd=REPO)
            if not waited:
                waited = True
                while_waiting()
            for i, p in list(running.items()):
                if p.poll() is not None:
                    codes[i] = p.returncode
                    del running[i]
            if time.monotonic() > deadline:
                raise SystemExit(f"training took over {timeout_s} s; stopped it")
            time.sleep(0.2)
    finally:
        for p in running.values():
            p.kill()
        for f in files:
            f.close()
    return codes


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == WORKER:
        return train_worker(argv[1:])
    args, extra = parse_args(argv)
    # Before any pool starts: the spawned decode and encode processes read it.
    os.environ.update(thread_env())
    clock = Clock()
    for d in (args.cache_dir, args.runs_dir):
        why = pm.refuse_reason(d)
        if why:
            raise SystemExit(why)
        os.makedirs(d, exist_ok=True)

    if not args.no_pull:
        rc = subprocess.call([os.path.join(REPO, "scripts", "pull-recordings.sh")])
        if rc != 0:
            print(f"pull-recordings.sh failed ({rc}); going on with what is on this Mac")
    clock.lap("pull")

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    run_dir = os.path.join(args.runs_dir, stamp)
    os.makedirs(run_dir)
    code = decode_code_hash()
    cands = [Candidate(p, code, args.cache_dir) for p in candidates(args.mcap_dir)]
    decoded = ensure_cache(cands, args.cache_dir, args.jobs)
    stubs = sum(c.size < STUB_BYTES for c in cands)
    print(f"{len(cands)} recordings, {len(decoded)} decoded now, "
          f"{len(cands) - len(decoded) - stubs} from the cache ({args.cache_dir}), "
          f"{stubs} too small to open")
    clock.lap("scan")

    excludes = pd.read_excludes(args.exclude_file)
    kept = apply_filter(cands, excludes)
    for c in cands:
        if c.why:
            print(f"  junk {c.name}: {'; '.join(c.why)}")
    drives = sum(c.drive for c in kept)
    print(f"{len(kept)} sessions train ({len(kept) - drives} record, {drives} drive logs), "
          f"{len(cands) - len(kept)} junk")
    clock.lap("filter")

    representation, labels = _train_flags(extra)
    dkey = dataset_key(code, kept, excludes)
    data_path = os.path.join(args.cache_dir, f"dataset-{dkey}.npz")
    x_path = os.path.join(args.cache_dir, f"dataset-{dkey}.x-{representation}.npy")
    if os.path.exists(data_path):
        print(f"dataset: the same sessions as last time, {data_path}")
    else:
        tmp = os.path.join(args.cache_dir, f"dataset-{dkey}.part.npz")
        build_dataset(kept, tmp, os.path.join(run_dir, "dataset.log"), args.cache_dir,
                      args.exclude_file)
        os.replace(tmp, data_path)
        for old in glob.glob(os.path.join(args.cache_dir, "dataset-*")):
            if not os.path.basename(old).startswith(f"dataset-{dkey}."):
                os.remove(old)
        print(f"dataset: built {data_path} (report in {run_dir}/dataset.log)")
    data = pe.load(data_path)
    if not os.path.exists(x_path):
        x = encode_frames_parallel(data["frames"], representation, args.jobs)
        np.save(x_path + ".part.npy", x)
        os.replace(x_path + ".part.npy", x_path)
    n_ticks = len(data["t_ns"])
    print(f"  {n_ticks} ticks, {int(data['decision'].sum())} decisions, "
          f"{len(data['frames'])} frames")
    clock.lap("dataset")

    registry = md.load_registry(args.models_dir)
    if args.baseline:
        dep, how = md.find_entry(registry, args.baseline), "--baseline"
    else:
        # Every decoded file, junk too: a short drive still says what drove.
        dep, how = deployed_entry(registry, [c.stats for c in cands])
    blocks = pt.time_blocks(data, args.blocks)
    names = data["meta"]["sessions"]
    dep_seen = trained_on(dep)
    any_graded = np.any([held for held, _ in blocks], axis=0) & ~pe.bad_marked(data)
    held_names = unseen_sessions(names, data["session"], any_graded, dep_seen) if dep else []

    base = [data_path]
    recipe = recipe_flags(args, extra)
    pieces = [str(k + 1) for k in range(args.blocks)] + ["final"]
    outs = [os.path.join(run_dir, f"fold-{p}.onnx") for p in pieces[:-1]]
    outs.append(os.path.join(run_dir, f"policy-{stamp}.onnx"))
    child = [sys.executable, os.path.abspath(__file__), WORKER, x_path, representation, "--"]
    commands = [[*child, *base, out, "--blocks", str(args.blocks), *recipe, "--fold", p]
                for p, out in zip(pieces, outs)]
    if held_names:
        # The head-to-head: trained on every session but the ones the deployed
        # model never saw, as it was, and graded with it on those.
        pieces.append("vs-deployed")
        outs.append(os.path.join(run_dir, "vs-deployed.onnx"))
        commands.append([*child, *base, outs[-1], *recipe,
                         *[a for n in held_names for a in ("--val", n)]])
    logs = [os.path.splitext(o)[0] + ".log" for o in outs]
    final = outs[args.blocks]
    steer_ok, throttle_ok = pe.label_masks(data, labels)
    labelled = steer_ok | throttle_ok
    graded = [held & ~pe.bad_marked(data) for held, _ in blocks]
    medians = [pe.median_training_magnitude(data, pt.training_mask(data, guard, labelled))
               for _, guard in blocks]
    dep_scores, dep_model = [None] * args.blocks, None
    grade_s = [0.0]

    def grade_deployed():
        nonlocal dep_model
        if dep is None:
            return
        t = time.perf_counter()
        try:
            dep_model = pl.PolicyModel(os.path.join(args.models_dir, dep["onnx"]))
            for k in range(args.blocks):
                dep_scores[k] = pe.score_ticks(dep_model, data, np.flatnonzero(graded[k]),
                                               labels, medians[k])
        except Exception as e:  # a contract this dataset cannot feed
            print(f"cannot grade {dep['label']} on this dataset: {type(e).__name__}: {e}")
            dep_model = None
        grade_s[0] = time.perf_counter() - t

    print(f"training {args.blocks} folds and the final model"
          + (" and the head-to-head" if held_names else "")
          + f" at once, {args.epochs} epochs, takeover weight {TAKEOVER_WEIGHT:g}"
          + (f", {shlex.join(extra)}" if extra else ""))
    codes = run_children(commands, logs, args.jobs, grade_deployed)
    failed = [(p, log) for p, rc, log in zip(pieces, codes, logs) if rc != 0]
    if failed:
        for p, log in failed:
            print(f"--fold {p} failed; the end of {log}:")
            with open(log) as f:
                print("".join(f.readlines()[-15:]))
        return 1
    clock.lap("train")

    fold_scores = []
    for out in outs[:args.blocks]:
        with open(os.path.splitext(out)[0] + ".json") as f:
            fold_scores.append(json.load(f)["fold"]["scores"])
    mean = pt._mean_scores(fold_scores)
    sess = np.asarray(data["session"])
    held_ids = [i for i, n in enumerate(names) if n in held_names]
    unseen_mask = ~np.isin(sess, [i for i, n in enumerate(names) if n in dep_seen])
    rows = []
    for k in range(args.blocks):
        ticks = np.flatnonzero(graded[k])
        seen = 1.0 - float(unseen_mask[ticks].mean()) if dep_model and len(ticks) else None
        rows.append((f"block {k + 1}", len(ticks), seen, fold_scores[k], dep_scores[k]))
    total = sum(r[1] for r in rows)
    rows.append(("pooled", total, None, pooled(fold_scores), pooled(dep_scores)))
    unseen = None
    if dep_model is not None and held_names:
        sub = np.flatnonzero(any_graded & np.isin(sess, held_ids))
        med = pe.median_training_magnitude(
            data, pt.training_mask(data, np.isin(sess, held_ids), labelled))
        unseen = (pe.score_ticks(pl.PolicyModel(outs[-1]), data, sub, labels, med),
                  pe.score_ticks(dep_model, data, sub, labels, med))
        rows.append(("never seen by deployed", unseen[0]["ticks"], 0.0, *unseen))
    clock.lap("grade")

    with open(os.path.splitext(final)[0] + ".json") as f:
        meta = json.load(f)
    meta["cv"] = {"blocks": args.blocks, "guard_s": pt.BLOCK_GUARD_S, "mean": mean,
                  "folds": fold_scores}
    meta["retrain"] = {"dataset": data_path, "sessions": [c.name for c in kept],
                       "junk": {c.name: c.why for c in cands if c.why},
                       "baseline": dep["label"] if dep else None}
    pt.write_sidecar(final, meta)
    dep_label = dep["label"] if dep else None
    note = note_text(len(kept), drives, args, extra, fold_scores, mean, dep_label, unseen)
    tag = args.tag or f"retrain-{stamp}"
    entry = None
    same = md.find_by_sha(registry, pm.sha256_local(final))
    if args.dry_run:
        print(f"--dry-run: not registered; the model is {final}")
    elif same:
        # Same sessions, code and seed train the same bytes: nothing new to add.
        print(f"the final model is byte for byte {same['label']} ({same['onnx']}); "
              "not registered again")
    else:
        entry = md.register(final, args.models_dir, args.mcap_dir, tag=tag, notes=note)
    clock.lap("register")

    print()
    print(f"held out, {args.blocks} blocks; deployed: "
          f"{dep_label + ' (' + how + ')' if dep else how}"
          + (f"; graded in {grade_s[0]:.1f} s while the folds trained" if dep else ""))
    print(comparison_table(rows, dep_label))
    print("  dir: direction accuracy, bal: balanced over forward/reverse/stop, rev_r/rev_p:"
          " reverse recall/precision, steer: MAE as a fraction of full lock;"
          " seen: share of the block the deployed model trained on"
          + (f"; never seen: {', '.join(held_names)}, the candidate trained on every"
             " other session" if unseen else ""))
    if entry:
        print(f"\nregistered {entry['label']} ({entry['onnx']}); not deployed. On the car: "
              f"./scripts/models.sh deploy candidate-{tag}")
    print(f"run folder: {run_dir}")
    print("\ntime per stage")
    print(clock.text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
