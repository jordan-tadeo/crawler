#!/usr/bin/env python3
"""Named driving models: candidates, the releases that earned a number, and the car's pick.

    python3 tools/models.py list                    # releases, candidates, which one drives
    python3 tools/models.py register ~/crawler-data/models/policy-<stamp>.onnx --tag imu-gyro
    python3 tools/models.py deploy candidate-imu-gyro --dry-run   # say what it would do
    python3 tools/models.py deploy --candidate imu-gyro           # a test drive on the car
    python3 tools/models.py promote imu-gyro --evidence ab.json --approved-on-car \\
        --notes "held the line through the turn"
    python3 tools/models.py deploy 1.3              # put kblock 1.3 on the car
    python3 tools/models.py rollback                # back to what drove before
    python3 tools/models.py deploy candidate-sweep2-1029 --fallback   # the map model's fallback
    python3 tools/models.py status                  # what the car points at
    python3 tools/models.py approve 1.2 --notes ".."  # record that a release drove well
    python3 tools/models.py note 1.2 "loops on a 3-point turn"   # a dated drive note
    python3 tools/models.py notes 1.2               # every note on that model
    python3 tools/models.py attach 1.2 --evidence ab.json --gate-note "won held-out A/B"
    python3 tools/models.py demote 2.0 --tag imu-1028 --reason "..."

scripts/models.sh runs this. Standard library only, so plain python3 on the Mac runs
it; nothing in here imports pigpio or commands a wheel.

Names. A version number means "better than the version before it", the way a
product's does, so a model only gets one by proving it. A newly trained model
registers as a CANDIDATE with no number: "kblock candidate <tag>", files
candidate-<tag>.onnx and .json, the tag from --tag or else the policy-<stamp>
stem. A candidate can go on the car for a test drive like any release.

promote <tag> turns a candidate into the next RELEASE, "kblock X.Y", and
refuses unless both gates pass:

  1. The held-out test. --evidence is a results JSON comparing the
     candidate's training recipe with the latest release's on identical
     held-out folds (leave-one-session-out). Its pooled numbers must have
     reverse recall not worse, reverse precision not worse by more than
     GATE_REVERSE_PRECISION_SLACK and steering MAE not worse by more than
     GATE_STEER_MAE_SLACK. --force-gate "<reason>" skips the rule and the
     reason is kept in the registry. The evidence file is copied in beside
     the release as kblock-X.Y.evidence.json.
  2. My approval on the car: --approved-on-car, with an optional --notes on
     how it drove. promote records the date.

The number: MINOR up when the candidate's contract (CONTRACT_KEYS) is the same
as the latest release's, MAJOR up when it differs -- the input contract,
representation, an input added (IMU, the previous action), the outputs, the
steering head, the steering scale. Only releases count, so a number given back by demote is
given out again to the next model that earns it. policy_train.py does not
write a number into a sidecar, and register ignores one if it is there.

The evidence schema (EVIDENCE_SCHEMA = 1), one JSON object:

    evidence_schema  1
    candidate        {"label": "kblock candidate <tag>", "recipe": "..."}
    baseline         {"label": "kblock X.Y" -- the latest release, "recipe": "..."}
    method           how the folds were made and graded, in words
    sessions         every session the folds were cut from
    folds            [{"name", "held_out": [session], "candidate": M, "baseline": M}]
    pooled           {"candidate": M, "baseline": M} over every fold
    M                {"reverse_recall", "reverse_precision", "steer_mae",
                      "direction_balanced_accuracy", ...counts as wanted}

promote never deletes: the candidate's files and entry stay, linked to the
release (promoted_to / candidate). demote <X.Y> --tag <tag> is the way back
for a release that should not have had its number: it copies kblock-X.Y.*
to candidate-<tag>.*, keeps kblock-X.Y.* on disk, and drops the release
entry. It refuses the model the car is running.

The registry is ~/crawler-data/models/registry.json, beside the model files,
never inside a git tree (the repo is public; the guard is
pull_mcaps.refuse_reason). Registering copies: the policy-<stamp> files the
trainer wrote stay exactly where and what they were, the copy is hashed against
the original before it is renamed into place, and the copy's sidecar gets
model_name / model_version ("1.3", or "candidate <tag>") so the car can say
which model drove (policy.model_label, the /footprint policy_model field).
Nothing is overwritten.

Deploy. The service runs `python -m crawler --policy models/policy.onnx`, and
models/policy.onnx and models/policy.json are symlinks on the car. deploy
copies the model's files into the car's models/ if they are not there, checks
their sha256 on the car, and then -- in one ssh, behind a fresh parked check --
re-points each symlink atomically (ln -sfn to a temp name, mv -T over the
real one), remembers the old target in models/.policy-previous for rollback,
and restarts crawler.service. It refuses unless `pigs gpw 18` reads 1575, the
ESC's neutral pulse: a car with the throttle off neutral is a car someone is
driving, and a restart under it is a restart at speed. After the restart it
checks the service is active, has not crash-looped (NRestarts), is listening
on 8765, is still at neutral, and has written no Traceback since the restart.
--dry-run does only the read-only half and prints the rest. A candidate goes
through exactly the same steps. Once the links point at it, the registry's
"deployed" says which model that was and when.

--fallback (deploy, rollback) does the same to the second slot,
models/fallback.onnx and fallback.json with models/.fallback-previous: the
map-free model a map --policy drives on while its map is lost, started with
--fallback-policy models/fallback.onnx. A map model is refused there, and the
registry's "deployed_fallback" says which model it was.

The car is CRAWLER_HOST, else jt@jtpi.local, the same as pull_mcaps.py: by name,
because the DHCP lease moves (tests/test_car_address.py).
"""

import argparse
import datetime
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time

import pull_mcaps as pm

FAMILY = "kblock"
MODELS_DIR = os.path.expanduser(os.environ.get("CRAWLER_MODELS", "~/crawler-data/models"))
MCAP_DIR = os.path.expanduser(os.environ.get("CRAWLER_DATA", "~/crawler-data/mcap"))
REGISTRY = "registry.json"
# 1: one "versions" list. 2: "releases" and "candidates"; a schema 1 registry
# is read as releases that predate the gate (upgrade_registry).
SCHEMA = 2

# The held-out gate, on the evidence's pooled numbers, candidate against the
# latest release. Reverse recall may not drop at all: reversing out of a
# corner is the thing these models are still bad at.
GATE_REVERSE_PRECISION_SLACK = 0.05
GATE_STEER_MAE_SLACK = 0.02
EVIDENCE_SCHEMA = 1
EVIDENCE_METRICS = ("reverse_recall", "reverse_precision", "steer_mae",
                    "direction_balanced_accuracy")

CAR_REPO = "/home/jt/Documents/py/crawler"
CAR_MODELS = CAR_REPO + "/models"
# What the service is started with: python -m crawler --policy models/policy.onnx, and
# PolicyModel reads the .json beside whatever path it is given.
LINK_ONNX = "policy.onnx"
LINK_JSON = "policy.json"
PREVIOUS = ".policy-previous"
# The second slot: the map-free model a map model hands the wheels to while
# the map is lost (--fallback-policy models/fallback.onnx). Same links, same
# swap, same rollback, its own names; deploy/rollback --fallback.
SLOTS = {
    "policy": ("policy.onnx", "policy.json", ".policy-previous", "deployed"),
    "fallback": ("fallback.onnx", "fallback.json", ".fallback-previous", "deployed_fallback"),
}
SERVICE = pm.SERVICE
PORT = 8765
# The ESC's GPIO and its neutral pulse, read with pigs and never written here.
THROTTLE_GPIO = 18
PARKED_US = 1575

# What a car has to know to run a model. Any of these changing is a MAJOR.
# The defaults are what a sidecar from before the key existed means.
CONTRACT_KEYS = {
    "input_contract": None,
    "representation": None,
    "actions": None,
    "uses_prev_action": False,
    "uses_imu": False,
    "imu_features": None,
    "imu_window": None,
    "imu_tilt": False,
    # False: the no-yaw IMU list (policy_imu.NO_YAW_FEATURES). An IMU model
    # from before the key fed yaw.
    "imu_yaw": True,
    "outputs": None,
    "steer_scale": None,
    "steer_modes": None,
    # How the steering is read (policy.decode_steer): one tanh, or a
    # left/straight/right class and a magnitude. Before the key, a regression.
    "steer_head": "regression",
    # The roof lidar as an input (policy_lidar), and the binning it was
    # trained on. Before the key, no lidar.
    "uses_lidar": False,
    "lidar_layout": None,
    # Action chunks (policy.ChunkEnsembler): how many decisions the gear and
    # steer heads answer for. Before the key, one.
    "chunk": 1,
}

# The scores list shows, off the CV mean or the held-out sessions.
SHORT_SCORES = (
    ("dir", "direction_accuracy"),
    ("dir_bal", "direction_balanced_accuracy"),
    ("steer_mae", "steer_mae"),
    ("mode", "mode_accuracy"),
)

STAMP = re.compile(r"(\d{8})-(\d{6})Z")
VERSION = re.compile(r"^(\d+)\.(\d+)$")
TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$")
CANDIDATE_PREFIX = "candidate-"


# ---------------------------------------------------------------------------
# Pure: versions, contracts, the registry document. The tests drive these.
# ---------------------------------------------------------------------------

def parse_version(text):
    """"1.1" -> (1, 1). ValueError for anything not MAJOR.MINOR."""
    m = VERSION.match(str(text).strip())
    if not m:
        raise ValueError(f"{text!r} is not a MAJOR.MINOR version")
    return int(m.group(1)), int(m.group(2))


def format_version(v):
    return f"{v[0]}.{v[1]}"


def bump(version, part):
    """The next version after `version` ("1.1") for part "major" or "minor"."""
    major, minor = parse_version(version)
    if part == "major":
        return format_version((major + 1, 0))
    if part == "minor":
        return format_version((major, minor + 1))
    raise ValueError(f"bump {part!r}: major or minor")


def check_tag(tag):
    """A candidate's tag, or ValueError: a short slug, safe as a file name."""
    if not isinstance(tag, str) or not TAG.match(tag):
        raise ValueError(f"tag {tag!r}: letters, digits, '.', '_' and '-', "
                         f"at most 48, starting with a letter or digit")
    return tag


def default_tag(source_path):
    """The policy-<stamp> stem the trainer wrote, as the tag."""
    return os.path.splitext(os.path.basename(source_path))[0]


def contract_of(meta):
    """The part of a sidecar the car's runtime depends on."""
    out = {}
    for key, default in CONTRACT_KEYS.items():
        value = meta.get(key, default)
        if isinstance(value, tuple):
            value = list(value)
        out[key] = value
    if not out["uses_imu"]:
        # Without the IMU input the IMU details mean nothing.
        out["imu_features"] = out["imu_window"] = None
        out["imu_tilt"] = out["imu_yaw"] = False
    if not out["uses_lidar"]:
        out["lidar_layout"] = None
    return out


def contract_diff(a, b):
    """The contract keys that differ between two sidecars' contracts. Each
    goes through contract_of first, so a contract stored before a key
    existed reads as that key's default, not as a change."""
    a, b = contract_of(a), contract_of(b)
    return [k for k in CONTRACT_KEYS if a.get(k) != b.get(k)]


def label(name, version):
    return f"{name} {version}"


def candidate_label(name, tag):
    return f"{name} candidate {tag}"


def file_stem(name, version):
    return f"{name}-{version}"


def candidate_stem(tag):
    return CANDIDATE_PREFIX + tag


def empty_registry():
    return {"schema": SCHEMA, "releases": [], "candidates": []}


def upgrade_registry(doc):
    """A schema 1 registry as schema 2: every version it has is a release that
    predates the gate, with no on-car approval recorded."""
    if doc.get("schema") == 1 and isinstance(doc.get("versions"), list):
        releases = [{**e, "gate": e.get("gate", "pre-gate"),
                     "approval": e.get("approval")} for e in doc["versions"]]
        return {"schema": SCHEMA, "releases": releases, "candidates": []}
    if (doc.get("schema") == SCHEMA and isinstance(doc.get("releases"), list)
            and isinstance(doc.get("candidates"), list)):
        return doc
    raise ValueError(f"not a schema 1 or {SCHEMA} registry")


def versions_of(registry, name):
    """That family's releases, oldest version first. Candidates have no
    number and never count."""
    return sorted((e for e in registry["releases"] if e["name"] == name),
                  key=lambda e: parse_version(e["version"]))


def find(registry, name, version):
    for e in registry["releases"]:
        if e["name"] == name and parse_version(e["version"]) == parse_version(version):
            return e
    return None


def find_candidate(registry, tag):
    for e in registry["candidates"]:
        if e["tag"] == tag:
            return e
    return None


def find_by_sha(registry, sha, onnx_name=None):
    """The entry with that onnx sha256. A promoted candidate and its release
    are the same bytes, so the file name decides first, then a release wins."""
    hits = [e for e in registry["releases"] + registry["candidates"]
            if e["onnx_sha256"] == sha]
    for e in hits:
        if onnx_name and e["onnx"] == onnx_name:
            return e
    return hits[0] if hits else None


def next_version(registry, name, contract):
    """(version, why) for the next release of `name`: MINOR on the latest
    release's contract, MAJOR off it, 1.0 for the first."""
    existing = versions_of(registry, name)
    if not existing:
        return "1.0", "first release"
    latest = existing[-1]
    changed = contract_diff(contract, latest["contract"])
    if changed:
        return (bump(latest["version"], "major"),
                f"contract differs from {latest['version']} in {', '.join(changed)}")
    return bump(latest["version"], "minor"), f"same contract as {latest['version']}"


def created_of(meta, source_path, mtime=None):
    """When the model was trained: the sidecar's created, else the stamp in
    policy-<YYYYMMDD-HHMMSS>Z, else the file's mtime, as UTC ISO 8601."""
    if meta.get("created"):
        return str(meta["created"])
    m = STAMP.search(os.path.basename(source_path))
    if m:
        t = datetime.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        return t.strftime("%Y-%m-%dT%H:%M:%SZ")
    when = os.path.getmtime(source_path) if mtime is None else mtime
    return datetime.datetime.fromtimestamp(when, datetime.timezone.utc) \
        .strftime("%Y-%m-%dT%H:%M:%SZ")


def clean_numbers(value):
    """A copy with NaN and infinities as None: registry.json is strict JSON,
    and a sidecar's scores have NaN wherever a fold had nothing to grade."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean_numbers(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_numbers(v) for v in value]
    return value


def scores_of(meta):
    """The CV mean and the held-out sessions' scores. Not the folds or the
    train fit: the sidecar beside the model keeps those."""
    cv = meta.get("cv")
    out = {"cv": None, "val": clean_numbers(meta.get("val"))}
    if isinstance(cv, dict):
        out["cv"] = clean_numbers({k: cv.get(k) for k in ("blocks", "guard_s", "mean")})
    return out


def short_scores(entry):
    """"cv5 dir 0.957 dir_bal 0.461 ..." for list."""
    scores = entry.get("scores") or {}
    cv = scores.get("cv")
    if cv and cv.get("mean"):
        where, numbers = f"cv{cv.get('blocks')}", cv["mean"]
    elif scores.get("val"):
        where, numbers = "val", scores["val"]
    else:
        return "no scores"
    parts = [where]
    for short, key in SHORT_SCORES:
        v = numbers.get(key)
        if isinstance(v, (int, float)):
            parts.append(f"{short} {v:.3f}")
    return " ".join(parts)


def make_candidate(name, tag, meta, source_onnx, source_json, onnx_sha,
                   json_sha, source_json_sha, sessions, parent, notes, registered):
    stem = candidate_stem(tag)
    return {
        "name": name,
        "tag": tag,
        "label": candidate_label(name, tag),
        "onnx": stem + ".onnx",
        "json": stem + ".json",
        "created": created_of(meta, source_onnx),
        "registered": registered,
        "onnx_sha256": onnx_sha,
        "json_sha256": json_sha,
        "source": {"onnx": os.path.basename(source_onnx),
                   "json": os.path.basename(source_json),
                   "json_sha256": source_json_sha},
        "train_sessions": sessions.get("train", []),
        "val_sessions": sessions.get("val", []),
        "train_commit": meta.get("train_commit"),
        "train_commit_dirty": meta.get("train_commit_dirty"),
        "train_command": meta.get("train_command"),
        "contract": contract_of(meta),
        "scores": scores_of(meta),
        "parent": parent,
        "notes": notes or "",
        "promoted_to": None,
    }


def registered_sidecar(meta, name, version):
    """The sidecar a registered copy carries: the trainer's, named. version
    is "1.3" for a release and "candidate <tag>" for a candidate, so the car
    reports "kblock 1.3" or "kblock candidate <tag>"."""
    return {**meta, "model_name": name, "model_version": version}


# ---------------------------------------------------------------------------
# Pure: the held-out gate.
# ---------------------------------------------------------------------------

def _metrics_problems(where, m):
    if not isinstance(m, dict):
        return [f"{where} is not an object"]
    return [f"{where}.{k} is not a number" for k in EVIDENCE_METRICS
            if not isinstance(m.get(k), (int, float)) or isinstance(m.get(k), bool)
            or not math.isfinite(m[k])]


def evidence_problems(doc):
    """What is wrong with an evidence document's shape, [] when nothing is."""
    if not isinstance(doc, dict):
        return ["evidence is not a JSON object"]
    problems = []
    if doc.get("evidence_schema") != EVIDENCE_SCHEMA:
        problems.append(f"evidence_schema is {doc.get('evidence_schema')!r}, "
                        f"not {EVIDENCE_SCHEMA}")
    for side in ("candidate", "baseline"):
        if not isinstance((doc.get(side) or {}).get("label"), str):
            problems.append(f"{side}.label is missing")
    if not isinstance(doc.get("method"), str) or not doc["method"].strip():
        problems.append("method is missing")
    sessions = doc.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        problems.append("sessions is missing or empty")
    folds = doc.get("folds")
    if not isinstance(folds, list) or not folds:
        problems.append("folds is missing or empty")
    else:
        for i, fold in enumerate(folds):
            if not isinstance(fold, dict):
                problems.append(f"folds[{i}] is not an object")
                continue
            held = fold.get("held_out")
            if not isinstance(held, list) or not held:
                problems.append(f"folds[{i}].held_out is missing or empty")
            elif isinstance(sessions, list) and any(s not in sessions for s in held):
                problems.append(f"folds[{i}].held_out names a session not in sessions")
            for side in ("candidate", "baseline"):
                problems += _metrics_problems(f"folds[{i}].{side}", fold.get(side))
    pooled = doc.get("pooled") or {}
    for side in ("candidate", "baseline"):
        problems += _metrics_problems(f"pooled.{side}", pooled.get(side))
    return problems


def gate_check(doc):
    """The pass rule on an evidence document's pooled numbers.
    -> {"passes", "checks": [...], "pooled": {...}}"""
    cand, base = doc["pooled"]["candidate"], doc["pooled"]["baseline"]
    rules = (
        ("reverse_recall", cand["reverse_recall"] >= base["reverse_recall"],
         "not worse"),
        ("reverse_precision",
         cand["reverse_precision"] >= base["reverse_precision"] - GATE_REVERSE_PRECISION_SLACK,
         f"not worse by more than {GATE_REVERSE_PRECISION_SLACK}"),
        ("steer_mae", cand["steer_mae"] <= base["steer_mae"] + GATE_STEER_MAE_SLACK,
         f"not worse by more than {GATE_STEER_MAE_SLACK}"),
    )
    checks = [{"metric": k, "candidate": cand[k], "baseline": base[k], "rule": rule,
               "ok": bool(ok)} for k, ok, rule in rules]
    return {"passes": all(c["ok"] for c in checks), "checks": checks,
            "pooled": {side: {k: doc["pooled"][side][k] for k in EVIDENCE_METRICS}
                       for side in ("candidate", "baseline")},
            "folds": len(doc["folds"]), "sessions": list(doc["sessions"])}


def describe_checks(check):
    return [f"{c['metric']} {c['candidate']:.3f} vs {c['baseline']:.3f} "
            f"({c['rule']}): {'ok' if c['ok'] else 'FAILS'}" for c in check["checks"]]


def load_evidence(path):
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise SystemExit(f"no evidence file {path}")
    with open(path) as f:
        try:
            doc = json.load(f)
        except ValueError as e:
            raise SystemExit(f"{path} is not JSON: {e}")
    problems = evidence_problems(doc)
    if problems:
        raise SystemExit(f"{path} is not schema {EVIDENCE_SCHEMA} evidence:\n  "
                         + "\n  ".join(problems))
    return path, doc


def evidence_record(path, doc, file_name, sha):
    return {"file": file_name, "sha256": sha, "source": os.path.basename(path),
            "candidate": doc["candidate"]["label"], "baseline": doc["baseline"]["label"],
            "summary": gate_check(doc)}


# ---------------------------------------------------------------------------
# The registry on disk.
# ---------------------------------------------------------------------------

def registry_path(models_dir=MODELS_DIR):
    return os.path.join(models_dir, REGISTRY)


def load_registry(models_dir=MODELS_DIR):
    path = registry_path(models_dir)
    if not os.path.exists(path):
        return empty_registry()
    with open(path) as f:
        doc = json.load(f)
    try:
        return upgrade_registry(doc)
    except ValueError as e:
        raise SystemExit(f"{path}: {e}")


def save_registry(registry, models_dir=MODELS_DIR):
    """Write the whole registry to a temp file and rename it over the old one,
    so a crash leaves the old registry or the new one, never half of one."""
    path = registry_path(models_dir)
    tmp = path + ".tmp"
    registry["releases"].sort(key=lambda e: (e["name"], parse_version(e["version"])))
    # Stable, so two registered in the same second keep their order.
    registry["candidates"].sort(key=lambda e: e["registered"])
    with open(tmp, "w") as f:
        json.dump(registry, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def copy_verified(src, dest, sha):
    """Copy src to dest through a temp name, and only rename it into place
    once it hashes to sha. Never over an existing file."""
    if os.path.exists(dest):
        raise SystemExit(f"{dest} already exists; nothing is overwritten")
    tmp = dest + ".part"
    shutil.copyfile(src, tmp)
    got = pm.sha256_local(tmp)
    if got != sha:
        os.remove(tmp)
        raise SystemExit(f"copy of {src} hashed {got}, not {sha}")
    os.replace(tmp, dest)


def write_verified_json(doc, dest):
    if os.path.exists(dest):
        raise SystemExit(f"{dest} already exists; nothing is overwritten")
    tmp = dest + ".part"
    with open(tmp, "w") as f:
        # The trainer writes NaN into its sidecars and policy.py reads them
        # with json.load, so the copy keeps them exactly as they were.
        json.dump(doc, f, indent=2)
        f.write("\n")
    with open(tmp) as f:
        # Compared as text: NaN is never equal to itself.
        if json.dumps(json.load(f), sort_keys=True) != json.dumps(doc, sort_keys=True):
            raise SystemExit(f"{tmp} does not read back as written")
    os.replace(tmp, dest)
    return pm.sha256_local(dest)


def copy_model(entry, models_dir, stem, model_version):
    """entry's onnx to <stem>.onnx and its sidecar, renamed, to <stem>.json.
    -> (onnx name, json name, json sha256)."""
    src = {k: os.path.join(models_dir, entry[k]) for k in ("onnx", "json")}
    if pm.sha256_local(src["onnx"]) != entry["onnx_sha256"]:
        raise SystemExit(f"{src['onnx']} is not what the registry recorded")
    with open(src["json"]) as f:
        meta = json.load(f)
    dest = os.path.join(models_dir, stem)
    copy_verified(src["onnx"], dest + ".onnx", entry["onnx_sha256"])
    json_sha = write_verified_json(registered_sidecar(meta, entry["name"], model_version),
                                   dest + ".json")
    return stem + ".onnx", stem + ".json", json_sha


def session_hashes(names, mcap_dir=MCAP_DIR, hash_file=None):
    """[{name, mcap_sha256}] with None for a session not on this Mac.

    A record session is in mcap_dir; an auto drive a model learned my
    takeovers from is in its drives/ folder (policy_dataset --drive-logs)."""
    hash_file = hash_file or pm.sha256_local
    out = []
    for name in names or []:
        path = os.path.join(mcap_dir, name + ".mcap")
        if not os.path.isfile(path):
            path = os.path.join(mcap_dir, "drives", name + ".mcap")
        out.append({"name": name,
                    "mcap_sha256": hash_file(path) if os.path.isfile(path) else None})
    return out


def utc_now(now=None):
    return now or datetime.datetime.now(datetime.timezone.utc)


def stamp(now=None):
    return utc_now(now).strftime("%Y-%m-%dT%H:%M:%SZ")


def register(onnx_path, models_dir=MODELS_DIR, mcap_dir=MCAP_DIR, name=None,
             tag=None, notes="", now=None, evidence=None):
    """Copy a trained model into the registry as a candidate. -> entry.

    evidence, a held-out A/B against the latest release, is checked and
    copied in beside it (candidate-<tag>.evidence.json) as a record of why it
    was worth registering. It earns nothing: promote still wants its own."""
    onnx_path = os.path.abspath(os.path.expanduser(onnx_path))
    json_path = os.path.splitext(onnx_path)[0] + ".json"
    for path in (onnx_path, json_path):
        if not os.path.isfile(path):
            raise SystemExit(f"no {path}")
    why = pm.refuse_reason(models_dir)
    if why:
        raise SystemExit(why.replace("Recordings", "Models"))
    os.makedirs(models_dir, exist_ok=True)
    with open(json_path) as f:
        meta = json.load(f)
    for key in ("input_contract", "representation", "actions", "steer_scale"):
        if key not in meta:
            raise SystemExit(f"{json_path} has no {key}: not a policy sidecar")
    tag = tag if tag is not None else default_tag(onnx_path)
    try:
        check_tag(tag)
    except ValueError as e:
        raise SystemExit(f"{e}; give one with --tag")

    registry = load_registry(models_dir)
    onnx_sha = pm.sha256_local(onnx_path)
    already = find_by_sha(registry, onnx_sha)
    if already:
        raise SystemExit(f"{os.path.basename(onnx_path)} is already "
                         f"{already['label']} ({already['onnx']})")
    if find_candidate(registry, tag):
        raise SystemExit(f"candidate {tag} is already registered; give another --tag")
    name = name or meta.get("model_name") or FAMILY
    releases = versions_of(registry, name)
    parent = releases[-1]["version"] if releases else None
    ev = None
    if evidence is not None:
        ev_path, ev = load_evidence(evidence)
        want = (candidate_label(name, tag),releases[-1]["label"] if releases else None)
        got = (ev["candidate"]["label"], ev["baseline"]["label"])
        if got[0] != want[0] or (want[1] is not None and got[1] != want[1]):
            raise SystemExit(f"Refusing: {ev_path} compares {got[0]!r} with {got[1]!r}, "
                             f"not {want[0]!r} with {want[1]!r}")

    stem = os.path.join(models_dir, candidate_stem(tag))
    copy_verified(onnx_path, stem + ".onnx", onnx_sha)
    json_sha = write_verified_json(registered_sidecar(meta, name, f"candidate {tag}"),
                                   stem + ".json")
    # The original is untouched: still the bytes that were registered.
    if pm.sha256_local(onnx_path) != onnx_sha:
        raise SystemExit(f"{onnx_path} changed while it was being registered")

    sessions = {"train": session_hashes(meta.get("train_sessions"), mcap_dir),
                "val": session_hashes(meta.get("val_sessions"), mcap_dir)}
    entry = make_candidate(name, tag, meta, onnx_path, json_path, onnx_sha, json_sha,
                           pm.sha256_local(json_path), sessions, parent, notes,
                           stamp(now))
    if ev is not None:
        ev_sha = pm.sha256_local(ev_path)
        ev_name = candidate_stem(tag) + ".evidence.json"
        copy_verified(ev_path, os.path.join(models_dir, ev_name), ev_sha)
        entry["evidence"] = evidence_record(ev_path, ev, ev_name, ev_sha)
    if meta.get("model_version") not in (None, ""):
        # Only promote gives a number; one the sidecar carries is kept as a note.
        entry["sidecar_model_version_ignored"] = str(meta["model_version"])
    registry["candidates"].append(entry)
    save_registry(registry, models_dir)
    return entry


def promote(tag, models_dir=MODELS_DIR, evidence=None, approved_on_car=False,
            notes="", force_gate=None, now=None):
    """Turn candidate `tag` into the next release, behind both gates. -> entry."""
    registry = load_registry(models_dir)
    cand = find_candidate(registry, tag)
    if cand is None:
        raise SystemExit(f"no candidate {tag} in the registry")
    if cand.get("promoted_to"):
        raise SystemExit(f"{cand['label']} is already "
                         f"{label(cand['name'], cand['promoted_to'])}")
    name = cand["name"]
    releases = versions_of(registry, name)
    latest = releases[-1] if releases else None

    if not approved_on_car:
        raise SystemExit(f"Refusing: {cand['label']} needs --approved-on-car. Deploy it "
                         f"(./scripts/models.sh deploy {cand['onnx'][:-5]}), drive it, and "
                         f"promote it once it drove well.")
    ev = None
    if evidence is not None:
        ev_path, ev = load_evidence(evidence)
        mismatch = []
        if ev["candidate"]["label"] != cand["label"]:
            mismatch.append(f"its candidate is {ev['candidate']['label']!r}, "
                            f"not {cand['label']!r}")
        if latest and ev["baseline"]["label"] != latest["label"]:
            mismatch.append(f"its baseline is {ev['baseline']['label']!r}, "
                            f"not the latest release {latest['label']!r}")
        if mismatch:
            raise SystemExit(f"Refusing: {ev_path} is evidence for another pair: "
                             + "; ".join(mismatch))
    check = gate_check(ev) if ev is not None else None
    if latest is None:
        gate = "first release"
    elif force_gate:
        gate = "forced"
    elif ev is None:
        raise SystemExit(f"Refusing: promoting {cand['label']} over {latest['label']} "
                         f"needs --evidence, a held-out comparison of the two "
                         f"(or --force-gate \"<reason>\")")
    elif not check["passes"]:
        raise SystemExit(f"Refusing: {cand['label']} does not beat {latest['label']} "
                         f"on the held-out folds:\n  " + "\n  ".join(describe_checks(check)))
    else:
        gate = "passed"

    version, why = next_version(registry, name, cand["contract"])
    stem = file_stem(name, version)
    set_aside_demoted(registry, models_dir, stem)
    onnx_name, json_name, json_sha = copy_model(cand, models_dir, stem, version)
    record = None
    if ev is not None:
        ev_name = stem + ".evidence.json"
        ev_sha = pm.sha256_local(ev_path)
        copy_verified(ev_path, os.path.join(models_dir, ev_name), ev_sha)
        record = evidence_record(ev_path, ev, ev_name, ev_sha)

    keep = {k: v for k, v in cand.items()
            if k not in ("tag", "promoted_to", "demoted", "sidecar_model_version_ignored")}
    entry = {**keep, "version": version, "label": label(name, version),
             "onnx": onnx_name, "json": json_name, "json_sha256": json_sha,
             "parent": latest["version"] if latest else None,
             "why_version": why, "candidate": tag, "promoted": stamp(now),
             "gate": gate, "evidence": record,
             "approval": {"status": "approved", "date": utc_now(now).strftime("%Y-%m-%d"),
                          "notes": notes or ""}}
    if gate == "forced":
        entry["gate_reason"] = force_gate
    cand["promoted_to"] = version
    registry["releases"].append(entry)
    save_registry(registry, models_dir)
    return entry


def set_aside_demoted(registry, models_dir, stem):
    """A number given back by demote leaves its kblock-X.Y.* files on disk.
    When the number is given out again, those files move to
    kblock-X.Y.demoted-<tag>.* first -- renamed, never deleted, and the
    demoted candidate's record says where they went. Any other file already
    under the name stops the promote in copy_verified."""
    for cand in registry["candidates"]:
        demoted = cand.get("demoted") or {}
        files = demoted.get("files") or {}
        for kind in ("onnx", "json"):
            old = files.get(kind)
            if old != f"{stem}.{kind}":
                continue
            path = os.path.join(models_dir, old)
            if not os.path.exists(path):
                continue
            if pm.sha256_local(path) != files.get(f"{kind}_sha256"):
                raise SystemExit(f"{path} is not the file {cand['label']} was demoted "
                                 f"from; not moving it")
            new = f"{stem}.demoted-{cand['tag']}.{kind}"
            if os.path.exists(os.path.join(models_dir, new)):
                raise SystemExit(f"{new} already exists; nothing is overwritten")
            os.rename(path, os.path.join(models_dir, new))
            files[kind] = new


def demote(version, tag, reason, car, models_dir=MODELS_DIR, name=None, now=None):
    """Release `version` back to a candidate: its number is given back.
    Refuses the model the car is running. -> the candidate entry."""
    registry = load_registry(models_dir)
    name = name or FAMILY
    entry = find(registry, name, version)
    if entry is None:
        raise SystemExit(f"no {label(name, version)} in the registry")
    if not reason or not reason.strip():
        raise SystemExit("demote needs --reason: the registry keeps why")
    try:
        check_tag(tag)
    except ValueError as e:
        raise SystemExit(str(e))
    if find_candidate(registry, tag):
        raise SystemExit(f"candidate {tag} is already registered; give another --tag")
    # Survey the car before anything moves; an unreachable car stops this here.
    _, state, sha = active_entry(registry, car)
    if sha == entry["onnx_sha256"]:
        raise SystemExit(f"Refusing: the car's policy.onnx ({state.get('onnx')}) is "
                         f"{entry['label']}. Deploy another model first.")

    onnx_name, json_name, json_sha = copy_model(
        entry, models_dir, candidate_stem(tag), f"candidate {tag}")
    keep = {k: v for k, v in entry.items()
            if k not in ("version", "label", "why_version", "gate", "gate_reason",
                         "evidence", "approval", "candidate", "promoted")}
    cand = {**keep, "tag": tag, "label": candidate_label(name, tag),
            "onnx": onnx_name, "json": json_name, "json_sha256": json_sha,
            "promoted_to": None,
            "demoted": {"from": entry["version"], "date": stamp(now), "reason": reason,
                        "files": {"onnx": entry["onnx"], "onnx_sha256": entry["onnx_sha256"],
                                  "json": entry["json"], "json_sha256": entry["json_sha256"]}}}
    registry["releases"].remove(entry)
    registry["candidates"].append(cand)
    save_registry(registry, models_dir)
    return cand


NOTE_KINDS = ("drive", "general")
NOTES_KEY = "notes_log"


def add_note(entry, kind, text, now=None):
    """Append a dated {date, kind, text} to the entry's note list. The single
    `notes` string stays what it was at registration."""
    entry.setdefault(NOTES_KEY, []).append(
        {"date": stamp(now), "kind": kind, "text": text})


def note(target, text, models_dir=MODELS_DIR, name=None, kind="drive", now=None):
    """Add a note to a release (1.2) or a candidate (candidate-<tag>); only the
    registry is written, never a model file. -> entry."""
    if kind not in NOTE_KINDS:
        raise SystemExit(f"note kind is one of {', '.join(NOTE_KINDS)}, not {kind!r}")
    if not (text or "").strip():
        raise SystemExit("a note needs some text")
    registry = load_registry(models_dir)
    entry = find_entry(registry, target, name)
    add_note(entry, kind, text.strip(), now)
    save_registry(registry, models_dir)
    return entry


def find_entry(registry, target, name=None):
    if target.startswith(CANDIDATE_PREFIX):
        entry = find_candidate(registry, target[len(CANDIDATE_PREFIX):])
        if entry is None:
            raise SystemExit(f"no candidate {target[len(CANDIDATE_PREFIX):]} in the registry")
        return entry
    entry = None
    try:
        entry = find(registry, name or FAMILY, target)
    except ValueError:
        pass
    if entry is None:
        raise SystemExit(f"no {label(name or FAMILY, target)} in the registry")
    return entry


def approve(version, models_dir=MODELS_DIR, name=None, notes="", now=None):
    """Record my on-car approval of a release that has none yet. -> entry."""
    registry = load_registry(models_dir)
    entry = find(registry, name or FAMILY, version)
    if entry is None:
        raise SystemExit(f"no {label(name or FAMILY, version)} in the registry")
    if (entry.get("approval") or {}).get("status") == "approved":
        raise SystemExit(f"{entry['label']} was approved on "
                         f"{entry['approval']['date']} already")
    entry["approval"] = {"status": "approved", "date": utc_now(now).strftime("%Y-%m-%d"),
                         "notes": notes or ""}
    if (notes or "").strip():
        add_note(entry, "approval", notes.strip(), now)
    save_registry(registry, models_dir)
    return entry


def attach(version, evidence, gate_note, models_dir=MODELS_DIR, name=None):
    """Attach held-out evidence to a release from before the gate, as a record.
    It does not make the release gated: the rule's verdict is kept beside it,
    pass or fail, and the gate reads "pre-gate, <note>". -> entry."""
    registry = load_registry(models_dir)
    entry = find(registry, name or FAMILY, version)
    if entry is None:
        raise SystemExit(f"no {label(name or FAMILY, version)} in the registry")
    if not str(entry.get("gate", "")).startswith("pre-gate"):
        raise SystemExit(f"{entry['label']} went through the gate ({entry.get('gate')}); "
                         f"its evidence is already its own")
    ev_path, ev = load_evidence(evidence)
    if ev["candidate"]["label"] != entry["label"]:
        raise SystemExit(f"{ev_path} is evidence for {ev['candidate']['label']!r}, "
                         f"not {entry['label']!r}")
    ev_name = file_stem(entry["name"], entry["version"]) + ".evidence.json"
    ev_sha = pm.sha256_local(ev_path)
    copy_verified(ev_path, os.path.join(models_dir, ev_name), ev_sha)
    entry["gate"] = f"pre-gate, {gate_note}" if gate_note else "pre-gate"
    entry["evidence"] = evidence_record(ev_path, ev, ev_name, ev_sha)
    if entry.get("approval") is None:
        entry["approval"] = {"status": "pending"}
    save_registry(registry, models_dir)
    return entry


# ---------------------------------------------------------------------------
# The car. Every command is built by a pure function, so the tests can read
# exactly what would run; Car is the one thing that runs them.
# ---------------------------------------------------------------------------

def parked_check_shell():
    """Shell that exits 3, printing NOT_PARKED <reading>, unless the throttle
    GPIO is at the ESC's neutral pulse."""
    return (f'p="$(pigs gpw {THROTTLE_GPIO} 2>&1)"; '
            f'[ "$p" = "{PARKED_US}" ] || {{ echo "NOT_PARKED $p"; exit 3; }}')


def is_parked(reading):
    return str(reading).strip() == str(PARKED_US)


def probe_command(slot="policy"):
    """Read-only: the throttle pulse, the slot's symlinks and what they point
    at, and the service's restart count."""
    m = shlex.quote(CAR_MODELS)
    link_onnx, link_json, previous, _ = SLOTS[slot]
    return (f"cd {m} && echo pulse=$(pigs gpw {THROTTLE_GPIO} 2>&1); "
            f"echo onnx=$(readlink {link_onnx}); echo json=$(readlink {link_json}); "
            f"echo previous=$(cat {previous} 2>/dev/null); "
            f"echo nrestarts=$(systemctl show {SERVICE} -p NRestarts --value); "
            f"echo active=$(systemctl is-active {SERVICE}); "
            f"echo since=$(systemctl show {SERVICE} -p ActiveEnterTimestamp "
            f"--timestamp=unix --value | tr -d @); "
            f"echo linked=$(stat -c %Y {link_onnx} 2>/dev/null)")


def parse_kv(text):
    out = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            out[key.strip()] = value.strip()
    return out


def sync_command(paths):
    """Flush the copied files and the models directory to the card. ext4
    holds new data in the page cache for seconds, and a sha256 then reads the
    cache: a power cut after it passed has left both files all zeros."""
    return "sync -- " + " ".join(shlex.quote(p) for p in [*paths, CAR_MODELS])


def swap_command(entry, previous_target, slot="policy"):
    """One shell: parked check, both symlinks re-pointed atomically, the old
    target remembered, all of it flushed to the card, the service restarted.
    Prints the car's clock just before the restart as restart_at=<unix s>."""
    m = shlex.quote(CAR_MODELS)
    link_onnx, link_json, previous, _ = SLOTS[slot]
    steps = [f"cd {m}", "{ " + parked_check_shell() + "; }"]
    if previous_target:
        steps.append(f"printf '%s\\n' {shlex.quote(previous_target)} > {previous}.tmp"
                     f" && mv -T {previous}.tmp {previous}")
    for target, link in ((entry["onnx"], link_onnx), (entry["json"], link_json)):
        steps.append(f"ln -sfn {shlex.quote(target)} .{link}.tmp && "
                     f"mv -T .{link}.tmp {link}")
    # A symlink cannot be fsynced through its own path, so the whole card.
    steps.append("sync")
    steps.append("echo restart_at=$(date +%s)")
    steps.append(f"sudo -n systemctl restart {SERVICE}")
    return " && ".join(steps)


def health_command(restart_at):
    return (f"echo active=$(systemctl is-active {SERVICE}); "
            f"echo nrestarts=$(systemctl show {SERVICE} -p NRestarts --value); "
            f"echo listening=$(ss -ltnH '( sport = :{PORT} )' | grep -c LISTEN); "
            f"echo pulse=$(pigs gpw {THROTTLE_GPIO} 2>&1); "
            f"echo tracebacks=$(journalctl -u {SERVICE} --since @{int(restart_at)} "
            f"--no-pager 2>&1 | grep -c Traceback)")


def health_problems(h, nrestarts_before):
    """What is wrong with a health readout, [] when nothing is."""
    problems = []
    if h.get("active") != "active":
        problems.append(f"service is {h.get('active')!r}, not active")
    try:
        if int(h.get("nrestarts", "x")) > int(nrestarts_before):
            problems.append(f"NRestarts went {nrestarts_before} -> {h['nrestarts']}: "
                            f"it crashed and systemd restarted it")
    except ValueError:
        problems.append(f"NRestarts unreadable: {h.get('nrestarts')!r}")
    if h.get("listening", "0") in ("", "0"):
        problems.append(f"nothing listening on {PORT}")
    if not is_parked(h.get("pulse", "")):
        problems.append(f"throttle GPIO reads {h.get('pulse')!r}, not {PARKED_US}")
    if h.get("tracebacks", "0") not in ("0", ""):
        problems.append(f"{h['tracebacks']} Traceback(s) in the journal since the restart")
    return problems


def copy_plan(entry, remote_sha):
    """The registered files the car is missing -> [name]. SystemExit for one
    it has under the same name with other bytes: that is not this model."""
    missing = []
    for name, sha in ((entry["onnx"], entry["onnx_sha256"]),
                      (entry["json"], entry["json_sha256"])):
        got = remote_sha.get(f"{CAR_MODELS}/{name}")
        if got is None:
            missing.append(name)
        elif got != sha:
            raise SystemExit(f"the car's {name} hashes {got}, the registry says "
                             f"{sha}; not touching it")
    return missing


class Car:
    """ssh and scp to the car. The tests replace this."""

    def __init__(self, host=None):
        self.host = host or pm.CAR

    def run(self, command, check=True):
        out = subprocess.run(["ssh", "-o", "ConnectTimeout=10", self.host, command],
                             capture_output=True, text=True)
        if check and out.returncode != 0:
            raise SystemExit(f"ssh {self.host} failed ({out.returncode}): "
                             f"{(out.stdout + out.stderr).strip()}")
        return out.returncode, out.stdout

    def sha256(self, paths):
        return pm.sha256_remote(self.host, paths)

    def put(self, local, remote):
        """Copy to remote via remote.part, renamed only once scp finished."""
        part = remote + ".part"
        out = subprocess.run(["scp", "-q", "-o", "ConnectTimeout=10", local,
                              f"{self.host}:{part}"], capture_output=True, text=True)
        if out.returncode != 0:
            raise SystemExit(f"scp {local} failed: {out.stderr.strip()}")
        self.run(f"mv -n {shlex.quote(part)} {shlex.quote(remote)} && "
                 f"test ! -e {shlex.quote(part)}")


def survey(car, slot="policy"):
    _, out = car.run(probe_command(slot))
    return parse_kv(out)


def active_entry(registry, car, state=None):
    """(entry or None, state, onnx sha on the car) for what policy.onnx points at."""
    state = state or survey(car)
    target = state.get("onnx", "")
    if not target:
        return None, state, None
    path = target if target.startswith("/") else f"{CAR_MODELS}/{target}"
    sha = car.sha256([path]).get(path)
    entry = find_by_sha(registry, sha, os.path.basename(target)) if sha else None
    return entry, state, sha


def deploy(registry, entry, car, models_dir=MODELS_DIR, dry_run=False,
           out=print, wait_s=30.0, sleep=time.sleep, clock=time.monotonic,
           slot="policy"):
    """Put `entry`, a release or a candidate, on the car in `slot` (SLOTS:
    the policy, or the map model's fallback) and restart the service under
    it. -> exit code."""
    link_onnx = SLOTS[slot][0]
    say = (lambda s: out(f"[dry-run] {s}")) if dry_run else out
    local = {n: os.path.join(models_dir, entry[n]) for n in ("onnx", "json")}
    for kind, path in local.items():
        if not os.path.isfile(path) or pm.sha256_local(path) != entry[f"{kind}_sha256"]:
            raise SystemExit(f"{path} is missing or not what the registry recorded")
    if slot == "fallback":
        # The car refuses a map fallback at engage (auto unavailable), so
        # it is refused here before anything moves.
        with open(local["json"]) as f:
            if json.load(f).get("uses_map", False):
                raise SystemExit(f"{entry['label']} drives off the map: a fallback "
                                 "is the model for when there is none")

    state = survey(car, slot)
    say(f"car: {link_onnx} -> {state.get('onnx') or '(none)'}, throttle pulse "
        f"{state.get('pulse')}, service {state.get('active')}, NRestarts "
        f"{state.get('nrestarts')}")
    if not is_parked(state.get("pulse", "")):
        out(f"Refusing: pigs gpw {THROTTLE_GPIO} reads {state.get('pulse')!r}, not "
            f"{PARKED_US}. Park the car (manual, hands off) and try again.")
        return 3

    remote = {n: f"{CAR_MODELS}/{entry[n]}" for n in ("onnx", "json")}
    missing = copy_plan(entry, car.sha256(list(remote.values())))
    for name in missing:
        kind = "onnx" if name.endswith(".onnx") else "json"
        say(f"copy {local[kind]} -> {car.host}:{remote[kind]}")
        if not dry_run:
            car.put(local[kind], remote[kind])
    if not missing:
        say(f"{entry['onnx']} and {entry['json']} already on the car, sha256 match")
    if missing:
        command = sync_command([remote["onnx"], remote["json"]])
        say(f"ssh {car.host} '{command}'")
        if not dry_run:
            car.run(command)
    if not dry_run and missing:
        if copy_plan(entry, car.sha256(list(remote.values()))):
            raise SystemExit("copied files are not on the car after the copy")
        out("sha256 verified on the car")

    previous = state.get("onnx", "")
    command = swap_command(entry, previous if previous != entry["onnx"] else
                           state.get("previous", ""), slot)
    say(f"ssh {car.host} '{command}'")
    if dry_run:
        say(f"then health: {health_command(0).replace('@0', '@<restart_at>')}")
        say(f"would deploy {entry['label']} ({entry['onnx_sha256'][:12]})")
        return 0

    code, text = car.run(command, check=False)
    if code == 3 or "NOT_PARKED" in text:
        out(f"Refusing: the car stopped being parked ({text.strip()}). Nothing changed.")
        return 3
    if code != 0:
        out(f"The swap or the restart failed ({code}): {text.strip()}")
        return 1
    kv = parse_kv(text)
    restart_at = int(kv.get("restart_at", "0") or 0)
    out(f"{link_onnx} -> {entry['onnx']}; restarted {SERVICE}")
    mark_deployed(registry, entry, models_dir, slot=slot)

    deadline = clock() + wait_s
    while True:
        sleep(2.0)
        _, htext = car.run(health_command(restart_at), check=False)
        h = parse_kv(htext)
        problems = health_problems(h, state.get("nrestarts", "0"))
        ready = h.get("active") == "active" and h.get("listening", "0") not in ("", "0")
        if (ready and not problems) or clock() >= deadline:
            break
    if problems:
        out("UNHEALTHY after the restart:")
        for p in problems:
            out(f"  {p}")
        out("Roll back with: ./scripts/models.sh rollback")
        return 1
    out(f"healthy: active, NRestarts {h.get('nrestarts')}, listening on {PORT}, "
        f"throttle {h.get('pulse')}, no Traceback. {entry['label']} drives.")
    if "tag" in entry:
        out(f"A test drive: promote {entry['tag']} --approved-on-car once it drove well.")
    return 0


def mark_deployed(registry, entry, models_dir=MODELS_DIR, now=None, slot="policy"):
    """Write down which model the car was last pointed at, so the registry
    can say without an ssh (tools/retrain.py grades against it)."""
    registry[SLOTS[slot][3]] = {"label": entry["label"], "onnx": entry["onnx"],
                            "onnx_sha256": entry["onnx_sha256"], "date": stamp(now)}
    save_registry(registry, models_dir)


def rollback_entry(registry, car, slot="policy"):
    """The registered model models/.policy-previous (or the slot's) names,
    by its hash."""
    state = survey(car, slot)
    target = state.get("previous", "")
    if not target:
        raise SystemExit(f"the car has no models/{SLOTS[slot][2]}: deploy a model by name")
    path = f"{CAR_MODELS}/{target}"
    sha = car.sha256([path]).get(path)
    entry = find_by_sha(registry, sha, os.path.basename(target)) if sha else None
    if entry is None:
        raise SystemExit(f"{target} on the car is not a registered model")
    return entry


# ---------------------------------------------------------------------------
# The commands.
# ---------------------------------------------------------------------------

def gate_text(e):
    gate = e.get("gate") or "-"
    if e.get("gate_reason"):
        gate += f" ({e['gate_reason']})"
    approval = e.get("approval")
    if approval is None:
        on_car = "not recorded"
    elif approval.get("status") == "approved":
        on_car = f"approved {approval.get('date')}"
        if approval.get("notes"):
            on_car += f": {approval['notes']}"
    else:
        on_car = approval.get("status", "?")
    ev = e.get("evidence")
    if ev:
        verdict = "passes" if ev["summary"]["passes"] else "fails"
        gate += f"; evidence {ev['file']} ({verdict} the rule vs {ev['baseline']})"
    return f"gate {gate}; on-car {on_car}"


def note_summary(e):
    """'N notes' and the newest drive note cut to one line, or '' for none."""
    notes = e.get(NOTES_KEY) or []
    if not notes:
        return ""
    text = f"{len(notes)} note{'s' if len(notes) != 1 else ''}"
    drive = [n for n in notes if n["kind"] == "drive"]
    if drive:
        one = " ".join(drive[-1]["text"].split())
        if len(one) > 60:
            one = one[:59].rstrip() + "…"
        text += f", latest drive {drive[-1]['date'][:10]}: {one}"
    return text


def list_lines(registry, active, models_dir, offline):
    lines = []

    def row(e):
        mark = "*" if active is e else " "
        sessions = len(e.get("train_sessions") or [])
        lines.append(f"{mark} {e['label']:<28} {e['created']}  {e['onnx_sha256'][:12]}  "
                     f"{sessions} session{'s' if sessions != 1 else ''}  "
                     f"parent {e['parent'] or '-':<4} {short_scores(e)}")

    if not registry["releases"] and not registry["candidates"]:
        return [f"no models in {registry_path(models_dir)}"]
    lines.append("Releases")
    if not registry["releases"]:
        lines.append("  (none)")
    for e in sorted(registry["releases"],
                    key=lambda e: (e["name"], parse_version(e["version"]))):
        row(e)
        lines.append(f"    {gate_text(e)}")
        if e.get("candidate"):
            lines.append(f"    promoted from candidate {e['candidate']}")
        if e.get("notes"):
            lines.append(f"    {e['notes']}")
        if note_summary(e):
            lines.append(f"    {note_summary(e)}")
    lines.append("Candidates")
    if not registry["candidates"]:
        lines.append("  (none)")
    for e in registry["candidates"]:
        row(e)
        if e.get("promoted_to"):
            lines.append(f"    promoted to {e['promoted_to']}")
        if e.get("demoted"):
            d = e["demoted"]
            lines.append(f"    demoted from {d['from']} on {d['date'][:10]}: {d['reason']}")
        if e.get("notes"):
            lines.append(f"    {e['notes']}")
        if note_summary(e):
            lines.append(f"    {note_summary(e)}")
    if not offline:
        lines.append("* = what the car's models/policy.onnx points at"
                     if active else "(the car's policy.onnx is no registered model)")
    return lines


def cmd_list(args, car_factory=Car):
    registry = load_registry(args.models_dir)
    active = None
    if not args.offline:
        try:
            active, _, _ = active_entry(registry, car_factory())
        except SystemExit as e:
            print(f"(car not reachable: {e})")
    for line in list_lines(registry, active, args.models_dir, args.offline):
        print(line)
    return 0


def cmd_register(args):
    entry = register(args.onnx, args.models_dir, args.mcap_dir, args.name, args.tag,
                     args.notes, evidence=args.evidence)
    print(f"registered {entry['label']} as {os.path.join(args.models_dir, entry['onnx'])}")
    print(f"  onnx sha256 {entry['onnx_sha256']}")
    print(f"  measured against {entry['parent'] or '(no release yet)'}, {short_scores(entry)}")
    if entry.get("sidecar_model_version_ignored"):
        print(f"  the sidecar said {entry['sidecar_model_version_ignored']}: ignored, "
              f"only promote gives a number")
    if entry.get("evidence"):
        print(f"  evidence {entry['evidence']['file']} vs {entry['evidence']['baseline']}:")
        for line in describe_checks(entry["evidence"]["summary"]):
            print(f"    {line}")
    print(f"  test drive: ./scripts/models.sh deploy {entry['onnx'][:-5]}")
    return 0


def cmd_promote(args):
    entry = promote(args.tag, args.models_dir, args.evidence, args.approved_on_car,
                    args.notes, args.force_gate)
    print(f"promoted candidate {args.tag} to {entry['label']} ({entry['why_version']}) as "
          f"{os.path.join(args.models_dir, entry['onnx'])}")
    print(f"  {gate_text(entry)}")
    if entry.get("evidence"):
        for line in describe_checks(entry["evidence"]["summary"]):
            print(f"    {line}")
    return 0


def cmd_demote(args, car_factory=Car):
    cand = demote(args.version, args.tag, args.reason, car_factory(), args.models_dir,
                  args.name)
    print(f"{label(cand['name'], args.version)} is now {cand['label']} "
          f"({os.path.join(args.models_dir, cand['onnx'])}); "
          f"{cand['demoted']['files']['onnx']} stays on disk")
    return 0


def cmd_approve(args):
    entry = approve(args.version, args.models_dir, args.name, args.notes)
    print(f"{entry['label']}: {gate_text(entry)}")
    return 0


def cmd_note(args):
    entry = note(args.target, args.text, args.models_dir, args.name, args.kind)
    print(f"{entry['label']}: {len(entry[NOTES_KEY])} notes")
    return 0


def cmd_notes(args):
    entry = find_entry(load_registry(args.models_dir), args.target, args.name)
    print(entry["label"])
    if entry.get("notes"):
        print(f"  registered: {entry['notes']}")
    for n in entry.get(NOTES_KEY) or []:
        print(f"  {n['date']}  [{n['kind']}]  {n['text']}")
    if not entry.get("notes") and not entry.get(NOTES_KEY):
        print("  (no notes)")
    return 0


def cmd_attach(args):
    entry = attach(args.version, args.evidence, args.gate_note, args.models_dir, args.name)
    print(f"{entry['label']}: {gate_text(entry)}")
    for line in describe_checks(entry["evidence"]["summary"]):
        print(f"    {line}")
    return 0


def _entry_for(registry, args):
    target = args.candidate
    if target is None and args.target and args.target.startswith(CANDIDATE_PREFIX):
        target = args.target[len(CANDIDATE_PREFIX):]
    if target is not None:
        entry = find_candidate(registry, target)
        if entry is None:
            raise SystemExit(f"no candidate {target} in the registry")
        return entry
    if not args.target:
        raise SystemExit("deploy what? a release (1.2) or a candidate "
                         "(candidate-<tag>, or --candidate <tag>)")
    name = args.name or FAMILY
    entry = find(registry, name, args.target)
    if entry is None:
        raise SystemExit(f"no {label(name, args.target)} in the registry")
    return entry


def _slot(args):
    return "fallback" if getattr(args, "fallback", False) else "policy"


def cmd_deploy(args, car_factory=Car):
    registry = load_registry(args.models_dir)
    return deploy(registry, _entry_for(registry, args), car_factory(),
                  args.models_dir, args.dry_run, slot=_slot(args))


def cmd_rollback(args, car_factory=Car):
    registry = load_registry(args.models_dir)
    car = car_factory()
    entry = rollback_entry(registry, car, _slot(args))
    print(f"rolling back to {entry['label']}")
    return deploy(registry, entry, car, args.models_dir, args.dry_run, slot=_slot(args))


def cmd_status(args, car_factory=Car):
    registry = load_registry(args.models_dir)
    car = car_factory()
    entry, state, sha = active_entry(registry, car)
    print(f"car {car.host}")
    print(f"  models/policy.onnx -> {state.get('onnx') or '(no link)'}")
    print(f"  models/policy.json -> {state.get('json') or '(no link)'}")
    print(f"  sha256 {sha or '(unreadable)'}")
    if entry is None:
        print("  NOT a registered model")
        rc = 1
    else:
        json_path = (state.get("json") if state.get("json", "").startswith("/")
                     else f"{CAR_MODELS}/{state.get('json', '')}")
        json_sha = car.sha256([json_path]).get(json_path)
        which = ("the registered sidecar" if json_sha == entry["json_sha256"]
                 else "the original trainer sidecar" if json_sha == entry["source"]["json_sha256"]
                 else "a sidecar the registry does not know")
        print(f"  = {entry['label']} (sha256 verified); policy.json is {which}")
        rc = 0 if json_sha in (entry["json_sha256"], entry["source"]["json_sha256"]) else 1
    print(f"  service {state.get('active')}, NRestarts {state.get('nrestarts')}, "
          f"throttle pulse {state.get('pulse')}")
    try:
        if int(state.get("linked") or 0) > int(state.get("since") or 0):
            print("  the link changed after the service started: it is still "
                  "driving the old model until a restart")
    except ValueError:
        pass
    if state.get("previous"):
        print(f"  rollback would go to {state['previous']}")
    return rc


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--models-dir", default=MODELS_DIR)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("list", help="releases and candidates, and which one the car runs")
    s.add_argument("--offline", action="store_true", help="do not ask the car")

    s = sub.add_parser("register", help="copy a trained model in as a candidate")
    s.add_argument("onnx")
    s.add_argument("--tag", default=None, help="short slug (default the policy-<stamp> stem)")
    s.add_argument("--name", default=None, help=f"family (default the sidecar's, else {FAMILY})")
    s.add_argument("--notes", default="")
    s.add_argument("--mcap-dir", default=MCAP_DIR)
    s.add_argument("--evidence", default=None,
                   help="a held-out A/B against the latest release, kept as a record")

    s = sub.add_parser("promote", help="give a candidate the next release number")
    s.add_argument("tag")
    s.add_argument("--evidence", default=None,
                   help=f"held-out comparison JSON (evidence schema {EVIDENCE_SCHEMA})")
    s.add_argument("--approved-on-car", action="store_true",
                   help="I drove it and it drove well")
    s.add_argument("--notes", default="", help="how it drove")
    s.add_argument("--force-gate", default=None, metavar="REASON",
                   help="promote without the held-out rule; the reason is kept")

    s = sub.add_parser("demote", help="give a release's number back; it becomes a candidate")
    s.add_argument("version")
    s.add_argument("--tag", required=True)
    s.add_argument("--reason", required=True)
    s.add_argument("--name", default=None)

    s = sub.add_parser("approve", help="record that a release drove well on the car")
    s.add_argument("version")
    s.add_argument("--notes", default="", help="how it drove")
    s.add_argument("--name", default=None)

    s = sub.add_parser("note", help="add a dated note to a release or candidate")
    s.add_argument("target", help="a release (1.2) or candidate-<tag>")
    s.add_argument("text")
    s.add_argument("--kind", choices=NOTE_KINDS, default="drive")
    s.add_argument("--name", default=None)

    s = sub.add_parser("notes", help="print every note on a release or candidate")
    s.add_argument("target")
    s.add_argument("--name", default=None)

    s = sub.add_parser("attach", help="attach held-out evidence to a pre-gate release")
    s.add_argument("version")
    s.add_argument("--evidence", required=True)
    s.add_argument("--gate-note", default="", help='e.g. "won held-out A/B"')
    s.add_argument("--name", default=None)

    for name, text in (("deploy", "put a release or a candidate on the car and restart"),
                       ("rollback", "deploy what drove before the last deploy")):
        s = sub.add_parser(name, help=text)
        if name == "deploy":
            s.add_argument("target", nargs="?", default=None,
                           help="a release (1.2) or candidate-<tag>")
            s.add_argument("--candidate", default=None, metavar="TAG")
        s.add_argument("--name", default=None)
        s.add_argument("--dry-run", action="store_true")
        s.add_argument("--fallback", action="store_true",
                       help="the map model's fallback slot (models/fallback.onnx), "
                            "not the policy")

    sub.add_parser("status", help="what the car's symlink points at, by sha256")
    args = p.parse_args(argv)
    args.models_dir = os.path.abspath(os.path.expanduser(args.models_dir))
    return args


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    return {"list": cmd_list, "register": cmd_register, "promote": cmd_promote,
            "demote": cmd_demote, "approve": cmd_approve, "note": cmd_note,
            "notes": cmd_notes, "attach": cmd_attach,
            "deploy": cmd_deploy, "rollback": cmd_rollback,
            "status": cmd_status}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
