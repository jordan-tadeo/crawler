#!/usr/bin/env python3
"""Named, versioned driving models: the registry on the Mac and the car's pick.

    python3 tools/models.py list                    # every version, which one drives
    python3 tools/models.py register ~/crawler-data/models/policy-<stamp>.onnx
    python3 tools/models.py register X.onnx --version 2.0 --notes "IMU in"
    python3 tools/models.py register X.onnx --bump minor
    python3 tools/models.py deploy 1.1 --dry-run    # say what it would do
    python3 tools/models.py deploy 1.1              # put kblock 1.1 on the car
    python3 tools/models.py rollback                # back to what drove before
    python3 tools/models.py status                  # what the car points at

models.sh runs this. Standard library only, so plain python3 on the Mac runs
it; nothing in here imports pigpio or commands a wheel.

Names. A model is a family name and MAJOR.MINOR, like a product: "kblock 1.1".
MAJOR goes up when the car's runtime has to know about the change -- the
input contract, representation, an input added (IMU, the previous action),
the outputs, the steering scale. MINOR goes up for a retrain on the same
contract. register works that out by comparing the new sidecar's contract
(CONTRACT_KEYS) with the newest registered version's, unless told with
--version or --bump, or the sidecar already carries model_version (written by
policy_train.py --model-version).

The registry is ~/crawler-data/models/registry.json, beside kblock-X.Y.onnx and
kblock-X.Y.json, never inside a git tree (the repo is public; the guard is
pull_mcaps.refuse_reason). Registering copies: the policy-<stamp> files the
trainer wrote stay exactly where and what they were, the copy is hashed against
the original before it is renamed into place, and the copy's sidecar gets
model_name / model_version so the car can say which model drove
(policy.model_label, the /footprint policy_model field). Nothing is ever
deleted and no version number is ever reused.

Deploy. The service runs `main.py --policy models/policy.onnx`, and
models/policy.onnx and models/policy.json are symlinks on the car. deploy
copies kblock-X.Y.* into the car's models/ if they are not there, checks their
sha256 on the car, and then -- in one ssh, behind a fresh parked check --
re-points each symlink atomically (ln -sfn to a temp name, mv -T over the
real one), remembers the old target in models/.policy-previous for rollback,
and restarts crawler.service. It refuses unless `pigs gpw 18` reads 1575, the
ESC's neutral pulse: a car with the throttle off neutral is a car someone is
driving, and a restart under it is a restart at speed. After the restart it
checks the service is active, has not crash-looped (NRestarts), is listening
on 8765, is still at neutral, and has written no Traceback since the restart.
--dry-run does only the read-only half and prints the rest.

The car is CRAWLER_HOST, else jt@jtpi.local, the same as pull_mcaps.py: by name,
because the DHCP lease moves (tests/test_car_address.py).
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pull_mcaps as pm  # noqa: E402

FAMILY = "kblock"
MODELS_DIR = os.path.expanduser(os.environ.get("CRAWLER_MODELS", "~/crawler-data/models"))
MCAP_DIR = os.path.expanduser(os.environ.get("CRAWLER_DATA", "~/crawler-data/mcap"))
REGISTRY = "registry.json"
SCHEMA = 1

CAR_REPO = "/home/jt/Documents/py/crawler"
CAR_MODELS = CAR_REPO + "/models"
# What the service is started with: main.py --policy models/policy.onnx, and
# PolicyModel reads the .json beside whatever path it is given.
LINK_ONNX = "policy.onnx"
LINK_JSON = "policy.json"
PREVIOUS = ".policy-previous"
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
    "outputs": None,
    "steer_scale": None,
    "steer_modes": None,
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
        out["imu_tilt"] = False
    return out


def contract_diff(a, b):
    """The contract keys that differ between two sidecars' contracts."""
    return [k for k in CONTRACT_KEYS if a.get(k) != b.get(k)]


def label(name, version):
    return f"{name} {version}"


def file_stem(name, version):
    return f"{name}-{version}"


def empty_registry():
    return {"schema": SCHEMA, "versions": []}


def versions_of(registry, name):
    """That family's entries, oldest version first."""
    return sorted((e for e in registry["versions"] if e["name"] == name),
                  key=lambda e: parse_version(e["version"]))


def find(registry, name, version):
    for e in registry["versions"]:
        if e["name"] == name and parse_version(e["version"]) == parse_version(version):
            return e
    return None


def find_by_sha(registry, sha):
    for e in registry["versions"]:
        if e["onnx_sha256"] == sha:
            return e
    return None


def choose_version(registry, name, meta, version=None, part=None, force=False):
    """(version, why) for a new model of `name` with sidecar `meta`.

    Order: --version, then --bump, then the sidecar's own model_version, then
    the contract against the newest version. ValueError for a version already
    taken, one that does not sort after every version of its major... and for
    a contract change kept under the same MAJOR, unless forced.
    """
    existing = versions_of(registry, name)
    latest = existing[-1] if existing else None
    if version is not None:
        chosen, why = format_version(parse_version(version)), "given"
    elif part is not None:
        chosen = bump(latest["version"], part) if latest else "1.0"
        why = f"--bump {part}" + (f" from {latest['version']}" if latest else "")
    elif meta.get("model_version") not in (None, ""):
        chosen, why = format_version(parse_version(meta["model_version"])), "sidecar"
    elif latest is None:
        chosen, why = "1.0", "first version"
    else:
        changed = contract_diff(contract_of(meta), latest["contract"])
        if changed:
            chosen = bump(latest["version"], "major")
            why = f"contract differs from {latest['version']} in {', '.join(changed)}"
        else:
            chosen = bump(latest["version"], "minor")
            why = f"same contract as {latest['version']}"

    if find(registry, name, chosen):
        raise ValueError(f"{label(name, chosen)} is already registered; "
                         f"versions are never reused")
    major = parse_version(chosen)[0]
    same_major = [e for e in existing if parse_version(e["version"])[0] == major]
    if same_major:
        top = same_major[-1]
        if parse_version(chosen) < parse_version(top["version"]):
            raise ValueError(f"{label(name, chosen)} sorts before {top['version']}, "
                             f"the newest {major}.x")
        changed = contract_diff(contract_of(meta), top["contract"])
        if changed and not force:
            raise ValueError(
                f"{label(name, chosen)} keeps MAJOR {major} but its contract "
                f"differs from {top['version']} in {', '.join(changed)}: that is "
                f"a MAJOR bump (--bump major), or --force if it really is not")
    return chosen, why


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


def make_entry(name, version, meta, source_onnx, source_json, onnx_sha,
               json_sha, source_json_sha, sessions, parent, notes, registered):
    stem = file_stem(name, version)
    return {
        "name": name,
        "version": version,
        "label": label(name, version),
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
    }


def registered_sidecar(meta, name, version):
    """The sidecar the registered copy carries: the trainer's, named."""
    return {**meta, "model_name": name, "model_version": version}


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
    if doc.get("schema") != SCHEMA or not isinstance(doc.get("versions"), list):
        raise SystemExit(f"{path}: not a schema {SCHEMA} registry")
    return doc


def save_registry(registry, models_dir=MODELS_DIR):
    """Write the whole registry to a temp file and rename it over the old one,
    so a crash leaves the old registry or the new one, never half of one."""
    path = registry_path(models_dir)
    tmp = path + ".tmp"
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


def session_hashes(names, mcap_dir=MCAP_DIR, hash_file=None):
    """[{name, mcap_sha256}] with None for a session not on this Mac."""
    hash_file = hash_file or pm.sha256_local
    out = []
    for name in names or []:
        path = os.path.join(mcap_dir, name + ".mcap")
        out.append({"name": name,
                    "mcap_sha256": hash_file(path) if os.path.isfile(path) else None})
    return out


def register(onnx_path, models_dir=MODELS_DIR, mcap_dir=MCAP_DIR, name=None,
             version=None, part=None, parent=None, notes="", force=False,
             now=None):
    """Copy a trained model into the registry as the next version. -> entry."""
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

    registry = load_registry(models_dir)
    onnx_sha = pm.sha256_local(onnx_path)
    already = find_by_sha(registry, onnx_sha)
    if already:
        raise SystemExit(f"{os.path.basename(onnx_path)} is already "
                         f"{already['label']} ({already['onnx']})")
    name = name or meta.get("model_name") or FAMILY
    try:
        chosen, reason = choose_version(registry, name, meta, version, part, force)
    except ValueError as e:
        raise SystemExit(str(e))
    if parent is None:
        before = [e for e in versions_of(registry, name)
                  if parse_version(e["version"]) < parse_version(chosen)]
        parent = before[-1]["version"] if before else None
    elif not find(registry, name, parent):
        raise SystemExit(f"--parent {parent}: no {label(name, parent)}")

    stem = os.path.join(models_dir, file_stem(name, chosen))
    copy_verified(onnx_path, stem + ".onnx", onnx_sha)
    json_sha = write_verified_json(registered_sidecar(meta, name, chosen), stem + ".json")
    # The original is untouched: still the bytes that were registered.
    if pm.sha256_local(onnx_path) != onnx_sha:
        raise SystemExit(f"{onnx_path} changed while it was being registered")

    sessions = {"train": session_hashes(meta.get("train_sessions"), mcap_dir),
                "val": session_hashes(meta.get("val_sessions"), mcap_dir)}
    registered = (now or datetime.datetime.now(datetime.timezone.utc)) \
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = make_entry(name, chosen, meta, onnx_path, json_path, onnx_sha,
                       json_sha, pm.sha256_local(json_path), sessions, parent,
                       notes, registered)
    entry["why_version"] = reason
    registry["versions"].append(entry)
    registry["versions"].sort(key=lambda e: (e["name"], parse_version(e["version"])))
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


def probe_command():
    """Read-only: the throttle pulse, the symlinks and what they point at,
    and the service's restart count."""
    m = shlex.quote(CAR_MODELS)
    return (f"cd {m} && echo pulse=$(pigs gpw {THROTTLE_GPIO} 2>&1); "
            f"echo onnx=$(readlink {LINK_ONNX}); echo json=$(readlink {LINK_JSON}); "
            f"echo previous=$(cat {PREVIOUS} 2>/dev/null); "
            f"echo nrestarts=$(systemctl show {SERVICE} -p NRestarts --value); "
            f"echo active=$(systemctl is-active {SERVICE}); "
            f"echo since=$(systemctl show {SERVICE} -p ActiveEnterTimestamp "
            f"--timestamp=unix --value | tr -d @); "
            f"echo linked=$(stat -c %Y {LINK_ONNX} 2>/dev/null)")


def parse_kv(text):
    out = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            out[key.strip()] = value.strip()
    return out


def swap_command(entry, previous_target):
    """One shell: parked check, both symlinks re-pointed atomically, the old
    target remembered, the service restarted. Prints the car's clock just
    before the restart as restart_at=<unix s>."""
    m = shlex.quote(CAR_MODELS)
    steps = [f"cd {m}", "{ " + parked_check_shell() + "; }"]
    if previous_target:
        steps.append(f"printf '%s\\n' {shlex.quote(previous_target)} > {PREVIOUS}.tmp"
                     f" && mv -T {PREVIOUS}.tmp {PREVIOUS}")
    for target, link in ((entry["onnx"], LINK_ONNX), (entry["json"], LINK_JSON)):
        steps.append(f"ln -sfn {shlex.quote(target)} .{link}.tmp && "
                     f"mv -T .{link}.tmp {link}")
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
    it has under the same name with other bytes: that is not this version."""
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


def survey(car):
    _, out = car.run(probe_command())
    return parse_kv(out)


def active_entry(registry, car, state=None):
    """(entry or None, state, onnx sha on the car) for what policy.onnx points at."""
    state = state or survey(car)
    target = state.get("onnx", "")
    if not target:
        return None, state, None
    path = target if target.startswith("/") else f"{CAR_MODELS}/{target}"
    sha = car.sha256([path]).get(path)
    return (find_by_sha(registry, sha) if sha else None), state, sha


def deploy(registry, entry, car, models_dir=MODELS_DIR, dry_run=False,
           out=print, wait_s=30.0, sleep=time.sleep, clock=time.monotonic):
    """Put `entry` on the car and restart the service under it. -> exit code."""
    say = (lambda s: out(f"[dry-run] {s}")) if dry_run else out
    local = {n: os.path.join(models_dir, entry[n]) for n in ("onnx", "json")}
    for kind, path in local.items():
        if not os.path.isfile(path) or pm.sha256_local(path) != entry[f"{kind}_sha256"]:
            raise SystemExit(f"{path} is missing or not what the registry recorded")

    state = survey(car)
    say(f"car: policy.onnx -> {state.get('onnx') or '(none)'}, throttle pulse "
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
    if not dry_run and missing:
        if copy_plan(entry, car.sha256(list(remote.values()))):
            raise SystemExit("copied files are not on the car after the copy")
        out("sha256 verified on the car")

    previous = state.get("onnx", "")
    command = swap_command(entry, previous if previous != entry["onnx"] else
                           state.get("previous", ""))
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
    out(f"policy.onnx -> {entry['onnx']}; restarted {SERVICE}")

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
        out("Roll back with: ./models.sh rollback")
        return 1
    out(f"healthy: active, NRestarts {h.get('nrestarts')}, listening on {PORT}, "
        f"throttle {h.get('pulse')}, no Traceback. {entry['label']} drives.")
    return 0


def rollback_entry(registry, car):
    """The registered version models/.policy-previous names, by its hash."""
    state = survey(car)
    target = state.get("previous", "")
    if not target:
        raise SystemExit("the car has no models/.policy-previous: deploy a version by name")
    path = f"{CAR_MODELS}/{target}"
    sha = car.sha256([path]).get(path)
    entry = find_by_sha(registry, sha) if sha else None
    if entry is None:
        raise SystemExit(f"{target} on the car is not a registered version")
    return entry


# ---------------------------------------------------------------------------
# The commands.
# ---------------------------------------------------------------------------

def cmd_list(args, car_factory=Car):
    registry = load_registry(args.models_dir)
    active = None
    if not args.offline:
        try:
            active, _, _ = active_entry(registry, car_factory())
        except SystemExit as e:
            print(f"(car not reachable: {e})")
    if not registry["versions"]:
        print(f"no versions in {registry_path(args.models_dir)}")
        return 0
    for e in registry["versions"]:
        mark = "*" if active is e else " "
        sessions = len(e.get("train_sessions") or [])
        print(f"{mark} {e['label']:<12} {e['created']}  {e['onnx_sha256'][:12]}  "
              f"{sessions} session{'s' if sessions != 1 else ''}  "
              f"parent {e['parent'] or '-':<4} {short_scores(e)}")
        if e.get("notes"):
            print(f"    {e['notes']}")
    if not args.offline:
        print("* = what the car's models/policy.onnx points at"
              if active else "(the car's policy.onnx is no registered version)")
    return 0


def cmd_register(args):
    entry = register(args.onnx, args.models_dir, args.mcap_dir, args.name,
                     args.version, args.bump, args.parent, args.notes, args.force)
    print(f"registered {entry['label']} ({entry['why_version']}) as "
          f"{os.path.join(args.models_dir, entry['onnx'])}")
    print(f"  onnx sha256 {entry['onnx_sha256']}")
    print(f"  parent {entry['parent']}, {short_scores(entry)}")
    return 0


def _entry_for(registry, args):
    name = args.name or FAMILY
    entry = find(registry, name, args.version)
    if entry is None:
        raise SystemExit(f"no {label(name, args.version)} in the registry")
    return entry


def cmd_deploy(args, car_factory=Car):
    registry = load_registry(args.models_dir)
    return deploy(registry, _entry_for(registry, args), car_factory(),
                  args.models_dir, args.dry_run)


def cmd_rollback(args, car_factory=Car):
    registry = load_registry(args.models_dir)
    car = car_factory()
    entry = rollback_entry(registry, car)
    print(f"rolling back to {entry['label']}")
    return deploy(registry, entry, car, args.models_dir, args.dry_run)


def cmd_status(args, car_factory=Car):
    registry = load_registry(args.models_dir)
    car = car_factory()
    entry, state, sha = active_entry(registry, car)
    print(f"car {car.host}")
    print(f"  models/policy.onnx -> {state.get('onnx') or '(no link)'}")
    print(f"  models/policy.json -> {state.get('json') or '(no link)'}")
    print(f"  sha256 {sha or '(unreadable)'}")
    if entry is None:
        print("  NOT a registered version")
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

    s = sub.add_parser("list", help="every version, and which one the car runs")
    s.add_argument("--offline", action="store_true", help="do not ask the car")

    s = sub.add_parser("register", help="copy a trained model in as the next version")
    s.add_argument("onnx")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--version", help="MAJOR.MINOR, instead of working it out")
    g.add_argument("--bump", choices=("major", "minor"))
    s.add_argument("--name", default=None, help=f"family (default the sidecar's, else {FAMILY})")
    s.add_argument("--parent", default=None, help="parent version (default the one before)")
    s.add_argument("--notes", default="")
    s.add_argument("--force", action="store_true",
                   help="allow a contract change under the same MAJOR")
    s.add_argument("--mcap-dir", default=MCAP_DIR)

    for name, text in (("deploy", "put a version on the car and restart"),
                       ("rollback", "deploy what drove before the last deploy")):
        s = sub.add_parser(name, help=text)
        if name == "deploy":
            s.add_argument("version")
        s.add_argument("--name", default=None)
        s.add_argument("--dry-run", action="store_true")

    sub.add_parser("status", help="what the car's symlink points at, by sha256")
    args = p.parse_args(argv)
    args.models_dir = os.path.abspath(os.path.expanduser(args.models_dir))
    return args


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    return {"list": cmd_list, "register": cmd_register, "deploy": cmd_deploy,
            "rollback": cmd_rollback, "status": cmd_status}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
