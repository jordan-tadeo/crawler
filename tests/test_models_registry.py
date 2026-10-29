"""The model registry: version numbers, the registry on disk, and deploy.

Deploy is the one that can hurt something. It restarts the service the car
drives under, so every test of it runs against FakeCar, which records the
commands instead of sending them, and the ones that matter are the refusals:
a car whose throttle pulse is not neutral gets nothing copied, nothing
re-pointed and no restart.
"""

import json
import math
import os
import pathlib
import sys

import pytest

import models as md
import policy_train as pt


def sidecar(**over):
    meta = {
        "input_contract": "depth_small",
        "representation": "profile",
        "actions": ["forward", "reverse", "stop"],
        "uses_prev_action": False,
        "outputs": ["logits", "throttle", "steer", "mode_logits"],
        "steer_scale": 0.7,
        "steer_modes": ["unified", "split"],
        "train_sessions": ["crawler-record-a", "crawler-record-b"],
        "val_sessions": [],
        "cv": {"blocks": 5, "guard_s": 1.0,
               "mean": {"direction_accuracy": 0.95, "steer_mae": 0.48,
                        "mode_switch_recall": float("nan")},
               "folds": [{"mode_window_accuracy": float("nan")}]},
    }
    meta.update(over)
    return meta


def write_model(folder, stem, payload, meta):
    onnx = folder / f"{stem}.onnx"
    onnx.write_bytes(payload)
    (folder / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    return onnx


def strict_json(path):
    def no_nan(token):
        raise ValueError(f"{token} in strict JSON")
    return json.loads(path.read_text(), parse_constant=no_nan)


# ---------------------------------------------------------------------------
# Versions.
# ---------------------------------------------------------------------------

def test_versions_parse_and_bump():
    assert md.parse_version("1.1") == (1, 1)
    assert md.parse_version("10.0") == (10, 0)
    assert md.bump("1.1", "minor") == "1.2"
    assert md.bump("1.1", "major") == "2.0"
    assert md.bump("2.9", "minor") == "2.10"
    for bad in ("1", "1.x", "v1.0", "1.0.0", "", "1.-1"):
        with pytest.raises(ValueError):
            md.parse_version(bad)
    with pytest.raises(ValueError):
        md.bump("1.0", "patch")


def test_versions_sort_as_numbers_not_text():
    reg = md.empty_registry()
    for v in ("1.10", "1.9", "1.2"):
        reg["releases"].append({"name": "kblock", "version": v, "contract": {}})
    assert [e["version"] for e in md.versions_of(reg, "kblock")] == ["1.2", "1.9", "1.10"]


def registry_with(*versions, meta=None):
    reg = md.empty_registry()
    for v in versions:
        reg["releases"].append({"name": "kblock", "version": v,
                                "contract": md.contract_of(meta or sidecar())})
    return reg


def test_a_retrain_on_the_same_contract_is_a_minor():
    v, why = md.next_version(registry_with("1.0", "1.1"), "kblock", md.contract_of(sidecar()))
    assert v == "1.2" and "same contract" in why


def test_a_new_input_is_a_major():
    imu = sidecar(uses_imu=True, imu_features=["gx"], imu_window=5, imu_tilt=True)
    assert md.next_version(registry_with("1.0", "1.1"), "kblock", md.contract_of(imu))[0] == "2.0"
    prev = sidecar(uses_prev_action=True)
    v, why = md.next_version(registry_with("1.1"), "kblock", md.contract_of(prev))
    assert v == "2.0" and "uses_prev_action" in why


def test_the_first_release_is_1_0():
    assert md.next_version(md.empty_registry(), "kblock", md.contract_of(sidecar()))[0] == "1.0"


def test_candidates_never_count_toward_the_next_number():
    reg = registry_with("1.0", "1.1")
    reg["candidates"].append({"name": "kblock", "tag": "x", "version": "9.9",
                              "contract": md.contract_of(sidecar(uses_imu=True))})
    assert md.next_version(reg, "kblock", md.contract_of(sidecar()))[0] == "1.2"


def test_a_schema_1_registry_reads_as_pre_gate_releases():
    old = {"schema": 1, "versions": [{"name": "kblock", "version": "1.0"},
                                     {"name": "kblock", "version": "1.1"}]}
    new = md.upgrade_registry(old)
    assert new["schema"] == md.SCHEMA and new["candidates"] == []
    assert [(e["version"], e["gate"], e["approval"]) for e in new["releases"]] == \
        [("1.0", "pre-gate", None), ("1.1", "pre-gate", None)]
    with pytest.raises(ValueError):
        md.upgrade_registry({"schema": 7})


def test_tags_are_short_file_safe_slugs():
    for good in ("imu-1028", "policy-20261028-195833Z", "a", "x.y_z"):
        assert md.check_tag(good) == good
    for bad in ("", "-x", "a/b", "a b", "..", "x" * 49, None):
        with pytest.raises(ValueError):
            md.check_tag(bad)


def test_imu_details_do_not_count_without_the_imu_input():
    assert md.contract_of(sidecar(imu_tilt=True)) == md.contract_of(sidecar())
    assert md.contract_of(sidecar(imu_yaw=True)) == md.contract_of(sidecar())


def test_dropping_the_yaw_is_a_major_and_an_old_contract_is_not_a_change():
    """A contract stored before imu_yaw existed (1.2's, imu-1028's) reads as
    the default: a retrain on it is still a minor. A no-yaw model off a
    full-IMU release is a major, and says why."""
    full = sidecar(**pt.imu_meta(True, False))
    stored = {k: v for k, v in md.contract_of(full).items() if k != "imu_yaw"}
    reg = md.empty_registry()
    reg["releases"].append({"name": "kblock", "version": "2.0", "contract": stored})
    assert md.next_version(reg, "kblock", md.contract_of(full))[0] == "2.1"
    no_yaw = sidecar(**pt.imu_meta(True, False, yaw=False))
    v, why = md.next_version(reg, "kblock", md.contract_of(no_yaw))
    assert v == "3.0" and "imu_yaw" in why
    base = registry_with("1.2", meta=sidecar())
    base["releases"][0]["contract"].pop("imu_yaw")
    assert md.next_version(base, "kblock", md.contract_of(sidecar()))[0] == "1.3"
    assert md.contract_of(no_yaw)["imu_yaw"] is False


# ---------------------------------------------------------------------------
# The registry on disk.
# ---------------------------------------------------------------------------

def test_a_new_model_registers_as_a_candidate_with_no_number(tmp_path):
    models = tmp_path / "models"
    mcaps = tmp_path / "mcap"
    models.mkdir()
    mcaps.mkdir()
    (mcaps / "crawler-record-a.mcap").write_bytes(b"session a")
    first = write_model(models, "policy-20261027-231046Z", b"onnx one",
                        sidecar(train_sessions=["crawler-record-a"]))
    # A sidecar stamped with a number by an older trainer still lands as a candidate.
    second = write_model(models, "policy-20261028-170131Z", b"onnx two",
                         sidecar(model_name="kblock", model_version="2.0"))
    originals = {p: p.read_bytes() for p in models.iterdir()}

    e1 = md.register(str(first), str(models), str(mcaps), notes="first model")
    e2 = md.register(str(second), str(models), str(mcaps), tag="imu-gyro")
    assert (e1["label"], e2["label"]) == ("kblock candidate policy-20261027-231046Z",
                                          "kblock candidate imu-gyro")
    assert "version" not in e1 and "version" not in e2
    assert e2["sidecar_model_version_ignored"] == "2.0"

    # The originals are exactly where and what they were.
    for path, data in originals.items():
        assert path.read_bytes() == data
    assert not list(models.glob("kblock-*"))

    reg = strict_json(models / "registry.json")   # no NaN made it in
    assert reg == md.load_registry(str(models))
    assert reg["releases"] == []
    c1, c2 = reg["candidates"]
    assert c1["tag"] == "policy-20261027-231046Z" and c1["notes"] == "first model"
    assert c1["onnx"] == "candidate-policy-20261027-231046Z.onnx"
    assert c2["onnx"] == "candidate-imu-gyro.onnx" and c2["promoted_to"] is None
    assert c1["parent"] is None
    assert c1["created"] == "2026-10-27T23:10:46Z"
    assert c1["onnx_sha256"] == md.pm.sha256_local(str(first))
    assert (models / "candidate-imu-gyro.onnx").read_bytes() == b"onnx two"
    assert c1["train_sessions"] == [{"name": "crawler-record-a",
                                     "mcap_sha256": md.pm.sha256_local(
                                         str(mcaps / "crawler-record-a.mcap"))}]
    assert c2["train_sessions"][1] == {"name": "crawler-record-b", "mcap_sha256": None}
    assert c1["train_commit"] is None
    assert c1["scores"]["cv"]["mean"]["direction_accuracy"] == 0.95
    assert c1["scores"]["cv"]["mean"]["mode_switch_recall"] is None
    assert "folds" not in c1["scores"]["cv"]

    copy = json.loads((models / "candidate-imu-gyro.json").read_text())
    assert (copy["model_name"], copy["model_version"]) == ("kblock", "candidate imu-gyro")
    assert math.isnan(copy["cv"]["mean"]["mode_switch_recall"])  # the copy is the sidecar
    assert c2["json_sha256"] == md.pm.sha256_local(str(models / "candidate-imu-gyro.json"))

    with pytest.raises(SystemExit, match="already kblock candidate imu-gyro"):
        md.register(str(second), str(models), str(mcaps), tag="again")
    third = write_model(models, "policy-3", b"onnx three", sidecar())
    with pytest.raises(SystemExit, match="already registered"):
        md.register(str(third), str(models), str(mcaps), tag="imu-gyro")
    with pytest.raises(SystemExit, match="--tag"):
        md.register(str(third), str(models), str(mcaps), tag="no/slash")
    assert len(md.load_registry(str(models))["candidates"]) == 2


def test_register_never_overwrites_a_file(tmp_path):
    src = write_model(tmp_path, "policy-x", b"bytes", sidecar())
    (tmp_path / "candidate-policy-x.onnx").write_bytes(b"something else")
    with pytest.raises(SystemExit, match="already exists"):
        md.register(str(src), str(tmp_path), str(tmp_path))
    assert (tmp_path / "candidate-policy-x.onnx").read_bytes() == b"something else"
    assert not (tmp_path / "registry.json").exists()


def test_a_bad_copy_is_never_renamed_into_place(tmp_path, monkeypatch):
    src = tmp_path / "a.onnx"
    src.write_bytes(b"real")
    with pytest.raises(SystemExit, match="hashed"):
        md.copy_verified(str(src), str(tmp_path / "b.onnx"), "0" * 64)
    assert not (tmp_path / "b.onnx").exists()
    assert not (tmp_path / "b.onnx.part").exists()


def test_the_registry_is_refused_inside_a_git_tree(tmp_path):
    (tmp_path / ".git").mkdir()
    src = write_model(tmp_path, "policy-x", b"bytes", sidecar())
    with pytest.raises(SystemExit, match="git work tree"):
        md.register(str(src), str(tmp_path / "models"), str(tmp_path))


# ---------------------------------------------------------------------------
# Deploy, against a car that only records.
# ---------------------------------------------------------------------------

class FakeCar:
    host = "jt@fake"

    def __init__(self, pulse="1575", on_car=None, swap_pulse=None, health=None):
        self.pulse = pulse
        self.files = dict(on_car or {})     # remote path -> sha256
        self.swap_pulse = swap_pulse or pulse
        self.health = health or {}
        self.commands = []
        self.puts = []

    def run(self, command, check=True):
        self.commands.append(command)
        if command.startswith(f"cd {md.CAR_MODELS} && echo pulse="):
            return 0, (f"pulse={self.pulse}\nonnx=policy-old.onnx\njson=policy-old.json\n"
                       f"previous=\nnrestarts=0\nactive=active\nsince=100\nlinked=50\n")
        if "systemctl restart" in command:
            if self.swap_pulse != "1575":
                return 3, f"NOT_PARKED {self.swap_pulse}\n"
            return 0, "restart_at=1234\n"
        if command.startswith("echo active="):
            h = {"active": "active", "nrestarts": "0", "listening": "1",
                 "pulse": "1575", "tracebacks": "0", **self.health}
            return 0, "".join(f"{k}={v}\n" for k, v in h.items())
        return 0, ""

    def sha256(self, paths):
        return {p: self.files[p] for p in paths if p in self.files}

    def put(self, local, remote):
        self.puts.append((local, remote))
        self.files[remote] = md.pm.sha256_local(local)


@pytest.fixture
def registered(tmp_path):
    """kblock 1.0, the first release: a candidate promoted with no baseline."""
    src = write_model(tmp_path, "policy-20261028-170131Z", b"the model", sidecar())
    md.register(str(src), str(tmp_path), str(tmp_path), tag="first")
    md.promote("first", str(tmp_path), approved_on_car=True)
    reg = md.load_registry(str(tmp_path))
    return reg, reg["releases"][0], str(tmp_path)


def deploy(registered, car, **kw):
    reg, entry, folder = registered
    lines = []
    rc = md.deploy(reg, entry, car, folder, out=lines.append, sleep=lambda s: None,
                   **kw)
    return rc, lines


def swaps(car):
    return [c for c in car.commands if "systemctl restart" in c]


def test_deploy_refuses_a_car_that_is_not_parked(registered):
    car = FakeCar(pulse="1700")
    rc, lines = deploy(registered, car)
    assert rc == 3
    assert car.puts == [] and swaps(car) == []
    assert any("Refusing" in line for line in lines)


def test_deploy_copies_verifies_repoints_and_restarts(registered):
    car = FakeCar()
    rc, lines = deploy(registered, car)
    assert rc == 0, lines
    assert [r for _, r in car.puts] == [f"{md.CAR_MODELS}/kblock-1.0.onnx",
                                        f"{md.CAR_MODELS}/kblock-1.0.json"]
    (swap,) = swaps(car)
    # The parked check is in the same shell as the restart, before anything moves.
    assert swap.index('[ "$p" = "1575" ]') < swap.index("ln -sfn")
    assert "ln -sfn kblock-1.0.onnx .policy.onnx.tmp && mv -T .policy.onnx.tmp policy.onnx" in swap
    assert "ln -sfn kblock-1.0.json .policy.json.tmp && mv -T .policy.json.tmp policy.json" in swap
    assert "policy-old.onnx > .policy-previous.tmp" in swap
    assert swap.endswith("sudo -n systemctl restart crawler")
    assert any("--since @1234" in c for c in car.commands)
    # Nothing here ever writes a pulse.
    assert not any("pigs s " in c or "pigs servo" in c or "pigs w " in c
                   for c in car.commands)


def test_deploy_skips_the_copy_of_files_already_there(registered):
    _, entry, _ = registered
    car = FakeCar(on_car={f"{md.CAR_MODELS}/{entry['onnx']}": entry["onnx_sha256"],
                          f"{md.CAR_MODELS}/{entry['json']}": entry["json_sha256"]})
    rc, _ = deploy(registered, car)
    assert rc == 0 and car.puts == [] and len(swaps(car)) == 1


def test_deploy_refuses_a_car_file_with_other_bytes(registered):
    _, entry, _ = registered
    car = FakeCar(on_car={f"{md.CAR_MODELS}/{entry['onnx']}": "f" * 64})
    with pytest.raises(SystemExit, match="not touching it"):
        deploy(registered, car)
    assert swaps(car) == []


def test_dry_run_changes_nothing(registered):
    car = FakeCar()
    rc, lines = deploy(registered, car, dry_run=True)
    assert rc == 0
    assert car.puts == [] and swaps(car) == []
    assert any("systemctl restart" in line for line in lines)


def test_a_deploy_writes_down_what_the_car_runs_and_a_dry_run_does_not(registered):
    _, entry, folder = registered
    deploy(registered, FakeCar(), dry_run=True)
    assert "deployed" not in md.load_registry(folder)
    deploy(registered, FakeCar(pulse="1700"))
    assert "deployed" not in md.load_registry(folder)
    rc, _ = deploy(registered, FakeCar())
    assert rc == 0
    mark = md.load_registry(folder)["deployed"]
    assert (mark["label"], mark["onnx"], mark["onnx_sha256"]) == (
        entry["label"], entry["onnx"], entry["onnx_sha256"])


def test_a_car_that_moves_before_the_swap_is_left_alone(registered):
    rc, lines = deploy(registered, FakeCar(swap_pulse="1650"))
    assert rc == 3 and any("stopped being parked" in line for line in lines)


@pytest.mark.parametrize("bad", [{"tracebacks": "2"}, {"nrestarts": "1"},
                                 {"pulse": "1600"}, {"active": "failed"},
                                 {"listening": "0"}])
def test_an_unhealthy_restart_says_so(registered, bad):
    rc, lines = deploy(registered, FakeCar(health=bad), wait_s=0.0)
    assert rc == 1 and "UNHEALTHY after the restart:" in lines


def test_a_candidate_deploys_through_the_same_checks(registered):
    reg, _, folder = registered
    src = os.path.join(folder, "policy-20261029-100000Z")
    with open(src + ".onnx", "wb") as f:
        f.write(b"a test drive")
    with open(src + ".json", "w") as f:
        json.dump(sidecar(), f)
    md.register(src + ".onnx", folder, folder, tag="try-me")
    reg = md.load_registry(folder)
    args = md.parse_args(["--models-dir", folder, "deploy", "candidate-try-me"])
    assert md._entry_for(reg, args)["tag"] == "try-me"
    args = md.parse_args(["--models-dir", folder, "deploy", "--candidate", "try-me"])
    cand = md._entry_for(reg, args)
    assert cand["label"] == "kblock candidate try-me"

    car = FakeCar()
    lines = []
    rc = md.deploy(reg, cand, car, folder, out=lines.append, sleep=lambda s: None)
    assert rc == 0, lines
    assert [r for _, r in car.puts] == [f"{md.CAR_MODELS}/candidate-try-me.onnx",
                                        f"{md.CAR_MODELS}/candidate-try-me.json"]
    (swap,) = swaps(car)
    assert swap.index('[ "$p" = "1575" ]') < swap.index("ln -sfn")
    assert "ln -sfn candidate-try-me.onnx .policy.onnx.tmp" in swap
    assert any("test drive" in line for line in lines)

    # Not parked: nothing copied, nothing re-pointed, as for a release.
    car = FakeCar(pulse="1700")
    assert md.deploy(reg, cand, car, folder, out=lambda s: None) == 3
    assert car.puts == [] and swaps(car) == []


# ---------------------------------------------------------------------------
# Promote: the held-out gate and my approval on the car.
# ---------------------------------------------------------------------------

def metrics(recall=0.5, precision=0.5, steer=0.5, bal=0.5):
    return {"reverse_recall": recall, "reverse_precision": precision,
            "steer_mae": steer, "direction_balanced_accuracy": bal}


def evidence(folder, candidate="kblock candidate next", baseline="kblock 1.0",
             cand=None, base=None, name="ab.json"):
    cand, base = cand or metrics(), base or metrics()
    doc = {"evidence_schema": 1,
           "candidate": {"label": candidate, "recipe": "new"},
           "baseline": {"label": baseline, "recipe": "old"},
           "method": "leave-one-session-out", "sessions": ["s1", "s2"],
           "folds": [{"name": s, "held_out": [s], "candidate": cand, "baseline": base}
                     for s in ("s1", "s2")],
           "pooled": {"candidate": cand, "baseline": base}}
    path = os.path.join(folder, name)
    with open(path, "w") as f:
        json.dump(doc, f)
    return path


@pytest.fixture
def with_candidate(registered):
    """kblock 1.0 released, and candidate "next" on the same contract."""
    _, _, folder = registered
    src = os.path.join(folder, "policy-20261029-120000Z")
    with open(src + ".onnx", "wb") as f:
        f.write(b"the next model")
    with open(src + ".json", "w") as f:
        json.dump(sidecar(), f)
    md.register(src + ".onnx", folder, folder, tag="next")
    return folder


def test_register_keeps_evidence_for_the_right_pair_as_a_record(registered):
    _, _, folder = registered
    src = os.path.join(folder, "policy-20261030-120000Z")
    with open(src + ".onnx", "wb") as f:
        f.write(b"a no-yaw model")
    with open(src + ".json", "w") as f:
        json.dump(sidecar(), f)
    wrong = evidence(folder, candidate="kblock candidate other", name="wrong.json")
    with pytest.raises(SystemExit, match="Refusing"):
        md.register(src + ".onnx", folder, folder, tag="nyaw", evidence=wrong)
    stale = evidence(folder, candidate="kblock candidate nyaw", baseline="kblock 0.9",
                     name="stale.json")
    with pytest.raises(SystemExit, match="Refusing"):
        md.register(src + ".onnx", folder, folder, tag="nyaw", evidence=stale)
    assert md.find_candidate(md.load_registry(folder), "nyaw") is None
    assert not os.path.exists(os.path.join(folder, "candidate-nyaw.onnx"))

    good = evidence(folder, candidate="kblock candidate nyaw", name="good.json")
    entry = md.register(src + ".onnx", folder, folder, tag="nyaw", evidence=good)
    kept = os.path.join(folder, "candidate-nyaw.evidence.json")
    assert entry["evidence"]["file"] == "candidate-nyaw.evidence.json"
    assert entry["evidence"]["sha256"] == md.pm.sha256_local(good) == md.pm.sha256_local(kept)
    assert entry["evidence"]["baseline"] == "kblock 1.0"
    assert "passes" in entry["evidence"]["summary"]
    assert "version" not in entry
    assert md.find_candidate(md.load_registry(folder), "nyaw")["evidence"] == entry["evidence"]


def test_promote_refuses_without_evidence(with_candidate):
    with pytest.raises(SystemExit, match="needs --evidence"):
        md.promote("next", with_candidate, approved_on_car=True)
    assert [e["version"] for e in md.load_registry(with_candidate)["releases"]] == ["1.0"]


def test_promote_refuses_a_failing_gate(with_candidate):
    base = metrics(recall=0.5, precision=0.6, steer=0.50)
    for cand, metric in ((metrics(recall=0.49, precision=0.6, steer=0.50), "reverse_recall"),
                         (metrics(recall=0.6, precision=0.54, steer=0.50), "reverse_precision"),
                         (metrics(recall=0.6, precision=0.6, steer=0.53), "steer_mae")):
        path = evidence(with_candidate, cand=cand, base=base)
        with pytest.raises(SystemExit, match=f"{metric} .*FAILS"):
            md.promote("next", with_candidate, evidence=path, approved_on_car=True)
    assert len(md.load_registry(with_candidate)["releases"]) == 1
    assert not os.path.exists(os.path.join(with_candidate, "kblock-1.1.onnx"))


def test_the_gate_allows_the_slack_and_no_more():
    def doc(cand, base):
        return {"pooled": {"candidate": cand, "baseline": base}, "folds": [{}],
                "sessions": ["s"]}
    base = metrics(recall=0.5, precision=0.6, steer=0.5)
    assert md.gate_check(doc(metrics(recall=0.5, precision=0.551, steer=0.5195), base))["passes"]
    assert not md.gate_check(doc(metrics(recall=0.5, precision=0.549, steer=0.5), base))["passes"]
    assert not md.gate_check(doc(metrics(recall=0.5, precision=0.6, steer=0.521), base))["passes"]


def test_promote_refuses_evidence_for_another_pair_or_a_bad_shape(with_candidate):
    path = evidence(with_candidate, baseline="kblock 0.9")
    with pytest.raises(SystemExit, match="another pair"):
        md.promote("next", with_candidate, evidence=path, approved_on_car=True)
    path = evidence(with_candidate, candidate="kblock candidate other")
    with pytest.raises(SystemExit, match="another pair"):
        md.promote("next", with_candidate, evidence=path, approved_on_car=True)
    path = evidence(with_candidate, cand={"reverse_recall": 0.9})
    with pytest.raises(SystemExit, match="reverse_precision is not a number"):
        md.promote("next", with_candidate, evidence=path, approved_on_car=True)


def test_promote_refuses_without_my_approval_on_the_car(with_candidate):
    path = evidence(with_candidate, cand=metrics(recall=0.9))
    with pytest.raises(SystemExit, match="--approved-on-car"):
        md.promote("next", with_candidate, evidence=path)
    assert len(md.load_registry(with_candidate)["releases"]) == 1


def test_promote_on_the_same_contract_is_the_next_minor(with_candidate):
    folder = with_candidate
    path = evidence(folder, cand=metrics(recall=0.7, precision=0.47, steer=0.51))
    entry = md.promote("next", folder, evidence=path, approved_on_car=True,
                       notes="clean three-point turn",
                       now=md.datetime.datetime(2026, 10, 29, 18, 0, tzinfo=md.datetime.timezone.utc))
    assert entry["label"] == "kblock 1.1" and entry["parent"] == "1.0"
    assert entry["gate"] == "passed" and entry["candidate"] == "next"
    assert entry["approval"] == {"status": "approved", "date": "2026-10-29",
                                 "notes": "clean three-point turn"}
    assert entry["evidence"]["summary"]["passes"] is True
    assert entry["evidence"]["file"] == "kblock-1.1.evidence.json"
    assert entry["evidence"]["sha256"] == md.pm.sha256_local(path)
    assert md.pm.sha256_local(os.path.join(folder, "kblock-1.1.evidence.json")) == \
        md.pm.sha256_local(path)

    strict_json(pathlib.Path(folder) / "registry.json")    # no NaN made it in
    reg = md.load_registry(folder)
    cand = md.find_candidate(reg, "next")
    assert cand["promoted_to"] == "1.1"     # kept, and linked
    for kind in ("onnx", "json"):
        assert os.path.exists(os.path.join(folder, f"candidate-next.{kind}"))
    with open(os.path.join(folder, "kblock-1.1.onnx"), "rb") as f:
        assert f.read() == b"the next model"
    with open(os.path.join(folder, "kblock-1.1.json")) as f:
        assert json.load(f)["model_version"] == "1.1"
    assert entry["onnx_sha256"] == cand["onnx_sha256"]
    with pytest.raises(SystemExit, match="already kblock 1.1"):
        md.promote("next", folder, evidence=path, approved_on_car=True)


def test_promote_off_the_contract_is_the_next_major(registered):
    _, _, folder = registered
    src = os.path.join(folder, "policy-imu")
    with open(src + ".onnx", "wb") as f:
        f.write(b"imu model")
    with open(src + ".json", "w") as f:
        json.dump(sidecar(uses_imu=True, imu_features=["gx"], imu_window=5), f)
    md.register(src + ".onnx", folder, folder, tag="imu")
    path = evidence(folder, candidate="kblock candidate imu")
    entry = md.promote("imu", folder, evidence=path, approved_on_car=True)
    assert entry["label"] == "kblock 2.0" and "uses_imu" in entry["why_version"]


def test_force_gate_is_recorded(with_candidate):
    path = evidence(with_candidate, cand=metrics(recall=0.1))
    entry = md.promote("next", with_candidate, evidence=path, approved_on_car=True,
                       force_gate="the folds miss the new terrain")
    assert entry["gate"] == "forced"
    assert entry["gate_reason"] == "the folds miss the new terrain"
    assert entry["evidence"]["summary"]["passes"] is False
    saved = md.find(md.load_registry(with_candidate), "kblock", "1.1")
    assert saved["gate_reason"] == "the folds miss the new terrain"


def test_force_gate_still_needs_the_car(with_candidate):
    with pytest.raises(SystemExit, match="--approved-on-car"):
        md.promote("next", with_candidate, force_gate="trust me")


# ---------------------------------------------------------------------------
# Demote: a number given back.
# ---------------------------------------------------------------------------

class CarOn(FakeCar):
    """A car whose policy.onnx points at `target` with sha256 `sha`."""

    def __init__(self, target, sha):
        super().__init__(on_car={f"{md.CAR_MODELS}/{target}": sha})
        self.target = target

    def run(self, command, check=True):
        code, text = super().run(command, check)
        return code, text.replace("policy-old.onnx", self.target)


@pytest.fixture
def with_imu_release(registered):
    """kblock 1.0, and kblock 2.0 the IMU model forced in over it."""
    _, _, folder = registered
    src = os.path.join(folder, "policy-20261028-195833Z")
    with open(src + ".onnx", "wb") as f:
        f.write(b"imu model")
    with open(src + ".json", "w") as f:
        json.dump(sidecar(uses_imu=True, imu_features=["gx"], imu_window=5), f)
    md.register(src + ".onnx", folder, folder, tag="imu-a")
    md.promote("imu-a", folder, approved_on_car=True, force_gate="before the gate")
    return folder


def test_demote_refuses_the_model_the_car_runs(with_imu_release):
    folder = with_imu_release
    two = md.find(md.load_registry(folder), "kblock", "2.0")
    with pytest.raises(SystemExit, match="Refusing: the car"):
        md.demote("2.0", "imu-1028", "lost the A/B",
                  CarOn("kblock-2.0.onnx", two["onnx_sha256"]), folder)
    assert md.find(md.load_registry(folder), "kblock", "2.0") is not None
    assert not os.path.exists(os.path.join(folder, "candidate-imu-1028.onnx"))


def test_demote_gives_the_number_back(with_imu_release):
    folder = with_imu_release
    one = md.find(md.load_registry(folder), "kblock", "1.0")
    cand = md.demote("2.0", "imu-1028", "lost the held-out A/B",
                     CarOn("kblock-1.0.onnx", one["onnx_sha256"]), folder)
    assert cand["label"] == "kblock candidate imu-1028"
    assert cand["demoted"]["from"] == "2.0"
    assert cand["demoted"]["reason"] == "lost the held-out A/B"
    reg = md.load_registry(folder)
    assert [e["version"] for e in reg["releases"]] == ["1.0"]
    assert md.find_candidate(reg, "imu-1028")["onnx_sha256"] == \
        md.pm.sha256_local(os.path.join(folder, "kblock-2.0.onnx"))
    # The old files stay on disk; the candidate's own copy carries its new name.
    for kind in ("onnx", "json"):
        assert os.path.exists(os.path.join(folder, f"kblock-2.0.{kind}"))
    with open(os.path.join(folder, "candidate-imu-1028.json")) as f:
        assert json.load(f)["model_version"] == "candidate imu-1028"

    # The next number comes from releases only: the next IMU winner is 2.0 again,
    # and the files 2.0 left behind are set aside, never overwritten.
    left = md.pm.sha256_local(os.path.join(folder, "kblock-2.0.onnx"))
    src = os.path.join(folder, "policy-imu-b")
    with open(src + ".onnx", "wb") as f:
        f.write(b"a better imu model")
    with open(src + ".json", "w") as f:
        json.dump(sidecar(uses_imu=True, imu_features=["gx"], imu_window=5), f)
    md.register(src + ".onnx", folder, folder, tag="imu-b")
    path = evidence(folder, candidate="kblock candidate imu-b")
    entry = md.promote("imu-b", folder, evidence=path, approved_on_car=True)
    assert entry["label"] == "kblock 2.0"
    with open(os.path.join(folder, "kblock-2.0.onnx"), "rb") as f:
        assert f.read() == b"a better imu model"
    assert md.pm.sha256_local(os.path.join(folder, "kblock-2.0.demoted-imu-1028.onnx")) == left
    files = md.find_candidate(md.load_registry(folder), "imu-1028")["demoted"]["files"]
    assert files["onnx"] == "kblock-2.0.demoted-imu-1028.onnx"


def test_demote_refuses_when_the_car_cannot_be_asked(with_imu_release):
    class Gone(FakeCar):
        def run(self, command, check=True):
            raise SystemExit("ssh jt@fake failed (255)")
    with pytest.raises(SystemExit, match="ssh"):
        md.demote("2.0", "imu-1028", "x", Gone(), with_imu_release)
    assert md.find(md.load_registry(with_imu_release), "kblock", "2.0") is not None


def test_approve_and_attach_on_a_pre_gate_release(tmp_path):
    reg = md.upgrade_registry({"schema": 1, "versions": [
        {"name": "kblock", "version": "1.0", "label": "kblock 1.0"},
        {"name": "kblock", "version": "1.1", "label": "kblock 1.1"}]})
    md.save_registry(reg, str(tmp_path))
    path = evidence(str(tmp_path), candidate="kblock 1.1",
                    cand=metrics(recall=0.5, precision=0.3), base=metrics(precision=0.6))
    entry = md.attach("1.1", path, "won held-out A/B", str(tmp_path))
    assert entry["gate"] == "pre-gate, won held-out A/B"
    assert entry["approval"] == {"status": "pending"}
    assert entry["evidence"]["summary"]["passes"] is False   # the verdict is kept honest
    with pytest.raises(SystemExit, match="evidence for"):
        md.attach("1.0", path, "", str(tmp_path))
    entry = md.approve("1.1", str(tmp_path), notes="drove the loop twice")
    assert entry["approval"]["status"] == "approved"
    with pytest.raises(SystemExit, match="approved on"):
        md.approve("1.1", str(tmp_path))


# ---------------------------------------------------------------------------
# list.
# ---------------------------------------------------------------------------

def test_list_shows_releases_and_candidates_apart_and_marks_the_car(with_imu_release):
    folder = with_imu_release
    reg = md.load_registry(folder)
    one = md.find(reg, "kblock", "1.0")
    md.demote("2.0", "imu-1028", "lost the A/B", CarOn("kblock-1.0.onnx", one["onnx_sha256"]),
              folder)
    reg = md.load_registry(folder)
    active, _, _ = md.active_entry(reg, CarOn("kblock-1.0.onnx", one["onnx_sha256"]))
    lines = md.list_lines(reg, active, folder, offline=False)
    text = "\n".join(lines)
    releases, candidates = lines.index("Releases"), lines.index("Candidates")
    assert releases < candidates
    rel = lines[releases + 1:candidates]
    can = lines[candidates + 1:]
    assert rel[0].startswith("* kblock 1.0 ")
    assert not any("candidate imu-1028" in line for line in rel)
    assert any(line.startswith("  kblock candidate imu-1028") for line in can)
    assert any("demoted from 2.0" in line for line in can)
    assert any("promoted to 1.0" in line for line in can)    # the first one
    assert "gate first release; on-car approved" in text
    assert text.endswith("* = what the car's models/policy.onnx points at")


def test_the_car_running_a_promoted_candidate_is_named_by_its_file(with_candidate):
    folder = with_candidate
    path = evidence(folder, cand=metrics(recall=0.9))
    release = md.promote("next", folder, evidence=path, approved_on_car=True)
    reg = md.load_registry(folder)
    sha = release["onnx_sha256"]
    assert md.active_entry(reg, CarOn("kblock-1.1.onnx", sha))[0]["label"] == "kblock 1.1"
    assert md.active_entry(reg, CarOn("candidate-next.onnx", sha))[0]["label"] == \
        "kblock candidate next"


# ---------------------------------------------------------------------------
# The trainer's half.
# ---------------------------------------------------------------------------

def test_the_trainer_writes_where_a_model_came_from():
    meta = pt.provenance_meta(["data.npz", "out.onnx", "--imu"], "kblock",
                              commit=lambda: ("abc123", False))
    assert meta["train_commit"] == "abc123" and meta["train_commit_dirty"] is False
    assert meta["train_command"] == "tools/policy_train.py data.npz out.onnx --imu"
    assert meta["model_name"] == "kblock" and "model_version" not in meta
    assert "model_name" not in pt.provenance_meta([], commit=lambda: (None, None))


def test_the_trainer_cannot_stamp_a_release_number(capsys):
    assert pt.main(["data.npz", "out.onnx", "--model-version", "2.0"]) == 2
    assert "promote" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Per-model drive notes.
# ---------------------------------------------------------------------------

def noted(tmp_path):
    row = {"name": "kblock", "created": "2026-10-01", "onnx_sha256": "ab" * 32,
           "parent": None, "train_sessions": []}
    reg = md.upgrade_registry({"schema": 1, "versions": [
        {**row, "version": "1.0", "label": "kblock 1.0"},
        {**row, "version": "1.1", "label": "kblock 1.1"}]})
    md.save_registry(reg, str(tmp_path))
    return str(tmp_path)


def test_a_note_is_appended_dated_and_kinded(tmp_path):
    import datetime
    folder = noted(tmp_path)
    at = datetime.datetime(2026, 10, 28, 9, 30, tzinfo=datetime.timezone.utc)
    md.note("1.1", "loops on a 3-point turn", folder, now=at)
    md.note("1.1", "good on carpet", folder, kind="general", now=at)
    reg = md.load_registry(folder)
    assert md.find(reg, "kblock", "1.1")["notes_log"] == [
        {"date": "2026-10-28T09:30:00Z", "kind": "drive", "text": "loops on a 3-point turn"},
        {"date": "2026-10-28T09:30:00Z", "kind": "general", "text": "good on carpet"}]
    assert md.find(reg, "kblock", "1.0").get("notes_log") is None


def test_notes_keep_the_order_they_were_added(tmp_path):
    folder = noted(tmp_path)
    for text in ("first", "second", "third"):
        md.note("1.0", text, folder)
    log = md.find(md.load_registry(folder), "kblock", "1.0")["notes_log"]
    assert [n["text"] for n in log] == ["first", "second", "third"]


def test_a_candidate_takes_a_note_and_the_registration_note_stays(tmp_path):
    folder = noted(tmp_path)
    reg = md.load_registry(folder)
    reg["candidates"].append({"name": "kblock", "tag": "side", "label": "kblock candidate side",
                              "registered": "2026-10-28T00:00:00Z", "notes": "reg note"})
    md.save_registry(reg, folder)
    md.note("candidate-side", "wobbles", folder)
    cand = md.find_candidate(md.load_registry(folder), "side")
    assert cand["notes"] == "reg note"
    assert cand["notes_log"][0]["text"] == "wobbles"


def test_the_list_shows_the_count_and_the_newest_drive_note(tmp_path):
    folder = noted(tmp_path)
    md.note("1.1", "old drive note", folder)
    md.note("1.1", "newest drive note " + "x" * 200, folder)
    md.note("1.1", "a general one", folder, kind="general")
    lines = md.list_lines(md.load_registry(folder), None, folder, offline=True)
    shown = [line for line in lines if "3 notes" in line]
    assert len(shown) == 1
    assert "newest drive note" in shown[0] and "old drive note" not in shown[0]
    assert "a general one" not in shown[0]
    assert len(shown[0]) < 140
    assert not any("note" in line for line in lines if "kblock 1.0" in line)


def test_approve_notes_land_in_the_list_as_an_approval(tmp_path):
    folder = noted(tmp_path)
    md.approve("1.1", folder, notes="drove the loop twice")
    log = md.find(md.load_registry(folder), "kblock", "1.1")["notes_log"]
    assert [(n["kind"], n["text"]) for n in log] == [("approval", "drove the loop twice")]
    md.approve("1.0", folder)    # no notes, no entry
    assert md.find(md.load_registry(folder), "kblock", "1.0").get("notes_log") is None


def test_a_note_on_an_unknown_model_is_refused_and_writes_nothing(tmp_path):
    folder = noted(tmp_path)
    before = (tmp_path / "registry.json").read_text()
    for target in ("9.9", "candidate-nope", "banana"):
        with pytest.raises(SystemExit, match="no "):
            md.note(target, "hello", folder)
    with pytest.raises(SystemExit, match="kind"):
        md.note("1.0", "hello", folder, kind="rant")
    with pytest.raises(SystemExit, match="text"):
        md.note("1.0", "  ", folder)
    assert (tmp_path / "registry.json").read_text() == before


def test_the_notes_command_prints_them_all(tmp_path, capsys):
    folder = noted(tmp_path)
    md.note("1.1", "one", folder)
    md.note("1.1", "two", folder, kind="general")
    assert md.main(["--models-dir", folder, "notes", "1.1"]) == 0
    out = capsys.readouterr().out
    assert "[drive]  one" in out and "[general]  two" in out
    assert out.index("one") < out.index("two")
