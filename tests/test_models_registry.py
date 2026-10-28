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
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "tools"))

import models as md  # noqa: E402
import policy_train as pt  # noqa: E402


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
        reg["versions"].append({"name": "kblock", "version": v, "contract": {}})
    assert [e["version"] for e in md.versions_of(reg, "kblock")] == ["1.2", "1.9", "1.10"]


def registry_with(*versions, meta=None):
    reg = md.empty_registry()
    for v in versions:
        reg["versions"].append({"name": "kblock", "version": v,
                                "contract": md.contract_of(meta or sidecar())})
    return reg


def test_a_retrain_on_the_same_contract_is_a_minor():
    v, why = md.choose_version(registry_with("1.0", "1.1"), "kblock", sidecar())
    assert v == "1.2" and "same contract" in why


def test_a_new_input_is_a_major():
    imu = sidecar(uses_imu=True, imu_features=["gx"], imu_window=5, imu_tilt=True)
    assert md.choose_version(registry_with("1.0", "1.1"), "kblock", imu)[0] == "2.0"
    prev = sidecar(uses_prev_action=True)
    v, why = md.choose_version(registry_with("1.1"), "kblock", prev)
    assert v == "2.0" and "uses_prev_action" in why


def test_the_first_version_is_1_0():
    assert md.choose_version(md.empty_registry(), "kblock", sidecar())[0] == "1.0"


def test_an_explicit_version_or_the_sidecars_own_wins():
    reg = registry_with("1.0", "1.1")
    imu = sidecar(uses_imu=True, imu_features=["gx"], imu_window=5)
    assert md.choose_version(reg, "kblock", imu, version="2.0")[0] == "2.0"
    assert md.choose_version(reg, "kblock", sidecar(model_version="1.5"))[0] == "1.5"
    assert md.choose_version(reg, "kblock", sidecar(), part="major")[0] == "2.0"


def test_a_taken_or_backwards_version_is_refused():
    reg = registry_with("1.0", "1.1")
    with pytest.raises(ValueError, match="already registered"):
        md.choose_version(reg, "kblock", sidecar(), version="1.1")
    with pytest.raises(ValueError, match="sorts before"):
        md.choose_version(registry_with("1.0", "1.2"), "kblock", sidecar(), version="1.1")


def test_a_contract_change_under_the_same_major_is_refused_unless_forced():
    imu = sidecar(uses_imu=True, imu_features=["gx"], imu_window=5)
    reg = registry_with("1.0", "1.1")
    with pytest.raises(ValueError, match="MAJOR"):
        md.choose_version(reg, "kblock", imu, part="minor")
    assert md.choose_version(reg, "kblock", imu, part="minor", force=True)[0] == "1.2"


def test_imu_details_do_not_count_without_the_imu_input():
    assert md.contract_of(sidecar(imu_tilt=True)) == md.contract_of(sidecar())


# ---------------------------------------------------------------------------
# The registry on disk.
# ---------------------------------------------------------------------------

def test_register_round_trip(tmp_path):
    models = tmp_path / "models"
    mcaps = tmp_path / "mcap"
    models.mkdir()
    mcaps.mkdir()
    (mcaps / "crawler-record-a.mcap").write_bytes(b"session a")
    first = write_model(models, "policy-20261027-231046Z", b"onnx one",
                        sidecar(train_sessions=["crawler-record-a"]))
    second = write_model(models, "policy-20261028-170131Z", b"onnx two", sidecar())
    originals = {p: p.read_bytes() for p in models.iterdir()}

    e1 = md.register(str(first), str(models), str(mcaps), notes="first model")
    e2 = md.register(str(second), str(models), str(mcaps))
    assert (e1["label"], e2["label"]) == ("kblock 1.0", "kblock 1.1")

    # The originals are exactly where and what they were.
    for path, data in originals.items():
        assert path.read_bytes() == data

    reg = strict_json(models / "registry.json")   # no NaN made it in
    assert reg == md.load_registry(str(models))
    v10, v11 = reg["versions"]
    assert v10["version"] == "1.0" and v10["parent"] is None and v10["notes"] == "first model"
    assert v11["version"] == "1.1" and v11["parent"] == "1.0"
    assert v10["created"] == "2026-10-27T23:10:46Z"
    assert v10["onnx_sha256"] == md.pm.sha256_local(str(first))
    assert (models / "kblock-1.0.onnx").read_bytes() == b"onnx one"
    assert v10["train_sessions"] == [{"name": "crawler-record-a",
                                      "mcap_sha256": md.pm.sha256_local(
                                          str(mcaps / "crawler-record-a.mcap"))}]
    assert v11["train_sessions"][1] == {"name": "crawler-record-b", "mcap_sha256": None}
    assert v10["train_commit"] is None
    assert v10["scores"]["cv"]["mean"]["direction_accuracy"] == 0.95
    assert v10["scores"]["cv"]["mean"]["mode_switch_recall"] is None
    assert "folds" not in v10["scores"]["cv"]

    copy = json.loads((models / "kblock-1.1.json").read_text())
    assert (copy["model_name"], copy["model_version"]) == ("kblock", "1.1")
    assert math.isnan(copy["cv"]["mean"]["mode_switch_recall"])  # the copy is the sidecar
    assert v11["json_sha256"] == md.pm.sha256_local(str(models / "kblock-1.1.json"))

    with pytest.raises(SystemExit, match="already kblock 1.1"):
        md.register(str(second), str(models), str(mcaps))
    assert len(md.load_registry(str(models))["versions"]) == 2


def test_register_never_overwrites_a_file(tmp_path):
    src = write_model(tmp_path, "policy-x", b"bytes", sidecar())
    (tmp_path / "kblock-1.0.onnx").write_bytes(b"something else")
    with pytest.raises(SystemExit, match="already exists"):
        md.register(str(src), str(tmp_path), str(tmp_path))
    assert (tmp_path / "kblock-1.0.onnx").read_bytes() == b"something else"
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
    src = write_model(tmp_path, "policy-20261028-170131Z", b"the model", sidecar())
    md.register(str(src), str(tmp_path), str(tmp_path))
    reg = md.load_registry(str(tmp_path))
    return reg, reg["versions"][0], str(tmp_path)


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


def test_a_car_that_moves_before_the_swap_is_left_alone(registered):
    rc, lines = deploy(registered, FakeCar(swap_pulse="1650"))
    assert rc == 3 and any("stopped being parked" in line for line in lines)


@pytest.mark.parametrize("bad", [{"tracebacks": "2"}, {"nrestarts": "1"},
                                 {"pulse": "1600"}, {"active": "failed"},
                                 {"listening": "0"}])
def test_an_unhealthy_restart_says_so(registered, bad):
    rc, lines = deploy(registered, FakeCar(health=bad), wait_s=0.0)
    assert rc == 1 and "UNHEALTHY after the restart:" in lines


# ---------------------------------------------------------------------------
# The trainer's half.
# ---------------------------------------------------------------------------

def test_the_trainer_writes_where_a_model_came_from():
    meta = pt.provenance_meta(["data.npz", "out.onnx", "--imu"], "kblock", "2.0",
                              commit=lambda: ("abc123", False))
    assert meta["train_commit"] == "abc123" and meta["train_commit_dirty"] is False
    assert meta["train_command"] == "tools/policy_train.py data.npz out.onnx --imu"
    assert (meta["model_name"], meta["model_version"]) == ("kblock", "2.0")
    assert "model_name" not in pt.provenance_meta([], commit=lambda: (None, None))
