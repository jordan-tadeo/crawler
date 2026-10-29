"""tools/retrain.py: the per-session cache, which recordings train, and which
model it grades against. The decode and the training need torch and mcap, so
these drive the pieces around them with a stand-in decode."""

import json
import os

import numpy as np
import pytest

import models as md
import policy_train as pt
import retrain as rt


def recording(folder, name, size=rt.STUB_BYTES + 10):
    p = folder / f"{name}.mcap"
    p.write_bytes(b"x" * size)
    return str(p)


class FakeDecode:
    """Writes a cache entry the way decode_into_cache does, and counts."""

    def __init__(self):
        self.paths = []

    def __call__(self, path, adapter_arg, pkl, js):
        self.paths.append(os.path.basename(path))
        stats = {"footprint": True, "caps": "ok", "usable": 100, "adapter": "footprint",
                 "commands": 10}
        with open(pkl, "wb") as f:
            f.write(b"session")
        with open(js, "w") as f:
            json.dump(stats, f)
        return stats


def scan(mcap, cache, code="code-a", decode=None):
    cands = [rt.Candidate(p, code, str(cache)) for p in rt.candidates(str(mcap))]
    return cands, rt.ensure_cache(cands, str(cache), jobs=1, decode=decode)


# ---------------------------------------------------------------------------
# The cache key.
# ---------------------------------------------------------------------------

def test_the_key_moves_with_the_code_the_size_the_mtime_and_the_reader(tmp_path):
    p = recording(tmp_path, "crawler-record-a")
    k = rt.cache_key(p, "code-a", False)
    assert rt.cache_key(p, "code-a", False) == k
    assert rt.cache_key(p, "code-b", False) != k
    assert rt.cache_key(p, "code-a", True) != k
    st = os.stat(p)
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
    assert rt.cache_key(p, "code-a", False) != k
    k2 = rt.cache_key(p, "code-a", False)
    with open(p, "ab") as f:
        f.write(b"y")
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
    assert rt.cache_key(p, "code-a", False) != k2


def test_the_code_hash_covers_every_file_the_decode_reaches():
    names = {os.path.basename(f) for f in rt.source_files(rt.pd)}
    assert {"policy_dataset.py", "policy_adapters.py", "policy.py", "policy_imu.py",
            "policy_lidar.py"} <= names
    assert all(f.startswith(rt.REPO + os.sep) for f in rt.source_files(rt.pd))


def test_the_code_hash_moves_with_a_byte(tmp_path):
    a = tmp_path / "a.py"
    a.write_text("x = 1\n")
    h = rt.code_hash([str(a)], str(tmp_path))
    a.write_text("x = 2\n")
    assert rt.code_hash([str(a)], str(tmp_path)) != h


def test_only_a_new_or_changed_recording_decodes_again(tmp_path):
    mcap, cache = tmp_path / "mcap", tmp_path / "cache"
    mcap.mkdir()
    cache.mkdir()
    recording(mcap, "crawler-record-a")
    b = recording(mcap, "crawler-record-b")
    recording(mcap, "crawler-record-stub", size=100)

    first = FakeDecode()
    cands, _ = scan(mcap, cache, decode=first)
    assert sorted(first.paths) == ["crawler-record-a.mcap", "crawler-record-b.mcap"]
    assert all(c.stats for c in cands if c.size >= rt.STUB_BYTES)

    again = FakeDecode()
    scan(mcap, cache, decode=again)
    assert again.paths == []

    recording(mcap, "crawler-record-c")
    st = os.stat(b)
    os.utime(b, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    third = FakeDecode()
    scan(mcap, cache, decode=third)
    assert sorted(third.paths) == ["crawler-record-b.mcap", "crawler-record-c.mcap"]
    # b's entry under its old key is gone, not left to pile up.
    assert len([p for p in os.listdir(cache) if p.startswith("crawler-record-b.")]) == 2

    code = FakeDecode()
    scan(mcap, cache, code="code-b", decode=code)
    assert len(code.paths) == 3


def test_a_half_written_entry_decodes_again(tmp_path):
    mcap, cache = tmp_path / "mcap", tmp_path / "cache"
    mcap.mkdir()
    cache.mkdir()
    recording(mcap, "crawler-record-a")
    cands, _ = scan(mcap, cache, decode=FakeDecode())
    os.remove(cands[0].pkl)
    redo = FakeDecode()
    scan(mcap, cache, decode=redo)
    assert redo.paths == ["crawler-record-a.mcap"]


# ---------------------------------------------------------------------------
# Which recordings train.
# ---------------------------------------------------------------------------

def test_candidates_skip_repair_copies_subfolders_and_other_files(tmp_path):
    recording(tmp_path, "crawler-record-a")
    recording(tmp_path, "crawler-20261029-0101Z-p1-001")
    (tmp_path / "crawler-record-a.mcap.orig").write_bytes(b"x")
    (tmp_path / "other.mcap").write_bytes(b"x")
    (tmp_path / "drives").mkdir()
    recording(tmp_path / "drives", "crawler-20261029-0101Z-p1-001")
    names = [os.path.basename(p) for p in rt.candidates(str(tmp_path))]
    assert names == ["crawler-20261029-0101Z-p1-001.mcap", "crawler-record-a.mcap"]


def test_a_drive_log_is_read_as_footprint_and_a_record_session_as_itself(tmp_path):
    rec = rt.Candidate(recording(tmp_path, "crawler-record-a"), "c", str(tmp_path))
    drive = rt.Candidate(recording(tmp_path, "crawler-20261029-0101Z-p1-001"), "c",
                         str(tmp_path))
    assert rec.adapter_arg is None and not rec.drive
    assert drive.adapter_arg == "footprint" and drive.drive


GOOD = {"footprint": True, "caps": "ok", "usable": 100, "adapter": "footprint",
        "commands": 10}


@pytest.mark.parametrize("size,stats,word", [
    (100, None, "stub"),
    (rt.STUB_BYTES, {**GOOD, "footprint": False, "commands": 0}, "no /footprint"),
    (rt.STUB_BYTES, {**GOOD, "caps": rt.pd.CAPS_EXCLUDED + ": +0.200/-0.600"}, "old caps"),
    (rt.STUB_BYTES, {**GOOD, "usable": rt.MIN_USABLE - 1}, "no labels"),
])
def test_each_junk_rule_says_why(size, stats, word):
    why = rt.junk_reasons(size, stats)
    assert len(why) == 1 and why[0].startswith(word)


def test_a_good_session_is_not_junk():
    assert rt.junk_reasons(rt.STUB_BYTES, {**GOOD, "usable": rt.MIN_USABLE}) == []


def test_an_excluded_run_is_out_in_every_segment_and_says_why(tmp_path):
    """The exclude file: a run's prefix takes every segment of it, a comment
    is a comment, and the rest of a good session's neighbours still train."""
    ex = tmp_path / "exclude.txt"
    ex.write_text("# takeovers I made because the car kept stopping\n\n"
                  "crawler-20261029-174641Z-p1300 car kept stopping, 10-29\n"
                  "crawler-20261029-175158Z-p1251\n")
    rules = rt.pd.read_excludes(str(ex))
    assert rules == [("crawler-20261029-174641Z-p1300", "car kept stopping, 10-29"),
                     ("crawler-20261029-175158Z-p1251", "on the exclude list")]
    mcap = tmp_path / "mcap"
    mcap.mkdir()
    for name in ("crawler-20261029-174641Z-p1300-001", "crawler-20261029-174641Z-p1300-002",
                 "crawler-20261029-175158Z-p1251-001", "crawler-20261029-052302Z-p2120-001",
                 "crawler-record-20261029-035203Z-p1092-001"):
        recording(mcap, name)
    (tmp_path / "cache").mkdir()
    cands, _ = scan(mcap, tmp_path / "cache", decode=FakeDecode())
    kept = rt.apply_filter(cands, rules)
    assert sorted(c.name for c in kept) == ["crawler-20261029-052302Z-p2120-001",
                                            "crawler-record-20261029-035203Z-p1092-001"]
    out = {c.name: c.why for c in cands if c.why}
    assert out["crawler-20261029-174641Z-p1300-002"] == ["excluded: car kept stopping, 10-29"]
    assert rt.apply_filter(cands, rt.pd.read_excludes(str(tmp_path / "none.txt"))) \
        and len(rt.apply_filter(cands, [])) == 5, "no file, nothing excluded"


def test_a_plain_dataset_run_leaves_out_the_excluded_drive_logs(tmp_path, monkeypatch, capsys):
    """policy_dataset.main reads the same file, so a run by hand leaves them
    out too, before a single one is opened."""
    ex = tmp_path / "exclude.txt"
    ex.write_text("crawler-20261029-174641Z-p1300 car kept stopping\n")
    drives = tmp_path / "drives"
    drives.mkdir()
    recording(drives, "crawler-20261029-174641Z-p1300-001")
    recording(drives, "crawler-20261029-052302Z-p2120-001")
    opened = []

    class Empty:
        commands = ()
        name = "empty"

    def read_session(path, adapter=None):
        opened.append(os.path.basename(path))
        return Empty(), False     # a drive with no rows: skipped, and on

    monkeypatch.setattr(rt.pd, "read_session", read_session)
    rec = recording(tmp_path, "crawler-20261029-174641Z-p1300-002")
    try:   # with nothing to keep it may stop or write an empty set; either is fine
        rt.pd.main([str(tmp_path / "out.npz"), rec, "--drive-logs", str(drives),
                    "--exclude-file", str(ex)])
    except (SystemExit, Exception):
        pass
    assert opened == ["crawler-20261029-052302Z-p2120-001.mcap"]
    assert "crawler-20261029-174641Z-p1300-001.mcap: excluded (car kept stopping)" \
        in capsys.readouterr().out


# ---------------------------------------------------------------------------
# What it grades against, and what it refuses.
# ---------------------------------------------------------------------------

def entry(label, sha, notes=()):
    return {"label": label, "onnx": f"{label}.onnx", "onnx_sha256": sha,
            "notes_log": [{"date": d, "text": t} for d, t in notes]}


def test_the_deploy_marker_wins_over_notes():
    a = entry("a", "1" * 64, [("2026-10-29T02:00:00Z", "Deployed for a drive")])
    b = entry("b", "2" * 64)
    reg = {"releases": [], "candidates": [a, b],
           "deployed": {"onnx": "b.onnx", "onnx_sha256": "2" * 64, "date": "x"}}
    assert rt.deployed_entry(reg)[0] is b


def test_without_a_marker_the_newest_deployed_note_says():
    a = entry("a", "1" * 64, [("2026-10-28T01:00:00Z", "Deployed for a drive")])
    b = entry("b", "2" * 64, [("2026-10-29T01:00:00Z", "deployed again"),
                              ("2026-10-30T01:00:00Z", "drove badly")])
    c = entry("c", "3" * 64, [("2026-10-31T01:00:00Z", "not on the car")])
    reg = {"releases": [a], "candidates": [b, c]}
    assert rt.deployed_entry(reg)[0] is b
    assert rt.deployed_entry({"releases": [], "candidates": [c]})[0] is None


def at(text):
    import datetime
    t = datetime.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    return int(t.replace(tzinfo=datetime.timezone.utc).timestamp() * 1e9)


def test_the_model_that_last_drove_beats_an_older_note():
    # The notes say a went on the car; the newest drive log rows say b drove,
    # and a model the registry does not know drove after it.
    a = entry("a", "1" * 64, [("2026-10-29T01:00:00Z", "Deployed for a drive")])
    b = entry("b", "2" * 64)
    reg = {"releases": [], "candidates": [a, b]}
    stats = [None, {"policy_models": {"1" * 64: at("2026-10-29T02:00:00Z")}},
             {"policy_models": {"2" * 64: at("2026-10-29T03:00:00Z"),
                                "9" * 64: at("2026-10-29T04:00:00Z")}}]
    e, how = rt.deployed_entry(reg, stats)
    assert e is b and "2026-10-29T03:00:00Z" in how


def test_a_deploy_after_the_last_drive_beats_it():
    a, b = entry("a", "1" * 64), entry("b", "2" * 64)
    stats = [{"policy_models": {"1" * 64: at("2026-10-29T02:00:00Z")}}]
    later = {"releases": [], "candidates": [a, b],
             "deployed": {"onnx": "b.onnx", "onnx_sha256": "2" * 64,
                          "date": "2026-10-29T05:00:00Z"}}
    assert rt.deployed_entry(later, stats)[0] is b
    earlier = {**later, "deployed": {**later["deployed"], "date": "2026-10-29T01:00:00Z"}}
    assert rt.deployed_entry(earlier, stats)[0] is a


def test_the_flags_it_sets_itself_are_refused_after_the_dashes():
    assert rt.parse_args(["--", "--lidar"])[1] == ["--lidar"]
    for flag in ("--epochs=5", "--blocks", "--fold", "--val"):
        with pytest.raises(SystemExit):
            rt.parse_args(["--", flag, "1"])


def test_pooled_scores_sum_the_confusions_and_weight_the_steer():
    a = {"ticks": 2, "confusion": [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
         "steer_mae": 0.2, "steer_ticks": 1}
    b = {"ticks": 4, "confusion": [[2, 1, 0], [0, 1, 0], [0, 0, 0]],
         "steer_mae": 0.5, "steer_ticks": 3}
    p = rt.pooled([a, b, None])
    assert p["ticks"] == 6
    assert np.array_equal(p["confusion"], [[3, 1, 0], [0, 2, 0], [0, 0, 0]])
    assert p["steer_mae"] == pytest.approx((0.2 + 1.5) / 4)


@pytest.mark.parametrize("fold,blocks", [("final", None), ("0", "5"), ("6", "5"),
                                         ("two", "5")])
def test_policy_train_refuses_a_fold_that_is_not_one(tmp_path, fold, blocks, capsys):
    argv = [str(tmp_path / "none.npz"), str(tmp_path / "out.onnx"), "--fold", fold]
    if blocks:
        argv += ["--blocks", blocks]
    assert pt.main(argv) == 2
    assert "--fold is final or a block" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The recipe, and the head-to-head with the deployed model.
# ---------------------------------------------------------------------------

def last_value(flags, name):
    return [flags[i + 1] for i, f in enumerate(flags) if f == name][-1]


def test_the_recipe_is_20_epochs_at_takeover_weight_1_and_a_flag_after_dashes_wins():
    args, extra = rt.parse_args([])
    flags = rt.recipe_flags(args, extra)
    assert last_value(flags, "--epochs") == "20"
    assert last_value(flags, "--takeover-weight") == "1"
    args, extra = rt.parse_args(["--", "--takeover-weight", "3", "--lidar"])
    flags = rt.recipe_flags(args, extra)
    assert last_value(flags, "--takeover-weight") == "3" and "--lidar" in flags


def test_the_head_to_head_holds_out_every_unseen_session_with_graded_ticks():
    names = ["rec-a", "rec-b", "drive-c", "rec-d"]
    session = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    graded = np.array([1, 1, 1, 0, 0, 0, 0, 1], dtype=bool)  # drive-c: never graded
    assert rt.unseen_sessions(names, session, graded, {"rec-a"}) == ["rec-b", "rec-d"]
    assert rt.unseen_sessions(names, session, graded, {"rec-a", "rec-b", "rec-d"}) == []


def test_trained_on_reads_names_and_dicts():
    assert rt.trained_on(None) == set()
    assert rt.trained_on({"train_sessions": ["a", {"name": "b", "sha256": "x"}]}) == {"a", "b"}
