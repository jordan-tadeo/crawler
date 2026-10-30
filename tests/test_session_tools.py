"""tools/pull_mcaps.py and tools/session_report.py: the parts that decide.

Both tools are mostly I/O -- ssh, rsync, an MCAP reader -- and none of that is
tested here, because none of it is mine. What is tested is the logic that can
be wrong quietly: which file is still being written, which Pi copies are safe
to delete, whether a destination is inside a repo, and how reversals and close
passes are counted. Every input is synthetic.

The tools are scripts in tools/, not modules on the path, so they are loaded
by file the same way test_mcap_check.py loads deploy/mcap_check.py. Neither
imports anything outside the standard library at module scope.
"""

import importlib.util
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    path = os.path.join(REPO, "tools", name + ".py")
    spec = importlib.util.spec_from_file_location(name + "_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pm = _load("pull_mcaps")
sr = _load("session_report")


# ---------------------------------------------------------------------------
# The destination guard
# ---------------------------------------------------------------------------

def test_a_folder_inside_a_repo_is_found(tmp_path):
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    (tmp_path / "repo" / "sub").mkdir()
    assert pm.git_work_tree_of(str(tmp_path / "repo" / "sub")) == \
        os.path.realpath(str(tmp_path / "repo"))


def test_a_destination_that_does_not_exist_yet_is_still_caught(tmp_path):
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    dest = tmp_path / "repo" / "data" / "mcap"
    assert pm.git_work_tree_of(str(dest)) is not None
    assert not dest.exists()


def test_a_linked_worktree_has_a_git_file_not_a_dir(tmp_path):
    (tmp_path / "wt").mkdir()
    (tmp_path / "wt" / ".git").write_text("gitdir: /elsewhere\n")
    assert pm.git_work_tree_of(str(tmp_path / "wt" / "x")) is not None


def test_a_symlink_into_a_repo_is_followed(tmp_path):
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    (tmp_path / "repo" / "rec").mkdir()
    (tmp_path / "out").mkdir()
    os.symlink(tmp_path / "repo" / "rec", tmp_path / "out" / "link")
    assert pm.git_work_tree_of(str(tmp_path / "out" / "link")) is not None


def test_a_folder_outside_any_repo_is_allowed(tmp_path):
    (tmp_path / "data").mkdir()
    assert pm.git_work_tree_of(str(tmp_path / "data")) is None
    assert pm.refuse_reason(str(tmp_path / "data")) is None


def test_this_checkout_is_refused_before_the_car_is_contacted(monkeypatch):
    def no_car(*args, **kwargs):
        raise AssertionError("contacted the car with a bad destination")
    monkeypatch.setattr(pm, "ssh", no_car)
    assert pm.main(["--dest", os.path.join(REPO, "recordings"), "--list"]) == 2


# ---------------------------------------------------------------------------
# What is live on the car
# ---------------------------------------------------------------------------

D = "/home/jt/Documents/py/crawler/log/mcap"
CLOSED = pm.MCAP_MAGIC_HEX
OPEN_TAIL = "0000000000000000"


def listing(state="active", pid=2002, fds=True, opened=(), files=()):
    lines = [f"SERVICE {state} {pid}"]
    if fds:
        lines.append("FDS_READABLE")
        lines += [f"OPEN {D}/{name}" for name in opened]
    lines.append("DF /dev/mmcblk0p2 121000000 12000000 100000000 11% /")
    lines += [f"FILE {size} {mtime} {tail} {D}/{name}"
              for name, size, mtime, tail in files]
    return "\n".join(lines)


def live_names(text):
    files, state, pid, open_paths, _ = pm.parse_listing(text)
    pm.mark_live(files, state, pid, open_paths)
    return {f.name for f in files if f.live}


def test_the_file_the_service_holds_open_is_live():
    text = listing(opened=["crawler-1-p2002-001.mcap"], files=[
        ("crawler-1-p2002-001.mcap", 75, 2, OPEN_TAIL),
        ("crawler-0-p1999-001.mcap", 900, 1, CLOSED)])
    assert live_names(text) == {"crawler-1-p2002-001.mcap"}


def test_a_footerless_file_from_a_dead_process_is_not_live():
    """The battery-out run: no footer, nobody has it open. Pull it."""
    text = listing(opened=[], files=[
        ("crawler-0-p1168-001.mcap", 118542336, 1, OPEN_TAIL)])
    assert live_names(text) == set()


def test_without_the_fd_table_the_running_pids_footerless_file_is_live():
    text = listing(fds=False, files=[
        ("crawler-1-p2002-002.mcap", 75, 2, OPEN_TAIL),
        ("crawler-1-p2002-001.mcap", 9000, 1, CLOSED),
        ("crawler-0-p1168-001.mcap", 9000, 0, OPEN_TAIL)])
    assert live_names(text) == {"crawler-1-p2002-002.mcap"}


def test_a_stopped_service_has_nothing_live():
    # A MainPID can outlive "active" while the unit is stopping; the file it
    # left is a finished run all the same.
    text = listing(state="deactivating", pid=2002, fds=False, files=[
        ("crawler-1-p2002-001.mcap", 75, 2, OPEN_TAIL)])
    assert live_names(text) == set()


def test_the_pid_comes_out_of_a_record_session_name_too():
    assert pm.pid_of("crawler-record-20261027-220000Z-p2002-001.mcap") == 2002
    assert pm.pid_of("crawler-20261027-200959Z-p1143-003.mcap") == 1143


# ---------------------------------------------------------------------------
# Prune safety
# ---------------------------------------------------------------------------

def rf(name, live=False):
    f = pm.RemoteFile(f"{D}/{name}", 10, 0, True)
    f.live = live
    return f


def test_prune_deletes_only_a_hash_match():
    files = [rf("a.mcap"), rf("b.mcap"), rf("c.mcap"), rf("d.mcap"),
             rf("e.mcap", live=True)]
    remote = {f"{D}/a.mcap": "aa", f"{D}/b.mcap": "bb", f"{D}/c.mcap": "cc",
              f"{D}/e.mcap": "ee"}
    local = {"a.mcap": ["aa"], "b.mcap": ["zz"], "e.mcap": ["ee"],
             "d.mcap": ["dd"]}
    delete, keep = pm.prune_plan(files, remote, local)
    assert [f.name for f in delete] == ["a.mcap"]
    reasons = {f.name: why for f, why in keep}
    assert reasons == {
        "b.mcap": "the copy here does not match",
        "c.mcap": "no copy on this Mac",
        "d.mcap": "could not hash it on the car",
        "e.mcap": "still being written",
    }


def test_the_orig_beside_a_repaired_copy_counts():
    """mcap_check rewrites the copy and keeps the original as .orig, so the
    repaired file no longer hashes like the car's -- the .orig does."""
    delete, _ = pm.prune_plan([rf("a.mcap")], {f"{D}/a.mcap": "aa"},
                              {"a.mcap": ["repaired", "aa"]})
    assert [f.name for f in delete] == ["a.mcap"]


def _fake_car(monkeypatch, tmp_path, contents):
    """A car with these files on it, and a Mac folder with matching copies."""
    sent = []
    import hashlib

    def ssh(car, command, check=True):
        sent.append(command)
        return ""

    monkeypatch.setattr(pm, "ssh", ssh)
    monkeypatch.setattr(pm, "sha256_remote", lambda car, paths: {
        p: hashlib.sha256(contents[os.path.basename(p)]).hexdigest()
        for p in paths})
    for name, data in contents.items():
        (tmp_path / name).write_bytes(data)
    return sent


def test_prune_without_yes_sends_no_rm(monkeypatch, tmp_path):
    sent = _fake_car(monkeypatch, tmp_path, {"a.mcap": b"one"})
    pm.prune("car", [rf("a.mcap")], str(tmp_path), yes=False)
    assert not any("rm" in c for c in sent)


def test_prune_with_yes_removes_exactly_the_verified_ones(monkeypatch, tmp_path):
    sent = _fake_car(monkeypatch, tmp_path, {"a.mcap": b"one",
                                             "b.mcap": b"two"})
    (tmp_path / "b.mcap").write_bytes(b"corrupted on the way")
    pm.prune("car", [rf("a.mcap"), rf("b.mcap"), rf("c.mcap", live=True)],
             str(tmp_path), yes=True)
    rms = [c for c in sent if c.startswith("rm ")]
    assert rms == [f"rm -- {D}/a.mcap"]


def test_yes_without_prune_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "ssh", lambda *a, **k: pytest.fail("ssh"))
    assert pm.main(["--dest", str(tmp_path), "--yes"]) == 2


def test_a_copy_at_the_same_size_is_already_here(tmp_path):
    f = pm.RemoteFile(f"{D}/a.mcap", 3, 0, True)
    assert not pm.already_have(f, str(tmp_path))
    (tmp_path / "a.mcap.orig").write_bytes(b"abc")
    assert pm.already_have(f, str(tmp_path))
    f.size = 4
    assert not pm.already_have(f, str(tmp_path))


def test_the_listing_calls_a_footerless_file_unclosed(capsys):
    """No footer means the writer never closed it, not a missing topic."""
    live = pm.RemoteFile(f"{D}/live.mcap", 3, 0, False)
    live.live = True
    files = [live, pm.RemoteFile(f"{D}/cut.mcap", 3, 0, False),
             pm.RemoteFile(f"{D}/done.mcap", 3, 0, True)]
    pm.show(files, "active", 1, set())
    out = capsys.readouterr().out.splitlines()
    rows = {n: next(line for line in out if n in line)
            for n in ("live.mcap", "cut.mcap", "done.mcap")}
    assert rows["cut.mcap"].split()[0] == "unclosed"
    assert rows["live.mcap"].split()[0] == "LIVE"
    assert rows["done.mcap"].split()[0] != "unclosed"
    assert "nofoot" not in "".join(rows.values())
    # the size column still lines up
    assert len({line.index("MiB") for line in rows.values()}) == 1



# ---------------------------------------------------------------------------
# --records-only
# ---------------------------------------------------------------------------

REC = ["crawler-record-20261027T1900Z-p900-001.mcap",
       "crawler-record-20261027T1800Z-p800-001.mcap",
       "crawler-record-20261027T1700Z-p700-001.mcap"]
LOG = ["crawler-20261027T1930Z-p2002-004.mcap",
       "crawler-20261027T1830Z-p2002-003.mcap"]


def mixed():
    """Newest first, the drive logs interleaved with the sessions."""
    return [rf(n) for n in (LOG[0], REC[0], LOG[1], REC[1], REC[2])]


def test_records_only_keeps_just_the_record_sessions():
    assert [f.name for f in pm.select(mixed(), records_only=True)] == REC
    assert [f.name for f in pm.select(mixed())] == [
        f.name for f in mixed()]


def test_count_is_taken_after_the_filter():
    """-n 2 with the filter is the newest two sessions, not whatever sessions
    happen to be among the newest two files."""
    assert [f.name for f in pm.select(mixed(), records_only=True,
                                      count=2)] == REC[:2]


def _main_against(monkeypatch, tmp_path, argv):
    """main() with the car faked: what got listed, pulled and pruned."""
    seen = {}
    monkeypatch.setattr(pm, "survey", lambda car, dirs: (
        mixed(), "active", 2002, set(), []))
    monkeypatch.setattr(pm, "show", lambda files, *a, **k: seen.setdefault(
        "shown", [f.name for f in files]))

    def pull(car, files, dest, pulled_list=None):
        seen["pulled"] = [f.name for f in files]
        return files, []
    monkeypatch.setattr(pm, "pull", pull)

    def prune(car, files, dest, yes):
        seen["pruned"] = [f.name for f in files]
    monkeypatch.setattr(pm, "prune", prune)
    assert pm.main(["--dest", str(tmp_path)] + argv) == 0
    return seen


def test_list_shows_what_the_filter_selects(monkeypatch, tmp_path):
    seen = _main_against(monkeypatch, tmp_path, ["--records-only", "--list"])
    assert seen["shown"] == REC
    assert "pulled" not in seen


def test_records_only_pulls_and_prunes_no_drive_log(monkeypatch, tmp_path):
    seen = _main_against(monkeypatch, tmp_path,
                         ["--records-only", "-n", "2", "--prune"])
    assert seen["pulled"] == REC[:2]
    assert seen["pruned"] == REC[:2]


def test_without_the_flag_everything_is_pulled(monkeypatch, tmp_path):
    seen = _main_against(monkeypatch, tmp_path, [])
    assert seen["pulled"] == [f.name for f in mixed()]


def test_a_filtered_out_drive_log_is_never_hashed_or_removed(monkeypatch,
                                                             tmp_path):
    """The real prune, given what --records-only kept, with a drive log on
    the Mac that hashes the same as the car's: it still never comes up."""
    contents = {n: n.encode() for n in REC + LOG}
    sent = _fake_car(monkeypatch, tmp_path, contents)
    hashed = []
    real = pm.sha256_remote
    monkeypatch.setattr(pm, "sha256_remote", lambda car, paths: (
        hashed.extend(paths), real(car, paths))[1])
    pm.prune("car", pm.select(mixed(), records_only=True), str(tmp_path),
             yes=True)
    assert not any(os.path.basename(p) in LOG for p in hashed)
    rms = [c for c in sent if c.startswith("rm ")]
    assert len(rms) == 1
    assert not any(n in rms[0] for n in LOG)
    assert all(n in rms[0] for n in REC)

# ---------------------------------------------------------------------------
# Counting reversals
# ---------------------------------------------------------------------------

def ticks(throttles, mode="manual", hz=20.0, steer=0.0):
    return [sr.Tick(i / hz, mode, th, steer, -steer, "unified")
            for i, th in enumerate(throttles)]


def test_one_reversal_with_its_duration_and_throttle():
    t = ticks([0.0] * 10 + [-0.5] * 20 + [0.0] * 10)
    events = sr.reverse_events(t)
    assert len(events) == 1
    assert events[0].duration == pytest.approx(1.0)
    assert events[0].mean_throttle == pytest.approx(-0.5)
    assert events[0].mode == "manual"


def test_a_one_tick_dip_through_stop_is_still_one_reversal():
    t = ticks([0.0] * 5 + [-0.6] * 10 + [0.0] + [-0.6] * 10 + [0.0] * 5)
    assert len(sr.reverse_events(t)) == 1


def test_two_reversals_a_second_apart_are_two():
    t = ticks([-0.6] * 10 + [0.0] * 20 + [-0.6] * 10 + [0.0])
    assert len(sr.reverse_events(t)) == 2


def test_a_twitch_is_not_a_reversal():
    t = ticks([0.0] * 5 + [-0.6] * 2 + [0.0] * 5)  # 0.1 s
    assert sr.reverse_events(t) == []


def test_a_resting_finger_is_not_a_reversal():
    t = ticks([-0.04] * 40)
    assert sr.reverse_events(t) == []
    assert sr.throttle_counts(t) == (0, 0, 40)


def test_a_file_that_ends_mid_reverse_still_counts_it():
    t = ticks([0.0] * 5 + [-0.6] * 10)
    events = sr.reverse_events(t)
    assert len(events) == 1 and events[0].peak_throttle == -0.6


def test_histogram_edges_and_overflow():
    edges = [-1.0, 0.0, 1.0]
    assert sr.histogram([-1.0, -0.5, 0.0, 0.5, 1.0], edges) == [2, 3]
    assert sr.histogram([-5.0, 5.0], edges) == [1, 1]


# ---------------------------------------------------------------------------
# Near obstacles and depth
# ---------------------------------------------------------------------------

def test_a_reversal_after_a_close_approach_is_near():
    t = ticks([0.3] * 40 + [-0.6] * 20 + [0.0] * 10, mode="record")
    start = sr.reverse_events(t)[0].start  # 2.0 s
    depth = sr.DepthSeries([(start - 0.5, 420.0), (start + 0.5, 900.0)])
    assert sr.near_reversals(sr.reverse_events(t), depth) == (1, 1)
    far = sr.DepthSeries([(start - 0.5, 800.0)])
    assert sr.near_reversals(sr.reverse_events(t), far) == (0, 1)


def test_a_reading_from_long_before_the_reversal_does_not_count():
    t = ticks([0.3] * 80 + [-0.6] * 20 + [0.0], mode="record")
    events = sr.reverse_events(t)
    depth = sr.DepthSeries([(events[0].start - 3.0, 200.0)])
    assert sr.near_reversals(events, depth) == (0, 0)


def test_close_time_counts_camera_modes_and_fresh_depth_only():
    t = (ticks([0.2] * 20, mode="record")
         + [tk._replace(t=tk.t + 1.0) for tk in ticks([0.2] * 20,
                                                     mode="manual")])
    depth = sr.DepthSeries([(i / 20.0, 250.0) for i in range(5)])
    out = sr.depth_tick_stats(t, depth)
    # 20 record ticks; readings stop at t=0.20, so the five ticks from 0.75
    # on are more than 0.5 s past it and stale. Manual ticks never count.
    assert out["depth_s"] == pytest.approx(1.0)
    assert out["stale_ticks"] == 5
    assert out["close_s"][sr.NEAR_MM] == pytest.approx(0.75)
    assert out["close_s"][sr.VERY_NEAR_MM] == pytest.approx(0.75)


def test_the_footprint_adapter_reads_infinite_probes_as_nothing_near():
    row = {"mode": "record", "throttle": -0.6, "steering_front": 0.1,
           "steering_rear": -0.1, "steering_mode": "unified", "seq": 7,
           "probe_distance_mm": [-1, 812.0, 455.0, -1]}
    out = sr.footprint_row(row)
    assert out["nearest_mm"] == 455.0 and out["seq"] == 7
    assert sr.footprint_row({"probe_distance_mm": [-1, -1]})["nearest_mm"] is None


def test_the_image_bytes_come_out_of_a_compressed_image():
    # timestamp (1), frame_id (4), data (2) with a 200-byte payload so the
    # length needs a two-byte varint, format (3).
    payload = b"\xff\xd8" + bytes(198)
    msg = (b"\x0a\x04\x08\x01\x10\x02"
           + b"\x22\x04oakd"
           + b"\x12" + bytes([0xC8, 0x01]) + payload
           + b"\x1a\x04jpeg")
    assert sr._proto_fields(msg)[2] == payload


# ---------------------------------------------------------------------------
# Gaps
# ---------------------------------------------------------------------------

def test_a_lost_pad_is_a_hole_in_the_ticks():
    times = [i / 20.0 for i in range(100)] + [10.0 + i / 20.0 for i in range(20)]
    holes = sr.tick_holes(times, span_end=30.0)
    assert len(holes) == 2
    assert holes[0][0] == pytest.approx(4.95)
    assert holes[1][1] == pytest.approx(30.0 - 10.95)


def test_disconnects_from_diagnostics_rows():
    rows = [(0.0, True), (1.0, False), (3.0, False), (4.0, True), (5.0, False)]
    drops, down = sr.disconnects(rows)
    assert drops == 2
    assert down == pytest.approx(3.0)


def test_a_depth_dropout_only_counts_while_the_camera_is_ready():
    times = [0.0, 0.1, 0.2, 2.0, 2.1]
    assert sr.dropouts(times, [(0.0, 2.1)]) == [(0.2, pytest.approx(1.8))]
    assert sr.dropouts(times, []) == []
    # A stream that dies and never comes back is the worst dropout of all.
    assert sr.dropouts([0.0, 0.1], [(0.0, 3.0)]) == [(0.1, pytest.approx(2.9))]


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def stats(**kw):
    base = {"depth_s": 600.0, "has_depth": True, "near_reversals": 5,
            "close_s": {sr.NEAR_MM: 120.0}, "moving_ticks": 100,
            "straight_fraction": 0.5, "mode_s": {"record": 600.0}}
    base.update(kw)
    return base


def test_verdicts():
    assert sr.verdict(stats()) == "good recovery coverage"
    assert sr.verdict(stats(close_s={sr.NEAR_MM: 30.0})).startswith(
        "mostly open-floor")
    assert "back out of more" in sr.verdict(stats(near_reversals=1))
    assert sr.verdict(stats(depth_s=0.0)).startswith("no auto or record")
    assert sr.verdict(stats(depth_s=30.0)).startswith("too short")
    assert "steer more" in sr.verdict(stats(straight_fraction=0.9))
    assert "autopilot" in sr.verdict(stats(mode_s={"auto": 600.0}))
