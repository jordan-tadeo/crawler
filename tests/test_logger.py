"""Logger makes its own log folder.

log/ holds nothing but generated CSVs, so it is gitignored, so it is not in a
fresh clone -- and open(..., 'a') does not create a missing parent. The csv()
call sits behind "a controller is connected and the state has changed", so a
clone that cannot write it comes up fine and then falls over a few seconds in,
on the first state change. With Restart=on-failure that is a restart loop.

Every test here points REPO_DIR at a tmp_path, because _prepare deliberately
anchors to the checkout rather than the current directory and the real thing
would write into the repo.
"""

import os

import pytest

import Logger as lg


def test_prepare_creates_a_missing_parent_directory(tmp_path, monkeypatch):
    """The fix, directly. Without the makedirs this returns a path whose
    parent does not exist and the first open(..., 'a') raises."""
    monkeypatch.setattr(lg, "REPO_DIR", str(tmp_path))

    path = lg._prepare("log/vehicle_state_log.csv")

    assert os.path.isdir(tmp_path / "log")
    assert path == str(tmp_path / "log" / "vehicle_state_log.csv")


def test_the_prepared_path_can_actually_be_opened_for_append(tmp_path, monkeypatch):
    """The assertion that matters -- a directory that exists is only the means."""
    monkeypatch.setattr(lg, "REPO_DIR", str(tmp_path))

    path = lg._prepare("log/deeper/still/state.csv")
    with open(path, "a") as f:
        f.write("ok\n")

    assert os.path.exists(path)


def test_a_path_with_no_parent_is_still_anchored_to_the_checkout(tmp_path, monkeypatch):
    """app.log is the no-parent case, and it must not land in whatever
    directory the process happens to be standing in."""
    monkeypatch.setattr(lg, "REPO_DIR", str(tmp_path))

    assert lg._prepare("app.log") == str(tmp_path / "app.log")


def test_prepare_is_idempotent(tmp_path, monkeypatch):
    """csv() calls it on every write, which on the car is once a second."""
    monkeypatch.setattr(lg, "REPO_DIR", str(tmp_path))

    first = lg._prepare("log/state.csv")
    second = lg._prepare("log/state.csv")

    assert first == second
    assert os.path.isdir(tmp_path / "log")


def test_constructing_a_logger_creates_the_folder_its_file_needs(tmp_path, monkeypatch):
    """The whole point: a fresh clone with no log/ comes up rather than
    raising on the first write."""
    monkeypatch.setattr(lg, "REPO_DIR", str(tmp_path))

    logger = lg.Logger(log_file="log/app.log")
    logger.log("info", "test", "a line")

    assert os.path.exists(tmp_path / "log" / "app.log")


def test_csv_writes_a_header_then_a_row_into_a_folder_that_did_not_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(lg, "REPO_DIR", str(tmp_path))
    logger = lg.Logger.__new__(lg.Logger)

    logger.csv("log/state.csv", "vehicle", {"throttle": 0.2, "front_s": 90})

    lines = (tmp_path / "log" / "state.csv").read_text().strip().splitlines()
    assert lines[0] == "timestamp,subject,throttle,front_s"
    assert lines[1].endswith("vehicle,0.2,90")


def test_csv_appends_in_the_column_order_the_file_already_has(tmp_path, monkeypatch):
    """A file with rows has a column order and its first row says what it is.
    A sample arriving with the keys in another order must not write a second
    shape into the same file."""
    monkeypatch.setattr(lg, "REPO_DIR", str(tmp_path))
    logger = lg.Logger.__new__(lg.Logger)

    logger.csv("log/state.csv", "vehicle", {"throttle": 0.2, "front_s": 90})
    logger.csv("log/state.csv", "vehicle", {"front_s": 95, "throttle": 0.4})

    lines = (tmp_path / "log" / "state.csv").read_text().strip().splitlines()
    assert len(lines) == 3
    assert lines[0] == "timestamp,subject,throttle,front_s"
    assert lines[2].endswith("vehicle,0.4,95")


def test_csv_writes_whatever_keys_the_state_dict_has(tmp_path, monkeypatch):
    """It used to index message['pan'] and message['tilt'] straight, so the
    moment get_state stopped reporting the gimbal this raised a KeyError once
    a second."""
    monkeypatch.setattr(lg, "REPO_DIR", str(tmp_path))
    logger = lg.Logger.__new__(lg.Logger)

    logger.csv("log/state.csv", "vehicle", {"throttle": 0.0, "front_s": 90, "rear_s": 90})

    header = (tmp_path / "log" / "state.csv").read_text().splitlines()[0]
    assert header == "timestamp,subject,throttle,front_s,rear_s"


def test_an_unknown_log_level_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(lg, "REPO_DIR", str(tmp_path))
    logger = lg.Logger(log_file="log/app.log")

    with pytest.raises(ValueError):
        logger.log("critical", "test", "not a level this takes")
