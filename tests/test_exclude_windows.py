"""A stretch of a drive left out of training, not the whole drive.

The 2026-10-29 evening drive on goal3 (crawler-20261030-003550Z-p1201) has a
tight-spot takeover worth learning from at 00:40:19Z, and then a battery at
the ESC's cutoff from 00:41:03Z, where every takeover was forced by the motor
cutting out. A rule on the run's prefix would lose the first to keep out the
second. So a rule can carry a window, prefix@FROM..TO, either end open, and
only the ticks inside it go.
"""

import datetime
from typing import NamedTuple

import pytest

import policy_dataset as pd
import retrain as rt
from test_retrain import FakeDecode, recording, scan


class T(NamedTuple):
    t_ns: int


def ns(stamp):
    return int(datetime.datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp() * 1e9)


RUN = "crawler-20261030-003550Z-p1201"


def rules_of(tmp_path, text):
    ex = tmp_path / "exclude.txt"
    ex.write_text(text)
    return pd.read_excludes(str(ex))


def test_a_window_rule_never_takes_the_whole_session(tmp_path):
    rules = rules_of(tmp_path, f"{RUN}@2026-10-30T00:41:03Z battery at the cutoff\n"
                               "crawler-20261029-174641Z-p1300 car kept stopping\n")
    assert pd.excluded(f"/x/{RUN}-001.mcap", rules) is None
    assert pd.excluded("/x/crawler-20261029-174641Z-p1300-001.mcap", rules) == "car kept stopping"


def test_only_the_ticks_inside_the_window_go(tmp_path):
    rules = rules_of(tmp_path, f"{RUN}@2026-10-30T00:41:03Z..2026-10-30T00:41:10Z battery\n")
    ticks = [T(ns("2026-10-30T00:41:02.9Z")), T(ns("2026-10-30T00:41:03Z")),
             T(ns("2026-10-30T00:41:09.9Z")), T(ns("2026-10-30T00:41:10Z"))]
    kept, cut = pd.drop_excluded_windows(f"{RUN}-001", ticks, rules)
    assert kept == [ticks[0], ticks[3]]
    assert cut == {"battery": 2}


def test_an_open_end_runs_to_the_end_of_the_session(tmp_path):
    rules = rules_of(tmp_path, f"{RUN}@2026-10-30T00:41:03Z battery\n"
                               f"{RUN}@..2026-10-30T00:39:00Z setting up\n")
    ticks = [T(ns("2026-10-30T00:38:59Z")), T(ns("2026-10-30T00:40:19Z")),
             T(ns("2026-10-30T09:00:00Z"))]
    kept, cut = pd.drop_excluded_windows(f"{RUN}-001", ticks, rules)
    assert kept == [ticks[1]]
    assert cut == {"battery": 1, "setting up": 1}


def test_a_window_on_another_run_cuts_nothing(tmp_path):
    rules = rules_of(tmp_path, f"{RUN}@2026-10-30T00:41:03Z battery\n")
    ticks = [T(ns("2026-10-30T00:42:00Z"))]
    assert pd.drop_excluded_windows("crawler-20261029-221228Z-p1974-001", ticks, rules) \
        == (ticks, {})


def test_a_window_that_does_not_parse_is_refused(tmp_path):
    with pytest.raises(ValueError):
        rules_of(tmp_path, f"{RUN}@00:41:03 battery\n")


def test_retrain_keeps_a_session_with_only_a_window_on_it(tmp_path):
    rules = rules_of(tmp_path, f"{RUN}@2026-10-30T00:41:03Z battery\n")
    mcap = tmp_path / "mcap"
    mcap.mkdir()
    recording(mcap, f"{RUN}-001")
    (tmp_path / "cache").mkdir()
    cands, _ = scan(mcap, tmp_path / "cache", decode=FakeDecode())
    assert [c.name for c in rt.apply_filter(cands, rules)] == [f"{RUN}-001"]
