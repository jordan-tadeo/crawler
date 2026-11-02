"""The recordings' tags: set on the map page, kept across a restart, written
into every file, and split on by the trainer, the scorecard and the list.

Nothing here touches the wheels: a tag is a label on the data. These hold
that the command goes through the patrol's queue and acks like any other
map page command, that a change lands on /session/tags the pass it is made
and the file's metadata as it opens, that a tick takes the newest tags at
or before it (an untagged file is messy), and that the filters and
weights do what their flags say. No torch, no MCAP reader.
"""

import json
import types

import numpy as np
import pytest

from crawler import __main__ as cm
from crawler.autopilot import patrol as pt
from crawler.telemetry import recording
from crawler.telemetry import tags as tg
from crawler.telemetry import telemetry as tm
import policy_adapters as pa
import policy_dataset as pd
import policy_eval as pe
import policy_scorecard as sc
import policy_tags as ptg
import policy_train as ptr
import retrain as rt
import session_tags as st

from test_policy_tools import S, T0, footprint_session, record
from test_recording import FakeLogger, fake_sdk, make  # noqa: F401  (fake_sdk is a fixture)

CLEAN = tg.BITS["clean"]
# The bit kept free for the outdoor tag, on its own branch: a mask with it
# set means nothing here.
SPARE = 1


class Said:
    def __init__(self):
        self.lines = []

    def __call__(self, level, subject, text):
        self.lines.append((level, subject, text))


def store(tmp_path):
    return pt.Store(str(tmp_path / "session_tags.json"), background=False)


# ---- the car's side -------------------------------------------------------------

def test_clean_is_the_one_tag_and_keeps_its_bit():
    assert tg.TAGS == ("clean",) and tg.NAMES == ("clean", "messy")
    assert tg.BITS == {"clean": 2}, "the bit is the datasets' wire format"


def test_none_set_is_messy_and_said_at_boot(tmp_path):
    said = Said()
    tags = tg.SessionTags(store=store(tmp_path), say=said)
    assert tags.bits == 0 and tags.state() == {"clean": False}
    assert said.lines == [("info", "Tags", "Tags: messy (none set yet)")]
    assert tags.payload() == {"clean": False, "tags": "", "seq": 0}
    assert tags.metadata() == {"clean": "false", "tags": ""}


def test_a_tag_set_is_said_kept_and_back_after_a_restart(tmp_path):
    said = Said()
    tags = tg.SessionTags(store=store(tmp_path), say=said)
    assert tags.command({"cmd": "tags", "clean": True}) is True
    assert tags.bits == CLEAN and tags.pop_changed() and not tags.pop_changed()
    assert said.lines[-1] == ("info", "Tags", "Tags now clean")
    assert json.load(open(tmp_path / "session_tags.json")) == {"clean": True}
    assert tags.payload() == {"clean": True, "tags": "clean", "seq": 1}
    assert tags.metadata()["tags"] == "clean"
    # The same again is taken and changes nothing: no new line, no resend.
    n = len(said.lines)
    assert tags.command({"cmd": "tags", "clean": True}) is True
    assert len(said.lines) == n and not tags.pop_changed() and tags.seq == 1
    # Next boot.
    again = Said()
    back = tg.SessionTags(store=store(tmp_path), say=again)
    assert back.bits == CLEAN
    assert again.lines == [("info", "Tags", "Tags: clean (as last set)")]
    # Off again, and kept off.
    assert back.command({"cmd": "tags", "clean": False}) and back.bits == 0
    assert json.load(open(tmp_path / "session_tags.json")) == {"clean": False}


def test_a_bad_tags_command_is_refused_and_changes_nothing(tmp_path):
    said = Said()
    tags = tg.SessionTags(store=store(tmp_path), say=said)
    assert tags.command({"cmd": "tags"}) is False
    assert tags.command({"cmd": "tags", "clean": "yes"}) is False
    assert tags.command({"cmd": "tags", "clean": 1}) is False
    # outdoor is no tag here: a command with only it names none.
    assert tags.command({"cmd": "tags", "outdoor": True}) is False
    assert tags.command("tags") is False
    assert tags.bits == 0 and not tags.changed
    assert [line[0] for line in said.lines[1:]] == ["warning"] * 4
    assert not (tmp_path / "session_tags.json").exists()


def test_bits_read_a_message_a_metadata_record_or_nothing():
    assert tg.bits_of({"clean": True}) == CLEAN
    assert tg.bits_of({"outdoor": "true", "clean": "true", "tags": "outdoor,clean"}) == CLEAN
    assert tg.bits_of({"outdoor": True}) == 0
    assert tg.bits_of({"clean": 1}) == 0 and tg.bits_of(None) == 0 and tg.bits_of([]) == 0
    assert tg.said(0) == "messy" and tg.said(CLEAN) == "clean"
    assert tg.names_of(CLEAN) == ["clean"] and tg.names_of(SPARE) == []


def test_the_patrol_takes_the_map_pages_tags_command_and_acks_it(tmp_path):
    from test_patrol import Events, cmd
    said = Said()
    tags = tg.SessionTags(store=store(tmp_path), say=said)
    p = pt.Patrol(say=said, tags=tags)
    assert "tags" in pt.MAP_COMMANDS and p.page_state()["tags"] == {"clean": False}
    p.poll(Events(cmd("tags", clean=True, id="a-1")), None)
    assert tags.bits == CLEAN and p.page_state()["tags"]["clean"] is True
    assert list(p.acks)[-1] == ["a-1", True, ""]
    # Its line goes to the journal under Tags, and to the page's toasts.
    assert ("info", "Tags", "Tags now clean") in said.lines
    assert p.page_state()["said"][-1] == ["info", "Tags now clean"]
    p.poll(Events(cmd("tags", clean="no", id="a-2")), None)
    ack = list(p.acks)[-1]
    assert ack[:2] == ["a-2", False] and "clean is 'no'" in ack[2]
    # A tag never changes what the patrol drives.
    assert p.use == "goal" and not p.running


def test_a_patrol_with_no_tags_refuses_the_command():
    from test_patrol import Events, cmd
    p = pt.Patrol()
    p.poll(Events(cmd("tags", clean=True, id="b-1")), None)
    assert list(p.acks)[-1] == ["b-1", False, "Tags refused: this car keeps no tags"]
    assert p.page_state()["tags"] is None


class Tele:
    def __init__(self, wanted=()):
        self.wanted = set(wanted)
        self.offers, self.now_offers = [], []

    def wants(self, key, now):
        return key in self.wanted

    def offer(self, key, payload, now):
        self.offers.append((key, payload))

    def offer_now(self, key, payload, now):
        self.now_offers.append((key, payload))
        return True


def test_the_tags_go_on_their_channel_once_a_second_and_on_a_change(tmp_path):
    tags = tg.SessionTags(store=store(tmp_path))
    pilot = types.SimpleNamespace(tags=tags, clicks=None)
    tele = Tele()
    cm.goal_telemetry(tele, pilot, None, 1.0)
    assert tele.offers == [] and tele.now_offers == [], "nobody listening, nothing gated through"
    tele.wanted.add("session_tags")
    cm.goal_telemetry(tele, pilot, None, 2.0)
    assert tele.offers == [("session_tags", tags.payload())]
    tags.command({"cmd": "tags", "clean": True})
    cm.goal_telemetry(tele, pilot, None, 2.1, pad=False)
    assert tele.now_offers == [("session_tags", {"clean": True, "tags": "clean", "seq": 1})]
    # Without a SessionTags (an old AutoPilot) nothing is sent.
    cm.goal_telemetry(Tele(["session_tags"]), types.SimpleNamespace(clicks=None), None, 3.0)
    assert tm.JSON_TOPICS["session_tags"] == tg.TOPIC == "/session/tags"
    assert tm.INTERVALS["session_tags"] == 1.0
    assert set(tm.JSON_SCHEMAS["session_tags"]["schema"]["properties"]) == set(tags.payload())


def test_a_record_session_says_its_tags_as_it_starts(tmp_path):
    from test_record import FakeHub, SessionTele
    tags = tg.SessionTags(store=store(tmp_path))
    tags.command({"cmd": "tags", "clean": True})
    log = FakeLogger()
    session = cm.RecordSession(SessionTele(), FakeHub(), log, make_recorder=lambda: "rec",
                               tags=tags)
    driver = types.SimpleNamespace(forward_speed=0.4, reverse_speed=-1.0)
    assert session.begin(0.0, driver)
    assert any(t.startswith("Recording, throttle capped") and t.endswith("tagged clean")
               for _, _, t in log.lines), log.lines


class MetaWriter:
    def __init__(self):
        self.records = []

    def write_metadata(self, name, values):
        self.records.append((name, values))

    def flush(self):
        pass

    def close(self):
        pass


def test_every_file_opens_with_the_tags_as_they_are_then(tmp_path, fake_sdk):  # noqa: F811
    tags = tg.SessionTags(store=store(tmp_path))
    writers = []

    def open_mcap(path, writer_options=None):
        open(path, "wb").close()
        writers.append(MetaWriter())
        return writers[-1]

    fake_sdk_module = __import__("sys").modules["foxglove"]
    fake_sdk_module.open_mcap = open_mcap
    r = make(tmp_path, metadata=lambda: {tg.METADATA_NAME: tags.metadata()})
    assert r.start()
    assert writers[0].records == [("crawler.tags", {"clean": "false", "tags": ""})]
    tags.command({"cmd": "tags", "clean": True})
    r._rotate()
    assert writers[1].records[0][1] == {"clean": "true", "tags": "clean"}
    assert r.metadata_written == 2
    # One that raises costs the record, never the file.
    log = FakeLogger()
    bad = make(tmp_path, logger=log, metadata=lambda: 1 / 0)
    assert bad.start() and bad.active and log.warnings()


def test_a_real_writer_takes_the_metadata_record(tmp_path):
    pytest.importorskip("foxglove")
    tags = tg.SessionTags()
    tags.command({"cmd": "tags", "clean": True})
    r = recording.Recorder(logger=FakeLogger(), directory=str(tmp_path),
                           metadata=lambda: {tg.METADATA_NAME: tags.metadata()})
    assert r.start() and r.metadata_written == 1
    path = r.path
    r.stop()
    body = open(path, "rb").read()
    assert b"crawler.tags" in body and b"clean" in body


# ---- the tools' side --------------------------------------------------------------

def test_a_tick_takes_the_newest_tags_at_or_before_it():
    entries = [(T0 + 5 * S, 0), (0, CLEAN), (T0 + 9 * S, CLEAN)]
    t = [T0, T0 + 5 * S - 1, T0 + 5 * S, T0 + 8 * S, T0 + 10 * S]
    assert ptg.tags_at(entries, t).tolist() == [CLEAN, CLEAN, 0, 0, CLEAN]
    # No metadata record: before the first message, the first message's.
    assert ptg.tags_at([(T0 + 5 * S, CLEAN)], [T0, T0 + 6 * S]).tolist() == [CLEAN, CLEAN]
    assert ptg.tags_at([], [T0, T0 + S]).tolist() == [0, 0]


def test_the_adapters_read_the_channel_and_the_metadata_record():
    for cls in (pa.FootprintAdapter, pa.LegacyAdapter):
        ad = cls("s")
        assert tg.TOPIC in ad.topics
        ad.feed(tg.TOPIC, T0 + 3 * S, {"clean": False, "tags": ""})
        ad.feed_metadata(tg.METADATA_NAME, {"clean": "true"})
        ad.feed_metadata("something.else", {"clean": "true"})
        assert ad.finish().tags == ((0, CLEAN), (T0 + 3 * S, 0))
    assert pa.Session("s", "x", "c", [], [], [], [], (), False).tags == ()


def tagged_dataset(tags):
    session = footprint_session(record([(4, 0.3, 0.2, 0.1)]))._replace(tags=tags)
    ticks, _, _ = pd.align(session, rules=None)
    return session, ticks, pd.to_arrays([(session, ticks)])


def test_the_dataset_carries_each_ticks_tags_and_untagged_is_zero():
    _, ticks, a = tagged_dataset(((0, 0), (T0 + 2 * S, CLEAN)))
    t = (a["t_ns"] - T0) / S
    assert a["tags"].dtype == np.uint8 and len(a["tags"]) == len(ticks)
    assert (a["tags"][t < 2] == 0).all() and (a["tags"][t >= 2] == CLEAN).all()
    _, _, plain = tagged_dataset(())
    assert not plain["tags"].any()


def test_filters_and_weights_name_a_tag_or_its_opposite():
    # The spare bit (an outdoor tag, recorded on its branch) is neither.
    bits = np.array([0, SPARE, CLEAN, SPARE | CLEAN], np.uint8)
    assert ptg.keep_mask(bits).all()
    assert ptg.keep_mask(bits, ["clean"]).tolist() == [False, False, True, True]
    assert ptg.keep_mask(bits, ["messy"]).tolist() == [True, True, False, False]
    assert ptg.keep_mask(bits, exclude=["messy"]).tolist() == [False, False, True, True]
    assert ptg.keep_mask(bits, ["clean"], ["clean"]).tolist() == [False] * 4
    assert ptg.weights(bits, {"clean": 3.0, "messy": 0.5}).tolist() == [0.5, 0.5, 3.0, 3.0]
    assert ptg.parse_names(" Clean, messy ") == ["clean", "messy"]
    assert ptg.parse_names(None) == [] and ptg.parse_weights(["messy=0"]) == {"messy": 0.0}
    for bad in (lambda: ptg.parse_names("muddy"), lambda: ptg.parse_names("outdoor"),
                lambda: ptg.parse_names("indoor"), lambda: ptg.parse_weights(["clean"]),
                lambda: ptg.parse_weights(["clean=-1"]), lambda: ptg.parse_weights(["x=2"]),
                lambda: ptg.parse_weights(["outdoor=2"]), lambda: ptg.parse_weights(["clean=nan"])):
        with pytest.raises(ValueError):
            bad()
    assert ptg.counts(bits, [True, False, True, False]) == {"messy": 1, "clean": 1}
    assert not hasattr(ptg, "place_mask") and not hasattr(ptg, "PLACES")
    # A dataset built before the tags: all messy.
    assert ptg.tick_tags({"t_ns": np.zeros(3)}).tolist() == [0, 0, 0]


def test_the_tag_weight_rides_on_every_other_tick_weight():
    _, _, a = tagged_dataset(((0, 0), (T0 + 2 * S, CLEAN)))
    data = dict(a, meta={"sessions": ["s"], "sources": list(pd.SOURCES)})
    idx = np.arange(len(a["t_ns"]))
    before = ptr.tick_weights(data, idx, 2.0)
    data["tag_w"] = ptg.weights(ptg.tick_tags(data), {"clean": 3.0})
    after = ptr.tick_weights(data, idx, 2.0)
    clean = (a["tags"] & CLEAN) != 0
    assert clean.any() and (~clean).any()
    assert np.allclose(after[clean], 3.0 * before[clean])
    assert np.allclose(after[~clean], before[~clean])


def test_val_grades_only_the_ticks_the_filter_kept():
    data = {"decision": np.array([True, True, True]), "session": np.array([0, 0, 0]),
            "meta": {"sessions": ["s"]}}
    graded = []
    pe_score_ticks = pe.score_ticks
    pe.median_training_magnitude = pe.median_training_magnitude
    try:
        pe.score_ticks = lambda model, d, ticks, *a: graded.append(list(ticks)) or {}
        orig_bad = pe.bad_marked
        pe.bad_marked = lambda d: np.zeros(3, bool)
        orig_med = pe.median_training_magnitude
        pe.median_training_magnitude = lambda d, m: 1.0
        pe.score(None, data, [0], "pad", None, keep=np.array([True, False, True]))
        pe.score(None, data, [0], "pad", None)
    finally:
        pe.score_ticks, pe.bad_marked = pe_score_ticks, orig_bad
        pe.median_training_magnitude = orig_med
    assert graded[0] == [0, 2] and graded[2] == [0, 1, 2]


def test_the_trainer_and_retrain_take_the_same_flags():
    source = open(ptr.__file__).read()
    for flag in ("--include-tags", "--exclude-tags", "--tag-weight"):
        assert f'add_argument("{flag}"' in source
    data = {"t_ns": np.zeros(3), "tags": np.array([0, CLEAN, CLEAN], np.uint8)}
    assert rt.tag_keep(data, []).all()
    assert rt.tag_keep(data, ["--include-tags", "clean", "--epochs", "3"]).tolist() == [
        False, True, True]
    assert rt.tag_keep(data, ["--exclude-tags=clean"]).tolist() == [True, False, False]


def test_the_scorecard_filters_on_the_tags_and_has_no_place():
    data = {"t_ns": np.zeros(3), "tags": np.array([0, CLEAN, SPARE], np.uint8)}
    assert sc.tag_keep(data).all()
    assert sc.tag_keep(data, ["clean"]).tolist() == [False, True, False]
    assert sc.tag_keep(data, exclude=["clean"]).tolist() == [True, False, True]
    assert sc.tag_keep({"t_ns": np.zeros(2)}, ["messy"]).all(), "untagged is messy"
    assert sc.main(["x.npz", "m.onnx", "--map", "d", "--include-tags", "muddy"]) == 2
    assert not hasattr(sc, "OUTDOOR_NOTE")
    with pytest.raises(SystemExit):
        sc.main(["x.npz", "m.onnx", "--map", "d", "--place", "indoor"])


def test_the_list_says_each_sessions_tags_ticks_and_minutes(tmp_path, capsys):
    s1, t1, a1 = tagged_dataset(((0, 0), (T0 + 2 * S, CLEAN)))
    data = dict(a1, meta={"sessions": [s1.name]})
    rows = st.dataset_rows(data)
    (name, mixes, n, labelled, minutes), = rows
    assert name == s1.name and n == len(t1) and minutes == pytest.approx(n / 600.0)
    assert set(mixes) == {"messy", "clean"} and sum(mixes.values()) == n
    text = st.format_rows(rows, "ticks")
    assert "totals by tags" in text and "clean" in text
    path = tmp_path / "d.npz"
    np.savez(path, meta=json.dumps(data["meta"]), **{k: v for k, v in a1.items()})
    assert st.main([str(path)]) == 0
    assert s1.name in capsys.readouterr().out
    assert st.main([str(path), "x.mcap"]) == 2
