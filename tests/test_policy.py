"""The learned policy's runtime, and the vetoes it cannot get past.

pigpio holds the last pulse width forever and nothing on this car fails
safe on its own, so the paths that matter most here are the ones where the
model is not trusted: no frame, a blind frame, something under the floor, a
stuck car, a nan out of the network, and an exception halfway through a
tick. Every one of them has to put a command on the wheels: the floor takes
the throttle off a forward and lets the steering through, the stuck detector
backs out, and everything else is neutral.

No torch anywhere in here. The real-model tests run the committed smoke
model through onnxruntime, which is what the car runs.
"""

import hashlib
import json
import math
import pathlib

import numpy as np
import pytest

import camera
import column_profile as cp
import driver as dv
import main as mn
import policy as pl

from conftest import FakeCamera, FakeController

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "policy-smoke.onnx"

T0 = 100.0  # past the recover cooldown, so a stuck can fire


class StubModel:
    """A model that says whatever it is told, and counts how often it is asked."""

    contract = pl.CONTRACT_LEGACY_JET
    representation = pl.PROFILE
    uses_prev = False

    def __init__(self, logits=(5.0, 0.0, 0.0), throttle=1.0, steer=0.0, raises=None):
        self.out = {"logits": np.array([logits], dtype=np.float32),
                    "throttle": np.array([[throttle]], dtype=np.float32),
                    "steer": np.array([[steer]], dtype=np.float32)}
        self.raises = raises
        self.calls = []

    def run(self, x, prev):
        self.calls.append((x.shape, prev.copy()))
        if self.raises:
            raise self.raises
        return self.out


FORWARD = (5.0, 0.0, 0.0)
REVERSE = (0.0, 5.0, 0.0)
STOP = (0.0, 0.0, 5.0)


def clear(mm=3000):
    return np.full((400, 640), mm, dtype=np.uint16)


def policy_driver(model, depth=None, imu=None):
    return pl.PolicyDriver(FakeController(), FakeCamera(depth=depth, imu=imu), model)


def neutral(d):
    return d.controller.throttles[-1] == 0 and d.controller.steerings[-1] == (0, 0)


# --- what the model asks for is what the wheels get -------------------------

def test_forward_is_a_fraction_of_forward_speed_never_more():
    d = policy_driver(StubModel(FORWARD, throttle=0.5, steer=0.5), clear())
    d.tick(T0)
    assert d.controller.throttles[-1] == pytest.approx(0.5 * d.forward_speed)
    assert d.controller.steerings[-1] == dv.steering_pair(0.5 * pl.STEER_SCALE)
    assert d.last_decision.action == pl.FORWARD


def test_stop_takes_the_throttle_off_and_still_steers():
    d = policy_driver(StubModel(STOP, throttle=0.9, steer=-0.25), clear())
    d.tick(T0)
    assert d.controller.throttles == [0]
    assert d.controller.steerings[-1] == dv.steering_pair(-0.25 * pl.STEER_SCALE)


def test_full_steer_is_manuals_full_lock():
    """The model learns in manual's units, so 1.0 is the pad's full lock."""
    assert pl.STEER_SCALE == mn.STEERING_LIMIT_PERC / 100
    d = policy_driver(StubModel(FORWARD, steer=-1.0), clear())
    d.tick(T0)
    assert d.controller.steerings[-1] == dv.steering_pair(-mn.STEERING_LIMIT_PERC / 100)


def test_a_model_reverse_is_decided_again_every_tick():
    model = StubModel(REVERSE, throttle=0.5, steer=1.0)
    d = policy_driver(model, clear())
    d.tick(T0)
    assert d._phase is None and not d.recovering(T0 + 0.1)
    assert d.controller.throttles[-1] == pytest.approx(0.5 * d.reverse_speed)
    assert d.controller.steerings[-1] == dv.steering_pair(pl.STEER_SCALE)
    # The next tick asks again, and the model can change its mind at once.
    model.out["logits"] = np.array([FORWARD], dtype=np.float32)
    model.out["steer"] = np.array([[-0.5]], dtype=np.float32)
    d.tick(T0 + 0.1)
    assert len(model.calls) == 2
    assert d.controller.throttles[-1] == pytest.approx(0.5 * d.forward_speed)
    assert d.controller.steerings[-1] == dv.steering_pair(-0.5 * pl.STEER_SCALE)


def test_last_verdict_speaks_the_autopilots_vocabulary_for_footprint_rows():
    d = policy_driver(StubModel(FORWARD, steer=0.3), clear())
    d.tick(T0)
    assert d.last_verdict.action == dv.ACT_FORWARD
    assert d.last_verdict.reason.startswith("policy")


# --- the vetoes ---------------------------------------------------------------

def test_no_frame_is_neutral_and_the_model_is_not_asked():
    d = policy_driver(StubModel(FORWARD), None)
    d.tick(T0)
    assert neutral(d)
    assert d.model.calls == []


def test_a_frame_that_is_not_an_image_is_neutral():
    d = policy_driver(StubModel(FORWARD), np.zeros(10, dtype=np.uint16))
    d.tick(T0)
    assert neutral(d)
    assert d.model.calls == []


def test_a_blind_frame_is_neutral_whatever_the_model_would_say():
    d = policy_driver(StubModel(FORWARD), np.zeros((400, 640), dtype=np.uint16))
    d.tick(T0)
    assert neutral(d)
    assert d.model.calls == []


def chair_leg():
    depth = clear()
    depth[:, 300:340] = 150  # inside REVERSE_FLOOR_MM, dead ahead
    return depth


def test_the_floor_takes_the_throttle_off_a_forward_and_lets_it_steer():
    d = policy_driver(StubModel(FORWARD, throttle=1.0, steer=0.4), chair_leg())
    d.tick(T0)
    assert d.floor_veto.vetoes()
    assert d._phase is None
    assert d.controller.throttles[-1] == 0.0
    assert d.controller.steerings[-1] == pytest.approx(dv.steering_pair(0.4 * pl.STEER_SCALE))
    assert d.last_decision.reason == "model floor"
    assert d.last_decision.throttle == 0.0


def centre_band(mm):
    """Probes 3-5 at `mm` (0 reads nothing), open floor either side -- so
    never the all-blind frame, which is neutral before the model is asked."""
    depth = clear()
    edges = dv.footprint_geometry(depth.shape).col_edges
    depth[:, edges[3]:edges[6]] = mm
    return depth


def drive_through(d, frames, start=T0):
    """One tick per frame, a decision apart. The reasons, in order."""
    reasons = []
    for k, frame in enumerate(frames):
        d.camera.depth = frame
        d.tick(start + k * 0.1)
        reasons.append(d.last_decision.reason)
    return reasons


def test_a_model_forward_stays_held_when_the_probe_goes_blind_on_the_way_in():
    d = policy_driver(StubModel(FORWARD), centre_band(390))
    reasons = drive_through(d, [centre_band(mm) for mm in (390, 300, 190, 0, 0)])
    assert reasons == ["model", "model floor", "model floor", "model floor",
                       "model floor"]
    assert d.controller.throttles[-1] == 0.0
    assert d.floor_veto.latch.latched[4]


def test_a_model_forward_through_a_dark_patch_is_not_held():
    d = policy_driver(StubModel(FORWARD), clear())
    reasons = drive_through(d, [clear(), clear(), centre_band(0), centre_band(0)])
    assert reasons == ["model"] * 4
    assert d.controller.throttles[-1] == pytest.approx(d.forward_speed)


def test_a_model_floor_lets_go_on_open_floor_and_on_backing_off():
    d = policy_driver(StubModel(FORWARD), centre_band(190))
    drive_through(d, [centre_band(190), centre_band(0)])
    assert d.floor_veto.vetoes()
    drive_through(d, [clear()], start=T0 + 0.2)
    assert d.last_decision.reason == "model"

    # Latched again, then the model backs up at the cap while the probe stays
    # blind: about 1.22 s of it lets go, and forward is the model's again.
    d = policy_driver(StubModel(REVERSE), centre_band(190))
    drive_through(d, [centre_band(190), centre_band(0)])
    drive_through(d, [centre_band(0) for _ in range(10)], start=T0 + 0.2)
    assert d.floor_veto.vetoes()  # 1.1 s of reverse: 180 mm, not yet
    drive_through(d, [centre_band(0) for _ in range(3)], start=T0 + 1.2)
    assert not d.floor_veto.vetoes()  # 1.4 s: 230 mm
    d.model.out["logits"] = np.array([FORWARD], dtype=np.float32)
    drive_through(d, [centre_band(0)], start=T0 + 1.5)
    assert d.last_decision.reason == "model"
    assert d.controller.throttles[-1] == pytest.approx(d.forward_speed)


def test_the_floor_never_holds_a_model_reverse():
    d = policy_driver(StubModel(REVERSE, throttle=1.0, steer=-0.4), chair_leg())
    d.tick(T0)
    assert d.controller.throttles[-1] == pytest.approx(d.reverse_speed)
    assert d.controller.steerings[-1] == pytest.approx(dv.steering_pair(-0.4 * pl.STEER_SCALE))
    assert d.last_decision.reason == "model"


def test_the_stuck_detector_still_fires_under_a_model_that_says_forward():
    still = {"accel": (0.0, 0.0, 0.0), "gyro": (0.001, 0.001, 0.001)}
    d = policy_driver(StubModel(FORWARD), clear(), imu=still)
    now = T0
    for _ in range(40):
        d.tick(now)
        if d._phase is not None:
            break
        now += 0.1
    assert d._phase == "reverse"
    assert d.last_decision.reason == "stuck"
    assert d.last_recovery_time == now


def test_stuck_cannot_fire_while_my_hand_has_the_throttle():
    still = {"accel": (0.0, 0.0, 0.0), "gyro": (0.001, 0.001, 0.001)}
    d = policy_driver(StubModel(FORWARD), clear(), imu=still)
    d.human_throttle = True
    now = T0
    for _ in range(40):
        d.tick(now)
        now += 0.1
    assert d._phase is None
    assert d.last_recovery_time == 0.0
    assert len(d.model.calls) == 40


BAD_OUTPUTS = [
    ("nan logit", dict(logits=(math.nan, 0.0, 0.0))),
    ("inf logit", dict(logits=(math.inf, 0.0, 0.0))),
    ("nan throttle", dict(throttle=math.nan)),
    ("throttle over 1", dict(throttle=1.2)),
    ("negative throttle", dict(throttle=-0.1)),
    ("nan steer", dict(steer=math.nan)),
    ("inf steer", dict(steer=-math.inf)),
    ("steer over 1", dict(steer=1.5)),
]


@pytest.mark.parametrize("name,kwargs", BAD_OUTPUTS, ids=[b[0] for b in BAD_OUTPUTS])
def test_model_output_that_is_not_finite_or_in_range_is_neutral(name, kwargs):
    """The nan case is the one that matters: min(1.0, nan) is 1.0, so a nan
    clamped on its way to the ESC is a full-forward pulse."""
    d = policy_driver(StubModel(**{"logits": FORWARD, **kwargs}), clear())
    d.tick(T0)
    assert neutral(d)
    assert all(t == 0 for t in d.controller.throttles)
    # Refused, not merely harmless: "model" alone is an accepted decision.
    assert d.last_decision.reason.startswith("model ")
    assert d.last_decision.reason != "model"


def test_model_output_of_the_wrong_shape_is_neutral():
    model = StubModel(FORWARD)
    model.out["logits"] = np.array([[5.0, 0.0]], dtype=np.float32)
    d = policy_driver(model, clear())
    d.tick(T0)
    assert neutral(d)


def test_a_missing_output_is_neutral():
    model = StubModel(FORWARD)
    del model.out["throttle"]
    d = policy_driver(model, clear())
    d.tick(T0)
    assert neutral(d)


def test_an_exception_in_the_model_commands_neutral_then_raises():
    d = policy_driver(StubModel(FORWARD, raises=RuntimeError("onnx fell over")), clear())
    d.controller.set_throttle(0.2)  # last tick was driving
    with pytest.raises(RuntimeError):
        d.tick(T0)
    assert neutral(d)


def test_an_exception_in_the_steering_write_still_zeroes_the_throttle():
    class SteeringRaises(FakeController):
        def set_steering(self, front, rear):
            raise OSError("i2c went away")

    d = pl.PolicyDriver(SteeringRaises(), FakeCamera(depth=clear()), StubModel(FORWARD))
    with pytest.raises(OSError):
        d.tick(T0)
    assert d.controller.throttles[-1] == 0


def test_output_within_tolerance_is_accepted_at_the_edge():
    decided, _ = pl.read_output({"logits": [[0, 0, 1]], "throttle": [[1.00005]],
                                 "steer": [[-1.00005]]})
    assert decided == (pl.STOP, 1.0, -1.0, None)


def test_a_rear_head_is_read_and_checked_when_there_is_one():
    ok, _ = pl.read_output({"logits": [[1, 0, 0]], "throttle": [[0.5]],
                            "steer": [[0.2]], "rear": [[0.1]]})
    assert ok[3] == pytest.approx(0.1)
    bad, why = pl.read_output({"logits": [[1, 0, 0]], "throttle": [[0.5]],
                               "steer": [[0.2]], "rear": [[math.nan]]})
    assert bad is None and why == "not finite"


# --- the inputs -----------------------------------------------------------------

def test_the_depth_small_contract_is_the_recordings_own_function():
    import policy_input

    assert pl.downsample_for(pl.CONTRACT_DEPTH_SMALL) is policy_input.depth_small


def test_an_unknown_contract_is_refused():
    with pytest.raises(ValueError):
        pl.downsample_for("jpeg_vibes")


def test_legacy_quantisation_matches_the_telemetry_colourmap_window():
    raw = np.zeros((400, 640), dtype=np.uint16)
    raw[:, :320] = 2000
    raw[:, 320:] = 250  # nearer than the colourmap's 300 mm floor
    mm = pl.legacy_mm_from_raw(raw)
    assert mm.shape == pl.LEGACY_SHAPE
    assert abs(int(mm[0, 0]) - 2000) <= 11  # one level of 2700/255 mm
    assert mm[0, -1] == 0  # clipped to the near end: indistinguishable from invalid
    small = pl.legacy_small_from_raw(raw)
    assert small.shape == (pl.INPUT_HEIGHT, pl.INPUT_WIDTH)
    assert small.dtype == np.uint16


def test_encode_depth_scales_and_masks():
    small = np.zeros((pl.INPUT_HEIGHT, pl.INPUT_WIDTH), dtype=np.uint16)
    small[0, 0] = 1500
    small[0, 1] = 9000
    x = pl.encode_depth(small)
    assert x.shape == (1, 2, pl.INPUT_HEIGHT, pl.INPUT_WIDTH)
    assert x[0, 0, 0, 0] == pytest.approx(0.5)
    assert x[0, 0, 0, 1] == 1.0
    assert x[0, 1, 0, 2] == 0.0 and x[0, 1, 0, 0] == 1.0


def test_encode_prev_never_carries_a_nan():
    prev = pl.encode_prev(pl.REVERSE, math.nan, 0.5)
    assert prev.tolist() == [[0.0, 1.0, 0.0, 0.0, 0.5]]


def _profile_by_loop(small, columns=cp.PROFILE_COLUMNS):
    top, bottom = cp.profile_rows(small.shape)
    band = small[top:bottom].astype(np.float32)
    edges = np.linspace(0, small.shape[1], columns + 1).round().astype(int)
    out = np.empty((3, columns), np.float32)
    for i in range(columns):
        cell = band[:, edges[i]:edges[i + 1]].ravel()
        valid = cell[cell > 0]
        if valid.size:
            out[0, i] = min(1.0, valid.min() / cp.PROFILE_SCALE_MM)
            out[1, i] = min(1.0, float(np.median(valid)) / cp.PROFILE_SCALE_MM)
        else:
            out[0, i] = out[1, i] = 1.0
        out[2, i] = valid.size / max(1, cell.size)
    return out


def test_the_column_profile_is_the_plain_per_column_loop():
    rng = np.random.default_rng(7)
    for k in range(50):
        small = rng.integers(0, 5000, (pl.INPUT_HEIGHT, pl.INPUT_WIDTH)).astype(np.uint16)
        small[rng.random(small.shape) < rng.random()] = 0
        if k % 5 == 0:
            small[:, 20:30] = 0
        np.testing.assert_allclose(cp.column_profile(small), _profile_by_loop(small),
                                   atol=1e-6)


def test_the_column_profile_reads_the_footprints_rows_only():
    small = np.full((pl.INPUT_HEIGHT, pl.INPUT_WIDTH), 2400, dtype=np.uint16)
    top, bottom = cp.profile_rows(small.shape)
    small[:top] = 300       # the ceiling and the floor right under the lens
    small[bottom:] = 300    # are not in the band
    profile = cp.column_profile(small)
    assert profile.shape == (cp.PROFILE_FEATURES, cp.PROFILE_COLUMNS)
    assert np.allclose(profile[0], 0.8) and np.allclose(profile[2], 1.0)


def test_a_blind_column_reads_far_with_no_evidence():
    small = np.full((pl.INPUT_HEIGHT, pl.INPUT_WIDTH), 1200, dtype=np.uint16)
    small[:, :5] = 0
    profile = cp.column_profile(small)
    assert profile[:, 0].tolist() == [1.0, 1.0, 0.0]
    assert profile[2, -1] == 1.0


# --- the real exported model, through onnxruntime -----------------------------

def test_the_committed_smoke_model_loads_and_answers_in_range():
    model = pl.PolicyModel(str(FIXTURE))
    assert model.representation in pl.REPRESENTATIONS
    assert set(model.output_names) >= {"logits", "throttle", "steer"}
    rng = np.random.default_rng(3)
    for _ in range(20):
        small = rng.integers(0, 4000, (pl.INPUT_HEIGHT, pl.INPUT_WIDTH)).astype(np.uint16)
        out = model.run(pl.encode_input(model.representation, small),
                        np.zeros((1, pl.PREV_SIZE), np.float32))
        decided, why = pl.read_output(out)
        assert decided is not None, why


def test_the_smoke_model_drives_a_clear_frame_inside_the_caps():
    model = pl.PolicyModel(str(FIXTURE))
    d = pl.PolicyDriver(FakeController(), FakeCamera(depth=clear()), model)
    d.tick(T0)
    t = d.controller.throttles[-1]
    assert d.reverse_speed <= t <= d.forward_speed
    front, rear = d.controller.steerings[-1]
    assert abs(front) <= pl.STEER_SCALE and math.isfinite(front) and math.isfinite(rear)


def test_a_sidecar_at_another_steering_scale_is_refused(tmp_path):
    meta = json.loads(FIXTURE.with_suffix(".json").read_text())
    assert meta["steer_scale"] == pl.STEER_SCALE
    meta["steer_scale"] = 0.6
    (tmp_path / "m.json").write_text(json.dumps(meta))
    (tmp_path / "m.onnx").write_bytes(FIXTURE.read_bytes())
    with pytest.raises(ValueError):
        pl.PolicyModel(str(tmp_path / "m.onnx"))


def test_policy_fields_name_the_model_by_its_hash():
    model = pl.PolicyModel(str(FIXTURE))
    d = pl.PolicyDriver(FakeController(), FakeCamera(depth=clear()), model)
    d.tick(T0)
    fields = d.policy_fields()
    assert fields["autopilot"] == pl.AUTOPILOT_POLICY
    assert fields["policy_model"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert fields["policy_action"] == d.last_decision.action
    assert fields["policy_throttle"] == d.controller.throttles[-1]
    assert fields["policy_steering_front"] == d.controller.steerings[-1][0]
    assert fields["policy_infer_ms"] > 0.0


def test_a_sidecar_with_an_unknown_contract_is_refused(tmp_path):
    (tmp_path / "m.json").write_text('{"input_contract": "x", "representation": "profile",'
                                     ' "actions": ["forward", "reverse", "stop"]}')
    (tmp_path / "m.onnx").write_bytes(FIXTURE.read_bytes())
    with pytest.raises(ValueError):
        pl.PolicyModel(str(tmp_path / "m.onnx"))


# --- takeover, over the policy -----------------------------------------------

def run_policy(monkeypatch, model, passes, plan, tele=None, tap=None):
    import test_takeover as tk
    return tk.run_auto(monkeypatch, passes, tk.Camera(clear()), plan=plan,
                       tele=tele, tap=tap,
                       make_driver=lambda command, cam: pl.PolicyDriver(command, cam, model))


def test_my_steering_takes_over_from_the_model_and_leaves_it_the_throttle(monkeypatch):
    import test_takeover as tk
    out = run_policy(monkeypatch, StubModel(FORWARD, throttle=1.0, steer=0.0), 60,
                     tk.at(20, LEFT_X=0.5))
    driver, s = out["driver"], out["snaps"]
    assert isinstance(driver, pl.PolicyDriver) and driver.model.calls
    assert s[10]["steer_by"] == mn.DRIVER
    assert s[10]["steering"] == pytest.approx(dv.steering_pair(0.0))
    assert s[10]["throttle"] == pytest.approx(driver.forward_speed)
    for snap in s[20:]:
        assert (snap["steer_by"], snap["throttle_by"]) == (mn.PAD, mn.DRIVER)
        assert snap["steering"] == pytest.approx(dv.steering_pair(tk.limited(0.5)))
        assert snap["throttle"] == pytest.approx(driver.forward_speed)


def test_my_throttle_takes_over_from_the_model_and_leaves_it_the_steering(monkeypatch):
    import test_takeover as tk
    out = run_policy(monkeypatch, StubModel(FORWARD, throttle=1.0, steer=0.5), 60,
                     tk.at(20, throttle=0.5))
    driver = out["driver"]
    for snap in out["snaps"][20:]:
        assert (snap["steer_by"], snap["throttle_by"]) == (mn.DRIVER, mn.PAD)
        assert snap["steering"] == pytest.approx(dv.steering_pair(0.5 * pl.STEER_SCALE))
        assert snap["throttle"] == pytest.approx(0.5 * driver.forward_speed)
    assert driver.human_throttle


def test_footprint_rows_carry_the_models_own_command_under_my_hand(monkeypatch):
    import footprint as fp
    import telemetry as tm
    import test_takeover as tk
    from test_record import SessionTele

    tele = SessionTele()
    out = run_policy(monkeypatch, StubModel(FORWARD, throttle=1.0, steer=-0.5), 60,
                     tk.at(20, LEFT_X=0.5), tele=tele, tap=fp.FootprintTap(tele=tele))
    driver = out["driver"]
    rows = [row for row, _ in tele.of("footprint")]
    assert rows
    model_front = pytest.approx(-0.5 * pl.STEER_SCALE)
    for row in rows:
        assert set(row) == set(tm.JSON_SCHEMAS["footprint"]["schema"]["properties"])
        assert row["autopilot"] == "policy"
        assert row["policy_action"] == pl.FORWARD and row["policy_reason"] == "model"
        assert row["policy_throttle"] == pytest.approx(driver.forward_speed)
        assert row["policy_steering_front"] == model_front
    mine = [row for row in rows if row["steer_by"] == mn.PAD]
    assert mine and all(row["steering_front"] == pytest.approx(tk.limited(0.5))
                        for row in mine)
    assert any(row["steering_front"] == model_front for row in rows
               if row["steer_by"] == mn.DRIVER)


def test_rows_under_the_rules_say_rules_and_record_rows_say_nobody():
    import footprint as fp

    assert fp.autopilot_fields(None) == fp.AUTOPILOT_FIELDS
    assert fp.AUTOPILOT_FIELDS["autopilot"] == ""
    rules = fp.autopilot_fields(dv.Driver(FakeController(), FakeCamera()))
    assert rules == {**fp.AUTOPILOT_FIELDS, "autopilot": pl.AUTOPILOT_RULES}
    d = policy_driver(StubModel(REVERSE, throttle=0.5, steer=0.2), clear())
    d.tick(T0)
    fields = fp.autopilot_fields(d)
    assert fields["autopilot"] == pl.AUTOPILOT_POLICY
    assert fields["policy_action"] == pl.REVERSE
    assert fields["policy_throttle"] == pytest.approx(0.5 * d.reverse_speed)


# --- main.py: opt-in, and loud when it fails ----------------------------------

class ReadyHub:
    state = camera.READY
    error = ""

    def view(self, name):
        return FakeCamera()


class Log:
    def __init__(self):
        self.lines = []

    def log(self, level, subject, message):
        self.lines.append((level, subject, message))


def test_no_policy_flag_is_the_hand_written_autopilot():
    pilot = mn.AutoPilot(FakeController(), Log(), hub=ReadyHub())
    assert pilot.resume() is True
    assert type(pilot.driver) is dv.Driver
    assert mn.parse_args([]).policy is None


def test_the_policy_flag_builds_a_policy_driver():
    args = mn.parse_args(["--policy", str(FIXTURE)])
    pilot = mn.AutoPilot(FakeController(), Log(), hub=ReadyHub(), policy_path=args.policy)
    assert pilot.resume() is True
    assert isinstance(pilot.driver, pl.PolicyDriver)
    # Its commands go into the autopilot's held command, never to the
    # hardware, so takeover mixes my hand in over it.
    assert pilot.driver.controller is pilot.command


def test_a_model_that_will_not_load_is_auto_unavailable_not_the_autopilot(tmp_path):
    pilot = mn.AutoPilot(FakeController(), Log(), hub=ReadyHub(),
                         policy_path=str(tmp_path / "missing.onnx"))
    assert pilot.resume() is False
    assert pilot.unavailable is True
    assert pilot.driver is None
