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

from crawler.hardware import camera
from crawler.autopilot import column_profile as cp
from crawler.autopilot import driver as dv
from crawler.autopilot import footprint as fp
import crawler.__main__ as mn
from crawler.autopilot import policy as pl

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


# --- the goal: the model's stop is only committed once the car has arrived ---

def test_the_goal_is_never_reached_unless_something_says_so():
    from crawler.autopilot import goal as gl
    assert gl.Goal().reached() is False
    d = policy_driver(StubModel(FORWARD), clear())
    assert d.goal.reached() is False
    pilot = mn.AutoPilot(None, None)
    assert pilot.goal.reached() is False


@pytest.mark.parametrize("logits,want,speed", [
    ((0.0, 1.0, 5.0), pl.REVERSE, "reverse_speed"),
    ((1.0, 0.0, 5.0), pl.FORWARD, "forward_speed"),
])
def test_a_model_stop_short_of_the_goal_drives_the_better_of_forward_and_reverse(
        logits, want, speed):
    d = policy_driver(StubModel(logits, throttle=0.5, steer=-0.25), clear())
    d.tick(T0)
    # At the model's magnitude, slowed by its confidence no further than the
    # moving floor for that direction.
    floor = pl.MOVING_FLOOR_REVERSE if want == pl.REVERSE else pl.MOVING_FLOOR_FORWARD
    assert d.controller.throttles[-1] == pytest.approx(
        pl.slowed_throttle(0.5 * getattr(d, speed), d.confidence.scale, floor))
    # A reverse out of a stop is a fresh one, so the commit steers it: to the
    # model's own side, at REVERSE_STEER.
    front = -pl.REVERSE_STEER if want == pl.REVERSE else -0.25 * pl.STEER_SCALE
    assert d.controller.steerings[-1] == pytest.approx(dv.steering_pair(front))
    assert d.last_decision.action == want
    assert d.last_decision.model_action == pl.STOP
    assert d.last_decision.stop_masked
    assert d.last_verdict.action == {pl.FORWARD: dv.ACT_FORWARD,
                                     pl.REVERSE: dv.ACT_REVERSE}[want]
    fields = d.policy_fields()
    assert fields["policy_action"] == pl.STOP
    assert fields["policy_stop_masked"] is True
    assert fields["goal_reached"] is False


def test_a_model_that_does_not_say_stop_is_not_masked():
    d = policy_driver(StubModel(REVERSE, throttle=0.5), clear())
    d.tick(T0)
    assert not d.last_decision.stop_masked
    assert d.policy_fields()["policy_action"] == pl.REVERSE


def test_the_goal_reached_is_neutral_whatever_the_model_says():
    from crawler.autopilot import goal as gl
    goal = gl.Goal()
    d = pl.PolicyDriver(FakeController(), FakeCamera(depth=clear()),
                        StubModel(FORWARD, throttle=1.0, steer=0.5), goal=goal)
    d.tick(T0)
    assert d.controller.throttles[-1] > 0
    goal.set_reached(True)
    d.tick(T0 + 0.1)
    assert neutral(d)
    assert d.last_decision.action == pl.STOP
    assert d.last_decision.reason == "goal reached"
    assert d.last_verdict.action == pl.STOP
    fields = d.policy_fields()
    assert fields["goal_reached"] is True
    assert fields["policy_action"] == pl.FORWARD
    assert fields["policy_stop_masked"] is False
    # And a model stop with the goal reached is committed as a stop.
    d.model.out["logits"] = np.array([STOP], dtype=np.float32)
    d.tick(T0 + 0.2)
    assert neutral(d) and d.last_decision.reason == "goal reached"
    assert d.policy_fields()["policy_action"] == pl.STOP


def test_the_floor_still_holds_a_masked_stop_that_comes_out_forward():
    d = policy_driver(StubModel((1.0, 0.0, 5.0), throttle=1.0, steer=0.4), chair_leg())
    d.tick(T0)
    assert d.floor_veto.vetoes()
    assert d.controller.throttles[-1] == 0.0
    assert d.last_decision.reason == "model floor"
    assert d.last_decision.stop_masked


def test_every_safety_stop_still_stops_while_the_goal_is_not_reached():
    # No frame, a blind frame, a bad output: neutral, and nothing masked.
    for depth, out in ((None, None), (np.zeros((400, 640), dtype=np.uint16), None),
                       (clear(), float("nan"))):
        model = StubModel(STOP, throttle=1.0 if out is None else out)
        d = policy_driver(model, depth)
        d.tick(T0)
        assert neutral(d), depth
        assert not d.goal_reached
        assert not d.policy_fields()["policy_stop_masked"]


def test_full_steer_is_manuals_full_lock():
    """The model learns in manual's units, so 1.0 is the pad's full lock."""
    assert pl.STEER_SCALE == mn.STEERING_LIMIT_PERC / 100
    d = policy_driver(StubModel(FORWARD, steer=-1.0), clear())
    d.tick(T0)
    assert d.controller.steerings[-1] == dv.steering_pair(-mn.STEERING_LIMIT_PERC / 100)


def test_a_model_reverse_is_decided_again_every_tick_once_its_commit_is_up():
    model = StubModel(REVERSE, throttle=0.5, steer=1.0)
    d = policy_driver(model, clear())
    d.tick(T0)
    assert d._phase is None and not d.recovering(T0 + 0.1)
    assert d.controller.throttles[-1] == pytest.approx(0.5 * d.reverse_speed)
    assert d.controller.steerings[-1] == dv.steering_pair(pl.STEER_SCALE)
    # Past the commit, still reversing, it is the model's tick by tick: its
    # own steering, and it can change its mind at once.
    model.out["steer"] = np.array([[0.1]], dtype=np.float32)
    d.tick(T0 + pl.REVERSE_COMMIT_S)
    assert not d.reverse_commit
    assert d.controller.steerings[-1] == pytest.approx(dv.steering_pair(0.1 * pl.STEER_SCALE))
    model.out["logits"] = np.array([FORWARD], dtype=np.float32)
    model.out["steer"] = np.array([[-0.5]], dtype=np.float32)
    d.tick(T0 + pl.REVERSE_COMMIT_S + 0.1)
    assert len(model.calls) == 3
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
    assert d.controller.steerings[-1] == pytest.approx(dv.steering_pair(-pl.REVERSE_STEER))
    assert d.last_decision.reason == "model commit"
    d.tick(T0 + pl.REVERSE_COMMIT_S)
    assert d.controller.throttles[-1] == pytest.approx(d.reverse_speed)
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
    from crawler.autopilot import policy_input

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


def test_policy_fields_name_the_model_by_its_stem_and_its_hash():
    model = pl.PolicyModel(str(FIXTURE))
    d = pl.PolicyDriver(FakeController(), FakeCamera(depth=clear()), model)
    d.tick(T0)
    fields = d.policy_fields()
    assert fields["autopilot"] == pl.AUTOPILOT_POLICY
    # An unregistered sidecar: named by its file.
    assert fields["policy_model"] == "policy-smoke"
    assert fields["policy_model_sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
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
    from crawler.autopilot import footprint as fp
    from crawler.telemetry import telemetry as tm
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


def test_footprint_rows_say_the_goal_and_the_masked_stop(monkeypatch):
    from crawler.autopilot import footprint as fp
    from crawler.telemetry import telemetry as tm
    from test_record import SessionTele

    tele = SessionTele()
    out = run_policy(monkeypatch, StubModel((0.0, 1.0, 5.0), throttle=0.5), 30,
                     lambda n, pad, cam: None, tele=tele, tap=fp.FootprintTap(tele=tele))
    rows = [row for row, _ in tele.of("footprint")]
    assert rows
    props = tm.JSON_SCHEMAS["footprint"]["schema"]["properties"]
    assert props["goal_reached"]["type"] == "boolean"
    assert props["policy_stop_masked"]["type"] == "boolean"
    for row in rows:
        assert row["goal_reached"] is False
        assert row["policy_stop_masked"] is True
        assert row["policy_action"] == pl.STOP
        assert row["committed_action"] == dv.ACT_REVERSE
    assert fp.AUTOPILOT_FIELDS["goal_reached"] is False
    assert fp.AUTOPILOT_FIELDS["policy_stop_masked"] is False
    assert out["driver"].goal_reached is False


def test_rows_under_the_rules_say_rules_and_record_rows_say_nobody():
    from crawler.autopilot import footprint as fp

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


# --- crawler/__main__.py: opt-in, and loud when it fails ----------------------------------

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


# --- the steering mode: 2WS / 4WS ---------------------------------------------

SWITCH = {"threshold": pl.MODE_THRESHOLD, "dwell_s": pl.MODE_DWELL_S,
          "min_interval_s": pl.MODE_MIN_INTERVAL_S}


def split_logits(p):
    """Mode logits whose softmax gives p(split) = p."""
    return np.array([[0.0, math.log(p / (1.0 - p))]], dtype=np.float32)


class ModeStub(StubModel):
    """StubModel with a mode head: p(split) is whatever it is told."""

    has_mode = True
    mode_switch = SWITCH

    def __init__(self, p_split=0.5, **kw):
        # A stop with no magnitude: driven on as a forward at 0, so it still
        # steers and never arms STUCK.
        kw.setdefault("logits", STOP)
        kw.setdefault("throttle", 0.0)
        kw.setdefault("steer", 0.5)
        super().__init__(**kw)
        self.set_p(p_split)
        self.modes = []

    def set_p(self, p):
        self.out[pl.MODE_OUTPUT] = split_logits(p)

    def run(self, x, prev, mode):
        self.modes.append(pl.STEER_MODES[int(np.argmax(mode))])
        return super().run(x, prev)


FRONT = 0.5 * pl.STEER_SCALE
TWO_WHEEL = (FRONT, 0.0)
FOUR_WHEEL = dv.steering_pair(FRONT)


def mode_driver(p_split=0.5, pad_split=False):
    d = policy_driver(ModeStub(p_split), clear())
    d.pad_split = pad_split
    return d


def ticks_from(d, start, seconds):
    """Tick at 10 Hz over [start, start + seconds); the mode after each."""
    out = []
    for k in range(int(round(seconds * 10))):
        d.tick(start + k * 0.1)
        out.append(d.mode.split)
    return out


def test_the_latch_switches_after_its_dwell_and_not_inside_its_interval():
    latch = pl.ModeLatch(0.7, 0.5, 2.0)
    latch.set(False, 0.0)
    flips = [k for k in range(40) if latch.update(0.9, k * 0.1)]
    assert flips == [20]                 # wanted from 0.0, but the interval is 2 s
    flips = [k for k in range(40, 80) if latch.update(0.0, k * 0.1)]
    assert flips == [45]                 # wanted from 4.0: the interval is past, the dwell is not

    latch.set(False, 0.0)
    assert not any(latch.update(0.9, 10.0 + k * 0.1) for k in range(5))  # 0.4 s
    assert not latch.update(0.5, 10.5)   # one tick of doubt starts the dwell over
    flips = [k for k in range(6, 20) if latch.update(0.9, 10.0 + k * 0.1)]
    assert flips == [11]                 # 0.5 s after it came back
    latch.set(False, 0.0)
    assert not any(latch.update(0.7, 10.0 + k * 0.1) for k in range(50))  # past it, not at it


def test_a_head_that_flickers_never_flaps_the_mode():
    latch = pl.ModeLatch()
    latch.set(False, 0.0)
    assert not any(latch.update(0.95 if k % 2 else 0.05, k * 0.1) for k in range(200))
    # One that always wants the other mode switches no faster than the interval.
    latch.set(False, 0.0)
    flips = [k * 0.1 for k in range(200)
             if latch.update(0.01 if latch.split else 0.99, k * 0.1)]
    assert 0 < len(flips) <= 20 / pl.MODE_MIN_INTERVAL_S
    assert all(b - a >= pl.MODE_MIN_INTERVAL_S - 1e-6 for a, b in zip(flips, flips[1:]))


def test_auto_engages_in_my_mode_and_drives_split_as_two_wheel_steering():
    d = mode_driver(p_split=0.99, pad_split=True)
    d.tick(T0)
    assert d.controller.steerings[-1] == TWO_WHEEL
    assert d.model.modes == [pl.SPLIT]   # the model is told the mode it is in
    fields = d.policy_fields()
    assert fields["policy_steer_mode"] == pl.SPLIT
    assert fields["policy_steer_mode_p"] == pytest.approx(0.99, abs=1e-6)

    d = mode_driver(p_split=0.5, pad_split=False)
    d.tick(T0)
    assert d.controller.steerings[-1] == FOUR_WHEEL
    assert d.model.modes == [pl.UNIFIED]


def test_the_policy_changes_mode_by_itself_only_after_dwell_and_interval():
    d = mode_driver(p_split=0.01, pad_split=True)
    modes = ticks_from(d, T0, 3.0)
    assert modes[:20] == [True] * 20 and modes[20:] == [False] * 10
    assert d.controller.steerings[-1] == FOUR_WHEEL
    assert d.model.modes[19] == pl.SPLIT and d.model.modes[21] == pl.UNIFIED
    assert d.policy_fields()["policy_steer_mode"] == pl.UNIFIED


def test_my_hand_on_the_steering_holds_my_mode_and_hands_back_resynced():
    d = mode_driver(p_split=0.99, pad_split=False)
    d.human_steering = True
    held = ticks_from(d, T0, 3.0)
    assert not any(held)                 # the head wants split for 3 s; my hand says no
    d.pad_split = True                   # L3 under my hand: the latch follows me at once
    d.tick(T0 + 3.0)
    assert d.mode.split
    d.pad_split = False
    d.tick(T0 + 3.1)
    assert not d.mode.split
    d.human_steering = False             # handed back at 3.2: the interval starts again
    after = ticks_from(d, T0 + 3.2, 3.0)
    assert after[:20] == [False] * 20 and after[20] is True


def test_l3_while_the_policy_drives_sets_its_mode_and_restarts_its_interval():
    d = mode_driver(p_split=0.5, pad_split=False)
    assert not any(ticks_from(d, T0, 5.0))
    d.pad_split = True                   # L3
    d.model.set_p(0.01)                  # and the head would rather be unified
    modes = ticks_from(d, T0 + 5.0, 3.0)
    assert modes[:20] == [True] * 20     # my press holds for the whole interval
    assert modes[20] is False
    assert d.controller.steerings[-1] == FOUR_WHEEL


def test_pausing_auto_forgets_the_mode_and_reengages_at_mine():
    d = mode_driver(p_split=0.99, pad_split=False)
    assert ticks_from(d, T0, 3.0)[-1] is True
    d.abort()
    d.model.set_p(0.5)
    d.tick(T0 + 10.0)
    assert not d.mode.split and d.controller.steerings[-1] == FOUR_WHEEL


def test_a_model_with_no_mode_head_drives_unified_whatever_my_mode():
    d = policy_driver(StubModel(STOP, steer=0.5), clear())
    d.pad_split = True
    d.tick(T0)
    assert d.controller.steerings[-1] == FOUR_WHEEL
    fields = d.policy_fields()
    assert fields["policy_steer_mode"] == pl.UNIFIED and fields["policy_steer_mode_p"] == -1.0
    assert pl.PolicyModel(str(FIXTURE)).has_mode is False   # the smoke model is one


def test_a_mode_head_that_is_not_a_number_is_neutral():
    d = mode_driver()
    d.model.out[pl.MODE_OUTPUT] = np.array([[0.0, np.nan]], np.float32)
    d.tick(T0)
    assert neutral(d) and d.last_decision.reason == "model mode not finite"
    d.model.out[pl.MODE_OUTPUT] = np.zeros((1, 3), np.float32)
    d.tick(T0 + 0.1)
    assert neutral(d) and d.last_decision.reason == "model mode shape"
    del d.model.out[pl.MODE_OUTPUT]
    d.tick(T0 + 0.2)
    assert neutral(d) and d.last_decision.reason == "model mode missing"


GRAPH = (["x", "prev", pl.MODE_INPUT], ["logits", "throttle", "steer", pl.MODE_OUTPUT])


def test_a_sidecar_with_the_mode_head_must_match_the_graph():
    meta = {"steer_modes": list(pl.STEER_MODES), "mode_switch": dict(SWITCH)}
    assert pl.check_sidecar(meta, *GRAPH) == SWITCH
    for bad in ({**meta, "steer_modes": [pl.SPLIT, pl.UNIFIED]},
                {**meta, "steer_modes": "unified"},
                {"steer_modes": list(pl.STEER_MODES)},
                {**meta, "mode_switch": {**SWITCH, "threshold": 0.3}},
                {**meta, "mode_switch": {**SWITCH, "min_interval_s": -1}},
                {**meta, "mode_switch": {"threshold": 0.7}}):
        with pytest.raises(ValueError):
            pl.check_sidecar(bad, *GRAPH)
    with pytest.raises(ValueError):
        pl.check_sidecar(meta, ["x", "prev"], ["logits", "throttle", "steer"])


def test_a_sidecar_without_the_mode_head_is_fixed_unified_unless_the_graph_has_one():
    assert pl.check_sidecar({}, ["x", "prev"], ["logits", "throttle", "steer"]) is None
    with pytest.raises(ValueError):
        pl.check_sidecar({}, *GRAPH)


def test_l3_in_policy_auto_reaches_the_model_and_the_footprint_rows(monkeypatch):
    """Through the real control loop: L3 at 1 s switches the policy to split,
    the rear goes straight, and the rows say so; my left stick after that
    takes the steering in my mode."""
    from crawler.autopilot import footprint as fp
    import test_takeover as tk
    from test_record import SessionTele

    def press(n, pad, cam):
        if n == 64:
            pad.down["LS"] = True
        elif n == 65:
            pad.down["LS"] = False

    tele = SessionTele()
    model = ModeStub(p_split=0.5)
    out = run_policy(monkeypatch, model, 192, tk.chain(press, tk.at(160, LEFT_X=0.5)),
                     tele=tele, tap=fp.FootprintTap(tele=tele))
    driver, s = out["driver"], out["snaps"]
    assert s[50]["steering"] == pytest.approx(FOUR_WHEEL)
    assert s[100]["steering"] == pytest.approx(TWO_WHEEL)
    assert driver.pad_split is True and driver.mode.split
    assert s[-1]["steer_by"] == mn.PAD and driver.human_steering
    assert s[-1]["steering"] == pytest.approx((tk.limited(0.5), 0.0))  # my split
    rows = [row for row, _ in tele.of("footprint")]
    modes = [row["policy_steer_mode"] for row in rows]
    assert modes[0] == pl.UNIFIED and modes[-1] == pl.SPLIT
    assert all(row["policy_steer_mode_p"] == pytest.approx(0.5, abs=1e-6) for row in rows)


def test_a_registered_sidecar_loads_and_names_the_model_in_every_row(tmp_path):
    meta = json.loads(FIXTURE.with_suffix(".json").read_text())
    meta.update(model_name="kblock", model_version="1.1",
                train_commit=None, train_command="tools/policy_train.py x y")
    (tmp_path / "kblock-1.1.json").write_text(json.dumps(meta))
    (tmp_path / "kblock-1.1.onnx").write_bytes(FIXTURE.read_bytes())
    # The car loads models/policy.onnx, a symlink: the name comes off the
    # sidecar, not the link.
    (tmp_path / "policy.onnx").symlink_to("kblock-1.1.onnx")
    (tmp_path / "policy.json").symlink_to("kblock-1.1.json")
    model = pl.PolicyModel(str(tmp_path / "policy.onnx"))
    assert model.label == "kblock 1.1"
    d = pl.PolicyDriver(FakeController(), FakeCamera(depth=clear()), model)
    d.tick(T0)
    row = fp.autopilot_fields(d)
    assert row["policy_model"] == "kblock 1.1"
    assert row["policy_model_sha256"] == model.sha256


def test_an_old_sidecar_behind_the_symlink_is_named_by_the_real_file(tmp_path):
    meta = json.loads(FIXTURE.with_suffix(".json").read_text())
    assert "model_name" not in meta
    (tmp_path / "policy-20261028-170131Z.json").write_text(json.dumps(meta))
    (tmp_path / "policy-20261028-170131Z.onnx").write_bytes(FIXTURE.read_bytes())
    (tmp_path / "policy.onnx").symlink_to("policy-20261028-170131Z.onnx")
    (tmp_path / "policy.json").symlink_to("policy-20261028-170131Z.json")
    model = pl.PolicyModel(str(tmp_path / "policy.onnx"))
    assert model.label == "policy-20261028-170131Z"


def test_model_label_needs_both_name_and_version():
    assert pl.model_label({"model_name": "kblock", "model_version": "2.0"}, "/x/a.onnx") == "kblock 2.0"
    assert pl.model_label({"model_name": "kblock"}, "/x/a.onnx") == "a"
    assert pl.model_label({"model_version": "2.0"}, "/x/a.onnx") == "a"


# --- confidence: the less sure the model is, the slower it drives -------------

def sure_of(p, action=pl.FORWARD):
    """Action logits whose softmax gives `action` probability p, the rest even."""
    rest = math.log((1.0 - p) / 2.0)
    return tuple(math.log(p) if a == action else rest for a in pl.ACTIONS)


def set_logits(d, logits):
    d.model.out["logits"] = np.array([logits], dtype=np.float32)


def run_ticks(d, n, start=T0):
    for k in range(n):
        d.tick(start + 0.1 * k)
    return start + 0.1 * n


@pytest.mark.parametrize("action,speed", [(pl.FORWARD, "forward_speed"),
                                          (pl.REVERSE, "reverse_speed")])
def test_a_sure_model_drives_its_throttle_unchanged(action, speed):
    d = policy_driver(StubModel(sure_of(0.95, action), throttle=0.75), clear())
    run_ticks(d, 10)
    assert d.confidence.scale == 1.0
    assert d.controller.throttles[-1] == 0.75 * getattr(d, speed)


def test_an_unsure_model_slows_to_the_speed_floor():
    d = policy_driver(StubModel(sure_of(0.4), throttle=1.0), clear())
    # A cap high enough that the moving floor is not what holds it.
    d.forward_speed = 1.0
    run_ticks(d, 20)
    assert d.confidence.scale == pytest.approx(pl.SPEED_FLOOR)
    assert d.controller.throttles[-1] == pytest.approx(pl.SPEED_FLOOR * 1.0)
    # Half way up the ramp is half way between the floor and full speed.
    assert pl.speed_scale_for((pl.CONF_FLOOR + pl.CONF_FULL) / 2) == pytest.approx(
        (pl.SPEED_FLOOR + 1.0) / 2)


def test_one_unsure_tick_moves_the_scale_one_step_at_most():
    d = policy_driver(StubModel(sure_of(0.99), throttle=1.0), clear())
    d.forward_speed = 1.0
    t = run_ticks(d, 10)
    assert d.confidence.scale == 1.0
    # A stop the goal masks into a forward the model gives 1%: the most
    # unsure a committed action gets.
    set_logits(d, (math.log(0.01), math.log(0.005), math.log(0.985)))
    d.tick(t)
    assert d.last_decision.action == pl.FORWARD and d.last_decision.stop_masked
    assert d.confidence.raw == pytest.approx(0.01, abs=1e-6)
    assert d.confidence.scale == pytest.approx(1.0 - pl.MAX_SCALE_STEP)
    assert d.controller.throttles[-1] == pytest.approx(1.0 - pl.MAX_SCALE_STEP)
    # And back up no faster than it came down.
    for k in range(20):
        d.tick(t + 0.1 * (k + 1))
    set_logits(d, sure_of(0.99))
    scales = [d.confidence.scale]
    assert scales[0] == pytest.approx(pl.SPEED_FLOOR)
    for k in range(20):
        d.tick(t + 5.0 + 0.1 * k)
        scales.append(d.confidence.scale)
    steps = np.diff(scales)
    assert scales[-1] == 1.0
    assert steps.max() == pytest.approx(pl.MAX_SCALE_STEP)
    assert steps.min() >= 0.0


@pytest.mark.parametrize("action,floor", [(pl.FORWARD, pl.MOVING_FLOOR_FORWARD),
                                          (pl.REVERSE, -pl.MOVING_FLOOR_REVERSE)])
def test_slowing_never_takes_the_throttle_under_what_moves_the_car(action, floor):
    # 0.75 of either cap is over its floor, and half of it is under.
    d = policy_driver(StubModel(sure_of(0.4, action), throttle=0.75), clear())
    run_ticks(d, 20)
    assert d.confidence.scale == pytest.approx(pl.SPEED_FLOOR)
    assert d.controller.throttles[-1] == pytest.approx(floor)


def test_a_throttle_already_under_the_moving_floor_is_left_where_it_was():
    d = policy_driver(StubModel(sure_of(0.4), throttle=0.25), clear())
    run_ticks(d, 20)
    assert d.controller.throttles[-1] == pytest.approx(0.25 * d.forward_speed)
    assert pl.slowed_throttle(0.0, 0.5, pl.MOVING_FLOOR_FORWARD) == 0.0


def slowed_driver():
    """A policy driver that has been driving unsure, so the scale is down."""
    d = policy_driver(StubModel(sure_of(0.4), throttle=1.0), clear())
    t = run_ticks(d, 20)
    assert d.confidence.scale == pytest.approx(pl.SPEED_FLOOR)
    return d, t


def test_every_stop_is_immediate_and_unscaled_however_sure_the_model_was():
    from crawler.autopilot import goal as gl

    # Sure and at full speed, then a stop: zero on the very next tick, no slew.
    for stop in ("no frame", "blind", "bad output", "goal reached", "model floor"):
        d = policy_driver(StubModel(sure_of(0.99), throttle=1.0), clear())
        d.goal = gl.Goal()
        t = run_ticks(d, 10)
        assert d.controller.throttles[-1] == pytest.approx(d.forward_speed)
        if stop == "no frame":
            d.camera.depth = None
        elif stop == "blind":
            d.camera.depth = np.zeros((400, 640), dtype=np.uint16)
        elif stop == "bad output":
            d.model.out["throttle"] = np.array([[float("nan")]], dtype=np.float32)
        elif stop == "goal reached":
            d.goal.set_reached(True)
        else:
            d.camera.depth = chair_leg()
        d.tick(t)
        assert d.controller.throttles[-1] == 0.0, stop
        if stop != "model floor":
            assert neutral(d), stop
            # Nothing was committed, so nothing was sure and the scale held.
            assert d.policy_fields()["policy_confidence"] == -1.0, stop
            assert d.confidence.scale == 1.0, stop


def test_a_stuck_recover_backs_out_slowly_not_at_the_reverse_cap():
    """The confidence scale does not touch it and neither does the cap: the
    back-out eases in toward ESCAPE_SPEED, and never commands -1.0."""
    d, t = slowed_driver()
    d.recovering = lambda now: False
    d.stuck = lambda imu, now: True
    d.tick(t)
    first = dv.ESCAPE_SPEED / dv.ESCAPE_RAMP_DECISIONS
    assert d.controller.throttles[-1] == pytest.approx(first)
    assert d.last_decision.reason == "stuck"
    assert d.last_decision.throttle == pytest.approx(first)
    assert d._backout_reason == dv.BACKOUT_STUCK


def test_pausing_auto_starts_the_confidence_over():
    d, t = slowed_driver()
    d.abort()
    assert d.confidence.smoothed is None and d.confidence.scale == 1.0
    set_logits(d, sure_of(0.95))
    d.tick(t + 10.0)
    # Seeded off the first decision, not dragged up from the old EMA.
    assert d.confidence.smoothed == pytest.approx(0.95, abs=1e-6)
    assert d.confidence.scale == 1.0
    assert d.controller.throttles[-1] == pytest.approx(d.forward_speed)


def test_the_confidence_is_logged_in_every_footprint_row(monkeypatch):
    from crawler.autopilot import footprint as fp
    from crawler.telemetry import telemetry as tm
    from test_record import SessionTele

    names = ("policy_confidence", "policy_confidence_smoothed", "policy_speed_scale")
    props = tm.JSON_SCHEMAS["footprint"]["schema"]["properties"]
    for name in names:
        assert name in props and name in fp.AUTOPILOT_FIELDS

    tele = SessionTele()
    out = run_policy(monkeypatch, StubModel(sure_of(0.4), throttle=1.0), 60,
                     lambda n, pad, cam: None, tele=tele, tap=fp.FootprintTap(tele=tele))
    rows = [row for row, _ in tele.of("footprint")]
    assert rows
    for row in rows:
        assert set(row) == set(props)
    last = rows[-1]
    assert last["policy_confidence"] == pytest.approx(0.4, abs=1e-6)
    assert last["policy_confidence_smoothed"] == pytest.approx(0.4, abs=1e-6)
    assert last["policy_speed_scale"] == pytest.approx(pl.SPEED_FLOOR)
    fields = out["driver"].policy_fields()
    assert {k: fields[k] for k in names} == {k: last[k] for k in names}


# --- the standoff with the floor ---------------------------------------------

def floor_held(d, ticks, start=T0):
    """`ticks` decisions 0.1 s apart against the chair leg. The reasons."""
    return drive_through(d, [chair_leg()] * ticks, start)


def test_the_floor_holding_a_forward_for_standoff_s_runs_the_stuck_recover():
    model = StubModel(FORWARD, throttle=1.0, steer=0.3)
    d = policy_driver(model, chair_leg())
    n = round(pl.STANDOFF_S / 0.1)
    reasons = floor_held(d, n + 1)
    assert reasons == ["model floor"] * n + ["standoff"]
    assert d._phase == dv._PHASE_REVERSE
    assert d._backout_reason == dv.BACKOUT_STANDOFF
    assert dv.ESCAPE_SPEED <= d.controller.throttles[-1] < 0
    assert d.last_verdict.reason == "policy standoff"
    assert d.policy_fields()["standoff_ticks"] == n + 1
    assert d.last_recovery_time == pytest.approx(T0 + pl.STANDOFF_S)
    # The recover owns the wheels, then the model has them back.
    calls = len(model.calls)
    now = T0 + pl.STANDOFF_S
    for _ in range(MAX_BACKOUT_TICKS):
        now += 0.1
        d.tick(now)
        if len(model.calls) > calls:
            break
        assert d.policy_fields()["standoff_ticks"] == 0
    assert len(model.calls) > calls, "the back-out never handed back"
    # The settle ends and the same tick asks the model.
    assert now - (T0 + pl.STANDOFF_S) == pytest.approx(
        dv.REVERSE_SECONDS + dv.SETTLE_SECONDS, abs=0.25)  # a tick each phase
    assert d.last_decision.reason == "model floor"
    assert d.standoff_ticks == 1


MOVING_GYRO = {"accel": (0.0, 0.0, 0.0), "gyro": (0.5, 0.0, 0.0)}
# Bounds every loop that waits for a back-out to hand back: the longest is the
# escalated cap plus the settle, plus a few ticks of slack.
MAX_BACKOUT_TICKS = round(
    (dv.ESCALATED_REVERSE_SECONDS + dv.SETTLE_SECONDS) / 0.1) + 5


def standoff_back_out(model):
    """A policy driver into a standoff with the chair leg, its back-out
    started. The driver and the time of the standoff."""
    d = policy_driver(model, chair_leg(), imu=MOVING_GYRO)
    n = round(pl.STANDOFF_S / 0.1)
    assert floor_held(d, n + 1)[-1] == "standoff"
    return d, T0 + n * 0.1


def test_a_standoff_back_out_ends_when_the_chair_leg_clears():
    """Not the old fixed two seconds: the scene opens, three decisions later
    the reverse stops, and the model has the car back after the settle --
    leaning toward the side that opened, on top of its own steer."""
    model = StubModel(FORWARD, throttle=1.0, steer=0.0)
    d, now = standoff_back_out(model)
    side = d._backout_side
    d.camera.depth = clear()
    calls = len(model.calls)
    throttles = []
    for _ in range(MAX_BACKOUT_TICKS):
        now = round(now + 0.1, 3)
        d.tick(now)
        throttles.append(d.controller.throttles[-1])
        if len(model.calls) > calls:
            break
    assert len(model.calls) > calls, "the back-out never handed back"
    assert d._backout_exit == dv.EXIT_OPENED
    assert d._backout_reverse_s < dv.REVERSE_SECONDS
    assert d._backout_reverse_s == pytest.approx(dv.BACKOUT_MIN_S, abs=0.11)
    assert min(throttles) == pytest.approx(dv.ESCAPE_SPEED)
    assert all(t >= dv.ESCAPE_SPEED for t in throttles)
    # Handed back: the model's forward at full, leaning the other way from the
    # reverse.
    assert d.last_decision.action == pl.FORWARD
    assert d.controller.throttles[-1] > 0.0
    lean = -side * dv.HANDBACK_STEER * d.max_turn
    assert d.last_decision.front == pytest.approx(lean)
    assert d.controller.steerings[-1] == pytest.approx(dv.steering_pair(lean))


def test_the_floor_veto_watches_the_back_out():
    """Every back-out scan reaches the policy's floor veto, and every throttle
    it commands is credited, so a latch can let go during the back-out."""
    d, now = standoff_back_out(StubModel(FORWARD, throttle=1.0))
    d.camera.depth = clear()
    d.tick(round(now + 0.1, 3))
    assert d.floor_veto.scan is d.last_scan
    assert d.floor_veto._throttle == pytest.approx(d._backout_throttle)
    assert d._backout_throttle < 0.0


def test_the_back_out_fields_reach_the_footprint_row():
    d, now = standoff_back_out(StubModel(FORWARD, throttle=1.0))
    d.tick(round(now + 0.1, 3))
    row = fp.autopilot_fields(d, round(now + 0.1, 3))
    assert row["autopilot"] == "policy"
    assert row["backout_phase"] == dv._PHASE_REVERSE
    assert row["backout_reason"] == dv.BACKOUT_STANDOFF
    assert row["backout_speed"] < 0.0
    assert row["backout_elapsed_s"] == pytest.approx(0.1)
    assert row["backout_side"] in (-1.0, 1.0)
    rules = fp.autopilot_fields(dv.Driver(FakeController(), FakeCamera()), 0.0)
    assert rules["autopilot"] == "rules" and rules["backout_phase"] == ""


def test_a_floor_hold_shorter_than_standoff_s_does_not_recover():
    d = policy_driver(StubModel(FORWARD), chair_leg())
    n = round(pl.STANDOFF_S / 0.1)
    assert floor_held(d, n) == ["model floor"] * n
    # A tick the floor lets go starts the count over.
    d.camera.depth = clear()
    d.tick(T0 + n * 0.1)
    assert d.last_decision.reason == "model" and d.standoff_ticks == 0
    assert floor_held(d, n, start=T0 + (n + 1) * 0.1) == ["model floor"] * n
    assert d._phase is None and d.last_recovery_time == 0.0


def test_a_floor_hold_under_my_throttle_is_never_a_standoff():
    d = policy_driver(StubModel(FORWARD), chair_leg())
    d.human_throttle = True
    floor_held(d, 30)
    assert d._phase is None and d.standoff_ticks == 0


# --- the reverse commit --------------------------------------------------------

def commit_ticks():
    return round(pl.REVERSE_COMMIT_S / 0.1)


def test_a_fresh_reverse_is_held_for_the_commit_even_when_the_model_says_forward():
    model = StubModel(REVERSE, throttle=1.0, steer=0.0)
    d = policy_driver(model, clear())
    d.tick(T0)
    assert d.reverse_commit
    model.out["logits"] = np.array([FORWARD], dtype=np.float32)
    n = commit_ticks()
    for k in range(1, n):
        d.tick(T0 + k * 0.1)
        assert d.reverse_commit, k
        assert d.controller.throttles[-1] < 0.0
        assert d.last_decision.reason == "model commit"
        assert d.policy_fields()["policy_action"] == pl.FORWARD
        assert d.policy_fields()["reverse_commit"] is True
    assert len(model.calls) == n
    d.tick(T0 + n * 0.1)
    assert not d.reverse_commit
    # Forward, slowed: the model was not sure of the reverse it was held to.
    assert 0.0 < d.controller.throttles[-1] <= d.forward_speed
    assert d.policy_fields()["reverse_commit"] is False


def test_a_reverse_after_a_reverse_is_not_a_fresh_commit():
    model = StubModel(REVERSE, throttle=1.0, steer=0.1)
    d = policy_driver(model, clear())
    for k in range(commit_ticks() + 3):
        d.tick(T0 + k * 0.1)
    assert not d.reverse_commit
    assert d.controller.steerings[-1] == pytest.approx(dv.steering_pair(0.1 * pl.STEER_SCALE))


def test_the_commit_reverses_away_from_the_forward_turn_it_came_out_of():
    for steer in (0.5, -0.5):
        model = StubModel(FORWARD, throttle=1.0, steer=steer)
        d = policy_driver(model, clear())
        d.tick(T0)
        d.tick(T0 + 0.1)
        # The model's own reverse steer is the turn's side, and loses.
        model.out["logits"] = np.array([REVERSE], dtype=np.float32)
        d.tick(T0 + 0.2)
        want = -math.copysign(pl.REVERSE_STEER, steer)
        assert d.controller.steerings[-1] == pytest.approx(dv.steering_pair(want))
        assert d.last_decision.front == pytest.approx(want)


def test_the_commit_side_rule():
    f = pl.reverse_commit_front
    lock = pl.REVERSE_STEER
    # A recent turn: the other side, whatever the model says.
    assert f(0.35, 0.2, 0.5) == pytest.approx(-lock)
    assert f(-0.35, 0.2, -0.5) == pytest.approx(lock)
    # A turn too old, or too small, is no turn: the model's side.
    assert f(0.35, pl.REVERSE_TURN_WINDOW_S + 0.1, 0.3) == pytest.approx(lock)
    assert f(0.1, 0.2, -0.3) == pytest.approx(-lock)
    # Nothing either way: the default side.
    assert f(None, math.inf, 0.05) == pytest.approx(pl.REVERSE_DEFAULT_SIDE * lock)


def test_the_commit_steers_in_the_mode_the_policy_is_in():
    d = policy_driver(StubModel(REVERSE, throttle=1.0, steer=0.5), clear())
    d.tick(T0)
    assert d.controller.steerings[-1] == pytest.approx(dv.steering_pair(pl.REVERSE_STEER))
    # Split as I drive it: the rear straight.
    d = policy_driver(StubModel(REVERSE, throttle=1.0, steer=0.5), clear())
    d._sync_mode = lambda now: None
    d.mode.split = True
    d.tick(T0)
    assert d.controller.steerings[-1] == pytest.approx((pl.REVERSE_STEER, 0.0))


def test_the_confidence_still_slows_the_commit():
    model = StubModel(REVERSE, throttle=1.0)
    d = policy_driver(model, clear())
    d.tick(T0)
    model.out["logits"] = np.array([FORWARD], dtype=np.float32)  # p(reverse) tiny
    for k in range(1, commit_ticks()):
        d.tick(T0 + k * 0.1)
    assert d.reverse_commit
    assert d.confidence.scale < 1.0
    assert d.controller.throttles[-1] == pytest.approx(
        pl.slowed_throttle(d.reverse_speed, d.confidence.scale, pl.MOVING_FLOOR_REVERSE))


class ReachedLater:
    def __init__(self):
        self.now = False

    def reached(self):
        return self.now


STILL_IMU = {"accel": (0.0, 0.0, 0.0), "gyro": (0.001, 0.001, 0.001)}


def _no_frame(d):
    d.camera.depth = None


def _blind(d):
    d.camera.depth = np.zeros((400, 640), dtype=np.uint16)


def _bad_output(d):
    d.model.out["throttle"] = np.array([[math.nan]], dtype=np.float32)


def _goal(d):
    d.goal.now = True


def _stuck(d):
    d.stuck = lambda imu, now: True


@pytest.mark.parametrize("name, stop, reason", [
    ("no frame", _no_frame, "no frame"),
    ("blind", _blind, "blind"),
    ("bad output", _bad_output, "model not finite"),
    ("goal", _goal, "goal reached"),
    ("stuck", _stuck, "stuck"),
])
def test_every_safety_stop_preempts_the_commit(name, stop, reason):
    d = pl.PolicyDriver(FakeController(), FakeCamera(depth=clear()),
                        StubModel(REVERSE, throttle=1.0), goal=ReachedLater())
    d.tick(T0)
    d.tick(T0 + 0.1)
    assert d.reverse_commit
    stop(d)
    d.tick(T0 + 0.2)
    assert not d.reverse_commit
    assert d.last_decision.reason == reason
    if name == "stuck":
        assert d._phase == dv._PHASE_REVERSE
    else:
        assert neutral(d)


def test_the_rear_bumper_cuts_a_commit_long_enough_for_it(monkeypatch):
    # At 0.7 s the bumper cannot speak (it needs 0.9 s of reverse); a commit
    # I raise past that gets the recover's bumper.
    monkeypatch.setattr(pl, "REVERSE_COMMIT_S", 3.0)
    d = policy_driver(StubModel(REVERSE, throttle=1.0), clear(), imu=STILL_IMU)
    for k in range(20):
        d.tick(T0 + k * 0.1)
        if d.last_decision.reason != "model commit":
            break
    assert d.last_decision.reason == "rear bumper"
    # The first tick's sample is cleared, five fill the window, four confirm.
    assert k == d.imu_window_size + dv.REVERSE_STALL_SAMPLES - 1
    assert not d.reverse_commit and neutral(d)


@pytest.mark.parametrize("group", ["human_throttle", "human_steering"])
def test_my_hand_on_either_group_ends_the_commit(group):
    model = StubModel(REVERSE, throttle=1.0)
    d = policy_driver(model, clear())
    d.tick(T0)
    model.out["logits"] = np.array([FORWARD], dtype=np.float32)
    d.tick(T0 + 0.1)
    assert d.reverse_commit
    setattr(d, group, True)
    d.tick(T0 + 0.2)
    assert not d.reverse_commit
    # The model's own answer goes out under my hand, which the pad mixes over.
    assert d.controller.throttles[-1] > 0.0
    # And when I let go it is the model's, with no commit left to resume.
    setattr(d, group, False)
    d.tick(T0 + 0.3)
    assert not d.reverse_commit
    assert d.controller.throttles[-1] > 0.0


def test_my_trigger_wins_over_the_commit_through_the_control_loop(monkeypatch):
    import test_takeover as tk
    out = run_policy(monkeypatch, StubModel(REVERSE, throttle=1.0), 60,
                     tk.at(5, throttle=0.5))
    assert out["snaps"][4]["throttle"] < 0.0
    for snap in out["snaps"][6:]:
        assert snap["throttle_by"] == mn.PAD
        assert snap["throttle"] == pytest.approx(0.5 * out["driver"].forward_speed)
    assert not out["driver"].reverse_commit


def test_the_commit_and_the_standoff_are_logged_in_every_footprint_row(monkeypatch):
    from crawler.autopilot import footprint as fp
    from crawler.telemetry import telemetry as tm
    import test_takeover as tk
    from test_record import SessionTele

    props = tm.JSON_SCHEMAS["footprint"]["schema"]["properties"]
    assert props["reverse_commit"]["type"] == "boolean"
    assert props["standoff_ticks"]["type"] == "integer"
    assert fp.AUTOPILOT_FIELDS["reverse_commit"] is False
    assert fp.AUTOPILOT_FIELDS["standoff_ticks"] == 0

    tele = SessionTele()
    run_policy(monkeypatch, StubModel(REVERSE, throttle=1.0), 200,
               lambda n, pad, cam: None, tele=tele, tap=fp.FootprintTap(tele=tele))
    rows = [row for row, _ in tele.of("footprint")]
    assert rows and all(set(row) == set(props) for row in rows)
    flags = [row["reverse_commit"] for row in rows]
    assert True in flags and flags[-1] is False

    tele = SessionTele()
    tk.run_auto(monkeypatch, 80, tk.Camera(chair_leg()), tele=tele,
                tap=fp.FootprintTap(tele=tele),
                make_driver=lambda command, cam: pl.PolicyDriver(
                    command, cam, StubModel(FORWARD)))
    counts = [row["standoff_ticks"] for row, _ in tele.of("footprint")]
    assert max(counts) >= 2
    assert all(isinstance(c, int) and not isinstance(c, bool) for c in counts)
