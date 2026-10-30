"""The lidar as the near-field sensor at the front: under the stereo pair's
clamp the creep's stop comes off the roof lidar's front arc, when the lidar
sees the thing the camera does, and off the camera as before when it does not.
"""

import numpy as np
import pytest

pytest.importorskip("scipy")

from crawler import __main__ as mn  # noqa: E402
from crawler.autopilot import driver as dv  # noqa: E402
from crawler.autopilot import policy as pl  # noqa: E402
from crawler.hardware import lidar as ld  # noqa: E402

import test_policy_map as tpm  # noqa: E402

TICK = 0.1


def bins_with(front_off_bumper=None, bearing=0.0, far=2500):
    """A binned scan: far everywhere, and one return `front_off_bumper` mm
    past the front bumper straight along `bearing` (x off the lidar)."""
    b = np.full(ld.BINS, far, np.uint16)
    if front_off_bumper is not None:
        x = ld.FRONT_BUMPER_X_MM + front_off_bumper
        k = int(round(bearing / ld.BIN_DEG)) % ld.BINS
        b[k] = int(round(x / np.cos(np.radians(bearing))))
    return b


def wall_bins(front_off_bumper, far=2500):
    """A flat wall square to the car, front_off_bumper past the bumper: every
    bin whose ray meets it ahead reads its range along that ray."""
    b = np.full(ld.BINS, far, np.uint16)
    x = ld.FRONT_BUMPER_X_MM + front_off_bumper
    for k, brg in enumerate(ld.bin_bearings()):
        if abs(brg) < 60.0:
            b[k] = int(round(x / np.cos(np.radians(brg))))
    return b


def probe_frame(mm, probe=4):
    """A depth frame with one probe at `mm` (0 is unmeasurable) and open
    floor at 3 m across the rest of the fan."""
    frame = np.full((400, 640), 3000, dtype=np.uint16)
    edges = dv.footprint_geometry(frame.shape, dv.CLEARANCE_MM).col_edges
    frame[:, edges[probe]:edges[probe + 1]] = mm
    return frame


# --- the line --------------------------------------------------------------

def test_the_lidar_line_is_the_measured_overrun_its_age_and_the_margin():
    """Off the front bumper: the creep's measured overrun, the extra travel a
    lidar scan's age can add over a camera frame's, at the creep's speed, and
    the same 40 mm margin. Nearer than the camera's own line, which the
    stereo clamp holds where it was."""
    extra = ((dv.LIDAR_REVOLUTION_S + dv.LIDAR_NEAR_MAX_AGE_S)
             - (dv.FRAME_AGE_S + dv.DECISION_S)) * dv.ground_speed_mm_s(dv.CREEP_SPEED)
    assert dv.LIDAR_EXTRA_AGE_MM == pytest.approx(max(0.0, extra))
    assert dv.LIDAR_CREEP_STOP_CLEAR_MM == pytest.approx(
        dv.CREEP_OVERRUN_MM + dv.LIDAR_EXTRA_AGE_MM + dv.CREEP_MARGIN_MM)
    assert dv.LIDAR_CREEP_STOP_CLEAR_MM == pytest.approx(124.7, abs=0.1)
    assert dv.LIDAR_CREEP_STOP_CLEAR_MM < dv.CREEP_STOP_CLEAR_MM
    # Under the clamp: the camera cannot measure there, so this is the
    # lidar's line alone.
    assert dv.LIDAR_CREEP_STOP_CLEAR_MM < dv.STEREO_NEAR_MM - dv.LENS_TO_FRONT_BUMPER_MM
    # The worst overrun past it still leaves the margin.
    end = dv.LIDAR_CREEP_STOP_CLEAR_MM - dv.CREEP_OVERRUN_MM - dv.LIDAR_EXTRA_AGE_MM
    assert end >= dv.CREEP_MARGIN_MM - 1e-9
    # The camera's line has not moved, and the switch is off by default.
    assert dv.CREEP_FLOOR_MM == pytest.approx(178.0)
    assert dv.LIDAR_NEAR_FIELD is False


# --- the lidar's front arc ----------------------------------------------------

def test_the_front_band_clearance_reads_a_wall_ahead_off_the_bumper():
    c = dv.front_band_clearance_mm(wall_bins(140), -dv.FRONT_PATH_HALF_MM,
                                   dv.FRONT_PATH_HALF_MM)
    # The nearest edge of a 5 degree bin is taken, so a little under, never over.
    assert 130.0 <= c <= 140.5
    # 0 is a bin with no return: nothing ahead.
    assert dv.front_band_clearance_mm(bins_with(None, far=0), -150, 150) == np.inf
    # A probe's lane at the clamp is narrower than a bin's arc there, and is
    # still read.
    lane = dv.probe_band_y(2, dv.STEREO_NEAR_MM)
    assert lane[1] - lane[0] < 30
    # (A bin's range is a whole mm, so up to half a mm over.)
    assert 130.0 <= dv.front_band_clearance_mm(wall_bins(140), *lane) <= 141.0
    assert dv.front_band_clearance_mm(None, -150, 150) == np.inf


def test_returns_inside_the_car_and_beside_it_are_not_ahead_of_it():
    # Inside the bodywork: nearer than the bumper straight ahead.
    inside = np.full(ld.BINS, 2500, np.uint16)
    inside[0] = int(ld.FRONT_BUMPER_X_MM - 30)
    assert dv.front_band_clearance_mm(inside, -150, 150) > 1000
    # Beside the car at 90 degrees, and behind it.
    beside = bins_with(None)
    beside[int(90 / ld.BIN_DEG)] = 300
    beside[int(180 / ld.BIN_DEG)] = 300
    assert dv.front_band_clearance_mm(beside, -150, 150) > 1000
    # Ahead but out past the band to the left: not in a band on the right.
    left = bins_with(100, bearing=25.0)
    assert dv.front_band_clearance_mm(left, -150, -50) > 1000
    assert dv.front_band_clearance_mm(left, 50, 300) < 150


def test_the_probe_bands_tile_the_path_left_to_right():
    """Probe 0 is the image's left, the car's left (+y); the nine tile the
    footprint the camera's span is sized to, scaled to the distance read."""
    z = dv.CLEARANCE_MM
    bands = [dv.probe_band_y(i, z) for i in range(dv.PROBE_COLUMNS)]
    assert bands[0][1] == pytest.approx(dv.FRONT_PATH_HALF_MM)
    assert bands[-1][0] == pytest.approx(-dv.FRONT_PATH_HALF_MM)
    for (lo, hi), (lo2, hi2) in zip(bands, bands[1:]):
        assert lo == pytest.approx(hi2) and lo > lo2
    assert dv.probe_band_y(4, z / 2)[1] == pytest.approx(bands[4][1] / 2)


# --- the creep's veto with the lidar ------------------------------------------

def creep(frames, bins):
    """Each frame through the real probe and the latch; the near-field creep
    veto's answer after each, against `bins` (one per frame, or one for all)."""
    veto = dv.ForwardFloorVeto(forward_speed=dv.FORWARD_SPEED)
    said = []
    for k, f in enumerate(frames):
        veto.observe(dv.free_by_column(probe_frame(int(f))), k * TICK)
        b = bins[k] if isinstance(bins, list) else bins
        said.append(veto.creep_vetoes(b))
    return said, veto


def test_at_the_clamp_the_lidar_lets_the_creep_go_on_to_its_own_line():
    clamp = dv.STEREO_NEAR_MM
    said, _ = creep([300, 200, clamp], wall_bins(140))
    assert said == [False, False, False], "the camera's 150 would hold the last"
    said, _ = creep([300, 200, clamp], wall_bins(120))
    assert said[-1] is True, "inside the lidar's line"
    # Without the lidar it is the camera's veto exactly.
    said, veto = creep([300, 200, clamp], None)
    assert said == [False, False, True]
    assert veto.vetoes(floor_mm=dv.CREEP_FLOOR_MM)


def test_a_probe_latched_blind_lets_go_to_the_lidar_only_while_it_agrees():
    said, veto = creep([300, 200, 0, 0], wall_bins(135))
    assert veto.latch.latched[4], "the blind latch is still the camera's"
    assert said[2:] == [False, False]
    said, _ = creep([300, 200, 0, 0], wall_bins(118))
    assert said[2:] == [True, True]


def test_something_under_the_scan_plane_keeps_the_camera_line():
    """The camera reads a thing at its clamp the lidar does not see: a low
    rock under the roof's scan plane, with the wall behind it 400 mm out.
    The lidar is no evidence about the rock, so the camera holds."""
    said, _ = creep([300, 200, dv.STEREO_NEAR_MM], wall_bins(400))
    assert said[-1] is True
    said, _ = creep([300, 200, 0, 0], wall_bins(400))
    assert said[2:] == [True, True]
    # Agreement is within LIDAR_AGREE_MM of the camera's own clearance.
    cam = dv.STEREO_NEAR_MM - dv.LENS_TO_FRONT_BUMPER_MM
    said, _ = creep([300, 200, dv.STEREO_NEAR_MM], wall_bins(cam + dv.LIDAR_AGREE_MM + 5))
    assert said[-1] is True
    said, _ = creep([300, 200, dv.STEREO_NEAR_MM], wall_bins(cam + dv.LIDAR_AGREE_MM - 5))
    assert said[-1] is False


def test_the_lidar_holds_a_creep_the_camera_would_let_through():
    """Something at cab height in the path the camera's band misses."""
    said, _ = creep([3000, 3000], bins_with(100))
    assert said == [True, True]
    said, _ = creep([3000, 3000], bins_with(300))
    assert said == [False, False]


def test_no_picture_is_still_a_veto():
    veto = dv.ForwardFloorVeto()
    assert veto.creep_vetoes(wall_bins(500))
    assert veto.creep_vetoes(None)


# --- through the policy --------------------------------------------------------

class FrontLidar:
    """A LidarReader stand-in whose scan is `age` old, on its own clock."""

    def __init__(self, bins, age=0.0):
        self.bins, self.age = bins, age

    def latest(self, now=None, max_age=ld.STALE_S):
        if self.age > max_age:
            return None
        return ld.Scan(t=0.0, angles_deg=np.zeros(0), distances_mm=np.zeros(0, np.uint16),
                       intensities=np.zeros(0, np.uint8), speed_dps=3570.0, packets=40,
                       bins=self.bins)


def _at_the_clamp(lidar, near):
    wall = tpm.clear()
    wall[150:350, 100:540] = int(dv.STEREO_NEAR_MM)
    slam = tpm.FakeSlam()
    d = tpm.driver(tpm.MapStub(tpm.FORWARD, throttle=0.8, steer=0.7), slam,
                   depth=wall, lidar=lidar)
    d.lidar_near = near
    for k in range(10):
        slam.at(tpm.T0 + k * 0.1)
        d.tick(tpm.T0 + k * 0.1)
    return d


def test_a_map_models_forward_at_the_clamp_creeps_on_the_lidar_with_the_switch():
    d = _at_the_clamp(FrontLidar(wall_bins(145)), near=True)
    assert d.last_decision.reason == pl.CREEP_REASON
    assert d.controller.throttles[-1] == pytest.approx(pl.CREEP_THROTTLE)
    assert d.rear_fields()["front_lidar_clearance_mm"] == pytest.approx(145, abs=12)
    # Off, as deployed: the camera's line holds it.
    d = _at_the_clamp(FrontLidar(wall_bins(145)), near=False)
    assert d.last_decision.reason == "model floor"
    assert d.controller.throttles[-1] == 0
    # A scan older than the near field allows is no scan: the camera holds.
    d = _at_the_clamp(FrontLidar(wall_bins(145), age=dv.LIDAR_NEAR_MAX_AGE_S + 0.05),
                      near=True)
    assert d.last_decision.reason == "model floor"
    # And inside the lidar's own line it holds.
    d = _at_the_clamp(FrontLidar(wall_bins(110)), near=True)
    assert d.last_decision.reason == "model floor"
    assert d.controller.throttles[-1] == 0


def test_the_switch_is_a_flag_that_reaches_the_driver():
    assert mn.parse_args([]).lidar_near is False
    assert mn.parse_args(["--lidar", "--lidar-near"]).lidar_near is True
    pilot = mn.AutoPilot(None, None, lidar_near=True)
    assert pilot.lidar_near is True
