"""The driver's safety windows in seconds, against the counts they replaced.

STUCK, the rear bumper, the smoothing window, the back-out's open run and its
ease-in were all counts of decisions, tuned at the 0.1 s decision. They are
spans of time now (driver "the decision clock"). Two promises, both checked
here by running the old count logic and the driver side by side on the same
sample streams:

  at exactly 0.1 s a step, every decision comes out the same as the count's,
  on any stream at all, missing IMU packets included;

  on jittered steps of 90-200 ms (the measured spread), nothing fires later
  than the count did, and nothing fires on less than its window of time.

The old logic below is the code as it was, cut down to what decides.
"""

import math
import random

import numpy as np
import pytest

from crawler.autopilot import driver as dv
from conftest import FakeCamera, FakeController


T0 = 100.0


def imu(g, a=0.0):
    return {"accel": (a, 0.0, 0.0), "gyro": (g, 0.0, 0.0)}


def clear_depth(mm=3000):
    return np.full((400, 640), mm, dtype=np.uint16)


def blocked_depth(mm=200):
    return np.full((400, 640), mm, dtype=np.uint16)


# --- the count logic, as it was -----------------------------------------------

class CountEvidence:
    """motion_evidence() and the two counters before the decision clock."""

    WINDOW = 5

    def __init__(self, stuck_threshold=0.2):
        self.gyro = []
        self.accel = []
        self.still = 0
        self.reverse_still = 0
        self.stuck_threshold = stuck_threshold

    def evidence(self, sample):
        if sample is None:
            return dv.UNKNOWN
        self.accel.append(float(np.linalg.norm(sample["accel"])))
        self.accel = self.accel[-self.WINDOW:]
        self.gyro.append(float(np.linalg.norm(sample["gyro"])))
        self.gyro = self.gyro[-self.WINDOW:]
        if (np.mean(self.gyro) > dv.GYRO_MOVING_MIN
                or np.mean(self.accel) > self.stuck_threshold):
            return dv.MOVING
        return dv.STILL

    def stuck(self, sample, now, armed=True, cooled=True):
        ev = self.evidence(sample)
        if ev == dv.MOVING or not armed:
            self.still = 0
            return False
        if ev == dv.STILL:
            self.still += 1
        if self.still < dv.STUCK_CONFIRM_SAMPLES or not cooled:
            return False
        self.still = 0
        return True

    def bumper(self, sample):
        ev = self.evidence(sample)
        if len(self.gyro) < self.WINDOW:
            return False
        if ev != dv.UNKNOWN:
            shaking = (np.mean(self.gyro) > dv.REVERSE_GYRO_STILL_MAX
                       or np.mean(self.accel) > self.stuck_threshold)
            ev = dv.MOVING if shaking else dv.STILL
        if ev == dv.MOVING:
            self.reverse_still = 0
            return False
        if ev == dv.STILL:
            self.reverse_still += 1
        return self.reverse_still >= dv.REVERSE_STALL_SAMPLES


def count_backout(samples, opens, times, cap_until, min_s):
    """The back-out's reverse as the counts ran it, from begin_recover at
    times[0]: the throttle on each decision and (decision, exit) at the end,
    or (None, "") if it never ended. No lidar, a frame every decision."""
    ev = CountEvidence()
    throttles = []
    decisions = 1
    throttles.append(dv.ESCAPE_SPEED * min(1.0, 1 / dv.ESCAPE_RAMP_DECISIONS))
    open_run = 0
    for i in range(1, len(times)):
        now = times[i]
        hit = ev.bumper(samples[i])
        open_run = open_run + 1 if opens[i] else 0
        if hit:
            exit_reason = dv.EXIT_BUMPER
        elif now >= cap_until:
            exit_reason = dv.EXIT_CAP
        elif (open_run >= dv.BACKOUT_OPEN_DECISIONS
                and now - times[0] >= min_s - 1e-3):
            exit_reason = dv.EXIT_OPENED
        else:
            decisions += 1
            throttles.append(dv.ESCAPE_SPEED
                             * min(1.0, decisions / dv.ESCAPE_RAMP_DECISIONS))
            continue
        throttles.append(0)
        return throttles, (i, exit_reason)
    return throttles, (None, "")


# --- the driver, on the same streams -------------------------------------------

def driver_stuck(samples, times):
    """Driver.stuck() over the stream, armed and cooled from the start, the
    caller stamping the cooldown on a fire as tick() does."""
    d = dv.Driver(FakeController(), FakeCamera())
    d._throttle_since = 0.0
    d.last_recovery_time = -1e9
    out = []
    for s, t in zip(samples, times):
        fired = d.stuck(s, t)
        out.append(fired)
        if fired:
            d.last_recovery_time = t
    return out


def count_stuck(samples, times):
    ev = CountEvidence()
    last = -1e9
    out = []
    for s, t in zip(samples, times):
        fired = ev.stuck(s, t, cooled=t - last > 5.0)
        out.append(fired)
        if fired:
            last = t
    return out


def driver_backout(samples, opens, times, reason=dv.BACKOUT_OBSTACLE):
    d = dv.Driver(FakeController(), FakeCamera())
    d.camera.imu = samples[0]
    d.camera.depth = clear_depth() if opens[0] else blocked_depth()
    d.begin_recover(times[0], reason)
    cap_until, min_s = d._phase_until, d._backout_min_s
    end = (None, "")
    for i in range(1, len(times)):
        d.camera.imu = samples[i]
        d.camera.depth = clear_depth() if opens[i] else blocked_depth()
        d.recovering(times[i])
        if d._phase != dv._PHASE_REVERSE:
            end = (i, d._backout_exit)
            break
    return list(d.controller.throttles), end, cap_until, min_s


# --- the streams ------------------------------------------------------------------

def exact_times(n, start=T0):
    """0.1 s steps the way a running clock adds them up: float sums, not
    rounded, so the windows' slack is what keeps them on the count."""
    t, out = start, []
    for _ in range(n):
        out.append(t)
        t += dv.DECISION_S
    return out


def jittered_times(n, rng, lo=0.09, hi=0.2):
    t, out = T0, []
    for _ in range(n):
        out.append(t)
        t += rng.uniform(lo, hi)
    return out


def measured_times(n, rng):
    """Shaped like the measured decision periods: p50 about 110 ms, p95 about
    170 ms, nothing under 90 or over 200 (the spread the windows are set on)."""
    t, out = T0, []
    for _ in range(n):
        out.append(t)
        t += min(0.2, max(0.09, rng.lognormvariate(math.log(0.11), 0.2)))
    return out


def noisy_stream(n, rng, p_unknown):
    """Anything at all: gyro and accel either side of their lines, so the
    verdict flips, and missing packets. Only exact equality survives this."""
    out = []
    for _ in range(n):
        if rng.random() < p_unknown:
            out.append(None)
        else:
            out.append(imu(rng.uniform(0.0, 0.03), rng.uniform(0.0, 0.35)))
    return out


def stop_stream(n, rng, p_unknown, moving=(0.015, 0.1), still=(0.0, 0.006)):
    """A car rolling, then stopped: rolling gyro over the line, parked gyro
    under it (the measured parked floor tops out at 0.0069)."""
    roll = rng.randrange(0, 25)
    out = []
    for k in range(n):
        if rng.random() < p_unknown:
            out.append(None)
            continue
        lo, hi = moving if k < roll else still
        out.append(imu(rng.uniform(lo, hi), rng.uniform(0.0, 0.05)))
    return out


def first(flags):
    return next((i for i, f in enumerate(flags) if f), None)


SEEDS = range(200)


# --- at exactly 0.1 s: the same, decision for decision ------------------------------

@pytest.mark.parametrize("p_unknown", [0.0, 0.15, 0.4])
def test_stuck_at_the_tuned_step_is_the_count_on_every_decision(p_unknown):
    for seed in SEEDS:
        rng = random.Random(seed)
        samples = (noisy_stream(60, rng, p_unknown)
                   + stop_stream(60, rng, p_unknown))
        times = exact_times(len(samples))
        assert driver_stuck(samples, times) == count_stuck(samples, times), seed


@pytest.mark.parametrize("p_unknown", [0.0, 0.15, 0.4])
def test_the_evidence_at_the_tuned_step_is_the_count_on_every_decision(p_unknown):
    """The smoothing window itself: same verdict, same samples in it."""
    for seed in SEEDS:
        rng = random.Random(seed)
        samples = noisy_stream(80, rng, p_unknown)
        d = dv.Driver(FakeController(), FakeCamera())
        ev = CountEvidence()
        for s, t in zip(samples, exact_times(len(samples))):
            d._evidence_now = t
            got = d.motion_evidence(s)
            d._evidence_now = None
            assert got == ev.evidence(s), seed
            assert d.gyro_history == ev.gyro, seed


@pytest.mark.parametrize("p_unknown", [0.0, 0.15, 0.4])
@pytest.mark.parametrize("reason", [dv.BACKOUT_OBSTACLE, dv.BACKOUT_STUCK])
def test_a_back_out_at_the_tuned_step_is_the_count_on_every_decision(p_unknown, reason):
    """Bumper, open run, cap and ease-in together: the same throttle on
    every decision and the same exit on the same decision."""
    for seed in SEEDS:
        rng = random.Random(seed)
        n = 45
        if seed % 3 == 0:
            samples = stop_stream(n, rng, p_unknown, moving=(0.33, 0.6),
                                  still=(0.0, 0.08))
        else:
            samples = [None if rng.random() < p_unknown
                       else imu(rng.uniform(0.0, 0.4), rng.uniform(0.0, 0.3))
                       for _ in range(n)]
        opens = [rng.random() < 0.6 for _ in range(n)]
        times = exact_times(n)
        got, end, cap_until, min_s = driver_backout(samples, opens, times, reason)
        want, want_end = count_backout(samples, opens, times, cap_until, min_s)
        assert end == want_end, seed
        assert got == want, seed


# --- jittered 90-200 ms: never later, never on less than the window -----------------

TIMINGS = [jittered_times, measured_times]


@pytest.mark.parametrize("timing", TIMINGS)
@pytest.mark.parametrize("p_unknown", [0.0, 0.15])
def test_a_stop_is_never_called_stuck_later_than_the_count_did(timing, p_unknown):
    fired = 0
    for seed in SEEDS:
        rng = random.Random(seed)
        samples = stop_stream(60, rng, p_unknown)
        times = timing(len(samples), rng)
        new, old = first(driver_stuck(samples, times)), first(count_stuck(samples, times))
        assert old is not None, seed
        assert new is not None and new <= old, (seed, new, old)
        fired += new < old
    # And the jitter really does move it: on slow steps it fires sooner.
    assert fired > 0


@pytest.mark.parametrize("timing", TIMINGS)
def test_stuck_never_fires_on_less_than_its_window_of_stillness(timing):
    """Earlier is fewer samples, never less time: from the first STILL of the
    run to the fire is STUCK_CONFIRM_S at least, on any stream."""
    for seed in SEEDS:
        rng = random.Random(seed)
        samples = noisy_stream(60, rng, 0.1) + stop_stream(60, rng, 0.1)
        times = timing(len(samples), rng)
        d = dv.Driver(FakeController(), FakeCamera())
        d._throttle_since = 0.0
        d.last_recovery_time = -1e9
        run_start = None
        for s, t in zip(samples, times):
            before = d._still_samples
            fired = d.stuck(s, t)
            if before == 0 and (d._still_samples == 1 or fired):
                run_start = t
            if fired:
                assert t - run_start >= dv.STUCK_CONFIRM_S - 1e-9, seed
                d.last_recovery_time = t
                run_start = None


@pytest.mark.parametrize("timing", TIMINGS)
def test_a_reverse_into_a_wall_is_never_stopped_later_than_the_count_did(timing):
    """Rolling back at a real reverse's shake, then against something at the
    ESC's buzz, wall in front the whole way: only the bumper or the cap ends
    it, and the bumper no later than it did."""
    sooner = 0
    for seed in SEEDS:
        rng = random.Random(seed)
        n = 45
        samples = stop_stream(n, rng, 0.1, moving=(0.33, 0.6), still=(0.0, 0.08))
        opens = [False] * n
        times = timing(n, rng)
        got, end, cap_until, min_s = driver_backout(samples, opens, times)
        _, want_end = count_backout(samples, opens, times, cap_until, min_s)
        assert end[0] is not None and want_end[0] is not None, seed
        assert end[0] <= want_end[0], (seed, end, want_end)
        sooner += end[0] < want_end[0]
    assert sooner > 0


@pytest.mark.parametrize("timing", TIMINGS)
def test_an_opening_never_ends_the_reverse_later_than_the_count_did(timing):
    for seed in SEEDS:
        rng = random.Random(seed)
        n = 45
        samples = [imu(rng.uniform(0.33, 0.6)) for _ in range(n)]
        opens = [rng.random() < 0.7 for _ in range(n)]
        times = timing(n, rng)
        got, end, cap_until, min_s = driver_backout(samples, opens, times)
        _, want_end = count_backout(samples, opens, times, cap_until, min_s)
        assert end[0] is not None and want_end[0] is not None, seed
        assert end[0] <= want_end[0], (seed, end, want_end)


@pytest.mark.parametrize("timing", TIMINGS)
def test_the_ease_in_on_jittered_steps_is_still_half_then_all(timing):
    """Every step of 90 ms or more reaches ESCAPE_RAMP_S, so the ease-in is
    exactly the count's: half on the first decision, all after."""
    for seed in SEEDS:
        rng = random.Random(seed)
        n = 30
        samples = [imu(0.5)] * n
        opens = [False] * n
        times = timing(n, rng)
        got, end, cap_until, min_s = driver_backout(samples, opens, times)
        want, want_end = count_backout(samples, opens, times, cap_until, min_s)
        assert got == want and end == want_end, seed


# --- the lines themselves, and what they buy at other rates -------------------------

@pytest.mark.parametrize("n", [2, 3, 4, 5, 10])
def test_each_window_line_sits_where_the_count_did(n):
    """N - 2 steps of 0.1 s never reach it, N - 1 always do, and N - 1 steps
    of the fastest measured 90 ms do too."""
    line = dv.run_window_s(n)
    assert sum([dv.DECISION_S] * (n - 2)) < line
    assert sum([dv.DECISION_S] * (n - 1)) >= line
    assert sum([dv.FASTEST_DECISION_S] * (n - 1)) >= line


def test_the_smoothing_window_holds_five_at_the_tuned_and_fastest_steps():
    for step in (dv.DECISION_S, dv.FASTEST_DECISION_S):
        assert sum([step] * 4) < dv.IMU_WINDOW_S <= sum([step] * 5)


def test_the_lines_are_the_counts_they_replace():
    assert dv.STUCK_CONFIRM_S == pytest.approx(0.809)
    assert dv.REVERSE_STALL_S == pytest.approx(0.269)
    assert dv.IMU_WINDOW_S == pytest.approx(0.449)
    assert dv.IMU_WINDOW_FULL_S == pytest.approx(0.359)
    assert dv.BACKOUT_OPEN_S == pytest.approx(0.179)
    assert dv.ESCAPE_RAMP_S == pytest.approx(0.089)


def steady(step, n, start=T0):
    return [start + step * k for k in range(n)]


@pytest.mark.parametrize("hz,fires_on", [(10, 9), (16, 13), (5, 5)])
def test_stuck_takes_the_same_time_at_any_rate(hz, fires_on):
    """Parked and armed: the decision STUCK lands on, counted from the first
    still one. At 10 Hz the tenth, as before; at 16 Hz the fourteenth, 0.81 s
    in, where a count of ten would have fired at 0.56 s; on a 0.2 s step the
    sixth, 1.0 s in, where the count waited 1.8 s."""
    samples = [imu(0.002)] * 40
    times = steady(1.0 / hz, 40)
    assert first(driver_stuck(samples, times)) == fires_on
    assert times[fires_on] - times[0] >= dv.STUCK_CONFIRM_S


def test_the_decision_length_is_main_s_auto_tick():
    """DECISION_S stands in for main.AUTO_TICK_INTERVAL, which the driver
    cannot import. Read off the source, not imported."""
    import ast
    import pathlib

    src = pathlib.Path(dv.__file__).parents[1] / "__main__.py"
    tree = ast.parse(src.read_text())
    values = [node.value.value for node in ast.walk(tree)
              if isinstance(node, ast.Assign)
              and any(getattr(t, "id", None) == "AUTO_TICK_INTERVAL"
                      for t in node.targets)]
    assert values == [dv.DECISION_S]
