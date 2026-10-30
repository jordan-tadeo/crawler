"""Follow a person: the person's place on the SLAM map is the goal.

The goal model drives. All this does is say where the goal is: the person
the windshield camera sees, put on the map, fed to the model through the
goal input it already has (policy_goal.encode: present, distance over FAR_M,
the bearing's sin and cos), exactly as a clicked goal is. Nothing here
steers and nothing turns the truck toward anybody; by the owner's ruling only
the model ever owns the wheels. The one thing it adds to the wheels is a
hold, the same neutral as the arrival hold (reached()): inside the follow gap,
and with nowhere to point the goal (nobody seen yet, or a search with no map
to pick from). With the person gone it chases where they were and then
searches (autopilot/search.py), which is only ever choosing the goal too.

Where a person is
-----------------
The detector (crawler/vision/) answers one to three times a second with a
box per person, a few hundred milliseconds after the frame was read
(runner.Sighting). Each box becomes a point on the map:

  bearing    the ray through the box's centre, off the camera's yaw and
             pitch (vision/track.ray), from the camera's place on the truck
             (mount.x_m, y_m).
  distance   in order, the first that has an answer:
             lidar  the roof lidar's returns inside the box's bearings, seen
                    from the camera, in the scan nearest the frame's time:
                    the nearest cluster of them (a person stands in front
                    of whatever is behind them). The lidar sees all round,
                    so this is the usual one.
             oakd   the OAK-D's depth down the box's bearing, when it is
                    inside the OAK-D's view and the lidar had nothing.
             box    the box's foot on the floor off the camera's height, on
                    a camera that sees the floor. Not the windshield one,
                    so on the truck as built this never answers (box_range).
             None of them: the sighting is not placed.
  the map    the truck's map pose at the frame's time (slam.pose_at), not
             now's: the answer is late, and the truck has moved since. So
             the point holds still on the map between the slow sightings,
             and the truck's own motion does not drag it along.

Who to follow
-------------
With nobody followed, the nearest placed person. Once one is followed,
the one nearest the last point within GATE_M (plus WALK_M_S for each second
since), so somebody walking past does not take over. Nobody in the gate is
a sighting of someone else, and the followed point stays where it was.

When they are not seen
----------------------
The detector misses a person it can see often: on the drive of 2026-10-30
(04:25-04:27Z, follow on) it answered every 0.96 s (median; p90 1.59 s,
the rate governor's slowest 2.0 s) and found the person in 16 of 90
answers, and after an answer that found them the next one did too only 6
times in 15 (20 of 55 over the whole recording). Boxes at a metre were
partial people at the frame's edges. So a person standing in view goes
unseen for 3 s or more about one re-sighting in five (8 of the 41
hit-to-hit gaps under 10 s), and the old 3 s "lost" fired on that. What
counts is whether they are still where they were seen, so it asks:

  following  seen within FRESH_S, or since then the newest lidar scan has
             returns within PRESENT_R_M of the point (somebody standing
             there, unseen by the camera), for up to CONFIRM_MAX_S. With no
             lidar, FRESH_NO_LIDAR_S. The goal is the point and the gap
             holds, as ever.
  chasing    gone from the point: the goal is where they were, carried on by
             the walk their last few sightings make (TRACK_S of them, at
             most EXTRAP_S of it and EXTRAP_MAX_M), fixed when the chase
             begins. Until CHASE_ARRIVE_M of it or CHASE_S since the chase
             began. No gap hold: the person is not there to keep it from.
  searching  no person for longer than that, or never one since Follow on
             (after WAIT_S): search.Search picks the goal, first their
             point carried EXTEND_M on down the bearing they were seen on,
             then the edges of the map. A hold only when it has nowhere to
             go yet.

This is what the walk-away of 21:26 local was: every sighting was inside
the gap (0.44 to 1.33 m), the gap only lets go when a sighting is past
RESUME_M, the person walked off the side of the frame before one was, and
the truck held neutral for 52 s.

The follow gap: within GAP_M of the person the truck holds neutral, as at a
clicked goal's ARRIVE_M; it drives again once the person is RESUME_M off.
Two numbers, so a person standing on the line does not flicker it.

Across a reseed the point is carried along the map links like a clicked
goal (goal.ClickGoal), and the next sighting replaces it anyway.

/follow/status says all of it, with the goal input as fed and the truck's
pose on the same map frame, so a recorded follow session can be turned into
labels later: the goal input, and what the wheels were told (/vehicle).
"""

import collections
import math

import numpy as np

from crawler.autopilot import goal as gl
from crawler.autopilot import policy_goal as pg

# Inside this of the person, neutral (the follow gap)...
GAP_M = 1.2
# ...until the person is this far off again.
RESUME_M = 1.6
# A sighting this recent is the person, where they were seen...
FRESH_S = 1.5
# ...this, with no lidar to look at the place with.
FRESH_NO_LIDAR_S = 4.0
# Past FRESH_S: lidar returns (MIN_HITS) this close to the point are them
# standing there unseen, for up to this long since the sighting.
PRESENT_R_M = 0.45
CONFIRM_MAX_S = 20.0
# Gone from the point: where they were, plus their walk, until this long
# since the chase began, or the truck is this close to it.
CHASE_S = 10.0
CHASE_ARRIVE_M = 0.5
# Their walk: off the sightings of the last TRACK_S, at least MIN_TRACK_DT_S
# apart end to end, at most WALK_M_S fast; carried on for at most EXTRAP_S
# and EXTRAP_MAX_M.
TRACK_S = 4.0
MIN_TRACK_DT_S = 0.5
EXTRAP_S = 2.0
EXTRAP_MAX_M = 1.0
# Follow on with nobody seen: this long for the detector before searching.
WAIT_S = 3.0
# A followed person can move this far between sightings and still be them...
GATE_M = 1.0
# ...plus this for every second since the last.
WALK_M_S = 1.5
# The newest of a smoothed point: this much of the new sighting.
SMOOTH = 0.7
# A sighting older than this when it arrives is not used.
SIGHTING_MAX_AGE_S = 2.0
# Lidar returns closer to the camera than this are the truck.
NEAR_M = 0.25
FAR_M = 8.0
# The nearest cluster: returns within this of the nearest, at least MIN_HITS.
CLUSTER_M = 0.3
MIN_HITS = 3
# The box's bearings widened by this each side, for the lidar's slice.
SLICE_PAD_DEG = 1.0
# Scans kept for matching a late sighting to the scan of its time.
SCANS_KEPT = 30
SCAN_MAX_DT_S = 0.3
# A box's bottom edge this close to the frame's edge is cut off.
EDGE = 0.02
# Commands on /follow/cmd (nav/click.FOLLOW_TOPIC).
COMMANDS = ("on", "off", "toggle")


class CameraGeometry:
    """Where the camera is and which way it looks, off a pantilt config."""

    def __init__(self, hfov_deg, vfov_deg, yaw_deg=0.0, pitch_deg=0.0,
                 x_m=0.0, y_m=0.0, z_m=0.22):
        self.hfov_deg, self.vfov_deg = float(hfov_deg), float(vfov_deg)
        self.yaw_deg, self.pitch_deg = float(yaw_deg), float(pitch_deg)
        self.x_m, self.y_m, self.z_m = float(x_m), float(y_m), float(z_m)

    @classmethod
    def from_config(cls, cfg):
        c, m = cfg.camera, cfg.mount
        return cls(c.hfov_deg, c.vfov_deg, m.yaw_deg, m.pitch_deg,
                   m.x_m, m.y_m, getattr(m, "z_m", 0.22))

    def ray(self, u, v, pan_deg=0.0, tilt_deg=0.0):
        from crawler.vision import track as tk
        return tk.ray(u, v, self.hfov_deg, self.vfov_deg,
                      self.yaw_deg + pan_deg, self.pitch_deg + tilt_deg)


def lidar_points(scan):
    """Car-frame (x, y) metres of a lidar.Scan's returns, or None."""
    if scan is None:
        return None
    from crawler.hardware import lidar as ld
    d = np.asarray(scan.distances_mm, float)
    ok = d > 0
    if not ok.any():
        return None
    x, y = ld.to_car(np.asarray(scan.angles_deg, float)[ok], d[ok])
    return np.asarray(x, float) / 1000.0, np.asarray(y, float) / 1000.0


def lidar_range(points, cam, b_lo, b_hi):
    """Metres from the camera to the nearest cluster of lidar returns with a
    bearing from the camera in [b_lo, b_hi] degrees, or None."""
    if points is None:
        return None
    x, y = points
    dx, dy = x - cam.x_m, y - cam.y_m
    r = np.hypot(dx, dy)
    b = np.degrees(np.arctan2(dy, dx))
    keep = (r > NEAR_M) & (r < FAR_M) & (b >= b_lo) & (b <= b_hi)
    r = np.sort(r[keep])
    if len(r) < MIN_HITS:
        return None
    # The nearest MIN_HITS that sit together, so one stray return in front is
    # not the person.
    for i in range(len(r) - MIN_HITS + 1):
        if r[i + MIN_HITS - 1] - r[i] <= CLUSTER_M:
            near = r[(r >= r[i]) & (r <= r[i] + CLUSTER_M)]
            return float(np.median(near))
    return None


def box_range(det, cam, pan_deg=0.0, tilt_deg=0.0):
    """Metres along the floor from the camera to a person off the box alone,
    or None: the foot on the floor off the camera's height.

    Never the box's height against a person's: on the recorded drive the
    boxes at 1-1.5 m were mostly partial people, and a height read put
    them 1.5-2.2x too far, which is a goal past the person. And on the
    windshield camera, pitched 45 up with the frame's bottom about 9
    degrees above level, there is no floor in view, so this is None there
    and the lidar or the OAK-D places everybody."""
    _, foot = cam.ray(det.cx, det.y1, pan_deg, tilt_deg)
    if det.y1 < 1.0 - EDGE and foot < -1.0:
        d = cam.z_m / -math.tan(math.radians(foot))
        return d if 0.3 < d < 15.0 else None
    return None


class OakdRange:
    """The OAK-D's depth down a bearing, for a person the lidar missed. A
    camera hub depth listener (hub.depth_listeners): it only keeps the
    newest few frames, on the reader's thread; __call__ reads them on the
    control loop's.

    The column for a bearing is off the lens's published fx
    (driver.FX_AT_640); the 20th percentile of the valid depth in a band
    COLUMN_HALF wide over the middle rows, which is the person when they
    are in front of whatever is behind them. The bearing is the windshield
    camera's and the depth is from the OAK-D's lens, 0.235 m ahead of the
    lidar to the camera's 0.15, so a few centimetres off; it is the second
    choice for that."""

    COLUMN_HALF = 4
    ROWS = (0.2, 0.8)
    HALF_VIEW_DEG = 33.0
    MAX_DT_S = 0.3
    KEPT = 4

    def __init__(self):
        self._frames = collections.deque(maxlen=self.KEPT)

    def on_depth(self, now, frame, capture_t=None):
        if frame is not None:
            self._frames.append((float(capture_t if capture_t is not None else now), frame))

    def __call__(self, bearing_deg, t):
        if abs(bearing_deg) > self.HALF_VIEW_DEG or not self._frames:
            return None
        ft, frame = min(list(self._frames), key=lambda f: abs(f[0] - t))
        if abs(ft - t) > self.MAX_DT_S:
            return None
        from crawler.autopilot import driver as dv
        h, w = frame.shape[:2]
        fx = dv.FX_AT_640 * w / 640.0
        col = int(round(w / 2.0 - fx * math.tan(math.radians(bearing_deg))))
        c0, c1 = max(col - self.COLUMN_HALF, 0), min(col + self.COLUMN_HALF + 1, w)
        r0, r1 = int(h * self.ROWS[0]), int(h * self.ROWS[1])
        if c1 <= c0:
            return None
        band = np.asarray(frame[r0:r1, c0:c1], float).ravel()
        band = band[(band > 200.0) & (band < FAR_M * 1000.0)]
        if len(band) < 20:
            return None
        z = float(np.percentile(band, 20)) / 1000.0
        return z / math.cos(math.radians(bearing_deg))


class PersonGoal(gl.Goal):
    """The followed person's place on the SLAM map, as a goal, and the
    follow gap; the chase after them and the search for them when they are
    gone. Control loop only: the vision runner's thread only ever swaps its
    Sighting, which poll() reads.

    sightings  something with sighting() -> runner.Sighting or None
    lidar      something with latest() -> lidar.Scan or None (LidarReader)
    pose_at    t -> a pose with .x .y .yaw (mapping runner.pose_at), or None:
               the truck's map pose at a monotonic time
    oakd       optional (bearing_deg off the truck, t) -> metres or None
    map_view   optional () -> runner.MapView or None, for the search
    """

    def __init__(self, camera, sightings=None, lidar=None, pose_at=None,
                 oakd=None, gap_m=GAP_M, resume_m=RESUME_M, map_view=None,
                 say=None):
        from crawler.autopilot import search as sr
        super().__init__()
        self.cam = camera
        self.sightings = sightings
        self.lidar = lidar
        self.pose_at = pose_at
        self.oakd = oakd
        self.gap_m, self.resume_m = float(gap_m), float(resume_m)
        self.arrive_m = self.gap_m
        self._say = say
        self.search = sr.Search(map_view=map_view, say=say)
        self.active = False
        self.mode = "off"           # off / waiting / following / gap / chasing / searching
        self.point = None           # (x, y, frame) on the map, or None
        self.seen_t = None          # the frame time it was last placed at
        self.seen_from = None       # the truck's (x, y) then, on point's frame
        self.chase = None           # (x, y, frame) chased to, or None
        self.now = None             # the newest crop's time
        self.frame_now = None
        self.present = False
        self.last = None            # (distance, bearing) to the goal from the newest crop
        self.pose = None            # the newest present crop's (x, y, yaw)
        self.source = ""            # what the last placed distance came off
        self.people = 0             # people in the newest sighting
        self.placed = 0
        self.unplaced = 0
        self.ignored = 0            # sightings of somebody else
        self.acquired = 0
        self.confirmed = False      # the lidar sees somebody at the point
        self._on_t = None
        self._gap = False
        self._chase_done = False
        self._chase_t = None
        self._track = collections.deque()   # (t, x, y) raw placed points, point's frame
        self._seq = 0               # the newest egomap.Link taken
        self._sighting_seq = None
        self._scans = collections.deque(maxlen=SCANS_KEPT)
        self._scan_t = None
        self._spot = (None, None, False)    # (scan t, point, taken)
        self._said_mode = "off"

    def _log(self, level, text):
        print(text)
        if self._say is not None:
            try:
                self._say(level, "Follow", text)
            except Exception:
                pass

    # ---- the switch -----------------------------------------------------------------

    def command(self, msg):
        """One {"cmd": "on" | "off" | "toggle"}: True if it did something,
        None if it is not one."""
        name = msg.get("cmd") if isinstance(msg, dict) else None
        if name not in COMMANDS:
            self._log("warning", f"{name!r} is not a follow command; one of "
                      f"{', '.join(COMMANDS)}")
            return None
        on = (not self.active) if name == "toggle" else name == "on"
        if on and not self.active:
            self.active = True
            self.point, self.seen_t, self.seen_from = None, None, None
            self.chase, self._chase_done, self._gap = None, False, False
            self._track.clear()
            self.search.reset()
            self._on_t = None
            self.mode = self._said_mode = "waiting"
            self._log("info", "Follow on: the nearest person is the goal")
        elif not on and self.active:
            self.active = False
            self.search.reset()
            self.mode = self._said_mode = "off"
            self._log("info", "Follow off")
        return True

    # ---- the goal the model is fed --------------------------------------------------

    @property
    def goal(self):
        """(x, y, frame) the model drives to now, or None."""
        if not self.active:
            return None
        if self.mode in ("following", "gap"):
            return self.point
        if self.mode == "chasing":
            return self.chase
        if self.mode == "searching":
            return self.search.target
        return None

    def lost(self):
        """Not following anybody where they were seen."""
        return self.mode not in ("following", "gap")

    def reached(self):
        """Neutral inside the gap, while waiting for a first sighting, and
        searching with nowhere to go. Off, never: the switch hands the
        wheels back to the patrol."""
        if not self.active:
            return False
        return self.goal is None or self.mode == "gap"

    def encode(self, crop):
        g = self.goal
        return pg.missing() if g is None else pg.encode(crop, g)

    # ---- each pass --------------------------------------------------------------------

    def poll(self, crop):
        """Every control pass, any mode: the newest scan into the history,
        the newest sighting onto the map, and where it is off this crop."""
        self._keep_scan()
        if self.active:
            self._take_sighting(crop)
        self.observe(crop)

    def age(self):
        """Seconds since the person was last placed, or None."""
        if self.seen_t is None or self.now is None:
            return None
        return self.now - self.seen_t

    def observe(self, crop):
        if crop is not None:
            self.now = float(getattr(crop, "t", 0.0))
            self.frame_now = int(getattr(crop, "frame", 0))
            self._follow(getattr(crop, "links", ()))
            if getattr(crop, "present", False):
                self.pose = (crop.x, crop.y, crop.yaw)
        self.present = bool(crop is not None and getattr(crop, "present", False))
        if self.active:
            self._decide(crop)
        g = self.goal
        self.last = (pg.distance_bearing(crop, g)
                     if (g is not None and self.present) else None)
        if self.active and self.mode in ("following", "gap") and self.last is not None:
            d = self.last[0]
            if not self._gap and d <= self.gap_m:
                self._gap = True
            elif self._gap and d >= self.resume_m:
                self._gap = False
            self.mode = "gap" if self._gap else "following"
        self._say_mode()

    def _decide(self, crop):
        """The mode off the sighting's age, the lidar at the point, and the
        chase."""
        if self.now is not None and self._on_t is None:
            self._on_t = self.now
        age = self.age()
        if self.point is None or age is None:
            if self.mode == "searching":
                self.search.step(crop)
            elif self.now is not None and self.now - self._on_t >= WAIT_S:
                self._begin_search(crop)
            else:
                self.mode = "waiting"
            return
        if self._is_there(age, crop):
            if self.mode not in ("following", "gap"):
                self.mode = "gap" if self._gap else "following"
            return
        self._gap = False
        if self.mode in ("following", "gap", "waiting") and not self._chase_done:
            self.chase = self._chase_point()
            self._chase_t = self.now
            self.mode = "chasing"
        if self.mode == "chasing":
            near = (self.present and self.chase is not None
                    and self.chase[2] == self.frame_now
                    and math.hypot(self.chase[0] - crop.x, self.chase[1] - crop.y)
                    <= CHASE_ARRIVE_M)
            if near or self.now - self._chase_t > CHASE_S or self.chase is None:
                self._chase_done = True
                self._begin_search(crop)
            return
        if self.mode != "searching":
            self._begin_search(crop)
            return
        self.search.step(crop)

    def _is_there(self, age, crop):
        """Whether the person is still where they were seen: seen just now,
        or the lidar has somebody at the point."""
        self.confirmed = False
        fresh = FRESH_S if self.lidar is not None else FRESH_NO_LIDAR_S
        if age <= fresh:
            return True
        if age > CONFIRM_MAX_S or crop is None or not self.present:
            return False
        if self.point[2] != self.frame_now:
            return False
        self.confirmed = self._spot_taken(crop)
        return self.confirmed

    def _spot_taken(self, crop):
        """MIN_HITS returns of the newest scan within PRESENT_R_M of the
        point, off this crop's pose. Once per scan."""
        scan = self._scans[-1] if self._scans else None
        if scan is None:
            return False
        if self._spot[0] == scan.t and self._spot[1] == self.point:
            return self._spot[2]
        taken = False
        pts = lidar_points(scan)
        if pts is not None:
            x, y = pts
            keep = np.hypot(x, y) > NEAR_M
            c, s = math.cos(crop.yaw), math.sin(crop.yaw)
            mx = crop.x + c * x[keep] - s * y[keep]
            my = crop.y + s * x[keep] + c * y[keep]
            near = np.hypot(mx - self.point[0], my - self.point[1]) <= PRESENT_R_M
            taken = int(near.sum()) >= MIN_HITS
        self._spot = (scan.t, self.point, taken)
        return taken

    def walk(self):
        """(vx, vy) m/s off the track, or (0, 0)."""
        if len(self._track) < 2:
            return 0.0, 0.0
        t0, x0, y0 = self._track[0]
        t1, x1, y1 = self._track[-1]
        dt = t1 - t0
        if dt < MIN_TRACK_DT_S:
            return 0.0, 0.0
        vx, vy = (x1 - x0) / dt, (y1 - y0) / dt
        v = math.hypot(vx, vy)
        if v > WALK_M_S:
            vx, vy = vx * WALK_M_S / v, vy * WALK_M_S / v
        return vx, vy

    def _chase_point(self):
        if self.point is None:
            return None
        vx, vy = self.walk()
        age = self.age() or 0.0
        dx, dy = vx * min(age, EXTRAP_S), vy * min(age, EXTRAP_S)
        r = math.hypot(dx, dy)
        if r > EXTRAP_MAX_M:
            dx, dy = dx * EXTRAP_MAX_M / r, dy * EXTRAP_MAX_M / r
        return (self.point[0] + dx, self.point[1] + dy, self.point[2])

    def _begin_search(self, crop):
        hint = None
        if self.point is not None and self.seen_from is not None:
            hint = (self.seen_from, self.point[:2], self.point[2])
        self.mode = "searching"
        self._say_mode()
        self.search.start(crop, hint)
        self.search.step(crop)

    def _say_mode(self):
        if self.mode == self._said_mode:
            return
        was, self._said_mode = self._said_mode, self.mode
        age = self.age()
        if self.mode == "following":
            if was in ("chasing", "searching", "waiting"):
                return  # the sighting said it
            text = "Following: the person is past the gap again"
        elif self.mode == "gap":
            text = f"Holding the gap: the person is within {self.gap_m:g} m"
        elif self.mode == "chasing":
            text = (f"Chasing where the person was last seen, {age:.1f} s ago"
                    if age is not None else "Chasing where the person was last seen")
        elif self.mode == "searching":
            text = ("Searching: nobody seen since Follow on" if self.point is None
                    else "Searching: the person is gone from where they were seen")
        else:
            return
        self._log("info", text)

    def _keep_scan(self):
        if self.lidar is None:
            return
        try:
            scan = self.lidar.latest()
        except Exception:
            return
        if scan is None or scan.t == self._scan_t:
            return
        self._scan_t = scan.t
        self._scans.append(scan)

    def _scan_at(self, t):
        best = None
        for s in self._scans:
            if best is None or abs(s.t - t) < abs(best.t - t):
                best = s
        if best is None or abs(best.t - t) > SCAN_MAX_DT_S:
            return None
        return best

    def _take_sighting(self, crop):
        s = self.sightings.sighting() if self.sightings is not None else None
        if s is None or s.seq == self._sighting_seq:
            return
        self._sighting_seq = s.seq
        self.people = len(s.people)
        if crop is None or not getattr(crop, "present", False):
            return
        if self.now is not None and float(getattr(crop, "t", 0.0)) - s.t > SIGHTING_MAX_AGE_S:
            return
        pose = self._pose_at(s.t, crop)
        if pose is None:
            return
        frame = int(getattr(crop, "frame", 0))
        points = lidar_points(self._scan_at(s.t))
        placed = []
        for det in s.people:
            got = self.place(det, s.pan_deg, s.tilt_deg, points, s.t)
            if got is None:
                self.unplaced += 1
                continue
            (ax, ay), source = got
            c, sn = math.cos(pose[2]), math.sin(pose[2])
            placed.append((pose[0] + c * ax - sn * ay, pose[1] + sn * ax + c * ay, source))
        if not placed:
            return
        age = max(s.t - self.seen_t, 0.0) if self.seen_t is not None else None
        follow = (self.point is not None and self.point[2] == frame
                  and age is not None and self.mode in ("following", "gap", "chasing"))
        if follow:
            gate = GATE_M + WALK_M_S * age
            near = min(placed, key=lambda p: math.hypot(p[0] - self.point[0],
                                                       p[1] - self.point[1]))
            if math.hypot(near[0] - self.point[0], near[1] - self.point[1]) > gate:
                self.ignored += 1
                return
            k = SMOOTH
            x = self.point[0] + k * (near[0] - self.point[0])
            y = self.point[1] + k * (near[1] - self.point[1])
            if self.mode not in ("following", "gap"):
                self._log("info", f"Following again: seen {math.hypot(x - pose[0], y - pose[1]):.1f} m "
                          f"off after {age:.1f} s (distance off the {near[2]})")
        else:
            near = min(placed, key=lambda p: math.hypot(p[0] - pose[0], p[1] - pose[1]))
            x, y = near[0], near[1]
            self.acquired += 1
            self._track.clear()
            self._log("info", f"Following a person {math.hypot(x - pose[0], y - pose[1]):.1f} m "
                      f"off (distance off the {near[2]})")
        self.point = (x, y, frame)
        self.seen_t = s.t
        self.seen_from = (pose[0], pose[1])
        self.source = near[2]
        self.placed += 1
        self._track.append((s.t, near[0], near[1]))
        while self._track and s.t - self._track[0][0] > TRACK_S:
            self._track.popleft()
        if self.mode not in ("following", "gap"):
            self.mode = "following"
            self._said_mode = "following"
            self._gap = False
        self.chase, self._chase_done = None, False
        self.search.reset()

    def _pose_at(self, t, crop):
        if self.pose_at is not None:
            try:
                p = self.pose_at(t)
            except Exception:
                p = None
            if p is not None:
                return (float(p.x), float(p.y), float(p.yaw))
            return None
        return (crop.x, crop.y, crop.yaw)

    def place(self, det, pan_deg, tilt_deg, points, t):
        """((ahead, left) metres off the lidar's centre, source) for one box,
        or None."""
        cam = self.cam
        b, _ = cam.ray(det.cx, det.cy, pan_deg, tilt_deg)
        bl, _ = cam.ray(max(det.x0, 0.0), det.cy, pan_deg, tilt_deg)
        br, _ = cam.ray(min(det.x1, 1.0), det.cy, pan_deg, tilt_deg)
        lo, hi = min(bl, br) - SLICE_PAD_DEG, max(bl, br) + SLICE_PAD_DEG
        r, source = lidar_range(points, cam, lo, hi), "lidar"
        if r is None and self.oakd is not None:
            try:
                r, source = self.oakd(b, t), "oakd"
            except Exception:
                r = None
        if r is None:
            r, source = box_range(det, cam, pan_deg, tilt_deg), "box"
        if r is None or not math.isfinite(r):
            return None
        br_ = math.radians(b)
        return (cam.x_m + r * math.cos(br_), cam.y_m + r * math.sin(br_)), source

    # ---- across a reseed ------------------------------------------------------------

    def _follow(self, links):
        for L in links:
            if L.seq <= self._seq:
                continue
            self._seq = L.seq
            self.search.follow(L)
            c, s = math.cos(L.yaw), math.sin(L.yaw)

            def tf(x, y):
                return c * x - s * y + L.x, s * x + c * y + L.y
            if self.chase is not None and self.chase[2] == L.frm:
                self.chase = (*tf(self.chase[0], self.chase[1]), L.to)
            if self.point is None or self.point[2] != L.frm:
                continue
            self.point = (*tf(self.point[0], self.point[1]), L.to)
            if self.seen_from is not None:
                self.seen_from = tf(*self.seen_from)
            self._track = collections.deque((t, *tf(x, y)) for t, x, y in self._track)

    # ---- out ------------------------------------------------------------------------

    def state(self):
        """off / waiting / following / gap / chasing / searching."""
        return self.mode if self.active else "off"

    def status(self):
        """/goal/status while following: the goal, in goal.ClickGoal's schema."""
        g = self.goal
        d, b = self.last if self.last is not None else (-1.0, 0.0)
        x, y, frame = g if g is not None else (0.0, 0.0, -1)
        return {"goal_set": g is not None, "x": float(x), "y": float(y),
                "frame": int(frame),
                "frame_now": -1 if self.frame_now is None else int(self.frame_now),
                "on_map": bool(g is not None and self.last is not None),
                "distance_m": float(d) if g is not None else -1.0,
                "bearing_deg": float(math.degrees(b)) if g is not None else 0.0,
                "arrived": bool(self.reached()), "arrive_m": self.gap_m,
                "clicks": int(self.placed), "refused": int(self.unplaced),
                "approx": False, "approx_m": 0.0}

    def follow_status(self, crop=None):
        """/follow/status. With the goal input as fed and the truck's map pose
        on the same frame: a recorded follow session pairs them with /vehicle
        for labels, with nothing else to decode."""
        g = self.goal
        d, b = self.last if self.last is not None else (-1.0, 0.0)
        x, y, frame = self.point if self.point is not None else (0.0, 0.0, -1)
        gx, gy = (g[0], g[1]) if g is not None else (0.0, 0.0)
        tx, ty, tyaw = self.pose if self.pose is not None else (0.0, 0.0, 0.0)
        enc = self.encode(crop).reshape(-1) if crop is not None else pg.missing().reshape(-1)
        age = self.age()
        se = self.search
        return {"active": bool(self.active), "state": self.state(),
                "x": float(x), "y": float(y), "frame": int(frame),
                "frame_now": -1 if self.frame_now is None else int(self.frame_now),
                "goal_x": float(gx), "goal_y": float(gy),
                "distance_m": float(d) if g is not None else -1.0,
                "bearing_deg": float(math.degrees(b)) if g is not None else 0.0,
                "seen_age_s": float(age) if age is not None else -1.0,
                "source": self.source, "confirmed": bool(self.confirmed),
                "people": int(self.people), "placed": int(self.placed),
                "unplaced": int(self.unplaced), "ignored": int(self.ignored),
                "acquired": int(self.acquired), "holding": bool(self.reached()),
                "search_kind": se.kind if self.mode == "searching" else "",
                "search_targets": int(se.targets), "search_arrived": int(se.arrived),
                "search_skipped": int(se.skipped), "frontiers": int(se.frontiers),
                "gap_m": self.gap_m, "resume_m": self.resume_m,
                "fresh_s": FRESH_S, "chase_s": CHASE_S,
                "goal_input": [float(v) for v in enc],
                "truck_x": float(tx), "truck_y": float(ty),
                "truck_yaw_deg": float(math.degrees(tyaw))}

    def marker(self):
        """/follow/marker's payload: the person on the map the truck is on,
        or {} which clears it."""
        if self.point is None or self.frame_now is None or self.point[2] != self.frame_now:
            return {}
        return {"person": (self.point[0], self.point[1]), "lost": bool(self.lost()),
                "gap_m": self.gap_m}

class _Events:
    def __init__(self, events):
        self.events = events

    def take_events(self):
        out, self.events = self.events, []
        return out


class FollowSwitch(gl.Goal):
    """The patrol (patrol.Patrol, with its single clicked goal), or following
    a person (PersonGoal), whichever /follow/cmd last said. AutoPilot's goal
    with --goal and --pantilt. Clicks and patrol commands always go to the
    patrol, which keeps its list and its place while following; the follow
    commands are taken here. Control loop only."""

    def __init__(self, patrol, person):
        super().__init__()
        self.patrol = patrol
        self.person = person

    @property
    def following(self):
        return self.person.active

    @property
    def use(self):
        return "follow" if self.following else getattr(self.patrol, "use", "goal")

    @property
    def goal(self):
        return self.person.goal if self.following else self.patrol.goal

    @property
    def single(self):
        return self.patrol.single

    @property
    def ahead(self):
        # A person or a search target has no next goal known; the patrol's.
        return [] if self.following else getattr(self.patrol, "ahead", [])

    def passed(self, goal, crop=None):
        if self.following or not hasattr(self.patrol, "passed"):
            return False
        return self.patrol.passed(goal, crop)

    def reached(self):
        return self.person.reached() if self.following else self.patrol.reached()

    def encode(self, crop):
        return self.person.encode(crop) if self.following else self.patrol.encode(crop)

    def poll(self, clicks, crop):
        events = clicks.take_events() if clicks is not None and hasattr(clicks, "take_events") else []
        rest = []
        for kind, what in events:
            if kind == "follow":
                self.person.command(what)
            else:
                rest.append((kind, what))
        self.patrol.poll(_Events(rest), crop)
        self.person.poll(crop)

    def observe(self, crop):
        if self.following:
            self.person.observe(crop)
        else:
            self.patrol.observe(crop)

    def status(self):
        return self.person.status() if self.following else self.patrol.status()

    def patrol_status(self):
        return self.patrol.patrol_status()

    def marker(self):
        return self.patrol.marker()

    def follow_status(self, crop=None):
        return self.person.follow_status(crop)

    def follow_marker(self):
        return self.person.marker()

    def approx_m(self):
        return 0.0 if self.following else self.patrol.approx_m()
