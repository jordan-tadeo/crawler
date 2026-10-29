"""The roof lidar as Foxglove's 3D panel wants it, seen from above.

Three topics, all in Foxglove's own schemas so the 3D panel draws them with
no converter and no script:

  /tf            foxglove.FrameTransforms, base_link -> lidar, one per scan
  /lidar/scan    foxglove.LaserScan in the lidar frame, every revolution
  /lidar/truck   foxglove.SceneUpdate in base_link, an outline of the truck
                 with a triangle at the front, once a second

/lidar, the 72 bins, is untouched: it is what a clearance stop or a model
reads. This is the picture.

The frames
----------
base_link is the car frame lidar.py already uses: the middle of the truck,
x forward, y left, z up.

lidar is the puck: on its spin axis, x along the arrow on the lid, z up, y a
quarter turn counter-clockwise from the arrow seen from above. That is NOT
the sensor's own handedness -- it counts clockwise from above -- so a raw
angle phi is bearing -phi in this frame, and the scan's slots are laid out
that way below. Keeping z up means the transform from base_link is the
mount's yaw and nothing else, the same -90 as lidar.MOUNT, and the frame
axes the 3D panel draws for it sit upright.

Checked end to end in tests/test_lidar_view.py, through the wire bytes: a
raw 270 return comes out at +x in base_link, straight ahead, which is where
the box off the front bumper was measured.

Why the scan is put on a grid
-----------------------------
A LaserScan has no per-point angles. It is a start angle, an end angle and
ranges assumed evenly spaced between them, and the sensor's points are only
nearly even: each packet's 12 are, the gaps between packets wobble with the
motor, and a packet lost to a CRC error is a hole that would slide every
point after it round by its span. So each real point goes into the slot of
a fixed SCAN_STEP_DEG grid nearest its own angle, and a slot nothing landed
in is NaN, which the panel does not draw.

0.5 degrees, 720 slots. The sensor lays a point every 0.072 * f degrees at
f Hz, 0.71 at the 9.9 Hz measured on the car, so above 6.9 Hz no two points
share a slot and every real point is drawn. Each is at most a quarter of a
degree off its own angle: 4 mm at 1 m. Below 6.9 Hz two can meet and the
nearer one is kept. A fixed grid is also a fixed size, so the message cannot
grow with anything.
"""

import math

import numpy as np

from crawler.hardware import lidar as ld

BASE_FRAME = "base_link"
LIDAR_FRAME = "lidar"

SCAN_STEP_DEG = 0.5
SCAN_SLOTS = int(round(360.0 / SCAN_STEP_DEG))

# The outline. The length is the two bumpers in lidar.py: the front one
# estimated off a box, the back one inferred from the sensor being halfway
# along. The width is a GUESS: nobody has measured it (lidar.HALF_WIDTH_MM is
# None), and 300 mm is a typical 1/10 crawler body. It is there so I can tell
# which way the truck faces, not to measure clearance against.
TRUCK_FRONT_MM = ld.FRONT_BUMPER_X_MM
TRUCK_REAR_MM = ld.REAR_BUMPER_X_MM
TRUCK_WIDTH_GUESS_MM = 300.0

# The arrow at the front: tip on the front bumper, base this far back.
ARROW_LENGTH_MM = 150.0
ARROW_HALF_WIDTH_MM = 90.0

TRUCK_ID = "truck"


def truck_width_mm() -> float:
    if ld.HALF_WIDTH_MM:
        return 2.0 * float(ld.HALF_WIDTH_MM)
    return TRUCK_WIDTH_GUESS_MM


# ---- the numbers, no SDK --------------------------------------------------

def scan_slots(angles_deg, distances_mm, mount=ld.MOUNT, slots=SCAN_SLOTS):
    """Sensor polar -> ranges in metres on the lidar frame's grid.

    Slot i is bearing i * 360/slots counter-clockwise from the arrow in the
    lidar frame. NaN where nothing came back; two points in one slot keep
    the nearer.
    """
    d = np.asarray(distances_mm, dtype=np.float64)
    a = np.asarray(angles_deg, dtype=np.float64)
    ranges = np.full(slots, np.inf)
    ok = (d >= ld.MIN_RANGE_MM) & (d <= ld.MAX_RANGE_MM)
    if ok.any():
        # The sensor counts clockwise from above; this frame, z up, counts
        # the other way. Upside down, the two agree.
        sign = 1.0 if mount.upside_down else -1.0
        step = 360.0 / slots
        bearing = np.mod(sign * a[ok], 360.0)
        idx = np.rint(bearing / step).astype(np.intp) % slots
        np.minimum.at(ranges, idx, d[ok] / 1000.0)
    ranges[~np.isfinite(ranges)] = np.nan
    return ranges


def mount_quaternion(mount=ld.MOUNT):
    """base_link -> lidar rotation, (x, y, z, w): the yaw about z, only."""
    half = math.radians(mount.yaw_deg) / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


def mount_translation(mount=ld.MOUNT):
    """base_link -> lidar, metres. z is the cab top, unmeasured, so 0."""
    return (mount.x_mm / 1000.0, mount.y_mm / 1000.0, 0.0)


def truck_outline_m():
    """The outline's corners in base_link, metres, counter-clockwise."""
    w = truck_width_mm() / 2000.0
    f = TRUCK_FRONT_MM / 1000.0
    r = TRUCK_REAR_MM / 1000.0
    return [(f, w), (r, w), (r, -w), (f, -w)]


def truck_arrow_m():
    """The front triangle in base_link, metres: tip, left, right."""
    tip = TRUCK_FRONT_MM / 1000.0
    base = (TRUCK_FRONT_MM - ARROW_LENGTH_MM) / 1000.0
    half = ARROW_HALF_WIDTH_MM / 1000.0
    return [(tip, 0.0), (base, half), (base, -half)]


# ---- the messages ------------------------------------------------------------

def _stamp(log_time):
    from foxglove.messages import Timestamp
    return Timestamp(sec=int(log_time) // 1_000_000_000,
                     nsec=int(log_time) % 1_000_000_000)


def frame_transforms(log_time, mount=ld.MOUNT):
    """/tf: base_link -> lidar, as a foxglove.FrameTransforms."""
    from foxglove.messages import (FrameTransform, FrameTransforms,
                                   Quaternion, Vector3)
    x, y, z = mount_translation(mount)
    qx, qy, qz, qw = mount_quaternion(mount)
    return FrameTransforms(transforms=[FrameTransform(
        timestamp=_stamp(log_time),
        parent_frame_id=BASE_FRAME,
        child_frame_id=LIDAR_FRAME,
        translation=Vector3(x=x, y=y, z=z),
        rotation=Quaternion(x=qx, y=qy, z=qz, w=qw),
    )])


def laser_scan(scan, log_time, mount=ld.MOUNT):
    """/lidar/scan: one revolution as a foxglove.LaserScan in the lidar frame.

    No intensities. The schema's numbers are doubles, so they would double
    the message for a colour the panel is not set to use: measured through
    the SDK's own MCAP writer on a synthetic room at 10 scans a second,
    20.7 KiB/s with them and 10.3 without, about 36 MiB an hour.
    """
    from foxglove.messages import LaserScan, Pose, Quaternion, Vector3
    ranges = scan_slots(scan.angles_deg, scan.distances_mm, mount)
    return LaserScan(
        timestamp=_stamp(log_time),
        frame_id=LIDAR_FRAME,
        pose=Pose(position=Vector3(x=0.0, y=0.0, z=0.0),
                  orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
        start_angle=0.0,
        end_angle=math.radians((SCAN_SLOTS - 1) * SCAN_STEP_DEG),
        ranges=ranges.tolist(),
    )


def truck_scene(log_time):
    """/lidar/truck: the outline and the front arrow, as a foxglove.SceneUpdate.

    lifetime unset is forever and frame_locked keeps it on base_link, so a
    client that joins late has it within a second and it never fades.
    """
    from foxglove.messages import (Color, LinePrimitive, LinePrimitiveLineType,
                                   Point3, Pose, Quaternion, SceneEntity,
                                   SceneUpdate, TriangleListPrimitive, Vector3)
    identity = Pose(position=Vector3(x=0.0, y=0.0, z=0.0),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))
    outline = LinePrimitive(
        type=LinePrimitiveLineType.LineLoop,
        pose=identity,
        thickness=0.02,
        scale_invariant=False,
        points=[Point3(x=x, y=y, z=0.0) for x, y in truck_outline_m()],
        color=Color(r=1.0, g=1.0, b=1.0, a=0.9),
    )
    arrow = TriangleListPrimitive(
        pose=identity,
        points=[Point3(x=x, y=y, z=0.0) for x, y in truck_arrow_m()],
        color=Color(r=1.0, g=0.55, b=0.1, a=0.9),
    )
    return SceneUpdate(entities=[SceneEntity(
        timestamp=_stamp(log_time),
        frame_id=BASE_FRAME,
        id=TRUCK_ID,
        frame_locked=True,
        lines=[outline],
        triangles=[arrow],
    )])
