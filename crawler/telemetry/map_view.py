"""The SLAM map as Foxglove's 3D panel wants it, with --slam.

  /map           foxglove.Grid in the map frame, once a second: the map
                 round the truck, 10 cm pixels, RGBA (unknown clear, free
                 dark grey, occupied white)
  /tf            also carries map -> base_link, the SLAM pose, every scan
                 (the lidar's base_link -> lidar goes in the same message)
  /slam/path     foxglove.SceneUpdate in the map frame: where it has been
  /slam/status   JSON, every scan: crawler.mapping.runner.status_row()

The map frame: x, y in metres from where the truck booted, heading 0 the
boot heading, z up, the same handedness as base_link. Set the 3D panel's
display frame to "map" and the scan, the truck outline and the map all line
up; leave it on base_link and the map turns round the truck instead.
deploy/foxglove-drive.json's map panel does the first.
"""

import math

import numpy as np

from crawler.mapping import grid as gr
from crawler.telemetry import lidar_view as lv

MAP_FRAME = "map"

# code -> RGBA. Unknown is fully clear so the panel's own background shows.
COLOURS = np.zeros((3, 4), np.uint8)
COLOURS[gr.FREE] = (60, 60, 66, 200)
COLOURS[gr.OCCUPIED] = (240, 240, 240, 255)

PATH_ID = "path"


def rgba(codes):
    """uint8 codes [nx, ny] -> the Grid's bytes: rows along y, columns along
    x, four bytes a cell."""
    return COLOURS[np.asarray(codes).T].tobytes()


def grid_message(view, log_time):
    """A runner.MapView as a foxglove.Grid."""
    from foxglove.messages import (Grid, PackedElementField,
                                   PackedElementFieldNumericType, Pose,
                                   Quaternion, Vector2, Vector3)
    nx, ny = view.codes.shape
    u8 = PackedElementFieldNumericType.Uint8
    return Grid(
        timestamp=lv._stamp(log_time),
        frame_id=MAP_FRAME,
        pose=Pose(position=Vector3(x=view.x0, y=view.y0, z=-0.01),
                  orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
        column_count=nx,
        cell_size=Vector2(x=view.res, y=view.res),
        row_stride=4 * nx,
        cell_stride=4,
        fields=[PackedElementField(name=n, offset=i, type=u8)
                for i, n in enumerate(("red", "green", "blue", "alpha"))],
        data=rgba(view.codes),
    )


def yaw_quaternion(yaw):
    h = yaw / 2.0
    return (0.0, 0.0, math.sin(h), math.cos(h))


def frame_transforms(log_time, mount, pose, pose_ns):
    """/tf with SLAM: base_link -> lidar as always, and map -> base_link at
    the scan's own time `pose_ns` (epoch ns), so the panel interpolates the
    truck between scans instead of jumping it."""
    from foxglove.messages import (FrameTransform, FrameTransforms,
                                   Quaternion, Timestamp, Vector3)
    mx, my, mz = lv.mount_translation(mount)
    mqx, mqy, mqz, mqw = lv.mount_quaternion(mount)
    x, y, yaw = pose
    qx, qy, qz, qw = yaw_quaternion(yaw)
    return FrameTransforms(transforms=[
        FrameTransform(
            timestamp=lv._stamp(log_time),
            parent_frame_id=lv.BASE_FRAME,
            child_frame_id=lv.LIDAR_FRAME,
            translation=Vector3(x=mx, y=my, z=mz),
            rotation=Quaternion(x=mqx, y=mqy, z=mqz, w=mqw)),
        FrameTransform(
            timestamp=Timestamp(sec=int(pose_ns) // 1_000_000_000,
                                nsec=int(pose_ns) % 1_000_000_000),
            parent_frame_id=MAP_FRAME,
            child_frame_id=lv.BASE_FRAME,
            translation=Vector3(x=float(x), y=float(y), z=0.0),
            rotation=Quaternion(x=qx, y=qy, z=qz, w=qw)),
    ])


def path_scene(points, log_time):
    """/slam/path: the driven (x, y) points as one line strip in map."""
    from foxglove.messages import (Color, LinePrimitive, LinePrimitiveLineType,
                                   Point3, Pose, Quaternion, SceneEntity,
                                   SceneUpdate, Vector3)
    line = LinePrimitive(
        type=LinePrimitiveLineType.LineStrip,
        pose=Pose(position=Vector3(x=0.0, y=0.0, z=0.0),
                  orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
        thickness=0.03,
        scale_invariant=False,
        points=[Point3(x=float(x), y=float(y), z=0.0) for x, y in points],
        color=Color(r=0.2, g=0.8, b=1.0, a=0.9),
    )
    return SceneUpdate(entities=[SceneEntity(
        timestamp=lv._stamp(log_time),
        frame_id=MAP_FRAME,
        id=PATH_ID,
        lines=[line],
    )])
