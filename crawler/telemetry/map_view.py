"""The SLAM map as Foxglove's 3D panel wants it, with --slam.

  /map           foxglove.Grid in the map frame, once a second: the map
                 round the truck, 10 cm pixels, RGBA (unknown clear, free
                 dark grey, occupied white), cropped to what has been seen
                 and coarsened past MAX_CELLS so it cannot outgrow the wifi
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

# The most cells a Grid message carries: 200 x 200, 160 KB of RGBA. The live
# view is not compressed, so a picture of the whole map at full resolution
# is a picture that grows with the drive -- the 40 m /map window is 640 KB at
# 10 cm and /voxel/low was 625 KB every second from the first minute. Past
# this the pixels get coarser instead: 10 cm becomes 20 cm, never a bigger
# message.
MAX_CELLS = 200 * 200


def bounded(codes, x0, y0, res, max_cells=MAX_CELLS):
    """uint8 codes [nx, ny] (0 is unknown, and higher wins) -> the same
    picture as (codes, x0, y0, res) in at most max_cells cells.

    Cropped to the cells that are not 0 first, which costs nothing to look
    at: unknown is drawn clear. Then, only if it is still too big, `step`
    cells to a pixel, each pixel the highest code in it -- occupied over
    free over unknown, so a wall never thins out of the picture."""
    codes = np.asarray(codes)
    seen = codes > 0
    rows, cols = np.flatnonzero(seen.any(1)), np.flatnonzero(seen.any(0))
    if len(rows) == 0:
        return codes[:1, :1], x0, y0, res
    i0, i1, j0, j1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    codes = codes[i0:i1, j0:j1]
    x0, y0 = x0 + i0 * res, y0 + j0 * res
    nx, ny = codes.shape
    step = 1
    while -(-nx // step) * -(-ny // step) > max_cells:
        step += 1
    if step > 1:
        px, py = -(-nx // step) * step, -(-ny // step) * step
        pad = np.zeros((px, py), codes.dtype)
        pad[:nx, :ny] = codes
        codes = pad.reshape(px // step, step, py // step, step).max((1, 3))
        res = res * step
    return codes, float(x0), float(y0), float(res)


def grid_rgba(codes, x0, y0, res, colours, log_time, z=0.0):
    """uint8 codes [nx, ny] -> a foxglove.Grid in the map frame, under the
    MAX_CELLS budget, coloured by `colours` (code -> RGBA)."""
    from foxglove.messages import (Grid, PackedElementField,
                                   PackedElementFieldNumericType, Pose,
                                   Quaternion, Vector2, Vector3)
    codes, x0, y0, res = bounded(codes, x0, y0, res)
    nx, ny = codes.shape
    u8 = PackedElementFieldNumericType.Uint8
    return Grid(
        timestamp=lv._stamp(log_time),
        frame_id=MAP_FRAME,
        pose=Pose(position=Vector3(x=x0, y=y0, z=z),
                  orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
        column_count=nx,
        cell_size=Vector2(x=res, y=res),
        row_stride=4 * nx,
        cell_stride=4,
        fields=[PackedElementField(name=n, offset=i, type=u8)
                for i, n in enumerate(("red", "green", "blue", "alpha"))],
        # rows along y, columns along x, four bytes a cell
        data=colours[codes.T].tobytes(),
    )


def rgba(codes):
    """uint8 codes [nx, ny] -> the Grid's bytes: rows along y, columns along
    x, four bytes a cell."""
    return COLOURS[np.asarray(codes).T].tobytes()


def grid_message(view, log_time):
    """A runner.MapView as a foxglove.Grid."""
    return grid_rgba(view.codes, view.x0, view.y0, view.res, COLOURS,
                     log_time, z=-0.01)


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
