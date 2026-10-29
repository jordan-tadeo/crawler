"""The OAK-D's maps as Foxglove's 3D panel wants them, with --voxel-map.

  /voxel/points  foxglove.PointCloud in the map frame, once a second: the
                 centre of every occupied voxel (at most voxel.POINTS_MAX),
                 x, y, z float32
  /voxel/low     foxglove.Grid in the map frame, once a second: the low
                 obstacles round the truck, 5 cm cells, RGBA (unseen clear,
                 floor faint green, low obstacle orange)

Both hang off the SLAM half of /tf like /map does, so with the 3D panel's
display frame on "map" they sit on the lidar's walls.
"""

import numpy as np

from crawler.mapping import voxel as vx
from crawler.telemetry import lidar_view as lv
from crawler.telemetry.map_view import MAP_FRAME

COLOURS = np.zeros((3, 4), np.uint8)
COLOURS[vx.LOW_FLOOR] = (40, 120, 60, 90)
COLOURS[vx.LOW_OBSTACLE] = (255, 140, 0, 255)


def points_message(points, log_time):
    """(N, 3) float32 map-frame points -> foxglove.PointCloud."""
    from foxglove.messages import (PackedElementField,
                                   PackedElementFieldNumericType, PointCloud,
                                   Pose, Quaternion, Vector3)
    pts = np.ascontiguousarray(np.asarray(points, np.float32).reshape(-1, 3))
    f32 = PackedElementFieldNumericType.Float32
    return PointCloud(
        timestamp=lv._stamp(log_time),
        frame_id=MAP_FRAME,
        pose=Pose(position=Vector3(x=0.0, y=0.0, z=0.0),
                  orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
        point_stride=12,
        fields=[PackedElementField(name=n, offset=4 * i, type=f32)
                for i, n in enumerate(("x", "y", "z"))],
        data=pts.tobytes(),
    )


def rgba(codes):
    """uint8 codes [nx, ny] -> the Grid's bytes: rows along y, columns along x."""
    return COLOURS[np.asarray(codes).T].tobytes()


def low_message(view, log_time):
    """A voxel.LowView -> foxglove.Grid."""
    from foxglove.messages import (Grid, PackedElementField,
                                   PackedElementFieldNumericType, Pose,
                                   Quaternion, Vector2, Vector3)
    nx, ny = view.codes.shape
    u8 = PackedElementFieldNumericType.Uint8
    return Grid(
        timestamp=lv._stamp(log_time),
        frame_id=MAP_FRAME,
        pose=Pose(position=Vector3(x=view.x0, y=view.y0, z=0.0),
                  orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
        column_count=nx,
        cell_size=Vector2(x=view.res, y=view.res),
        row_stride=4 * nx,
        cell_stride=4,
        fields=[PackedElementField(name=n, offset=i, type=u8)
                for i, n in enumerate(("red", "green", "blue", "alpha"))],
        data=rgba(view.codes),
    )
