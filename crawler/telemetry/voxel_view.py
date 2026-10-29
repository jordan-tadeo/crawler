"""The OAK-D's maps as Foxglove's 3D panel wants them, with --voxel-map.

  /voxel/points  foxglove.PointCloud in the map frame, every two seconds:
                 the centre of every occupied voxel, thinned evenly to at
                 most LIVE_POINTS_MAX, x, y, z float32
  /voxel/low     foxglove.Grid in the map frame, every two seconds: the low
                 obstacles round the truck, 5 cm cells, RGBA (unseen clear,
                 floor faint green, low obstacle orange), cropped and
                 coarsened under map_view.MAX_CELLS like /map

Both hang off the SLAM half of /tf like /map does, so with the 3D panel's
display frame on "map" they sit on the lidar's walls.
"""

import numpy as np

from crawler.mapping import voxel as vx
from crawler.telemetry import lidar_view as lv
from crawler.telemetry import map_view as mv
from crawler.telemetry.map_view import MAP_FRAME

COLOURS = np.zeros((3, 4), np.uint8)
COLOURS[vx.LOW_FLOOR] = (40, 120, 60, 90)
COLOURS[vx.LOW_OBSTACLE] = (255, 140, 0, 255)

# The most points one /voxel/points message carries: 120 KB. The voxel map
# hands over up to voxel.POINTS_MAX, 480 KB, and it gets there by filling up
# as the truck drives -- on the 10-29 drive the cloud went from 60 to 344
# KiB/s in seven minutes, which is the live view slowing down over a drive.
LIVE_POINTS_MAX = 10000


def points_message(points, log_time):
    """(N, 3) float32 map-frame points -> foxglove.PointCloud."""
    from foxglove.messages import (PackedElementField,
                                   PackedElementFieldNumericType, PointCloud,
                                   Pose, Quaternion, Vector3)
    pts = np.asarray(points, np.float32).reshape(-1, 3)
    if len(pts) > LIVE_POINTS_MAX:
        pts = pts[:: -(-len(pts) // LIVE_POINTS_MAX)]
    pts = np.ascontiguousarray(pts)
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
    return mv.grid_rgba(view.codes, view.x0, view.y0, view.res, COLOURS,
                        log_time)
