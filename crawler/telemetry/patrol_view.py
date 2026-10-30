"""/patrol/marker: the patrol's waypoints, drawn in the 3D panel's map frame.

One SceneUpdate off autopilot/patrol.Patrol.marker(), built on the publisher
thread: a small disc and its number (from 1, the order it is driven in) at
each waypoint, a line through them in order (closed round for a loop), and
the current one as a bigger amber disc. A list carried across a SLAM reseed
also gets a faint disc round the current one as wide as it may be off, as
the goal's disc is (nav_view.py). An empty payload clears everything this
ever drew, so a cleared list never stays on screen.
"""

from crawler.telemetry import lidar_view as lv
from crawler.telemetry import map_view
from crawler.telemetry import nav_view

LINE_ID = "patrol_line"
CURRENT_ID = "patrol_current"
APPROX_ID = "patrol_approx"
POINT_DISC_M = 0.18
CURRENT_DISC_M = 0.4
TEXT_M = 0.3
# Waypoint ids are point_id(k); this many are deleted on a clear, which is
# more than patrol.MAX_WAYPOINTS.
MAX_IDS = 128


def point_id(k):
    return f"patrol_wp_{k}"


def scene(payload, log_time):
    from foxglove.messages import (Color, CylinderPrimitive, LinePrimitive,
                                   LinePrimitiveLineType, Point3, Pose,
                                   Quaternion, SceneEntity, SceneEntityDeletion,
                                   SceneEntityDeletionType, SceneUpdate,
                                   TextPrimitive, Vector3)
    stamp = lv._stamp(log_time)
    payload = payload or {}
    pts = [(float(x), float(y)) for x, y in (payload.get("points") or [])]
    current = payload.get("current")
    ident = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

    def at(x, y, z):
        return Pose(position=Vector3(x=x, y=y, z=z), orientation=ident)

    def disc(xy, size, color, z=0.03):
        return CylinderPrimitive(pose=at(xy[0], xy[1], z),
                                 size=Vector3(x=size, y=size, z=0.06),
                                 bottom_scale=1.0, top_scale=1.0, color=color)

    entities, gone = [], []
    point_color = Color(r=0.3, g=0.6, b=1.0, a=0.9)
    for k, xy in enumerate(pts):
        entities.append(SceneEntity(
            timestamp=stamp, frame_id=map_view.MAP_FRAME, id=point_id(k),
            cylinders=[disc(xy, POINT_DISC_M, point_color)],
            texts=[TextPrimitive(pose=at(xy[0] + 0.15, xy[1] + 0.15, 0.1),
                                 billboard=True, font_size=TEXT_M, scale_invariant=False,
                                 color=Color(r=1.0, g=1.0, b=1.0, a=1.0),
                                 text=str(k + 1))]))
    gone.extend(point_id(k) for k in range(len(pts), MAX_IDS))

    line = pts + pts[:1] if payload.get("closed") and len(pts) >= 3 else pts
    if len(line) >= 2:
        entities.append(SceneEntity(
            timestamp=stamp, frame_id=map_view.MAP_FRAME, id=LINE_ID,
            lines=[LinePrimitive(type=LinePrimitiveLineType.LineStrip,
                                 pose=at(0.0, 0.0, 0.02), thickness=0.03,
                                 scale_invariant=False,
                                 points=[Point3(x=x, y=y, z=0.0) for x, y in line],
                                 color=Color(r=0.3, g=0.6, b=1.0, a=0.7))]))
    else:
        gone.append(LINE_ID)

    if current is not None and 0 <= int(current) < len(pts):
        xy = pts[int(current)]
        entities.append(SceneEntity(
            timestamp=stamp, frame_id=map_view.MAP_FRAME, id=CURRENT_ID,
            cylinders=[disc(xy, CURRENT_DISC_M, Color(r=1.0, g=0.7, b=0.1, a=0.9),
                            z=0.035)]))
        approx_m = float(payload.get("approx_m") or 0.0)
        if approx_m > 0.0:
            size = 2.0 * min(max(approx_m, CURRENT_DISC_M / 2.0), nav_view.APPROX_MAX_M)
            entities.append(SceneEntity(
                timestamp=stamp, frame_id=map_view.MAP_FRAME, id=APPROX_ID,
                cylinders=[disc(xy, size, Color(r=1.0, g=0.7, b=0.1,
                                                a=nav_view.APPROX_ALPHA), z=0.025)]))
        else:
            gone.append(APPROX_ID)
    else:
        gone.extend((CURRENT_ID, APPROX_ID))
    deletions = [SceneEntityDeletion(timestamp=stamp,
                                     type=SceneEntityDeletionType.MatchingId, id=i)
                 for i in gone]
    return SceneUpdate(deletions=deletions, entities=entities)
