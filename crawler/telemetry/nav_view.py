"""/nav/path: what the navigator is doing, drawn in the 3D panel's map frame.

One SceneUpdate off nav/pointnav.PointNavDriver.nav_path(), built on the
publisher thread: the path still to drive as a line strip, the goal as a
green disc, and, when the goal cannot be reached, the reachable point the
truck is going to (or holding at) instead as an amber one. An empty payload
-- no goal -- clears all three, so a stale path never stays on screen.

A goal carried across a SLAM reseed (autopilot/goal.py) comes with
approx_m, how far off it may be: its disc is that wide each way (up to
APPROX_MAX_M) and faint, so a guess never looks like a click.
"""

from crawler.telemetry import lidar_view as lv
from crawler.telemetry import map_view

PATH_ID = "nav_path"
GOAL_ID = "nav_goal"
AIM_ID = "nav_aim"
DISC_M = 0.25              # the arrival tolerance's diameter is 0.5 m; this is a marker
APPROX_MAX_M = 3.0         # the widest an approximate goal's disc is drawn, radius
APPROX_ALPHA = 0.35


def path_scene(payload, log_time):
    from foxglove.messages import (Color, CylinderPrimitive, LinePrimitive,
                                   LinePrimitiveLineType, Point3, Pose,
                                   Quaternion, SceneEntity,
                                   SceneEntityDeletion,
                                   SceneEntityDeletionType, SceneUpdate,
                                   Vector3)
    stamp = lv._stamp(log_time)
    payload = payload or {}
    pts = payload.get("path") or []
    goal = payload.get("goal")
    aim = payload.get("aim")
    ident = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

    entities, gone = [], []
    if len(pts) >= 2:
        entities.append(SceneEntity(
            timestamp=stamp, frame_id=map_view.MAP_FRAME, id=PATH_ID,
            lines=[LinePrimitive(
                type=LinePrimitiveLineType.LineStrip,
                pose=Pose(position=Vector3(x=0.0, y=0.0, z=0.02), orientation=ident),
                thickness=0.04, scale_invariant=False,
                points=[Point3(x=float(x), y=float(y), z=0.0) for x, y in pts],
                color=Color(r=0.3, g=1.0, b=0.4, a=0.95))]))
    else:
        gone.append(PATH_ID)

    def disc(ident_, xy, color, size=DISC_M):
        return SceneEntity(
            timestamp=stamp, frame_id=map_view.MAP_FRAME, id=ident_,
            cylinders=[CylinderPrimitive(
                pose=Pose(position=Vector3(x=float(xy[0]), y=float(xy[1]), z=0.03),
                          orientation=ident),
                size=Vector3(x=size, y=size, z=0.06),
                bottom_scale=1.0, top_scale=1.0, color=color)])

    approx_m = float(payload.get("approx_m") or 0.0)
    if goal is not None and approx_m > 0.0:
        size = 2.0 * min(max(approx_m, DISC_M / 2.0), APPROX_MAX_M)
        entities.append(disc(GOAL_ID, goal, Color(r=0.2, g=0.9, b=0.3, a=APPROX_ALPHA),
                             size))
    elif goal is not None:
        entities.append(disc(GOAL_ID, goal, Color(r=0.2, g=0.9, b=0.3, a=0.8)))
    else:
        gone.append(GOAL_ID)
    if aim is not None:
        entities.append(disc(AIM_ID, aim, Color(r=1.0, g=0.7, b=0.1, a=0.85)))
    else:
        gone.append(AIM_ID)
    deletions = [SceneEntityDeletion(timestamp=stamp,
                                     type=SceneEntityDeletionType.MatchingId, id=i)
                 for i in gone]
    return SceneUpdate(deletions=deletions, entities=entities)
