"""/path: the path the goal model is pointed along, drawn in the 3D panel's
map frame.

One SceneUpdate off autopilot/pathgoal.PathPlanner.marker(), built on the
publisher thread: the path from where it was planned to the goal as a cyan
line, and the point the model is fed this decision as a magenta disc (a
point along the path), or a faint one on the goal itself when that is what
it is fed. The goal's own disc is /goal/marker's or /patrol/marker's. An
empty payload clears both, so a stale path never stays on screen.
"""

from crawler.telemetry import lidar_view as lv
from crawler.telemetry import map_view

LINE_ID = "path_line"
AIM_ID = "path_aim"
AIM_DISC_M = 0.2


def scene(payload, log_time):
    from foxglove.messages import (Color, CylinderPrimitive, LinePrimitive,
                                   LinePrimitiveLineType, Point3, Pose,
                                   Quaternion, SceneEntity, SceneEntityDeletion,
                                   SceneEntityDeletionType, SceneUpdate, Vector3)
    stamp = lv._stamp(log_time)
    payload = payload or {}
    pts = payload.get("path") or []
    aim = payload.get("aim")
    ident = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    entities, gone = [], []
    if len(pts) >= 2:
        entities.append(SceneEntity(
            timestamp=stamp, frame_id=map_view.MAP_FRAME, id=LINE_ID,
            lines=[LinePrimitive(
                type=LinePrimitiveLineType.LineStrip,
                pose=Pose(position=Vector3(x=0.0, y=0.0, z=0.025), orientation=ident),
                thickness=0.035, scale_invariant=False,
                points=[Point3(x=float(x), y=float(y), z=0.0) for x, y in pts],
                color=Color(r=0.2, g=0.85, b=1.0, a=0.9))]))
    else:
        gone.append(LINE_ID)
    if aim is not None:
        alpha = 0.9 if payload.get("fed") == "subgoal" else 0.35
        entities.append(SceneEntity(
            timestamp=stamp, frame_id=map_view.MAP_FRAME, id=AIM_ID,
            cylinders=[CylinderPrimitive(
                pose=Pose(position=Vector3(x=float(aim[0]), y=float(aim[1]), z=0.04),
                          orientation=ident),
                size=Vector3(x=AIM_DISC_M, y=AIM_DISC_M, z=0.08),
                bottom_scale=1.0, top_scale=1.0,
                color=Color(r=1.0, g=0.2, b=0.9, a=alpha))]))
    else:
        gone.append(AIM_ID)
    deletions = [SceneEntityDeletion(timestamp=stamp,
                                     type=SceneEntityDeletionType.MatchingId, id=i)
                 for i in gone]
    return SceneUpdate(deletions=deletions, entities=entities)
