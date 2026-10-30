"""/follow/marker: the person being followed, drawn in the 3D panel's map frame.

One SceneUpdate off autopilot/follow.PersonGoal.marker(), built on the
publisher thread: a pillar where the person is on the map (yellow, grey
once lost) and a faint disc as wide as the follow gap round them, inside
which the truck holds. An empty payload clears both.
"""

from crawler.telemetry import lidar_view as lv
from crawler.telemetry import map_view

PERSON_ID = "follow_person"
GAP_ID = "follow_gap"
PILLAR_M = 0.3
PILLAR_H_M = 1.0
GAP_ALPHA = 0.15


def scene(payload, log_time):
    from foxglove.messages import (Color, CylinderPrimitive, Pose, Quaternion,
                                   SceneEntity, SceneEntityDeletion,
                                   SceneEntityDeletionType, SceneUpdate, Vector3)
    stamp = lv._stamp(log_time)
    payload = payload or {}
    person = payload.get("person")
    ident = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    if person is None:
        return SceneUpdate(deletions=[
            SceneEntityDeletion(timestamp=stamp, type=SceneEntityDeletionType.MatchingId, id=i)
            for i in (PERSON_ID, GAP_ID)], entities=[])
    x, y = float(person[0]), float(person[1])
    lost = bool(payload.get("lost"))
    colour = (Color(r=0.6, g=0.6, b=0.6, a=0.8) if lost
              else Color(r=1.0, g=0.85, b=0.1, a=0.9))
    gap = 2.0 * float(payload.get("gap_m") or 0.0)

    def cyl(ident_, size, height, z, c):
        return SceneEntity(
            timestamp=stamp, frame_id=map_view.MAP_FRAME, id=ident_,
            cylinders=[CylinderPrimitive(
                pose=Pose(position=Vector3(x=x, y=y, z=z), orientation=ident),
                size=Vector3(x=size, y=size, z=height),
                bottom_scale=1.0, top_scale=1.0, color=c)])

    entities = [cyl(PERSON_ID, PILLAR_M, PILLAR_H_M, PILLAR_H_M / 2.0, colour)]
    if gap > 0.0:
        entities.append(cyl(GAP_ID, gap, 0.02, 0.01,
                            Color(r=1.0, g=0.85, b=0.1, a=GAP_ALPHA)))
    return SceneUpdate(deletions=[], entities=entities)
