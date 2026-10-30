"""A goal clicked in Foxglove's 3D panel.

How the click gets here. The 3D panel has a Publish tool (the icon in its
toolbar; right-click or hold it to pick which kind). On a server that is not
ROS -- the car runs foxglove-sdk's own WebSocket server -- the one kind that
works is "2D pose" ("Publish pose"): press on the map where the truck should
go, drag the way it should face, let go. Foxglove publishes that as a
foxglove.PoseInFrame on the tool's topic (the panel's Publish settings;
/move_base_simple/goal unless changed), with frame_id the panel's display
frame. Point and pose-estimate are ROS-only and never arrive here.

What the server has to do for that to work, per the Foxglove WebSocket
protocol: advertise the clientPublish capability, and list the message
encodings it accepts in supportedEncodings. Foxglove picks the first of
those it can write (json, ros1, cdr) and advertises a client channel with
it; then every click is one binary message on that channel. server_options()
is exactly those three start_server() arguments: ClientPublish, ["json"],
and this listener. JSON only, so the payload is always JSON text and parsing
it needs nothing compiled.

The listener callbacks run on the server's own thread, not the control loop,
so ClickGoals only parses and stores under a lock; the navigator picks the
newest goal up with take() when it next decides.

Frames. The goal is wanted in the map frame. A click with frame_id "map"
(the display frame set to the SLAM's map) is used as it is. One in
"base_link" (the display frame left on the truck) is relative to the truck,
and to_map() puts it on the map off the pose at the time it is used. Any
other frame is refused, never guessed at.

The orientation is kept (yaw) but the navigator only drives to the position;
a goal heading is an extra the planner can take later.

Patrol commands. With --goal, Foxglove's Publish panel sends a JSON object
on PATROL_TOPIC, one button a command ({"cmd": "start"} and so on, see
autopilot/patrol.py). Any schema name, JSON only, and it must be an object
with a string "cmd". Clicks and commands go into one queue in the order they
came (take_events), because "edit" then three clicks is three waypoints and
three clicks then "goal" is not.

Follow commands. With --goal and --pantilt, the Follow on and Follow off
buttons send {"cmd": "on"} or {"cmd": "off"} on FOLLOW_TOPIC
(autopilot/follow.py), the same way, as ("follow", dict) in the same queue.
"""

import collections
import json
import math
import threading
import time
from typing import NamedTuple, Optional

MAP_FRAME = "map"
BASE_FRAME = "base_link"

POSE_SCHEMAS = ("foxglove.PoseInFrame",
                "geometry_msgs/PoseStamped", "geometry_msgs/msg/PoseStamped")
POINT_SCHEMAS = ("geometry_msgs/PointStamped", "geometry_msgs/msg/PointStamped")
ENCODING = "json"
PATROL_TOPIC = "/patrol/cmd"
FOLLOW_TOPIC = "/follow/cmd"
# Events held for a reader that is not draining them (--nav only take()s).
EVENTS_KEPT = 256


class ClickGoal(NamedTuple):
    x: float
    y: float
    yaw: Optional[float]
    frame_id: str
    topic: str
    received: float        # time.monotonic() when it arrived


def _num(v):
    f = float(v)
    if not math.isfinite(f):
        raise ValueError("not finite")
    return f


def yaw_of(q) -> Optional[float]:
    """Yaw off a quaternion dict, or None if there is none or it is not one."""
    try:
        x, y, z, w = (_num(q.get(k, 0.0)) for k in ("x", "y", "z", "w"))
    except Exception:
        return None
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-9:
        return None
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def parse(schema_name, data, topic="", now=None) -> Optional[ClickGoal]:
    """A ClickGoal off one JSON message, or None if it is not a goal.

    foxglove.PoseInFrame:  {timestamp, frame_id, pose: {position, orientation}}
    geometry_msgs/PoseStamped:  {header: {stamp, frame_id}, pose: {...}}
    geometry_msgs/PointStamped: {header: {stamp, frame_id}, point: {x, y, z}}
    """
    try:
        msg = json.loads(bytes(data).decode("utf-8"))
        if not isinstance(msg, dict):
            return None
        if schema_name in POSE_SCHEMAS:
            frame = msg.get("frame_id")
            if frame is None:
                frame = (msg.get("header") or {}).get("frame_id", "")
            pose = msg["pose"]
            pos = pose["position"]
            x, y = _num(pos["x"]), _num(pos["y"])
            yaw = yaw_of(pose.get("orientation") or {})
        elif schema_name in POINT_SCHEMAS:
            frame = (msg.get("header") or {}).get("frame_id", "")
            pt = msg["point"]
            x, y = _num(pt["x"]), _num(pt["y"])
            yaw = None
        else:
            return None
    except Exception:
        return None
    return ClickGoal(x, y, yaw, str(frame or ""), str(topic),
                     time.monotonic() if now is None else float(now))


def parse_command(data):
    """A patrol command dict off one JSON message, or None if it is not one."""
    try:
        msg = json.loads(bytes(data).decode("utf-8"))
    except Exception:
        return None
    if not isinstance(msg, dict) or not isinstance(msg.get("cmd"), str):
        return None
    return msg


def to_map(goal, pose):
    """(x, y, yaw) of a ClickGoal in the map frame, given the truck's map
    pose (x, y, yaw); None for a frame it cannot place."""
    if goal is None:
        return None
    if goal.frame_id in (MAP_FRAME, ""):
        return (goal.x, goal.y, goal.yaw)
    if goal.frame_id == BASE_FRAME and pose is not None:
        px, py, pyaw = pose[:3]
        c, s = math.cos(pyaw), math.sin(pyaw)
        yaw = None if goal.yaw is None else goal.yaw + pyaw
        return (px + c * goal.x - s * goal.y, py + s * goal.x + c * goal.y, yaw)
    return None


class ClickGoals:
    """A foxglove ServerListener that keeps the newest clicked goal.

    Only on_client_advertise, on_client_unadvertise and on_message_data do
    anything. Never raises into the server's thread.

    The SDK (0.30) calls on_subscribe and on_unsubscribe on whatever listener
    it is given, for every channel every client opens or closes, and logs
    "Callback failed: AttributeError" for each one the listener lacks: 16 of
    them in the car's journal from one --nav run on 10-29. So those, and the
    connection graph pair, are here and do nothing.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._channels = {}     # (client id, channel id) -> (topic, schema, encoding)
        self._goal = None
        # ("goal", ClickGoal) and ("cmd", dict), oldest first.
        self._events = collections.deque(maxlen=EVENTS_KEPT)
        self.received = 0
        self.rejected = 0

    def on_subscribe(self, client, channel):
        return None

    def on_unsubscribe(self, client, channel):
        return None

    def on_connection_graph_subscribe(self):
        return None

    def on_connection_graph_unsubscribe(self):
        return None

    def on_client_advertise(self, client, channel):
        try:
            key = (int(client.id), int(channel.id))
            info = (str(channel.topic), str(channel.schema_name), str(channel.encoding))
            with self._lock:
                self._channels[key] = info
        except Exception:
            pass

    def on_client_unadvertise(self, client, client_channel_id):
        try:
            with self._lock:
                self._channels.pop((int(client.id), int(client_channel_id)), None)
        except Exception:
            pass

    def on_message_data(self, client, client_channel_id, data):
        try:
            with self._lock:
                info = self._channels.get((int(client.id), int(client_channel_id)))
            if info is None:
                return
            topic, schema, encoding = info
            if topic in (PATROL_TOPIC, FOLLOW_TOPIC):
                command = parse_command(data) if encoding == ENCODING else None
                kind = "cmd" if topic == PATROL_TOPIC else "follow"
                with self._lock:
                    if command is None:
                        self.rejected += 1
                    else:
                        self.received += 1
                        self._events.append((kind, command))
                return
            if schema not in POSE_SCHEMAS + POINT_SCHEMAS:
                return
            goal = parse(schema, data, topic) if encoding == ENCODING else None
            with self._lock:
                if goal is None:
                    self.rejected += 1
                else:
                    self.received += 1
                    self._goal = goal
                    self._events.append(("goal", goal))
        except Exception:
            pass

    def push(self, kind, command):
        """A command from somewhere other than Foxglove (the map page's
        HTTP POST, telemetry/buttons.py) into the same queue, in order with
        the rest: ("cmd", dict) or ("follow", dict). False if refused."""
        if kind not in ("cmd", "follow") or not isinstance(command, dict) \
                or not isinstance(command.get("cmd"), str):
            with self._lock:
                self.rejected += 1
            return False
        with self._lock:
            self.received += 1
            self._events.append((kind, command))
        return True

    def take(self) -> Optional[ClickGoal]:
        """The newest goal since the last take(), or None."""
        with self._lock:
            goal, self._goal = self._goal, None
        return goal

    def take_events(self):
        """Every click and patrol command since the last take_events(), in
        the order they came."""
        with self._lock:
            out = list(self._events)
            self._events.clear()
        return out

    def server_options(self):
        """The start_server() arguments that let Foxglove publish to it."""
        from foxglove.websocket import Capability
        return {"capabilities": [Capability.ClientPublish],
                "supported_encodings": [ENCODING],
                "server_listener": self}
