"""deploy/foxglove-*.json, the layouts I import into Foxglove.

Foxglove says nothing about a panel on a topic nobody publishes: it just
sits empty. So every topic a layout names is checked against the channels
crawler/telemetry/telemetry.py actually opens.
"""

import glob
import json
import os
import re

import pytest

from crawler.telemetry import telemetry as tm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYOUTS = sorted(glob.glob(os.path.join(ROOT, "deploy", "foxglove-*.json")))

PUBLISHED = (set(tm.JSON_TOPICS.values()) | set(tm.IMAGE_CHANNELS.values())
             | set(tm.RAW_DEPTH_CHANNELS.values())
             | set(tm.LIDAR_VIEW_CHANNELS.values())
             | set(tm.SLAM_CHANNELS.values()) | set(tm.VOXEL_CHANNELS.values())
             | set(tm.NAV_CHANNELS.values()))

# The one topic Foxglove publishes to the car: a goal from the 3D panel's
# pose tool, which nav/click.py listens for.
GOAL_TOPIC = "/move_base_simple/goal"


def panels(node):
    if isinstance(node, str):
        return [node]
    return panels(node["first"]) + panels(node["second"])


def path_topic(path):
    """"/vehicle.throttle" or "/controller.axes[0]" -> the topic."""
    return re.match(r"(/[^.\[{]*)", path).group(1)


def topics_read(cfg):
    """Every topic one panel config reads, as (topic, where) pairs."""
    out = []
    image = cfg.get("imageMode", {}).get("imageTopic")
    if image:
        out.append((image, "imageTopic"))
    for topic in cfg.get("topics", {}):
        out.append((topic, "topics"))
    for p in cfg.get("paths", []):
        out.append((path_topic(p["value"]), p["value"]))
    if "topicPath" in cfg:
        out.append((path_topic(cfg["topicPath"]), cfg["topicPath"]))
    return out


def fields_of(topic):
    for key, t in tm.JSON_TOPICS.items():
        if t == topic:
            return tm.JSON_SCHEMAS[key]["schema"]["properties"]
    return None


def test_there_are_two_layouts_one_to_drive_by_and_one_of_numbers():
    names = [os.path.basename(p) for p in LAYOUTS]
    assert names == ["foxglove-drive.json", "foxglove-telemetry.json"]


@pytest.mark.parametrize("path", LAYOUTS, ids=os.path.basename)
def test_a_layout_reads_only_topics_the_car_publishes(path):
    layout = json.load(open(path))
    assert set(panels(layout["layout"])) == set(layout["configById"])
    seen = 0
    for key, cfg in layout["configById"].items():
        for topic, where in topics_read(cfg):
            seen += 1
            assert topic in PUBLISHED, f"{key}: {where}"
            # A plot of a JSON field the schema does not have plots nothing.
            props = fields_of(topic)
            if props is not None and where.startswith(topic + "."):
                field = re.match(r"[A-Za-z_0-9]+", where[len(topic) + 1:]).group(0)
                assert field in props, f"{key}: {where}"
        publish = cfg.get("publish")
        if publish:
            assert publish["poseTopic"] == GOAL_TOPIC, key
    assert seen > 0


def test_the_drive_layout_is_pictures_and_the_telemetry_one_is_not():
    drive, tele = (json.load(open(p))["configById"] for p in LAYOUTS)
    cams = set(tm.IMAGE_CHANNELS.values())
    assert {cfg["imageMode"]["imageTopic"] for k, cfg in drive.items()
            if k.startswith("Image!")} == cams
    assert {"/map", "/lidar/scan", "/voxel/points", "/voxel/low",
            "/nav/path"} <= set(drive["3D!slam"]["topics"])
    # A click on the map says what the goto made of it.
    assert any(cfg.get("topicPath") == "/nav/status" for cfg in drive.values())
    assert not any(k.startswith(("Image!", "3D!")) for k in tele)
    read = {t for cfg in tele.values() for t, _ in topics_read(cfg)}
    assert read == {"/vehicle", "/controller", "/imu", "/diagnostics",
                    "/slam/status", "/footprint"}
