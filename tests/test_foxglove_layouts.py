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

# A goal from the 3D panel's pose tool, which nav/click.py listens for. The
# other topic Foxglove publishes to the car is /patrol/cmd, the Publish panel
# buttons (test_patrol.py checks those).
GOAL_TOPIC = "/move_base_simple/goal"


def panels(node, configs=None):
    """Every panel id a mosaic node lays out. With configs, a Tab panel's tabs
    are walked too: Studio keeps a tab's own mosaic in the Tab panel's config
    ({"activeTabIdx": 0, "tabs": [{"title": ..., "layout": <mosaic node>}]},
    TabPanelConfig in types/layouts.ts of the open Lichtblick fork), and the
    panels inside it keep their configs in the one flat configById
    (getAllPanelIds in util/layout.ts there)."""
    if node is None:
        return []
    if isinstance(node, str):
        out = [node]
        if configs is not None and node.startswith("Tab!"):
            for tab in configs[node]["tabs"]:
                out += panels(tab.get("layout"), configs)
        return out
    return panels(node["first"], configs) + panels(node["second"], configs)


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
    assert set(panels(layout["layout"], layout["configById"])) == set(layout["configById"])
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
    # Every picture but disparity: the hood camera took its pane. Disparity
    # is still published and recorded, and any Image panel can pick it.
    cams = set(tm.IMAGE_CHANNELS.values()) - {"/camera/disparity"}
    assert "/pantilt/image" in cams
    shown = [cfg["imageMode"]["imageTopic"] for k, cfg in drive.items()
             if k.startswith("Image!")]
    assert sorted(shown) == sorted(cams)
    assert {"/map", "/lidar/scan", "/voxel/points", "/voxel/low",
            "/goal/marker"} <= set(drive["3D!slam"]["topics"])
    # The planner is not on the drive layout: the model drives to a click.
    assert not any(t.startswith("/nav/") for cfg in drive.values()
                   for t, _ in topics_read(cfg))
    # A click on the map says where the goal is from the truck.
    assert any(cfg.get("topicPath") == "/goal/status" for cfg in drive.values())
    assert not any(k.startswith(("Image!", "3D!")) for k in tele)
    read = {t for cfg in tele.values() for t, _ in topics_read(cfg)}
    assert read == {"/vehicle", "/controller", "/imu", "/diagnostics",
                    "/slam/status", "/footprint"}


def test_a_click_on_the_drive_map_publishes_a_goal_pose_in_the_map_frame():
    """The 3D map panel's Publish tool is the pose kind on the goal topic,
    and the panel's display frame is the map, so a click arrives with
    frame_id "map", which goal.ClickGoal takes as it is."""
    drive = json.load(open(LAYOUTS[0]))["configById"]["3D!slam"]
    assert drive["publish"]["type"] == "pose"
    assert drive["publish"]["poseTopic"] == GOAL_TOPIC
    assert drive["followTf"] == "map"
    assert drive["topics"]["/goal/marker"]["visible"]


@pytest.mark.parametrize("path", LAYOUTS, ids=os.path.basename)
def test_every_panel_is_laid_out_once_tabs_included_and_every_config_is_used(path):
    layout = json.load(open(path))
    cfg = layout["configById"]
    shown = panels(layout["layout"], cfg)
    # A panel in two places would be one config drawn twice; Studio's own
    # layout code assumes each id is a single leaf.
    assert sorted(shown) == sorted(cfg)
    for key in shown:
        if key.startswith("Tab!"):
            tab = cfg[key]
            assert isinstance(tab["activeTabIdx"], int), key
            assert 0 <= tab["activeTabIdx"] < len(tab["tabs"]), key
            for t in tab["tabs"]:
                assert set(t) == {"title", "layout"} and t["title"], key


def test_the_drive_layout_keeps_the_statuses_in_tabs_over_a_grid_of_buttons():
    layout = json.load(open(LAYOUTS[0]))
    cfg = layout["configById"]
    right = layout["layout"]["second"]["second"]
    assert right["first"] == "3D!lidar"
    status, grid = right["second"]["first"], right["second"]["second"]
    tab = cfg[status]
    assert tab["activeTabIdx"] == 0
    assert [(t["title"], t["layout"]) for t in tab["tabs"]] == [
        ("Goal", "RawMessages!goal"), ("Patrol", "RawMessages!patrol"),
        ("Path", "RawMessages!path"), ("Follow", "RawMessages!follow")]
    rows = []
    while isinstance(grid, dict) and grid["direction"] == "column":
        rows.append(grid["first"])
        grid = grid["second"]
    rows.append(grid)
    pairs = [(r["first"], r["second"]) for r in rows]
    assert [tuple(k.split("!")[1] for k in p) for p in pairs] == [
        ("patrol_edit", "patrol_goal"), ("patrol_start", "patrol_pause"),
        ("patrol_mode", "patrol_trace"), ("patrol_undo", "patrol_clear"),
        ("follow_on", "follow_off")]
    for k, v in cfg.items():
        if k.startswith("RawMessages!"):
            # "all" opens every field; FONT_SIZE_OPTIONS in Lichtblick's
            # RawMessagesCommon/constants.ts is the list the panel offers.
            assert v["expansion"] == "all", k
            assert v["fontSize"] in (8, 9, 10, 11, 12, 14, 16, 18, 24), k
