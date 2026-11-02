"""The map page's own behaviour, in a phone-sized headless Chromium with
touch events: tests/js/map_page.js against a stand-in car.

What it holds the page to: a stroke is one command, sent on the lift, with
nothing sent, polled or redrawn whole while the finger is down; a state poll
never takes an unanswered stroke off the screen; no answer in a few seconds,
or a refusal, is a "car didn't take it" with the points kept and a Retry
(a stroke only: a refused Undo is said and dropped, never resent); a
waypoint lands under the finger after a banner resizes the map box;
the not-ready banner says why; unknown space is drawn apart from free.

Needs node and the playwright package with its Chromium. NODE_PATH (or
CRAWLER_NODE_PATH) is where node finds playwright; without it this skips.
"""

import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from crawler.telemetry import buttons as bt

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "js", "map_page.js")


def _env():
    env = dict(os.environ)
    extra = os.environ.get("CRAWLER_NODE_PATH")
    if extra:
        env["NODE_PATH"] = extra + (os.pathsep + env["NODE_PATH"] if env.get("NODE_PATH") else "")
    return env


def _have_playwright():
    node = shutil.which("node")
    if node is None:
        return False
    try:
        return subprocess.run([node, "-e", "require.resolve('playwright')"], env=_env(),
                              capture_output=True, timeout=20).returncode == 0
    except Exception:
        return False


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    if not _have_playwright():
        pytest.skip("no node with playwright (set NODE_PATH or CRAWLER_NODE_PATH)")
    png = tmp_path_factory.mktemp("map") / "map.png"
    png.write_bytes(bt.png_of(np.ones((40, 40), np.uint8)))     # 4 m of free floor
    run = subprocess.run(["node", SCRIPT, bt.MAP_PAGE, str(png)], env=_env(),
                         capture_output=True, text=True, timeout=240)
    assert run.returncode == 0, run.stderr[-2000:]
    return json.loads(run.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("name", [
    "stroke_is_one_command_on_lift",
    "poll_never_clobbers_an_unanswered_stroke",
    "no_answer_is_an_error_with_retry",
    "a_refusal_says_why_and_keeps_the_points",
    "a_waypoint_lands_under_the_finger_after_a_banner",
    "a_refused_undo_never_comes_back_for_the_next_loop",
    "a_finger_down_never_times_out_a_stroke_the_car_took",
    "the_not_ready_banner_says_why",
    "unknown_is_drawn_apart_from_free",
    "explore_is_a_toggle_next_to_start_and_draws_its_target",
    "the_clean_button_sends_the_tag_and_shows_the_cars",
])
def test_the_map_page(results, name):
    assert results.get(name) == "ok", results.get(name)
