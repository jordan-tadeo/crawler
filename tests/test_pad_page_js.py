"""The pad page's own behaviour, and the map page's tab to it, in a
phone-sized headless Chromium: tests/js/pad_page.js against a stand-in car.

What it holds the page to: the button sends {"cmd": "repair_pad"} and
nothing else, every PadRepair state is shown in words with the button off
while a run is going, a refusal ("Stop auto first") is said and leaves the
button usable, and the map page's Pad tab sits in the one row and goes there.

Needs node and playwright, as tests/test_map_page_js.py does; skips without.
"""

import json
import os
import subprocess

import pytest

from crawler.telemetry import buttons as bt

from test_map_page_js import _env, _have_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "js", "pad_page.js")


@pytest.fixture(scope="module")
def results():
    if not _have_playwright():
        pytest.skip("no node with playwright (set NODE_PATH or CRAWLER_NODE_PATH)")
    run = subprocess.run(["node", SCRIPT, bt.PAD_PAGE, bt.MAP_PAGE], env=_env(),
                         capture_output=True, text=True, timeout=180)
    assert run.returncode == 0, run.stderr[-2000:]
    return json.loads(run.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("name", [
    "the_button_sends_repair_pad_and_shows_each_state",
    "a_refusal_says_why_and_leaves_the_button_usable",
    "the_map_page_has_a_pad_tab",
])
def test_the_pad_page(results, name):
    assert results.get(name) == "ok", results.get(name)
