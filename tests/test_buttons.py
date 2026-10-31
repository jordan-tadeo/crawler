"""telemetry/buttons.py and buttons.html, the phone page for patrol and follow.

The page sends its commands straight over the Foxglove WebSocket, so the
things that can go wrong are the page drifting from the Publish panels it
stands in for, the page reaching for the internet the car may not have, the
server refusing the page's handshake, and the HTTP side itself.
"""

import base64
import json
import os
import re
import socket
import struct
import time
import types
import urllib.error
import urllib.request

import pytest

from crawler import __main__ as cm
from crawler.nav import click
from crawler.telemetry import buttons as bt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVE = os.path.join(ROOT, "deploy", "foxglove-drive.json")
HTML = open(bt.PAGE, encoding="utf-8").read()
MAP_HTML = open(bt.MAP_PAGE, encoding="utf-8").read()


def page_commands():
    """The page's {"patrol_start": {topic, schema, value}, ...} block."""
    block = re.search(r'<script type="application/json" id="cmds">(.*?)</script>',
                      HTML, re.S).group(1)
    return json.loads(block)


def layout_publish_panels():
    cfg = json.load(open(DRIVE))["configById"]
    return {k.split("!", 1)[1]: v for k, v in cfg.items() if k.startswith("Publish!")}


def test_the_page_sends_exactly_what_the_drive_layouts_publish_panels_send():
    page = page_commands()
    panels = layout_publish_panels()
    assert set(page) == set(panels)
    for key, panel in panels.items():
        assert page[key] == {"topic": panel["topicName"], "schema": panel["datatype"],
                             "value": panel["value"]}, key
        assert click.parse_command(page[key]["value"].encode()) is not None, key


def test_every_button_on_the_page_is_one_of_its_commands():
    page = page_commands()
    used = set(re.findall(r'data-cmd="([a-z_]+)"', HTML))
    used |= set(re.findall(r'"(follow_(?:on|off))"', HTML.split('id="cmds"', 1)[1].split("</script>", 1)[1]))
    assert used == set(page)


def test_the_page_only_ever_sends_patrol_and_follow_commands():
    # Never throttle or steering: the only topics it can write are these two.
    assert {c["topic"] for c in page_commands().values()} == {
        click.PATROL_TOPIC, click.FOLLOW_TOPIC}


@pytest.mark.parametrize("html", [HTML, MAP_HTML], ids=["buttons", "map"])
def test_the_pages_load_nothing_from_anywhere(html):
    # On the car's own hotspot there is no internet: no CDN, no web font.
    # The only links are the two tabs, to this server's own pages.
    assert not re.search(r"""https?://|["'(]//""", html)
    assert not re.search(r"""\bsrc\s*=""", html)
    assert set(re.findall(r"""\bhref\s*=\s*"([^"]*)\"""", html)) == {"/", "/map"}
    assert "@import" not in html and "url(" not in html
    assert len(html.encode()) < 40_000


def test_the_map_page_only_sends_patrol_commands():
    from crawler.autopilot import patrol as pt
    sent = set(re.findall(r"""\bsend\(\s*"([a-z_]+)\"""", MAP_HTML))
    sent |= set(re.findall(r"""\?\s*"([a-z_]+)"\s*:\s*"([a-z_]+)"\)""", MAP_HTML)[0])
    assert {"stroke", "unstroke", "move", "delete", "save", "load", "forget", "place",
            "clear", "start", "pause"} <= sent
    assert sent <= set(pt.COMMANDS)
    # Its one way out is /api/cmd, and nothing on it says throttle or steering.
    assert re.findall(r"""fetch\("([^"]+)\"""", MAP_HTML) and all(
        u in ("/api/cmd", "/api/state", "/api/map.png", "/api/pending")
        for u in re.findall(r"""fetch\("([^"]+)\"""", MAP_HTML))
    assert not re.search(r"throttle|steer|/vehicle|/camera|image_raw", MAP_HTML, re.I)


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, b""


def test_the_page_is_served_with_the_websocket_port_it_should_dial():
    page = bt.ButtonPage(host="127.0.0.1", port=0, ws_port=9876)
    assert page.start()
    try:
        status, headers, body = get(f"http://127.0.0.1:{page.port}/")
        assert status == 200
        assert headers["Content-Type"].startswith("text/html")
        assert b'<meta name="ws-port" content="9876">' in body
        assert bt.PORT_TAG.encode() not in body
        assert get(f"http://127.0.0.1:{page.port}/index.html")[0] == 200
        assert get(f"http://127.0.0.1:{page.port}/../buttons.py")[0] == 404
        assert get(f"http://127.0.0.1:{page.port}/favicon.ico")[0] == 404
    finally:
        page.stop()
        page.stop()


def test_a_taken_port_costs_the_page_and_nothing_else():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        page = bt.ButtonPage(host="127.0.0.1", port=s.getsockname()[1])
        assert page.start() is False
        assert page.error and not page.serving
        page.stop()


def test_the_page_is_up_only_when_the_server_takes_commands():
    class Log:
        def log(self, *a):
            pass

    serving = types.SimpleNamespace(serving=True, host="127.0.0.1", port=8765)
    args = types.SimpleNamespace(buttons_port=0)
    assert cm.start_buttons(args, serving, object(), Log()) is None
    args.buttons_port = 8080
    assert cm.start_buttons(args, serving, None, Log()) is None
    off = types.SimpleNamespace(serving=False, host="127.0.0.1", port=8765)
    assert cm.start_buttons(args, off, object(), Log()) is None
    assert cm.parse_args([]).buttons_port == 8080


# ---- the page's WebSocket half, against the real Foxglove server ----------

def ws_frame(payload, opcode):
    """One masked client frame (RFC 6455), FIN set."""
    mask = os.urandom(4)
    n = len(payload)
    head = bytes([0x80 | opcode])
    if n < 126:
        head += bytes([0x80 | n])
    else:
        head += bytes([0x80 | 126]) + struct.pack(">H", n)
    return head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload))


def page_protocols():
    return json.loads(re.search(r"var PROTOCOLS=(\[.*?\]);", HTML).group(1))


def test_the_server_takes_the_pages_handshake_and_its_commands():
    """What the page does in a browser, byte for byte: offer its subprotocols,
    advertise one client channel per command topic, send each command as a
    messageData frame. Every one has to come out of ClickGoals."""
    pytest.importorskip("foxglove")
    from crawler.telemetry import telemetry as tm

    goals = click.ClickGoals()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    tele = tm.Telemetry(host="127.0.0.1", port=port, server_options=goals.server_options())
    assert tele.start(serve=True) and tele.serving
    try:
        sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        key = base64.b64encode(os.urandom(16)).decode()
        sock.sendall((f"GET / HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\n"
                      f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                      "Sec-WebSocket-Version: 13\r\n"
                      f"Sec-WebSocket-Protocol: {', '.join(page_protocols())}\r\n\r\n").encode())
        reply = b""
        while b"\r\n\r\n" not in reply:
            reply += sock.recv(4096)
        assert reply.startswith(b"HTTP/1.1 101"), reply[:200]

        cmds = page_commands()
        chans = {}
        for c in cmds.values():
            chans.setdefault(c["topic"], {"id": len(chans) + 1, "topic": c["topic"],
                                          "encoding": "json", "schemaName": c["schema"]})
        sock.sendall(ws_frame(json.dumps({"op": "advertise", "channels": list(chans.values())}).encode(), 1))
        time.sleep(0.3)
        for c in cmds.values():
            sock.sendall(ws_frame(b"\x01" + struct.pack("<I", chans[c["topic"]]["id"])
                                  + c["value"].encode(), 2))
        got = []
        deadline = time.time() + 5
        while len(got) < len(cmds) and time.time() < deadline:
            got += goals.take_events()
            time.sleep(0.05)
        sock.close()
        want = [("cmd" if c["topic"] == click.PATROL_TOPIC else "follow", json.loads(c["value"]))
                for c in cmds.values()]
        assert got == want
    finally:
        tele.stop()


# ---- the map page's HTTP half ---------------------------------------------

def decode_png(body):
    """(width, height, palette indices [h][w], transparent count) of an
    8-bit indexed PNG with filter 0 rows, the way png_of writes one."""
    import zlib
    assert body[:8] == b"\x89PNG\r\n\x1a\n"
    pos, chunks = 8, {}
    while pos < len(body):
        n = struct.unpack(">I", body[pos:pos + 4])[0]
        kind = body[pos + 4:pos + 8]
        chunks.setdefault(kind, b"")
        chunks[kind] += body[pos + 8:pos + 8 + n]
        pos += 12 + n
    w, h, depth, colour = struct.unpack(">IIBB", chunks[b"IHDR"][:10])
    assert depth == 8 and colour == 3 and len(chunks[b"PLTE"]) == 9
    raw = zlib.decompress(chunks[b"IDAT"])
    rows = [list(raw[r * (w + 1) + 1:(r + 1) * (w + 1)]) for r in range(h)]
    assert all(raw[r * (w + 1)] == 0 for r in range(h))
    return w, h, rows, len(chunks[b"tRNS"])


def map_view():
    import numpy as np
    from crawler.mapping import runner as rn
    codes = np.zeros((4, 3), np.uint8)       # x 0..3, y 0..2
    codes[0, 0] = 2                          # occupied, lowest x and y
    codes[3, 2] = 1                          # free, highest x and y
    return rn.MapView(t=1.0, version=7, x0=-1.0, y0=2.0, res=0.1, codes=codes)


def post(url, body):
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, None


class Library:
    def names(self):
        return [{"name": "kitchen", "points": 4, "saved_at": 1.0}]


@pytest.fixture
def mapped():
    goals = click.ClickGoals()
    clock = [100.0]
    page = bt.ButtonPage(host="127.0.0.1", port=0, goals=goals, map_source=map_view,
                         library=Library(), turn_radius_m=0.7, clock=lambda: clock[0])
    assert page.start()
    try:
        yield page, goals, clock, f"http://127.0.0.1:{page.port}"
    finally:
        page.stop()


def test_the_map_page_and_its_state_are_served(mapped):
    page, goals, clock, url = mapped
    status, headers, body = get(url + "/map")
    assert status == 200 and headers["Content-Type"].startswith("text/html")
    assert b'href="/map"' in body
    # Nothing is built for a page nobody has open.
    assert not page.wants_state()
    st = json.loads(get(url + "/api/state")[2])
    assert st["ready"] is False and st["turn_radius_m"] == 0.7
    assert st["tracks"][0]["name"] == "kitchen"
    assert page.wants_state(), "asked for: due"
    walls = [[i * 0.01, 1.0] for i in range(4000)]
    page.offer_state({"patrol": {"points": [[1.0, 2.0], None],
                                 "pending": {"name": "kitchen", "points": [[0, 0]],
                                             "walls": walls, "id": 5}},
                      "path": [[0, 0], [1, 1]], "aim": [1, 1]})
    assert not page.wants_state(), "just built"
    clock[0] += bt.STATE_EVERY_S
    assert page.wants_state()
    clock[0] += bt.STATE_WANTED_S + 1
    assert not page.wants_state(), "the page went away"
    st = json.loads(get(url + "/api/state")[2])
    assert st["ready"] and st["patrol"]["points"] == [[1.0, 2.0], None]
    assert st["patrol"]["pending"]["walls"] == 4000, "a count, not the walls"
    assert st["path"] == [[0, 0], [1, 1]]
    pend = json.loads(get(url + "/api/pending")[2])
    assert pend["id"] == 5 and len(pend["walls"]) == bt.PENDING_WALLS
    assert pend["walls"][0] == [0.0, 1.0]


def test_the_map_picture_is_north_up_with_its_frame_in_a_header(mapped):
    page, goals, clock, url = mapped
    status, headers, body = get(url + "/api/map.png")
    assert status == 200 and headers["Content-Type"] == "image/png"
    assert headers["X-Map"] == "-1.000 2.000 0.1000 4 3"
    w, h, rows, transparent = decode_png(body)
    assert (w, h) == (4, 3) and transparent == 1
    # Row 0 is the highest y: the free cell (3, 2) top right, the wall
    # (0, 0) bottom left.
    assert rows[0][3] == 1 and rows[2][0] == 2 and sum(map(sum, rows)) == 3
    get(url + "/api/map.png")
    assert page.pngs_built == 1, "made again only when the map moves on"


def test_a_posted_command_goes_into_the_click_queue(mapped):
    page, goals, clock, url = mapped
    status, reply = post(url + "/api/cmd",
                         json.dumps({"cmd": "stroke", "points": [[0, 0], [1, 0]]}).encode())
    assert status == 202 and reply["ok"]
    assert goals.take_events() == [("cmd", {"cmd": "stroke", "points": [[0, 0], [1, 0]]})]
    for bad in (b'{"cmd": "throttle", "value": 1}', b'{"cmd": "on"}', b"[1]", b"nope",
                b'{"steering": 0.5}'):
        status, reply = post(url + "/api/cmd", bad)
        assert status == 400 and not reply["ok"], bad
    assert post(url + "/api/cmd", b"x" * (bt.CMD_MAX_BYTES + 1))[0] == 413
    assert post(url + "/api/state", b'{"cmd": "start"}')[0] == 404
    assert goals.take_events() == []


def test_without_a_map_there_is_no_map_page():
    page = bt.ButtonPage(host="127.0.0.1", port=0)
    assert page.start()
    try:
        url = f"http://127.0.0.1:{page.port}"
        assert get(url + "/")[0] == 200
        for path in ("/map", "/api/state", "/api/map.png", "/api/pending"):
            assert get(url + path)[0] == 404, path
        assert post(url + "/api/cmd", b'{"cmd": "start"}')[0] == 404
    finally:
        page.stop()


def test_no_map_yet_is_an_empty_answer():
    page = bt.ButtonPage(host="127.0.0.1", port=0, goals=click.ClickGoals(),
                         map_source=lambda: None)
    assert page.start()
    try:
        assert get(f"http://127.0.0.1:{page.port}/api/map.png")[0] == 204
    finally:
        page.stop()


def test_the_map_page_state_is_the_patrols_and_the_planners():
    from crawler.autopilot import patrol as pt
    p = pt.Patrol()
    ap = types.SimpleNamespace(patrol=p, path=types.SimpleNamespace(
        marker=lambda: {"path": [(0.123, 4.567)], "aim": (1.0, 2.0), "fed": "path"}))
    st = cm.map_page_state(ap)
    json.dumps(st)
    assert st["path"] == [[0.12, 4.57]] and st["aim"] == [1.0, 2.0]
    assert st["patrol"]["points"] == []
    assert st["pad"] is True and st["mode"] is None
    st = cm.map_page_state(ap, pad=False, mode="manual")
    assert st["pad"] is False and st["mode"] == "manual"


def test_a_connection_that_says_nothing_holds_up_nobody_else(mapped):
    """A phone browser opens a spare connection and sends nothing on it. With
    one thread for everything that held every request for REQUEST_TIMEOUT_S
    (5.00 s measured)."""
    page, goals, clock, url = mapped
    idle = [socket.create_connection(("127.0.0.1", page.port)) for _ in range(3)]
    try:
        t = time.monotonic()
        assert get(url + "/api/state")[0] == 200
        assert post(url + "/api/cmd", b'{"cmd": "start"}')[0] == 202
        assert time.monotonic() - t < 1.0
    finally:
        for s in idle:
            s.close()


def test_the_same_map_picture_again_is_a_304(mapped):
    page, goals, clock, url = mapped
    status, headers, body = get(url + "/api/map.png")
    tag = headers["ETag"]
    assert status == 200 and tag
    req = urllib.request.Request(url + "/api/map.png", headers={"If-None-Match": tag})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=5)
    assert e.value.code == 304
    req = urllib.request.Request(url + "/api/map.png", headers={"If-None-Match": '"0"'})
    with urllib.request.urlopen(req, timeout=5) as r:
        assert r.status == 200 and r.read() == body


def test_a_command_with_an_id_is_answered_in_the_patrols_acks():
    """The page's ack: [id, taken, why] once the control loop has applied
    it, taken or refused, and nothing for a command with no id (Foxglove's).
    why is the refusal's own warning, word for word."""
    from test_goal_reseed import crop_at
    from crawler.autopilot import patrol as pt
    goals = click.ClickGoals()
    page = bt.ButtonPage(goals=goals)
    p = pt.Patrol()
    for body in ({"cmd": "stroke", "points": [[0, 0], [1, 0]], "id": "k-1"},
                 {"cmd": "move", "i": 9, "x": 0, "y": 0, "id": "k-2"},
                 {"cmd": "start"},
                 {"cmd": "pause", "id": 7}):
        assert page.command(json.dumps(body).encode())[0]
    p.poll(goals, crop_at())
    assert p.page_state()["acks"] == [["k-1", True, ""],
                                      ["k-2", False, "Move refused: no waypoint 9"]]
    assert len(p.points) == 2


def test_a_silent_refusal_still_says_why_in_its_ack():
    """An Undo with nothing left to undo was refused with no word said, and
    the page fell back on whatever warning was last said (or "refused"),
    then kept it on screen next to the next loop drawn. Each refusal now
    names itself, and an older warning is never borrowed for it."""
    from test_goal_reseed import crop_at
    from crawler.autopilot import patrol as pt
    goals = click.ClickGoals()
    page = bt.ButtonPage(goals=goals)
    p = pt.Patrol()
    p._log("warning", "Something old and unrelated")
    for body in ({"cmd": "unstroke", "id": "u-1"},
                 {"cmd": "stroke", "points": [], "id": "u-2"},
                 {"cmd": "stroke", "points": [[0, 0], [1, 0]], "id": "u-3"},
                 {"cmd": "unstroke", "id": "u-4"},
                 {"cmd": "unstroke", "id": "u-5"}):
        assert page.command(json.dumps(body).encode())[0]
    p.poll(goals, crop_at())
    acks = p.page_state()["acks"]
    assert [a[:2] for a in acks] == [["u-1", False], ["u-2", False], ["u-3", True],
                                     ["u-4", True], ["u-5", False]]
    assert acks[0][2] == acks[4][2] == "Nothing to undo: every stroke is already taken back"
    assert acks[1][2].startswith("Stroke refused")
    assert all("unrelated" not in a[2] for a in acks)
