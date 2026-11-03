"""The phone pages: patrol and follow buttons, and the map, from a phone
browser, nothing installed.

    http://jtpi.local:8080       the buttons (or 192.168.68.57, or 10.42.0.1 on the hotspot)
    http://jtpi.local:8080/map   the map
    http://jtpi.local:8080/pad   re-pair the pad (hardware/pad_pair.py)

Three static files next to this one, buttons.html, map.html and pad.html,
each with a Buttons / Map / Pad tab bar at the top, and nothing from the
internet.

The buttons page does not talk to this server again after it has loaded.
It opens the Foxglove WebSocket on the telemetry port itself (subprotocol
foxglove.sdk.v1, the only one foxglove-sdk 0.30 accepts: it refuses the
older name foxglove.websocket.v1 with a 400; same messages as Studio's),
advertises a client channel on /patrol/cmd and one on /follow/cmd, and
sends exactly the JSON the drive layout's Publish panels send. So the
commands arrive through nav/click.ClickGoals like any button press from
Studio, and there is no second command path to keep in step: a test holds
the page's payloads to the layout's. It never sends throttle or steering;
there is no topic on this car that would take them. It subscribes to
/patrol/status and /follow/status for the state it shows.

The map page talks to this server instead, because what it draws (the SLAM
map as a picture) is cheaper made once here than decoded from Foxglove
messages in a phone, and it only asks while it is open and on screen:

  GET  /api/state     JSON, the page polls it about twice a second: the
                      patrol's page_state() (the waypoints on the map the
                      truck is on, a saved track not placed yet, the fit),
                      the truck's pose, the planner's path and the point it
                      feeds, the saved track names, the turning radius for
                      the drive line. The control loop builds it only while
                      the page is asking (wants_state), STATE_EVERY_S apart.
  GET  /api/map.png   the SLAM map round the truck (runner.MapView, 10 cm)
                      as a 3-colour PNG, about once a second, made again
                      only when the map has moved on. Its frame is in the
                      X-Map header: x0 y0 res width height, north up.
  GET  /api/pending   the walls of a saved track not placed yet, in its
                      own frame, for the nudge (thinned to PENDING_WALLS).
  POST /api/cmd       one patrol command, {"cmd": ...}, into the same
                      ClickGoals queue the Foxglove buttons go through
                      (ClickGoals.push), so the control loop takes it in
                      order with everything else. Patrol commands only:
                      autopilot/patrol.COMMANDS; never throttle or steering.
                      202 is "queued", not "done": a command with an "id"
                      is answered in the state's patrol.acks once the
                      control loop has applied it, and the page keeps what
                      it drew until then. The loop takes them with or
                      without a pad (__main__.padless_pass).

The pad page is its own page and not a button on the map, because the map
page is held under 40,000 bytes and had 47 of them left, and because the pad
wants re-pairing on a car with no SLAM map too: /pad and its two calls are
served whenever there is a PadRepair, map page or not.

  GET  /api/pad       JSON, PadRepair.status(): the state, why, the pad's
                      address once one is found, the seconds of scan left.
                      The same dict is /api/state's pad_repair.
  POST /api/cmd       {"cmd": "repair_pad"} starts the re-pair. Answered at
                      once, 202 started or 400 with the reason ("Stop auto
                      first", "Already re-pairing"). Not a patrol command
                      and never queued: it goes to the PadRepair, which runs
                      on a thread of its own and never touches the wheels.

Why stdlib http.server and not something in the WebSocket server: the
foxglove SDK's server speaks WebSocket and nothing else, and every other
option is a dependency. A thread a connection (ThreadingHTTPServer, at most
MAX_CONNECTIONS at once): it was one thread for everything, and a phone
browser opens a spare connection and sends nothing on it, which held every
other request -- the state, the map, a stroke -- for REQUEST_TIMEOUT_S
(5.00 s measured behind one idle socket). The threads are niced on Linux
(they inherit the serving thread's nice), so a page can never get ahead of
the control loop.

Like the rest of telemetry, start() never raises: a port already taken
costs the pages and nothing else.
"""

import http.server
import json
import os
import struct
import sys
import threading
import time
import zlib

DEFAULT_PORT = 8080
HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "buttons.html")
MAP_PAGE = os.path.join(HERE, "map.html")
PAD_PAGE = os.path.join(HERE, "pad.html")
REPAIR_COMMAND = "repair_pad"
# The page reads the WebSocket port from this tag; it ships with the default.
PORT_TAG = '<meta name="ws-port" content="8765">'
POLL_S = 1.0
NICE = 10
# A client that connects and sends nothing is dropped after this, so it
# cannot hold a thread for ever.
REQUEST_TIMEOUT_S = 5.0
# Connections served at once; one past it is closed straight away.
MAX_CONNECTIONS = 16
# The control loop builds the map page's state while the page has asked
# within STATE_WANTED_S, at most every STATE_EVERY_S. Long enough to last
# through a finger on the glass, when the page does not ask.
STATE_WANTED_S = 10.0
STATE_EVERY_S = 0.4
CMD_MAX_BYTES = 32_000
PENDING_WALLS = 1500
# The map picture's colours, by grid code: unknown is clear (the page draws
# its own hatching there), free a slate plainly lighter than the page,
# walls near white.
PALETTE = bytes((0, 0, 0, 68, 78, 94, 235, 235, 235))


def page_bytes(ws_port=8765):
    """The buttons page as served, with the WebSocket port it should dial."""
    with open(PAGE, encoding="utf-8") as f:
        html = f.read()
    return html.replace(PORT_TAG, f'<meta name="ws-port" content="{int(ws_port)}">').encode("utf-8")


def map_page_bytes():
    with open(MAP_PAGE, encoding="utf-8") as f:
        return f.read().encode("utf-8")


def pad_page_bytes():
    with open(PAD_PAGE, encoding="utf-8") as f:
        return f.read().encode("utf-8")


def _chunk(kind, data):
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))


def png_of(codes):
    """uint8 codes [nx, ny] (x, y) -> an indexed PNG, x across and y up the
    picture (row 0 is the highest y), unknown transparent."""
    import numpy as np
    img = np.ascontiguousarray(np.asarray(codes, np.uint8).T[::-1])
    img = np.minimum(img, 2)
    h, w = img.shape
    raw = np.zeros((h, w + 1), np.uint8)
    raw[:, 1:] = img
    return (b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 3, 0, 0, 0))
            + _chunk(b"PLTE", PALETTE)
            + _chunk(b"tRNS", b"\x00")
            + _chunk(b"IDAT", zlib.compress(raw.tobytes(), 6))
            + _chunk(b"IEND", b""))


class _Server(http.server.ThreadingHTTPServer):
    """A thread a connection, at most MAX_CONNECTIONS of them. Each thread
    is started by the serving thread, so on Linux it starts at its nice."""

    daemon_threads = True

    def __init__(self, *args, **kwargs):
        self._slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


def _handler(page):
    class Handler(http.server.BaseHTTPRequestHandler):
        timeout = REQUEST_TIMEOUT_S
        server_version = "crawler"
        sys_version = ""

        def _send(self, code, body, kind, extra=(), head_only=False):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            for k, v in extra:
                self.send_header(k, v)
            self.end_headers()
            if not head_only:
                self.wfile.write(body)

        def _get(self, head_only):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                return self._send(200, page.body, "text/html; charset=utf-8",
                                  head_only=head_only)
            if page.pad_body is not None:
                if path in ("/pad", "/pad.html"):
                    return self._send(200, page.pad_body, "text/html; charset=utf-8",
                                      head_only=head_only)
                if path == "/api/pad":
                    return self._send(200, page.pad_json(), "application/json",
                                      head_only=head_only)
            if page.map_body is not None:
                if path in ("/map", "/map.html"):
                    return self._send(200, page.map_body, "text/html; charset=utf-8",
                                      head_only=head_only)
                if path == "/api/state":
                    return self._send(200, page.state_json(), "application/json",
                                      head_only=head_only)
                if path == "/api/pending":
                    return self._send(200, page.pending_json(), "application/json",
                                      head_only=head_only)
                if path == "/api/map.png":
                    got = page.map_png()
                    if got is None:
                        return self._send(204, b"", "image/png", head_only=head_only)
                    body, header = got
                    # The same picture again is a 304 and nothing to decode.
                    tag = '"%08x"' % (zlib.crc32(body) & 0xFFFFFFFF)
                    if self.headers.get("If-None-Match") == tag:
                        return self._send(304, b"", "image/png", [("ETag", tag)],
                                          head_only=True)
                    return self._send(200, body, "image/png",
                                      [("X-Map", header), ("ETag", tag)],
                                      head_only=head_only)
            self.send_error(404)

        def do_GET(self):
            self._get(False)

        def do_HEAD(self):
            self._get(True)

        def do_POST(self):
            if self.path.split("?", 1)[0] != "/api/cmd" or (
                    page.map_body is None and page.pad_body is None):
                self.send_error(404)
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if not 0 < n <= CMD_MAX_BYTES:
                self.send_error(413 if n > CMD_MAX_BYTES else 411)
                return
            ok, why = page.command(self.rfile.read(n))
            self._send(202 if ok else 400, json.dumps({"ok": ok, "why": why}).encode(),
                       "application/json")

        def log_message(self, *args):
            pass

    return Handler


class ButtonPage:
    """The pages' HTTP server on a thread of its own.

    goals: the nav/click.ClickGoals the map page's commands go into; with
    none there is no map page. map_source: a callable for the newest
    runner.MapView (SlamRunner.map). library: the tracks.Library whose
    names the page lists. turn_radius_m: the drive line's radius.
    repair: the hardware/pad_pair.PadRepair behind /pad; with none there is
    no pad page."""

    def __init__(self, logger=None, host="0.0.0.0", port=DEFAULT_PORT, ws_port=8765,
                 goals=None, map_source=None, library=None, turn_radius_m=None,
                 clock=time.monotonic, repair=None):
        self.logger = logger
        self.host, self.port, self.ws_port = host, int(port), int(ws_port)
        self.goals, self.map_source, self.library = goals, map_source, library
        self.turn_radius_m = turn_radius_m
        self.clock = clock
        self.error = None
        self.body = b""
        self.map_body = None
        self.repair = repair
        self.pad_body = None
        self._server = None
        self._thread = None
        self._lock = threading.Lock()
        self._state = None            # (built at, dict) from the control loop
        self._asked = None            # when the page last asked for the state
        self._png = None              # ((MapView.t, version), body, header)
        self.requests = 0
        self.states_built = 0
        self.pngs_built = 0

    @property
    def serving(self):
        return self._server is not None

    # ---- the control loop's side -------------------------------------------------

    def wants_state(self, now=None):
        """True while the map page is open and the state is due."""
        now = self.clock() if now is None else now
        with self._lock:
            asked, state = self._asked, self._state
        if asked is None or now - asked > STATE_WANTED_S:
            return False
        return state is None or now - state[0] >= STATE_EVERY_S

    def offer_state(self, state, now=None):
        """The control loop's snapshot for the page: plain data, not
        changed after this (the page's thread reads it as it is)."""
        now = self.clock() if now is None else now
        with self._lock:
            self._state = (now, state)
            self.states_built += 1

    # ---- the page's side -----------------------------------------------------------

    def state_json(self):
        now = self.clock()
        with self._lock:
            self._asked = now
            state = self._state
            self.requests += 1
        out = {"ready": state is not None, "age_s": None if state is None else now - state[0],
               "turn_radius_m": self.turn_radius_m}
        if state is not None:
            snap = dict(state[1])
            patrol = dict(snap.get("patrol") or {})
            pending = patrol.get("pending")
            if pending is not None:
                pending = dict(pending)
                pending["walls"] = len(pending.get("walls") or [])
                patrol["pending"] = pending
            snap["patrol"] = patrol
            out.update(snap)
        if self.repair is not None:
            out["pad_repair"] = self._repair_status()
        try:
            out["tracks"] = self.library.names() if self.library is not None else []
        except Exception:
            out["tracks"] = []
        return json.dumps(out, separators=(",", ":")).encode("utf-8")

    def _repair_status(self):
        try:
            return self.repair.status()
        except Exception as e:
            return {"state": "failed", "why": str(e), "running": False}

    def pad_json(self):
        return json.dumps(self._repair_status(), separators=(",", ":")).encode("utf-8")

    def pending_json(self):
        with self._lock:
            state = self._state
        pending = ((state[1].get("patrol") or {}).get("pending")
                   if state is not None else None) or {}
        walls = pending.get("walls") or []
        if len(walls) > PENDING_WALLS:
            step = len(walls) / PENDING_WALLS
            walls = [walls[int(k * step)] for k in range(PENDING_WALLS)]
        return json.dumps({"id": pending.get("id"),
                           "walls": [[round(x, 2), round(y, 2)] for x, y in walls]},
                          separators=(",", ":")).encode("utf-8")

    def map_png(self):
        """(PNG bytes, X-Map header) of the newest map, or None."""
        try:
            view = self.map_source() if self.map_source is not None else None
        except Exception:
            view = None
        if view is None:
            return None
        key = (view.t, view.version)
        with self._lock:
            cached = self._png
        if cached is None or cached[0] != key:
            nx, ny = view.codes.shape
            header = f"{view.x0:.3f} {view.y0:.3f} {view.res:.4f} {nx} {ny}"
            cached = (key, png_of(view.codes), header)
            with self._lock:
                self._png = cached
                self.pngs_built += 1
        return cached[1], cached[2]

    def command(self, body):
        """(ok, why) for one POSTed command."""
        from crawler.autopilot import patrol as pt
        from crawler.nav import click as ck
        msg = ck.parse_command(body)
        if msg is None:
            return False, "not a command"
        if msg["cmd"] == REPAIR_COMMAND:
            if self.repair is None:
                return False, "no re-pair on this car"
            try:
                return self.repair.start()
            except Exception as e:
                return False, str(e)
        if msg["cmd"] not in pt.COMMANDS:
            return False, f"{msg['cmd']!r} is not a patrol command"
        if self.goals is None or not self.goals.push("cmd", msg):
            return False, "not taking commands"
        return True, ""

    # ---- the server ------------------------------------------------------------------

    def start(self) -> bool:
        try:
            self.body = page_bytes(self.ws_port)
            self.map_body = map_page_bytes() if self.goals is not None else None
            self.pad_body = pad_page_bytes() if self.repair is not None else None
            server = _Server((self.host, self.port), _handler(self))
            server.timeout = REQUEST_TIMEOUT_S
        except Exception as e:
            self.error = str(e)
            self._log("warning", f"No button page: {e}")
            return False
        self._server = server
        # Port 0 is the OS's pick (tests); say the one actually bound.
        self.port = server.server_address[1]
        self._thread = threading.Thread(target=self._serve, name="buttons", daemon=True)
        self._thread.start()
        self._log("info", f"Button page on http://{self.host}:{self.port}"
                  + (", the map on /map" if self.map_body is not None else "")
                  + (", the pad on /pad" if self.pad_body is not None else ""))
        return True

    def _serve(self):
        # Linux takes a thread id for PRIO_PROCESS and nices that one thread.
        # Only Linux: on macOS the same number is a pid, somebody else's.
        if sys.platform.startswith("linux"):
            try:
                os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), NICE)
            except Exception:
                pass
        try:
            self._server.serve_forever(poll_interval=POLL_S)
        except Exception as e:
            self._log("warning", f"Button page stopped: {e}")

    def stop(self):
        """Safe to call twice, and never raises."""
        server, self._server = self._server, None
        if server is None:
            return
        try:
            server.shutdown()
            server.server_close()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _log(self, level, message):
        if self.logger is not None:
            try:
                self.logger.log(level, "Buttons", message)
                return
            except Exception:
                pass
        print(f"[buttons] {message}")
