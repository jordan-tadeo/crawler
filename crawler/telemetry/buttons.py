"""The button page: patrol and follow from a phone browser, nothing installed.

    http://jtpi.local:8080   (or 192.168.68.57, or 10.42.0.1 on the hotspot)

One static file, buttons.html next to this one, and that is all this serves.
The page does not talk to this server again after it has loaded. It opens
the Foxglove WebSocket on the telemetry port itself (subprotocol
foxglove.sdk.v1, the only one foxglove-sdk 0.30 accepts: it refuses the
older name foxglove.websocket.v1 with a 400; same messages as Studio's), advertises a client channel on /patrol/cmd
and one on /follow/cmd, and sends exactly the JSON the drive layout's Publish
panels send. So the commands arrive through nav/click.ClickGoals like any
button press from Studio, and there is no second command path to keep in
step: a test holds the page's payloads to the layout's. It never sends
throttle or steering; there is no topic on this car that would take them.

It subscribes to /patrol/status and /follow/status for the state it shows.
Those are JSON at 2 and 5 Hz, and with the MCAP ring recording they are
built anyway.

Why stdlib http.server and not something in the WebSocket server: the
foxglove SDK's server speaks WebSocket and nothing else, and every other
option is a dependency. A plain HTTPServer answers one request at a time on
one thread, which is plenty for a 15 KB page fetched once per phone. The
thread sleeps in select() between requests (a wake every POLL_S to check for
shutdown) and is niced on Linux, so a page load can never get ahead of the
control loop.

Like the rest of telemetry, start() never raises: a port already taken
costs the page and nothing else.
"""

import http.server
import os
import sys
import threading

DEFAULT_PORT = 8080
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "buttons.html")
# The page reads the WebSocket port from this tag; it ships with the default.
PORT_TAG = '<meta name="ws-port" content="8765">'
POLL_S = 1.0
NICE = 10
# A client that connects and sends nothing is dropped after this, so it
# cannot hold the one serving thread.
REQUEST_TIMEOUT_S = 5.0


def page_bytes(ws_port=8765):
    """The page as served, with the WebSocket port it should dial."""
    with open(PAGE, encoding="utf-8") as f:
        html = f.read()
    return html.replace(PORT_TAG, f'<meta name="ws-port" content="{int(ws_port)}">').encode("utf-8")


def _handler(body):
    class Handler(http.server.BaseHTTPRequestHandler):
        timeout = REQUEST_TIMEOUT_S
        server_version = "crawler"
        sys_version = ""

        def _head(self):
            if self.path.split("?", 1)[0] not in ("/", "/index.html"):
                self.send_error(404)
                return False
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            return True

        def do_GET(self):
            if self._head():
                self.wfile.write(body)

        def do_HEAD(self):
            self._head()

        def log_message(self, *args):
            pass

    return Handler


class ButtonPage:
    """The page's HTTP server on a thread of its own."""

    def __init__(self, logger=None, host="0.0.0.0", port=DEFAULT_PORT, ws_port=8765):
        self.logger = logger
        self.host, self.port, self.ws_port = host, int(port), int(ws_port)
        self.error = None
        self._server = None
        self._thread = None

    @property
    def serving(self):
        return self._server is not None

    def start(self) -> bool:
        try:
            body = page_bytes(self.ws_port)
            server = http.server.HTTPServer((self.host, self.port), _handler(body))
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
        self._log("info", f"Button page on http://{self.host}:{self.port}")
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
