"""Re-pair the Xbox pad from the phone, no laptop.

An Xbox One pad remembers one Bluetooth host. Pair it with the PC and the Pi
is left holding a bond the pad has forgotten, so every connect fails with
errno 52, EBADE, which is HCI 0x06 "PIN or Key Missing". The fix by hand, as
jt on the Pi, was always the same five commands:

    bluetoothctl remove <mac>
    scan on              ...then hold the pair button until the light flashes fast
    pair <mac>
    trust <mac>
    connect <mac>

PadRepair is those five, started by {"cmd": "repair_pad"} on POST /api/cmd
(telemetry/buttons.py) and run on a thread of its own, never on the control
loop. It finds the pad by name, never by MAC, so a different pad works too:
every known device called "Xbox Wireless Controller" is removed, and the first
one a scan turns up is paired, trusted and connected.

What it never does: touch the wheels. Removing the pad's bond drops the pad
like any other pad loss, and the control loop does what it always does about
that: auto parks, record ends, the wheels hold neutral, and Joystick.try_connect
picks the new js device up once it connects. Which is also why it refuses to
start in auto: losing the pad is not something to do to a car that is driving.

The scan is its own bluetoothctl, held open for the length of the scan, because
BlueZ ties discovery to the D-Bus client that asked for it: a one-shot
`bluetoothctl scan on` stops discovering the moment it exits. It is stopped
before the pair rather than after. The Pi's radio is one chip for both WiFi
and Bluetooth, and an inquiry running under a page and an SMP exchange is the
one thing that made pairing flaky; the pad keeps its place in BlueZ's device
list for TemporaryTimeout (30 s) after the scan stops, plenty for a pair.

Every bluetoothctl call has a timeout. A user that cannot reach bluetoothd on
the system bus gets a bluetoothctl that prints "Waiting to connect to
bluetoothd..." and hangs for ever, and that has to come back as a failure
with a reason, not a thread stuck until the next boot.
"""

import re
import shutil
import subprocess
import threading
import time

PAD_NAME = "Xbox Wireless Controller"
SCAN_S = 90.0
POLL_S = 1.0
# Seconds for each one-shot call. Pairing is the slow one: the pad has to
# answer, and Just Works over SSP takes a few seconds on its own.
CALL_S = 10.0
PAIR_S = 30.0
CONNECT_S = 20.0
CONNECT_TRIES = 3

IDLE = "idle"
REMOVING = "removing"
WAITING = "waiting"
PAIRING = "pairing"
CONNECTED = "connected"
FAILED = "failed"
TIMED_OUT = "timed_out"
STATES = (IDLE, REMOVING, WAITING, PAIRING, CONNECTED, FAILED, TIMED_OUT)
RUNNING = (REMOVING, WAITING, PAIRING)

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x01|\x02")
_DEVICE = re.compile(r"^Device\s+((?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2})\s*(.*)$")


def is_pad(name) -> bool:
    """The pad by its name, whatever the case and the padding round it."""
    return isinstance(name, str) and name.strip().lower() == PAD_NAME.lower()


def parse_devices(text):
    """[(mac, name)] off `bluetoothctl devices`. A device still without a
    name shows its address, dashed, as the name; that is kept as it reads."""
    out = []
    for line in _ANSI.sub("", text or "").splitlines():
        # The interactive prompt's [CHG]/[NEW] lines are not in a one-shot
        # call, but a line that starts with one is still a device.
        line = re.sub(r"^\s*\[[A-Z]+\]\s*", "", line).strip()
        m = _DEVICE.match(line)
        if m:
            out.append((m.group(1).upper(), m.group(2).strip()))
    return out


class BluetoothCtl:
    """bluetoothctl, one subprocess a command, every one with a timeout.

    `run` and `popen` are subprocess.run and subprocess.Popen; tests pass
    their own. Never raises: a call that failed is (False, what it said)."""

    def __init__(self, exe="bluetoothctl", run=subprocess.run, popen=subprocess.Popen):
        self.exe = exe
        self._run = run
        self._popen = popen
        self._scan = None

    def available(self) -> bool:
        return shutil.which(self.exe) is not None

    def call(self, *args, timeout=CALL_S):
        try:
            done = self._run([self.exe, *args], stdin=subprocess.DEVNULL,
                             capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, (f"bluetoothctl {args[0]} did not answer in {timeout:.0f} s "
                           "(is bluetoothd up, and can this user reach it?)")
        except Exception as e:
            return False, f"bluetoothctl {args[0]}: {e}"
        text = _ANSI.sub("", (done.stdout or "") + (done.stderr or ""))
        return done.returncode == 0, text

    def devices(self):
        ok, text = self.call("devices")
        return parse_devices(text) if ok else []

    def remove(self, mac):
        ok, text = self.call("remove", mac)
        return ok or "not available" in text, _said(text)

    def pair(self, mac):
        ok, text = self.call("pair", mac, timeout=PAIR_S)
        done = "Pairing successful" in text or "AlreadyExists" in text
        return (ok and "Failed" not in text) or done, _said(text)

    def trust(self, mac):
        ok, text = self.call("trust", mac)
        return ok and "succeeded" in text, _said(text)

    def connect(self, mac):
        ok, text = self.call("connect", mac, timeout=CONNECT_S)
        return (ok and "Failed" not in text) or "Connection successful" in text, _said(text)

    def connected(self, mac) -> bool:
        ok, text = self.call("info", mac)
        return ok and bool(re.search(r"^\s*Connected:\s*yes\s*$", text, re.M))

    def scan_on(self, seconds):
        """Discovery for `seconds`, held by a bluetoothctl of its own."""
        self.scan_off()
        try:
            self._scan = self._popen(
                [self.exe, "--timeout", str(int(seconds) + 5), "scan", "on"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
        except Exception as e:
            self._scan = None
            return False, f"bluetoothctl scan: {e}"
        return True, ""

    def scan_off(self):
        """Its client gone, BlueZ stops the discovery it asked for."""
        scan, self._scan = self._scan, None
        if scan is None:
            return
        try:
            scan.terminate()
            scan.wait(timeout=3)
        except Exception:
            try:
                scan.kill()
                scan.wait(timeout=3)
            except Exception:
                pass


def _said(text):
    """The line worth quoting from bluetoothctl's output: the failure, or the last."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for ln in lines:
        if "Failed" in ln or "Error" in ln or "not available" in ln:
            return ln
    return lines[-1] if lines else ""


class PadRepair:
    """The re-pair, one run at a time, on a thread of its own.

    mode: a callable for the drive mode now ("auto" refuses). bt: a
    BluetoothCtl, or a stand-in with the same methods. Never raises into
    whoever calls start() or status()."""

    def __init__(self, mode=None, logger=None, bt=None, clock=time.monotonic,
                 sleep=time.sleep, scan_s=SCAN_S, poll_s=POLL_S, background=True):
        self.mode = mode
        self.logger = logger
        self.bt = bt if bt is not None else BluetoothCtl()
        self.clock = clock
        self.sleep = sleep
        self.scan_s = float(scan_s)
        self.poll_s = float(poll_s)
        self.background = background
        self._lock = threading.Lock()
        self._state = IDLE
        self._why = ""
        self._mac = None
        self._deadline = None
        self._thread = None
        self.runs = 0

    # ---- the page's side ------------------------------------------------------------

    def start(self):
        """(ok, why). Refused in auto, and a no-op while one is running."""
        try:
            mode = self.mode() if self.mode is not None else None
        except Exception:
            mode = None
        if mode == "auto":
            return False, "Stop auto first"
        with self._lock:
            if self._state in RUNNING:
                return False, "Already re-pairing"
            self._set(REMOVING, "", locked=True)
            self._mac, self._deadline = None, None
            self.runs += 1
        if not self.background:
            self._run()
            return True, ""
        try:
            self._thread = threading.Thread(target=self._run, name="pad-repair", daemon=True)
            self._thread.start()
        except Exception as e:
            self._set(FAILED, f"could not start: {e}")
            return False, self._why
        return True, ""

    def status(self):
        with self._lock:
            left = None
            if self._state == WAITING and self._deadline is not None:
                left = max(0, int(round(self._deadline - self.clock())))
            return {"state": self._state, "why": self._why, "mac": self._mac,
                    "left_s": left, "scan_s": int(self.scan_s),
                    "running": self._state in RUNNING}

    # ---- the run ----------------------------------------------------------------------

    def _run(self):
        try:
            self._steps()
        except Exception as e:
            self._set(FAILED, f"{type(e).__name__}: {e}")
        finally:
            try:
                self.bt.scan_off()
            except Exception:
                pass

    def _steps(self):
        bt = self.bt
        if hasattr(bt, "available") and not bt.available():
            return self._set(FAILED, "bluetoothctl is not installed")
        for mac, name in bt.devices():
            if is_pad(name):
                ok, said = bt.remove(mac)
                if not ok:
                    return self._set(FAILED, f"could not remove {mac}: {said}")
                self._say("info", f"Removed the old pairing with {mac}")
        ok, said = bt.scan_on(self.scan_s)
        if not ok:
            return self._set(FAILED, said)
        with self._lock:
            self._deadline = self.clock() + self.scan_s
        self._set(WAITING, "")
        mac = None
        while mac is None:
            mac = next((m for m, n in bt.devices() if is_pad(n)), None)
            if mac is not None:
                break
            if self.clock() >= self._deadline:
                return self._set(TIMED_OUT, f"no pad in pairing mode in {self.scan_s:.0f} s")
            self.sleep(self.poll_s)
        bt.scan_off()
        with self._lock:
            self._mac = mac
        self._set(PAIRING, mac)
        ok, said = bt.pair(mac)
        if not ok:
            return self._set(FAILED, f"pair: {said}")
        ok, said = bt.trust(mac)
        if not ok:
            return self._set(FAILED, f"trust: {said}")
        # The pad often connects itself straight after the pair, and then a
        # connect of ours says InProgress or AlreadyConnected. What counts is
        # Connected: yes.
        for attempt in range(CONNECT_TRIES):
            ok, said = bt.connect(mac)
            if ok or bt.connected(mac):
                return self._set(CONNECTED, mac)
            if attempt + 1 < CONNECT_TRIES:
                self.sleep(2.0)
        return self._set(FAILED, f"connect: {said}")

    def _set(self, state, why="", locked=False):
        if locked:
            self._state, self._why = state, why
        else:
            with self._lock:
                self._state, self._why = state, why
        level = "warning" if state in (FAILED, TIMED_OUT) else "info"
        self._say(level, f"Re-pair: {state}" + (f" ({why})" if why else ""))

    def _say(self, level, message):
        if self.logger is not None:
            try:
                self.logger.log(level, "Pad", message)
                return
            except Exception:
                pass
        print(f"[pad] {message}")
