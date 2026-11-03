"""hardware/pad_pair.py: re-pairing the Xbox pad from the phone's /pad.

bluetoothctl is never run here. PadRepair is driven against a stand-in with
BluetoothCtl's methods, and BluetoothCtl itself against a stand-in
subprocess.run and Popen, with output copied off a real bluetoothctl.
"""

import json
import re
import subprocess
import threading
import urllib.error
import urllib.request

import pytest

from crawler.hardware import pad_pair as pp
from crawler.telemetry import buttons as bt

from test_main import FakeLogger

PAD = "9C:AA:1B:B6:6B:34"
OTHER = "11:22:33:44:55:66"


class FakeBt:
    """BluetoothCtl's methods. `scans` is what each devices() call returns
    once the scan is on, the last one repeating."""

    def __init__(self, known=(), scans=((),), pair_ok=True, connect_ok=True,
                 connected_after=True, gate=None):
        self.known = list(known)
        self.scans = [list(s) for s in scans]
        self.pair_ok, self.connect_ok = pair_ok, connect_ok
        self.connected_after = connected_after
        self.gate = gate
        self.calls = []
        self.scanning = False

    def available(self):
        return True

    def devices(self):
        self.calls.append(("devices",))
        if self.gate is not None:
            self.gate.wait(5)
        if not self.scanning:
            return list(self.known)
        return self.scans.pop(0) if len(self.scans) > 1 else list(self.scans[0])

    def remove(self, mac):
        self.calls.append(("remove", mac))
        self.known = [d for d in self.known if d[0] != mac]
        return True, "Device has been removed"

    def scan_on(self, seconds):
        self.calls.append(("scan_on", seconds))
        self.scanning = True
        return True, ""

    def scan_off(self):
        if self.scanning:
            self.calls.append(("scan_off",))
        self.scanning = False

    def pair(self, mac):
        self.calls.append(("pair", mac, self.scanning))
        return (True, "Pairing successful") if self.pair_ok else (
            False, "Failed to pair: org.bluez.Error.AuthenticationFailed")

    def trust(self, mac):
        self.calls.append(("trust", mac))
        return True, f"Changing {mac} trust succeeded"

    def connect(self, mac):
        self.calls.append(("connect", mac))
        return (True, "Connection successful") if self.connect_ok else (
            False, "Failed to connect: org.bluez.Error.InProgress")

    def connected(self, mac):
        return self.connected_after


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def repair(fake, mode="manual", scan_s=90.0):
    clock = Clock()
    log = FakeLogger()
    states = []
    r = pp.PadRepair(mode=lambda: mode, logger=log, bt=fake, clock=clock,
                     sleep=clock.sleep, scan_s=scan_s, background=False)
    orig = r._set

    def spy(state, why="", locked=False):
        states.append(state)
        orig(state, why, locked)

    r._set = spy
    return r, log, states, clock


# --- the run --------------------------------------------------------------------


def test_a_run_forgets_the_old_bond_waits_for_the_pad_then_pairs_trusts_and_connects():
    fake = FakeBt(known=[(PAD, "Xbox Wireless Controller"), (OTHER, "JBL Flip")],
                  scans=[[(OTHER, "JBL Flip")], [("AA:BB:CC:DD:EE:01", "AA-BB-CC-DD-EE-01")],
                         [("AA:BB:CC:DD:EE:01", "Xbox Wireless Controller")]])
    r, log, states, _ = repair(fake)
    assert r.status()["state"] == pp.IDLE
    assert r.start() == (True, "")
    assert states == [pp.REMOVING, pp.WAITING, pp.PAIRING, pp.CONNECTED]
    st = r.status()
    assert st["state"] == pp.CONNECTED and st["mac"] == "AA:BB:CC:DD:EE:01"
    assert st["running"] is False
    names = [c[0] for c in fake.calls]
    # The other device is left alone; the old pad is removed before the scan.
    assert ("remove", PAD) in fake.calls and ("remove", OTHER) not in fake.calls
    assert names.index("remove") < names.index("scan_on")
    assert names.index("scan_off") < names.index("pair") < names.index("trust") < names.index("connect")
    assert ("pair", "AA:BB:CC:DD:EE:01", False) in fake.calls, "paired with the scan still running"
    # And the journal has every step.
    said = [m for _, subject, m in log.lines if subject == "Pad"]
    assert any("Removed the old pairing with " + PAD in m for m in said)
    for s in ("removing", "waiting", "pairing", "connected"):
        assert any(m.startswith("Re-pair: " + s) for m in said), s


def test_no_bonded_pad_goes_straight_to_the_scan():
    fake = FakeBt(scans=[[(PAD, "Xbox Wireless Controller")]])
    r, _, states, _ = repair(fake)
    r.start()
    assert not [c for c in fake.calls if c[0] == "remove"]
    assert states[-1] == pp.CONNECTED


def test_no_pad_in_pairing_mode_times_out_and_stops_the_scan():
    fake = FakeBt(scans=[[(OTHER, "JBL Flip")]])
    r, log, states, clock = repair(fake, scan_s=90.0)
    t0 = clock.t
    r.start()
    st = r.status()
    assert st["state"] == pp.TIMED_OUT and "90 s" in st["why"]
    assert 90.0 <= clock.t - t0 <= 92.0
    assert fake.scanning is False
    assert not [c for c in fake.calls if c[0] in ("pair", "trust", "connect")]
    assert any(level == "warning" and "timed_out" in m for level, _, m in log.lines)


def test_the_seconds_left_count_down_while_waiting():
    fake = FakeBt(scans=[[]])
    r, _, _, clock = repair(fake, scan_s=30.0)
    seen = []
    real_sleep = clock.sleep

    def sleep(s):
        seen.append(r.status()["left_s"])
        real_sleep(s)

    r.sleep = sleep
    r.start()
    assert seen[0] == 30 and seen[-1] == 1 and seen == sorted(seen, reverse=True)


def test_a_failed_pair_is_failed_with_bluetoothctls_reason():
    fake = FakeBt(scans=[[(PAD, "Xbox Wireless Controller")]], pair_ok=False)
    r, _, states, _ = repair(fake)
    r.start()
    st = r.status()
    assert st["state"] == pp.FAILED and "AuthenticationFailed" in st["why"]
    assert not [c for c in fake.calls if c[0] in ("trust", "connect")]


def test_a_pad_that_connected_itself_counts_as_connected():
    fake = FakeBt(scans=[[(PAD, "Xbox Wireless Controller")]], connect_ok=False,
                  connected_after=True)
    r, _, _, _ = repair(fake)
    r.start()
    assert r.status()["state"] == pp.CONNECTED


def test_a_connect_that_never_lands_is_failed_after_its_tries():
    fake = FakeBt(scans=[[(PAD, "Xbox Wireless Controller")]], connect_ok=False,
                  connected_after=False)
    r, _, _, _ = repair(fake)
    r.start()
    st = r.status()
    assert st["state"] == pp.FAILED and st["why"].startswith("connect:")
    assert len([c for c in fake.calls if c[0] == "connect"]) == pp.CONNECT_TRIES


def test_a_crash_in_the_run_is_failed_and_the_scan_is_stopped():
    fake = FakeBt(scans=[[(PAD, "Xbox Wireless Controller")]])

    def boom(mac):
        raise OSError("bus gone")

    fake.pair = boom
    r, _, _, _ = repair(fake)
    r.start()
    assert r.status()["state"] == pp.FAILED and "bus gone" in r.status()["why"]
    assert fake.scanning is False


def test_no_bluetoothctl_is_failed_and_says_so():
    fake = FakeBt()
    fake.available = lambda: False
    r, _, _, _ = repair(fake)
    r.start()
    assert r.status() ["state"] == pp.FAILED
    assert "not installed" in r.status()["why"]


# --- when it runs ------------------------------------------------------------------


def test_refused_in_auto_and_nothing_is_touched():
    fake = FakeBt(known=[(PAD, "Xbox Wireless Controller")])
    r, _, states, _ = repair(fake, mode="auto")
    assert r.start() == (False, "Stop auto first")
    assert fake.calls == [] and states == [] and r.status()["state"] == pp.IDLE
    assert r.runs == 0


def test_one_run_at_a_time_and_a_second_press_does_nothing():
    gate = threading.Event()
    fake = FakeBt(scans=[[(PAD, "Xbox Wireless Controller")]], gate=gate)
    r = pp.PadRepair(mode=lambda: "manual", logger=FakeLogger(), bt=fake,
                     sleep=lambda s: None)
    try:
        assert r.start() == (True, "")
        assert r.status()["running"] is True
        assert r.start() == (False, "Already re-pairing")
        assert r.start() == (False, "Already re-pairing")
        assert r.runs == 1
    finally:
        gate.set()
    r._thread.join(5)
    assert r.status()["state"] == pp.CONNECTED
    assert len([c for c in fake.calls if c[0] == "pair"]) == 1
    # Done, so the next press is a new run.
    fake.scanning = False
    assert r.start() == (True, "")
    r._thread.join(5)
    assert r.runs == 2


def test_it_never_reaches_the_wheels():
    src = open(pp.__file__, encoding="utf-8").read()
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    code = re.sub(r'"""[\s\S]*?"""', "", code)
    assert not re.search(r"vecon|throttle|steer|pigpio|vehicle", code, re.I)


# --- names ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,pad", [
    ("Xbox Wireless Controller", True),
    ("xbox wireless controller", True),
    ("  Xbox Wireless Controller ", True),
    ("Xbox Elite Wireless Controller", False),
    ("9C-AA-1B-B6-6B-34", False),
    ("Xbox", False),
    ("", False),
    (None, False),
])
def test_the_pad_is_found_by_its_name(name, pad):
    assert pp.is_pad(name) is pad


def test_devices_are_read_off_bluetoothctls_own_output():
    text = ("\x1b[0;94m[bluetooth]\x1b[0m# \r"
            "Device 9C:AA:1B:B6:6B:34 Xbox Wireless Controller\n"
            "Device 4c:aa:16:00:11:22 4C-AA-16-00-11-22\n"
            "[NEW] Device 11:22:33:44:55:66 JBL Flip 5\n"
            "Agent registered\n")
    assert pp.parse_devices(text) == [
        (PAD, "Xbox Wireless Controller"),
        ("4C:AA:16:00:11:22", "4C-AA-16-00-11-22"),
        (OTHER, "JBL Flip 5")]


# --- BluetoothCtl against a stand-in subprocess ----------------------------------------


class Done:
    def __init__(self, out, code=0):
        self.stdout, self.stderr, self.returncode = out, "", code


def test_bluetoothctl_calls_have_timeouts_and_their_answers_are_read():
    seen = []
    answers = {
        "pair": Done("Attempting to pair with 9C:AA:1B:B6:6B:34\n"
                     "[CHG] Device 9C:AA:1B:B6:6B:34 Paired: yes\nPairing successful\n"),
        "trust": Done("[CHG] Device 9C:AA:1B:B6:6B:34 Trusted: yes\n"
                      "Changing 9C:AA:1B:B6:6B:34 trust succeeded\n"),
        "connect": Done("Attempting to connect to 9C:AA:1B:B6:6B:34\n"
                        "Failed to connect: org.bluez.Error.InProgress\n", 1),
        "info": Done("Device 9C:AA:1B:B6:6B:34 (public)\n\tName: Xbox Wireless Controller\n"
                     "\tPaired: yes\n\tConnected: yes\n"),
        "remove": Done("Device 9C:AA:1B:B6:6B:34 not available\n", 1),
    }

    def run(argv, **kw):
        seen.append((argv, kw.get("timeout")))
        return answers[argv[1]]

    ctl = pp.BluetoothCtl(run=run)
    assert ctl.pair(PAD) == (True, "Pairing successful")
    assert ctl.trust(PAD)[0] is True
    ok, why = ctl.connect(PAD)
    assert ok is False and "InProgress" in why
    assert ctl.connected(PAD) is True
    assert ctl.remove(PAD)[0] is True, "a device already gone is removed"
    assert all(t is not None and t > 0 for _, t in seen)
    assert seen[0] == (["bluetoothctl", "pair", PAD], pp.PAIR_S)


def test_a_bluetoothctl_that_hangs_is_a_failure_with_a_reason():
    # What bluetoothctl does as a user that cannot reach bluetoothd: prints
    # "Waiting to connect to bluetoothd..." and never exits.
    def run(argv, **kw):
        raise subprocess.TimeoutExpired(argv, kw["timeout"])

    ctl = pp.BluetoothCtl(run=run)
    ok, why = ctl.pair(PAD)
    assert ok is False and "did not answer" in why
    assert ctl.devices() == []


def test_the_scan_is_held_open_by_its_own_bluetoothctl_and_stopped_by_ending_it():
    procs = []

    class Proc:
        def __init__(self, argv, **kw):
            self.argv, self.ended = argv, False
            procs.append(self)

        def terminate(self):
            self.ended = True

        def wait(self, timeout=None):
            return 0

    ctl = pp.BluetoothCtl(popen=Proc)
    assert ctl.scan_on(90) == (True, "")
    assert procs[0].argv == ["bluetoothctl", "--timeout", "95", "scan", "on"]
    ctl.scan_off()
    assert procs[0].ended is True
    ctl.scan_off()   # twice is fine


# --- the page's side ---------------------------------------------------------------------


def _req(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


@pytest.fixture
def served():
    mode = {"now": "manual"}
    fake = FakeBt(scans=[[(PAD, "Xbox Wireless Controller")]])
    r = pp.PadRepair(mode=lambda: mode["now"], logger=FakeLogger(), bt=fake,
                     sleep=lambda s: None, background=False)
    # No goals, so no map page: the pad page is served anyway.
    page = bt.ButtonPage(host="127.0.0.1", port=0, repair=r)
    assert page.start()
    try:
        yield page, r, mode, f"http://127.0.0.1:{page.port}"
    finally:
        page.stop()


def test_the_pad_page_and_its_status_are_served_without_a_map(served):
    page, r, _, base = served
    assert page.map_body is None
    code, body = _req(base + "/pad")
    assert code == 200 and b"Re-pair controller" in body
    code, body = _req(base + "/api/pad")
    assert code == 200 and json.loads(body)["state"] == "idle"
    assert _req(base + "/map")[0] == 404


def test_repair_pad_on_the_command_path_starts_it_and_auto_refuses(served):
    page, r, mode, base = served
    mode["now"] = "auto"
    code, body = _req(base + "/api/cmd", {"cmd": "repair_pad"})
    assert code == 400 and json.loads(body) == {"ok": False, "why": "Stop auto first"}
    assert r.runs == 0
    mode["now"] = "manual"
    code, body = _req(base + "/api/cmd", {"cmd": "repair_pad"})
    assert code == 202 and json.loads(body)["ok"] is True
    assert json.loads(_req(base + "/api/pad")[1])["state"] == "connected"
    # Every other command still needs the map's patrol.
    assert _req(base + "/api/cmd", {"cmd": "start"})[0] == 400


def test_the_map_pages_state_carries_the_repair():
    r = pp.PadRepair(mode=lambda: "manual", bt=FakeBt())

    class Goals:
        def push(self, kind, msg):
            return True

    page = bt.ButtonPage(goals=Goals(), repair=r)
    assert json.loads(page.state_json())["pad_repair"]["state"] == "idle"
    assert "pad_repair" not in json.loads(bt.ButtonPage(goals=Goals()).state_json())
    assert bt.ButtonPage().command(b'{"cmd":"repair_pad"}') == (False, "no re-pair on this car")


# --- the pad that comes back --------------------------------------------------------------


def test_a_fresh_pairing_is_picked_up_by_the_running_service(monkeypatch):
    """The re-pair drops the old js device and a new one appears. The
    Joystick sees the count go to zero, lets go, and try_connect opens the
    new device at index 0 and measures its triggers again: no restart."""
    import pygame
    from crawler.hardware import joystick as js

    count = [1]

    class Handle:
        def __init__(self, rest):
            self.rest = rest

        def init(self):
            pass

        def get_axis(self, i):
            return self.rest if i in (2, 5) else 0.0

    handles = [Handle(0.0), Handle(-1.0)]
    opened = []
    monkeypatch.setattr(pygame.joystick, "get_init", lambda: True)
    monkeypatch.setattr(pygame.joystick, "get_count", lambda: count[0])
    monkeypatch.setattr(pygame.joystick, "Joystick",
                        lambda i: opened.append(i) or handles[len(opened) - 1])
    monkeypatch.setattr(pygame.event, "pump", lambda: None)
    pad = js.Joystick(wait=False)
    assert pad.is_connected() and pad._trigger_rest["RT"] == 0.0
    count[0] = 0                        # bluetoothctl remove
    pad.update_connection_status()
    assert pad.is_connected() is False
    count[0] = 1                        # paired, trusted, connected
    assert pad.try_connect(10.0) is True
    assert opened == [0, 0] and pad.joystick is handles[1]
    assert pad._trigger_rest["RT"] == -1.0, "the new pad's triggers measured again"
