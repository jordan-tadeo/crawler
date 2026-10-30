"""Find the hood camera servos' centre, limits and direction, and save them.

On the car, from the checkout, with the service up or down:

    ./.venv/bin/python tools/pantilt_cal.py
    ./.venv/bin/python tools/pantilt_cal.py --pan-channel 10 --tilt-channel 11

It writes only the LEDn registers of the two channels in the config (or the
ones given), at the frequency the PCA9685 is already running: it never
resets the chip, never touches MODE1 or the prescale, and refuses the
steering channels 14 and 15. So it is safe beside the running service. It
does need the service to have set the chip up once since power-on
(auto-increment on): it says so and stops if not.

Both servos go to their centre pulse first. Then type commands, one a line:

    p+  p-          nudge pan by the step (left should be p+; see pinv)
    t+  t-          nudge tilt (up should be t+; see tinv)
    p 1600          pan to a pulse in microseconds (500-2500)
    t 1450          the same for tilt
    step 10         the nudge size in microseconds (default 20)
    pc  tc          this pulse is the centre
    pmin pmax       this pulse is a soft limit; stop short of where the
    tmin tmax       horn binds or the camera hits the hood
    pinv  tinv      flip the direction, when p+ turned right or t+ looked down
    pspan 150       degrees between pmin and pmax, by eye or a protractor,
    tspan 90        which sets deg_per_us
    show            what is set so far
    save            write it to models/pantilt.json (untracked; --out to change)
    release         stop driving both servos (they go limp)
    q               centre both and quit

Nudge in small steps toward each end and listen: a servo pushing against a
stop buzzes. Back off a step or two from there and mark it.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler.hardware import pantilt as pt  # noqa: E402


class Cal:
    """The calibration's state and its commands, apart from the terminal."""

    def __init__(self, cfg, writer):
        self.cfg = cfg
        self.writer = writer
        self.step = 20.0
        self.us = {"pan": cfg.mount.pan.centre_us,
                   "tilt": cfg.mount.tilt.centre_us}

    def axis(self, name):
        return getattr(self.cfg.mount, name)

    def move(self, name, us):
        if not pt.PULSE_FLOOR_US <= us <= pt.PULSE_CEIL_US:
            return f"{us:.0f} us is outside {pt.PULSE_FLOOR_US}-{pt.PULSE_CEIL_US}"
        self.writer.write_us(self.axis(name).channel, us)
        self.us[name] = us
        return f"{name} {us:.0f} us"

    def centre_both(self):
        for name in ("pan", "tilt"):
            self.move(name, self.axis(name).centre_us)

    def apply(self, line):
        """Run one command; returns what to print, or None to quit."""
        words = line.strip().split()
        if not words:
            return ""
        cmd, rest = words[0].lower(), words[1:]
        names = {"p": "pan", "t": "tilt"}
        try:
            if cmd in ("q", "quit", "exit"):
                self.centre_both()
                return None
            if cmd in ("p+", "p-", "t+", "t-"):
                name = names[cmd[0]]
                axis = self.axis(name)
                sign = 1.0 if cmd[1] == "+" else -1.0
                if axis.invert:
                    sign = -sign
                return self.move(name, self.us[name] + sign * self.step)
            if cmd in ("p", "t") and rest:
                return self.move(names[cmd], float(rest[0]))
            if cmd == "step" and rest:
                self.step = max(1.0, min(200.0, float(rest[0])))
                return f"step {self.step:.0f} us"
            if cmd in ("pc", "tc", "pmin", "pmax", "tmin", "tmax"):
                name = names[cmd[0]]
                key = {"c": "centre_us", "min": "min_us", "max": "max_us"}[cmd[1:]]
                setattr(self.axis(name), key, self.us[name])
                return f"{name}.{key} = {self.us[name]:.0f}"
            if cmd in ("pinv", "tinv"):
                axis = self.axis(names[cmd[0]])
                axis.invert = not axis.invert
                return f"{names[cmd[0]]}.invert = {axis.invert}"
            if cmd in ("pspan", "tspan") and rest:
                axis = self.axis(names[cmd[0]])
                width = axis.max_us - axis.min_us
                if width <= 0:
                    return "mark min and max first"
                axis.deg_per_us = float(rest[0]) / width
                return f"{names[cmd[0]]}.deg_per_us = {axis.deg_per_us:.4f}"
            if cmd == "show":
                return self.show()
            if cmd == "release":
                for name in ("pan", "tilt"):
                    self.writer.release(self.axis(name).channel)
                return "both released"
        except ValueError as e:
            return f"? {e}"
        return f"? {line.strip()}"

    def show(self):
        out = []
        for name in ("pan", "tilt"):
            a = self.axis(name)
            out.append(f"{name}: ch {a.channel}, now {self.us[name]:.0f} us, "
                       f"centre {a.centre_us:.0f}, {a.min_us:.0f}..{a.max_us:.0f}, "
                       f"invert {a.invert}, {a.deg_per_us:.4f} deg/us")
        return "\n".join(out)


# Where `save` writes. models/ is untracked, so a calibration on the car never
# dirties the checkout; run the service with --pantilt models/pantilt.json.
CAL_OUT = os.path.join("models", "pantilt.json")


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", help="start from this (default: the last "
                    f"save, {CAL_OUT}, or else {pt.DEFAULT_CONFIG})")
    ap.add_argument("--out", default=CAL_OUT)
    ap.add_argument("--pan-channel", type=int)
    ap.add_argument("--tilt-channel", type=int)
    return ap.parse_args(argv)


def paths(args, exists=os.path.exists):
    """(the config to start from, the file `save` writes)."""
    if args.config:
        return args.config, args.out
    return (CAL_OUT if exists(CAL_OUT) else pt.DEFAULT_CONFIG), args.out


def main(argv=None):
    args = parse(argv)
    src, out = paths(args)
    cfg = pt.load_config(src)
    cfg.mount.kind = "servo"
    if args.pan_channel is not None:
        cfg.mount.pan.channel = args.pan_channel
    if args.tilt_channel is not None:
        cfg.mount.tilt.channel = args.tilt_channel
    pt.validate(cfg)

    import board
    import busio
    writer = pt.RawPca9685(busio.I2C(board.SCL, board.SDA))
    print(f"PCA9685 at {writer.freq_hz:.2f} Hz; pan ch {cfg.mount.pan.channel}, "
          f"tilt ch {cfg.mount.tilt.channel}. Centring both.")
    cal = Cal(cfg, writer)
    cal.centre_both()
    print(cal.show())
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            line = "q"
        if line.strip().lower() == "save":
            try:
                os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
                pt.save_config(cal.cfg, out)
                print(f"saved {out}")
            except pt.ConfigError as e:
                print(f"not saved: {e}")
            continue
        msg = cal.apply(line)
        if msg is None:
            print("centred, bye")
            return 0
        if msg:
            print(msg)


if __name__ == "__main__":
    sys.exit(main())
