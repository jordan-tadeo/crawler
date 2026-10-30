"""Saved tracks: named waypoint lists, each with the walls it was drawn
among, kept until the owner deletes one.

A track is a dict, the same shape on disk and in the patrol:

  {"version": 2, "name": "living room loop" or None, "mode": "loop",
   "points": [[x, y], ...], "walls": [[x, y], ...], "saved_at": epoch s}

The points and the walls are in one frame, the map the track was last on;
which frame that was means nothing on the next boot (every boot starts a
fresh map), so mapping/trackfit.py turns and moves the walls onto the live
map and the points go with them. The walls are only for that.

The working track, the one the patrol has now, is log/patrol.json, as it
has been since the patrol was written (patrol.Store). The named ones are
log/tracks/<file>.json, one a file, written only when the owner saves one
and removed only when the owner deletes one; nothing automatic ever takes
one away or writes over one. migrate() reads the version 1 patrol.json.
"""

import json
import os
import re
import threading
import time

VERSION = 2
NAME_MAX = 40
# A name's file: letters, digits, '-' and '_'; anything else becomes '_'.
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


def clean_name(name):
    """A name as kept: stripped, inner whitespace one space, NAME_MAX long;
    None if nothing is left."""
    if not isinstance(name, str):
        return None
    name = " ".join(name.split())[:NAME_MAX]
    return name or None


def file_of(name):
    stem = _UNSAFE.sub("_", name).strip("_").lower() or "track"
    # Two names that clean to one stem ("Loop!" and "loop?") must not share
    # a file: the stem gets the name's own hash when it is not the name.
    if stem != name:
        stem = f"{stem}-{abs(hash_name(name)) % 100000:05d}"
    return stem + ".json"


def hash_name(name):
    # Stable across runs (str.__hash__ is salted per process).
    h = 0
    for ch in name.encode("utf-8"):
        h = (h * 131 + ch) % (1 << 31)
    return h


def _points(v):
    out = []
    for p in v or []:
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            continue
        if x == x and y == y and abs(x) < 1e6 and abs(y) < 1e6:
            out.append([x, y])
    return out


def migrate(data):
    """A stored track (version 1 patrol.json or version 2) as version 2, or
    None if it is not one. Version 1 kept the walls of map 0 and the points
    on `frame`; points on any other map were not in the walls' frame, so
    those walls are dropped: the track is kept, and placed by hand."""
    if not isinstance(data, dict):
        return None
    points = _points(data.get("points"))
    walls = _points(data.get("walls"))
    version = data.get("version", 1)
    if version == 1 and int(data.get("frame", 0) or 0) != 0:
        walls = []
    mode = data.get("mode") if data.get("mode") in ("once", "loop", "patrol") else "loop"
    return {"version": VERSION, "name": clean_name(data.get("name")), "mode": mode,
            "points": points, "walls": walls,
            "saved_at": float(data.get("saved_at") or 0.0)}


class Library:
    """The named tracks in one directory. Safe from any thread; writes are
    whole files renamed into place. The patrol writes through it on a
    thread of its own; synchronous=True has it write in line (tests)."""

    def __init__(self, path, synchronous=False):
        self.path = str(path)
        self.synchronous = bool(synchronous)
        self._lock = threading.Lock()
        self.errors = 0

    def _file(self, name):
        return os.path.join(self.path, file_of(name))

    def names(self):
        """[{"name", "points", "saved_at"}], newest first."""
        out = []
        try:
            files = os.listdir(self.path)
        except OSError:
            return out
        for fn in files:
            if not fn.endswith(".json"):
                continue
            data = self._read(os.path.join(self.path, fn))
            if data is None or not data.get("name"):
                continue
            out.append({"name": data["name"], "points": len(data["points"]),
                        "saved_at": data["saved_at"]})
        out.sort(key=lambda r: -r["saved_at"])
        return out

    def _read(self, path):
        try:
            with open(path) as f:
                return migrate(json.load(f))
        except Exception:
            return None

    def has(self, name):
        name = clean_name(name)
        return name is not None and os.path.exists(self._file(name))

    def load(self, name):
        name = clean_name(name)
        if name is None:
            return None
        data = self._read(self._file(name))
        return data if data is not None and data.get("name") == name else None

    def save(self, name, track):
        """Write `track` as `name`. True if written."""
        name = clean_name(name)
        if name is None:
            return False
        data = migrate(dict(track, name=name))
        data["saved_at"] = time.time()
        try:
            with self._lock:
                os.makedirs(self.path, exist_ok=True)
                path = self._file(name)
                tmp = path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(data, f)
                os.replace(tmp, path)
            return True
        except Exception:
            self.errors += 1
            return False

    def delete(self, name):
        name = clean_name(name)
        if name is None or self.load(name) is None:
            return False
        try:
            with self._lock:
                os.remove(self._file(name))
            return True
        except OSError:
            self.errors += 1
            return False
