"""What I say a drive is, written into the recording beside it.

Two tags, each on or off, set from the map page's Outdoor and Clean buttons
({"cmd": "tags", "outdoor": true, "clean": false} on /patrol/cmd or POST
/api/cmd), never from the pad:

  outdoor   driving outside. Every model up to now was trained indoors, and
            an outdoor tick must never be graded or trained on as if it
            were one more indoor one.
  clean     I am driving instructor-style: the model drives and I take over
            only when it goes wrong, with smooth corrections. Off is every
            session before this existed: takeovers made any old way.

A recording with no tags in it (every one before this) is indoor and not
clean. Nothing here touches the wheels, the mode or any safety rule: a tag
is a label on the data and that is all.

Kept across a restart in log/session_tags.json, as the patrol's settings
are (patrol.Store: written off the control thread). How they reach a
recording:

  /session/tags   crawler.SessionTags, the tags now, once a second
                  (telemetry.INTERVALS) and again on the pass a tag changes,
                  so each tick takes the newest at or before it and a tag
                  flipped halfway through a session splits it there. seq
                  counts the changes since boot.
  metadata        an MCAP metadata record named METADATA_NAME, the tags as
                  the file was opened (recording.Recorder.metadata), so the
                  first second of a session is labelled before the first
                  /session/tags lands.

The journal says the tags at boot, at the start of each record session and
on every change.
"""

TAGS = ("outdoor", "clean")
# The other side of each tag, for the tools' filters (--include-tags
# indoor): a tick without the tag has its opposite.
OPPOSITES = {"outdoor": "indoor", "clean": "messy"}
NAMES = TAGS + tuple(OPPOSITES[t] for t in TAGS)
BITS = {t: 1 << k for k, t in enumerate(TAGS)}
TOPIC = "/session/tags"
METADATA_NAME = "crawler.tags"
SETTINGS_FILE = "log/session_tags.json"


def _flag(value):
    """A tag's value off a message, a metadata string or a saved file:
    True, False, or None for anything else."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    return None


def bits_of(payload):
    """The bit mask (BITS) of a /session/tags message, a metadata record's
    dict or a saved file; 0, untagged, for anything that is not one."""
    if not isinstance(payload, dict):
        return 0
    return sum(BITS[t] for t in TAGS if _flag(payload.get(t)) is True)


def names_of(bits):
    """The tags on in a bit mask, in TAGS order."""
    return [t for t in TAGS if int(bits) & BITS[t]]


def said(bits):
    """How the journal and the tools say a mask: "outdoor, clean", or what
    no tags means."""
    on = names_of(bits)
    words = [t if t in on else OPPOSITES[t] for t in TAGS]
    return ", ".join(words)


class SessionTags:
    """The tags now. The control loop sets them (command); the publisher
    thread reads them for a file's metadata, off one tuple swapped whole."""

    def __init__(self, store=None, say=None):
        self.store = store            # a patrol.Store, or None
        self.say = say                # Logger.log(level, subject, text), or None
        saved = store.load() if store is not None else None
        self._on = tuple(t for t in TAGS
                         if isinstance(saved, dict) and _flag(saved.get(t)) is True)
        self.seq = 0
        self.changed = False          # a change not yet sent on /session/tags
        self._log("info", f"Tags: {said(self.bits)} "
                  f"({'as last set' if isinstance(saved, dict) else 'none set yet'})")

    @property
    def bits(self):
        return sum(BITS[t] for t in self._on)

    def state(self):
        """{tag: bool}, for the map page."""
        on = self._on
        return {t: t in on for t in TAGS}

    def payload(self):
        """/session/tags."""
        on = self._on
        out = {t: t in on for t in TAGS}
        out["tags"] = ",".join(on)
        out["seq"] = int(self.seq)
        return out

    def metadata(self):
        """The MCAP metadata record's dict: strings only, as MCAP wants."""
        on = self._on
        out = {t: "true" if t in on else "false" for t in TAGS}
        out["tags"] = ",".join(on)
        return out

    def command(self, msg):
        """{"cmd": "tags", "outdoor": bool, "clean": bool}, either or both.
        True if taken (a repeat of what is set is taken and changes
        nothing), False if refused, said why."""
        if not isinstance(msg, dict):
            return False
        given = {t: msg[t] for t in TAGS if t in msg}
        if not given:
            self._log("warning", f"Tags refused: none of {', '.join(TAGS)} given")
            return False
        bad = [t for t, v in given.items() if not isinstance(v, bool)]
        if bad:
            self._log("warning", f"Tags refused: {bad[0]} is {given[bad[0]]!r}, "
                      "not true or false")
            return False
        on = set(self._on)
        for t, v in given.items():
            (on.add if v else on.discard)(t)
        new = tuple(t for t in TAGS if t in on)
        if new == self._on:
            return True
        self._on = new
        self.seq += 1
        self.changed = True
        if self.store is not None:
            self.store.save(self.state())
        self._log("info", f"Tags now {said(self.bits)}")
        return True

    def pop_changed(self):
        """True once after each change: send /session/tags this pass."""
        was, self.changed = self.changed, False
        return was

    def _log(self, level, text):
        if self.say is not None:
            try:
                self.say(level, "Tags", text)
            except Exception:
                pass
