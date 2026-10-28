"""Whether the car has got where it was going. The one place auto asks.

A learned policy's "stop" is only committed when this says the goal is
reached; until then PolicyDriver drives on with the better of forward and
reverse (policy.py, "The goal"). Nothing sets it yet, so it is always
False and the model never gets to give up in place.

The seam: AutoPilot owns one Goal (pilot.goal) and hands it to the
PolicyDriver it builds, which calls reached() once every decision. Whatever
later decides the car has arrived -- a person follower that is close enough
to the person -- calls set_reached(True) on that same object, or replaces it
with anything else that has a reached() method. Nothing else changes.

This only ever stops the car. Every safety stop in the driver stops whatever
this says.
"""


class Goal:
    """A goal flag, False until something sets it."""

    def __init__(self, reached=False):
        self._reached = bool(reached)

    def reached(self):
        return self._reached

    def set_reached(self, reached):
        self._reached = bool(reached)
