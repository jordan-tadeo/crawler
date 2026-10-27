"""Nothing in the repo may be pinned to a DHCP lease.

This is a test about a confusing failure and not about a string. The address
the car used to hold moved, and the device that has it now answers a ping and
refuses SSH -- so every tool that had the address baked in failed in the one
way that looks like the car being broken. The fix is the name, which mDNS
resolves on the same wifi with nothing configured, and `CRAWLER_HOST` for the
network where it does not.

It is a grep, because the thing that went wrong was a literal in a file and
the only way to stop it coming back is to notice the literal.
"""

import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CAR_NAME = "jtpi.local"

# Where an address would be written if it were written anywhere.
SEARCHED = (
    "pull-recordings.sh",
    "run.sh",
    "telemetry.py",
    "main.py",
    "README.md",
    "deploy/mcap_check.py",
    "deploy/install-service.sh",
    "deploy/crawler.service.template",
)

ADDRESS = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")

# Addresses that mean something other than "the car". 0.0.0.0 is the bind
# address telemetry listens on and 127.0.0.1 is this machine; neither one is a
# lease and neither one can move.
NOT_A_LEASE = {"0.0.0.0", "127.0.0.1", "255.255.255.255"}


def read(name):
    with open(os.path.join(REPO, name)) as handle:
        return handle.read()


def test_pull_recordings_reaches_the_car_by_name():
    source = read("pull-recordings.sh")
    assert f'CAR="${{CRAWLER_HOST:-jt@{CAR_NAME}}}"' in source


def test_the_host_override_is_still_there():
    """A network with no mDNS still needs a way in."""
    assert "CRAWLER_HOST" in read("pull-recordings.sh")


def test_the_foxglove_url_is_the_name_too():
    assert f"ws://{CAR_NAME}:8765" in read("README.md")
    assert f"ws://{CAR_NAME}:8765" in read("telemetry.py")


def test_no_file_is_pinned_to_an_address():
    found = {}
    for name in SEARCHED:
        leases = [match for match in ADDRESS.findall(read(name))
                  if match not in NOT_A_LEASE]
        if leases:
            found[name] = leases
    assert not found, f"hardcoded addresses, which move: {found}"
