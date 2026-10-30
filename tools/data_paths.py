#!/usr/bin/env python3
"""Where the laptop keeps what never goes in a repo: the recordings, the
models and registry.json, the decode cache, the retrain runs, and the
training venv.

It all sits in one folder, beside the checkout:

    <folder>/repo              the checkout
    <folder>/worktrees/<name>  any worktrees of it
    <folder>/data              recordings (mcap/), models/, cache/, retrain/, exclude.txt
    <folder>/train             the training venv

CRAWLER_DATA names the data folder outright, and CRAWLER_TRAIN the venv.
Without them each is `data` (or `train`) beside the main checkout. The main
checkout comes off the .git file, so a worktree anywhere resolves to the same
folder as the checkout it came from. Failing that, the old ~/crawler-data and
~/crawler-train if they are still there, so a checkout in the old layout
keeps working.

    python3 tools/data_paths.py data     # prints the data folder, for the shell scripts
    python3 tools/data_paths.py train    # prints the training venv
"""

import os
import sys

CHECKOUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY = {"data": "~/crawler-data", "train": "~/crawler-train"}
ENV = {"data": "CRAWLER_DATA", "train": "CRAWLER_TRAIN"}


def main_checkout(checkout=CHECKOUT):
    """The main checkout of `checkout`: itself, or for a worktree the one
    its .git file points back to (<main>/.git/worktrees/<name>, whose
    commondir file says ../.. to <main>/.git)."""
    dot_git = os.path.join(checkout, ".git")
    if not os.path.isfile(dot_git):
        return checkout
    with open(dot_git) as f:
        line = f.read().strip()
    if not line.startswith("gitdir:"):
        return checkout
    gitdir = os.path.join(checkout, line[len("gitdir:"):].strip())
    common = os.path.join(gitdir, "commondir")
    if not os.path.isfile(common):
        return checkout
    with open(common) as f:
        common_dir = os.path.normpath(os.path.join(gitdir, f.read().strip()))
    return os.path.dirname(common_dir)


def resolve(kind, checkout=CHECKOUT):
    """The `data` or `train` folder: the env var, else beside the main
    checkout, else the old home-folder one if it exists, else beside the
    main checkout again (where a first run makes it)."""
    set_to = os.environ.get(ENV[kind])
    if set_to:
        return os.path.abspath(os.path.expanduser(set_to))
    beside = os.path.join(os.path.dirname(os.path.abspath(main_checkout(checkout))), kind)
    if os.path.isdir(beside):
        return beside
    old = os.path.expanduser(LEGACY[kind])
    if os.path.isdir(old):
        return old
    return beside


def data_root(checkout=CHECKOUT):
    return resolve("data", checkout)


def train_dir(checkout=CHECKOUT):
    return resolve("train", checkout)


DATA_ROOT = data_root()


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ENV:
        sys.exit("usage: data_paths.py data|train")
    print(resolve(sys.argv[1]))
