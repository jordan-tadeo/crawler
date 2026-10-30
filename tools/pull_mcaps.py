#!/usr/bin/env python3
"""Move the car's recordings onto the laptop, outside the repo, and prove it.

    python3 tools/pull_mcaps.py                    # everything new, into <data>/mcap (tools/data_paths.py)
    python3 tools/pull_mcaps.py --list             # what is on the car, and what is live
    python3 tools/pull_mcaps.py -n 3               # the newest three only
    python3 tools/pull_mcaps.py --records-only     # only the RECORD-mode sessions
    python3 tools/pull_mcaps.py --prune            # say what would go off the car
    python3 tools/pull_mcaps.py --prune --yes      # and do it

pull-recordings.sh runs this for the copy and then mcap_check.py over what it
fetched. Standard library only, so plain python3 on the Mac runs it.

Why the destination is outside the repo, and why that is enforced. This repo is
public. A recording is a long video of my house, and anything that lands in a
work tree is one `git add -A` away from being in history forever, where no
later `git rm` takes it back out. .gitignore covers log/, recordings/ and
*.mcap, but an ignore rule is a convenience and this is a guard: a destination
anywhere inside a git work tree is refused, with no flag to override it.

Which file is still being written. The car's recorder holds its current
segment open until it rotates or the run ends, and that file has no MCAP
footer yet. A size-stable check does not work here: in manual mode the writer
buffers whole chunks and the live file can sit at 75 bytes for minutes, so it
looks finished when it is not. So a file is live if the crawler service's
process has it open (read off /proc/<MainPID>/fd, which the jt user can read
for its own service), or, if that cannot be read, if it has no footer and its
p<pid> is the service's MainPID. A file with no footer from a process that is
gone is a run that lost power, which is exactly the one I want, so it is
pulled and mcap_check repairs the copy.

Verification is size plus sha256 on both ends, every file this run copied.
--prune is opt-in and re-hashes both ends at the moment it runs: a Pi file is
only deleted if a local copy (the file, or the .orig mcap_check keeps beside a
repaired one) has the same sha256 right now. It never touches a live file, it
prints what it would delete, and without --yes it deletes nothing.

--records-only narrows everything after the listing to the crawler-record-*
files, the training sessions, and leaves the rolling drive logs where they
are. -n counts within what it kept, --list shows what it kept, and --prune
only looks at what it kept, so a drive log is never even a candidate. Prune
was never deciding off the filter anyway: nothing goes without a local copy
that hashes the same.

On the car this only ever reads, except for the rm that --prune --yes sends.
"""

import argparse
import hashlib
import os
import shlex
import subprocess
import sys
import tempfile

import data_paths as dp

CAR = os.environ.get("CRAWLER_HOST", "jt@jtpi.local")
# The rolling ring the service writes in every mode, and where record mode
# (manual-recording branch) puts its one-file-per-session drives.
REMOTE_DIRS = (
    "/home/jt/Documents/py/crawler/log/mcap",
    "/home/jt/Documents/py/crawler/recordings",
)
DEFAULT_DEST = os.path.expanduser(os.environ.get("CRAWLER_MCAP", os.path.join(dp.DATA_ROOT, "mcap")))
SERVICE = "crawler"

# What a closed MCAP ends in. Only close() writes it.
MCAP_MAGIC_HEX = "894d434150300d0a"

HASH_CHUNK = 1 << 20


# ---------------------------------------------------------------------------
# Pure: the guard, parsing the car's listing, deciding what is live and what
# may be pruned. The tests drive these.
# ---------------------------------------------------------------------------

def existing_ancestor(path):
    """The path itself if it exists, else its nearest existing parent."""
    path = os.path.abspath(os.path.expanduser(path))
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path


def git_work_tree_of(path, exists=os.path.exists):
    """The work tree a path is inside, or None.

    Walks up from the real path (so a symlink into a repo is caught) looking
    for a .git entry -- a directory in a normal checkout, a file in a linked
    worktree or a submodule. The path does not have to exist yet.
    """
    here = os.path.realpath(existing_ancestor(path))
    rest = os.path.relpath(os.path.abspath(os.path.expanduser(path)),
                           existing_ancestor(path))
    if rest != ".":
        here = os.path.join(here, rest)
    while True:
        if exists(os.path.join(here, ".git")):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            return None
        here = parent


def git_says_work_tree(path):
    """Second opinion from git itself, for a GIT_DIR/odd layout the walk
    might miss. None if git is absent or says no."""
    try:
        out = subprocess.run(
            ["git", "-C", existing_ancestor(path), "rev-parse",
             "--show-toplevel"],
            capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def refuse_reason(dest):
    """Why dest is not allowed, or None if it is fine."""
    tree = git_work_tree_of(dest) or git_says_work_tree(dest)
    if tree:
        return (f"{dest} is inside the git work tree {tree}. Recordings never "
                f"go inside a repo; pick a folder outside one "
                f"(default {DEFAULT_DEST}).")
    return None


class RemoteFile:
    def __init__(self, path, size, mtime, closed):
        self.path = path
        self.size = size
        self.mtime = mtime
        self.closed = closed      # has the MCAP footer magic
        self.live = False
        self.live_why = ""

    @property
    def name(self):
        return os.path.basename(self.path)


def listing_script(dirs):
    """The one read-only shell command run on the car."""
    quoted = " ".join(shlex.quote(d) for d in dirs)
    return (
        f"state=$(systemctl is-active {SERVICE} 2>/dev/null); "
        f"pid=$(systemctl show -p MainPID --value {SERVICE} 2>/dev/null); "
        'echo "SERVICE ${state:-unknown} ${pid:-0}"; '
        'if [ -n "$pid" ] && [ "$pid" != 0 ]; then '
        '  if ls /proc/$pid/fd >/dev/null 2>&1; then echo FDS_READABLE; '
        '    for fd in /proc/$pid/fd/*; do l=$(readlink "$fd"); '
        '      case "$l" in *.mcap) echo "OPEN $l";; esac; done; fi; fi; '
        f"for d in {quoted}; do [ -d \"$d\" ] || continue; "
        '  echo "DF $(df -Pk "$d" | tail -1)"; '
        '  for f in "$d"/crawler-*.mcap; do [ -f "$f" ] || continue; '
        '    echo "FILE $(stat -c "%s %Y" "$f") '
        '$(tail -c 8 "$f" | od -An -tx1 | tr -d " \\n") $f"; done; done'
    )


def parse_listing(text):
    """The car's answer -> (files, service_state, main_pid, open_paths or
    None if the fd table could not be read, df lines)."""
    files, open_paths, df = [], set(), []
    state, pid, fds_readable = "unknown", 0, False
    for line in text.splitlines():
        head, _, rest = line.partition(" ")
        if head == "SERVICE":
            parts = rest.split()
            state = parts[0] if parts else "unknown"
            try:
                pid = int(parts[1]) if len(parts) > 1 else 0
            except ValueError:
                pid = 0
        elif line.strip() == "FDS_READABLE":
            fds_readable = True
        elif head == "OPEN":
            open_paths.add(rest.strip())
        elif head == "DF":
            df.append(rest.strip())
        elif head == "FILE":
            parts = rest.split(" ", 3)
            if len(parts) == 4:
                size, mtime, tail, path = parts
            elif len(parts) == 3:  # a file shorter than... an empty tail
                size, mtime, path = parts
                tail = ""
            else:
                continue
            try:
                files.append(RemoteFile(path, int(size), int(mtime),
                                        tail == MCAP_MAGIC_HEX))
            except ValueError:
                continue
    return files, state, pid, (open_paths if fds_readable else None), df


def pid_of(name):
    """The p<pid> in crawler-...-p1143-001.mcap, or None."""
    for part in name[:-len(".mcap")].split("-"):
        if part.startswith("p") and part[1:].isdigit():
            return int(part[1:])
    return None


def mark_live(files, state, main_pid, open_paths):
    """Set .live on every file that is still being written. See the module
    docstring for why it is these two tests and not a size check."""
    running = state == "active" and main_pid > 0
    for f in files:
        if open_paths is not None and f.path in open_paths:
            f.live, f.live_why = True, "open by the running service"
        elif (open_paths is None and running and not f.closed
              and pid_of(f.name) == main_pid):
            f.live, f.live_why = True, (
                "no footer, and written by the running service's pid")
    return files


RECORD_PREFIX = "crawler-record-"


def is_record_session(f):
    """A RECORD-mode training session, by its name."""
    return f.name.startswith(RECORD_PREFIX)


def select(files, records_only=False, count=0):
    """What this run works on: the filter first, then the newest count of
    what is left. files is already newest first."""
    if records_only:
        files = [f for f in files if is_record_session(f)]
    if count > 0:
        files = files[:count]
    return files


def newest_first(files):
    return sorted(files, key=lambda f: (f.mtime, f.name), reverse=True)


def already_have(f, dest, exists=os.path.exists, getsize=os.path.getsize):
    """True if the copy is here already: the file at the same size, or the
    .orig mcap_check left when it repaired it."""
    for local in (os.path.join(dest, f.name),
                  os.path.join(dest, f.name + ".orig")):
        if exists(local) and getsize(local) == f.size:
            return True
    return False


def prune_plan(files, remote_sha, local_sha):
    """(delete, keep) for --prune. Pure.

    remote_sha: {remote path: sha256} hashed just now on the car.
    local_sha:  {name: [sha256 of each local copy that exists]}.
    A file goes only if it is not live, its remote hash is known, and some
    local copy has that exact hash. Everything else is kept with a reason.
    """
    delete, keep = [], []
    for f in files:
        if f.live:
            keep.append((f, "still being written"))
            continue
        sha = remote_sha.get(f.path)
        if not sha:
            keep.append((f, "could not hash it on the car"))
            continue
        copies = local_sha.get(f.name) or []
        if not copies:
            keep.append((f, "no copy on this Mac"))
        elif sha in copies:
            delete.append(f)
        else:
            keep.append((f, "the copy here does not match"))
    return delete, keep


# ---------------------------------------------------------------------------
# The car and the disk.
# ---------------------------------------------------------------------------

def ssh(car, command, check=True):
    out = subprocess.run(["ssh", "-o", "ConnectTimeout=10", car, command],
                         capture_output=True, text=True)
    if check and out.returncode != 0:
        raise SystemExit(f"ssh {car} failed ({out.returncode}): "
                         f"{out.stderr.strip()}")
    return out.stdout


def sha256_local(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(HASH_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def sha256_remote(car, paths):
    """{path: sha256} for files on the car, one ssh for all of them.

    Under nice and ionice idle: the car may be driving while this runs, and a
    gigabyte of hashing is not allowed to take a core or the card off the
    control loop.
    """
    if not paths:
        return {}
    out = ssh(car, "nice -n 19 ionice -c 3 sha256sum -- "
              + " ".join(shlex.quote(p) for p in paths),
              check=False)
    sums = {}
    for line in out.splitlines():
        sha, _, path = line.partition("  ")
        if len(sha) == 64:
            sums[path] = sha
    return sums


def mib(n):
    return f"{n / 1048576.0:.1f} MiB"


def print_df(df, when):
    for line in df:
        parts = line.split()
        if len(parts) >= 6:
            free = int(parts[3]) * 1024
            print(f"  car free space {when}: {free / 1073741824.0:.1f} GiB "
                  f"({parts[4]} used) on {parts[5]}")


def survey(car, dirs):
    files, state, pid, open_paths, df = parse_listing(
        ssh(car, listing_script(dirs)))
    mark_live(files, state, pid, open_paths)
    return newest_first(files), state, pid, open_paths, df


def show(files, state, pid, open_paths, what="recordings"):
    print(f"On {CAR}: service {state}, MainPID {pid}"
          + ("" if open_paths is not None else
             " (its fd table is not readable; live = no footer + its pid)"))
    total = sum(f.size for f in files)
    print(f"  {len(files)} {what}, {mib(total)}")
    for f in files:
        # unclosed = no MCAP footer: the writer died before close()
        flag = "LIVE" if f.live else ("" if f.closed else "unclosed")
        print(f"  {flag:<8} {mib(f.size):>11}  {f.name}"
              + (f"  ({f.live_why})" if f.live else ""))


def pull(car, files, dest, pulled_list=None):
    """Copy what is not here yet. Returns (copied, failed)."""
    todo = [f for f in files if not f.live and not already_have(f, dest)]
    if not todo:
        print("Nothing new to copy.")
        return [], []
    os.makedirs(dest, exist_ok=True)
    print(f"Copying {len(todo)} into {dest}:")
    by_dir = {}
    for f in todo:
        by_dir.setdefault(os.path.dirname(f.path), []).append(f)
    for remote_dir, group in by_dir.items():
        with tempfile.NamedTemporaryFile("w", delete=False) as listing:
            listing.write("\n".join(f.name for f in group) + "\n")
        try:
            # --partial-dir so a half-copied file never sits in dest under its
            # real name. -t keeps the car's mtime, which is the newest-first
            # order everything else sorts on.
            subprocess.run(
                ["rsync", "-t", "--partial-dir=.rsync-partial",
                 f"--files-from={listing.name}", "--info=progress2"
                 if _rsync_has_info() else "--progress",
                 f"{car}:{remote_dir}/", dest + "/"], check=False)
        finally:
            os.unlink(listing.name)

    print("Verifying (size + sha256 on both ends):")
    remote = sha256_remote(car, [f.path for f in todo])
    copied, failed = [], []
    for f in todo:
        local = os.path.join(dest, f.name)
        if not os.path.exists(local):
            failed.append((f, "not copied"))
        elif os.path.getsize(local) != f.size:
            failed.append((f, f"size {os.path.getsize(local)} != {f.size}"))
        elif remote.get(f.path) != sha256_local(local):
            failed.append((f, "sha256 differs"))
        else:
            copied.append(f)
            print(f"  ok   {f.name}  {mib(f.size)}  {remote[f.path][:12]}")
    for f, why in failed:
        print(f"  FAIL {f.name}: {why}")
    if pulled_list:
        with open(pulled_list, "w") as handle:
            handle.write("".join(os.path.join(dest, f.name) + "\n"
                                 for f in copied))
    return copied, failed


def _rsync_has_info():
    # macOS ships openrsync / rsync 2.6.9, which have no --info.
    try:
        out = subprocess.run(["rsync", "--version"], capture_output=True,
                             text=True, timeout=5).stdout
    except Exception:
        return False
    return "version 3" in out


def local_hashes(files, dest):
    out = {}
    for f in files:
        for local in (os.path.join(dest, f.name),
                      os.path.join(dest, f.name + ".orig")):
            if os.path.exists(local):
                out.setdefault(f.name, []).append(sha256_local(local))
    return out


def prune(car, files, dest, yes):
    candidates = [f for f in files if not f.live]
    print("Prune: hashing both ends again...")
    remote = sha256_remote(car, [f.path for f in candidates])
    delete, keep = prune_plan(files, remote, local_hashes(candidates, dest))
    for f, why in keep:
        print(f"  keep   {f.name}  ({why})")
    if not delete:
        print("Nothing on the car is safe to delete.")
        return 0
    total = sum(f.size for f in delete)
    for f in delete:
        print(f"  delete {f.name}  {mib(f.size)}")
    print(f"  {len(delete)} files, {mib(total)}")
    if not yes:
        print("Nothing deleted. Run again with --prune --yes to delete those.")
        return 0
    ssh(car, "rm -- " + " ".join(shlex.quote(f.path) for f in delete))
    print(f"Deleted {len(delete)} files off the car.")
    return 0


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Copy the car's MCAP recordings to this Mac, verified.")
    p.add_argument("-o", "--dest", default=DEFAULT_DEST,
                   help=f"where they go (default {DEFAULT_DEST}); never inside "
                        "a git work tree")
    p.add_argument("-n", "--count", type=int, default=0,
                   help="only the newest N recordings (default: all)")
    p.add_argument("--records-only", action="store_true",
                   help="only the crawler-record-* training sessions, not "
                        "the rolling drive logs")
    p.add_argument("--list", action="store_true",
                   help="say what is on the car and stop")
    p.add_argument("--prune", action="store_true",
                   help="after copying, list the car's files that have a "
                        "verified copy here, for deletion")
    p.add_argument("--yes", action="store_true",
                   help="with --prune, actually delete them")
    p.add_argument("--pulled-list", default=None, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    dest = os.path.abspath(os.path.expanduser(args.dest))
    why = refuse_reason(dest)
    if why:
        print(why, file=sys.stderr)
        return 2
    if args.yes and not args.prune:
        print("--yes only means something with --prune.", file=sys.stderr)
        return 2

    files, state, pid, open_paths, df = survey(CAR, REMOTE_DIRS)
    if args.records_only:
        others = len(files)
        files = select(files, records_only=True)
        show(files, state, pid, open_paths, "record sessions")
        print(f"  ({others - len(files)} drive logs left on the car)")
    else:
        show(files, state, pid, open_paths)
    print_df(df[:1], "before")
    if args.list:
        return 0
    files = select(files, count=args.count)

    copied, failed = pull(CAR, files, dest, args.pulled_list)
    if args.prune:
        if failed:
            print("Not pruning: a copy failed verification this run.")
        else:
            prune(CAR, files, dest, args.yes)
            if args.yes:
                print_df(survey(CAR, REMOTE_DIRS)[4][:1], "after")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
