"""Starting the mapping and detector worker processes: one BLAS thread each,
and niced as a whole.

numpy's OpenBLAS starts a pool of nproc - 1 threads the moment numpy is
imported, and a spawned child imports numpy (through its target's module)
before the target runs, so neither a cap nor an os.nice() inside the target
reaches that pool. On the Pi the voxel worker's three pool threads ran at
nice 0, ~28% of a core each and more user time than the worker's own thread,
mostly spinning between small matrix products. Timed there on a synthetic
depth frame (the 3 Hz cap, nice 10, the service running beside it), one
update was:

  OpenBLAS threads   wall median   CPU per update
  4 (default)          87-93 ms       322 ms
  1                    60-64 ms        65 ms

One thread is faster and costs a fifth of the CPU. The SLAM step was the
same either way (37 ms, 40 ms of CPU), so it is capped too: its pools only
sit there.

start() puts the caps in the environment for the moment the child is
spawned (a spawn child inherits it at exec, so numpy reads it at import),
then puts the parent's environment back. lower_priority() renices every
thread the process has, not just the caller: on Linux nice is per thread.
"""

import os
import threading

BLAS_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}

_env_lock = threading.Lock()


def start(proc, env=BLAS_ENV):
    """proc.start() with `env` in the environment it is spawned with. The
    parent's environment is the same afterwards. Raises what start() raises."""
    with _env_lock:
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            proc.start()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def _tids():
    try:
        return [int(t) for t in os.listdir("/proc/self/task")]
    except (OSError, ValueError):
        return []


def lower_priority(nice):
    """Raise the niceness of every thread of this process by `nice`. Only
    for a worker process: in the service it would lower the control loop.
    Never raises."""
    nice = int(nice)
    if not nice:
        return
    tids = _tids()
    if not tids:                 # not Linux: the call is per process there
        try:
            os.nice(nice)
        except Exception:
            pass
        return
    for tid in tids:
        try:
            now = os.getpriority(os.PRIO_PROCESS, tid)
            os.setpriority(os.PRIO_PROCESS, tid, min(19, now + nice))
        except Exception:        # a thread that ended, or no permission
            pass
