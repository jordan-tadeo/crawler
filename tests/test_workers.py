"""The worker processes start with one BLAS thread and nice every thread.

A spawned child imports numpy before its target runs, so the caps have to
be in the environment it is spawned with; see crawler/workers.py.
"""

import multiprocessing as mp
import os
import subprocess
import sys
import textwrap

import pytest

from crawler import workers


class RecordingProc:
    """A Process stand-in: what the environment held when start() ran."""

    def __init__(self, fail=False):
        self.fail = fail
        self.env = None

    def start(self):
        self.env = {k: os.environ.get(k) for k in workers.BLAS_ENV}
        if self.fail:
            raise OSError("no fork")


def test_the_child_is_started_with_one_blas_thread_and_the_parent_keeps_its_env(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    p = RecordingProc()
    workers.start(p)
    assert p.env == {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                     "MKL_NUM_THREADS": "1"}
    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert "OPENBLAS_NUM_THREADS" not in os.environ


def test_a_start_that_raises_still_puts_the_env_back(monkeypatch):
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    with pytest.raises(OSError):
        workers.start(RecordingProc(fail=True))
    assert "OPENBLAS_NUM_THREADS" not in os.environ


def test_a_real_spawned_child_sees_the_caps(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    mark = tmp_path / "capped"
    ctx = mp.get_context("spawn")
    # os.system runs in the child, so its shell has the child's environment.
    p = ctx.Process(target=os.system, args=(
        f'[ "$OPENBLAS_NUM_THREADS" = 1 ] && [ "$OMP_NUM_THREADS" = 1 ] && touch "{mark}"',))
    workers.start(p)
    p.join(timeout=30)
    assert mark.exists()


@pytest.mark.parametrize("site", ["voxel", "slam", "person"])
def test_every_worker_process_is_started_through_the_caps(monkeypatch, tmp_path, site):
    started = []

    def record(proc, env=workers.BLAS_ENV):
        started.append(proc.name)
        proc.start()

    monkeypatch.setattr(workers, "start", record)
    if site == "voxel":
        from crawler.mapping import voxel as vx
        w = vx.VoxelMapper(None, mode=vx.MODE_PROCESS)
        name = "voxel"
    elif site == "slam":
        pytest.importorskip("scipy")
        from crawler.mapping import runner as rn
        w = rn.SlamRunner(mode=rn.MODE_PROCESS)
        name = "slam"
    else:
        from crawler.vision import person as ps
        model = tmp_path / "none.onnx"
        model.write_bytes(b"")
        w = ps.DetectorClient(str(model), worker="process")
        name = "person-detect"
    try:
        w.start()
        assert started == [name]
    finally:
        w.close()


@pytest.mark.skipif(not os.path.isdir("/proc/self/task"), reason="per-thread nice is Linux's")
def test_lower_priority_renices_threads_started_before_it():
    # In a child of its own: a nice cannot be taken back.
    code = textwrap.dedent("""
        import os, threading
        from crawler import workers
        go = threading.Event()
        t = threading.Thread(target=go.wait, daemon=True)
        t.start()
        workers.lower_priority(5)
        print(sorted({os.getpriority(os.PRIO_PROCESS, int(tid))
                      for tid in os.listdir("/proc/self/task")}))
        go.set()
    """)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True,
                         text=True, timeout=30)
    base = os.getpriority(os.PRIO_PROCESS, 0)
    assert out.stdout.strip() == f"[{min(19, base + 5)}]", out.stderr
