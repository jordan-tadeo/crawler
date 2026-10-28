"""policy_input.depth_small: the one downsample training and the car share."""

import pathlib

import numpy as np
import pytest

from crawler.autopilot import policy_input as pi


def test_640x400_comes_out_80x50_uint16_millimetres():
    frame = np.full((400, 640), 1234, dtype=np.uint16)
    small = pi.depth_small(frame)
    assert small.shape == (50, 80) == pi.SHAPE
    assert small.dtype == np.uint16
    assert (small == 1234).all()


def test_each_block_is_its_nearest_valid_pixel():
    """A pole one pixel wide survives the shrink at its own distance."""
    frame = np.full((400, 640), 3000, dtype=np.uint16)
    frame[:, 83] = 450          # inside column block 10 (pixels 80..87)
    small = pi.depth_small(frame)
    assert (small[:, 10] == 450).all()
    assert (small[:, 9] == 3000).all() and (small[:, 11] == 3000).all()


def test_no_reading_stays_zero_and_never_becomes_near():
    frame = np.full((400, 640), 2000, dtype=np.uint16)
    frame[0:8, 0:8] = 0                       # a block with nothing
    frame[0:8, 8:16] = 0
    frame[0, 8:16] = 900                      # 8 of 64 valid: under a quarter
    frame[0:8, 16:24] = 0
    frame[0:2, 16:24] = 700                   # 16 of 64 valid: a quarter
    small = pi.depth_small(frame)
    assert small[0, 0] == 0
    assert small[0, 1] == 0
    assert small[0, 2] == 700
    assert small[0, 3] == 2000


def test_the_far_end_of_the_range_is_not_mistaken_for_no_reading():
    frame = np.full((400, 640), 65535, dtype=np.uint16)
    assert (pi.depth_small(frame) == 65535).all()


def test_values_are_not_clipped_or_scaled():
    frame = (np.arange(400 * 640, dtype=np.uint32) % 60000 + 1).astype(
        np.uint16).reshape(400, 640)
    small = pi.depth_small(frame)
    blocks = frame.reshape(50, 8, 80, 8)
    assert np.array_equal(small, blocks.min(axis=(1, 3)))


def reference(frame):
    """The obvious way to write it: a 4-D reshape, valid-only min, count."""
    rows, cols = frame.shape
    bh, bw = rows // pi.HEIGHT, cols // pi.WIDTH
    blocks = frame[:pi.HEIGHT * bh, :pi.WIDTH * bw].reshape(
        pi.HEIGHT, bh, pi.WIDTH, bw)
    valid = blocks != 0
    nearest = np.where(valid, blocks, 65535).min(axis=(1, 3))
    need = max(1, int(np.ceil(pi.MIN_VALID_FRACTION * bh * bw)))
    return np.where(valid.sum(axis=(1, 3)) >= need, nearest, 0).astype(np.uint16)


@pytest.mark.parametrize("shape", [(400, 640), (800, 1280), (403, 645)])
def test_the_fast_version_is_the_obvious_one_exactly(shape):
    rng = np.random.default_rng(7)
    for holes in (0.0, 0.5, 0.74, 0.76, 0.9, 1.0):
        frame = rng.integers(1, 65536, shape, dtype=np.uint16)
        frame[rng.random(shape) < holes] = 0
        frame[:8, :8] = 65535      # the far end, valid, next to the wrap
        assert np.array_equal(pi.depth_small(frame), reference(frame)), holes


def test_anything_but_2d_uint16_at_least_80x50_is_refused():
    with pytest.raises(ValueError):
        pi.depth_small(np.zeros((400, 640), dtype=np.uint8))
    with pytest.raises(ValueError):
        pi.depth_small(np.zeros((400, 640, 3), dtype=np.uint16))
    with pytest.raises(ValueError, match="smaller"):
        pi.depth_small(np.zeros((40, 64), dtype=np.uint16))


def test_another_size_is_cut_into_its_own_blocks_and_still_80x50():
    frame = np.full((800, 1280), 800, dtype=np.uint16)
    frame[:, 1279] = 300           # the last column of the last 16-wide block
    small = pi.depth_small(frame)
    assert small.shape == pi.SHAPE
    assert (small[:, 79] == 300).all() and (small[:, 78] == 800).all()


def test_nothing_else_in_the_repo_shrinks_a_frame_for_the_policy():
    """The function must not drift: the tap and the dump both call this one."""
    root = pathlib.Path(__file__).resolve().parent.parent
    for path in (root / "crawler" / "autopilot" / "footprint.py",
                 root / "deploy" / "footprint_dump.py"):
        assert "policy_input.depth_small(" in path.read_text(), path
