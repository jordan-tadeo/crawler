"""The depth picture the steering policy sees. One function, used twice.

The policy is a small CNN over a downsampled depth image. It is trained on
what record mode wrote to /policy/depth_small and it will run on the car over
what this same function makes of the live frame, so this file is the one place
the two can drift apart -- and so it is the one place it is written. Training
code reads it from here; the car calls it from here. Nothing else in the repo
shrinks a depth frame for the policy.

What it does
------------
640x400 uint16 millimetres in, 80x50 uint16 millimetres out: the frame cut
into 8x8-pixel blocks, each block replaced by the NEAREST VALID depth in it.

  * Valid means non-zero. The OAK-D writes 0 where stereo found no match.
  * A block with fewer than MIN_VALID valid pixels comes out 0. So 0 still
    means exactly what it means in the raw frame, "no reading here", and is
    never mistaken for "something touching the lens".
  * Nearest, not mean or median, because the thing this image is for is
    not hitting what is in it. A chair leg two pixels wide is 3% of its
    block: a mean puts it a metre further away than it is, a median deletes
    it, and the minimum keeps it. The cost is that one speckle of bad near
    depth survives too. MIN_VALID throws out the blocks that are mostly
    speckle; the rest is the trainer's to learn around, which it can, because
    the same speckle will be there at runtime.

No clipping, no scaling, no normalising. The values are the camera's
millimetres; turning them into network inputs is the trainer's job, and
doing it in here would make every recording carry one choice of it forever.

A frame that is not 640x400 is cut into blocks of (height // 50) x
(width // 80) from the top-left corner and any remainder rows and columns are
dropped; the output is 80x50 whatever goes in, as long as it is at least that
big.
"""

import numpy as np

# Output size, (rows, columns). 80 wide by 50 tall.
HEIGHT = 50
WIDTH = 80
SHAPE = (HEIGHT, WIDTH)

# Of the 64 pixels in an 8x8 block, at least this many must be valid for the
# block to say a distance at all. A quarter.
MIN_VALID_FRACTION = 0.25

_ONE = np.uint16(1)


def depth_small(frame) -> np.ndarray:
    """A depth frame as the policy's 80x50 uint16 picture. See the docstring.

    Raises ValueError on anything that is not a 2-D uint16 frame at least
    80x50: an input the policy was not trained on is a bug to hear about, not
    a picture to guess.

    How, since it runs on the control loop at 10 Hz: depth - 1 wraps a 0 to
    65535, so a plain min over a block never picks a no-reading unless the
    whole block is one, and + 1 wraps that back to 0. The mins and the valid
    counts are taken one block-column of pixels at a time with strided slices
    (8 passes across, then 8 down) rather than over a 4-D reshape, which numpy
    reduces slowly: 2.6 ms a frame on the Pi 4 against 4.6. The output is
    identical; tests/test_policy_input.py holds it to the plain reshape
    version over random frames.
    """
    frame = np.asarray(frame)
    if frame.ndim != 2 or frame.dtype != np.uint16:
        raise ValueError(f"need 2-D uint16 depth, got {frame.dtype} {frame.shape}")
    rows, cols = frame.shape
    bh, bw = rows // HEIGHT, cols // WIDTH
    if bh < 1 or bw < 1:
        raise ValueError(f"depth frame {frame.shape} is smaller than {SHAPE}")

    frame = frame[:HEIGHT * bh, :WIDTH * bw]
    shifted = frame - _ONE
    valid = (frame != 0).view(np.uint8)

    # Across: (rows, 80).
    nearest = shifted[:, 0::bw].copy()
    count = valid[:, 0::bw].astype(np.uint16)
    for k in range(1, bw):
        np.minimum(nearest, shifted[:, k::bw], out=nearest)
        np.add(count, valid[:, k::bw], out=count)

    # Down: (50, 80).
    small = nearest[0::bh].copy()
    total = count[0::bh].copy()
    for k in range(1, bh):
        np.minimum(small, nearest[k::bh], out=small)
        np.add(total, count[k::bh], out=total)

    small += _ONE
    need = max(1, int(np.ceil(MIN_VALID_FRACTION * bh * bw)))
    small[total < need] = 0
    return small
