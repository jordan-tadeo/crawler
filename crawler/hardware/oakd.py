"""OAK-D stereo depth, read without ever blocking the control thread.

Every getter here is a tryGet: it returns None when no packet has landed
rather than waiting for one. That is the whole point of this file. The
blocking queue.get() it used to use meant that if the USB link wedged, the
control thread parked inside get() forever -- the loop stopped, the stop
signal was never checked, and because pigpio latches a servo pulse width
and keeps emitting it, the car carried on at whatever throttle it had until
the battery died. There is no hardware failsafe and no receiver to fall
back on, so a stall in here is a runaway.

Note for the Pi: a Luxonis device needs a udev rule before a non-root
service user can open it, or dai.Device() just fails. Put this in
/etc/udev/rules.d/80-movidius.rules and reload:

    SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"

I am not installing that from install-service.sh -- it is a one-time
system change and I want to run it myself.

Who owns one of these: camera.CameraHub, and nothing else. dai.Device() takes
exclusive USB access and blocks for seconds while the device boots, so there
is one instance per process, it is constructed on a background thread, and the
hub is the only thing that reads the queues. Consumers take a
camera.StreamView instead of an OakD -- same getters, one slot each, nobody
stealing anybody's frames.

The colourmapping that used to live in here has moved to StreamView for that
reason. get_depth_colormap() pulled a frame off the depth queue as a side
effect of drawing a picture, so a debug pane and the driver were competing for
the same packets. Everything in here now is a raw frame off one queue.
"""

import depthai as dai
import numpy as np

from crawler.autopilot import policy_imu


def frame_from_payload(packet, label=""):
    """An ImgFrame's pixels, worked out from the payload instead of the label.

    getFrame() trusts ImgFrame.getType(), and with extended disparity on, that
    label is wrong. depthai 2.30 reports maxDisparity as 8170 rather than 190
    as soon as any post-processing filter is enabled -- spatial, temporal or
    speckle, any one of them is enough -- and because 8170 does not fit in a
    byte the disparity output gets stamped Type.RAW16. The firmware still sends
    one byte per pixel, so getFrame() asks for 512000 bytes, finds 256000 and
    raises. Measured on the car: extended plus filters is RAW16 carrying
    1.00 bytes/px, extended with no filters is RAW8 carrying 1.00, and subpixel
    plus filters is RAW16 genuinely carrying 2.00. So only the combination this
    pipeline now runs is mislabelled, which is why it never showed up before.

    Depth is not affected -- it is real millimetres, RAW16, two bytes a pixel in
    every configuration -- so nothing the driver reads goes through here. It is
    the disparity stream, which only ever becomes a picture for me to look at.

    Returns None rather than raising on anything it cannot make sense of, which
    is the contract every getter in this file keeps: a missing frame is a case
    the callers already handle, an exception on the camera thread is not.
    """
    data = packet.getData()
    width, height = packet.getWidth(), packet.getHeight()
    pixels = width * height
    if pixels <= 0 or data.size == 0 or data.size % pixels:
        print(f"[OakD] {label}: {data.size} bytes for {width}x{height}, dropped.")
        return None
    per_pixel = data.size // pixels
    if per_pixel == 1:
        return data.reshape(height, width)
    if per_pixel == 2:
        return data.view(np.uint16).reshape(height, width)
    print(f"[OakD] {label}: {per_pixel} bytes per pixel, dropped.")
    return None


# How fast the mono pair is allowed to shutter, and how much of the Myriad X
# the depth post-processing gets. These two are one setting, because each one
# is useless without the other, and together they are the whole of why the
# depth frame the driver reads is 88 ms old instead of 642.
#
# What was happening. The monos defaulted to 30 fps and nothing capped them.
# The stereo node with 2 post-processing shaves can only finish 12.25 frames a
# second, so for every frame it completed, the sensors handed it two and a half
# more. The surplus does not get dropped at the front -- it sits in the
# device's frame pools -- so a steady-state queue formed and every depth frame
# arrived having waited behind about eight others. Measured on the car, 600
# frames, static scene: 12.25 fps delivered and a mean age of 641 ms between
# the packet's own timestamp and dai.Clock.now() at receipt. 641 ms is 7.9
# frame intervals, and that is what those eight queued frames cost.
#
# It is not the USB link and never was. The whole config was measured with
# depth, disparity, the rectified mono and the IMU all streaming, and again
# with disparity and the rectified stream unlinked -- half the transported
# bytes gone, 12.13 fps against 12.25. Transport is free here: the device
# re-enumerates as 03e7:f63b on the Pi's 5 Gbps bus once the pipeline is
# uploaded, and getUsbSpeed() says SUPER. The 480 Mbps that lsusb shows at
# idle is the Myriad bootloader ROM, which is USB-2 only. Do not go looking
# for bandwidth; this node is compute-bound.
#
# So the fix is both halves:
#
# The shaves raise what the pipeline can finish. Measured capability, uncapped:
# 12.25 fps at 2 shaves, 15.17 at 3, 17.29 at 4, 18.69 at 5, 19.50 at 6. The
# device asks for 3 in its own log -- "It's recommended to assign more than 2
# post-processing shaves using 'setPostProcessingHardwareResources' due to
# performance reasons. Recommended: 3" -- and 3 is genuinely where the big
# step is. 5 is past the recommendation on purpose, for the headroom reason
# below. There is nothing else on this device to starve: the pipeline is two
# MonoCameras, a StereoDepth, an IMU and some XLinkOuts, no NeuralNetwork, and
# the IMU measured 104.7 Hz at 2 shaves and 104.7 Hz at 5. A blob would change
# that calculation and the shaves would have to come back.
#
# The fps cap stops the surplus being produced at all, and this is the half
# that actually collapses the latency. Capping alone does nothing -- 15 fps at
# 2 shaves is 12.28 fps and 622 ms, because 15 still over-produces against a
# 12.25 capability. Capping below capability empties the queue: 5 shaves and
# 16 fps is 16.00 fps delivered at a mean age of 87.5 ms, min 84.3, max 90.8
# over 600 frames. One and a bit frame intervals, which is as fresh as a frame
# can be.
#
# Which is why 16 and not 19. The cap has to sit under capability with room to
# spare, because going over is not a graceful degradation -- it is the whole
# backlog back. Measured: 3 shaves capped at 20 gives 511 ms, 5 shaves capped
# at 18 gives 88 ms but has only 3.7% of margin left. 16 against a measured
# 18.69 leaves 14%, which is what a scene harder than the one I measured on
# gets to eat. And 16 fps is already 1.6 frames per autopilot decision at
# AUTO_TICK_INTERVAL = 0.1, so there is nothing upstairs asking for more.
#
# None of this moves depth. Same static scene, 600 frames each: median over a
# 100x100 centre ROI 519.4 mm at baseline and 519.6 mm here, valid-pixel
# density 0.890 and 0.892, first-percentile depth 323 mm and 325 mm. Depth is
# still real millimetres and every threshold in driver.py still means what it
# said.
#
# One thing I have not measured and should: a 16 fps cap lets auto-exposure
# reach a 62 ms exposure where 30 fps bounded it at 33, so a fast pan could
# blur more than it used to. Static bench scene cannot show that. It wants a
# driving run.
CAPTURE_FPS = 16
POST_PROCESSING_SHAVES = 5
POST_PROCESSING_SLICES = 3


# The IMU, a BNO086. LINEAR_ACCELERATION and GYROSCOPE_RAW are what the
# stuck detector and every recording so far read, and they do not change.
# GAME_ROTATION_VECTOR is new, and it is there for one thing: which way is
# down. LINEAR_ACCELERATION has gravity taken out, so a parked car reads ~0
# and nothing recorded says whether the car is on a slope.
#
# Why this one, off the depthai docs (dai.IMUSensor's own docstrings and the
# IMU node page):
#
#   * It is "an orientation output ... expressed as a quaternion with no
#     specific reference for heading, while roll and pitch are referenced
#     against gravity", fused from the accelerometer and the gyro. Roll and
#     pitch are the whole of what tilt is; the heading drift it admits to is
#     a number I never read.
#   * No magnetometer. ROTATION_VECTOR fuses one in for heading, and a
#     magnetometer a few centimetres from a brushed motor and an ESC is a
#     liability for a number I do not want.
#   * It has its own slot. An IMUPacket carries one acceleroMeter, one
#     gyroscope, one magneticField and one rotationVector. ACCELEROMETER_RAW
#     (or GRAVITY) would report into the same acceleroMeter slot that
#     LINEAR_ACCELERATION already fills, so the existing imu_accel_* columns
#     would no longer say which of the two they were. The rotation vector
#     lands in rotationVector and nothing that exists moves.
#   * Fused, not raw. Tilt off ACCELEROMETER_RAW is gravity plus every bump,
#     and a bump is exactly the moment the policy needs tilt and jolt told
#     apart. RAW is also "without any postprocessing", uncalibrated.
#
# The rate. 100 Hz is what LINEAR_ACCELERATION and GYROSCOPE_RAW have always
# been asked for. The BNO "rounds up" to its next available rate, and the
# documented ones are 15/31/62/125/250/500 Hz for the accelerometer family and
# 25/33/50/100/200/400 Hz for the gyro family, so by that table the accel
# runs at 125 Hz and the gyro at 100. The hub measured 87.6 Hz of delivered
# IMU messages on the car (driver.py). The rotation vector is fused at up to 400 Hz; 100 is
# plenty for a car that decides at 10. The ~16 /imu messages a second a
# recording shows is not the sensor: telemetry.INTERVALS gates /imu at 20 Hz
# (0.05 s) off the sample it let through, so each period loses part of a
# sample's worth. /imu/batch records every sample (telemetry.ImuBatcher).
#
# The "imu" XLinkOut queue is opened shallow and non-blocking like every
# other one (OakD.__init__), so a third report in each packet cannot back
# anything up.
IMU_SENSORS = [dai.IMUSensor.LINEAR_ACCELERATION, dai.IMUSensor.GYROSCOPE_RAW,
               dai.IMUSensor.GAME_ROTATION_VECTOR]
IMU_RATE_HZ = 100


def imu_reading(data):
    """An IMUData message -> the sample dict every consumer reads, or None.

    "accel" and "gyro" are the first packet's, as they always were. "up" is
    present only when a packet in the message carries a real rotation vector
    (policy_imu.up_from_quaternion); depthai leaves the quaternion all zeros
    in a packet the rotation vector did not report into.
    """
    if data is None or not data.packets:
        return None
    packet = data.packets[0]
    accel = packet.acceleroMeter
    gyro = packet.gyroscope
    sample = {"accel": (accel.x, accel.y, accel.z),
              "gyro": (gyro.x, gyro.y, gyro.z)}
    for p in data.packets:
        rv = p.rotationVector
        up = policy_imu.up_from_quaternion(rv.i, rv.j, rv.k, rv.real)
        if up is not None:
            sample["up"] = up
            break
    return sample


def build_pipeline(rectified: bool = True) -> dai.Pipeline:
    """The whole device config, with no device attached.

    Split out of OakD.__init__ so the settings above can be asserted without
    an OAK-D plugged in -- dai.Pipeline() is pure host-side bookkeeping and
    dai.Device() is the only line that needs hardware. Everything OakD used to
    build inline is built here and nothing else changed.
    """
    pipeline = dai.Pipeline()

    # Mono cameras
    cam_left = pipeline.create(dai.node.MonoCamera)
    cam_right = pipeline.create(dai.node.MonoCamera)
    cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    cam_left.setFps(CAPTURE_FPS)
    cam_right.setFps(CAPTURE_FPS)

    # Stereo depth node
    stereo = pipeline.create(dai.node.StereoDepth)

    # Create IMU node
    imu = pipeline.create(dai.node.IMU)
    imu.enableIMUSensor(IMU_SENSORS, IMU_RATE_HZ)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)

    # IMU output
    imu_out = pipeline.create(dai.node.XLinkOut)
    imu_out.setStreamName("imu")
    imu.out.link(imu_out.input)

    # === GUI-EQUIVALENT SETTINGS ===
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setConfidenceThreshold(245)
    stereo.setLeftRightCheck(True)
    # Extended disparity rather than subpixel. These two are mutually
    # exclusive on RVC2, so it is a trade and not an addition: extended
    # runs the stereo engine twice, once at half resolution, and
    # concatenates the result into a 190 level disparity search instead of
    # 95. That halves the minimum measurable depth --
    # fx * baseline / levels = 441.25 * 75 / 190 = 174 mm against 348 --
    # which is what I actually want, because the obstacle trigger is at
    # 400 mm and with subpixel on there were only 52 mm of measurable
    # warning above the blind zone. Now there are 226 mm.
    #
    # What I give up is subpixel interpolation, so depth comes back
    # quantised to whole disparity levels. One level is Z^2 / (fx*baseline)
    # = 4.8 mm at the 400 mm trigger and 24.5 mm at the 900 mm floor clip,
    # where subpixel's eighth-of-a-level gave 0.6 mm and 3.1 mm. Neither
    # matters here: 4.8 mm of jitter on a 400 mm threshold is 1.2%, and the
    # floor clip at 899 is a band edge with 12 degrees of pitch margin
    # behind it, not a measurement anybody reads. 174 mm of range is worth
    # more than a millimetre of precision I never look at.
    #
    # Measured, 300 frames a side, same scene, camera not moved: nearest
    # valid depth went from 319 mm on subpixel to 165 mm on extended, and
    # readings in the 200-250 mm band exist only on extended -- 25130 pixel
    # samples against zero. 12.1 fps and 0.64 s of frame latency either
    # way, so the second stereo pass costs nothing measurable here. Those
    # two figures are the old 2-shave, uncapped pipeline and are no longer
    # what the car runs -- see CAPTURE_FPS above -- but the point survives,
    # which was that extended and subpixel cost the same.
    #
    # One surprise worth writing down: 348 was never the hard floor the
    # arithmetic says. Subpixel returned depth down to 319 mm, roughly 4%
    # of valid pixels sitting under 348, because the spatial and temporal
    # filters below interpolate across the edge of the search range. So
    # the gain is real but it is a floor dropping from ~320 to ~165, not
    # from 348 to 174.
    stereo.setSubpixel(False)
    stereo.setExtendedDisparity(True)
    stereo.setRectifyEdgeFillColor(0)
    # The align stays, and the argument for it is valid-pixel density -- not
    # depth. It looks like a no-op and the device itself encourages that
    # reading: there is no ColorCamera in this pipeline, and the log says
    # "Align to RGB requested, but ColorCamera not configured. Can't determine
    # output resolution automatically. Using MonoCamera resolution." It is not
    # a no-op. The EEPROM carries RGB calibration whether or not that node
    # exists, so the warp happens and the mono output size is kept.
    #
    # What it buys, measured on the car, 300 frames per config on one scene:
    # 94.1% of the full frame valid with the align, 80.0% without. That 14
    # point gap reproduces an earlier run on a different scene, 89.3% against
    # 76.8%, so the density effect is the robust part of this. The depth shift
    # that same earlier run reported did not reproduce -- against a tape
    # measure the two configs read 1048.0 mm and 1051.0 mm on a 1041.4 mm
    # target -- so treat the align as free of geometric cost and see FX_AT_640
    # in driver.py for that whole measurement.
    #
    # Why density is worth buying: driver.py's MIN_VALID_FRACTION = 0.1 marks a
    # footprint probe NO_RETURN when it cannot produce a tenth of its samples
    # as valid-and-far, a NO_RETURN probe is blocked, and a blocked column is a
    # reverse. Invalid pixels turn into reverses, so the align is buying
    # forward motion. And it costs no frame rate at all: 16.00 fps measured
    # either way. Anybody about to delete this as dead config is deleting 14
    # points of density and paying for it in reverses.
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    stereo.setPostProcessingHardwareResources(
        POST_PROCESSING_SHAVES, POST_PROCESSING_SLICES)

    # stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

    config = stereo.initialConfig.get()
    config.postProcessing.spatialFilter.enable = True
    config.postProcessing.spatialFilter.holeFillingRadius = 2
    config.postProcessing.spatialFilter.numIterations = 1

    config.postProcessing.temporalFilter.enable = True

    config.postProcessing.speckleFilter.enable = True
    config.postProcessing.speckleFilter.speckleRange = 28

    stereo.initialConfig.set(config)
    # === END SETTINGS ===

    # Link stereo to cameras
    cam_left.out.link(stereo.left)
    cam_right.out.link(stereo.right)

    # Depth output
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    # Disparity output (optional)
    xout_disp = pipeline.create(dai.node.XLinkOut)
    xout_disp.setStreamName("disparity")
    stereo.disparity.link(xout_disp.input)

    # The rectified left mono image: the actual camera picture, which
    # neither depth nor disparity shows me. Worth having in Foxglove for
    # checking where the thing is pointed and whether a lens is dirty --
    # a depth map looks much the same either way.
    #
    # Still opt-in, but no longer for the reason I wrote down. I had it that
    # an XLink stream costs USB bandwidth from the moment it is linked, so
    # auto mode should not pay for a picture it does not look at. Measured,
    # 200 frames a config, same scene: depth plus disparity plus this one is
    # 12.25 fps, and unlinking both this and disparity -- half the transported
    # bytes -- is 12.13. Both of those are the old 2-shave, uncapped pipeline
    # and neither is the rate the car runs at now -- see CAPTURE_FPS above --
    # but the comparison is the point and it survives: the stream is free on
    # the wire. What it is not free of is the JPEG encode on the Pi's publisher
    # thread, which is real, so the gate stays where it is and the hub still
    # only asks for it when telemetry is going to carry it.
    if rectified:
        xout_rect = pipeline.create(dai.node.XLinkOut)
        xout_rect.setStreamName("rectified_left")
        stereo.rectifiedLeft.link(xout_rect.input)

    return pipeline


class OakD:
    def __init__(self, rectified: bool = True):
        self.pipeline = build_pipeline(rectified)
        self.rectified = rectified

        # Start device
        self.device = dai.Device(self.pipeline)
        self.closed = False
        # self.device.setLogLevel(dai.LogLevel.INFO)
        # self.device.setLogOutputLevel(dai.LogLevel.INFO)

        # Every queue here is opened shallow and non-blocking -- the small
        # maxSize and the False -- and that is load-bearing, not a style
        # choice. depthai's default queue for a stream nobody asks for is deep
        # AND blocking, so every XLinkOut stream in the pipeline has to be
        # either drained or opened non-blocking. One stream left to fill will
        # jam XLink and take every other stream down with it, depth included.
        #
        # It cost me an afternoon because it does not look like a host problem.
        # I wrote a diagnostic straight against build_pipeline() and opened
        # only the "depth" queue, leaving disparity and imu undrained. Depth
        # delivered exactly 43 frames at an apparent 1.44 fps and then stopped
        # dead -- and the 43 frames that did arrive carried a healthy 87 ms
        # latency, which reads like the device dying rather than my own unread
        # queues strangling it. Re-run with all three drained non-blocking:
        # 300 frames at 16.08 fps, clean.
        #
        # So a stream that quits after a few dozen frames with good latency on
        # the ones it did deliver is this, every time. Anyone writing a
        # diagnostic against build_pipeline() has to open all of them.
        self.depth_queue = self.device.getOutputQueue("depth", 4, False)
        self.disparity_queue = self.device.getOutputQueue("disparity", 4, False)
        self.imu_queue = self.device.getOutputQueue("imu", 10, False)
        self.rectified_queue = (
            self.device.getOutputQueue("rectified_left", 2, False)
            if rectified else None)

    def get_imu_sample(self):
        if self.imu_queue is None:
            return None
        return imu_reading(self.imu_queue.tryGet())

    # There used to be a reset_pipeline() here that called self.__init__() on
    # an X_LINK error. It is gone on purpose. Rebuilding the pipeline takes
    # seconds, blocks whoever called it, and can raise halfway through and
    # leave the object half-built -- all of which is exactly what must never
    # happen on the thread that commands throttle. If the link dies, the
    # getters return None, driver.tick() zeroes the throttle, and the hub
    # (camera.py) sees lost() or the depth stall and opens a new one on its
    # own thread.

    def _try_frame(self, queue, label):
        """Newest frame off one queue, or None. Never waits, never raises.

        The X_LINK_ERROR case is a real one and it is not fatal: the link
        drops, this hands back None, and whoever asked treats a missing frame
        the way it already has to -- driver.tick() zeroes the throttle.
        """
        if queue is None:
            return None
        try:
            packet = queue.tryGet()
        except RuntimeError as e:
            if "X_LINK_ERROR" in str(e):
                print(f"[OakD] Stream error - no {label} this tick.")
                return None
            raise
        if packet is None:
            return None
        self._note_capture(label, packet)
        try:
            return packet.getFrame()
        except RuntimeError as e:
            if "enough data" not in str(e):
                raise
            return frame_from_payload(packet, label)

    def _note_capture(self, label, packet):
        """Keep when the packet's frame was taken, for capture_time()."""
        try:
            t = packet.getTimestamp().total_seconds()
        except Exception:
            t = None
        self.__dict__.setdefault("_captures", {})[label] = t

    def capture_time(self, label):
        """When the frame last handed out on `label` ("depth", ...) was
        taken: the packet's getTimestamp(), in seconds. depthai syncs that
        to the host's steady clock, which on Linux is the CLOCK_MONOTONIC
        that time.monotonic() reads. None if there has been none or it had
        no stamp.

        The voxel map places a frame by this rather than by when the reader
        got it: the frame is about 88 ms old by then (see CAPTURE_FPS) and
        the truck has moved on."""
        return self.__dict__.get("_captures", {}).get(label)

    def get_depth_frame(self):
        """Latest depth frame in millimetres, or None if none is waiting."""
        return self._try_frame(self.depth_queue, "depth")

    def get_disparity_frame(self):
        """Latest raw disparity frame, or None."""
        return self._try_frame(self.disparity_queue, "disparity")

    def get_rectified_left_frame(self):
        """Latest rectified left mono frame, or None if not enabled."""
        return self._try_frame(self.rectified_queue, "rectified left")

    def lost(self):
        """True once depthai has closed the device under me: its watchdog does
        that when the USB link dies. camera.CameraHub asks twice a second and
        opens a new one. Never raises."""
        if self.closed:
            return False
        try:
            return bool(self.device.isClosed())
        except Exception:
            return False

    def close(self):
        """Release the device. Safe to call twice."""
        if self.closed:
            return
        self.closed = True
        self.depth_queue = None
        self.disparity_queue = None
        self.imu_queue = None
        self.rectified_queue = None
        try:
            self.device.close()
        except Exception as e:
            print(f"[OakD] close failed: {e}")
