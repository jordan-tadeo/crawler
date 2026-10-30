"""The hood camera: finding a person in its picture and keeping them centred.

person.py   the detector: a YOLO .onnx through onnxruntime, in a process of
            its own, at a rate that backs off when the Pi is busy
track.py    pure arithmetic: pixel to angle, picking the person, the pan and
            tilt targets, the rate governor
runner.py   the thread that ties the camera, the detector, the mount and
            telemetry together with --pantilt

None of it ever commands throttle or steering. The mount it drives is the
pan/tilt pair and nothing else (crawler/hardware/pantilt.py).
"""
