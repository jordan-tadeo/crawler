"""Finding people in the hood camera's picture.

The model is a YOLO detector exported to ONNX (YOLO11n or YOLOv8n, COCO,
class 0 is a person) and run by onnxruntime, which the car already has for
the driving policy, so there is nothing new to install but the model file.
It is not in the repo: models/ is untracked, like the policy.
tools/person_model.py makes it on the Mac: YOLO11n exported with

    uvx --python 3.11 --from 'ultralytics==8.3.40' --with onnx --with onnxslim \
        --with 'torch<2.6' --with 'torchvision<0.21' \
        yolo export model=yolo11n.pt format=onnx imgsz=192,320 opset=12

then quantised to INT8 (static, QDQ, per channel, calibrated on coco128
frames in a 16:9 canvas, the detection head left in float). Input
1x3x192x320, output 1x84x1260. Copy it to models/person.onnx on the car. The
weights are Ultralytics' and AGPL-3.0.

The input is shaped for the windshield camera: 1280x720 is 16:9, so it
scales by 0.25 to 320x180 with 6 grey rows top and bottom. The first model
here was 256x320, shaped for a 4:3 webcam, and a 16:9 frame in it is 30%
padding the network still pays for. Same 0.25 scale, so a person is the
same number of pixels in either and recall is the same.

Why this and not the others
---------------------------
- The OAK-D's Myriad X would cost the Pi nothing: a MobileNet-SSD there runs
  20-50 fps by Luxonis' and users' numbers, and the hood frames could go to
  it over XLink. But that chip is already the bottleneck for the one thing
  the car drives on. oakd.py measured the stereo node compute-bound (12.25
  fps at 2 shaves, 18.69 at 5) and gave it 5 shaves for headroom, noting a
  blob would change that. A detector there takes shaves off depth, changes
  the pipeline every drive depends on, and goes down whenever the OAK-D
  does. Its own colour camera is fixed forward, too, so it cannot follow
  anybody the pan/tilt has turned to.
- MediaPipe's ObjectDetector (EfficientDet-Lite0) is a new, large
  dependency with its own protobuf and numpy pins, reported anywhere from 2
  to 20 fps on a Pi, and no better a detector than YOLO11n.
- A Coral USB stick is not on the car, and pycoral's wheels stop before
  Python 3.11.
- NCNN runs the same YOLO about 1.5-2x faster than onnxruntime on ARM, and is
  the obvious next step if the measured cost here is too high. It is a new
  dependency; onnxruntime is not.

The cost, measured on the car (Pi 4B, onnxruntime 1.31, one thread at nice
10, the service running with SLAM and the voxel map), session.run on a
1280x720 frame:

    model                p50     p95    coco128 person recall, conf 0.4
    256x320 float        232 ms  249 ms  89/254
    192x320 float        182 ms  205 ms  91/254 (Mac)
    192x320 INT8          95 ms  108 ms  93/254
    288x512 INT8         230 ms  237 ms  109/254 (Mac, float)

Letterboxing the frame is 4 ms and the pipe 4-7 ms. More threads are no
faster on a busy Pi: two cut 192x320 INT8 to 77 ms but cost 1.4 cores for
it, three and four were slower than one. At the default 5 Hz, capped by
max_duty 0.5, the 192x320 INT8 model measured 0.52 of a core (p95 121 ms),
against 0.27 for the old model at the 1.2 Hz it managed. RateGovernor halves the rate whenever the whole Pi
is under 15% idle. The journal's "Detection ... Hz, ... ms each" line every
minute is the real number. It runs in a process of its own at nice 10 on
one thread and shares nothing with the driving pipeline -- not the OAK-D,
not its USB link, not the control loop's GIL.

The worker
----------
DetectorClient owns a child process (spawn, like the SLAM worker) and a
pipe. submit() hands it one frame if it is not busy and returns at once;
poll() returns the answer when there is one and never waits. The child
drops its own priority, loads the model, and answers each frame with the
people in it, in fractions of the frame, and how long inference took.
"""

import os
import threading
import time

from crawler.vision.track import Detection

PERSON_CLASS = 0
PAD_VALUE = 114


def padding(frame_w, frame_h, in_h, in_w):
    """The fraction of a frame_w x frame_h frame's letterboxed input that is
    grey padding: work the network does for nothing."""
    scale = min(in_w / float(frame_w), in_h / float(frame_h))
    nw = min(int(round(frame_w * scale)), in_w)
    nh = min(int(round(frame_h * scale)), in_h)
    return 1.0 - (nw * nh) / float(in_w * in_h)


def letterbox(frame, in_h, in_w):
    """frame (HxWx3 BGR uint8) scaled to fit in_h x in_w without changing
    its shape, padded grey. Returns (1x3xHxW float32 RGB in 0..1, scale,
    pad_x, pad_y)."""
    import cv2
    import numpy as np
    h, w = frame.shape[:2]
    scale = min(in_w / float(w), in_h / float(h))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    if (nw, nh) != (w, h):
        frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x = (in_w - nw) // 2
    pad_y = (in_h - nh) // 2
    canvas = np.full((in_h, in_w, 3), PAD_VALUE, dtype=np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = frame
    tensor = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)
    tensor *= 1.0 / 255.0
    return tensor, scale, pad_x, pad_y


def decode(output, frame_w, frame_h, scale, pad_x, pad_y, conf=0.4, iou=0.45,
           max_people=10):
    """People in a YOLO output (1 x (4+classes) x N, boxes as centre x, centre
    y, width, height in input pixels, class scores already through a sigmoid),
    as Detections in fractions of the original frame, best first."""
    import cv2
    import numpy as np
    out = np.asarray(output)
    if out.ndim == 3:
        out = out[0]
    if out.shape[0] < 5:
        return []
    scores = out[4 + PERSON_CLASS]
    keep = np.nonzero(scores >= conf)[0]
    if keep.size == 0:
        return []
    cx, cy, bw, bh = (out[i, keep] for i in range(4))
    s = scores[keep]
    x0 = (cx - bw / 2 - pad_x) / scale
    y0 = (cy - bh / 2 - pad_y) / scale
    x1 = (cx + bw / 2 - pad_x) / scale
    y1 = (cy + bh / 2 - pad_y) / scale
    boxes = [[float(a), float(b), float(c - a), float(d - b)]
             for a, b, c, d in zip(x0, y0, x1, y1)]
    picked = cv2.dnn.NMSBoxes(boxes, [float(v) for v in s], conf, iou)
    picked = np.array(picked).reshape(-1)
    people = []
    for i in picked[:max_people]:
        people.append(Detection(
            score=float(s[i]),
            x0=float(min(max(x0[i] / frame_w, 0.0), 1.0)),
            y0=float(min(max(y0[i] / frame_h, 0.0), 1.0)),
            x1=float(min(max(x1[i] / frame_w, 0.0), 1.0)),
            y1=float(min(max(y1[i] / frame_h, 0.0), 1.0))))
    people.sort(key=lambda d: -d.score)
    return people


class OnnxPersonDetector:
    """A YOLO .onnx on onnxruntime's CPU provider, `threads` threads."""

    def __init__(self, model_path, conf=0.4, iou=0.45, threads=1,
                 session=None):
        if session is None:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = int(threads)
            opts.inter_op_num_threads = 1
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session = ort.InferenceSession(model_path, opts,
                                           providers=["CPUExecutionProvider"])
        self.session = session
        inp = session.get_inputs()[0]
        self.input_name = inp.name
        _, _, self.in_h, self.in_w = inp.shape
        self.conf = conf
        self.iou = iou

    def __call__(self, frame):
        """(people, inference seconds) for one BGR frame."""
        h, w = frame.shape[:2]
        tensor, scale, px, py = letterbox(frame, self.in_h, self.in_w)
        t0 = time.perf_counter()
        out = self.session.run(None, {self.input_name: tensor})[0]
        dt = time.perf_counter() - t0
        return decode(out, w, h, scale, px, py, self.conf, self.iou), dt


def _lower_priority(nice):
    try:
        os.nice(int(nice))
    except Exception:
        pass


def _serve(conn, model_path, conf, iou, threads, nice, make=None):
    """The worker: load, say ready, then answer frames until the pipe closes."""
    if nice:
        _lower_priority(nice)
        # One core: OpenCV's resize would otherwise fan out over all four.
        # The thread worker passes nice 0, so this is the process's alone.
        try:
            import cv2
            cv2.setNumThreads(1)
        except Exception:
            pass
    try:
        detector = (make or OnnxPersonDetector)(model_path, conf=conf, iou=iou,
                                                threads=threads)
    except BaseException as e:
        try:
            conn.send(("error", f"{type(e).__name__}: {e}"))
        except Exception:
            pass
        return
    conn.send(("ready", (detector.in_h, detector.in_w)))
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg is None:
            return
        seq, frame = msg
        try:
            people, dt = detector(frame)
            conn.send(("people", seq,
                       [(d.score, d.x0, d.y0, d.x1, d.y1) for d in people], dt))
        except Exception as e:
            conn.send(("failed", seq, f"{type(e).__name__}: {e}"))


class DetectorClient:
    """The detector in a child process (or a thread, for tests and the Mac).

    states: "starting", "ready", "failed" (error says why), "closed".
    """

    def __init__(self, model_path, conf=0.4, iou=0.45, threads=1, nice=10,
                 worker="process", make=None):
        self.model_path = model_path
        self.args = (model_path, conf, iou, threads, nice)
        self.worker = worker
        self._make = make
        self.state = "starting"
        self.error = ""
        self.input_shape = None
        self.busy = False
        self._seq = None
        self._conn = None
        self._proc = None

    def start(self):
        if not os.path.exists(self.model_path) and self._make is None:
            self.state = "failed"
            self.error = f"no model at {self.model_path}"
            return False
        if self.worker == "process":
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            parent, child = ctx.Pipe()
            self._proc = ctx.Process(target=_serve, args=(child, *self.args),
                                     name="person-detect", daemon=True)
            self._proc.start()
            child.close()
        else:
            import multiprocessing as mp
            parent, child = mp.Pipe()
            # No nice for a thread: it would lower the whole process.
            args = (child, *self.args[:4], 0, self._make)
            self._proc = threading.Thread(target=_serve, args=args,
                                          name="person-detect", daemon=True)
            self._proc.start()
        self._conn = parent
        return True

    def submit(self, seq, frame):
        """Hand one frame over if the worker is free. Never waits."""
        if self.state != "ready" or self.busy:
            return False
        try:
            self._conn.send((seq, frame))
        except Exception as e:
            self.state = "failed"
            self.error = f"send: {e}"
            return False
        self.busy = True
        self._seq = seq
        return True

    def poll(self):
        """("people", seq, [Detection], seconds), ("failed", seq, why), or
        None when there is nothing new. Never waits."""
        conn = self._conn
        if conn is None:
            return None
        try:
            if not conn.poll(0):
                if self.worker == "process" and self._proc is not None \
                        and not self._proc.is_alive() and self.state != "closed":
                    self.state = "failed"
                    self.error = self.error or "the detector process died"
                    self.busy = False
                return None
            msg = conn.recv()
        except (EOFError, OSError) as e:
            self.state = "failed"
            self.error = self.error or f"pipe: {e}"
            self.busy = False
            return None
        kind = msg[0]
        if kind == "ready":
            self.state = "ready"
            self.input_shape = msg[1]
            return None
        if kind == "error":
            self.state = "failed"
            self.error = msg[1]
            return None
        self.busy = False
        if kind == "people":
            _, seq, rows, dt = msg
            return ("people", seq, [Detection(*r) for r in rows], dt)
        return msg

    def close(self):
        self.state = "closed"
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.send(None)
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
        proc = self._proc
        if proc is not None:
            proc.join(timeout=2.0)
            if self.worker == "process" and proc.is_alive():
                proc.terminate()
