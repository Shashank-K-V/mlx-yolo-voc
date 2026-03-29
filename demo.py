#!/usr/bin/env python3
"""
Real-time webcam / video inference with the trained YOLO detector.

Usage
-----
# Webcam (camera 0)
python demo.py

# Specific camera
python demo.py --source 1

# Video file → annotated output
python demo.py --source path/to/video.mp4 --output out.mp4

# Load non-default weights
python demo.py --weights checkpoints/best_model.npz
"""

import argparse
import os
import sys

import numpy as np
import mlx.core as mx
from PIL import Image

from src.model.backbone import FPNBackbone
from src.model.head import YOLOHead
from src.model.detector import YOLODetector
from src.model.anchors import COCO_CLASSES
from src.utils.box_ops import decode_yolo_multi, nms_multi_class

# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #
IMG_SIZE    = 224
CONF_THRESH = 0.30

# BGR colour per class (OpenCV), cycles across 80 classes
_PALETTE = [
    (255,  56,  56), ( 56, 255,  56), ( 56,  56, 255),
    (255, 157,  56), (157,  56, 255), ( 56, 255, 157),
    (255, 255,  56), ( 56, 157, 255), (200, 100, 255),
    (100, 255, 200),
] * 8


# ------------------------------------------------------------------ #
# Model
# ------------------------------------------------------------------ #

def load_model(weights_path: str):
    backbone = FPNBackbone()
    head     = YOLOHead(in_channels=128)
    model    = YOLODetector(backbone, head)
    model.load_weights(weights_path)
    mx.eval(model.parameters())
    return model


# ------------------------------------------------------------------ #
# Inference on a single BGR frame
# ------------------------------------------------------------------ #

def infer_frame(model, frame_bgr):
    try:
        import cv2
    except ImportError:
        return frame_bgr

    h_orig, w_orig = frame_bgr.shape[:2]

    # Pre-process
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img   = Image.fromarray(frame_rgb).resize((IMG_SIZE, IMG_SIZE))
    img_np    = np.asarray(pil_img, dtype=np.float32) / 255.0   # (224,224,3)
    x         = mx.array(img_np)[None, ...]                      # (1,224,224,3)

    preds    = model(x)
    preds_np = [np.array(p[0]) for p in preds]

    boxes, scores, cids = decode_yolo_multi(
        preds_np, img_size=IMG_SIZE, conf_thresh=CONF_THRESH
    )
    if len(boxes) == 0:
        return frame_bgr

    keep   = nms_multi_class(boxes, scores, cids)
    boxes  = boxes [keep]
    scores = scores[keep]
    cids   = cids  [keep]

    sx = w_orig / IMG_SIZE
    sy = h_orig / IMG_SIZE

    for box, score, cid in zip(boxes, scores, cids):
        if not np.all(np.isfinite(box)):
            continue
        x1 = max(0, int(box[0] * sx))
        y1 = max(0, int(box[1] * sy))
        x2 = min(w_orig, int(box[2] * sx))
        y2 = min(h_orig, int(box[3] * sy))
        color = _PALETTE[int(cid) % len(_PALETTE)]
        label = f"{COCO_CLASSES[int(cid)]}: {score:.2f}"
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame_bgr, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return frame_bgr


# ------------------------------------------------------------------ #
# Webcam / video runners
# ------------------------------------------------------------------ #

def run_webcam(model, camera_id: int = 0):
    import cv2
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Cannot open camera {camera_id}")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated = infer_frame(model, frame)
        cv2.imshow("YOLO Detector", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_video(model, video_path: str, out_path: str | None = None):
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    writer = None
    if out_path:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated = infer_frame(model, frame)
        if writer:
            writer.write(annotated)
        n += 1
        if n % 30 == 0:
            print(f"  Processed {n} frames …")

    cap.release()
    if writer:
        writer.release()
        print(f"Saved to {out_path}")


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="YOLO real-time demo")
    parser.add_argument("--weights", default="checkpoints/best_model.npz",
                        help="Path to .npz weights file")
    parser.add_argument("--source",  default="0",
                        help="Camera index (int) or path to a video file")
    parser.add_argument("--output",  default=None,
                        help="Save annotated video to this path (optional)")
    args = parser.parse_args()

    # Check OpenCV
    try:
        import cv2   # noqa: F401
    except ImportError:
        print("OpenCV not found. Install it with:\n  pip install opencv-python")
        sys.exit(1)

    # Check weights
    if not os.path.isfile(args.weights):
        print(f"Weights not found: {args.weights}")
        print("Run training first:  python -m src.train")
        sys.exit(1)

    model = load_model(args.weights)
    print(f"Loaded weights from {args.weights}")

    try:
        run_webcam(model, camera_id=int(args.source))
    except ValueError:
        run_video(model, args.source, args.output)


if __name__ == "__main__":
    main()
