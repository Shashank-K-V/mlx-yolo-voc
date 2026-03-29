"""Visualisation helpers — draw predicted boxes on images."""

import numpy as np
from PIL import Image, ImageDraw

from src.model.anchors import COCO_CLASSES

# One colour per class (cycles for safety)
_PALETTE = [
    "red", "dodgerblue", "limegreen", "gold", "darkorchid", "darkorange",
    "deepskyblue", "tomato", "mediumseagreen", "hotpink", "royalblue",
    "coral", "springgreen", "violet", "turquoise", "chocolate",
] * 5   # 80 entries


def draw_boxes_multi(image_path, boxes, scores, class_ids, out_path,
                     box_img_size=224):
    """
    Draw multi-class predictions on *image_path* and save to *out_path*.

    Parameters
    ----------
    boxes      : (N, 4) float  [x1, y1, x2, y2] in box_img_size coords
    scores     : (N,)   float
    class_ids  : (N,)   int    COCO class index 0-79
    """
    img  = Image.open(image_path).convert("RGB")
    w, h = img.size
    sx   = w / box_img_size
    sy   = h / box_img_size
    draw = ImageDraw.Draw(img)

    for box, score, cid in zip(boxes, scores, class_ids):
        if not np.all(np.isfinite(box)):
            continue
        x1s = float(box[0]) * sx
        y1s = float(box[1]) * sy
        x2s = float(box[2]) * sx
        y2s = float(box[3]) * sy
        color = _PALETTE[int(cid) % len(_PALETTE)]
        draw.rectangle([x1s, y1s, x2s, y2s], outline=color, width=2)
        label = f"{COCO_CLASSES[int(cid)]}: {score:.2f}"
        draw.text((x1s, max(0.0, y1s - 12)), label, fill=color)

    img.save(out_path)


def draw_boxes(image_path, boxes, scores, out_path, box_img_size=224):
    """Legacy single-class draw — red boxes, score label."""
    img  = Image.open(image_path).convert("RGB")
    w, h = img.size
    sx   = w / box_img_size
    sy   = h / box_img_size
    draw = ImageDraw.Draw(img)

    for box, score in zip(boxes, scores):
        if not np.all(np.isfinite(box)):
            continue
        x1s = float(box[0]) * sx
        y1s = float(box[1]) * sy
        x2s = float(box[2]) * sx
        y2s = float(box[3]) * sy
        draw.rectangle([x1s, y1s, x2s, y2s], outline="red", width=2)
        draw.text((x1s, max(0.0, y1s - 12)), f"{score:.2f}", fill="red")

    img.save(out_path)
