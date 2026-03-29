"""
Box utilities: multi-scale/multi-anchor decode, IoU, NMS, class-aware NMS.
"""

import numpy as np

from src.model.anchors import ANCHORS, STRIDES, NUM_ANCHORS, NUM_CLASSES


# ------------------------------------------------------------------ #
# Primitives
# ------------------------------------------------------------------ #

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))


def iou(a, b):
    """IoU of two boxes in [x1,y1,x2,y2] format."""
    x1 = max(a[0], b[0]);  y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]);  y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


compute_iou = iou   # legacy alias


# ------------------------------------------------------------------ #
# Decode
# ------------------------------------------------------------------ #

def decode_yolo_multi(preds, img_size=224, conf_thresh=0.3):
    """
    Decode predictions from all three FPN scales.

    Parameters
    ----------
    preds       : list/tuple of numpy arrays, each (S, S, A*(5+C))
    img_size    : input image size in pixels
    conf_thresh : minimum obj × cls score to keep a box

    Returns
    -------
    boxes     : (N, 4) float32  [x1, y1, x2, y2] in pixel coords
    scores    : (N,)   float32  objectness × class_prob
    class_ids : (N,)   int32
    """
    all_boxes, all_scores, all_cids = [], [], []

    for si, pred in enumerate(preds):
        anchors = ANCHORS[si]
        stride  = STRIDES[si]
        S       = pred.shape[0]
        A       = NUM_ANCHORS
        C       = NUM_CLASSES

        pred = pred.reshape(S, S, A, 5 + C)

        for j in range(S):
            for i in range(S):
                for a in range(A):
                    tx, ty, tw, th = pred[j, i, a, :4]
                    obj_logit      = pred[j, i, a, 4]
                    obj            = sigmoid(obj_logit)

                    if obj < conf_thresh:
                        continue

                    cls_probs = sigmoid(pred[j, i, a, 5:])
                    cls_id    = int(np.argmax(cls_probs))
                    score     = obj * cls_probs[cls_id]

                    if score < conf_thresh:
                        continue

                    aw, ah = anchors[a]
                    cx = (i + sigmoid(tx)) * stride
                    cy = (j + sigmoid(ty)) * stride
                    bw = aw * np.exp(np.clip(tw, -10, 10))
                    bh = ah * np.exp(np.clip(th, -10, 10))

                    all_boxes.append([cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2])
                    all_scores.append(float(score))
                    all_cids.append(cls_id)

    if not all_boxes:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros(0,      dtype=np.float32),
            np.zeros(0,      dtype=np.int32),
        )

    return (
        np.array(all_boxes,  dtype=np.float32),
        np.array(all_scores, dtype=np.float32),
        np.array(all_cids,   dtype=np.int32),
    )


# ------------------------------------------------------------------ #
# NMS
# ------------------------------------------------------------------ #

def nms(boxes, scores, thresh=0.5):
    """Single-class greedy NMS. Returns kept indices."""
    if len(boxes) == 0:
        return []
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        cur = idxs[0]
        keep.append(int(cur))
        idxs = np.array([i for i in idxs[1:] if iou(boxes[cur], boxes[i]) < thresh])
    return keep


def nms_multi_class(boxes, scores, class_ids, iou_thresh=0.45, max_det=100):
    """
    Class-aware NMS: run independent NMS per class, then sort global by score.

    Returns a list of kept original indices.
    """
    if len(boxes) == 0:
        return []

    keep_global = []
    for c in np.unique(class_ids):
        mask    = (class_ids == c)
        c_idxs  = np.where(mask)[0]
        c_keep  = nms(boxes[mask], scores[mask], iou_thresh)
        keep_global.extend(c_idxs[c_keep].tolist())

    # Sort by descending score and cap
    keep_global = sorted(keep_global, key=lambda x: -scores[x])[:max_det]
    return keep_global


# ------------------------------------------------------------------ #
# Legacy single-scale decode (kept for backward compat)
# ------------------------------------------------------------------ #

def decode_yolo(pred, img_size=224, conf_thresh=0.5):
    """
    Legacy YOLOv1-style single-scale decoder.
    pred : (S, S, 6)  [tx, ty, tw, th, obj, cls]
    """
    S    = pred.shape[0]
    cell = img_size / S
    boxes, sc = [], []

    for j in range(S):
        for i in range(S):
            tx, ty, tw, th, obj, _ = pred[j, i]
            obj = sigmoid(obj)
            if obj < conf_thresh:
                continue
            cx = (i + sigmoid(tx)) * cell
            cy = (j + sigmoid(ty)) * cell
            bw = sigmoid(tw) * img_size
            bh = sigmoid(th) * img_size
            boxes.append([cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2])
            sc.append(obj)

    return np.array(boxes, dtype=np.float32), np.array(sc, dtype=np.float32)
