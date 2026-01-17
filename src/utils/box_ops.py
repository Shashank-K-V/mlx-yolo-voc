# src/utils/box_ops.py

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode_yolo_v1(pred, img_size=224, conf_thresh=0.25):
    """
    pred: numpy array (S, S, 5 + C) for a single image
    Returns: boxes (N,4), scores (N,), class_ids (N,)
    Boxes are in absolute coords (x1,y1,x2,y2)
    """

    S, _, D = pred.shape
    C = D - 5
    cell_size = img_size / S

    boxes = []
    scores = []
    class_ids = []

    for j in range(S):
        for i in range(S):
            tx, ty, tw, th = pred[j, i, 0:4]
            obj = pred[j, i, 4]
            cls_logits = pred[j, i, 5:]

            obj = sigmoid(obj)

            # For now: class prob via sigmoid (works fine for 1 class)
            cls_probs = sigmoid(cls_logits) if C > 0 else np.array([1.0], dtype=np.float32)
            cls_id = int(np.argmax(cls_probs)) if C > 0 else 0
            cls_score = float(cls_probs[cls_id]) if C > 0 else 1.0

            score = obj * cls_score
            if score < conf_thresh:
                continue

            tx = sigmoid(tx)
            ty = sigmoid(ty)
            tw = sigmoid(tw)
            th = sigmoid(th)

            cx = (i + tx) * cell_size
            cy = (j + ty) * cell_size
            bw = tw * img_size
            bh = th * img_size

            x1 = cx - bw / 2.0
            y1 = cy - bh / 2.0
            x2 = cx + bw / 2.0
            y2 = cy + bh / 2.0

            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            class_ids.append(cls_id)

    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    return np.array(boxes, dtype=np.float32), np.array(scores, dtype=np.float32), np.array(class_ids, dtype=np.int64)


def iou_xyxy(a, b):
    """
    a: (4,), b: (4,)
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area + 1e-9
    return inter_area / union


def nms(boxes, scores, iou_thresh=0.5):
    """
    boxes: (N,4), scores: (N,)
    Returns indices of selected boxes
    """
    if len(boxes) == 0:
        return []

    idxs = np.argsort(scores)[::-1]
    keep = []

    while len(idxs) > 0:
        cur = idxs[0]
        keep.append(cur)
        rest = idxs[1:]

        filtered = []
        for r in rest:
            if iou_xyxy(boxes[cur], boxes[r]) < iou_thresh:
                filtered.append(r)

        idxs = np.array(filtered, dtype=np.int64)

    return keep
