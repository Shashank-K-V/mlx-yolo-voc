# src/model/yolo_loss.py

import mlx.core as mx
import numpy as np


def yolo_v1_loss(pred, boxes, labels, img_size=224, lambda_coord=5.0, lambda_noobj=0.05):
    """
    pred: (B, S, S, 5 + C)
    boxes: (N, 4) in absolute image coords (x1, y1, x2, y2)
    labels: list of class ids
    """

    B, S, _, D = pred.shape
    C = D - 5

    # Build target in NumPy (clean + explicit)
    target_np = np.zeros((B, S, S, 5 + C), dtype=np.float32)

    # Week-2 scope: one image, one object
    box = boxes[0]
    cls = labels[0] - 1  # COCO category ids start at 1

    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1

    cell_size = img_size / S
    i = int(cx / cell_size)
    j = int(cy / cell_size)

    # Normalize targets
    tx = (cx - i * cell_size) / cell_size
    ty = (cy - j * cell_size) / cell_size
    tw = w / img_size
    th = h / img_size

    target_np[0, j, i, 0] = tx
    target_np[0, j, i, 1] = ty
    target_np[0, j, i, 2] = tw
    target_np[0, j, i, 3] = th
    target_np[0, j, i, 4] = 1.0  # objectness
    target_np[0, j, i, 5 + cls] = 1.0

    target = mx.array(target_np)

    obj_mask = target[..., 4:5]
    noobj_mask = 1.0 - obj_mask

    # Loss components
    box_loss = lambda_coord * mx.sum(obj_mask * (pred[..., :4] - target[..., :4]) ** 2)
    obj_loss = mx.sum(obj_mask * (pred[..., 4:5] - target[..., 4:5]) ** 2)
    #noobj_loss = lambda_noobj * mx.sum(noobj_mask * (pred[..., 4:5]) ** 2)
    noobj_loss = 0.0

    #cls_loss = mx.sum(obj_mask * (pred[..., 5:] - target[..., 5:]) ** 2)
    cls_loss =0.0
    return box_loss + obj_loss + noobj_loss + cls_loss
