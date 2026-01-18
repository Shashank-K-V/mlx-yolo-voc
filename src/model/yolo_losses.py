import mlx.core as mx
import numpy as np

def yolo_v1_loss(pred, boxes, labels, img_size=224):
    B, S, _, D = pred.shape

    if len(boxes) == 0:
        return mx.sum(pred[..., 4] ** 2)

    target = np.zeros((B, S, S, 6), dtype=np.float32)

    x1, y1, x2, y2 = boxes[0]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    cell = img_size / S
    i = min(int(cx / cell), S - 1)
    j = min(int(cy / cell), S - 1)

    target[0, j, i, 0] = (cx - i * cell) / cell
    target[0, j, i, 1] = (cy - j * cell) / cell
    target[0, j, i, 2] = w / img_size
    target[0, j, i, 3] = h / img_size
    target[0, j, i, 4] = 1.0
    target[0, j, i, 5] = 1.0

    target = mx.array(target)
    return mx.sum((pred - target) ** 2)
