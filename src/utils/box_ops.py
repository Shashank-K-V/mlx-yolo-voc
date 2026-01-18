import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def decode_yolo(pred, img_size=224, conf_thresh=0.3):
    S = pred.shape[0]
    cell = img_size / S

    boxes, scores = [], []

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

            boxes.append([
                cx - bw / 2,
                cy - bh / 2,
                cx + bw / 2,
                cy + bh / 2
            ])
            scores.append(obj)

    return np.array(boxes), np.array(scores)

def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)

def nms(boxes, scores, thresh=0.5):
    idxs = np.argsort(scores)[::-1]
    keep = []

    while len(idxs) > 0:
        cur = idxs[0]
        keep.append(cur)
        idxs = np.array([i for i in idxs[1:] if iou(boxes[cur], boxes[i]) < thresh])
    return keep
