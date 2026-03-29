"""
Mean Average Precision (mAP@50) evaluation.

Follows the VOC 2010+ / COCO-style AP:
  • IoU threshold = 0.50
  • 11-point interpolated precision–recall curve per class
  • mAP = mean AP across classes that have at least one GT instance
"""

import numpy as np
import mlx.core as mx

from src.model.anchors import NUM_CLASSES
from src.utils.box_ops import decode_yolo_multi, nms_multi_class, iou


# ------------------------------------------------------------------ #
# AP helpers
# ------------------------------------------------------------------ #

def _ap_11point(recalls, precisions):
    """11-point interpolated AP (recall thresholds 0, 0.1, …, 1.0)."""
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 11):
        p = precisions[recalls >= t]
        ap += float(np.max(p)) if len(p) > 0 else 0.0
    return ap / 11.0


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def compute_map(model, dataset, indices, img_size=224,
                conf_thresh=0.3, iou_thresh=0.5):
    """
    Compute mAP@50 over the images specified by *indices*.

    Parameters
    ----------
    model     : YOLODetector (MLX)
    dataset   : COCODataset
    indices   : iterable of dataset indices to evaluate
    img_size  : input resolution
    conf_thresh: detection confidence threshold
    iou_thresh : IoU threshold for a TP match

    Returns
    -------
    float  mAP value in [0, 1]
    """
    # Per-class accumulators
    class_dets  = {c: [] for c in range(NUM_CLASSES)}   # [(score, is_tp), …]
    class_n_gt  = {c: 0  for c in range(NUM_CLASSES)}

    for idx in indices:
        img, gt_boxes, gt_labels = dataset[idx]

        # ── Run inference ────────────────────────────────────────────
        x      = mx.array(img).transpose(1, 2, 0)[None, ...]
        preds  = model(x)
        preds_np = [np.array(p[0]) for p in preds]

        pred_boxes, pred_scores, pred_cids = decode_yolo_multi(
            preds_np, img_size=img_size, conf_thresh=conf_thresh
        )

        if len(pred_boxes) > 0:
            keep        = nms_multi_class(pred_boxes, pred_scores, pred_cids)
            pred_boxes  = pred_boxes [keep]
            pred_scores = pred_scores[keep]
            pred_cids   = pred_cids  [keep]

        # ── Count GT per class ────────────────────────────────────────
        for c in gt_labels:
            class_n_gt[int(c)] += 1

        # ── Match predictions to GT ───────────────────────────────────
        for c in (np.unique(pred_cids) if len(pred_cids) > 0 else []):
            c = int(c)
            det_mask = pred_cids == c
            gt_mask  = gt_labels == c

            c_pb  = pred_boxes [det_mask]
            c_ps  = pred_scores[det_mask]
            c_gt  = gt_boxes   [gt_mask] if len(gt_boxes) > 0 else np.zeros((0, 4))

            matched = set()
            for di in np.argsort(-c_ps):
                box   = c_pb[di]
                score = float(c_ps[di])

                if len(c_gt) == 0:
                    class_dets[c].append((score, False))
                    continue

                best_iou, best_j = 0.0, -1
                for j, gt_b in enumerate(c_gt):
                    v = iou(box, gt_b)
                    if v > best_iou:
                        best_iou, best_j = v, j

                if best_iou >= iou_thresh and best_j not in matched:
                    class_dets[c].append((score, True))
                    matched.add(best_j)
                else:
                    class_dets[c].append((score, False))

    # ── Compute per-class AP, then mean ──────────────────────────────
    aps = []
    for c in range(NUM_CLASSES):
        if class_n_gt[c] == 0:
            continue

        dets = sorted(class_dets[c], key=lambda x: -x[0])
        if not dets:
            aps.append(0.0)
            continue

        tp = np.cumsum([1 if d[1] else 0 for d in dets], dtype=np.float32)
        fp = np.cumsum([0 if d[1] else 1 for d in dets], dtype=np.float32)

        recall    = tp / (class_n_gt[c] + 1e-9)
        precision = tp / (tp + fp + 1e-9)

        aps.append(_ap_11point(recall, precision))

    return float(np.mean(aps)) if aps else 0.0
