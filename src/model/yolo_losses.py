"""
Multi-scale, multi-anchor, multi-class YOLO loss (YOLOv3-style).

For each GT box the best-matching anchor across all 9 anchors (3 scales × 3)
is found via shape-IoU (anchors centred at origin vs GT shape).  That anchor
is made "responsible" for predicting the box; all other anchors penalise any
positive objectness they predict (no-object loss).
"""

import mlx.core as mx
import numpy as np

from src.model.anchors import ANCHORS, STRIDES, NUM_ANCHORS, NUM_CLASSES


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _anchor_shape_iou(box_wh, anchors_flat):
    """IoU of a GT box vs every anchor shape (both centred at origin)."""
    bw, bh = box_wh
    ious = []
    for aw, ah in anchors_flat:
        inter = min(bw, aw) * min(bh, ah)
        union = bw * bh + aw * ah - inter + 1e-9
        ious.append(inter / union)
    return ious


def _bce(logits, targets):
    """Numerically-stable binary cross-entropy from logits."""
    return mx.mean(
        mx.maximum(logits, 0) - logits * targets
        + mx.log(1 + mx.exp(-mx.abs(logits)))
    )


# ------------------------------------------------------------------ #
# Per-scale loss
# ------------------------------------------------------------------ #

def _loss_single_scale(pred, assignments, scale_idx, img_size,
                       lambda_coord=5.0, lambda_obj=1.0, lambda_noobj=0.01):
    """
    pred        : (1, S, S, A*(5+C))  raw logits for one image
    assignments : dict  (j, i, a) → (tx, ty, tw, th, cls_idx)
    """
    S = img_size // STRIDES[scale_idx]
    A = NUM_ANCHORS
    C = NUM_CLASSES

    pred_r   = pred.reshape(1, S, S, A, 5 + C)
    pred_xy  = mx.sigmoid(pred_r[..., :2])     # tx, ty → offset 0-1
    pred_wh  = pred_r[..., 2:4]                # tw, th  (log-scale)
    pred_obj = mx.sigmoid(pred_r[..., 4])      # objectness
    pred_cls = pred_r[..., 5:]                 # class logits

    # Build numpy target buffers
    txy  = np.zeros((1, S, S, A, 2), dtype=np.float32)
    twh  = np.zeros((1, S, S, A, 2), dtype=np.float32)
    tobj = np.zeros((1, S, S, A),    dtype=np.float32)
    tcls = np.zeros((1, S, S, A, C), dtype=np.float32)
    mask = np.zeros((1, S, S, A),    dtype=np.float32)

    for (j, i, a), (tx, ty, tw, th, cls_idx) in assignments.items():
        mask [0, j, i, a]           = 1.0
        txy  [0, j, i, a]           = [tx, ty]
        twh  [0, j, i, a]           = [tw, th]
        tobj [0, j, i, a]           = 1.0
        if 0 <= cls_idx < C:
            tcls[0, j, i, a, cls_idx] = 1.0

    mask_mx    = mx.array(mask)
    nomask_mx  = 1.0 - mask_mx
    txy_mx     = mx.array(txy)
    twh_mx     = mx.array(twh)
    tobj_mx    = mx.array(tobj)
    tcls_mx    = mx.array(tcls)

    # Box loss (MSE on offset & log-scale dims)
    box_loss = lambda_coord * mx.sum(
        mask_mx[..., None] * (
            (pred_xy - txy_mx) ** 2 + (pred_wh - twh_mx) ** 2
        )
    )

    # Objectness loss
    obj_loss   = lambda_obj   * mx.sum(mask_mx   * (pred_obj - tobj_mx) ** 2)
    # Use mean so the noobj penalty is independent of grid size (S²×A cells).
    noobj_loss = lambda_noobj * S * S * A * mx.mean(nomask_mx * pred_obj ** 2)

    # Class loss – BCE from logits (only responsible cells)
    cls_loss = mx.sum(
        mask_mx[..., None] * (
            mx.maximum(pred_cls, 0) - pred_cls * tcls_mx
            + mx.log(1 + mx.exp(-mx.abs(pred_cls)))
        )
    )

    return box_loss + obj_loss + noobj_loss + cls_loss


# ------------------------------------------------------------------ #
# Public entry point
# ------------------------------------------------------------------ #

def yolo_multi_scale_loss(preds, boxes_list, labels_list, img_size=224):
    """
    preds       : tuple (pred_p3, pred_p4, pred_p5)
                  each tensor (B, S, S, A*(5+C))
    boxes_list  : list[np.ndarray(N,4)]   pixel coords, one per image
    labels_list : list[np.ndarray(N,)]    class indices 0-79, one per image
    """
    all_anchors_flat = [a for scale in ANCHORS for a in scale]   # 9 anchors
    total_loss = mx.array(0.0)
    B = len(boxes_list)

    for b in range(B):
        boxes  = boxes_list[b]    # (N, 4)
        labels = labels_list[b]   # (N,)

        # ── Empty image: penalise all objectness ─────────────────────
        if len(boxes) == 0:
            for si, pred in enumerate(preds):
                S = img_size // STRIDES[si]
                pr = pred[b:b+1].reshape(1, S, S, NUM_ANCHORS, 5 + NUM_CLASSES)
                total_loss = total_loss + 0.5 * mx.sum(mx.sigmoid(pr[..., 4]) ** 2)
            continue

        # ── Assign each GT box to its best anchor ────────────────────
        scale_assignments = [{} for _ in range(3)]

        for box, cls_idx in zip(boxes, labels):
            x1, y1, x2, y2 = box
            bw  = max(float(x2 - x1), 1.0)
            bh  = max(float(y2 - y1), 1.0)
            cx  = float(x1 + x2) / 2.0
            cy  = float(y1 + y2) / 2.0

            ious      = _anchor_shape_iou((bw, bh), all_anchors_flat)
            best_a    = int(np.argmax(ious))
            si        = best_a // NUM_ANCHORS   # scale index 0/1/2
            ai        = best_a %  NUM_ANCHORS   # anchor index within scale

            stride    = STRIDES[si]
            S         = img_size // stride
            aw, ah    = ANCHORS[si][ai]

            ci = min(int(cx / stride), S - 1)
            cj = min(int(cy / stride), S - 1)

            tx = cx / stride - ci                       # 0..1
            ty = cy / stride - cj                       # 0..1
            tw = float(np.log(bw / aw + 1e-9))
            th = float(np.log(bh / ah + 1e-9))

            key = (cj, ci, ai)
            if key not in scale_assignments[si]:        # first-come wins
                scale_assignments[si][key] = (tx, ty, tw, th, int(cls_idx))

        # ── Loss per scale ────────────────────────────────────────────
        for si, pred in enumerate(preds):
            total_loss = total_loss + _loss_single_scale(
                pred[b:b+1],
                scale_assignments[si],
                si,
                img_size,
            )

    return total_loss / B


# ── Legacy alias kept for any old imports ──────────────────────────
def yolo_v1_loss(pred, boxes, labels, img_size=224, **_):
    """Thin shim so old code referencing yolo_v1_loss still compiles."""
    return yolo_multi_scale_loss(
        (pred, pred, pred), [boxes], [np.array(labels)], img_size=img_size
    )
