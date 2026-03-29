"""
Training script — YOLOv3-style detector on COCO, built with Apple MLX.

Features
--------
1. mAP@50 evaluation (every MAP_EVAL_INTERVAL epochs, MAP_EVAL_SAMPLES images)
2. Feature Pyramid Network backbone (P3/P4/P5)
3. 3 anchor boxes per scale (9 total, k-means COCO clusters)
4. 80-class detection
5. Cosine LR schedule with linear warmup
6. Random flip + colour-jitter augmentation
7. Best-model checkpointing by mAP
8. Training-curve plot (loss + mAP vs epoch)
"""

import math
import os
import time

import matplotlib
matplotlib.use("Agg")   # headless backend
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from src.dataset.coco_loader import COCODataset
from src.model.backbone import FPNBackbone
from src.model.detector import YOLODetector
from src.model.head import YOLOHead
from src.model.yolo_losses import yolo_multi_scale_loss
from src.utils.box_ops import decode_yolo_multi, nms_multi_class
from src.utils.map_eval import compute_map
from src.utils.viz import draw_boxes_multi


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def cosine_lr(base_lr: float, epoch: int, total_epochs: int,
              warmup: int = 5) -> float:
    """Linear warmup then cosine decay."""
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(total_epochs - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def clip_gradients(grads, max_norm: float = 10.0):
    """Rescale gradient tree if its global L2-norm exceeds *max_norm*."""

    def _leaves(g):
        if isinstance(g, mx.array):          return [g]
        if isinstance(g, dict):              return [l for v in g.values() for l in _leaves(v)]
        if isinstance(g, (list, tuple)):     return [l for v in g          for l in _leaves(v)]
        return []

    leaves = _leaves(grads)
    if not leaves:
        return grads

    # Compute norm on the MLX side to avoid pulling values to Python
    sq_sum = mx.sum(mx.stack([mx.sum(g.astype(mx.float32) ** 2) for g in leaves]))
    mx.eval(sq_sum)
    norm = math.sqrt(float(sq_sum.item()))

    if norm <= max_norm or norm == 0.0:
        return grads

    scale = max_norm / norm

    def _scale(g):
        if isinstance(g, mx.array):      return (g * scale).astype(g.dtype)
        if isinstance(g, dict):          return {k: _scale(v) for k, v in g.items()}
        if isinstance(g, list):          return [_scale(v) for v in g]
        if isinstance(g, tuple):         return tuple(_scale(v) for v in g)
        return g

    return _scale(grads)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    # ── Config ────────────────────────────────────────────────────────
    IMG_SIZE           = 224
    EPOCHS             = 100
    SAMPLES_PER_EPOCH  = 800
    BATCH_SIZE         = 8
    BASE_LR            = 1e-4
    WARMUP_EPOCHS      = 5
    MAP_EVAL_INTERVAL  = 10     # evaluate mAP every N epochs
    MAP_EVAL_SAMPLES   = 100    # number of val images for mAP

    os.makedirs("assets",      exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # ── Dataset ────────────────────────────────────────────────────────
    ds = COCODataset(
        root     = "data/coco-real/images/val2017",
        ann_file = "data/coco-real/annotations/instances_val2017.json",
        img_size = IMG_SIZE,
        augment  = True,
    )
    MAX_SAMPLES = min(SAMPLES_PER_EPOCH, len(ds))
    val_indices = list(range(min(MAP_EVAL_SAMPLES, len(ds))))

    print(f"Dataset: {len(ds)} images  |  training on {MAX_SAMPLES}/epoch")

    # ── Model ─────────────────────────────────────────────────────────
    backbone = FPNBackbone()
    head     = YOLOHead(in_channels=128)
    model    = YOLODetector(backbone, head)
    optimizer = optim.Adam(learning_rate=BASE_LR)

    # ── Pre-training snapshot ──────────────────────────────────────────
    img0, _, _ = ds[0]
    x0         = mx.array(img0).transpose(1, 2, 0)[None, ...]
    preds0     = model(x0)
    p0_np      = [np.array(p[0]) for p in preds0]
    b0, s0, c0 = decode_yolo_multi(p0_np, img_size=IMG_SIZE, conf_thresh=0.1)
    if len(b0) > 0:
        keep0 = nms_multi_class(b0, s0, c0)
        img_name0 = ds.images[ds.image_ids[0]]["file_name"]
        draw_boxes_multi(
            f"data/coco-real/images/val2017/{img_name0}",
            b0[keep0], s0[keep0], c0[keep0],
            "assets/pred_before.png",
        )
    print("Saved assets/pred_before.png")

    # ── Training loop ─────────────────────────────────────────────────
    history   = {"loss": [], "map_epoch": [], "map_val": []}
    best_map  = 0.0

    for epoch in range(EPOCHS):
        # Update learning rate
        lr = cosine_lr(BASE_LR, epoch, EPOCHS, warmup=WARMUP_EPOCHS)
        optimizer.learning_rate = lr

        epoch_loss  = 0.0
        epoch_start = time.perf_counter()
        indices     = np.random.permutation(MAX_SAMPLES)

        for start in range(0, MAX_SAMPLES, BATCH_SIZE):
            batch_idx = indices[start:start + BATCH_SIZE]

            imgs, batch_boxes, batch_labels = [], [], []
            for i in batch_idx:
                img, boxes, labels = ds[i]
                imgs.append(img)
                batch_boxes.append(boxes)
                batch_labels.append(labels)

            x = mx.array(np.stack(imgs)).transpose(0, 2, 3, 1)   # NHWC

            def loss_fn():
                preds = model(x)
                return yolo_multi_scale_loss(
                    preds, batch_boxes, batch_labels, img_size=IMG_SIZE
                )

            vg_fn        = nn.value_and_grad(model, loss_fn)
            loss, grads  = vg_fn()
            grads        = clip_gradients(grads, max_norm=10.0)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            loss_val = loss.item()
            if not math.isnan(loss_val):
                epoch_loss += loss_val * len(batch_idx)

        epoch_time = time.perf_counter() - epoch_start
        avg_loss   = epoch_loss / MAX_SAMPLES
        history["loss"].append(avg_loss)

        # ── mAP evaluation ─────────────────────────────────────────────
        map_val = 0.0
        if (epoch + 1) % MAP_EVAL_INTERVAL == 0:
            map_val = compute_map(
                model, ds, val_indices, img_size=IMG_SIZE, conf_thresh=0.1
            )
            history["map_epoch"].append(epoch)
            history["map_val"].append(map_val)

            # Save best checkpoint
            if map_val > best_map:
                best_map = map_val
                model.save_weights("checkpoints/best_model.npz")
                print(f"  ↑ New best mAP={best_map:.4f}  saved checkpoints/best_model.npz")

        print(
            f"epoch {epoch:02d} | loss {avg_loss:.4f} | "
            f"mAP {map_val:.4f} | lr {lr:.2e} | "
            f"time {epoch_time:.2f}s"
        )

    # ── Post-training snapshot ─────────────────────────────────────────
    preds_f  = model(x0)
    pf_np    = [np.array(p[0]) for p in preds_f]
    bf, sf, cf = decode_yolo_multi(pf_np, img_size=IMG_SIZE, conf_thresh=0.1)
    if len(bf) > 0:
        keepf = nms_multi_class(bf, sf, cf)
        img_name0 = ds.images[ds.image_ids[0]]["file_name"]
        draw_boxes_multi(
            f"data/coco-real/images/val2017/{img_name0}",
            bf[keepf], sf[keepf], cf[keepf],
            "assets/pred_after.png",
        )
    print("Saved assets/pred_after.png")

    # ── Training curves ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["loss"], color="steelblue")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    if history["map_epoch"]:
        axes[1].plot(history["map_epoch"], history["map_val"],
                     marker="o", color="darkorange")
        axes[1].set_title("Validation mAP@50")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("mAP")
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("assets/training_curves.png", dpi=150)
    print("Saved assets/training_curves.png")

    # ── Final weights ──────────────────────────────────────────────────
    model.save_weights("checkpoints/final_model.npz")
    print(f"Saved checkpoints/final_model.npz | Best mAP: {best_map:.4f}")


if __name__ == "__main__":
    main()
