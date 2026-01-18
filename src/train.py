import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import os

from src.dataset.coco_loader import COCODataset
from src.model.backbone import SimpleBackbone
from src.model.head import YOLOHead
from src.model.detector import YOLODetector
from src.model.yolo_losses import yolo_v1_loss
from src.utils.box_ops import decode_yolo, nms
from src.utils.viz import draw_boxes


def main():
    # -----------------------------
    # Config
    # -----------------------------
    IMG_SIZE = 224
    EPOCHS = 20
    SAMPLES_PER_EPOCH = 10
    LR = 1e-3

    os.makedirs("assets", exist_ok=True)

    # -----------------------------
    # Dataset (COCO val2017, person-only)
    # -----------------------------
    ds = COCODataset(
        root="data/coco-real/images/val2017",
        ann_file="data/coco-real/annotations/instances_val2017.json",
        img_size=IMG_SIZE
    )

    # -----------------------------
    # Model
    # -----------------------------
    backbone = SimpleBackbone()              # outputs (1, 7, 7, 256)
    head = YOLOHead(in_channels=256)         # outputs (1, 7, 7, 6)
    model = YOLODetector(backbone, head)     # SINGLE MLX MODULE

    optimizer = optim.SGD(learning_rate=LR)

    # -----------------------------
    # BEFORE TRAINING (visual sanity check)
    # -----------------------------
    img, _, _ = ds[0]
    x = mx.array(img).transpose(1, 2, 0)[None, ...]

    y = model(x)
    boxes, scores = decode_yolo(np.array(y[0]), img_size=IMG_SIZE)
    keep = nms(boxes, scores)

    img_name = ds.images[ds.image_ids[0]]["file_name"]
    draw_boxes(
        f"data/coco-real/images/val2017/{img_name}",
        boxes[keep],
        scores[keep],
        "assets/pred_before.png"
    )

    print("Saved assets/pred_before.png")

    # -----------------------------
    # TRAINING LOOP (REAL GRADIENTS)
    # -----------------------------
    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        for i in range(min(SAMPLES_PER_EPOCH, len(ds))):
            img, boxes, labels = ds[i]
            x = mx.array(img).transpose(1, 2, 0)[None, ...]

            # MLX-correct autograd:
            # differentiate w.r.t. *model*, not params or functions
            vg_fn = nn.value_and_grad(
                model,
                lambda: yolo_v1_loss(model(x), boxes, labels, img_size=IMG_SIZE)
                
            )

            loss, grads = vg_fn()
            optimizer.update(model, grads)

            epoch_loss += loss.item()

        print(f"epoch {epoch:02d} | loss {epoch_loss / SAMPLES_PER_EPOCH:.4f}")

    # -----------------------------
    # AFTER TRAINING (qualitative check)
    # -----------------------------
    y = model(x)
    boxes, scores = decode_yolo(np.array(y[0]), img_size=IMG_SIZE)
    keep = nms(boxes, scores)

    draw_boxes(
        f"data/coco-real/images/val2017/{img_name}",
        boxes[keep],
        scores[keep],
        "assets/pred_after.png"
    )

    print("Saved assets/pred_after.png")


if __name__ == "__main__":
    main()
