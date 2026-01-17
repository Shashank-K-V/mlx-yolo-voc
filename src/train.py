import mlx.core as mx
import numpy as np
from src.dataset.coco_loader import COCOMiniDataset
from src.model.backbone import SimpleBackbone
from src.model.head import YOLOHead
from src.model.yolo_losses import yolo_v1_loss
from src.utils.box_ops import decode_yolo_v1, nms
from src.utils.viz import draw_boxes
from src.utils.optim import SGD
import mlx.optimizers as optim

def main():
    ds = COCOMiniDataset(
        root="data/coco-mini/images",
        ann_file="data/coco-mini/annotations/instances_train.json"
    )

    backbone = SimpleBackbone()
    head = YOLOHead(num_classes=1)

    params = {}
    params.update(backbone.parameters())
    params.update(head.parameters())

    optimizer = optim.SGD(learning_rate=1e-3)

    for epoch in range(20):
        img_chw, boxes, labels = ds[0]
        x = mx.array(img_chw).transpose(1, 2, 0)[None, ...]

        def loss_fn(p):
            feats = backbone(x)
            y = head(feats)
            return yolo_v1_loss(y, boxes, labels)

        loss, grads = mx.value_and_grad(loss_fn)(params)
        optimizer.update(params, grads)

        if epoch % 2 == 0:
            print(f"epoch {epoch:02d} | loss {loss.item():.4f}")


if __name__ == "__main__":
    main()
