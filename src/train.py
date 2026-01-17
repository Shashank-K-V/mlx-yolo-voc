import mlx.core as mx
import numpy as np
from src.dataset.coco_loader import COCOMiniDataset
from src.model.backbone import SimpleBackbone
from src.model.head import YOLOHead
from src.model.yolo_losses import yolo_v1_loss
from src.utils.box_ops import decode_yolo_v1, nms
from src.utils.viz import draw_boxes


def main():
    ds = COCOMiniDataset(
        root="data/coco-mini/images",
        ann_file="data/coco-mini/annotations/instances_train.json"
    )

    img_chw, boxes, labels = ds[0]

    x = mx.array(img_chw).transpose(1, 2, 0)[None, ...]  # (1, 224, 224, 3)


    backbone = SimpleBackbone()
    head = YOLOHead(num_classes=1)

    feats = backbone(x)
    y = head(feats)

    loss = yolo_v1_loss(y, boxes, labels)

    # Convert MLX array -> numpy for decoding
    pred_np = np.array(y[0])  # (S,S,6)

    boxes_pred, scores_pred, cls_pred = decode_yolo_v1(pred_np, img_size=224, conf_thresh=0.25)
    keep = nms(boxes_pred, scores_pred, iou_thresh=0.5)

    boxes_nms = boxes_pred[keep]
    scores_nms = scores_pred[keep]

    print("x:", x.shape)
    print("feats:", feats.shape)
    print("y:", y.shape)
    print("loss:", loss.item())
    print("boxes:", boxes.shape, "labels:", labels)
    print("num raw boxes:", len(boxes_pred), "num after nms:", len(boxes_nms))

    # Draw on the COCO-mini image (original file is 256, but we resize to 224 in loader)
    image_path = "data/coco-mini/images/000000000001.jpg"
    out_path = draw_boxes(image_path, boxes_nms, scores_nms, out_path="assets/pred.png")
    print("saved:", out_path)

if __name__ == "__main__":
    main()
