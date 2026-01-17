import mlx.core as mx

from src.dataset.coco_loader import COCOMiniDataset
from src.model.backbone import SimpleBackbone
from src.model.head import YOLOHead

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

    print("x:", x.shape)
    print("feats:", feats.shape)
    print("y:", y.shape)
    print("boxes:", boxes.shape, "labels:", labels)

if __name__ == "__main__":
    main()
