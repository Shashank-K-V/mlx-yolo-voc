import mlx.nn as nn
from src.model.anchors import NUM_ANCHORS, NUM_CLASSES


class YOLOHead(nn.Module):
    """
    Three independent prediction heads — one per FPN scale.

    Each head produces (B, S, S, NUM_ANCHORS * (5 + NUM_CLASSES)) raw logits:
        [tx, ty, tw, th, obj, cls_0, …, cls_79]  ×  NUM_ANCHORS

    Scales:
        head3 → P3 28×28  (small objects)
        head4 → P4 14×14  (medium objects)
        head5 → P5  7×7   (large objects)
    """

    def __init__(self, in_channels: int = 128):
        super().__init__()
        out = NUM_ANCHORS * (5 + NUM_CLASSES)   # 3 × 85 = 255

        self.head3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels, out, 1),
        )
        self.head4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels, out, 1),
        )
        self.head5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels, out, 1),
        )

    def __call__(self, p3, p4, p5):
        return self.head3(p3), self.head4(p4), self.head5(p5)
