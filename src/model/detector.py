import mlx.nn as nn


class YOLODetector(nn.Module):
    """Thin wrapper that connects the FPN backbone to the multi-scale head."""

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def __call__(self, x):
        p3, p4, p5 = self.backbone(x)
        # Returns a 3-tuple: (pred_p3, pred_p4, pred_p5)
        # Each tensor is (B, S, S, NUM_ANCHORS*(5+NUM_CLASSES))
        return self.head(p3, p4, p5)
