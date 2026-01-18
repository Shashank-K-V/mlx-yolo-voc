import mlx.nn as nn

class YOLODetector(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def __call__(self, x):
        feats = self.backbone(x)
        return self.head(feats)
