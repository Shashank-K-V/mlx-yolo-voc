import mlx.nn as nn

class YOLOHead(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.pred = nn.Conv2d(64, 5 + num_classes, 1)

    def __call__(self, feats):
        return self.pred(feats)

    def parameters(self):
        return self.pred.parameters()
