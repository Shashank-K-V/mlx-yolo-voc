import mlx.nn as nn

class YOLOHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pred = nn.Conv2d(in_channels, 6, 1)  # 5 + 1 class

    def __call__(self, x):
        return self.pred(x)

    def parameters(self):
        return self.pred.parameters()
