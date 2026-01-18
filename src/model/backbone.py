import mlx.nn as nn

class SimpleBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),   # 224 → 112
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 112 → 56
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 56 → 28
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 28 → 14
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 14 → 7
            nn.ReLU(),
        )

    def __call__(self, x):
        return self.net(x)

    def parameters(self):
        return self.net.parameters()
