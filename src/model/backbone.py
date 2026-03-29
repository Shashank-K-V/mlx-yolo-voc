import mlx.nn as nn


class FPNBackbone(nn.Module):
    """
    Simple 5-stage CNN backbone with a Feature Pyramid Network (FPN) top-down pathway.

    Output feature maps (all 128 channels):
        P3 → (B, 28, 28, 128)  stride=8   small objects
        P4 → (B, 14, 14, 128)  stride=16  medium objects
        P5 → (B,  7,  7, 128)  stride=32  large objects
    """

    def __init__(self):
        super().__init__()

        # ── Bottom-up stages ──────────────────────────────────────────
        self.stage1 = nn.Sequential(
            nn.Conv2d(3,   16,  3, stride=2, padding=1), nn.ReLU(),   # 224→112
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(16,  32,  3, stride=2, padding=1), nn.ReLU(),   # 112→56
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(32,  64,  3, stride=2, padding=1), nn.ReLU(),   # 56→28  C3
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(64,  128, 3, stride=2, padding=1), nn.ReLU(),   # 28→14  C4
        )
        self.stage5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),   # 14→7   C5
        )

        # ── FPN lateral 1×1 projections ──────────────────────────────
        self.lat5 = nn.Conv2d(256, 128, 1)
        self.lat4 = nn.Conv2d(128, 128, 1)
        self.lat3 = nn.Conv2d(64,  128, 1)

        # ── FPN output 3×3 convs (anti-aliasing after merge) ─────────
        self.out5 = nn.Conv2d(128, 128, 3, padding=1)
        self.out4 = nn.Conv2d(128, 128, 3, padding=1)
        self.out3 = nn.Conv2d(128, 128, 3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def __call__(self, x):
        # Bottom-up
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)   # (B, 28, 28, 64)
        c4 = self.stage4(c3)   # (B, 14, 14, 128)
        c5 = self.stage5(c4)   # (B,  7,  7, 256)

        # Top-down FPN
        p5 = nn.relu(self.out5(self.lat5(c5)))                          # (B, 7,  7,  128)
        p4 = nn.relu(self.out4(self.lat4(c4) + self.upsample(p5)))      # (B, 14, 14, 128)
        p3 = nn.relu(self.out3(self.lat3(c3) + self.upsample(p4)))      # (B, 28, 28, 128)

        return p3, p4, p5
