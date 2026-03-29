#!/usr/bin/env python3
"""
Export the trained YOLO detector to ONNX via a PyTorch mirror model.

Strategy
--------
1. Load MLX weights from a .npz checkpoint.
2. Build an identical architecture in PyTorch (NCHW convention).
3. Copy every weight tensor, transposing Conv2d filters from MLX's
   (out, kH, kW, in) layout to PyTorch's (out, in, kH, kW) layout.
4. Export with torch.onnx.export → model.onnx  (opset 11).

Requirements
------------
    pip install torch onnx onnxruntime

Usage
-----
    python export_onnx.py                              # defaults
    python export_onnx.py --weights checkpoints/best_model.npz --output model.onnx
    python export_onnx.py --verify                     # run onnxruntime check
"""

import argparse
import os
import sys

import numpy as np


# ------------------------------------------------------------------ #
# Architecture constants (must match src/model/)
# ------------------------------------------------------------------ #
from src.model.anchors import NUM_ANCHORS, NUM_CLASSES

IMG_SIZE = 224
OUT_CH   = NUM_ANCHORS * (5 + NUM_CLASSES)   # 255


# ------------------------------------------------------------------ #
# PyTorch mirror
# ------------------------------------------------------------------ #

def _pt_sequential_conv(*args):
    """Conv2d + ReLU block."""
    import torch.nn as nn
    layers = []
    for in_c, out_c, k, stride, pad in args:
        layers += [nn.Conv2d(in_c, out_c, k, stride=stride, padding=pad), nn.ReLU()]
    return nn.Sequential(*layers)


def build_pytorch_model():
    import torch
    import torch.nn as nn

    class FPNBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.stage1 = nn.Sequential(nn.Conv2d( 3,  16, 3, stride=2, padding=1), nn.ReLU())
            self.stage2 = nn.Sequential(nn.Conv2d(16,  32, 3, stride=2, padding=1), nn.ReLU())
            self.stage3 = nn.Sequential(nn.Conv2d(32,  64, 3, stride=2, padding=1), nn.ReLU())
            self.stage4 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU())
            self.stage5 = nn.Sequential(nn.Conv2d(128,256, 3, stride=2, padding=1), nn.ReLU())
            self.lat5 = nn.Conv2d(256, 128, 1)
            self.lat4 = nn.Conv2d(128, 128, 1)
            self.lat3 = nn.Conv2d( 64, 128, 1)
            self.out5 = nn.Conv2d(128, 128, 3, padding=1)
            self.out4 = nn.Conv2d(128, 128, 3, padding=1)
            self.out3 = nn.Conv2d(128, 128, 3, padding=1)

        def forward(self, x):
            c3 = torch.relu(self.stage3(self.stage2(self.stage1(x))))
            c4 = torch.relu(self.stage4(c3))
            c5 = torch.relu(self.stage5(c4))
            p5 = torch.relu(self.out5(self.lat5(c5)))
            p4 = torch.relu(self.out4(self.lat4(c4) + torch.nn.functional.interpolate(p5, scale_factor=2, mode="nearest")))
            p3 = torch.relu(self.out3(self.lat3(c3) + torch.nn.functional.interpolate(p4, scale_factor=2, mode="nearest")))
            return p3, p4, p5

    class YOLOHead(nn.Module):
        def __init__(self):
            super().__init__()
            def _head():
                return nn.Sequential(
                    nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(128, OUT_CH, 1),
                )
            self.head3 = _head()
            self.head4 = _head()
            self.head5 = _head()

        def forward(self, p3, p4, p5):
            return self.head3(p3), self.head4(p4), self.head5(p5)

    class YOLODetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = FPNBackbone()
            self.head     = YOLOHead()

        def forward(self, x):
            p3, p4, p5 = self.backbone(x)
            return self.head(p3, p4, p5)

    return YOLODetector()


# ------------------------------------------------------------------ #
# Weight transfer
# ------------------------------------------------------------------ #

def _mlx_conv_w(w_np):
    """MLX: (out, kH, kW, in) → PyTorch: (out, in, kH, kW)."""
    return np.transpose(w_np, (0, 3, 1, 2))


def load_weights_into_pytorch(pt_model, npz_path: str):
    import torch

    data = dict(np.load(npz_path))
    sd   = pt_model.state_dict()

    def _copy(pt_key: str, mlx_key: str, is_conv_weight: bool = False):
        if mlx_key not in data:
            print(f"  WARNING: {mlx_key} not found in checkpoint, skipping.")
            return
        w = data[mlx_key]
        if is_conv_weight:
            w = _mlx_conv_w(w)
        t = torch.from_numpy(w.copy())
        if pt_key in sd and sd[pt_key].shape == t.shape:
            sd[pt_key].copy_(t)
        else:
            exp = sd[pt_key].shape if pt_key in sd else "?"
            print(f"  SHAPE MISMATCH  {pt_key}: got {t.shape}, expected {exp}")

    # ── Backbone stages ──────────────────────────────────────────────
    for i, (pt_stage, mlx_stage) in enumerate([
        ("backbone.stage1.0", "backbone.stage1.layers.0"),
        ("backbone.stage2.0", "backbone.stage2.layers.0"),
        ("backbone.stage3.0", "backbone.stage3.layers.0"),
        ("backbone.stage4.0", "backbone.stage4.layers.0"),
        ("backbone.stage5.0", "backbone.stage5.layers.0"),
    ]):
        _copy(f"{pt_stage}.weight", f"{mlx_stage}.weight", is_conv_weight=True)
        _copy(f"{pt_stage}.bias",   f"{mlx_stage}.bias")

    # ── FPN laterals & outputs ────────────────────────────────────────
    for name in ("lat5", "lat4", "lat3", "out5", "out4", "out3"):
        _copy(f"backbone.{name}.weight", f"backbone.{name}.weight", is_conv_weight=True)
        _copy(f"backbone.{name}.bias",   f"backbone.{name}.bias")

    # ── Heads ─────────────────────────────────────────────────────────
    for hname in ("head3", "head4", "head5"):
        for (pt_idx, mlx_idx) in [(0, 0), (2, 2)]:
            _copy(f"head.{hname}.{pt_idx}.weight",
                  f"head.{hname}.layers.{mlx_idx}.weight",
                  is_conv_weight=True)
            _copy(f"head.{hname}.{pt_idx}.bias",
                  f"head.{hname}.layers.{mlx_idx}.bias")

    pt_model.load_state_dict(sd)
    print("Weights transferred successfully.")
    return pt_model


# ------------------------------------------------------------------ #
# Export
# ------------------------------------------------------------------ #

def export(weights_path: str, output_path: str, verify: bool = False):
    try:
        import torch
        import torch.onnx
    except ImportError:
        print("PyTorch not installed.  Run:  pip install torch")
        sys.exit(1)

    print(f"Loading weights from {weights_path} …")
    pt_model = build_pytorch_model()
    pt_model = load_weights_into_pytorch(pt_model, weights_path)
    pt_model.eval()

    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    print(f"Exporting to {output_path} …")
    torch.onnx.export(
        pt_model,
        dummy_input,
        output_path,
        opset_version   = 11,
        input_names     = ["images"],
        output_names    = ["pred_p3", "pred_p4", "pred_p5"],
        dynamic_axes    = {"images": {0: "batch_size"}},
    )
    print(f"Saved {output_path}  ({os.path.getsize(output_path) / 1e6:.1f} MB)")

    if verify:
        try:
            import onnxruntime as ort
            sess  = ort.InferenceSession(output_path)
            dummy = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
            outs  = sess.run(None, {"images": dummy})
            print("ONNX Runtime verification passed.")
            for name, o in zip(["pred_p3", "pred_p4", "pred_p5"], outs):
                print(f"  {name}: {o.shape}")
        except ImportError:
            print("onnxruntime not installed; skipping verification.")
            print("Install with:  pip install onnxruntime")


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Export YOLO to ONNX")
    parser.add_argument("--weights", default="checkpoints/best_model.npz")
    parser.add_argument("--output",  default="model.onnx")
    parser.add_argument("--verify",  action="store_true",
                        help="Run an onnxruntime inference check after export")
    args = parser.parse_args()

    if not os.path.isfile(args.weights):
        print(f"Weights not found: {args.weights}")
        print("Run training first:  python -m src.train")
        sys.exit(1)

    export(args.weights, args.output, verify=args.verify)


if __name__ == "__main__":
    main()
