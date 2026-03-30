# yolo-mlx 🎯

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![MLX](https://img.shields.io/badge/Apple%20MLX-0.18%2B-black?logo=apple&logoColor=white)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M--series-silver?logo=apple&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-macOS-lightgrey?logo=apple)

---

## Description

YOLO-style object detection implemented and optimised for the **Apple MLX** framework on Apple Silicon. This project ports the full detection pipeline — Feature Pyramid Network backbone, multi-scale anchor assignment, 80-class COCO detection, and mAP@50 evaluation — entirely within MLX, enabling fast on-device inference without CUDA. By leveraging the unified memory architecture of M-series chips, the model runs the complete forward and backward pass on the same memory pool as the CPU, eliminating PCIe transfer overhead and enabling real-time detection on a MacBook.

---

## Key Features

- YOLO architecture (FPN backbone + multi-scale heads) adapted natively for Apple MLX
- On-device inference on Apple M-series chips — no GPU server required
- No CUDA — runs fully on unified memory via MLX's lazy evaluation engine
- Real-time detection pipeline with webcam support via `demo.py`
- Multi-anchor assignment (3 anchors × 3 scales = 9 anchors, k-means clustered on COCO)
- 80-class COCO detection with class-aware Non-Maximum Suppression
- mAP@50 evaluation (full VOC-style precision–recall curve)
- Cosine LR scheduling with linear warmup + gradient clipping for stable training
- ONNX export for cross-platform deployment

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Core ML framework | [Apple MLX](https://github.com/ml-explore/mlx) |
| Language | Python 3.10+ |
| Hardware target | Apple Silicon (M1/M2/M3/M4) |
| Video / webcam | OpenCV |
| Numerical ops | NumPy |
| Visualisation | Matplotlib, Pillow |

---

## How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/your-username/Object-Detection-MLX
cd Object-Detection-MLX

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run inference on an image
python detect.py --source image.jpg

# 4. Run on webcam
python demo.py

# 5. Train from scratch
python -m src.train
```

**Requirements:** macOS 13.3+ with an M-series chip. MLX is not supported on Intel Macs or non-Apple hardware.

---

## Architecture

```
Input (224×224×3)
       │
  FPN Backbone (5 stages, strides 2→32)
       │
  ┌────┴────────────────────┐
  C3 (28²)  C4 (14²)  C5 (7²)
  Top-down lateral merging  │
  P3 (28²)  P4 (14²)  P5 (7²)
       │         │        │
  Head×3 → (S, S, 3×85)   │
                           │
  85 = 4 (box) + 1 (obj) + 80 (cls)
```

---

## Results

| Metric | Value |
|--------|-------|
| Dataset | COCO val2017 |
| Epochs | 100 |
| Best mAP@50 | 0.0121 |
| Final loss | 41.7 (from 254.3) |
| Hardware | Apple Silicon (MLX) |
| NaN explosions | 0 |

Training curves and qualitative detection results are saved to `assets/`.

### Before vs After Training

| Before | After |
|--------|-------|
| ![Before](assets/pred_before.png) | ![After](assets/pred_after.png) |

---

## Project Structure

```
├── src/
│   ├── model/          # FPN backbone, detection heads, YOLO loss
│   ├── dataset/        # COCO JSON loader + augmentation
│   └── utils/          # Box decoding, NMS, mAP, visualisation
├── train.py            # Training script
├── demo.py             # Real-time webcam inference
├── export_onnx.py      # ONNX export
├── assets/             # Training curves, prediction images
└── checkpoints/        # Saved model weights (.npz)
```

---

## Author

**Shashank Krishnamurthy Vinodh**
MS Computer Science — Machine Learning Systems · Computer Vision · Apple Silicon
