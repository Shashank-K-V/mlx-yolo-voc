# MLX-YOLO-VOC

First public implementation of a YOLO-style object detection pipeline using Apple's MLX framework, with benchmarks against PyTorch on Pascal VOC.

## Motivation
MLX is a new ML framework optimized for Apple Silicon, but its applicability to real-world computer vision tasks such as object detection remains largely unexplored. This project investigates whether MLX can support efficient training and inference for YOLO-style detectors.

## Scope (v0.1)
- YOLO-style single-stage object detection
- Pascal VOC dataset
- Training and inference on Apple Silicon
- Performance and memory benchmarking vs PyTorch

## Planned Extensions
- COCO dataset benchmarks
- Anchor-free detectors
- Deployment-oriented optimizations

## Project Structure
mlx-yolo-voc/
├─ src/
├─ docs/
├─ experiments/
├─ notebooks/
├─ assets/


## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

MLX installation instructions will be added once the core pipeline is implemented.



This already signals:
- research intent
- novelty
- seriousness

---

## STEP 5: Create repo LOCALLY (recommended)

Do **not** create code directly on GitHub.

### Local setup
```bash
mkdir mlx-yolo-voc
cd mlx-yolo-voc
git init
