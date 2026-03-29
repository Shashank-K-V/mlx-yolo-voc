# COCO anchor boxes (w, h) in pixels for 224×224 input.
# Derived from YOLOv3 COCO anchors (originally for 416 input, scaled × 224/416 ≈ 0.538).
#
# Three scales, three anchors each:
#   P3 → 28×28 grid, stride=8  → small objects
#   P4 → 14×14 grid, stride=16 → medium objects
#   P5 →  7×7 grid,  stride=32 → large objects

NUM_CLASSES = 80
NUM_ANCHORS = 3   # anchors per scale

ANCHORS = [
    # P3 – 28×28 (small objects)
    [(5,  7),  (9,  16), (18, 12)],
    # P4 – 14×14 (medium objects)
    [(16, 33), (33, 24), (32, 64)],
    # P5 – 7×7  (large objects)
    [(62, 48), (84, 107), (201, 176)],
]

STRIDES = [8, 16, 32]   # pixel stride for P3, P4, P5

# ------------------------------------------------------------------ #
# COCO 80 class names (index 0–79)
# ------------------------------------------------------------------ #
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# COCO category_id → class index (0–79).
# The 80 categories have non-contiguous IDs (12, 26, 29, 30 … are absent).
_COCO_IDS = [
    1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
    54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]
COCO_ID_TO_CLASS = {cid: idx for idx, cid in enumerate(_COCO_IDS)}
