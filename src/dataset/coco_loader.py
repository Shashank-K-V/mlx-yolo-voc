"""
COCO dataset loader — all 80 categories, optional augmentation.

Augmentations applied at full 224×224 resolution (after resize):
    • Random horizontal flip  (p=0.5) with box-coordinate mirroring
    • Brightness jitter       (p=0.5, factor 0.6–1.4)
    • Contrast  jitter        (p=0.5, factor 0.6–1.4)
    • Colour-saturation jitter(p=0.5, factor 0.6–1.4)
"""

import json
import os
import random

import numpy as np
from PIL import Image, ImageEnhance

from src.model.anchors import COCO_ID_TO_CLASS


class COCODataset:
    def __init__(self, root, ann_file, img_size=224, augment=False):
        with open(ann_file) as f:
            coco = json.load(f)

        self.root     = root
        self.img_size = img_size
        self.augment  = augment

        self.images      = {img["id"]: img for img in coco["images"]}
        self.annotations = {}

        for ann in coco["annotations"]:
            if ann["category_id"] not in COCO_ID_TO_CLASS:
                continue
            self.annotations.setdefault(ann["image_id"], []).append(ann)

        # Keep only images that have at least one valid annotation
        self.image_ids = list(self.annotations.keys())

    # ---------------------------------------------------------------- #

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]

        img_path = os.path.join(self.root, img_info["file_name"])
        image    = Image.open(img_path).convert("RGB")

        orig_w, orig_h = image.size
        image          = image.resize((self.img_size, self.img_size))

        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h

        boxes, labels = [], []
        for ann in self.annotations.get(image_id, []):
            cls_idx = COCO_ID_TO_CLASS.get(ann["category_id"], -1)
            if cls_idx < 0:
                continue

            x, y, w, h = ann["bbox"]
            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + w) * scale_x
            y2 = (y + h) * scale_y

            if x2 - x1 < 1 or y2 - y1 < 1:   # skip degenerate boxes
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(cls_idx)

        if self.augment and len(boxes) > 0:
            image, boxes = self._augment(image, boxes)

        img_np = np.asarray(image, dtype=np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))   # CHW

        return (
            img_np,
            np.array(boxes,  dtype=np.float32),
            np.array(labels, dtype=np.int32),
        )

    # ---------------------------------------------------------------- #
    # Augmentation
    # ---------------------------------------------------------------- #

    def _augment(self, image, boxes):
        W = self.img_size

        # Horizontal flip
        if random.random() < 0.5:
            image  = image.transpose(Image.FLIP_LEFT_RIGHT)
            boxes  = [[W - b[2], b[1], W - b[0], b[3]] for b in boxes]

        # Brightness
        if random.random() < 0.5:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.6, 1.4))

        # Contrast
        if random.random() < 0.5:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.6, 1.4))

        # Colour saturation
        if random.random() < 0.5:
            image = ImageEnhance.Color(image).enhance(random.uniform(0.6, 1.4))

        return image, boxes
