import json
import os
import numpy as np
from PIL import Image

class COCODataset:
    def __init__(self, root, ann_file, img_size=224):
        with open(ann_file, "r") as f:
            coco = json.load(f)

        self.root = root
        self.img_size = img_size

        self.images = {img["id"]: img for img in coco["images"]}
        self.annotations = {}
        for ann in coco["annotations"]:
            if ann["category_id"] != 1:  # PERSON ONLY
                continue
            self.annotations.setdefault(ann["image_id"], []).append(ann)

        self.image_ids = list(self.images.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]

        img_path = os.path.join(self.root, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        orig_w, orig_h = image.size
        image = image.resize((self.img_size, self.img_size))

        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h

        image = np.asarray(image, dtype=np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # CHW

        boxes = []
        labels = []

        for ann in self.annotations.get(image_id, []):
            x, y, w, h = ann["bbox"]
            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + w) * scale_x
            y2 = (y + h) * scale_y

            boxes.append([x1, y1, x2, y2])
            labels.append(0)  # person = class 0

        return image, np.array(boxes, dtype=np.float32), labels
