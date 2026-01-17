# src/dataset/voc_loader.py

import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np

class VOCDataset:
    def __init__(self, root, image_set="train"):
        self.root = root
        self.image_dir = os.path.join(root, "JPEGImages")
        self.ann_dir = os.path.join(root, "Annotations")

        split_file = os.path.join(root, "ImageSets", "Main", f"{image_set}.txt")
        with open(split_file) as f:
            self.ids = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.ids)

    def _load_annotation(self, image_id):
        ann_path = os.path.join(self.ann_dir, f"{image_id}.xml")
        tree = ET.parse(ann_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj.find("name").text)

        return np.array(boxes), labels

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img_path = os.path.join(self.image_dir, f"{image_id}.jpg")

        image = Image.open(img_path).convert("RGB")
        image = image.resize((224, 224))
        image = np.asarray(image, dtype=np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # CHW


        boxes, labels = self._load_annotation(image_id)

        return image, boxes, labels
