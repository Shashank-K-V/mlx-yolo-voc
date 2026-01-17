# src/utils/viz.py

from PIL import Image, ImageDraw


def draw_boxes(image_path, boxes, scores=None, out_path="assets/pred.png"):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for k, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], width=2)

        if scores is not None:
            draw.text((x1, y1), f"{scores[k]:.2f}")

    img.save(out_path)
    return out_path
