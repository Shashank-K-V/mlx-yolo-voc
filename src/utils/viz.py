from PIL import Image, ImageDraw

def draw_boxes(image_path, boxes, scores, out_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for box, score in zip(boxes, scores):
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), f"{score:.2f}", fill="red")

    img.save(out_path)
