import json
import os
from tqdm import tqdm

def convert_one(json_path, images_dir, labels_dir):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Create image lookup
    img_lookup = {img["id"]: img for img in data["images"]}

    # Group annotations by image
    ann_lookup = {}
    for ann in data["annotations"]:
        if ann["image_id"] not in ann_lookup:
            ann_lookup[ann["image_id"]] = []
        ann_lookup[ann["image_id"]].append(ann)

    os.makedirs(labels_dir, exist_ok=True)

    for img_id, anns in tqdm(ann_lookup.items(), desc=f"Converting {json_path}"):
        img = img_lookup[img_id]
        w, h = img["width"], img["height"]

        label_path = os.path.join(labels_dir, img["file_name"].replace(".jpg", ".txt"))
        
        with open(label_path, "w") as out:
            for ann in anns:
                cls = ann["category_id"] - 1   # YOLO classes 0-indexed

                seg = ann["segmentation"][0]   # list of [x1, y1, x2, y2, ...]
                normalized = []
                for i in range(0, len(seg), 2):
                    x = seg[i] / w
                    y = seg[i+1] / h
                    normalized.append(x)
                    normalized.append(y)

                out.write(str(cls) + " " + " ".join(f"{v:.6f}" for v in normalized) + "\n")


def main():
    convert_one("train/_annotations.coco.json", "train", "train/labels")
    convert_one("valid/_annotations.coco.json", "valid", "valid/labels")
    convert_one("test/_annotations.coco.json", "test", "test/labels")

if __name__ == "__main__":
    main()
