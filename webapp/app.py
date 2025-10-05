from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "data/images")
ANNOTATIONS_FILE = 'annotations/instances_default.json'

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/annotate/<int:image_id>')
def annotate(image_id):
    all_images= sorted(os.listdir(IMAGE_FOLDER))

    if image_id < 0 or image_id >= len(all_images):
        return "Invalid image ID",404

    image_file = all_images[image_id]
    image_path =f"/static_images/{image_file}"

    return render_template('annotate.html',
                            image_id=image_id,
                            image_path=image_path,
                            total_images=len(all_images)
                            )

def ensure_annotations_file():
    os.makedirs(os.path.dirname(ANNOTATIONS_FILE), exist_ok=True)
    if not os.path.exists(ANNOTATIONS_FILE) or os.path.getsize(ANNOTATIONS_FILE) == 0:
        initial = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "Laptop"},
                {"id": 2, "name": "Monitor"},
                {"id": 3, "name": "Keyboard"},
                {"id": 4, "name": "Mouse"},
                {"id": 5, "name": "Notebook"},
                {"id": 6, "name": "Neckband"},
                {"id": 7, "name": "Bottle"}
            ]
        }
        with open(ANNOTATIONS_FILE, 'w') as f:
            json.dump(initial, f, indent=4)

@app.route('/get_annotations/<int:image_id>')
def get_annotations(image_id):
    ensure_annotations_file()
    with open(ANNOTATIONS_FILE, 'r') as f:
        coco = json.load(f)
    # find annotations matching the image_id
    annots = [a for a in coco.get("annotations", []) if a.get("image_id") == image_id]
    return jsonify({"annotations": annots})


@app.route('/save_annotation', methods=['POST'])
def save_annotation():
    data = request.json
    image_id = data.get("image_id")
    boxes = data.get("boxes", [])
    file_name = data.get("file_name", None)
    width = int(data.get("width", 0))
    height = int(data.get("height", 0))

    ensure_annotations_file()

    # load coco
    with open(ANNOTATIONS_FILE, 'r') as f:
        try:
            coco = json.load(f)
        except Exception:
            coco = {"images": [], "annotations": [], "categories": []}

    # add image metadata if not present
    images = coco.get("images", [])
    existing_img_ids = [img["id"] for img in images]
    if image_id not in existing_img_ids:
        if file_name is None:
            # Attempt to infer filename from folder listing
            try:
                files = sorted(os.listdir(IMAGE_FOLDER))
                file_name = files[image_id]
            except Exception:
                file_name = f"image_{image_id}.jpg"
        images.append({
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })
        coco["images"] = images

    # ensure categories exist (we seeded 7 placeholders in ensure_annotations_file)
    categories = coco.get("categories", [])
    if not categories:
        # fallback: create 7 placeholders
        coco["categories"] = [{"id": i, "name": f"object_{i}"} for i in range(1, 8)]

    # append annotations
    annots = coco.get("annotations", [])
    next_id = (max([a["id"] for a in annots]) + 1) if annots else 1

    for b in boxes:
        # ensure numeric values
        x = float(b.get("x", 0))
        y = float(b.get("y", 0))
        w = float(b.get("w", 0))
        h = float(b.get("h", 0))
        category_id = int(b.get("category_id", 1))
        ann = {
            "id": next_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0
        }
        annots.append(ann)
        next_id += 1

    coco["annotations"] = annots

    # write back
    with open(ANNOTATIONS_FILE, 'w') as f:
        json.dump(coco, f, indent=4)

    return jsonify({"status": "success", "saved": len(boxes)})


@app.route('/static_images/<filename>')
def static_images(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

if __name__=='__main__':
    app.run(debug=True)