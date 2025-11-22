# webapp/app.py
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json

# --- existing project paths (kept relative per your preference) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "data/images")
ANNOTATIONS_FILE = 'annotations/instances_default.json'  # relative as requested

app = Flask(__name__)

# --- ML imports & model paths (added) ---
# Make sure src/models/inference.py exists and exposes the referenced functions
from src.models.inference import (
    train_detector,
    train_prototypical,
    run_detector_inference_and_classify
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DETECTOR_PATH = os.path.join(MODELS_DIR, "detector.pth")
PROTO_PATH = os.path.join(MODELS_DIR, "prototypes.pt")

# -------------------------
# Existing UI routes (unchanged)
# -------------------------
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

    with open(ANNOTATIONS_FILE, 'r') as f:
        coco = json.load(f)
    categories = {c["id"]: c["name"] for c in coco.get("categories", [])}

    return render_template('annotate.html',
                            image_id=image_id,
                            image_path=image_path,
                            categories=categories,
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

@app.route('/export_coco')
def export_coco():
    """Download the current COCO annotation file."""
    ensure_annotations_file()
    return send_from_directory(
        os.path.dirname(ANNOTATIONS_FILE),
        os.path.basename(ANNOTATIONS_FILE),
        as_attachment=True
    )

# -------------------------
# ML endpoints (inserted; do not modify earlier routes)
# -------------------------

@app.route('/train_detector', methods=['POST'])
def api_train_detector():
    """
    Train the Faster R-CNN detector synchronously.
    Request JSON (optional): { "epochs": int, "batch_size": int, "lr": float }
    """
    body = request.get_json() or {}
    epochs = int(body.get('epochs', 40))
    batch_size = int(body.get('batch_size', 2))
    lr = float(body.get('lr', 1e-4))

    
    os.makedirs(MODELS_DIR, exist_ok=True)

    
    try:
        with open(ANNOTATIONS_FILE, 'r') as f:
            coco = json.load(f)

        valid_annots = []
        for a in coco.get("annotations", []):
            x, y, w, h = a["bbox"]
            if w > 1 and h > 1:  # Ignore zero/negative boxes
                valid_annots.append(a)
            else:
                print(f" Skipped invalid box: {a}")

        coco["annotations"] = valid_annots

        
        with open(ANNOTATIONS_FILE, 'w') as f:
            json.dump(coco, f, indent=2)

    except Exception as e:
        return jsonify({"status": "error", "error": f"Failed to load/clean annotations: {str(e)}"}), 500

    
    try:
        out_path = train_detector(
            coco_json=ANNOTATIONS_FILE,
            img_root=os.path.join(PROJECT_ROOT, "data/images"),
            out_path=DETECTOR_PATH,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr
        )
        return jsonify({"status": "ok", "detector_path": out_path})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/train_classifier', methods=['POST'])
def api_train_classifier():
    """
    Train the prototypical classifier synchronously.
    Request JSON (optional): { "n_way":int, "k_shot":int, "q_query":int, "episodes_per_epoch":int, "epochs":int, "lr":float }
    """
    body = request.get_json() or {}
    n_way = int(body.get('n_way', 5))
    k_shot = int(body.get('k_shot', 5))
    q_query = int(body.get('q_query', 5))
    episodes_per_epoch = int(body.get('episodes_per_epoch', 100))
    epochs = int(body.get('epochs', 1))
    lr = float(body.get('lr', 1e-4))

    os.makedirs(MODELS_DIR, exist_ok=True)

    try:
        out_path = train_prototypical(
            coco_json=ANNOTATIONS_FILE,
            img_root=os.path.join(PROJECT_ROOT, "data/images"),
            out_path=PROTO_PATH,
            n_way=n_way, k_shot=k_shot, q_query=q_query,
            episodes_per_epoch=episodes_per_epoch, epochs=epochs, lr=lr
        )
        return jsonify({"status": "ok", "proto_path": out_path})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/auto_annotate/<int:image_id>', methods=['GET'])
def api_auto_annotate(image_id):
    """
    Run detector + prototypical classifier on the given image_id and return JSON detections.
    This endpoint DOES NOT append detections to your annotations file automatically.
    Frontend should call this and overlay returned boxes; user will Save manually if desired.
    """
    ensure_annotations_file()

    # load image filename from annotations if present, otherwise infer from folder
    with open(ANNOTATIONS_FILE, 'r') as f:
        coco = json.load(f)
    images_meta = {i['id']: i for i in coco.get('images', [])}
    all_files = sorted(os.listdir(IMAGE_FOLDER))

    if image_id in images_meta:
        file_name = images_meta[image_id]['file_name']
    else:
        if image_id < 0 or image_id >= len(all_files):
            return jsonify({"status": "error", "error": "Invalid image id"}), 404
        file_name = all_files[image_id]

    image_path = os.path.join(IMAGE_FOLDER, file_name)

    # Ensure models exist
    if not os.path.exists(DETECTOR_PATH) or not os.path.exists(PROTO_PATH):
        return jsonify({"status": "error", "error": "Detector or prototypes not found. Train or place models in models/ first."}), 400

    try:
        detections = run_detector_inference_and_classify(
            image_path=image_path,
            detector_ckpt=DETECTOR_PATH,
            proto_ckpt=PROTO_PATH,
            coco_json=ANNOTATIONS_FILE,
            img_root=os.path.join(PROJECT_ROOT, "data/images")
        )
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]  # assuming this format
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)

        det["bbox"] = [x, y, w, h]
    # Return detections to the frontend for overlay; each detection contains:
    # { "bbox":[x,y,w,h], "score":float, "category_id":int, "category_name":str, "pred_score":float }
    return jsonify({"status": "ok", "detections": detections})


if __name__=='__main__':
    app.run(debug=True)
