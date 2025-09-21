# backend/server.py
import os
import io
import time
import json
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify, send_file
from PIL import Image
import numpy as np

# ultralytics YOLO
from ultralytics import YOLO

# --- Configuration ---
MODEL_NAME = os.getenv("MODEL_NAME", "yolov8n.pt")  # yolov8n is lightweight
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Flask
app = Flask(__name__)

# Load model once at startup (will download weights if not present)
print(f"Loading model '{MODEL_NAME}' (this may take a few seconds)...")
model = YOLO(MODEL_NAME)
print("Model loaded.")

# Helper: convert numpy image to Flask response
def pil_image_to_bytes(img: Image.Image, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf

# Helper: format detection results to JSON-friendly dict
def format_results(results, model):
    """
    results: ultralytics Results object or list of them
    returns: formatted dict
    """
    # model.names is a dict idx->label
    names = model.names if hasattr(model, "names") else {}

    formatted_objects = []
    # results is a list-like object; iterate through each result (we pass one image so only one)
    for r in results:
        # r.boxes is Boxes object with .xyxy, .conf, .cls
        boxes = r.boxes
        if boxes is None:
            continue
        xyxy = boxes.xyxy.cpu().numpy()  # shape (N,4)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else np.zeros((xyxy.shape[0],))
        cls = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") else np.zeros((xyxy.shape[0],), dtype=int)

        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = [float(x) for x in box]
            label = names.get(int(cls[i]), str(int(cls[i])))
            conf = float(confs[i])
            formatted_objects.append({
                "label": label,
                "confidence": round(conf, 4),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
            })

    return formatted_objects

@app.route("/")
def index():
    return (
        "<h3>AI Backend (Object Detection)</h3>"
        "<p>POST an image to <code>/detect</code> (field name <code>image</code>)</p>"
    )

@app.route("/detect", methods=["POST"])
def detect():
    """
    Expects form-data with key 'image' containing the image file (jpg/png).
    Returns JSON with detections and saves an output image with bounding boxes and a JSON file.
    """
    if "image" not in request.files:
        return jsonify({"error": "no image file (form field 'image')"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400

    # read image into memory
    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"failed to read image: {e}"}), 400

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")[:-3]
    in_fname = Path(file.filename).stem
    out_base = f"{in_fname}_{timestamp}"
    out_img_path = OUTPUT_DIR / f"{out_base}.png"
    out_json_path = OUTPUT_DIR / f"{out_base}.json"

    # Run inference (ultralytics YOLO)
    # We use model.predict() but the high-level call model(img) also works.
    try:
        # model returns a Results object (list-like). We pass image as numpy array
        np_img = np.array(img)
        results = model(np_img)  # run inference (uses device='cpu' by default if no GPU)
    except Exception as e:
        return jsonify({"error": f"inference failed: {e}"}), 500

    # Format JSON output
    objects = format_results(results, model)
    output = {
        "source_filename": file.filename,
        "processed_at": timestamp,
        "num_detections": len(objects),
        "objects": objects
    }

    # Save the JSON
    with open(out_json_path, "w") as f:
        json.dump(output, f, indent=2)

    # Create and save a plotted image with boxes (results[0].plot())
    try:
        # ultralytics Results has .plot() which returns an image array with boxes drawn
        plotted = results[0].plot()  # numpy array (H,W,3)
        plotted_pil = Image.fromarray(plotted)
        plotted_pil.save(out_img_path)
    except Exception as e:
        # fallback: just save original
        img.save(out_img_path)

    # Respond with JSON and link info
    resp = {
        "status": "ok",
        "result_json": str(out_json_path),
        "result_image": str(out_img_path),
        "data": output
    }
    return jsonify(resp), 200

# Helper endpoint to download last output image by filename
@app.route("/outputs/<path:fname>", methods=["GET"])
def download_output(fname):
    p = OUTPUT_DIR / fname
    if not p.exists():
        return jsonify({"error": "file not found"}), 404
    return send_file(str(p), as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting AI backend server on port {port} ...")
    app.run(host="0.0.0.0", port=port, debug=False)
