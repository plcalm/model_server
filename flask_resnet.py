"""
Flask + ResNet-50 推理服务
启动: gunicorn -w 4 -k gevent -b 0.0.0.0:5000 flask_resnet:app
"""

import base64
import io
import sys

import torch
from flask import Flask, jsonify, request
from PIL import Image

from resnet_utils import _load_labels, load_model, preprocess, top5_predictions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading ResNet-50 on {device}...", file=sys.stderr)
MODEL = load_model().to(device)
LABELS = _load_labels()
print(f"Loaded {len(LABELS)} labels.", file=sys.stderr)

app = Flask(__name__)


def _run_inference(image: Image.Image) -> list:
    input_data = preprocess(image)
    input_tensor = torch.from_numpy(input_data).to(device)
    with torch.no_grad():
        logits = MODEL(input_tensor).cpu().numpy()
    return top5_predictions(logits, LABELS)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file in request"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    try:
        image = Image.open(io.BytesIO(file.read()))
    except Exception as exc:
        return jsonify({"error": f"Invalid image: {exc}"}), 400
    return jsonify(_run_inference(image))


@app.route("/predict_base64", methods=["POST"])
def predict_base64():
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "Missing 'image' field in JSON body"}), 400
    try:
        img_bytes = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(img_bytes))
    except Exception as exc:
        return jsonify({"error": f"Invalid base64 image: {exc}"}), 400
    return jsonify(_run_inference(image))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
