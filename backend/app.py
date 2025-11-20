# backend/app.py

import os
import json
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import io


app = Flask(__name__)
CORS(app)

# -------------------------------
# Paths
# -------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT, "model", "rice_disease_mobilenetv2.h5")
INFO_PATH = os.path.join(os.path.dirname(__file__), "disease_info.json")

# -------------------------------
# Load model + disease info
# -------------------------------
print("🔄 Loading TensorFlow model...")
MODEL = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded!")

with open(INFO_PATH, "r") as f:
    DISEASE_INFO = json.load(f)
print("✅ disease_info.json loaded!")

# -------------------------------
# Class names (correct order)
# -------------------------------
CLASS_NAMES = [
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Sheath Blight"
]

# -------------------------------
# Preprocess PIL image
# -------------------------------
def preprocess_pil_image(pil_img):
    img = pil_img.convert("RGB").resize((224, 224))
    img = np.array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img


# -------------------------------
# Simple heuristic: is this image a leaf?
# We check for presence of green pixels (G > 1.2*R and G > 1.2*B and G > 40)
# If the proportion of such pixels is above a small threshold we consider it a leaf.
# This keeps the backend independent of OpenCV or additional deps.
# -------------------------------
def is_leaf(pil_img, resize_to=(224, 224), green_prop_threshold=0.05):
    try:
        img = pil_img.convert("RGB").resize(resize_to)
        arr = np.array(img)
        # arr shape (H, W, 3)
        r = arr[..., 0].astype(np.int32)
        g = arr[..., 1].astype(np.int32)
        b = arr[..., 2].astype(np.int32)

        green_mask = (g > (r * 1.2)) & (g > (b * 1.2)) & (g > 40)
        prop_green = float(np.sum(green_mask)) / (arr.shape[0] * arr.shape[1])

        return prop_green >= green_prop_threshold
    except Exception:
        # If any error occurs while checking, be conservative and return True
        # to allow the prediction path to run (model may still handle it).
        return True

# -------------------------------
# Predict from PIL image
# -------------------------------
def predict_from_pil(pil_img):
    arr = preprocess_pil_image(pil_img)
    preds = MODEL.predict(arr)

    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    disease = CLASS_NAMES[idx]

    info = DISEASE_INFO.get(disease, {})
    treatment = info.get("treatment", "No treatment info available")
    govt_schemes = info.get("govt_schemes", [])

    return {
        "label": disease,
        "confidence": round(confidence, 3),
        "treatment": treatment,
        "govt_schemes": govt_schemes
    }

# -------------------------------
# Routes
# -------------------------------
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file with key 'image' provided"}), 400

    file = request.files["image"]

    try:
        # Read image into PIL
        img = Image.open(file.stream)

        # Quick check: is it a leaf image?
        if not is_leaf(img):
            return jsonify({
                "label": "Unknown Image",
                "confidence": 0.0,
                "treatment": "",
                "govt_schemes": [],
                "message": "Uploaded image does not appear to be a rice leaf. Please upload a clear leaf image."
            })

        # Otherwise run model prediction
        result = predict_from_pil(img)
        return jsonify(result)

    except Exception as e:
        print("❌ Prediction error:", e)
        print(traceback.format_exc())
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
