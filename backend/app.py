# backend/app.py
import os
import io
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT, "model", "rice_disease_mobilenetv2.h5")

# Try to import your existing detector.py (non-invasive).
detector = None
use_detector_module = False
try:
    import sys
    sys.path.insert(0, ROOT)  # allow importing detector.py from project root
    import backend.detector as detector_module  # assume file is named detector.py
    detector = detector_module
    use_detector_module = True
    app.logger.info("Using existing detector.py for predictions.")
except Exception as e:
    app.logger.warning("Could not import detector.py. Will attempt to load model directly. Error: %s", e)
    detector = None
    use_detector_module = False

# If detector.py doesn't expose a function we can use, load model fallback
tf = None
keras_model = None
if not use_detector_module:
    try:
        import tensorflow as tf
        from tensorflow import keras
        keras_model = keras.models.load_model(MODEL_PATH)
        app.logger.info("Loaded model from %s", MODEL_PATH)
    except Exception as e:
        app.logger.error("Failed to load model fallback: %s", e)
        keras_model = None


# helper preprocessing - match typical MobileNetV2 input size (224x224)
def preprocess_pil_image(pil_img, target_size=(224, 224)):
    img = pil_img.convert("RGB").resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# Generic decode — if your detector.py handles labels itself, that will be used.
DEFAULT_LABELS = ['Brown Spot', 'BLB', 'Blast', 'Healthy']


def decode_preds(preds, labels=DEFAULT_LABELS):
    # preds expected shape (1, n)
    try:
        probs = preds[0].tolist()
        idx = int(np.argmax(probs))
        return {"label": labels[idx], "confidence": float(probs[idx])}
    except Exception:
        return {"label": "Unknown", "confidence": 0.0}


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects multipart/form-data with key 'image' (file).
    Returns JSON: {label, confidence, treatment, schemes}
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided under key 'image'."}), 400

    file = request.files["image"]
    try:
        # read PIL image
        img = Image.open(file.stream)

        # If we have detector.py and it exposes a convenient function, use it.
        if use_detector_module:
            # Try multiple common function names — adapt if your detector uses different names.
            possible_fns = [
                "predict_from_pil",    # predict_from_pil(PIL.Image) -> dict or (label, conf)
                "predict_image_from_pil",
                "predict_image",       # maybe expects path or PIL
                "predict",             # generic
                "classify"             # generic
            ]
            for fn_name in possible_fns:
                fn = getattr(detector, fn_name, None)
                if callable(fn):
                    try:
                        # attempt to call with PIL image
                        res = fn(img)
                        # if returns tuple/list, try interpret
                        if isinstance(res, dict):
                            result = res
                        elif isinstance(res, (list, tuple)) and len(res) >= 2:
                            result = {"label": res[0], "confidence": float(res[1])}
                        else:
                            # if returned a numpy array of preds
                            try:
                                arr = np.array(res)
                                result = decode_preds(arr)
                            except Exception:
                                result = {"label": str(res), "confidence": 0.0}
                        break
                    except Exception as inner:
                        # maybe the function expected a file path — try to save temporary and call again
                        try:
                            tmp_path = os.path.join("/tmp", "tmp_predict_image.jpg")
                            img.save(tmp_path)
                            res2 = fn(tmp_path)
                            if isinstance(res2, dict):
                                result = res2
                            elif isinstance(res2, (list, tuple)) and len(res2) >= 2:
                                result = {"label": res2[0], "confidence": float(res2[1])}
                            else:
                                try:
                                    arr = np.array(res2)
                                    result = decode_preds(arr)
                                except Exception:
                                    result = {"label": str(res2), "confidence": 0.0}
                            break
                        except Exception:
                            # try next function
                            continue
            else:
                # no suitable function found — fall back to model if loaded
                if keras_model is not None:
                    arr = preprocess_pil_image(img)
                    preds = keras_model.predict(arr)
                    result = decode_preds(preds)
                else:
                    return jsonify({"error": "No compatible prediction function found in detector.py and no fallback model loaded."}), 500
        else:
            # No detector module — use fallback keras model
            if keras_model is None:
                return jsonify({"error": "No model available on server."}), 500
            arr = preprocess_pil_image(img)
            preds = keras_model.predict(arr)
            result = decode_preds(preds)

        # Add example treatment & scheme suggestions (edit these in utils or detector later)
        treatments = {
            'Brown Spot': 'Use fungicide X, remove affected leaves, rotate crop.',
            'BLB': 'Use certified seeds, apply appropriate bactericide, ensure good drainage.',
            'Blast': 'Reduce nitrogen, use blast-resistant varieties, apply fungicide Y.',
            'Healthy': 'No disease detected. Keep monitoring.'
        }
        schemes = {
            'Brown Spot': ['Scheme A: Subsidy for fungicides', 'Scheme B: Crop insurance info'],
            'BLB': ['Scheme C: Government bactericide support'],
            'Blast': ['Scheme D: Resistant seed subsidies'],
            'Healthy': ['Scheme E: Preventive advisory program']
        }

        label = result.get("label") if isinstance(result, dict) else str(result)
        confidence = result.get("confidence", 0.0) if isinstance(result, dict) else 0.0

        response = {
            "label": label,
            "confidence": float(confidence),
            "treatment": treatments.get(label, ""),
            "schemes": schemes.get(label, [])
        }
        return jsonify(response)

    except Exception as exc:
        app.logger.error("Prediction error: %s\n%s", exc, traceback.format_exc())
        return jsonify({"error": "Prediction failed: " + str(exc)}), 500


if __name__ == "__main__":
    # for local testing only
    app.run(host="0.0.0.0", port=5000, debug=True)
