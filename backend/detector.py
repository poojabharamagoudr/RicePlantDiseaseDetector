import tensorflow as tf
import numpy as np
import cv2
import json
import os

# -------------------------------
# Paths
# -------------------------------
MODEL_PATH = os.path.join("model", "rice_disease_mobilenetv2.h5")
INFO_PATH = os.path.join("disease_info.json")

# -------------------------------
# Load model and info
# -------------------------------
print("🔄 Loading model...")
MODEL = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

with open(INFO_PATH, "r") as f:
    DISEASE_INFO = json.load(f)
print("✅ Disease info loaded!")

# -------------------------------
# Define class labels (update if needed)
# -------------------------------
CLASS_NAMES = [
    "BrownSpot",
    "LeafBlast",
    "BacterialBlight",
    "Tungro",
    "Healthy"
]

# -------------------------------
# Preprocess image
# -------------------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"❌ Could not read image: {image_path}")
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -------------------------------
# Predict function
# -------------------------------
def predict_disease(image_path):
    img = preprocess_image(image_path)
    preds = MODEL.predict(img)
    idx = np.argmax(preds)
    confidence = float(np.max(preds))
    disease = CLASS_NAMES[idx]

    info = DISEASE_INFO.get(disease, {})
    treatment = info.get("treatment", "No treatment info available")
    schemes = info.get("schemes", [])

    result = {
        "prediction": disease,
        "confidence": round(confidence, 3),
        "treatment": treatment,
        "schemes": schemes
    }

    return result

# -------------------------------
# Test run (run this file directly)
# -------------------------------
if __name__ == "__main__":
    test_image = r"C:\Users\pooja\Desktop\Riceplantdiseasedetector\data\test\Healthy Rice Leaf\test_leaf.jpg"

    if not os.path.exists(test_image):
        print(f"⚠️ Image not found at: {test_image}")
    else:
        try:
            output = predict_disease(test_image)
            print(json.dumps(output, indent=2))
        except Exception as e:
            print("❌ Error during prediction:", e)
