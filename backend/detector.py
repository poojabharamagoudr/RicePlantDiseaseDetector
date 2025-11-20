# # THIS FILE IS DISABLED — DO NOT USE
# # (Kept only for backup)

# """
# All code below is commented out intentionally.
# """

# # ------------------------------
# # Everything below is disabled
# # ------------------------------

# import tensorflow as tf
# import numpy as np
# import cv2
# import json
# import os

# # -------------------------------
# # Paths
# # -------------------------------
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# MODEL_PATH = os.path.join(BASE_DIR, "model", "rice_disease_mobilenetv2.h5")
# INFO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "disease_info.json")

# # -------------------------------
# # Load model and info
# # -------------------------------
# print("🔄 Loading model...")
# MODEL = tf.keras.models.load_model(MODEL_PATH)
# print("✅ Model loaded successfully!")

# with open(INFO_PATH, "r") as f:
#     DISEASE_INFO = json.load(f)
# print("✅ Disease info loaded!")

# # -------------------------------
# # Define class labels (update if needed)
# # -------------------------------
# CLASS_NAMES = [
#     "Brown Spot",
#     "Healthy Rice Leaf",
#     "Leaf Blast",
#     "Sheath Blight"
# ]

# # -------------------------------
# # Preprocess image
# # -------------------------------
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# def preprocess_image(image_path):
#     img = cv2.imread(image_path)
#     if img is None:
#         raise ValueError(f"❌ Could not read image: {image_path}")

#     img = cv2.resize(img, (224,224))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = preprocess_input(img)
#     img = np.expand_dims(img, axis=0)
#     return img

# # -------------------------------
# # Predict function
# # -------------------------------
# def predict_from_pil(pil_img):
#     # Convert PIL to model input
#     img = pil_img.convert("RGB").resize((224, 224))
#     img = np.array(img)
#     img = preprocess_input(img)
#     img = np.expand_dims(img, axis=0)

#     preds = MODEL.predict(img)
#     idx = np.argmax(preds)
#     confidence = float(np.max(preds))
#     disease = CLASS_NAMES[idx]

#     info = DISEASE_INFO.get(disease, {})
#     treatment = info.get("treatment", "No treatment info available")
#     schemes = info.get("schemes", [])

#     return {
#         "label": disease,
#         "confidence": round(confidence, 3),
#         "treatment": treatment,
#         "schemes": schemes
#     }

#     return result

# # -------------------------------
# # Test run (run this file directly)
# # -------------------------------
# if __name__ == "__main__":
#     test_image = r"C:\Users\pooja\Desktop\Riceplantdiseasedetector\data\test\Healthy Rice Leaf\test_leaf.jpg"

#     if not os.path.exists(test_image):
#         print(f"⚠️ Image not found at: {test_image}")
#     else:
#         try:
#             output = predict_disease(test_image)
#             print(json.dumps(output, indent=2))
#         except Exception as e:
#             print("❌ Error during prediction:", e)
