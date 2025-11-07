import tensorflow as tf
import numpy as np
import cv2
import os

# Load trained model
model_path = 'model/rice_disease_mobilenetv2.h5'
model = tf.keras.models.load_model(model_path)

# Define classes (same as training)
classes = ['Healthy Rice Leaf', 'Leaf Blast', 'Brown Spot', 'Sheath Blight']

# Path to test folder
test_dir = 'data/test'

# Function to predict a single image
def predict_leaf(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Image not found: {img_path}")
        return
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    print(f"{os.path.basename(img_path)} --> {predicted_class} ({confidence:.2f}% confidence)")

# Loop through all subfolders and images in test folder
for folder in os.listdir(test_dir):
    folder_path = os.path.join(test_dir, folder)
    if os.path.isdir(folder_path):
        for f in os.listdir(folder_path):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder_path, f)
                predict_leaf(img_path)
