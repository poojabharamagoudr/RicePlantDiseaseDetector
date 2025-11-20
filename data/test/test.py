import tensorflow as tf
import numpy as np
import cv2
import os
import csv

# Load trained model
model_path = 'model/rice_disease_mobilenetv2.h5'
model = tf.keras.models.load_model(model_path)

# Classes (must match training)
classes = ['Healthy Rice Leaf', 'Leaf Blast', 'Brown Spot', 'Sheath Blight']

# Folder containing test images
test_dir = 'data/test'

# Prepare CSV to save results
csv_file = 'test_predictions.csv'
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Image', 'Predicted Class', 'Confidence'])

    # Loop through images
    for folder in os.listdir(test_dir):
        folder_path = os.path.join(test_dir, folder)
        if os.path.isdir(folder_path):
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(folder_path, img_name)
                    # Load and preprocess image
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Could not load image: {img_path}")
                        continue
                    img = cv2.resize(img, (224, 224))
                    img = img / 255.0
                    img = np.expand_dims(img, axis=0)

                    # Predict
                    prediction = model.predict(img, verbose=0)
                    pred_class = classes[np.argmax(prediction)]
                    confidence = np.max(prediction) * 100

                    print(f"{img_name} --> {pred_class} ({confidence:.2f}% confidence)")
                    writer.writerow([img_name, pred_class, f"{confidence:.2f}%"])

print(f"\nAll predictions saved to {csv_file}")
