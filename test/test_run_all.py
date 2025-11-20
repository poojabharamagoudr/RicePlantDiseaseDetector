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

def fix_filename(path):
    # Fix filenames ending with .jpg.jpg or similar
    dirname, filename = os.path.split(path)
    if filename.lower().endswith(".jpg.jpg"):
        new_name = filename.replace(".jpg.jpg", ".jpg")
        new_path = os.path.join(dirname, new_name)
        os.rename(path, new_path)
        print(f"Renamed: {path} --> {new_path}")
        return new_path
    return path

def predict_leaf(img_path):
    img_path = fix_filename(img_path)
    img = cv2.imread(img_path)
    if img is None:
        print(f"ERROR: Cannot load image {img_path}")
        return None, None
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img, verbose=0)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return predicted_class, confidence

if __name__ == "__main__":
    test_dir = 'data/test'
    results = []

    # Walk through all subfolders in test_dir
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                predicted_class, confidence = predict_leaf(img_path)
                if predicted_class:
                    print(f"{file} --> {predicted_class} ({confidence:.2f}%)")
                    results.append([file, predicted_class, f"{confidence:.2f}%"])

    # Save predictions to CSV
    if results:
        csv_path = os.path.join(test_dir, 'test_predictions.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Filename', 'Predicted Class', 'Confidence'])
            writer.writerows(results)
        print(f"\nAll predictions saved to {csv_path}")
    else:
        print("No images found to predict.")
