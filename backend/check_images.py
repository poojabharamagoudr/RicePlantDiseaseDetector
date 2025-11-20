import os

test_dir = 'data/test'
all_images = []

for folder in os.listdir(test_dir):
    folder_path = os.path.join(test_dir, folder)
    if os.path.isdir(folder_path):
        for f in os.listdir(folder_path):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(folder_path, f))

print("Found images:", all_images)
