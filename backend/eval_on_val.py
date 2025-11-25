import os
from PIL import Image

import importlib.util

# Load backend.app by file path so this script works when run directly
APP_PATH = os.path.join(os.path.dirname(__file__), 'app.py')
spec = importlib.util.spec_from_file_location('backend_app', APP_PATH)
backend_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend_app)


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
VAL_DIR = os.path.join(ROOT, 'data', 'val')

def iter_images(base_dir):
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                yield os.path.join(root, f)


def expected_label_from_path(path, base_dir):
    # expected label is the immediate directory under base_dir
    rel = os.path.relpath(path, base_dir)
    parts = rel.split(os.sep)
    if len(parts) >= 2:
        return parts[0]
    return parts[0] if parts else ''


def main():
    if not os.path.isdir(VAL_DIR):
        print('No validation directory found at', VAL_DIR)
        return
import os
import numpy as np
from PIL import Image
import importlib.util

# Load backend.app by file path so this script works when run directly
APP_PATH = os.path.join(os.path.dirname(__file__), 'app.py')
spec = importlib.util.spec_from_file_location('backend_app', APP_PATH)
backend_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend_app)


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
VAL_DIR = os.path.join(ROOT, 'data', 'val')


def iter_images(base_dir):
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                yield os.path.join(root, f)


def expected_label_from_path(path, base_dir):
    rel = os.path.relpath(path, base_dir)
    parts = rel.split(os.sep)
    return parts[0] if parts else ''


def main():
    if not os.path.isdir(VAL_DIR):
        print('No validation directory found at', VAL_DIR)
        return

    # Collect images and expected labels
    paths = list(iter_images(VAL_DIR))
    if not paths:
        print('No images found under', VAL_DIR)
        return

    imgs = []
    expects = []
    for p in paths:
        try:
            pil = Image.open(p)
            arr = backend_app.preprocess_pil_image(pil)  # shape (1, H, W, C)
            imgs.append(arr)
            expects.append(expected_label_from_path(p, VAL_DIR))
        except Exception as e:
            print('Error loading', p, e)

    if not imgs:
        print('No valid images to evaluate')
        return

    # Stack into a single batch
    X = np.vstack(imgs)

    # Run batched prediction quietly
    preds = backend_app.MODEL.predict(X, verbose=0)
    pred_idxs = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)
    labels = [backend_app.CLASS_NAMES[i] for i in pred_idxs]

    # Tally results
    total = len(labels)
    correct = 0
    conf_table = {}
    misclassified = []

    for path, exp, pred, conf in zip(paths, expects, labels, confidences):
        conf_table.setdefault(exp, {})
        conf_table[exp].setdefault(pred, 0)
        conf_table[exp][pred] += 1
        if pred == exp:
            correct += 1
        else:
            if len(misclassified) < 20:
                misclassified.append((path, exp, pred, float(conf)))

    print('\nValidation results:')
    print('  Total images:', total)
    print('  Correct:', correct)
    print('  Accuracy: {:.2%}'.format((correct / total) if total else 0))

    print('\nConfusion (actual -> predicted counts):')
    for actual, preds in conf_table.items():
        print('  ', actual)
        for p_label, cnt in preds.items():
            print('     -> {:20s}: {}'.format(p_label, cnt))

    if misclassified:
        print('\nSample misclassified images:')
        for p in misclassified:
            print(' ', p)


if __name__ == '__main__':
    main()
