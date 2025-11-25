#!/usr/bin/env python3
"""Select one sample image per class and POST to the backend /predict endpoint.

This script does NOT import the model; it only discovers files and sends them.
Usage:
  python backend/run_samples.py
It respects the environment variable `API_BASE_URL` (default http://127.0.0.1:5000).
"""
import os
import sys
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIRS = [
    os.path.join(ROOT, 'data', 'test'),
    os.path.join(ROOT, 'data', 'val'),
    os.path.join(ROOT, 'data', 'train'),
]

API_BASE = os.environ.get('API_BASE_URL', 'http://127.0.0.1:5000')

# Expected class folder names (match your training/labels)
CLASS_NAMES = [
    'Brown Spot',
    'Healthy Rice Leaf',
    'Leaf Blast',
    'Sheath Blight'
]


def find_sample_for_class(class_name):
    for base in DATA_DIRS:
        class_dir = os.path.join(base, class_name)
        if os.path.isdir(class_dir):
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    return os.path.join(class_dir, fname)
    return None


def post_image(path):
    url = API_BASE.rstrip('/') + '/predict'
    with open(path, 'rb') as f:
        files = {'image': (os.path.basename(path), f, 'image/jpeg')}
        r = requests.post(url, files=files, timeout=30)
        return r.status_code, r.text


def main():
    samples = {}
    for cls in CLASS_NAMES:
        p = find_sample_for_class(cls)
        samples[cls] = p

    print('Using API base:', API_BASE)
    print('\nSamples found:')
    for cls, p in samples.items():
        print(f' - {cls}:', p or '(not found)')

    print('\nPosting samples...')
    for cls, p in samples.items():
        if not p:
            print(f'[{cls}] No sample found, skipping')
            continue
        try:
            code, text = post_image(p)
            print(f'[{cls}] HTTP {code} -> {text}')
        except Exception as e:
            print(f'[{cls}] Request failed:', e)


if __name__ == '__main__':
    try:
        import requests  # ensure dependency
    except Exception:
        print('Please install requests: python -m pip install requests')
        sys.exit(1)
    main()
