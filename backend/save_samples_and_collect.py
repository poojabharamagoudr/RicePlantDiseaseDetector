#!/usr/bin/env python3
"""Post one sample image per class, save responses, and copy misclassified images.

Outputs:
 - backend/sample_results.json  (list of {class, path, response_json})
 - backend/misclassified_samples/  (copies of images where predicted != expected)

Run from repo root:
  python backend/save_samples_and_collect.py
"""
import os
import sys
import json
import shutil
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_JSON = os.path.join(os.path.dirname(__file__), 'sample_results.json')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'misclassified_samples')

DATA_DIRS = [
    os.path.join(ROOT, 'data', 'test'),
    os.path.join(ROOT, 'data', 'val'),
    os.path.join(ROOT, 'data', 'train'),
]

API_BASE = os.environ.get('API_BASE_URL', 'http://127.0.0.1:5000')

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
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {'raw': r.text}


def ensure_out():
    os.makedirs(OUT_DIR, exist_ok=True)


def main():
    try:
        import requests  # ensure installed
    except Exception:
        print('Please install requests: python -m pip install requests')
        sys.exit(1)

    samples = {}
    for cls in CLASS_NAMES:
        p = find_sample_for_class(cls)
        samples[cls] = p

    results = []
    ensure_out()

    for cls, p in samples.items():
        if not p:
            print(f'[{cls}] no sample, skipping')
            continue
        print(f'Posting {cls}:', p)
        try:
            status, body = post_image(p)
        except Exception as e:
            print('Request failed for', p, e)
            status, body = None, {'error': str(e)}

        rec = {'class': cls, 'path': p, 'status': status, 'response': body}
        results.append(rec)

        # determine predicted label
        pred = None
        if isinstance(body, dict):
            pred = body.get('label')

        expected = cls
        if pred and pred != expected:
            # copy file to misclassified_samples
            dest = os.path.join(OUT_DIR, f"{cls}__{os.path.basename(p)}")
            try:
                shutil.copy2(p, dest)
                print('  Copied misclassified to', dest)
            except Exception as e:
                print('  Failed to copy', e)

    # save JSON
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print('Wrote', OUT_JSON)
    print('Misclassified samples (if any) are in', OUT_DIR)


if __name__ == '__main__':
    main()
