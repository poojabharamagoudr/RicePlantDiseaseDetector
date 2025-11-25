#!/usr/bin/env python3
"""Quick backend connectivity tester.

Usage:
  python backend/test_backend_connection.py            # checks /health
  python backend/test_backend_connection.py path/to/image.jpg  # POST image to /predict

The script prefers the `requests` library; if not present it will
fall back to `urllib` for the health check only and instruct how to
install `requests` for full POST support.
"""
import sys
import os

API_BASE = os.environ.get('API_BASE_URL', 'http://127.0.0.1:5000')


def check_health_with_requests(session=None):
    import requests
    url = API_BASE.rstrip('/') + '/health'
    try:
        r = (session or requests).get(url, timeout=5)
        print('GET', url, '->', r.status_code)
        try:
            print(r.json())
        except Exception:
            print(r.text[:200])
    except Exception as e:
        print('Health check failed:', e)


def post_image_with_requests(path):
    import requests
    url = API_BASE.rstrip('/') + '/predict'
    if not os.path.exists(path):
        print('Image not found:', path)
        return
    with open(path, 'rb') as f:
        files = {'image': (os.path.basename(path), f, 'image/jpeg')}
        try:
            r = requests.post(url, files=files, timeout=30)
            print('POST', url, '->', r.status_code)
            try:
                print(r.json())
            except Exception:
                print(r.text[:1000])
        except Exception as e:
            print('Request failed:', e)


def main():
    img = sys.argv[1] if len(sys.argv) > 1 else None

    # Try to use requests if available
    try:
        import requests  # noqa: F401
        has_requests = True
    except Exception:
        has_requests = False

    if has_requests:
        check_health_with_requests()
        if img:
            post_image_with_requests(img)
        else:
            print('\nTo POST an image: python backend/test_backend_connection.py path/to/image.jpg')
    else:
        # Fallback: simple health check using urllib
        print('`requests` not found. Install it with: pip install requests')
        try:
            from urllib.request import urlopen
            url = API_BASE.rstrip('/') + '/health'
            with urlopen(url, timeout=5) as r:
                body = r.read(200)
                print('GET', url, '->', r.status)
                print(body.decode('utf-8', errors='replace'))
        except Exception as e:
            print('Health check failed (urllib):', e)


if __name__ == '__main__':
    main()
