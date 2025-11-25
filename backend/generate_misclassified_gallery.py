#!/usr/bin/env python3
"""Generate an HTML gallery of misclassified images from eval_summary.txt

Reads `backend/eval_summary.txt`, extracts misclassified tuples, creates
thumbnails and an HTML file `backend/misclassified_gallery.html` for review.
"""
import re
import os
import base64
from io import BytesIO
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
EVAL_SUM = os.path.join(os.path.dirname(__file__), 'eval_summary.txt')
OUT_HTML = os.path.join(os.path.dirname(__file__), 'misclassified_gallery.html')


def parse_misclassified(eval_text, max_items=20):
    # Pattern matches tuples like: ('C:\...\file.jpg', 'Actual', 'Predicted', 0.5982)
    pattern = re.compile(r"\('(?P<path>[^']+)'\s*,\s*'(?P<actual>[^']+)'\s*,\s*'(?P<pred>[^']+)'\s*,\s*(?P<conf>[0-9.eE+-]+)\)")
    items = []
    for m in pattern.finditer(eval_text):
        p = m.group('path')
        items.append((p, m.group('actual'), m.group('pred'), float(m.group('conf'))))
        if len(items) >= max_items:
            break
    return items


def make_thumb_datauri(img_path, thumb_size=(420, 320)):
    try:
        with Image.open(img_path) as im:
            im = im.convert('RGB')
            im.thumbnail(thumb_size)
            buf = BytesIO()
            im.save(buf, format='JPEG', quality=75)
            b64 = base64.b64encode(buf.getvalue()).decode('ascii')
            return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        return None


def generate_html(items):
    parts = [
        '<!doctype html>',
        '<html><head><meta charset="utf-8"><title>Misclassified Samples</title>',
        '<style>body{font-family:sans-serif} .card{display:inline-block;margin:8px;border:1px solid #ddd;padding:6px;width:440px;vertical-align:top} img{max-width:100%;height:auto;border-bottom:1px solid #eee} .meta{padding:6px;font-size:14px}</style>',
        '</head><body>',
        '<h2>Misclassified Samples (from backend/eval_summary.txt)</h2>',
        '<p>If images don't show, verify the paths exist on this machine.</p>',
        '<div id="gallery">'
    ]

    for path, actual, pred, conf in items:
        exists = os.path.exists(path)
        img_src = make_thumb_datauri(path) if exists else None
        parts.append('<div class="card">')
        if img_src:
            parts.append(f'<a href="file:///{path.replace("\\","/")}" target="_blank"><img src="{img_src}" alt="{os.path.basename(path)}"></a>')
        else:
            parts.append('<div style="width:420px;height:320px;display:flex;align-items:center;justify-content:center;background:#f8f8f8;color:#666">(image not found)</div>')

        parts.append('<div class="meta">')
        parts.append(f'<strong>File:</strong> {os.path.basename(path)}<br>')
        parts.append(f'<strong>Full path:</strong> {path}<br>')
        parts.append(f'<strong>Actual:</strong> {actual} &nbsp;&nbsp; <strong>Pred:</strong> {pred} &nbsp;&nbsp; <strong>Conf:</strong> {conf:.3f}')
        parts.append('</div></div>')

    parts.append('</div></body></html>')
    return '\n'.join(parts)


def main():
    if not os.path.exists(EVAL_SUM):
        print('Cannot find', EVAL_SUM)
        return

    with open(EVAL_SUM, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    items = parse_misclassified(text, max_items=20)
    if not items:
        print('No misclassified entries found in', EVAL_SUM)
        return

    html = generate_html(items)
    with open(OUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)

    print('Wrote', OUT_HTML)
    print('Open it in your browser (e.g. right-click -> Open).')


if __name__ == '__main__':
    main()
