Rice Plant Disease Detector

An AI-powered web application that detects diseases in rice plant leaves using deep learning.
It helps farmers and researchers identify plant diseases early, view treatment advice, and learn about government schemes.

Features :
Upload or capture leaf image for real-time prediction
Displays disease name, confidence, and treatment suggestions
Provides government schemes related to crop health
Works on both mobile and desktop

Technologies Used :
Frontend: HTML, CSS, JavaScript
Backend: Flask (Python)

Model: MobileNetV2 (Transfer Learning)
Dataset: Rice leaf disease images (around 1000+ per class)

How It Works :
User uploads or captures a leaf image.
The image is processed and passed to the trained MobileNetV2 model.

The model predicts the disease type and confidence.
Treatment details and government schemes are shown instantly.

Team
Developed by Pooja B Bharamagoudr and team as part of the final year engineering project.

Getting Started
---------------

Prerequisites
- Python 3.8+ installed
- (Optional) Virtual environment tool such as `venv` or `conda`

Install dependencies
1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install Python requirements:

```powershell
pip install -r requirements.txt
```

Run the backend (Flask)

```powershell
python backend\app.py
```

The backend will start on port 5000 by default (`http://127.0.0.1:5000`).

Serve the frontend (development)

You can open `frontend/index.html` directly in the browser, or serve it via a simple HTTP server so `fetch` requests use same-origin or deployed host:

```powershell
cd frontend
python -m http.server 8000
# open http://localhost:8000/
```

Configuring API base URL (for deployment)
----------------------------------------

The frontend supports a deploy-time API base URL via a meta tag in `frontend/index.html`:

```html
<meta name="api-base-url" content="">
```

- Leave `content` empty for same-origin (the client will call `/predict`).
- Set `content` to your API host (for example `https://api.example.com`) to have the client call `https://api.example.com/predict`.

If you set a different host, ensure your backend allows CORS from your frontend origin. By default the backend enables CORS for all origins (`CORS(app)`) in `backend/app.py` — for production restrict it to your frontend origin:

```python
from flask_cors import CORS
CORS(app, origins=["https://your-frontend.example.com"])
```

API: `/predict`
----------------
Endpoint: `POST /predict`

Request: multipart/form-data with field `image` containing the image file (from file chooser or camera blob).

Example curl (file upload):

```bash
curl -X POST -F "image=@/path/to/leaf.jpg" https://your-host.example.com/predict
```

Example curl (camera capture saved to file):

```bash
curl -X POST -F "image=@capture.jpg" https://your-host.example.com/predict
```

Responses
- Leaf image detected (successful prediction):

```json
{
	"label": "Leaf Blast",
	"confidence": 0.974,
	"treatment": "Use fungicides like tricyclazole, maintain proper spacing, avoid excessive nitrogen.",
	"govt_schemes": ["PM-Kisan", "Crop Insurance Scheme"]
}
```

- Non-leaf / unknown image:

```json
{
	"label": "Unknown Image",
	"confidence": 0.0,
	"treatment": "",
	"govt_schemes": [],
	"message": "Uploaded image does not appear to be a rice leaf. Please upload a clear leaf image."
}
```

Notes & Tips
- The backend uses a MobileNetV2 model (`model/rice_disease_mobilenetv2.h5`). Keep that file in `model/`.
- The leaf-check is a lightweight heuristic (green-pixel proportion). For production you can replace it with a small binary leaf/non-leaf classifier for improved robustness.
- If your frontend is hosted on a different origin than the backend, either set the `api-base-url` meta tag or configure a reverse proxy so `/predict` is forwarded to the API host.

Contact
-------
Project by Pooja B Bharamagoudr.
