const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const predictBtn = document.getElementById("predictBtn");
const resultBox = document.getElementById("resultBox");
const diseaseName = document.getElementById("diseaseName");
const confidence = document.getElementById("confidence");
const treatment = document.getElementById("treatment");
const schemesDiv = document.getElementById("schemes");
const messageP = document.getElementById("message");

// -------------------------------
// API base URL (configurable at deploy time)
// - Set a meta tag <meta name="api-base-url" content="https://api.example.com"> in `index.html`.
// - If the meta tag is empty or missing, the code falls back to same-origin.
// -------------------------------
const metaApi = document.querySelector('meta[name="api-base-url"]');
const API_BASE_RAW = metaApi ? (metaApi.getAttribute('content') || '').trim() : '';
const API_BASE = API_BASE_RAW.replace(/\/$/, ''); // remove trailing slash

function apiUrl(path) {
  if (!path.startsWith('/')) path = '/' + path;
  if (!API_BASE) return path; // same-origin
  return API_BASE + path;
}

const cameraBtn = document.getElementById("cameraBtn");
const cameraView = document.getElementById("cameraView");
const captureBtn = document.getElementById("captureBtn");
let stream;
let capturedImageBlob = null;

// 🎥 Open back camera
cameraBtn.addEventListener("click", async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { exact: "environment" } }
    });
  } catch {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
  }

  cameraView.srcObject = stream;
  cameraView.style.display = "block";
  captureBtn.style.display = "inline-block";
});

// 📸 Capture image from video
captureBtn.addEventListener("click", () => {
  const canvas = document.createElement("canvas");
  canvas.width = cameraView.videoWidth;
  canvas.height = cameraView.videoHeight;
  canvas.getContext("2d").drawImage(cameraView, 0, 0);
  canvas.toBlob(blob => {
    capturedImageBlob = blob;
    previewImage.src = URL.createObjectURL(blob);
    previewImage.style.display = "block";
    // Stop camera stream after capture
    try {
      if (stream) {
        stream.getTracks().forEach(t => t.stop());
      }
    } catch (err) {
      console.warn('Could not stop stream', err);
    }
    cameraView.style.display = "none";
    captureBtn.style.display = "none";
  }, "image/jpeg");
});

// 🖼️ Preview when uploaded
imageInput.addEventListener("change", e => {
  const file = e.target.files[0];
  if (file) {
    previewImage.src = URL.createObjectURL(file);
    previewImage.style.display = "block";
    capturedImageBlob = file;
  }
});

// 🔍 Predict
predictBtn.addEventListener("click", async () => {
  if (!capturedImageBlob) {
    alert("Please upload or capture a leaf image first!");
    return;
  }

  const formData = new FormData();
  formData.append("image", capturedImageBlob);

  resultBox.style.display = "none";

  try {
    const res = await fetch(apiUrl('/predict'), {
      method: "POST",
      body: formData
    });
    const data = await res.json();

    // Clear previous message
    messageP.style.display = "none";
    messageP.textContent = "";

    const label = data.label || "Unknown";
    const conf = typeof data.confidence === 'number' ? data.confidence : 0;
    const govtSchemes = data.govt_schemes || data.schemes || [];

    diseaseName.textContent = label;

    if (label === "Unknown Image" || label.toLowerCase().includes('unknown')) {
      // Non-leaf / unknown
      confidence.textContent = "--";
      treatment.textContent = "";
      schemesDiv.innerHTML = "";
      // Show backend message if present
      if (data.message) {
        messageP.textContent = data.message;
        messageP.style.display = "block";
      }
    } else {
      confidence.textContent = (conf * 100).toFixed(2) + "%";
      treatment.textContent = data.treatment || "No treatment info.";

      schemesDiv.innerHTML = "";
      if (govtSchemes && govtSchemes.length > 0) {
        schemesDiv.innerHTML = "<strong>Government Schemes:</strong><ul>" +
          govtSchemes.map(s => `<li>${s}</li>`).join("") +
          "</ul>";
      }
    }

    resultBox.style.display = "block";
  } catch (err) {
    alert("Prediction failed. Check backend connection.");
    console.error(err);
  }
});
