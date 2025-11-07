const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const predictBtn = document.getElementById("predictBtn");
const resultBox = document.getElementById("resultBox");
const diseaseName = document.getElementById("diseaseName");
const confidence = document.getElementById("confidence");
const treatment = document.getElementById("treatment");
const schemesDiv = document.getElementById("schemes");

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
    const res = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData
    });
    const data = await res.json();

    diseaseName.textContent = data.label || "Unknown";
    confidence.textContent = (data.confidence * 100).toFixed(2) + "%";
    treatment.textContent = data.treatment || "No treatment info.";

    schemesDiv.innerHTML = "";
    if (data.schemes && data.schemes.length > 0) {
      schemesDiv.innerHTML = "<strong>Government Schemes:</strong><ul>" +
        data.schemes.map(s => `<li>${s}</li>`).join("") +
        "</ul>";
    }

    resultBox.style.display = "block";
  } catch (err) {
    alert("Prediction failed. Check backend connection.");
    console.error(err);
  }
});
