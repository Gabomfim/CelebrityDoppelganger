const camera = document.querySelector("#camera");
const preview = document.querySelector("#preview");
const canvas = document.querySelector("#canvas");
const placeholder = document.querySelector("#camera-placeholder");
const cameraButton = document.querySelector("#camera-button");
const snapButton = document.querySelector("#snap-button");
const upload = document.querySelector("#upload");
const matchButton = document.querySelector("#match-button");
const status = document.querySelector("#status");
const results = document.querySelector("#results");
const resultGrid = document.querySelector("#result-grid");
const againButton = document.querySelector("#again-button");
const countdown = document.querySelector("#countdown");
const flash = document.querySelector("#flash");
const shotStrip = document.querySelector("#shot-strip");

let stream = null;
let selectedBlobs = [];
let previewUrls = [];

async function loadConfig() {
  try {
    const response = await fetch("/api/config");
    const config = await response.json();
    if (config.github_url) document.querySelector("#github-link").href = config.github_url;
    if (config.linkedin_url) {
      ["#linkedin-link", "#footer-linkedin"].forEach((selector) => {
        const link = document.querySelector(selector);
        link.href = config.linkedin_url;
        link.hidden = false;
      });
    }
  } catch (_) {
    // Static defaults remain usable.
  }
}

function showStatus(message = "") {
  status.textContent = message;
}

function revokePreviews() {
  previewUrls.forEach((url) => URL.revokeObjectURL(url));
  previewUrls = [];
  shotStrip.innerHTML = "";
}

function stopCamera() {
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  camera.srcObject = null;
}

function waitForVideo() {
  if (camera.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && camera.videoWidth > 0) {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => reject(new Error("Camera startup timed out.")), 10000);
    camera.addEventListener(
      "loadeddata",
      () => {
        window.clearTimeout(timer);
        resolve();
      },
      { once: true },
    );
  });
}

async function requestCameraStream() {
  let timer;
  const request = navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: { ideal: "user" },
      width: { ideal: 1280 },
      height: { ideal: 960 },
    },
    audio: false,
  });
  const timeout = new Promise((_, reject) => {
    timer = window.setTimeout(() => {
      reject(new DOMException("Camera permission timed out", "TimeoutError"));
    }, 15000);
  });
  try {
    return await Promise.race([request, timeout]);
  } catch (error) {
    if (error?.name === "TimeoutError") {
      request.then((lateStream) => lateStream.getTracks().forEach((track) => track.stop()));
    }
    throw error;
  } finally {
    window.clearTimeout(timer);
  }
}

function showSelectedImages(blobs) {
  selectedBlobs = blobs;
  revokePreviews();
  previewUrls = blobs.map((blob) => URL.createObjectURL(blob));
  preview.src = previewUrls.at(-1);
  preview.hidden = false;
  camera.hidden = true;
  placeholder.hidden = true;
  snapButton.hidden = true;
  snapButton.disabled = true;
  cameraButton.hidden = false;
  cameraButton.innerHTML = "<span>↻</span> Retake 3 selfies";
  matchButton.hidden = false;
  shotStrip.innerHTML = previewUrls
    .map((url, index) => `<img src="${url}" alt="Selfie ${index + 1} of 3">`)
    .join("");
}

cameraButton.addEventListener("click", async () => {
  showStatus("Requesting camera access…");
  selectedBlobs = [];
  matchButton.hidden = true;
  results.hidden = true;
  revokePreviews();

  if (!window.isSecureContext) {
    showStatus("Camera access requires HTTPS. Open the secure site or upload a photo instead.");
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    showStatus("This browser doesn't support camera capture here. Upload a photo instead.");
    return;
  }

  cameraButton.disabled = true;
  try {
    stopCamera();
    stream = await requestCameraStream();
    camera.srcObject = stream;
    camera.hidden = false;
    preview.hidden = true;
    placeholder.hidden = true;
    await camera.play();
    await waitForVideo();
    cameraButton.hidden = true;
    snapButton.hidden = false;
    snapButton.disabled = false;
    showStatus("Camera ready — center your face, then start the three-shot photobooth.");
  } catch (error) {
    stopCamera();
    camera.hidden = true;
    placeholder.hidden = false;
    const denied = error?.name === "NotAllowedError" || error?.name === "SecurityError";
    const timedOut = error?.name === "TimeoutError";
    showStatus(
      denied
        ? "Camera permission was blocked. Allow camera access in your browser settings or upload a photo."
        : timedOut
          ? "Camera permission timed out. Allow access when your browser asks, then try again."
          : "We couldn't start your camera. Try another browser or upload a photo instead.",
    );
  } finally {
    cameraButton.disabled = false;
  }
});

function delay(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function captureFrame() {
  canvas.width = camera.videoWidth;
  canvas.height = camera.videoHeight;
  const context = canvas.getContext("2d");
  context.translate(canvas.width, 0);
  context.scale(-1, 1);
  context.drawImage(camera, 0, 0);
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("We couldn't capture that frame."))),
      "image/jpeg",
      0.92,
    );
  });
}

async function runCountdown(shotNumber) {
  countdown.hidden = false;
  for (const value of [3, 2, 1]) {
    countdown.textContent = value;
    showStatus(`Photo ${shotNumber} of 3 — hold that pose…`);
    await delay(650);
  }
  countdown.textContent = "★";
}

snapButton.addEventListener("click", async () => {
  if (!stream || !camera.videoWidth || !camera.videoHeight) {
    showStatus("The camera is still getting ready. Wait a moment and try again.");
    return;
  }
  snapButton.disabled = true;
  cameraButton.disabled = true;
  const blobs = [];
  try {
    for (let shot = 1; shot <= 3; shot += 1) {
      await runCountdown(shot);
      blobs.push(await captureFrame());
      flash.hidden = false;
      await delay(120);
      flash.hidden = true;
      await delay(350);
    }
    stopCamera();
    showSelectedImages(blobs);
    showStatus("Three great shots! We'll combine them into your private face prototype.");
  } catch (error) {
    showStatus(error.message || "We couldn't finish the photobooth. Please try again.");
  } finally {
    countdown.hidden = true;
    flash.hidden = true;
    snapButton.disabled = false;
    cameraButton.disabled = false;
  }
});

upload.addEventListener("change", () => {
  const files = Array.from(upload.files || []);
  if (!files.length) return;
  if (files.length !== 3) {
    upload.value = "";
    showStatus("Choose exactly three photos to create your prototype.");
    return;
  }
  if (files.some((file) => file.size > 10 * 1024 * 1024)) {
    upload.value = "";
    showStatus("Each photo must be smaller than 10 MB.");
    return;
  }
  stopCamera();
  showSelectedImages(files);
  cameraButton.innerHTML = "<span>◉</span> Use camera";
  showStatus("Three photos ready. They will be processed in memory and never stored.");
});

function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("One of the photos could not be read."));
    reader.readAsDataURL(blob);
  });
}

matchButton.addEventListener("click", async () => {
  if (selectedBlobs.length !== 3) return;
  matchButton.disabled = true;
  matchButton.innerHTML = "Reading the stars… <span>✦</span>";
  showStatus();
  try {
    const images = await Promise.all(selectedBlobs.map(blobToDataUrl));
    const response = await fetch("/api/match-prototype", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ images }),
      cache: "no-store",
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Matching failed. Please try again.");
    renderResults(payload.matches);
    results.hidden = false;
    results.scrollIntoView({ behavior: "smooth" });
    selectedBlobs = [];
    revokePreviews();
    preview.hidden = true;
    placeholder.hidden = false;
    matchButton.hidden = true;
  } catch (error) {
    showStatus(error.message);
  } finally {
    matchButton.disabled = false;
    matchButton.innerHTML = "Find my celebrity twins <span>→</span>";
  }
});

function escapeHtml(value) {
  const node = document.createElement("div");
  node.textContent = value || "";
  return node.innerHTML;
}

function renderResults(matches) {
  resultGrid.innerHTML = matches
    .map(
      (match, index) => `<article class="match-card"><div class="match-photo"><img src="${encodeURI(match.photo_url)}" alt="${escapeHtml(match.name)}" loading="lazy"><span class="match-rank">#${index + 1} match</span><span class="similarity">${match.similarity}% twin</span></div><div class="match-copy"><div class="profession">${escapeHtml(match.profession || "Public figure")}</div><h3>${escapeHtml(match.name)}</h3><p class="curiosity">${escapeHtml(match.most_famous_for || match.why_famous || "A celebrated face with an uncanny resemblance to yours.")}</p>${match.wikipedia_url ? `<a class="wiki" href="${encodeURI(match.wikipedia_url)}" target="_blank" rel="noreferrer">Meet ${escapeHtml(match.name)} on Wikipedia ↗</a>` : ""}</div></article>`,
    )
    .join("");
}

againButton.addEventListener("click", () => {
  results.hidden = true;
  preview.hidden = true;
  placeholder.hidden = false;
  matchButton.hidden = true;
  upload.value = "";
  selectedBlobs = [];
  revokePreviews();
  showStatus();
  window.scrollTo({
    top: document.querySelector(".studio").offsetTop - 20,
    behavior: "smooth",
  });
});

window.addEventListener("pagehide", () => {
  stopCamera();
  revokePreviews();
});
loadConfig();
