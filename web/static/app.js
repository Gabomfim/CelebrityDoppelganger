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
let stream = null;
let selectedBlob = null;

async function loadConfig() {
  try {
    const response = await fetch("/api/config");
    const config = await response.json();
    if (config.github_url) document.querySelector("#github-link").href = config.github_url;
    if (config.linkedin_url) {
      ["#linkedin-link", "#footer-linkedin"].forEach((selector) => {
        const link = document.querySelector(selector); link.href = config.linkedin_url; link.hidden = false;
      });
    }
  } catch (_) { /* Static defaults remain usable. */ }
}

function stopCamera() { if (stream) stream.getTracks().forEach((track) => track.stop()); stream = null; }
function showStatus(message = "") { status.textContent = message; }

cameraButton.addEventListener("click", async () => {
  showStatus();
  if (!navigator.mediaDevices?.getUserMedia) return showStatus("Camera access isn't available here. Upload a photo instead.");
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 960 } }, audio: false });
    camera.srcObject = stream; camera.hidden = false; preview.hidden = true; placeholder.hidden = true;
    cameraButton.hidden = true; snapButton.hidden = false; matchButton.hidden = true;
  } catch (_) { showStatus("We couldn't access your camera. Check permission or upload a photo instead."); }
});

snapButton.addEventListener("click", () => {
  canvas.width = camera.videoWidth; canvas.height = camera.videoHeight;
  const context = canvas.getContext("2d"); context.translate(canvas.width, 0); context.scale(-1, 1); context.drawImage(camera, 0, 0);
  canvas.toBlob((blob) => { selectedBlob = blob; preview.src = URL.createObjectURL(blob); preview.hidden = false; camera.hidden = true; stopCamera(); snapButton.hidden = true; cameraButton.hidden = false; cameraButton.innerHTML = "<span>↻</span> Retake selfie"; matchButton.hidden = false; }, "image/jpeg", .92);
});

upload.addEventListener("change", () => {
  const file = upload.files?.[0]; if (!file) return;
  if (file.size > 10 * 1024 * 1024) return showStatus("Choose an image smaller than 10 MB.");
  selectedBlob = file; stopCamera(); camera.hidden = true; placeholder.hidden = true;
  preview.src = URL.createObjectURL(file); preview.hidden = false; snapButton.hidden = true;
  cameraButton.hidden = false; cameraButton.innerHTML = "<span>◉</span> Use camera"; matchButton.hidden = false; showStatus();
});

matchButton.addEventListener("click", async () => {
  if (!selectedBlob) return;
  matchButton.disabled = true; matchButton.innerHTML = "Reading the stars… <span>✦</span>"; showStatus();
  try {
    const response = await fetch("/api/match", { method: "POST", headers: { "Content-Type": selectedBlob.type || "image/jpeg" }, body: selectedBlob, cache: "no-store" });
    const payload = await response.json(); if (!response.ok) throw new Error(payload.detail || "Matching failed. Please try again.");
    renderResults(payload.matches); results.hidden = false; results.scrollIntoView({ behavior: "smooth" });
  } catch (error) { showStatus(error.message); }
  finally { selectedBlob = null; matchButton.disabled = false; matchButton.innerHTML = "Find my celebrity twins <span>→</span>"; }
});

function escapeHtml(value) { const node = document.createElement("div"); node.textContent = value || ""; return node.innerHTML; }
function renderResults(matches) {
  resultGrid.innerHTML = matches.map((match, index) => `<article class="match-card"><div class="match-photo"><img src="${encodeURI(match.photo_url)}" alt="${escapeHtml(match.name)}" loading="lazy"><span class="match-rank">#${index + 1} match</span><span class="similarity">${match.similarity}% twin</span></div><div class="match-copy"><div class="profession">${escapeHtml(match.profession || "Public figure")}</div><h3>${escapeHtml(match.name)}</h3><p class="curiosity">${escapeHtml(match.most_famous_for || match.why_famous || "A celebrated face with an uncanny resemblance to yours.")}</p>${match.wikipedia_url ? `<a class="wiki" href="${encodeURI(match.wikipedia_url)}" target="_blank" rel="noreferrer">Meet ${escapeHtml(match.name)} on Wikipedia ↗</a>` : ""}</div></article>`).join("");
}

againButton.addEventListener("click", () => { results.hidden = true; preview.hidden = true; placeholder.hidden = false; matchButton.hidden = true; upload.value = ""; showStatus(); window.scrollTo({ top: document.querySelector(".studio").offsetTop - 20, behavior: "smooth" }); });
window.addEventListener("pagehide", stopCamera); loadConfig();
