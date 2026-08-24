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
const analyticsStartedAt = Date.now();
let analyticsEngaged = false;
let analyticsEnded = false;

function analyticsSessionId() {
  let value = sessionStorage.getItem("celebrity-twin-session");
  if (!value) {
    value = window.crypto?.randomUUID?.() || `anon-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    sessionStorage.setItem("celebrity-twin-session", value);
  }
  return value;
}

function clientContext() {
  const agent = navigator.userAgent;
  let browser = "Other";
  if (/LinkedInApp|LIApp/i.test(agent)) browser = "LinkedIn";
  else if (/Instagram/i.test(agent)) browser = "Instagram";
  else if (/FBAN|FBAV/i.test(agent)) browser = "Facebook";
  else if (/Edg/i.test(agent)) browser = "Edge";
  else if (/Firefox|FxiOS/i.test(agent)) browser = "Firefox";
  else if (/Chrome|CriOS/i.test(agent)) browser = "Chrome";
  else if (/Safari/i.test(agent)) browser = "Safari";
  else if (/; wv\)|WebView/i.test(agent)) browser = "Other WebView";
  let os = "Other";
  if (/Android/i.test(agent)) os = "Android";
  else if (/iPhone|iPad|iPod/i.test(agent)) os = "iOS";
  else if (/Windows/i.test(agent)) os = "Windows";
  else if (/Macintosh/i.test(agent)) os = "macOS";
  else if (/Linux/i.test(agent)) os = "Linux";
  const shortestSide = Math.min(screen.width, screen.height);
  const device = /iPad|Tablet/i.test(agent) || (navigator.maxTouchPoints > 1 && shortestSide >= 600)
    ? "tablet"
    : /Mobile|iPhone|Android/i.test(agent) || shortestSide < 600
      ? "mobile"
      : "desktop";
  return { browser, device, os };
}

function sendAnalytics(event, details = {}, beacon = false) {
  const payload = JSON.stringify({
    event,
    session_id: analyticsSessionId(),
    ...clientContext(),
    ...details,
  });
  if (beacon && navigator.sendBeacon) {
    navigator.sendBeacon("/api/analytics", new Blob([payload], { type: "application/json" }));
    return;
  }
  fetch("/api/analytics", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: payload,
    keepalive: true,
    cache: "no-store",
  }).catch(() => {});
}

function markEngaged() {
  if (analyticsEngaged) return;
  analyticsEngaged = true;
  sendAnalytics("engaged");
}

document.addEventListener("pointerdown", markEngaged, { once: true, passive: true });
document.addEventListener("keydown", markEngaged, { once: true });
sendAnalytics("page_view");

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

function waitForVideoMetadata() {
  if (camera.readyState >= HTMLMediaElement.HAVE_METADATA && camera.videoWidth > 0) {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const events = ["loadedmetadata", "resize"];
    const cleanup = () => events.forEach((event) => camera.removeEventListener(event, finish));
    const finish = () => {
      if (camera.videoWidth <= 0) return;
      window.clearTimeout(timer);
      cleanup();
      resolve();
    };
    const timer = window.setTimeout(() => {
      cleanup();
      reject(new Error("Camera startup timed out."));
    }, 20000);
    events.forEach((event) => camera.addEventListener(event, finish));
  });
}

function waitForDecodedVideoFrame() {
  return new Promise((resolve, reject) => {
    let frameCallbackId;
    const startedAt = performance.now();
    const cleanup = () => {
      window.clearInterval(poll);
      window.clearTimeout(timer);
      if (frameCallbackId && camera.cancelVideoFrameCallback) {
        camera.cancelVideoFrameCallback(frameCallbackId);
      }
    };
    const finish = () => {
      cleanup();
      resolve();
    };
    const hasFrame = () =>
      camera.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
      camera.videoWidth > 0 &&
      camera.currentTime > 0;
    const poll = window.setInterval(() => {
      if (hasFrame()) finish();
    }, 100);
    const timer = window.setTimeout(() => {
      cleanup();
      reject(new DOMException("No video frame was received", "VideoFrameError"));
    }, 10000);
    if (camera.requestVideoFrameCallback) {
      frameCallbackId = camera.requestVideoFrameCallback(() => finish());
    }
    // Some Safari versions do not advance currentTime immediately after the first frame.
    camera.addEventListener("playing", () => {
      if (performance.now() - startedAt > 250 && hasFrame()) finish();
    }, { once: true });
  });
}

async function requestCameraStream() {
  let timer;
  const desktopSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent) &&
    !/iPhone|iPad|iPod/i.test(navigator.userAgent);
  let request = navigator.mediaDevices.getUserMedia({
    video: desktopSafari
      ? { facingMode: "user" }
      : {
          facingMode: { ideal: "user" },
          width: { ideal: 1280 },
          height: { ideal: 960 },
        },
    audio: false,
  });
  const timeout = new Promise((_, reject) => {
    timer = window.setTimeout(() => {
      reject(new DOMException("Camera permission timed out", "TimeoutError"));
    }, 30000);
  });
  try {
    return await Promise.race([request, timeout]);
  } catch (error) {
    if (error?.name === "OverconstrainedError") {
      request = navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      return await request;
    }
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
  preview.src = previewUrls[previewUrls.length - 1];
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
  sendAnalytics("camera_started");
  showStatus("Requesting camera access…");
  selectedBlobs = [];
  matchButton.hidden = true;
  results.hidden = true;
  revokePreviews();

  if (!window.isSecureContext) {
    sendAnalytics("camera_failed", { error_code: "insecure_context" });
    showStatus("Camera access requires HTTPS. Open the secure site or upload a photo instead.");
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    sendAnalytics("camera_failed", { error_code: "unsupported" });
    showStatus("This browser doesn't support camera capture here. Upload a photo instead.");
    return;
  }

  cameraButton.disabled = true;
  try {
    stopCamera();
    stream = await requestCameraStream();
    camera.autoplay = true;
    camera.muted = true;
    camera.playsInline = true;
    camera.setAttribute("autoplay", "");
    camera.setAttribute("muted", "");
    camera.setAttribute("playsinline", "");
    camera.hidden = false;
    preview.hidden = true;
    placeholder.hidden = true;
    camera.srcObject = stream;
    await waitForVideoMetadata();
    const playback = camera.play();
    if (playback) await playback;
    await waitForDecodedVideoFrame();
    cameraButton.hidden = true;
    snapButton.hidden = false;
    snapButton.disabled = false;
    showStatus(
      "Camera ready — remove sunglasses, keep only one face in frame, then start the photobooth.",
    );
    sendAnalytics("camera_ready");
  } catch (error) {
    stopCamera();
    camera.hidden = true;
    placeholder.hidden = false;
    const denied = error?.name === "NotAllowedError" || error?.name === "SecurityError";
    const timedOut = error?.name === "TimeoutError";
    const unavailable = error?.name === "NotReadableError" || error?.name === "AbortError";
    const blackPreview = error?.name === "VideoFrameError";
    sendAnalytics("camera_failed", {
      error_code: denied ? "permission_denied" : timedOut ? "timeout" : unavailable ? "unavailable" : blackPreview ? "black_preview" : "other",
    });
    showStatus(
      denied
        ? "Camera permission was blocked. In Safari, open Settings → Websites → Camera, choose Allow for this site, then reload."
        : timedOut
          ? "Camera permission timed out. Allow access when your browser asks, then try again."
          : unavailable
            ? "The camera is busy or unavailable. Close other camera apps and tabs, then try again."
            : blackPreview
              ? "Safari opened the camera but did not deliver a video frame. Reload the page, close other camera tabs, or upload three photos instead."
            : "We couldn't start your camera. Reload the page or upload three photos instead.",
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
    sendAnalytics("photobooth_completed");
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
  sendAnalytics("upload_selected");
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
  const matchStartedAt = performance.now();
  sendAnalytics("match_submitted");
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
    sendAnalytics("match_succeeded", { duration_ms: Math.round(performance.now() - matchStartedAt) });
  } catch (error) {
    showStatus(error.message);
    sendAnalytics("match_failed", {
      duration_ms: Math.round(performance.now() - matchStartedAt),
      error_code: "match_error",
    });
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
    .map((match, index) => {
      const wikipediaUrl =
        match.wikipedia_url ||
        `https://en.wikipedia.org/wiki/Special:Search?search=${encodeURIComponent(match.name)}`;
      return `<a class="match-card" href="${encodeURI(wikipediaUrl)}" target="_blank" rel="noreferrer" aria-label="Read about ${escapeHtml(match.name)} on Wikipedia"><div class="match-photo"><img src="${encodeURI(match.photo_url)}" alt="${escapeHtml(match.name)}" loading="lazy"><span class="match-rank">#${index + 1} match</span><span class="similarity">${match.similarity}% twin</span></div><div class="match-copy"><div class="profession">${escapeHtml(match.profession || "Biography")}</div><h3>${escapeHtml(match.name)}</h3><p class="curiosity">${escapeHtml(match.why_famous || match.most_famous_for || `${match.name} is known for a notable career in entertainment.`)}</p><span class="wiki">Meet ${escapeHtml(match.name)} on Wikipedia ↗</span></div></a>`;
    })
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
  if (!analyticsEnded) {
    analyticsEnded = true;
    sendAnalytics(
      "session_end",
      { duration_ms: Date.now() - analyticsStartedAt, engaged: analyticsEngaged ? 1 : 0 },
      true,
    );
  }
  stopCamera();
  revokePreviews();
});
loadConfig();
