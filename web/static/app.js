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
const cosineWheel = document.querySelector("#cosine-wheel");
const cosineWheelLegend = document.querySelector("#cosine-wheel-legend");
const cosineWheelTooltip = document.querySelector("#cosine-wheel-tooltip");
const againButton = document.querySelector("#again-button");
const countdown = document.querySelector("#countdown");
const flash = document.querySelector("#flash");
const shotStrip = document.querySelector("#shot-strip");
const shareRevealButton = document.querySelector("#share-reveal-button");
const shareDialog = document.querySelector("#share-dialog");
const shareClose = document.querySelector("#share-close");
const sharePreview = document.querySelector("#share-preview");
const shareStatus = document.querySelector("#share-status");
const shareCanvas = document.querySelector("#share-canvas");
const downloadShareButton = document.querySelector("#download-share");
const copyShareTextButton = document.querySelector("#copy-share-text");
const publicShareUrl = "https://gabomfim.github.io/CelebrityDoppelganger/";

let stream = null;
let selectedBlobs = [];
let previewUrls = [];
let currentMatches = [];
let shareBlob = null;
let sharePreviewUrl = null;
const analyticsStartedAt = Date.now();
let analyticsEngaged = false;
let analyticsEnded = false;
const analyticsPreference = new URLSearchParams(window.location.search).get("analytics");
if (analyticsPreference === "off") {
  localStorage.setItem("celebrity-twin-analytics", "off");
} else if (analyticsPreference === "on") {
  localStorage.removeItem("celebrity-twin-analytics");
}
const analyticsEnabled = localStorage.getItem("celebrity-twin-analytics") !== "off";

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
  if (!analyticsEnabled) return;
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
  status.textContent = window.CelebrityTwinI18n?.t(message) || message;
}

function translated(message) {
  return window.CelebrityTwinI18n?.t(message) || message;
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
    renderResults(payload.matches, payload.cosine_wheel || []);
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

function shareCaption(match) {
  if (window.CelebrityTwinI18n?.locale === "pt-BR") {
    return `Saiu o resultado ✨ Meu sósia famoso é ${match.name} — ${match.similarity}% de semelhança! Qual é o seu? ${publicShareUrl}`;
  }
  return `The verdict is in ✨ My Celebrity Twin is ${match.name} — a ${match.similarity}% match! Who is yours? ${publicShareUrl}`;
}

function roundedRectangle(context, x, y, width, height, radius) {
  const corner = Math.min(radius, width / 2, height / 2);
  context.beginPath();
  context.moveTo(x + corner, y);
  context.arcTo(x + width, y, x + width, y + height, corner);
  context.arcTo(x + width, y + height, x, y + height, corner);
  context.arcTo(x, y + height, x, y, corner);
  context.arcTo(x, y, x + width, y, corner);
  context.closePath();
}

function drawCoverImage(context, image, x, y, width, height, radius) {
  const scale = Math.max(width / image.naturalWidth, height / image.naturalHeight);
  const sourceWidth = width / scale;
  const sourceHeight = height / scale;
  const sourceX = (image.naturalWidth - sourceWidth) / 2;
  const sourceY = Math.max(0, (image.naturalHeight - sourceHeight) * 0.22);
  context.save();
  roundedRectangle(context, x, y, width, height, radius);
  context.clip();
  context.drawImage(image, sourceX, sourceY, sourceWidth, sourceHeight, x, y, width, height);
  const shade = context.createLinearGradient(0, y + height * 0.58, 0, y + height);
  shade.addColorStop(0, "rgba(30,4,12,0)");
  shade.addColorStop(1, "rgba(30,4,12,.72)");
  context.fillStyle = shade;
  context.fillRect(x, y, width, height);
  context.restore();
}

function fitHeadline(context, text, maxWidth, startSize, minimumSize = 54) {
  let size = startSize;
  do {
    context.font = `900 ${size}px Fraunces, Georgia, serif`;
    if (context.measureText(text).width <= maxWidth) return size;
    size -= 4;
  } while (size > minimumSize);
  return minimumSize;
}

async function loadSharePhoto(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(translated("We couldn't prepare the celebrity photo."));
  const objectUrl = URL.createObjectURL(await response.blob());
  const image = new Image();
  image.src = objectUrl;
  try {
    await image.decode();
    return image;
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

async function createShareCard(match) {
  await document.fonts?.ready;
  const photo = await loadSharePhoto(match.photo_url);
  const context = shareCanvas.getContext("2d");
  const width = shareCanvas.width;
  const height = shareCanvas.height;

  const background = context.createLinearGradient(0, 0, width, height);
  background.addColorStop(0, "#16070b");
  background.addColorStop(0.48, "#64142a");
  background.addColorStop(1, "#21080f");
  context.fillStyle = background;
  context.fillRect(0, 0, width, height);

  const spotlight = context.createRadialGradient(width / 2, 180, 20, width / 2, 420, 610);
  spotlight.addColorStop(0, "rgba(255,231,169,.28)");
  spotlight.addColorStop(1, "rgba(255,231,169,0)");
  context.fillStyle = spotlight;
  context.fillRect(0, 0, width, 900);

  context.strokeStyle = "#e5b84d";
  context.lineWidth = 6;
  roundedRectangle(context, 34, 34, width - 68, height - 68, 34);
  context.stroke();
  context.strokeStyle = "rgba(229,184,77,.34)";
  context.lineWidth = 2;
  roundedRectangle(context, 52, 52, width - 104, height - 104, 26);
  context.stroke();

  context.fillStyle = "#f0c761";
  context.font = "800 25px DM Sans, Arial, sans-serif";
  context.textAlign = "center";
  context.letterSpacing = "5px";
  context.fillText(translated("THE RED CARPET REVEAL"), width / 2, 116);
  context.letterSpacing = "0px";

  drawCoverImage(context, photo, 150, 155, 780, 650, 36);
  context.strokeStyle = "rgba(248,215,125,.9)";
  context.lineWidth = 5;
  roundedRectangle(context, 150, 155, 780, 650, 36);
  context.stroke();

  context.fillStyle = "#fff5df";
  context.font = "700 31px DM Sans, Arial, sans-serif";
  context.fillText(translated("MY CELEBRITY LOOKALIKE IS"), width / 2, 872);
  const name = String(match.name).toUpperCase();
  const nameSize = fitHeadline(context, name, 900, 94);
  context.fillStyle = "#f4cb68";
  context.font = `900 ${nameSize}px Fraunces, Georgia, serif`;
  context.fillText(name, width / 2, 972);

  const pillWidth = 270;
  roundedRectangle(context, (width - pillWidth) / 2, 1018, pillWidth, 70, 35);
  context.fillStyle = "#fff8e8";
  context.fill();
  context.fillStyle = "#5b1225";
  context.font = "900 30px DM Sans, Arial, sans-serif";
  context.fillText(`${match.similarity}% ${translated("TWIN")}`, width / 2, 1065);

  context.strokeStyle = "rgba(244,203,104,.45)";
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo(135, 1144);
  context.lineTo(945, 1144);
  context.stroke();
  context.fillStyle = "#fff8e8";
  context.font = "900 46px Fraunces, Georgia, serif";
  context.fillText("★  Celebrity Twin", width / 2, 1216);
  context.fillStyle = "rgba(255,248,232,.76)";
  context.font = "600 22px DM Sans, Arial, sans-serif";
  context.fillText("gabomfim.github.io/CelebrityDoppelganger", width / 2, 1262);

  return new Promise((resolve, reject) => {
    shareCanvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("We couldn't create the reveal card."))),
      "image/png",
    );
  });
}

function shareFilename(match) {
  const slug = match.name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  return `celebrity-twin-${slug || "reveal"}.png`;
}

function downloadShareCard() {
  if (!shareBlob || !currentMatches[0]) return;
  const link = document.createElement("a");
  link.href = URL.createObjectURL(shareBlob);
  link.download = shareFilename(currentMatches[0]);
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

async function copyShareCaption() {
  const caption = shareCaption(currentMatches[0]);
  await navigator.clipboard.writeText(caption);
  shareStatus.textContent = "Caption copied — paste it with your reveal card.";
}

function fallbackShareUrl(platform, caption) {
  const encodedCaption = encodeURIComponent(caption);
  const encodedSite = encodeURIComponent(publicShareUrl);
  if (platform === "whatsapp") return `https://wa.me/?text=${encodedCaption}`;
  if (platform === "facebook") return `https://www.facebook.com/sharer/sharer.php?u=${encodedSite}`;
  if (platform === "x") return `https://twitter.com/intent/tweet?text=${encodedCaption}`;
  if (platform === "threads") return `https://www.threads.net/intent/post?text=${encodedCaption}`;
  if (platform === "instagram") return "https://www.instagram.com/";
  return null;
}

async function shareToPlatform(platform) {
  if (!shareBlob || !currentMatches[0]) return;
  const match = currentMatches[0];
  const caption = shareCaption(match);
  const file = new File([shareBlob], shareFilename(match), { type: "image/png" });
  if (navigator.share && navigator.canShare?.({ files: [file] })) {
    try {
      await navigator.share({ title: "My Celebrity Twin", text: caption, url: publicShareUrl, files: [file] });
      shareStatus.textContent = "Your red carpet reveal is ready for its audience!";
      sendAnalytics("share_completed");
      return;
    } catch (error) {
      if (error?.name === "AbortError") return;
    }
  }

  downloadShareCard();
  const url = fallbackShareUrl(platform, caption);
  if (url) window.open(url, "_blank", "noopener,noreferrer");
  try {
    await copyShareCaption();
  } catch (_) {
    shareStatus.textContent = "Image downloaded. Add it to your post with the caption below.";
  }
  if (!url) shareStatus.textContent = "Image downloaded — upload it to your favorite social app.";
  sendAnalytics("share_completed");
}

async function openShareDialog() {
  if (!currentMatches[0]) return;
  shareDialog.showModal();
  shareStatus.textContent = "Creating your red carpet reveal…";
  shareRevealButton.disabled = true;
  sendAnalytics("share_opened");
  try {
    shareBlob = await createShareCard(currentMatches[0]);
    if (sharePreviewUrl) URL.revokeObjectURL(sharePreviewUrl);
    sharePreviewUrl = URL.createObjectURL(shareBlob);
    sharePreview.src = sharePreviewUrl;
    shareStatus.textContent = "Choose an app. On mobile, the image opens in your share sheet.";
  } catch (error) {
    shareStatus.textContent = error.message || "We couldn't create the reveal card.";
  } finally {
    shareRevealButton.disabled = false;
  }
}

function renderResults(matches, wheelPoints = []) {
  currentMatches = matches;
  const portuguese = window.CelebrityTwinI18n?.locale === "pt-BR";
  resultGrid.innerHTML = matches
    .map((match, index) => {
      const wikipediaUrl =
        match.wikipedia_url ||
        `https://en.wikipedia.org/wiki/Special:Search?search=${encodeURIComponent(match.name)}`;
      const aria = portuguese ? `Leia sobre ${match.name} na Wikipédia` : `Read about ${match.name} on Wikipedia`;
      const rank = portuguese ? `#${index + 1} resultado` : `#${index + 1} match`;
      const twin = portuguese ? "sósia" : "twin";
      const biography = portuguese ? "Biografia" : "Biography";
      const fallback = portuguese ? `${match.name} é conhecido por uma carreira de destaque no entretenimento.` : `${match.name} is known for a notable career in entertainment.`;
      const wiki = portuguese ? `Conheça ${match.name} na Wikipédia ↗` : `Meet ${match.name} on Wikipedia ↗`;
      return `<a class="match-card" href="${encodeURI(wikipediaUrl)}" target="_blank" rel="noreferrer" aria-label="${escapeHtml(aria)}"><div class="match-photo"><img src="${encodeURI(match.photo_url)}" alt="${escapeHtml(match.name)}" loading="lazy"><span class="match-rank">${rank}</span><span class="similarity">${match.similarity}% ${twin}</span></div><div class="match-copy"><div class="profession">${escapeHtml(match.profession || biography)}</div><h3>${escapeHtml(match.name)}</h3><p class="curiosity">${escapeHtml(match.why_famous || match.most_famous_for || fallback)}</p><span class="wiki">${escapeHtml(wiki)}</span></div></a>`;
    })
    .join("");
  renderCosineWheel(matches, wheelPoints);
}

function svgElement(name, attributes = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  return element;
}

function renderCosineWheel(matches, wheelPoints) {
  const portuguese = window.CelebrityTwinI18n?.locale === "pt-BR";
  const you = portuguese ? "Você" : "You";
  const distanceLabel = portuguese ? "distância cosseno" : "cosine distance";
  const centerX = 360;
  const centerY = 260;
  const radius = 190;
  const svg = svgElement("svg", {
    viewBox: "0 0 720 520",
    "aria-label": portuguese
      ? "Roda interativa de distância cosseno"
      : "Interactive cosine-distance wheel",
  });

  const defs = svgElement("defs");
  const glow = svgElement("filter", {
    id: "wheel-glow",
    x: "-50%",
    y: "-50%",
    width: "200%",
    height: "200%",
  });
  glow.append(svgElement("feGaussianBlur", { stdDeviation: "5", result: "blur" }));
  const merge = svgElement("feMerge");
  merge.append(
    svgElement("feMergeNode", { in: "blur" }),
    svgElement("feMergeNode", { in: "SourceGraphic" }),
  );
  glow.append(merge);
  defs.append(glow);
  svg.append(defs);

  const plot = svgElement("g", { class: "wheel-plot" });
  plot.append(
    svgElement("circle", { cx: centerX, cy: centerY, r: radius, class: "wheel-ring" }),
    svgElement("line", {
      x1: centerX - radius,
      y1: centerY,
      x2: centerX + radius,
      y2: centerY,
      class: "wheel-axis",
    }),
    svgElement("line", {
      x1: centerX,
      y1: centerY - radius,
      x2: centerX,
      y2: centerY + radius,
      class: "wheel-axis",
    }),
  );

  const originLabel = svgElement("text", {
    x: centerX,
    y: centerY - 12,
    class: "wheel-origin-label",
    "text-anchor": "middle",
  });
  originLabel.textContent = portuguese ? "círculo unitário normalizado" : "normalized unit circle";
  plot.append(originLabel);

  const highlighted = new Map(matches.slice(0, 3).map((match, index) => [match.class_name, index + 1]));
  const availablePoints = wheelPoints.length
    ? wheelPoints
    : matches.map((match) => ({
        class_name: match.class_name,
        name: match.name,
        similarity: Number(match.similarity) / 100,
      }));
  let pinnedPoint = null;

  function pointPosition(point, forcedSide = null) {
    const similarity = Math.max(-1, Math.min(1, Number(point.similarity)));
    const verticalMagnitude = Math.sqrt(Math.max(0, 1 - similarity ** 2));
    const hash = [...String(point.class_name)].reduce(
      (total, character) => total + character.charCodeAt(0),
      0,
    );
    const side = forcedSide || (hash % 2 === 0 ? 1 : -1);
    return {
      similarity,
      x: centerX + radius * similarity,
      y: centerY - side * radius * verticalMagnitude,
    };
  }

  function showPointDetails(point, target, pin = false) {
    if (pinnedPoint && !pin) return;
    pinnedPoint = pin ? point.class_name : pinnedPoint;
    const rect = cosineWheel.getBoundingClientRect();
    const targetRect = target.getBoundingClientRect();
    const name = document.createElement("strong");
    name.textContent = point.name;
    const metric = document.createElement("span");
    const similarityPercent = (Number(point.similarity) * 100).toFixed(1);
    metric.textContent = `${similarityPercent}% · ${distanceLabel} ${(1 - Number(point.similarity)).toFixed(3)}`;
    cosineWheelTooltip.replaceChildren(name, metric);
    cosineWheelTooltip.style.left = `${targetRect.left - rect.left + targetRect.width / 2}px`;
    cosineWheelTooltip.style.top = `${targetRect.top - rect.top}px`;
    cosineWheelTooltip.hidden = false;
  }

  availablePoints.forEach((point) => {
    if (highlighted.has(point.class_name)) return;
    const position = pointPosition(point);
    const dot = svgElement("circle", {
      cx: position.x,
      cy: position.y,
      r: 3.2,
      class: "wheel-background-point",
      tabindex: "0",
      role: "button",
      "aria-label": `${point.name}, ${distanceLabel} ${(1 - position.similarity).toFixed(3)}`,
    });
    dot.addEventListener("pointerenter", () => showPointDetails(point, dot));
    dot.addEventListener("pointerleave", () => {
      if (!pinnedPoint) cosineWheelTooltip.hidden = true;
    });
    dot.addEventListener("click", (event) => {
      event.stopPropagation();
      pinnedPoint = pinnedPoint === point.class_name ? null : point.class_name;
      if (pinnedPoint) showPointDetails(point, dot, true);
      else cosineWheelTooltip.hidden = true;
    });
    dot.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        pinnedPoint = point.class_name;
        showPointDetails(point, dot, true);
      }
    });
    plot.append(dot);
  });

  const userX = centerX + radius;
  const userY = centerY;
  const userPoint = svgElement("g", { class: "wheel-user" });
  userPoint.append(
    svgElement("circle", { cx: userX, cy: userY, r: 14, class: "wheel-user-halo" }),
    svgElement("circle", { cx: userX, cy: userY, r: 8, class: "wheel-user-point" }),
  );
  const userLabel = svgElement("text", {
    x: userX + 18,
    y: userY + 5,
    class: "wheel-user-label",
  });
  userLabel.textContent = you;
  userPoint.append(userLabel);
  plot.append(userPoint);

  cosineWheelLegend.replaceChildren();
  matches.slice(0, 3).forEach((match, index) => {
    const similarity = Math.max(-1, Math.min(1, Number(match.similarity) / 100));
    const cosineDistance = 1 - similarity;
    const side = index % 2 === 0 ? 1 : -1;
    const position = pointPosition({ ...match, similarity }, side);
    const angle = Math.acos(similarity);
    const pointX = position.x;
    const pointY = position.y;
    const labelRadius = radius + 25 + (index === 2 ? 24 : 0);
    const labelX = centerX + labelRadius * Math.cos(angle);
    const labelY = centerY - side * labelRadius * Math.sin(angle);

    plot.append(
      svgElement("line", {
        x1: userX,
        y1: userY,
        x2: pointX,
        y2: pointY,
        class: `wheel-connection wheel-rank-${index + 1}`,
      }),
    );
    const highlightedPoint = svgElement("circle", {
        cx: pointX,
        cy: pointY,
        r: index === 0 ? 9 : 7,
        class: `wheel-celebrity wheel-rank-${index + 1}`,
        "data-cosine-distance": cosineDistance.toFixed(3),
        tabindex: "0",
        role: "button",
        "aria-label": `${match.name}, ${distanceLabel} ${cosineDistance.toFixed(3)}`,
      });
    highlightedPoint.addEventListener("pointerenter", () =>
      showPointDetails({ ...match, similarity }, highlightedPoint),
    );
    highlightedPoint.addEventListener("pointerleave", () => {
      if (!pinnedPoint) cosineWheelTooltip.hidden = true;
    });
    highlightedPoint.addEventListener("click", (event) => {
      event.stopPropagation();
      pinnedPoint = pinnedPoint === match.class_name ? null : match.class_name;
      if (pinnedPoint) showPointDetails({ ...match, similarity }, highlightedPoint, true);
      else cosineWheelTooltip.hidden = true;
    });
    plot.append(highlightedPoint);
    const label = svgElement("text", {
      x: labelX,
      y: labelY,
      class: `wheel-celebrity-label wheel-rank-${index + 1}`,
      "text-anchor": similarity > 0.92 ? "start" : "middle",
    });
    label.textContent = `#${index + 1} ${match.name}`;
    plot.append(label);

    const item = document.createElement("li");
    const name = document.createElement("strong");
    name.textContent = match.name;
    const metric = document.createElement("span");
    metric.textContent = `${match.similarity}% · ${distanceLabel} ${cosineDistance.toFixed(3)}`;
    item.append(name, metric);
    cosineWheelLegend.append(item);
  });

  svg.append(plot);
  let viewBox = { x: 0, y: 0, width: 720, height: 520 };
  let drag = null;
  const applyViewBox = () =>
    svg.setAttribute("viewBox", `${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`);
  svg.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const rect = svg.getBoundingClientRect();
      const ratioX = (event.clientX - rect.left) / rect.width;
      const ratioY = (event.clientY - rect.top) / rect.height;
      const anchorX = viewBox.x + ratioX * viewBox.width;
      const anchorY = viewBox.y + ratioY * viewBox.height;
      const factor = event.deltaY < 0 ? 0.84 : 1.19;
      const nextWidth = Math.max(180, Math.min(1000, viewBox.width * factor));
      const nextHeight = nextWidth * (520 / 720);
      viewBox = {
        x: anchorX - ratioX * nextWidth,
        y: anchorY - ratioY * nextHeight,
        width: nextWidth,
        height: nextHeight,
      };
      applyViewBox();
    },
    { passive: false },
  );
  svg.addEventListener("pointerdown", (event) => {
    if (event.target.matches("circle[role='button']")) return;
    drag = { x: event.clientX, y: event.clientY, viewX: viewBox.x, viewY: viewBox.y };
    svg.setPointerCapture(event.pointerId);
    svg.classList.add("is-panning");
  });
  svg.addEventListener("pointermove", (event) => {
    if (!drag) return;
    const rect = svg.getBoundingClientRect();
    viewBox.x = drag.viewX - ((event.clientX - drag.x) / rect.width) * viewBox.width;
    viewBox.y = drag.viewY - ((event.clientY - drag.y) / rect.height) * viewBox.height;
    applyViewBox();
  });
  const stopPan = () => {
    drag = null;
    svg.classList.remove("is-panning");
  };
  svg.addEventListener("pointerup", stopPan);
  svg.addEventListener("pointercancel", stopPan);
  svg.addEventListener("click", () => {
    pinnedPoint = null;
    cosineWheelTooltip.hidden = true;
  });

  cosineWheel.replaceChildren(svg, cosineWheelTooltip);
}

shareRevealButton.addEventListener("click", openShareDialog);
shareClose.addEventListener("click", () => shareDialog.close());
shareDialog.addEventListener("click", (event) => {
  if (event.target === shareDialog) shareDialog.close();
});
document.querySelectorAll("[data-share-platform]").forEach((button) => {
  button.addEventListener("click", () => shareToPlatform(button.dataset.sharePlatform));
});
downloadShareButton.addEventListener("click", () => {
  downloadShareCard();
  shareStatus.textContent = "Reveal card downloaded.";
});
copyShareTextButton.addEventListener("click", () => {
  copyShareCaption().catch(() => {
    shareStatus.textContent = "Copying is unavailable here. Use More to share the caption.";
  });
});

againButton.addEventListener("click", () => {
  results.hidden = true;
  preview.hidden = true;
  placeholder.hidden = false;
  matchButton.hidden = true;
  upload.value = "";
  selectedBlobs = [];
  currentMatches = [];
  revokePreviews();
  if (shareDialog.open) shareDialog.close();
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
  if (sharePreviewUrl) URL.revokeObjectURL(sharePreviewUrl);
});
loadConfig();
