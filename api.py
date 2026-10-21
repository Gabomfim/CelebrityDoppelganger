"""FastAPI service for privacy-preserving celebrity face matching."""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
from base64 import b64decode
from binascii import Error as Base64Error
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
import lancedb
import torch
from facenet_pytorch import MTCNN
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from train import FaceClassifier, build_eval_transform, load_face_classifier_state

PROJECT_ROOT = Path(__file__).parent
STATIC_DIR = PROJECT_ROOT / "web" / "static"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
PROTOTYPE_IMAGE_COUNT = 3
MAX_PROTOTYPE_REQUEST_BYTES = 42 * 1024 * 1024
MAX_ANALYTICS_REQUEST_BYTES = 4096
MAX_CONCURRENT_MATCHES = max(1, int(os.getenv("MAX_CONCURRENT_MATCHES", "1")))
GITHUB_URL = "https://github.com/Gabomfim/CelebrityDoppelganger"
LINKEDIN_URL = "https://www.linkedin.com/in/gabrielabsilveira/"
ANALYTICS_EVENTS = {
    "page_view",
    "engaged",
    "camera_started",
    "camera_ready",
    "camera_failed",
    "photobooth_completed",
    "upload_selected",
    "match_submitted",
    "match_succeeded",
    "match_failed",
    "share_opened",
    "share_completed",
    "session_end",
}
ANALYTICS_CATEGORIES = {
    "browser": {
        "Chrome",
        "Edge",
        "Firefox",
        "Safari",
        "LinkedIn",
        "Instagram",
        "Facebook",
        "Other WebView",
        "Other",
    },
    "device": {"mobile", "tablet", "desktop"},
    "os": {"Android", "iOS", "Linux", "macOS", "Windows", "Other"},
}
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9-]{16,64}$")
PORTUGUESE_COUNTRIES = {"AO", "BR", "CV", "GQ", "GW", "MO", "MZ", "PT", "ST", "TL"}


class FaceNotDetectedError(ValueError):
    pass


class CelebrityMatcher:
    def __init__(self, checkpoint: str, database_uri: str, dataset_dir: Path | None) -> None:
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        state = self._load_checkpoint(checkpoint)
        self.class_names: list[str] = state["class_names"]
        self.model = FaceClassifier(
            len(self.class_names), int(state["config"]["embedding_dim"]), pretrained=None
        )
        load_face_classifier_state(self.model, state["model_state_dict"])
        self.model.to(self.device).eval()
        self.detector = MTCNN(keep_all=False, device=self.device, post_process=False)
        self.transform = build_eval_transform()
        self.table = lancedb.connect(database_uri).open_table("person_prototypes")
        prototype_rows = (
            self.table.to_arrow().select(["class_name", "display_name", "vector"]).to_pylist()
        )
        self.prototype_metadata = [
            {"class_name": row["class_name"], "name": row["display_name"]} for row in prototype_rows
        ]
        self.prototype_matrix = torch.nn.functional.normalize(
            torch.tensor([row["vector"] for row in prototype_rows], dtype=torch.float32), dim=1
        )
        self.dataset_dir = dataset_dir.resolve() if dataset_dir else None
        self.photos = self._index_photos()
        self.s3_photos = {
            row["class_name"]: row["photo_s3_uri"]
            for row in self.table.to_arrow().select(["class_name", "photo_s3_uri"]).to_pylist()
            if row.get("photo_s3_uri")
        }

    def _load_checkpoint(self, checkpoint: str) -> dict[str, Any]:
        if not checkpoint.startswith("s3://"):
            return torch.load(checkpoint, map_location=self.device, weights_only=False)
        bucket, key = checkpoint.removeprefix("s3://").split("/", 1)
        payload = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        return torch.load(io.BytesIO(payload), map_location=self.device, weights_only=False)

    def _index_photos(self) -> dict[str, Path]:
        photos: dict[str, Path] = {}
        if self.dataset_dir is None or not self.dataset_dir.is_dir():
            return photos
        for class_name in self.class_names:
            directory = (self.dataset_dir / class_name).resolve()
            if not directory.is_relative_to(self.dataset_dir) or not directory.is_dir():
                continue
            photo = next(
                (
                    path
                    for path in sorted(directory.iterdir())
                    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                ),
                None,
            )
            if photo:
                photos[class_name] = photo
        return photos

    def read_photo(self, class_name: str) -> tuple[bytes, str] | None:
        if class_name in self.photos:
            path = self.photos[class_name]
            media_type = Image.MIME.get(Image.open(path).format, "image/jpeg")
            return path.read_bytes(), media_type
        uri = self.s3_photos.get(class_name)
        if not uri:
            return None
        bucket, key = uri.removeprefix("s3://").split("/", 1)
        result = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        return result["Body"].read(), result.get("ContentType", "image/jpeg")

    def embed(self, image_bytes: bytes) -> list[float]:
        try:
            with Image.open(io.BytesIO(image_bytes)) as source:
                image = source.convert("RGB")
        except (UnidentifiedImageError, OSError) as error:
            raise ValueError("The uploaded file is not a supported image") from error
        boxes, probabilities = self.detector.detect(image)
        if boxes is None or probabilities is None or probabilities[0] < 0.90:
            raise FaceNotDetectedError(
                "No clear face was detected. Face the camera, improve the lighting, and try again."
            )
        box = boxes[0]
        width, height = image.size
        margin = 0.12 * max(box[2] - box[0], box[3] - box[1])
        crop_box = (
            max(0, int(box[0] - margin)),
            max(0, int(box[1] - margin)),
            min(width, int(box[2] + margin)),
            min(height, int(box[3] + margin)),
        )
        face = self.transform(image.crop(crop_box)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            embedding, _ = self.model(face)
        return embedding[0].cpu().tolist()

    def _match_vector(self, vector: list[float]) -> list[dict[str, Any]]:
        neighbors = self.table.search(vector).metric("cosine").limit(3).to_list()
        matches = []
        for neighbor in neighbors[:3]:
            class_name = neighbor["class_name"]
            distance = float(neighbor.get("_distance", 0.0))
            matches.append(
                {
                    "name": neighbor["display_name"],
                    "class_name": class_name,
                    "similarity": round(max(0.0, min(1.0, 1.0 - distance)) * 100, 1),
                    "photo_url": f"/api/celebrity-photo/{class_name}",
                    "why_famous": neighbor.get("why_famous"),
                    "most_famous_for": neighbor.get("most_famous_for"),
                    "profession": neighbor.get("profession"),
                    "wikipedia_url": neighbor.get("wikipedia_url"),
                }
            )
        return matches

    def match(self, image_bytes: bytes) -> list[dict[str, Any]]:
        return self._match_vector(self.embed(image_bytes))

    def match_prototype(self, images: list[bytes]) -> dict[str, list[dict[str, Any]]]:
        if len(images) != PROTOTYPE_IMAGE_COUNT:
            raise ValueError(f"Exactly {PROTOTYPE_IMAGE_COUNT} images are required")
        embeddings = torch.tensor([self.embed(image) for image in images])
        prototype = torch.nn.functional.normalize(embeddings.mean(dim=0), dim=0)
        matches = self._match_vector(prototype.tolist())
        similarities = torch.mv(self.prototype_matrix, prototype.cpu()).tolist()
        wheel = [
            {
                **metadata,
                "similarity": round(max(-1.0, min(1.0, similarity)), 6),
            }
            for metadata, similarity in zip(self.prototype_metadata, similarities, strict=True)
        ]
        return {"matches": matches, "cosine_wheel": wheel}


def create_matcher() -> CelebrityMatcher:
    checkpoint = os.getenv("MODEL_CHECKPOINT", "models/celebrity_face_classifier.pt")
    database_uri = os.getenv("PROTOTYPE_DATABASE_URI", "models/prototypes.lancedb")
    dataset_value = os.getenv("SOURCE_DATASET_DIR")
    dataset_dir = Path(dataset_value) if dataset_value else None
    return CelebrityMatcher(checkpoint, database_uri, dataset_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.inference_slots = asyncio.Semaphore(MAX_CONCURRENT_MATCHES)
    try:
        app.state.matcher = create_matcher()
        app.state.startup_error = None
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as error:
        app.state.matcher = None
        app.state.startup_error = str(error)
    yield
    app.state.matcher = None


async def run_bounded_inference(request: Request, operation):
    """Reject excess inference instead of building an unbounded, costly request queue."""
    slots: asyncio.Semaphore = request.app.state.inference_slots
    try:
        await asyncio.wait_for(slots.acquire(), timeout=0.01)
    except TimeoutError as error:
        raise HTTPException(
            status_code=429,
            detail="The face studio is busy. Please wait a moment and try again.",
            headers={"Retry-After": "10"},
        ) from error
    try:
        return await asyncio.to_thread(operation)
    finally:
        slots.release()


app = FastAPI(
    title="Celebrity Twin",
    description="Find celebrity lookalikes without retaining user images.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/api/celebrity-photo/"):
        response.headers["Cache-Control"] = "public, max-age=86400"
    elif request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    else:
        response.headers["Cache-Control"] = "public, max-age=300"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(self)"
    return response


@app.get("/api/config")
def config(request: Request) -> dict[str, str | None]:
    country = request.headers.get("cloudfront-viewer-country", "").upper()
    accept_language = request.headers.get("accept-language", "").lower()
    suggested_locale = (
        "pt-BR" if country in PORTUGUESE_COUNTRIES or "pt" in accept_language else "en"
    )
    return {
        "github_url": GITHUB_URL,
        "linkedin_url": os.getenv("LINKEDIN_URL", LINKEDIN_URL),
        "suggested_locale": suggested_locale,
    }


@app.get("/api/health")
def health(request: Request) -> JSONResponse:
    ready = request.app.state.matcher is not None
    return JSONResponse(
        {"status": "ready" if ready else "model_unavailable"}, status_code=200 if ready else 503
    )


@app.post("/api/analytics", status_code=204)
async def analytics(request: Request) -> Response:
    """Record allowlisted, anonymous product events without request identifiers."""
    if request.headers.get("content-type", "").split(";", 1)[0] != "application/json":
        raise HTTPException(status_code=415, detail="Analytics events must be JSON")
    body = await request.body()
    if not body or len(body) > MAX_ANALYTICS_REQUEST_BYTES:
        raise HTTPException(status_code=400, detail="Invalid analytics event")
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, ValueError) as error:
        raise HTTPException(status_code=400, detail="Invalid analytics event") from error
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid analytics event")
    event = payload.get("event")
    session_id = payload.get("session_id")
    if (
        event not in ANALYTICS_EVENTS
        or not isinstance(session_id, str)
        or not SESSION_ID_PATTERN.fullmatch(session_id)
    ):
        raise HTTPException(status_code=400, detail="Invalid analytics event")
    record: dict[str, Any] = {
        "record_type": "celebrity_twin_analytics",
        "timestamp": datetime.now(UTC).isoformat(),
        "event": event,
        "session_id": session_id,
        "test_event": int(session_id.startswith("test-")),
    }
    for field, allowed in ANALYTICS_CATEGORIES.items():
        value = payload.get(field)
        if value in allowed:
            record[field] = value
    for field in ("duration_ms", "engaged"):
        value = payload.get(field)
        if isinstance(value, int) and 0 <= value <= 3_600_000:
            record[field] = value
    error_code = payload.get("error_code")
    if isinstance(error_code, str) and re.fullmatch(r"[a-z0-9_]{1,40}", error_code):
        record["error_code"] = error_code
    print(json.dumps(record, separators=(",", ":")), flush=True)
    return Response(status_code=204)


@app.post("/api/match")
async def match(request: Request) -> dict[str, Any]:
    matcher: CelebrityMatcher | None = request.app.state.matcher
    if matcher is None:
        raise HTTPException(status_code=503, detail="The matching model is not available yet")
    content_type = request.headers.get("content-type", "").split(";", 1)[0]
    if content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image must be smaller than 10 MB")
    if not body:
        raise HTTPException(status_code=400, detail="The image is empty")
    try:
        image_bytes = bytes(body)
        results = await run_bounded_inference(request, lambda: matcher.match(image_bytes))
    except FaceNotDetectedError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        body.clear()
    return {"matches": results, "neighbor_count": 3, "image_stored": False}


@app.post("/api/match-prototype")
async def match_prototype(request: Request) -> dict[str, Any]:
    matcher: CelebrityMatcher | None = request.app.state.matcher
    if matcher is None:
        raise HTTPException(status_code=503, detail="The matching model is not available yet")
    if request.headers.get("content-type", "").split(";", 1)[0] != "application/json":
        raise HTTPException(status_code=415, detail="Send the three images as JSON")
    body = bytearray()
    images: list[bytes] = []
    try:
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_PROTOTYPE_REQUEST_BYTES:
                raise HTTPException(status_code=413, detail="The selfie set is too large")
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, ValueError) as error:
            raise HTTPException(status_code=400, detail="The selfie set is invalid") from error
        encoded_images = payload.get("images") if isinstance(payload, dict) else None
        if not isinstance(encoded_images, list) or len(encoded_images) != PROTOTYPE_IMAGE_COUNT:
            raise HTTPException(
                status_code=400,
                detail=f"Exactly {PROTOTYPE_IMAGE_COUNT} selfies are required",
            )
        for encoded in encoded_images:
            if not isinstance(encoded, str):
                raise HTTPException(status_code=400, detail="The selfie set is invalid")
            value = encoded.split(",", 1)[-1]
            try:
                image = b64decode(value, validate=True)
            except (Base64Error, ValueError) as error:
                raise HTTPException(status_code=400, detail="The selfie set is invalid") from error
            if not image or len(image) > MAX_IMAGE_BYTES:
                raise HTTPException(
                    status_code=413, detail="Each selfie must be smaller than 10 MB"
                )
            images.append(image)
        inference_images = list(images)
        match_result = await run_bounded_inference(
            request, lambda: matcher.match_prototype(inference_images)
        )
    except FaceNotDetectedError as error:
        raise HTTPException(
            status_code=422,
            detail="A clear face was not detected in all three selfies. Please retake the set.",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        body.clear()
        images.clear()
    if isinstance(match_result, dict):
        results = match_result["matches"]
        cosine_wheel = match_result.get("cosine_wheel", [])
    else:
        results = match_result
        cosine_wheel = []
    return {
        "matches": results,
        "cosine_wheel": cosine_wheel,
        "neighbor_count": 3,
        "prototype_image_count": PROTOTYPE_IMAGE_COUNT,
        "image_stored": False,
    }


@app.get("/api/celebrity-photo/{class_name}")
def celebrity_photo(class_name: str, request: Request) -> Response:
    matcher: CelebrityMatcher | None = request.app.state.matcher
    if matcher is None:
        raise HTTPException(status_code=404, detail="Celebrity photo not found")
    photo = matcher.read_photo(class_name)
    if photo is None:
        raise HTTPException(status_code=404, detail="Celebrity photo not found")
    return Response(
        content=photo[0], media_type=photo[1], headers={"Cache-Control": "public, max-age=86400"}
    )


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="web")
