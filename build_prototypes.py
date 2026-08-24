"""Build mean class embeddings and cache them with person metadata in LanceDB."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import boto3
import lancedb
import pyarrow as pa
import requests
import torch
from PIL import Image
from requests.adapters import HTTPAdapter
from torch.utils.data import DataLoader
from urllib3.util.retry import Retry

from train import (
    FaceClassifier,
    FaceDataset,
    build_eval_transform,
    discover_images,
    load_face_classifier_state,
)

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = (
    "CelebrityDoppelganger/1.0 "
    "(https://github.com/Gabomfim/CelebrityDoppelganger; prototype metadata enrichment)"
)


def metadata_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    retry = Retry(
        total=6,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fallback_display_name(class_name: str) -> str:
    """Turn a directory slug into a readable name without inventing identity metadata."""
    normalized = unicodedata.normalize("NFC", class_name).replace("_", " ").strip()
    return re.sub(r"\s+", " ", normalized).title()


def infer_profession(description: str | None) -> str | None:
    if not description:
        return None
    match = re.search(r"\b(?:is|was) (?:an?|the) ([^.]+)", description, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def fetch_person_metadata(
    class_name: str, timeout: float = 20.0, session: requests.Session | None = None
) -> dict[str, Any]:
    """Resolve a person through English Wikipedia with biography and canonical page URL."""
    fallback = fallback_display_name(class_name)
    client = session or metadata_session()
    response = client.get(
        WIKIPEDIA_API,
        params={
            "action": "query",
            "generator": "search",
            "gsrsearch": fallback,
            "gsrnamespace": 0,
            "gsrlimit": 1,
            "prop": "extracts|info|pageprops",
            "inprop": "url",
            "exintro": 1,
            "explaintext": 1,
            "format": "json",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})
    if not pages:
        return empty_metadata(fallback)
    page = next(iter(pages.values()))
    extract = re.sub(r"\s+", " ", page.get("extract", "")).strip()
    sentences = re.split(r"(?<=[.!?])\s+", extract)
    description = " ".join(sentences[:2]) or None
    return {
        "display_name": page.get("title", fallback),
        "why_famous": description,
        "most_famous_for": description,
        "wikipedia_url": page.get("fullurl"),
        "profession": infer_profession(description),
        "wikidata_id": page.get("pageprops", {}).get("wikibase_item"),
        "metadata_status": "resolved",
    }


def empty_metadata(display_name: str, status: str = "not_found") -> dict[str, Any]:
    return {
        "display_name": display_name,
        "why_famous": None,
        "most_famous_for": None,
        "wikipedia_url": None,
        "profession": None,
        "wikidata_id": None,
        "metadata_status": status,
    }


def load_metadata_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def enrich_classes(
    class_names: list[str], cache_path: Path, offline: bool
) -> dict[str, dict[str, Any]]:
    cache = load_metadata_cache(cache_path)
    session = metadata_session()
    for index, class_name in enumerate(class_names, start=1):
        cached = cache.get(class_name, {})
        if (
            cached.get("metadata_status") == "resolved"
            and cached.get("wikipedia_url")
            and cached.get("why_famous")
            and cached.get("profession")
        ):
            continue
        if offline:
            cache[class_name] = empty_metadata(fallback_display_name(class_name), "offline")
        else:
            try:
                cache[class_name] = fetch_person_metadata(class_name, session=session)
            except (requests.RequestException, KeyError, TypeError, ValueError) as error:
                cache[class_name] = empty_metadata(
                    fallback_display_name(class_name), f"error:{type(error).__name__}"
                )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(cache, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        print(f"metadata {index}/{len(class_names)}: {class_name}")
    return cache


def build_prototype_database(
    checkpoint_path: Path,
    data_dir: Path,
    database_uri: str,
    metadata_cache_path: Path | None = None,
    photo_s3_prefix: str | None = None,
    batch_size: int = 128,
    num_workers: int = 4,
    offline_metadata: bool = False,
) -> str:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names: list[str] = state["class_names"]
    paths, labels, discovered_names = discover_images(data_dir)
    if discovered_names != class_names:
        raise ValueError("Checkpoint class order does not match the current dataset")
    model = FaceClassifier(len(class_names), int(state["config"]["embedding_dim"]), pretrained=None)
    load_face_classifier_state(model, state["model_state_dict"])
    model.to(device).eval()
    loader = DataLoader(
        FaceDataset(paths, labels, build_eval_transform()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    sums: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = defaultdict(int)
    with torch.inference_mode():
        for images, batch_labels in loader:
            embeddings, _ = model(images.to(device, non_blocking=True))
            for embedding, label in zip(embeddings.cpu(), batch_labels.tolist(), strict=True):
                sums[label] = sums.get(label, torch.zeros_like(embedding)) + embedding
                counts[label] += 1
    cache_path = metadata_cache_path or Path("models/person_metadata.json")
    metadata = enrich_classes(class_names, cache_path, offline_metadata)
    records = []
    for label, class_name in enumerate(class_names):
        prototype = torch.nn.functional.normalize(sums[label] / counts[label], dim=0)
        records.append(
            {
                "class_id": label,
                "class_name": class_name,
                "vector": prototype.tolist(),
                "image_count": counts[label],
                "photo_s3_uri": upload_reference_photo(
                    paths[labels.index(label)], class_name, photo_s3_prefix
                ),
                **metadata[class_name],
            }
        )
    if not database_uri.startswith("s3://"):
        Path(database_uri).mkdir(parents=True, exist_ok=True)
    database = lancedb.connect(database_uri)
    database.create_table("person_prototypes", data=pa.Table.from_pylist(records), mode="overwrite")
    manifest = {
        "format_version": 1,
        "table": "person_prototypes",
        "checkpoint": checkpoint_path.name,
        "class_count": len(records),
        "embedding_dimension": len(records[0]["vector"]),
    }
    cache_path.with_name("prototype_manifest.json").write_text(
        json.dumps({**manifest, "database_uri": database_uri}, indent=2), encoding="utf-8"
    )
    return database_uri


def upload_reference_photo(
    source_path: Path, class_name: str, photo_s3_prefix: str | None
) -> str | None:
    if not photo_s3_prefix:
        return None
    if not photo_s3_prefix.startswith("s3://"):
        raise ValueError("photo_s3_prefix must be an s3:// URI")
    location = photo_s3_prefix.removeprefix("s3://").rstrip("/")
    bucket, prefix = location.split("/", 1)
    key = f"{prefix}/{class_name}{source_path.suffix.lower()}"
    boto3.client("s3").upload_file(
        str(source_path),
        bucket,
        key,
        ExtraArgs={"ContentType": Image.open(source_path).get_format_mimetype()},
    )
    return f"s3://{bucket}/{key}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/source_dataset"))
    parser.add_argument("--database-uri", default="models/prototypes.lancedb")
    parser.add_argument("--metadata-cache", type=Path, default=Path("models/person_metadata.json"))
    parser.add_argument("--photo-s3-prefix", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--offline-metadata", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = build_prototype_database(
        arguments.checkpoint,
        arguments.data_dir,
        arguments.database_uri,
        arguments.metadata_cache,
        arguments.photo_s3_prefix,
        arguments.batch_size,
        arguments.num_workers,
        arguments.offline_metadata,
    )
    print(f"Prototype database saved to {result}")
