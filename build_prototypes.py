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
from torch.utils.data import DataLoader

from train import (
    FaceClassifier,
    FaceDataset,
    build_eval_transform,
    discover_images,
    load_face_classifier_state,
)

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "CelebrityDoppelganger/1.0 (prototype metadata enrichment)"


def fallback_display_name(class_name: str) -> str:
    """Turn a directory slug into a readable name without inventing identity metadata."""
    normalized = unicodedata.normalize("NFC", class_name).replace("_", " ").strip()
    return re.sub(r"\s+", " ", normalized).title()


def fetch_person_metadata(class_name: str, timeout: float = 15.0) -> dict[str, Any]:
    """Resolve one identity through Wikidata's search API and English Wikipedia sitelink."""
    fallback = fallback_display_name(class_name)
    response = requests.get(
        WIKIDATA_API,
        params={
            "action": "wbsearchentities",
            "search": fallback,
            "language": "en",
            "uselang": "en",
            "type": "item",
            "limit": 5,
            "format": "json",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    results = response.json().get("search", [])
    if not results:
        return empty_metadata(fallback)
    exact = next(
        (item for item in results if item.get("label", "").casefold() == fallback.casefold()),
        results[0],
    )
    entity_id = exact["id"]
    entity_response = requests.get(
        WIKIDATA_API,
        params={
            "action": "wbgetentities",
            "ids": entity_id,
            "props": "labels|descriptions|claims|sitelinks",
            "languages": "en",
            "sitefilter": "enwiki",
            "format": "json",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    entity_response.raise_for_status()
    entity = entity_response.json()["entities"][entity_id]
    description = entity.get("descriptions", {}).get("en", {}).get("value")
    title = entity.get("sitelinks", {}).get("enwiki", {}).get("title")
    occupation_ids = [
        claim.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")
        for claim in entity.get("claims", {}).get("P106", [])
    ]
    notable_work_ids = [
        claim.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")
        for claim in entity.get("claims", {}).get("P800", [])
    ]
    occupations = resolve_entity_labels([item for item in occupation_ids if item], timeout)
    notable_works = resolve_entity_labels([item for item in notable_work_ids if item], timeout)
    wikipedia_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}" if title else None
    return {
        "display_name": entity.get("labels", {}).get("en", {}).get("value", fallback),
        "why_famous": description,
        "most_famous_for": ", ".join(notable_works) or description,
        "wikipedia_url": wikipedia_url,
        "profession": ", ".join(occupations) or None,
        "wikidata_id": entity_id,
        "metadata_status": "resolved",
    }


def resolve_entity_labels(entity_ids: list[str], timeout: float) -> list[str]:
    if not entity_ids:
        return []
    response = requests.get(
        WIKIDATA_API,
        params={
            "action": "wbgetentities",
            "ids": "|".join(entity_ids[:50]),
            "props": "labels",
            "languages": "en",
            "format": "json",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    entities = response.json().get("entities", {})
    return [
        entities[item]["labels"]["en"]["value"]
        for item in entity_ids
        if item in entities and "en" in entities[item].get("labels", {})
    ]


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
    for index, class_name in enumerate(class_names, start=1):
        if class_name in cache:
            continue
        if offline:
            cache[class_name] = empty_metadata(fallback_display_name(class_name), "offline")
        else:
            try:
                cache[class_name] = fetch_person_metadata(class_name)
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
