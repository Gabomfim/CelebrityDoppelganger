"""Refresh celebrity biographies and occupations without recomputing embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import lancedb
import pyarrow as pa

from build_prototypes import enrich_classes


def refresh_metadata(database_uri: str, cache_path: Path) -> int:
    database = lancedb.connect(database_uri)
    table = database.open_table("person_prototypes")
    records = table.to_arrow().to_pylist()
    class_names = [record["class_name"] for record in records]
    metadata = enrich_classes(class_names, cache_path, offline=False)
    for record in records:
        record.update(metadata[record["class_name"]])
    database.create_table("person_prototypes", pa.Table.from_pylist(records), mode="overwrite")
    return len(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-uri", required=True)
    parser.add_argument("--metadata-cache", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    count = refresh_metadata(arguments.database_uri, arguments.metadata_cache)
    print(f"Refreshed metadata for {count} celebrity prototypes")
