"""Normalize the offline CAS manifest to the frozen audit contract."""

from __future__ import annotations

import csv
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_DIR / "logs" / "cas" / "cas_source_manifest.csv"


def main() -> None:
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CAS source manifest has no header")
        rows = list(reader)
        fields = list(reader.fieldnames)

    for row in rows:
        path = str(row.get("local_path", ""))
        if path.endswith("cas_injury_2022_2025_snapshot.csv.gz"):
            row["artifact_role"] = "official_api_attribute_snapshot"
        elif path.endswith("cas_injury_2022_2025_snapshot.jsonl.gz"):
            row["artifact_role"] = "lossless_attribute_snapshot"

    roles = [row.get("artifact_role", "") for row in rows]
    if roles.count("official_api_attribute_snapshot") != 1:
        raise ValueError("CAS public manifest must have one CSV snapshot entry")
    if roles.count("lossless_attribute_snapshot") != 1:
        raise ValueError("CAS public manifest must have one JSONL snapshot entry")
    with MANIFEST.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("CAS_PUBLIC_MANIFEST=COMPATIBLE_WITH_FROZEN_AUDIT_TEST")


if __name__ == "__main__":
    main()
