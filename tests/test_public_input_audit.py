"""Standalone tests for the public STATS19 input audit."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_DIR / "code"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


public_data = load_module("public_data", CODE_DIR / "public_data.py")
public_input_audit = load_module(
    "public_input_audit", CODE_DIR / "public_input_audit.py"
)


def annual_payload(year: int, severity: int) -> bytes:
    header = [
        "collision_index",
        "collision_year",
        "collision_severity",
        "extra_field",
    ]
    row = [f"id-{year}", str(year), str(severity), "x"]
    return (",".join(header) + "\n" + ",".join(row) + "\n").encode("utf-8")


def make_fixture(root: Path) -> Path:
    files = []
    for offset, year in enumerate(public_input_audit.ANALYSIS_YEARS):
        payload = annual_payload(year, offset % 3 + 1)
        relative_path = f"data/raw/collisions/collision_{year}.csv"
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        files.append(
            {
                "relative_path": relative_path,
                "expected_rows": 1,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "role": "analysis_input",
                "source_url": "https://example.test/live",
                "immutable_download_url": "https://example.test/archive",
            }
        )
    manifest = {
        "schema_version": "PUBLIC_DATA_V1",
        "status": "TEST",
        "datasets": {
            "stats19": {
                "analysis_years": list(public_input_audit.ANALYSIS_YEARS),
                "files": files,
            }
        },
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class PublicInputAuditTests(unittest.TestCase):
    def test_repository_manifest_freezes_annual_row_counts(self) -> None:
        annual = public_input_audit.load_annual_inputs(
            PROJECT_DIR / "config" / "public_data_manifest.json"
        )
        self.assertEqual(
            [item.expected_rows for item in annual],
            [122635, 117536, 91199, 101087, 106004, 104258, 100927],
        )

    def test_builds_d1_compatible_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_fixture(root)
            output = root / "logs" / "d1_initial_audit.csv"
            audit = public_input_audit.build_initial_audit(root, manifest, output)
            self.assertEqual(tuple(audit.columns), public_input_audit.AUDIT_COLUMNS)
            self.assertEqual(audit["rows"].tolist(), [1] * 7)
            self.assertTrue(audit["extraction_row_match"].all())
            self.assertTrue(audit["year_values_correct"].all())
            with output.open(encoding="utf-8-sig", newline="") as handle:
                written = list(csv.DictReader(handle))
            self.assertEqual(len(written), 7)
            self.assertEqual(written[0]["year"], "2018")

    def test_hash_mismatch_stops_before_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_fixture(root)
            target = root / "data" / "raw" / "collisions" / "collision_2024.csv"
            target.write_bytes(b"changed")
            with self.assertRaises(public_input_audit.PublicInputAuditError):
                public_input_audit.build_initial_audit(
                    root, manifest, root / "logs" / "audit.csv"
                )

    def test_wrong_year_stops_after_verified_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = make_fixture(root)
            payload = annual_payload(2023, 1)
            target = root / "data" / "raw" / "collisions" / "collision_2024.csv"
            target.write_bytes(payload)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            item = manifest["datasets"]["stats19"]["files"][-1]
            item["size_bytes"] = len(payload)
            item["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(public_input_audit.PublicInputAuditError):
                public_input_audit.build_initial_audit(
                    root, manifest_path, root / "logs" / "audit.csv"
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
