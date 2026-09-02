"""Standalone tests for the public data manifest and verifier."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_DIR / "code" / "public_data.py"
SPEC = importlib.util.spec_from_file_location("public_data", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load public_data.py")
public_data = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = public_data
SPEC.loader.exec_module(public_data)


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.position = 0

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        block = self.payload[self.position : self.position + size]
        self.position += len(block)
        return block


def write_manifest(path: Path, payload: bytes, *, url: str | None) -> None:
    manifest = {
        "schema_version": "PUBLIC_DATA_V1",
        "status": "TEST",
        "datasets": {
            "sample": {
                "files": [
                    {
                        "relative_path": "data/raw/sample.bin",
                        "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "role": "analysis_input",
                        "source_url": "https://example.test/live",
                        "immutable_download_url": url,
                    }
                ]
            }
        },
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")


class PublicDataTests(unittest.TestCase):
    def test_repository_manifest_contract(self) -> None:
        payload, specs = public_data.load_manifest(
            PROJECT_DIR / "config" / "public_data_manifest.json"
        )
        self.assertEqual(payload["status"], "AWAITING_IMMUTABLE_DATA_ARCHIVE")
        self.assertEqual(len(specs), 9)
        self.assertEqual({spec.dataset for spec in specs}, {"stats19", "cas"})
        self.assertTrue(all(spec.immutable_download_url is None for spec in specs))

    def test_verify_pass_and_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"fixed snapshot\n"
            manifest = root / "manifest.json"
            write_manifest(manifest, payload, url=None)
            _, specs = public_data.load_manifest(manifest)
            target = root / specs[0].relative_path
            target.parent.mkdir(parents=True)
            target.write_bytes(payload)
            self.assertEqual(public_data.verify_file(root, specs[0]).status, "PASS")
            target.write_bytes(b"wrong snapshot")
            self.assertIn(
                public_data.verify_file(root, specs[0]).status,
                {"SIZE_MISMATCH", "SHA256_MISMATCH"},
            )

    def test_missing_immutable_url_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            write_manifest(manifest, b"fixed", url=None)
            _, specs = public_data.load_manifest(manifest)
            with self.assertRaises(public_data.DataDownloadError):
                public_data.download_file(root, specs[0])

    def test_verified_atomic_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"archived bytes"
            manifest = root / "manifest.json"
            write_manifest(manifest, payload, url="https://example.test/archive")
            _, specs = public_data.load_manifest(manifest)

            def opener(*_: object, **__: object) -> FakeResponse:
                return FakeResponse(payload)

            result = public_data.download_file(root, specs[0], opener=opener)
            self.assertEqual(result.status, "PASS")
            self.assertEqual((root / specs[0].relative_path).read_bytes(), payload)
            self.assertFalse((root / (specs[0].relative_path + ".part")).exists())

    def test_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            write_manifest(manifest, b"fixed", url=None)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["datasets"]["sample"]["files"][0]["relative_path"] = "../escape"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(public_data.DataManifestError):
                public_data.load_manifest(manifest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
