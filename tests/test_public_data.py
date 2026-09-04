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
    def __init__(
        self,
        payload: bytes,
        *,
        max_bytes: int | None = None,
        status: int = 200,
    ) -> None:
        self.payload = payload
        self.max_bytes = max_bytes
        self.status = status
        self.position = 0

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        if self.max_bytes is not None and self.position >= self.max_bytes:
            return b""
        if self.max_bytes is not None:
            size = min(size, self.max_bytes - self.position)
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
        self.assertEqual(payload["status"], "IMMUTABLE_DATA_ARCHIVES_PUBLISHED")
        self.assertEqual(len(specs), 9)
        self.assertEqual({spec.dataset for spec in specs}, {"stats19", "cas"})
        self.assertTrue(all(spec.immutable_download_url is not None for spec in specs))
        records = payload["archive_policy"]["data_records"]
        self.assertEqual(
            records["stats19"]["version_doi"],
            "10.5281/zenodo.22290566",
        )
        self.assertEqual(
            records["cas"]["version_doi"],
            "10.5281/zenodo.22296725",
        )
        for spec in specs:
            record_id = "22290566" if spec.dataset == "stats19" else "22296725"
            self.assertIn(f"/records/{record_id}/files/", spec.immutable_download_url)

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

    def test_interrupted_download_resumes_with_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"0123456789"
            manifest = root / "manifest.json"
            write_manifest(manifest, payload, url="https://example.test/archive")
            _, specs = public_data.load_manifest(manifest)
            calls: list[str | None] = []

            def opener(request: object, **_: object) -> FakeResponse:
                range_value = (
                    request.get_header("Range")
                    if hasattr(request, "get_header")
                    else None
                )
                calls.append(range_value)
                if len(calls) == 1:
                    return FakeResponse(payload, max_bytes=4)
                self.assertEqual(range_value, "bytes=4-")
                return FakeResponse(payload[4:], status=206)

            result = public_data.download_file(
                root,
                specs[0],
                retries=1,
                retry_backoff=0,
                opener=opener,
            )
            self.assertEqual(result.status, "PASS")
            self.assertEqual((root / specs[0].relative_path).read_bytes(), payload)
            self.assertEqual(calls, [None, "bytes=4-"])

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
