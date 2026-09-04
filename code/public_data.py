"""Download and verify immutable public reconstruction inputs.

Mutable upstream URLs are recorded for provenance but are never used as an
exact-snapshot fallback. Downloads use version-specific Zenodo record URLs and
are accepted only after byte-size and SHA-256 verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_DIR / "config" / "public_data_manifest.json"
BLOCK_SIZE = 8 * 1024 * 1024
DEFAULT_RETRIES = 5
DEFAULT_RETRY_BACKOFF = 2.0
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class DataManifestError(ValueError):
    """Raised when the public data manifest violates its contract."""


class DataDownloadError(RuntimeError):
    """Raised when an immutable input cannot be downloaded safely."""


@dataclass(frozen=True)
class FileSpec:
    dataset: str
    relative_path: str
    size_bytes: int
    sha256: str
    role: str
    source_url: str
    immutable_download_url: str | None


@dataclass(frozen=True)
class Verification:
    spec: FileSpec
    status: str
    observed_size_bytes: int | None
    observed_sha256: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise DataManifestError("relative_path must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise DataManifestError(f"Unsafe relative_path: {value!r}")
    return value


def _validate_https_or_null(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or urlparse(value).scheme != "https":
        raise DataManifestError(f"{field} must be an HTTPS URL or null")
    return value


def load_manifest(path: Path = DEFAULT_MANIFEST) -> tuple[dict[str, Any], list[FileSpec]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "PUBLIC_DATA_V1":
        raise DataManifestError("Unexpected public data manifest version")
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise DataManifestError("Manifest must define at least one dataset")

    specs: list[FileSpec] = []
    seen_paths: set[str] = set()
    for dataset, definition in datasets.items():
        if not isinstance(dataset, str) or not dataset:
            raise DataManifestError("Dataset names must be non-empty strings")
        files = definition.get("files") if isinstance(definition, dict) else None
        if not isinstance(files, list) or not files:
            raise DataManifestError(f"Dataset {dataset!r} has no files")
        for raw in files:
            if not isinstance(raw, dict):
                raise DataManifestError(f"Dataset {dataset!r} contains a non-object file")
            relative_path = _validate_relative_path(raw.get("relative_path"))
            if relative_path in seen_paths:
                raise DataManifestError(f"Duplicate relative_path: {relative_path}")
            seen_paths.add(relative_path)
            size = raw.get("size_bytes")
            if not isinstance(size, int) or size <= 0:
                raise DataManifestError(f"Invalid size for {relative_path}")
            sha256 = raw.get("sha256")
            if not isinstance(sha256, str) or not SHA256_PATTERN.fullmatch(sha256):
                raise DataManifestError(f"Invalid SHA-256 for {relative_path}")
            role = raw.get("role")
            if not isinstance(role, str) or not role:
                raise DataManifestError(f"Invalid role for {relative_path}")
            source_url = _validate_https_or_null(raw.get("source_url"), "source_url")
            if source_url is None:
                raise DataManifestError(f"source_url is required for {relative_path}")
            specs.append(
                FileSpec(
                    dataset=dataset,
                    relative_path=relative_path,
                    size_bytes=size,
                    sha256=sha256,
                    role=role,
                    source_url=source_url,
                    immutable_download_url=_validate_https_or_null(
                        raw.get("immutable_download_url"),
                        "immutable_download_url",
                    ),
                )
            )
    return payload, specs


def select_specs(specs: list[FileSpec], datasets: list[str]) -> list[FileSpec]:
    available = {spec.dataset for spec in specs}
    selected = available if not datasets or "all" in datasets else set(datasets)
    unknown = selected - available
    if unknown:
        raise DataManifestError(
            f"Unknown dataset(s): {', '.join(sorted(unknown))}; "
            f"available: {', '.join(sorted(available))}"
        )
    return [spec for spec in specs if spec.dataset in selected]


def local_path(root: Path, spec: FileSpec) -> Path:
    root = root.resolve()
    candidate = (root / Path(*PurePosixPath(spec.relative_path).parts)).resolve()
    if candidate != root and root not in candidate.parents:
        raise DataManifestError(f"Path escapes project root: {spec.relative_path}")
    return candidate


def verify_file(root: Path, spec: FileSpec) -> Verification:
    path = local_path(root, spec)
    if not path.is_file():
        return Verification(spec, "MISSING", None, None)
    size = path.stat().st_size
    if size != spec.size_bytes:
        return Verification(spec, "SIZE_MISMATCH", size, None)
    digest = sha256_file(path)
    status = "PASS" if digest == spec.sha256 else "SHA256_MISMATCH"
    return Verification(spec, status, size, digest)


def verify_files(root: Path, specs: list[FileSpec]) -> list[Verification]:
    return [verify_file(root, spec) for spec in specs]


def download_file(
    root: Path,
    spec: FileSpec,
    *,
    replace: bool = False,
    timeout: float = 120.0,
    retries: int = DEFAULT_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    opener: Callable[..., Any] = urlopen,
) -> Verification:
    current = verify_file(root, spec)
    if current.status == "PASS":
        return current
    if spec.immutable_download_url is None:
        raise DataDownloadError(
            f"No immutable archive URL is published for {spec.relative_path}. "
            "Do not substitute the mutable source URL."
        )
    destination = local_path(root, spec)
    if destination.exists() and not replace:
        raise DataDownloadError(
            f"Existing file failed verification: {destination}. "
            "Use --replace only after checking that this is not your only copy."
        )

    if retries < 0:
        raise DataDownloadError("retries must be non-negative")
    if retry_backoff < 0:
        raise DataDownloadError("retry_backoff must be non-negative")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        # Keep incomplete bytes only for resume. A complete partial is either
        # promoted after verification or discarded before another request.
        if partial.exists():
            partial_size = partial.stat().st_size
            if partial_size == spec.size_bytes:
                if sha256_file(partial) == spec.sha256:
                    os.replace(partial, destination)
                    return verify_file(root, spec)
                partial.unlink()
            elif partial_size > spec.size_bytes:
                partial.unlink()

        offset = partial.stat().st_size if partial.exists() else 0
        digest = hashlib.sha256()
        if offset:
            with partial.open("rb") as existing:
                for block in iter(lambda: existing.read(BLOCK_SIZE), b""):
                    digest.update(block)

        headers = {"User-Agent": "stats19-temporal-eval-public-reproduction/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = Request(spec.immutable_download_url, headers=headers)

        try:
            with opener(request, timeout=timeout) as response:
                status = getattr(response, "status", None)
                if status is None and hasattr(response, "getcode"):
                    status = response.getcode()

                # Never append a full response to a partial file if a server
                # ignores the Range request.
                if offset and status not in (None, 206):
                    partial.unlink(missing_ok=True)
                    if attempt < retries:
                        time.sleep(retry_backoff * (2**attempt))
                        continue
                    raise DataDownloadError(
                        f"Server did not honour Range for {spec.relative_path}"
                    )

                mode = "ab" if offset else "wb"
                with partial.open(mode) as output:
                    for block in iter(lambda: response.read(BLOCK_SIZE), b""):
                        output.write(block)
                        digest.update(block)

            size = partial.stat().st_size
            observed = digest.hexdigest()
            if size == spec.size_bytes and observed == spec.sha256:
                os.replace(partial, destination)
                return verify_file(root, spec)

            raise DataDownloadError(
                f"Downloaded bytes incomplete or failed verification for "
                f"{spec.relative_path}: size={size}/{spec.size_bytes}, "
                f"sha256={observed}/{spec.sha256}"
            )
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_backoff * (2**attempt))

    raise DataDownloadError(
        f"Download failed after {retries + 1} attempt(s) for "
        f"{spec.relative_path}; a partial file may be retained for resume: "
        f"{partial}"
    ) from last_error


def print_manifest_summary(payload: dict[str, Any], specs: list[FileSpec]) -> None:
    print(f"Manifest status: {payload['status']}")
    for dataset in sorted({spec.dataset for spec in specs}):
        group = [spec for spec in specs if spec.dataset == dataset]
        total = sum(spec.size_bytes for spec in group)
        ready = sum(spec.immutable_download_url is not None for spec in group)
        print(
            f"{dataset}: files={len(group)}, bytes={total}, "
            f"immutable_urls={ready}/{len(group)}"
        )
        for spec in group:
            print(f"  {spec.relative_path}  {spec.sha256}")


def print_verification(results: list[Verification]) -> None:
    for result in results:
        print(
            f"[{result.status}] {result.spec.dataset}: "
            f"{result.spec.relative_path}"
        )
    passed = sum(result.status == "PASS" for result in results)
    print(f"Verified: {passed}/{len(results)} files")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=PROJECT_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("list", "verify"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--dataset", action="append", default=[])
    download = subparsers.add_parser("download")
    download.add_argument("--dataset", action="append", default=[])
    download.add_argument("--replace", action="store_true")
    download.add_argument("--timeout", type=float, default=120.0)
    download.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    download.add_argument(
        "--retry-backoff", type=float, default=DEFAULT_RETRY_BACKOFF
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload, all_specs = load_manifest(args.manifest)
    specs = select_specs(all_specs, args.dataset)
    if args.command == "list":
        print_manifest_summary(payload, specs)
        return 0
    if args.command == "verify":
        results = verify_files(args.root, specs)
        print_verification(results)
        return 0 if all(result.status == "PASS" for result in results) else 1

    results: list[Verification] = []
    for spec in specs:
        result = download_file(
            args.root,
            spec,
            replace=args.replace,
            timeout=args.timeout,
            retries=args.retries,
            retry_backoff=args.retry_backoff,
        )
        print(f"[{result.status}] {spec.dataset}: {spec.relative_path}")
        results.append(result)
    print(f"Downloaded and verified: {len(results)}/{len(specs)} files")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DataManifestError, DataDownloadError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
