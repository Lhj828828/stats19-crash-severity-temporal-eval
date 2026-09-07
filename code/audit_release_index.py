"""Read-only pre-release audit of the exact staged Git blobs."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[1]
SEALED_ROOTS = tuple(
    f"{folder}/{branch}/"
    for folder in ("config", "logs", "results")
    for branch in ("final_analysis", "random_reference_revision", "stats19_feature_revision")
)
TEXT_SUFFIXES = {".py", ".ps1", ".md", ".txt", ".json", ".csv", ".cff", ".yaml", ".yml", ".toml"}
HOME = re.compile(r"[A-Z]:[\\/]+Users[\\/]+Administrator|/(?:Users|home)/[^/\s]+", re.I)
SECRET = re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----")
FORBIDDEN = re.compile(
    r"^(?:models/|data/(?:raw/(?:collisions|cas|downloads)|interim|processed)/)"
    r"|/(?:predictions|validation_predictions|shap_values)/"
    r"|\.(?:joblib|pkl|npz|npy|csv\.gz|jsonl\.gz|zip|7z)$"
)


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def main() -> None:
    records = []
    for row in git("ls-files", "--stage", "-z").split(b"\0"):
        if not row:
            continue
        meta, name = row.split(b"\t", 1)
        mode, oid, stage = meta.decode().split()
        if stage != "0" or mode not in {"100644", "100755"}:
            raise ValueError("Unexpected index mode or unresolved merge")
        records.append((name.decode("utf-8"), oid))

    # Batch-read the actual index objects, not normalized working-tree views.
    payload = subprocess.check_output(
        ["git", "cat-file", "--batch"], cwd=ROOT,
        input="".join(oid + "\n" for _, oid in records).encode("ascii"),
    )
    cursor = 0
    report = {"status": "RUNNING", "files": len(records), "bytes": 0,
              "sealed_bytes_checked": 0, "python_parsed": 0, "json_parsed": 0,
              "text_scanned": 0, "nonsealed_worktree_byte_differences": [], "failures": []}
    for name, oid in records:
        end = payload.index(b"\n", cursor)
        found, kind, size = payload[cursor:end].decode().split()
        size = int(size)
        data = payload[end + 1:end + 1 + size]
        cursor = end + 2 + size
        if found != oid or kind != "blob":
            raise ValueError("Git object stream mismatch")
        report["bytes"] += size
        path = ROOT / name
        sealed = name.startswith(SEALED_ROOTS) or path.suffix == ".py"
        if path.read_bytes() != data:
            if sealed:
                report["failures"].append({"file": name, "reason": "sealed bytes differ in index"})
            else:
                report["nonsealed_worktree_byte_differences"].append(name)
        if sealed:
            report["sealed_bytes_checked"] += 1
        if size > 10 * 1024 ** 2 or (FORBIDDEN.search(name) and not name.endswith(".gitkeep")):
            report["failures"].append({"file": name, "reason": "large or excluded artifact"})
        if path.suffix in TEXT_SUFFIXES:
            text = data.decode("utf-8-sig")
            report["text_scanned"] += 1
            if HOME.search(text) or SECRET.search(text):
                report["failures"].append({"file": name, "reason": "home path or credential signature"})
            if path.suffix == ".py":
                ast.parse(text, filename=name)
                report["python_parsed"] += 1
            elif path.suffix == ".json":
                json.loads(text)
                report["json_parsed"] += 1
            elif name == "CITATION.cff":
                citation = yaml.safe_load(text)
                if citation["version"] != "1.3.0" or citation["authors"][0]["family-names"] != "Liu":
                    report["failures"].append({"file": name, "reason": "release citation identity differs"})
    report["index_tree"] = git("write-tree").decode().strip()
    report["index_inventory_sha256"] = hashlib.sha256(
        json.dumps(records, separators=(",", ":")).encode()
    ).hexdigest()
    report["status"] = "PASS" if not report["failures"] else "FAIL"
    print(json.dumps(report, indent=2))
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
