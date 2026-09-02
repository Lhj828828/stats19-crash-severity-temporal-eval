"""Checks that public execution does not depend on an author desktop file."""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
EVIDENCE_FILE = PROJECT_DIR / "config" / "d14_exclude2020_planning_evidence.json"
FROZEN_PROTOCOL_FILE = PROJECT_DIR / "config" / "d14_exclude2020_protocol_v2.json"
SCRIPT_FILE = PROJECT_DIR / "code" / "d14_exclude2020_sensitivity.py"
TEXT_SUFFIXES = {
    ".cff",
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=PROJECT_DIR,
        check=True,
        capture_output=True,
    )
    return [
        PROJECT_DIR / raw.decode("utf-8")
        for raw in result.stdout.split(b"\0")
        if raw
    ]


class PortableProvenanceTests(unittest.TestCase):
    def test_original_outline_hash_is_preserved_without_runtime_dependency(self) -> None:
        evidence = json.loads(EVIDENCE_FILE.read_text(encoding="utf-8"))
        frozen = json.loads(FROZEN_PROTOCOL_FILE.read_text(encoding="utf-8"))
        original_hash = evidence["original_author_file"]["sha256"]
        self.assertIn(original_hash, frozen["upstream_sha256"].values())
        self.assertFalse(evidence["original_author_file"]["redistributed"])
        self.assertIn("not an external preregistration", evidence["evidence_boundary"]["not_supported"])

    def test_d14_script_uses_repository_evidence(self) -> None:
        source = SCRIPT_FILE.read_text(encoding="utf-8")
        self.assertIn("d14_exclude2020_planning_evidence.json", source)
        self.assertNotIn("PROJECT_DIR.parent", source)
        self.assertNotIn("OUTLINE_FILE", source)
        self.assertIn(
            '"upstream_sha256": {relative(path): hash_file(path) for path in required}',
            source,
        )

    def test_tracked_text_has_no_author_workstation_path(self) -> None:
        markers = (
            ":" + "/" + "Users" + "/",
            ":" + "\\" + "Users" + "\\",
            "One" + "Drive",
            "/" + "home" + "/",
        )
        hits: list[str] = []
        for path in tracked_files():
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            if any(marker.lower() in text.lower() for marker in markers):
                hits.append(path.relative_to(PROJECT_DIR).as_posix())
        self.assertEqual(hits, [], f"Author-workstation paths remain in: {hits}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
