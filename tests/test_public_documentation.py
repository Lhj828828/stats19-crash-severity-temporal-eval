"""Consistency checks for public reproduction documentation."""

from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


class PublicDocumentationTests(unittest.TestCase):
    def test_readme_names_public_runner_and_preserves_release_boundary(self) -> None:
        readme = (PROJECT_DIR / "README.md").read_text(encoding="utf-8")
        self.assertIn("python run_public_reproduction.py --all", readme)
        self.assertIn("clean D8-D14 run", readme)
        self.assertIn("not a command for third-party use", readme)

    def test_data_guide_does_not_require_complete_file_for_public_run(self) -> None:
        guide = (PROJECT_DIR / "DATA_SOURCES.md").read_text(encoding="utf-8")
        local_layout = guide.split("## Recreating the local data layout", maxsplit=1)[1]
        normalized = " ".join(local_layout.split())
        self.assertIn("download_and_verify_data.py download --dataset stats19", local_layout)
        self.assertNotIn("code/d1_acquire_and_audit.py", local_layout)
        self.assertIn("not exact-snapshot fallbacks", normalized)

    def test_reproduction_guide_discloses_unfinished_release_gates(self) -> None:
        guide = (PROJECT_DIR / "REPRODUCING.md").read_text(encoding="utf-8")
        self.assertIn("full clean D8-D14 run", guide)
        self.assertIn("immutable data-record URLs", guide)
        self.assertIn("--stop-after D7_training_inputs", guide)
        self.assertIn("author-side forensic comparator", guide)


if __name__ == "__main__":
    unittest.main(verbosity=2)
