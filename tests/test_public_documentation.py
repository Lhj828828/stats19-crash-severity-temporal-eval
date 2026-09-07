"""Consistency checks for public reproduction documentation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


class PublicDocumentationTests(unittest.TestCase):
    def test_current_journal_target_is_applied_sciences(self) -> None:
        readme = (PROJECT_DIR / "README.md").read_text(encoding="utf-8")
        checklist = PROJECT_DIR / "docs" / "APPLIED_SCIENCES_RELEASE_CHECKLIST.md"
        self.assertIn("Applied Sciences", readme)
        self.assertNotIn("docs/IEEE_ACCESS_RELEASE_CHECKLIST.md", readme)
        self.assertTrue(checklist.is_file())
        self.assertFalse(
            (PROJECT_DIR / "docs" / "IEEE_ACCESS_RELEASE_CHECKLIST.md").exists()
        )

    def test_readme_names_public_runner_and_preserves_release_boundary(self) -> None:
        readme = (PROJECT_DIR / "README.md").read_text(encoding="utf-8")
        self.assertIn("python run_public_reproduction.py --all", readme)
        self.assertIn("all D1-D14 stages", readme)
        self.assertIn("not an independent", readme)
        self.assertIn("not a command for third-party use", readme)

    def test_data_guide_does_not_require_complete_file_for_public_run(self) -> None:
        guide = (PROJECT_DIR / "DATA_SOURCES.md").read_text(encoding="utf-8")
        local_layout = guide.split("## Recreating the local data layout", maxsplit=1)[1]
        normalized = " ".join(local_layout.split())
        self.assertIn("download_and_verify_data.py download --dataset all", local_layout)
        self.assertNotIn("code/d1_acquire_and_audit.py", local_layout)
        self.assertIn("not exact-snapshot fallbacks", normalized)

    def test_reproduction_guide_discloses_validation_and_published_data(self) -> None:
        guide = (PROJECT_DIR / "REPRODUCING.md").read_text(encoding="utf-8")
        self.assertIn("seven fixed annual files through D14", guide)
        self.assertIn("not evidence of an independent", guide)
        self.assertIn("10.5281/zenodo.22290566", guide)
        self.assertIn("10.5281/zenodo.22296725", guide)
        self.assertIn("all nine files", guide)
        self.assertIn("--stop-after D7_training_inputs", guide)
        self.assertIn("author-side forensic comparator", guide)

    def test_validation_record_is_specific_and_does_not_claim_independence(self) -> None:
        record = (
            PROJECT_DIR / "docs" / "PUBLIC_REPRODUCTION_VALIDATION.md"
        ).read_text(encoding="utf-8")
        self.assertIn("baf63c3d9373c64d9845374c280848ae1a45ef25", record)
        self.assertIn("47/47", record)
        self.assertIn("21/21", record)
        self.assertIn("not a claim that an independent third party", record)
        self.assertIn("9/9 PASS", record)
        self.assertIn("10.5281/zenodo.22290566", record)
        self.assertIn("10.5281/zenodo.22296725", record)

    def test_stats19_redistribution_review_is_explicit(self) -> None:
        review = (
            PROJECT_DIR / "docs" / "STATS19_REDISTRIBUTION_REVIEW.md"
        ).read_text(encoding="utf-8")
        normalized_review = " ".join(review.split())
        manifest = json.loads(
            (PROJECT_DIR / "config" / "public_data_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        stats19 = manifest["datasets"]["stats19"]
        self.assertEqual(
            stats19["source_terms_review"],
            "REVIEW_COMPLETE_REDISTRIBUTION_PERMITTED_WITH_ATTRIBUTION",
        )
        self.assertEqual(stats19["license"]["id"], "ogl-uk-3.0")
        self.assertIn("derived fixed snapshots", normalized_review)
        self.assertIn("not legal advice", normalized_review)

        self.assertIn("must not imply", normalized_review)

    def test_cas_redistribution_review_is_explicit(self) -> None:
        review = (
            PROJECT_DIR / "docs" / "CAS_REDISTRIBUTION_REVIEW.md"
        ).read_text(encoding="utf-8")
        normalized_review = " ".join(review.split())
        manifest = json.loads(
            (PROJECT_DIR / "config" / "public_data_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        cas = manifest["datasets"]["cas"]
        self.assertEqual(
            cas["source_terms_review"],
            "REVIEW_COMPLETE_REDISTRIBUTION_PERMITTED_WITH_ATTRIBUTION",
        )
        self.assertEqual(cas["license"]["id"], "cc-by-4.0")
        self.assertEqual(
            cas["license"]["review_record"],
            "docs/CAS_REDISTRIBUTION_REVIEW.md",
        )
        self.assertIn("project-created serialisations", normalized_review)
        self.assertIn("not legal advice", normalized_review)
        self.assertIn("does not endorse", normalized_review)

if __name__ == "__main__":
    unittest.main(verbosity=2)
