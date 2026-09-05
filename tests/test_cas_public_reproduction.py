"""Unit checks for the isolated public CAS reproduction entry point."""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_DIR / "code" / "cas_public_reproduction.py"


def load_module():
    spec = importlib.util.spec_from_file_location("cas_public_reproduction_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cas_public = load_module()


class CASPublicReproductionTests(unittest.TestCase):
    def test_root_entry_lists_offline_pipeline(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(PROJECT_DIR / "run_cas_public_reproduction.py"),
                "--list-stages",
            ],
            cwd=PROJECT_DIR,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertIn("01 CAS_D1_OFFLINE_FEASIBILITY_AUDIT", completed.stdout)
        self.assertIn(
            "CAS_POSTHOC_FEATURE_ABLATION_SENSITIVITY",
            completed.stdout,
        )
        self.assertIn("CAS_PUBLIC_CLOSEOUT", completed.stdout)
        self.assertIn("TESTS_and_COMPACT_RESULT_VERIFICATION", completed.stdout)

    def test_pipeline_has_no_live_query_or_forensic_stage(self) -> None:
        arguments = [argument for stage in cas_public.PIPELINE_STAGES for argument in stage.arguments]
        self.assertEqual(
            cas_public.PIPELINE_STAGES[0].arguments,
            ("code/cas_offline_feasibility_audit.py",),
        )
        self.assertFalse(any("d16" in argument.lower() for argument in arguments))
        self.assertFalse(any("cas_feasibility_audit.py" == argument for argument in arguments))
        self.assertEqual(len(cas_public.STANDALONE_TESTS), 9)
        stage_names = [stage.name for stage in cas_public.PIPELINE_STAGES]
        self.assertLess(
            stage_names.index("CAS_D6_ONE_TIME_2025_EVALUATION"),
            stage_names.index("CAS_POSTHOC_BIND_FEATURE_ABLATION_PROTOCOL"),
        )
        self.assertLess(
            stage_names.index("CAS_POSTHOC_FEATURE_ABLATION_SENSITIVITY"),
            stage_names.index("CAS_D7_FREEZE_POST_ANALYSIS_PROTOCOL"),
        )

    def test_public_inputs_exclude_rebuildable_author_artifacts(self) -> None:
        inputs = cas_public.input_map(PROJECT_DIR)
        self.assertIn("config/cas_public_result_contract.json", inputs)
        self.assertIn(
            "config/cas_public_reference_protocols/cas_feature_ablation_protocol.json",
            inputs,
        )
        self.assertIn(
            "config/cas_public_reference_protocols/cas_cross_dataset_reporting_protocol.json",
            inputs,
        )
        self.assertIn("references/stats19_upstream/d14_final_summary.json", inputs)
        forbidden_prefixes = (
            "data/interim/",
            "data/processed/",
            "models/",
            "results/",
            "logs/",
        )
        self.assertFalse(
            any(path.startswith(forbidden_prefixes) for path in inputs),
            sorted(path for path in inputs if path.startswith(forbidden_prefixes)),
        )

    def test_public_inputs_do_not_embed_author_home_paths(self) -> None:
        patterns = (
            re.compile(r"(?i)[a-z]:" + r"\\+" + "users" + r"\\+"),
            re.compile("/" + "(?:users|home)" + "/", re.IGNORECASE),
            re.compile(r"\.\./" + "stats19论文", re.IGNORECASE),
        )
        offenders: list[str] = []
        text_suffixes = {".cff", ".json", ".md", ".ps1", ".py", ".txt"}
        for relative_path, path in cas_public.input_map(PROJECT_DIR).items():
            if path.suffix.lower() not in text_suffixes:
                continue
            content = path.read_text(encoding="utf-8")
            if any(pattern.search(content) for pattern in patterns):
                offenders.append(relative_path)
        self.assertEqual(
            offenders,
            [],
            f"Public CAS inputs contain author-specific paths: {offenders}",
        )

    def test_workspace_must_be_outside_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            with self.assertRaises(ValueError):
                cas_public.ensure_external_workspace(source, source)
            with self.assertRaises(ValueError):
                cas_public.ensure_external_workspace(source / "child", source)

    def test_fixed_snapshot_hash_is_pinned(self) -> None:
        self.assertTrue(cas_public.DEFAULT_SNAPSHOT.is_file())
        self.assertEqual(
            cas_public.hash_file(cas_public.DEFAULT_SNAPSHOT),
            cas_public.EXPECTED_SNAPSHOT_SHA256,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
