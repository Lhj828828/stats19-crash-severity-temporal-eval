"""Unit tests for the isolated public reproduction orchestrator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
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
public_reproduction = load_module(
    "public_reproduction", CODE_DIR / "public_reproduction.py"
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_workspace_marker(project: Path) -> None:
    write_json(
        project / "logs" / "public_workspace.json",
        {
            "version": public_reproduction.VERSION,
            "status": "PUBLIC_WORKSPACE_PREPARED",
            "source_project": str(project.parent / "source"),
            "workspace": str(project),
            "copied_inputs": {},
        },
    )


class PublicReproductionTests(unittest.TestCase):
    def test_child_output_is_forced_to_utf8(self) -> None:
        self.assertEqual(
            public_reproduction.THREAD_ENVIRONMENT["PYTHONIOENCODING"], "utf-8"
        )

    def test_root_entry_point_is_directly_runnable(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(PROJECT_DIR / "run_public_reproduction.py"), "--list-stages"],
            cwd=PROJECT_DIR,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertIn("01 PUBLIC_D1_input_audit", completed.stdout)
        self.assertIn("32 D14_closeout", completed.stdout)

    def test_pipeline_has_public_boundary_and_protocol_freezes(self) -> None:
        stages = public_reproduction.PIPELINE_STAGES
        names = [stage.name for stage in stages]
        arguments = [argument for stage in stages for argument in stage.arguments]
        self.assertEqual(stages[0].arguments, ("code/public_input_audit.py",))
        self.assertEqual(len(names), len(set(names)))
        self.assertNotIn("code/d1_acquire_and_audit.py", arguments)
        self.assertFalse(any("d16" in argument.lower() for argument in arguments))
        self.assertNotIn("code/d15_reproducibility.py", arguments)
        self.assertLess(names.index("D9_freeze_protocol"), names.index("D9_ordered_logit_subset"))
        self.assertLess(names.index("D10_freeze_protocol"), names.index("D10_tune"))
        self.assertLess(names.index("D11_freeze_protocol"), names.index("D11_one_time_evaluation"))

    def test_workspace_must_be_outside_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            with self.assertRaises(ValueError):
                public_reproduction.ensure_external_workspace(source, source)
            with self.assertRaises(ValueError):
                public_reproduction.ensure_external_workspace(source / "child", source)

    def test_bind_d10_changes_only_runtime_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "workspace"
            project.mkdir()
            make_workspace_marker(project)
            protocol = project / "config" / "d10_lightgbm_protocol.json"
            amendment = project / "config" / "d10_execution_amendment.json"
            write_json(protocol, {"version": "D10_V2", "upstream": {"x": "y"}})
            write_json(
                amendment,
                {
                    "version": "D10_EXECUTION_AMENDMENT_V1",
                    "original_protocol_sha256": "0" * 64,
                    "amended_n_jobs": 4,
                    "scientific_parameters_changed": False,
                    "candidate_set_changed": False,
                    "selection_rule_changed": False,
                },
            )
            original = json.loads(amendment.read_text(encoding="utf-8"))
            public_reproduction.bind_d10_amendment(project)
            bound = json.loads(amendment.read_text(encoding="utf-8"))
            self.assertEqual(
                bound["original_protocol_sha256"],
                public_reproduction.hash_file(protocol),
            )
            self.assertEqual(bound["amended_n_jobs"], original["amended_n_jobs"])
            self.assertEqual(
                bound["candidate_set_changed"], original["candidate_set_changed"]
            )
            self.assertFalse(
                bound["public_reproduction_instance"]["scientific_parameters_changed"]
            )
            public_reproduction.bind_d10_amendment(project)

    def test_bind_d10b_refreshes_only_declared_upstream_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "workspace"
            project.mkdir()
            make_workspace_marker(project)
            mappings = {
                "D5_schema_sha256": "config/d5_dataset_schema.json",
                "D6_protocol_sha256": "config/d6_analysis_protocol.json",
                "D6_assignments_sha256": "data/processed/d6_split_assignments.csv.gz",
                "D7_training_inputs_sha256": "config/d7_training_inputs.json",
                "D10_protocol_sha256": "config/d10_lightgbm_protocol.json",
                "D10_execution_amendment_sha256": "config/d10_execution_amendment.json",
                "D10_selected_model_sha256": "config/d10_selected_lightgbm.json",
                "D10_tuning_results_sha256": "results/d10/d10_tuning_results.csv",
                "D10_source_sha256": "code/d10_tune_lightgbm.py",
                "baseline_modeling_source_sha256": "code/baseline_modeling.py",
                "modeling_data_source_sha256": "code/modeling_data.py",
            }
            for index, raw_path in enumerate(mappings.values()):
                path = project / raw_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"input-{index}".encode("ascii"))
            protocol_path = project / "config" / "d10b_tree_sensitivity_protocol.json"
            write_json(
                protocol_path,
                {
                    "version": "D10B_V1",
                    "upstream": {key: "0" * 64 for key in mappings},
                    "model_lock": {"candidate_id": "C03", "n_jobs": 4},
                    "diagnostic": {
                        "single_fit_estimators": 2000,
                        "checkpoints": [1200, 1500, 2000],
                    },
                    "decision_lock": {"D11_change_allowed_from_D10b": False},
                },
            )
            public_reproduction.bind_d10b_protocol(project)
            bound = json.loads(protocol_path.read_text(encoding="utf-8"))
            for key, raw_path in mappings.items():
                expected = hashlib.sha256((project / raw_path).read_bytes()).hexdigest()
                self.assertEqual(bound["upstream"][key], expected)
            self.assertEqual(bound["model_lock"]["candidate_id"], "C03")
            self.assertEqual(bound["diagnostic"]["checkpoints"], [1200, 1500, 2000])
            self.assertFalse(
                bound["public_reproduction_instance"]["scientific_parameters_changed"]
            )
            public_reproduction.bind_d10b_protocol(project)


if __name__ == "__main__":
    unittest.main(verbosity=2)
