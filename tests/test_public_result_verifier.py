"""Tests for compact public-result verification."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_DIR / "code" / "public_result_verifier.py"
SPEC = importlib.util.spec_from_file_location("public_result_verifier", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load public_result_verifier.py")
verifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verifier
SPEC.loader.exec_module(verifier)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def base_contract(comparisons: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "PUBLIC_RESULT_CONTRACT_V1",
        "reference_root": "reference",
        "tolerances": {"absolute": 1e-10, "relative": 1e-8},
        "comparisons": comparisons,
        "required_nonempty_files": ["figures/required.png"],
    }


class PublicResultVerifierTests(unittest.TestCase):
    def test_csv_comparison_is_keyed_and_tolerant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference" / "results" / "sample.csv"
            candidate = root / "results" / "sample.csv"
            write_csv(
                reference,
                ["id", "label", "metric", "fit_seconds"],
                [
                    {"id": "a", "label": "first", "metric": 0.5, "fit_seconds": 1},
                    {"id": "b", "label": "second", "metric": 1.0, "fit_seconds": 2},
                ],
            )
            write_csv(
                candidate,
                ["metric", "id", "fit_seconds", "label"],
                [
                    {"id": "b", "label": "second", "metric": 1.000000001, "fit_seconds": 99},
                    {"id": "a", "label": "first", "metric": 0.5, "fit_seconds": 88},
                ],
            )
            result = verifier.compare_csv(
                reference,
                candidate,
                {
                    "key_columns": ["id"],
                    "ignore_columns": ["fit_seconds"],
                },
                atol=1e-10,
                rtol=1e-8,
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["reference_rows"], 2)

    def test_filter_excludes_ordered_logit_from_pass_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.csv"
            candidate = root / "candidate.csv"
            rows = [
                {"scope": "test", "model": "logistic_weighted", "metric": 0.4},
                {"scope": "test", "model": "ordered_logit_unweighted", "metric": 0.3},
            ]
            write_csv(reference, ["scope", "model", "metric"], rows)
            changed = [dict(row) for row in rows]
            changed[1]["metric"] = -999
            write_csv(candidate, ["scope", "model", "metric"], changed)
            result = verifier.compare_csv(
                reference,
                candidate,
                {
                    "key_columns": ["scope", "model"],
                    "filter": {
                        "column": "model",
                        "include": ["logistic_weighted"],
                    },
                },
                atol=0,
                rtol=0,
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["candidate_rows"], 1)

    def test_numeric_difference_outside_tolerance_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.csv"
            candidate = root / "candidate.csv"
            write_csv(reference, ["id", "metric"], [{"id": 1, "metric": 1.0}])
            write_csv(candidate, ["id", "metric"], [{"id": 1, "metric": 1.01}])
            result = verifier.compare_csv(
                reference,
                candidate,
                {"key_columns": ["id"]},
                atol=1e-10,
                rtol=1e-8,
            )
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(result["mismatch_count"], 1)

    def test_json_comparison_recursively_ignores_runtime_keys(self) -> None:
        result = verifier.compare_json_values(
            {
                "status": "PASS",
                "metric": 0.5,
                "nested": {"runtime_seconds": 1, "decision": False},
            },
            {
                "nested": {"decision": False, "runtime_seconds": 999},
                "metric": 0.500000001,
                "status": "PASS",
            },
            ignored_keys={"runtime_seconds"},
            atol=1e-10,
            rtol=1e-8,
        )
        self.assertEqual(result["status"], "PASS")

    def test_contract_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            contract_path = Path(directory) / "contract.json"
            contract = base_contract(
                [
                    {
                        "path": "../escape.csv",
                        "kind": "csv",
                        "key_columns": ["id"],
                    }
                ]
            )
            write_json(contract_path, contract)
            with self.assertRaises(verifier.ResultContractError):
                verifier.load_contract(contract_path)

    def test_repository_contract_uses_only_compact_artifacts(self) -> None:
        contract = verifier.load_contract(
            PROJECT_DIR / "config" / "public_result_contract.json"
        )
        forbidden_suffixes = {".npz", ".npy", ".joblib", ".csv.gz"}
        forbidden_parts = {"models", "predictions", "shap_values", "bootstrap_arrays"}
        for item in contract["comparisons"]:
            path = PurePosixPath(item["path"])
            self.assertFalse(any(str(path).lower().endswith(x) for x in forbidden_suffixes))
            self.assertTrue(forbidden_parts.isdisjoint(part.lower() for part in path.parts))

        reference_root = PROJECT_DIR / contract["reference_root"]
        files = [path for path in reference_root.rglob("*") if path.is_file()]
        self.assertEqual(len(files), len(contract["comparisons"]))
        self.assertLess(sum(path.stat().st_size for path in files), 1_000_000)

    def test_frozen_project_matches_compact_references(self) -> None:
        report = verifier.verify_results(candidate_root=PROJECT_DIR)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(
            report["comparison_summary"],
            {"passed": 21, "total": 21},
        )
if __name__ == "__main__":
    unittest.main(verbosity=2)
