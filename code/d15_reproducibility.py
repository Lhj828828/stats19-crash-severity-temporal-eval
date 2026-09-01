"""D15 read-only audit and packaging entrypoint for the frozen STATS19 project.

The script verifies the existing D1-D14 artifacts instead of refitting models.
All paths are resolved from this file's project root, so the project can be
moved without editing the analysis code. It writes only D15 audit outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as package_metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _name in THREAD_ENVIRONMENT_VARIABLES:
    os.environ.setdefault(_name, "4")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score


PROJECT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_DIR / "code"
CONFIG_DIR = PROJECT_DIR / "config"
DATA_DIR = PROJECT_DIR / "data"
FIGURE_DIR = PROJECT_DIR / "figures"
LOG_DIR = PROJECT_DIR / "logs"
MODEL_DIR = PROJECT_DIR / "models"
RESULT_DIR = PROJECT_DIR / "results"

VERSION = "D15_V1"
CONFIG_FILE = CONFIG_DIR / "d15_reproducibility_protocol.json"
REQUIREMENTS_FILE = PROJECT_DIR / "requirements.txt"
LOCK_FILE = PROJECT_DIR / "requirements-lock.txt"
REPORT_FILE = LOG_DIR / "d15_reproducibility_report.json"
MANIFEST_FILE = LOG_DIR / "d15_artifact_manifest.csv"
CHECKPOINT_FILE = LOG_DIR / "d15_checkpoint.md"

DATA_FILE = DATA_DIR / "processed" / "stats19_modeling_dataset.csv.gz"
ASSIGNMENTS_FILE = DATA_DIR / "processed" / "d6_split_assignments.csv.gz"
SCHEMA_FILE = CONFIG_DIR / "d5_dataset_schema.json"
D6_FILE = CONFIG_DIR / "d6_analysis_protocol.json"
D7_FILE = CONFIG_DIR / "d7_training_inputs.json"
D10_SELECTED_FILE = CONFIG_DIR / "d10_selected_lightgbm.json"
D11_FILE = CONFIG_DIR / "d11_evaluation_protocol.json"
D13_FILE = CONFIG_DIR / "d13_positioning_protocol.json"
D14_FILE = CONFIG_DIR / "d14_closeout_protocol_v2.json"
D14_SUMMARY_FILE = LOG_DIR / "d14_final_summary.json"
D11_METRICS_FILE = RESULT_DIR / "d11" / "d11_test_metrics.csv"
D13_MAIN_FILE = RESULT_DIR / "d13" / "d13_main_performance_table.csv"
D14_RANK_FILE = RESULT_DIR / "d14" / "d14_rank_stability_summary.csv"
D14_ERROR_FILE = RESULT_DIR / "d14" / "d14_error_structure.csv"

PRIMARY_MODELS = ("logistic_weighted", "lightgbm_weighted")
RANDOM_SEEDS = (1103, 2207, 3301, 4409, 5501)
TEST_FILES = (
    "code/test_cas_feasibility_audit.py",
    "code/test_d10_lightgbm.py",
    "code/test_d10b_tree_sensitivity.py",
    "code/test_d11_evaluation.py",
    "code/test_d12_bootstrap.py",
    "code/test_d13_checkpoint.py",
    "code/test_d14_closeout.py",
    "code/test_d14_exclude2020.py",
    "code/test_d14_shap.py",
    "code/test_d6_protocol.py",
    "code/test_d7_training_inputs.py",
    "code/test_d8_baselines.py",
    "code/test_d9_ordered_logit.py",
    "code/test_d9s1_matched_subset.py",
    "code/test_environment.py",
)


@dataclass
class CheckRecord:
    name: str
    status: str
    detail: str
    seconds: float


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_DIR.resolve()).as_posix()


def resolve_relative(raw_path: str) -> Path:
    """Resolve a manifest path while rejecting absolute/path-traversal entries."""
    normalized = str(raw_path).strip().replace("\\", "/")
    if not normalized or re.match(r"^[A-Za-z]:/", normalized) or normalized.startswith("/"):
        raise ValueError(f"Manifest path is not project-relative: {raw_path}")
    resolved = (PROJECT_DIR / Path(normalized)).resolve()
    if not resolved.is_relative_to(PROJECT_DIR.resolve()):
        raise ValueError(f"Manifest path escapes project root: {raw_path}")
    return resolved


def run_check(
    name: str,
    function: Callable[[], str],
    records: list[CheckRecord],
) -> bool:
    started = time.perf_counter()
    try:
        detail = function()
        elapsed = time.perf_counter() - started
        records.append(CheckRecord(name, "PASS", detail, round(elapsed, 3)))
        print(f"[PASS] {name}: {detail}")
        return True
    except Exception as exc:  # Keep all failures in one final report.
        elapsed = time.perf_counter() - started
        detail = f"{type(exc).__name__}: {exc}"
        records.append(CheckRecord(name, "FAIL", detail, round(elapsed, 3)))
        print(f"[FAIL] {name}: {detail}")
        return False


def check_protocol_contract() -> str:
    d15 = read_json(CONFIG_FILE)
    d6 = read_json(D6_FILE)
    d7 = read_json(D7_FILE)
    selected = read_json(D10_SELECTED_FILE)
    d11 = read_json(D11_FILE)
    d13 = read_json(D13_FILE)
    d14 = read_json(D14_FILE)
    summary = read_json(D14_SUMMARY_FILE)

    assert d15["version"] == VERSION
    assert d15["status"] == "REPRODUCIBILITY_PROTOCOL_FROZEN"
    assert d6["status"] == "FROZEN_BEFORE_MODEL_TRAINING"
    assert d7["status"] == "TRAINING_INPUTS_AUDITED_AND_FROZEN"
    assert selected["selected_candidate_id"] == "C03"
    assert int(selected["selected_n_estimators"]) == 1200
    assert d11["status"] == "FROZEN_BEFORE_D11_TEST_ACCESS"
    assert d13["status"] == "FROZEN_POST_D12_POSITIONING_RULES"
    assert d14["status"] == "FROZEN_BEFORE_D14_ERROR_CASE_EXTRACTION"
    assert summary["status"] == "D14_COMPLETE_WITH_REGIONAL_HOLDOUT_DEVIATION"
    assert summary["D15_ready"] is True
    assert summary["H1"]["broad_random_split_optimism_claim"] == "NOT_SUPPORTED"
    assert summary["H2"]["uniform_lightgbm_superiority_claim"] == "NOT_SUPPORTED"
    assert summary["H3"]["causal_claim_allowed"] is False
    assert summary["protocol_deviation"]["regional_holdout_executed"] is False
    assert d6["frozen_population"]["row_count"] == 743646
    assert d6["temporal_protocol"]["row_counts"] == {
        "train": 538461,
        "validation": 104258,
        "test": 100927,
    }
    assert d6["random_reference_protocol"]["seeds"] == list(RANDOM_SEEDS)
    assert d7["feature_contract"]["feature_count"] == 17
    return "D6-D14 statuses, selection lock, reporting boundaries and D15 handoff are consistent"


def check_data_contract() -> str:
    schema = read_json(SCHEMA_FILE)
    feature_columns = list(schema["feature_columns"])
    metadata_columns = list(schema["metadata_columns"])
    target_column = str(schema["target_column"])
    expected_columns = metadata_columns + [target_column] + feature_columns

    header = pd.read_csv(DATA_FILE, nrows=0).columns.tolist()
    assert header == expected_columns
    assert len(feature_columns) == 17
    assert len(metadata_columns) == 5

    trace_columns = ["meta_collision_index", "meta_collision_year", "target_severity"]
    data = pd.read_csv(
        DATA_FILE,
        usecols=trace_columns,
        dtype={"meta_collision_index": "string"},
    )
    assignments = pd.read_csv(
        ASSIGNMENTS_FILE,
        dtype={"meta_collision_index": "string"},
    )
    assert len(data) == 743646
    assert len(assignments) == len(data)
    assert data["meta_collision_index"].is_unique
    assert assignments["meta_collision_index"].is_unique
    assert data["meta_collision_index"].tolist() == assignments["meta_collision_index"].tolist()
    assert data["meta_collision_year"].tolist() == assignments["meta_collision_year"].tolist()
    assert data["target_severity"].tolist() == assignments["target_severity"].tolist()
    assert set(data["target_severity"].unique().tolist()) == {0, 1, 2}

    temporal_counts = assignments["temporal_role"].value_counts().to_dict()
    assert temporal_counts == {"train": 538461, "validation": 104258, "test": 100927}
    assert assignments.loc[assignments["meta_collision_year"].eq(2024), "temporal_role"].eq("test").all()
    assert assignments.loc[assignments["meta_collision_year"].ne(2024), "temporal_role"].ne("test").all()

    expected_random = {"train": 449903, "validation": 96408, "test": 96408, "locked_temporal_test": 100927}
    for seed in RANDOM_SEEDS:
        column = f"random_role_seed_{seed}"
        counts = assignments[column].value_counts().to_dict()
        assert counts == expected_random, (column, counts)
        future = assignments["meta_collision_year"].eq(2024)
        assert assignments.loc[future, column].eq("locked_temporal_test").all()
        assert assignments.loc[~future, column].isin(("train", "validation", "test")).all()

    return "743646 rows, 17 features, 5 metadata columns and all six frozen role columns match the contract"


def check_prediction_contract() -> str:
    metrics = pd.read_csv(D11_METRICS_FILE)
    # D11 stores five temporal rows, 25 random-reference rows and 25 optional
    # 2024 diagnostic rows in the same table.
    assert len(metrics) == 55
    checked = []
    for model in PRIMARY_MODELS:
        path = RESULT_DIR / "d11" / "predictions" / f"temporal__{model}__test.csv.gz"
        frame = pd.read_csv(path, dtype={"meta_collision_index": "string"})
        assert len(frame) == 100927
        assert frame["meta_collision_index"].is_unique
        assert frame["meta_collision_year"].eq(2024).all()
        assert set(frame["target_severity"].unique().tolist()) == {0, 1, 2}
        assert set(frame["predicted_severity"].unique().tolist()).issubset({0, 1, 2})
        probabilities = frame[["prob_slight", "prob_serious", "prob_fatal"]].to_numpy(dtype=float)
        assert np.isfinite(probabilities).all()
        assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=2e-8)

        y_true = frame["target_severity"].to_numpy(dtype=int)
        y_pred = frame["predicted_severity"].to_numpy(dtype=int)
        row = metrics.loc[
            metrics["protocol"].eq("temporal")
            & metrics["seed"].astype(str).eq("year_based")
            & metrics["model"].eq(model)
        ]
        assert len(row) == 1
        row = row.iloc[0]
        macro_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
        qwk = cohen_kappa_score(y_true, y_pred, labels=[0, 1, 2], weights="quadratic")
        fatal_recall = recall_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0]
        assert np.isclose(macro_f1, float(row["macro_f1"]), rtol=0, atol=1e-12)
        assert np.isclose(qwk, float(row["qwk"]), rtol=0, atol=1e-12)
        assert np.isclose(fatal_recall, float(row["fatal_recall"]), rtol=0, atol=1e-12)
        checked.append(f"{model}: macro_f1={macro_f1:.6f}, fatal_recall={fatal_recall:.6f}")

    assert len(pd.read_csv(D13_MAIN_FILE)) == 35
    assert len(pd.read_csv(D14_RANK_FILE)) == 4
    assert len(pd.read_csv(D14_ERROR_FILE)) == 12
    return "Primary 2024 prediction tables and recomputed metrics agree; " + "; ".join(checked)


def check_model_smoke() -> str:
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))
    from baseline_modeling import prepare_features
    from modeling_data import load_modeling_data

    features, _, _ = load_modeling_data(nrows=64)
    bundle = joblib.load(MODEL_DIR / "d10" / "temporal" / "preprocessing_bundle.joblib")
    artifact = joblib.load(MODEL_DIR / "d10" / "temporal" / "lightgbm_weighted.joblib")
    prepared = prepare_features(
        features,
        feature_columns=bundle["feature_columns"],
        categorical_columns=bundle["categorical_columns"],
        numeric_columns=bundle["numeric_columns"],
        category_vocabulary=bundle["category_vocabulary"],
    )
    estimator = artifact["estimator"]
    predicted = np.asarray(estimator.predict(prepared.frame))
    probabilities = np.asarray(estimator.predict_proba(prepared.frame), dtype=float)
    assert predicted.shape == (64,)
    assert probabilities.shape == (64, 3)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-8)
    assert set(predicted.tolist()).issubset({0, 1, 2})
    return "Frozen temporal C03-1200 LightGBM loaded and predicted 64 records with a valid 3-class probability matrix"


def check_dependency_contract() -> str:
    expected: dict[str, str] = {}
    for raw_line in LOCK_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise ValueError(f"Unpinned lock entry: {line}")
        name, version = line.split("==", 1)
        expected[name.strip()] = version.strip()

    mismatches: list[str] = []
    for name, expected_version in expected.items():
        try:
            observed = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            mismatches.append(f"{name}: missing (expected {expected_version})")
            continue
        if observed != expected_version:
            mismatches.append(f"{name}: {observed} != {expected_version}")
    if mismatches:
        raise AssertionError("; ".join(mismatches))
    python_version = platform.python_version()
    assert python_version == "3.13.15", python_version
    return f"Python {python_version}; {len(expected)} locked package versions match"


def check_portability_contract() -> str:
    portable_files = (
        Path(__file__),
        PROJECT_DIR / "run_d15_reproducibility.ps1",
        CONFIG_FILE,
        PROJECT_DIR / "README.md",
    )
    # Build forbidden markers from pieces so this audit does not match its own
    # detector expression.
    patterns = (
        ":" + "/" + "Users" + "/",
        ":" + chr(92) + "Users" + chr(92),
        "/" + "mnt" + "/" + "data",
        "One" + "Drive",
    )
    hits: list[str] = []
    for path in portable_files:
        text = path.read_text(encoding="utf-8")
        if any(pattern.lower() in text.lower() for pattern in patterns):
            hits.append(relative_path(path))
    if hits:
        raise AssertionError("D15 runtime files contain absolute workstation paths: " + ", ".join(hits))
    return "D15 runtime files resolve paths from the project root; historical absolute provenance records remain archival only"


def _manifest_store() -> dict[str, dict[str, Any]]:
    store: dict[str, dict[str, Any]] = {}

    def add(raw_path: str, source: str, expected_bytes: int | None = None, expected_sha256: str | None = None) -> None:
        path = resolve_relative(raw_path)
        rel = relative_path(path)
        entry = store.setdefault(
            rel,
            {"path": path, "sources": set(), "expected_bytes": set(), "expected_sha256": set()},
        )
        entry["sources"].add(source)
        if expected_bytes is not None:
            entry["expected_bytes"].add(int(expected_bytes))
        if expected_sha256:
            entry["expected_sha256"].add(str(expected_sha256))

    # Preserve and validate every prior frozen artifact manifest.
    for manifest in sorted(LOG_DIR.glob("*_artifact_manifest.csv")):
        if manifest.name.startswith("d15_"):
            continue
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"relative_path", "bytes", "sha256"}
            if not required.issubset(reader.fieldnames or set()):
                raise ValueError(f"Unexpected manifest columns: {manifest}")
            for row in reader:
                add(row["relative_path"], manifest.name, int(row["bytes"]), row["sha256"])

    # D1 uses a provenance-specific schema rather than the later artifact schema.
    d1_manifest = LOG_DIR / "d1_file_manifest.csv"
    with d1_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            add(row["local_path"], d1_manifest.name, int(row["size_bytes"]), row["sha256"])

    def add_tree(root: Path) -> None:
        if not root.exists():
            return
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = relative_path(path)
            parts = set(path.relative_to(PROJECT_DIR).parts)
            if "__pycache__" in parts or ".staging" in parts:
                continue
            if rel.startswith("logs/d15_"):
                continue
            add(rel, "D15_tree_snapshot")

    for root in (CODE_DIR, CONFIG_DIR, DATA_DIR, FIGURE_DIR, LOG_DIR, MODEL_DIR, RESULT_DIR, PROJECT_DIR / "manuscript"):
        add_tree(root)
    for root_file in (
        REQUIREMENTS_FILE,
        LOCK_FILE,
        PROJECT_DIR / "README.md",
        PROJECT_DIR / "run_d15_reproducibility.ps1",
    ):
        add(relative_path(root_file), "D15_project_files")
    return store


def validate_and_write_manifest() -> dict[str, Any]:
    store = _manifest_store()
    rows: list[dict[str, Any]] = []
    mismatch_count = 0
    total_bytes = 0
    started = time.perf_counter()
    items = sorted(store.items())
    for index, (rel, entry) in enumerate(items, start=1):
        path: Path = entry["path"]
        if not path.is_file():
            raise FileNotFoundError(rel)
        actual_bytes = path.stat().st_size
        expected_bytes = entry["expected_bytes"]
        if expected_bytes and expected_bytes != {actual_bytes}:
            mismatch_count += 1
            raise AssertionError(f"Size mismatch for {rel}: expected {expected_bytes}, observed {actual_bytes}")
        expected_hashes = entry["expected_sha256"]
        if len(expected_hashes) > 1:
            raise AssertionError(f"Conflicting frozen hashes for {rel}: {expected_hashes}")
        actual_hash = hash_file(path)
        if expected_hashes and actual_hash not in expected_hashes:
            mismatch_count += 1
            raise AssertionError(f"SHA-256 mismatch for {rel}: expected {expected_hashes}, observed {actual_hash}")
        total_bytes += actual_bytes
        rows.append(
            {
                "relative_path": rel,
                "bytes": actual_bytes,
                "sha256": actual_hash,
                "source": ";".join(sorted(entry["sources"])),
            }
        )
        if index == 1 or index % 50 == 0 or index == len(items):
            print(f"[HASH] {index}/{len(items)} files checked")

    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST_FILE.with_suffix(MANIFEST_FILE.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "bytes", "sha256", "source"])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(MANIFEST_FILE)
    return {
        "files": len(rows),
        "bytes": total_bytes,
        "seconds": round(time.perf_counter() - started, 3),
        "manifest_sha256": hash_file(MANIFEST_FILE),
        "mismatches": mismatch_count,
    }


def run_independent_tests() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(CODE_DIR) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    for variable in THREAD_ENVIRONMENT_VARIABLES:
        environment[variable] = "4"

    for relative_test in TEST_FILES:
        test_path = PROJECT_DIR / relative_test
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                [sys.executable, str(test_path)],
                cwd=PROJECT_DIR,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=900,
                check=False,
            )
            combined = (completed.stdout + "\n" + completed.stderr).strip()
            tail = combined[-2000:]
            passed = completed.returncode == 0
            result = {
                "file": relative_test,
                "status": "PASS" if passed else "FAIL",
                "returncode": completed.returncode,
                "seconds": round(time.perf_counter() - started, 3),
                "tail": tail,
            }
        except Exception as exc:
            result = {
                "file": relative_test,
                "status": "FAIL",
                "returncode": None,
                "seconds": round(time.perf_counter() - started, 3),
                "tail": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)
        print(f"[TEST {result['status']}] {relative_test} ({result['seconds']} s)")
    return results


def write_checkpoint(report: dict[str, Any]) -> None:
    passed = report["status"] == "D15_REPRODUCIBILITY_AUDIT_PASS"
    lines = [
        "# D15 reproducibility checkpoint",
        "",
        f"Status: **{report['status']}**",
        f"Audit time: {report['created_local']}",
        "",
        "## Scope",
        "",
        "- This checkpoint verifies the frozen D1-D14 artifacts and does not refit models.",
        "- Paths in the D15 runtime are resolved relative to the project root.",
        f"- Unified manifest: {report.get('manifest', {}).get('files', 0)} files, "
        f"{report.get('manifest', {}).get('bytes', 0)} bytes.",
        "",
        "## Checks",
        "",
    ]
    for check in report["checks"]:
        mark = "PASS" if check["status"] == "PASS" else "FAIL"
        lines.append(f"- [{mark}] {check['name']}: {check['detail']}")
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- H1 remains metric-dependent; no uniform random-split optimism claim is permitted.",
            "- H2 includes a fatal-recall trade-off; no uniform LightGBM superiority claim is permitted.",
            "- H3 is descriptive SHAP rank-change evidence, not a causal or binary instability claim.",
            "- The planned regional holdout was not executed and spatial generalization remains unverified.",
            "",
            "## Re-run",
            "",
            """Use `run_d15_reproducibility.ps1` for the full audit and independent checks.""",
        ]
    )
    if not passed:
        lines.append("")
        lines.append("At least one check failed; do not treat this project state as reproducible until the failure is resolved.")
    write_text_atomic(CHECKPOINT_FILE, "\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run all frozen standalone assertion scripts after the structural checks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    records: list[CheckRecord] = []
    test_results: list[dict[str, Any]] = []

    if args.run_tests:
        test_started = time.perf_counter()
        test_results = run_independent_tests()
        test_status = all(item["status"] == "PASS" for item in test_results)
        records.append(
            CheckRecord(
                "independent D1-D14 test suite",
                "PASS" if test_status else "FAIL",
                f"{sum(item['status'] == 'PASS' for item in test_results)}/{len(test_results)} standalone scripts passed",
                round(time.perf_counter() - test_started, 3),
            )
        )

    run_check("protocol contract", check_protocol_contract, records)
    run_check("data and split contract", check_data_contract, records)
    run_check("primary prediction contract", check_prediction_contract, records)
    run_check("frozen model smoke test", check_model_smoke, records)
    run_check("dependency contract", check_dependency_contract, records)
    run_check("path portability contract", check_portability_contract, records)

    manifest_summary: dict[str, Any] = {}
    try:
        manifest_summary = validate_and_write_manifest()
        records.append(
            CheckRecord(
                "artifact hash manifest",
                "PASS",
                f"{manifest_summary['files']} files and {manifest_summary['bytes']} bytes checked",
                float(manifest_summary["seconds"]),
            )
        )
        print(f"[PASS] artifact hash manifest: {manifest_summary['files']} files checked")
    except Exception as exc:
        records.append(CheckRecord("artifact hash manifest", "FAIL", f"{type(exc).__name__}: {exc}", 0.0))
        print(f"[FAIL] artifact hash manifest: {type(exc).__name__}: {exc}")

    status = "D15_REPRODUCIBILITY_AUDIT_PASS" if all(item.status == "PASS" for item in records) else "D15_REPRODUCIBILITY_AUDIT_FAIL"
    report = {
        "version": VERSION,
        "status": status,
        "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "project_root_resolution": "code/d15_reproducibility.py parent directory",
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "thread_environment": {name: os.environ.get(name) for name in THREAD_ENVIRONMENT_VARIABLES},
        "checks": [asdict(item) for item in records],
        "independent_tests": test_results,
        "manifest": manifest_summary,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "write_scope": [
            "logs/d15_reproducibility_report.json",
            "logs/d15_artifact_manifest.csv",
            "logs/d15_checkpoint.md",
            "logs/environment_info.txt when --run-tests is used",
        ],
        "interpretation": {
            "data_and_models": "Frozen artifacts were verified; no new model fit was performed.",
            "regional_holdout": "Not executed; spatial generalization remains unverified.",
        },
    }
    write_json_atomic(REPORT_FILE, report)
    write_checkpoint(report)
    print(f"D15_STATUS={status}")
    print(f"D15_REPORT={REPORT_FILE}")
    print(f"D15_CHECKPOINT={CHECKPOINT_FILE}")
    return 0 if status.endswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
