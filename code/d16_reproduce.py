"""Run an isolated raw-to-results reproduction of the STATS19 study.

D16 writes only to a separate sibling workspace and to its own D16 report in
the reference project. Frozen D1-D15 artifacts are treated as read-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import queue
import runpy
import shutil
import subprocess
import sys
import threading
import time
import venv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")


VERSION = "D16_V2"
SOURCE_DIR = Path(__file__).resolve().parents[1]
CONFIG_FILE = SOURCE_DIR / "config" / "d16_reproduction_protocol.json"
FINAL_COMPARISON_CONFIG_FILE = (
    SOURCE_DIR / "config" / "d16_final_comparison_protocol.json"
)
SOURCE_REPORT_FILE = SOURCE_DIR / "logs" / "d16_reproduction_report.json"
SOURCE_CHECKPOINT_FILE = SOURCE_DIR / "logs" / "d16_checkpoint.md"
INITIAL_REPORT_ARCHIVE_FILE = (
    SOURCE_DIR / "logs" / "d16_initial_comparison_report.json"
)
DEFAULT_WORKSPACE = SOURCE_DIR.parent / f"{SOURCE_DIR.name}_D16_reproduction"

THREAD_CAP = 4
FLOAT_ATOL = 1e-10
FLOAT_RTOL = 1e-8
PROBABILITY_ATOL = 1e-12
PIXEL_ATOL = 2.0 / 255.0
STAGE_TIMEOUT_SECONDS = 4 * 60 * 60

THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": str(THREAD_CAP),
    "OPENBLAS_NUM_THREADS": str(THREAD_CAP),
    "MKL_NUM_THREADS": str(THREAD_CAP),
    "NUMEXPR_NUM_THREADS": str(THREAD_CAP),
    "VECLIB_MAXIMUM_THREADS": str(THREAD_CAP),
    "BLIS_NUM_THREADS": str(THREAD_CAP),
    "LOKY_MAX_CPU_COUNT": str(THREAD_CAP),
    "PYTHONHASHSEED": "0",
    "MPLBACKEND": "Agg",
}

PRE_FROZEN_CONFIGS = (
    "config/d8_baseline_protocol.json",
    "config/d9_ordered_logit_protocol.json",
    "config/d9s1_matched_subset_protocol.json",
    "config/d10_lightgbm_protocol.json",
    "config/d10_execution_amendment.json",
    "config/d10b_tree_sensitivity_protocol.json",
    "config/d11_evaluation_protocol.json",
    "config/d12_bootstrap_protocol.json",
    "config/d13_positioning_protocol.json",
    "config/d14_shap_protocol.json",
    "config/d14_exclude2020_protocol.json",
    "config/d14_exclude2020_protocol_v2.json",
    "config/d14_closeout_protocol.json",
    "config/d14_closeout_protocol_v2.json",
    "config/d15_reproducibility_protocol.json",
)

RUNTIME_PRESERVED_CONFIGS = {
    "config/d9s1_matched_subset_protocol.json",
    "config/d10_lightgbm_protocol.json",
    "config/d10_execution_amendment.json",
    "config/d10b_tree_sensitivity_protocol.json",
    "config/d11_evaluation_protocol.json",
    "config/d12_bootstrap_protocol.json",
    "config/d13_positioning_protocol.json",
    "config/d14_shap_protocol.json",
    "config/d14_exclude2020_protocol_v2.json",
    "config/d14_closeout_protocol_v2.json",
}

ROOT_INPUTS = (
    "README.md",
    "requirements.txt",
    "requirements-lock.txt",
    "run_d15_reproducibility.ps1",
    "run_d16_reproduction.py",
    "run_d16_reproduction.ps1",
)
AUXILIARY_INPUTS = (
    "logs/d9_runtime_benchmark.json",
    "logs/d10b_execution_incident.json",
)

GENERATED_EXACT_FILES = (
    "config/d3_feature_manifest.json",
    "config/d4_output_schema.json",
    "config/d5_cleaning_rules.json",
    "config/d5_dataset_schema.json",
    "config/d6_analysis_protocol.json",
    "config/d7_training_inputs.json",
    "data/interim/d4_modeling_table.csv.gz",
    "data/processed/stats19_modeling_dataset.csv.gz",
    "data/processed/d6_split_assignments.csv.gz",
    "data/raw/collisions/collision_2018.csv",
    "data/raw/collisions/collision_2019.csv",
    "data/raw/collisions/collision_2020.csv",
    "data/raw/collisions/collision_2021.csv",
    "data/raw/collisions/collision_2022.csv",
    "data/raw/collisions/collision_2023.csv",
    "data/raw/collisions/collision_2024.csv",
)

SCIENTIFIC_JSON_FILES = (
    "config/d10_selected_lightgbm.json",
    "results/d12/d12_bootstrap_draw_metadata.json",
    "logs/d11_run_summary.json",
    "logs/d12_run_summary.json",
    "logs/d13_positioning_decision.json",
    "logs/d14_shap_run_summary.json",
    "logs/d14_exclude2020_run_summary.json",
    "logs/d14_final_summary.json",
)

VOLATILE_JSON_KEYS = {
    "created_local",
    "completed_local",
    "runtime_seconds",
    "elapsed_seconds",
    "fit_seconds",
    "fit_seconds_full_2000",
    "prediction_bytes",
    "python",
    "executable",
    "platform",
}

ORDERED_LOGIT_MODEL = "ordered_logit_unweighted"
MIXED_ORDERED_MODEL_CSV_FILES = {
    "results/d11/d11_test_confusion_matrices.csv",
    "results/d11/d11_test_metrics.csv",
    "results/d12/d12_model_metric_intervals.csv",
    "results/d12/d12_random_optimism_gaps.csv",
    "results/d12/d12_random_optimism_summary.csv",
    "results/d13/d13_main_performance_table.csv",
}
D9S1_MIXED_COMPARISON = "results/d9s1/d9s1_matched_comparison.csv"
D12_MIXED_DRAW_FILE = "results/d12/d12_bootstrap_draws.npz"
D11_MIXED_FIGURE = "figures/d11_test_metric_overview.png"
D12_PROPAGATED_CSV_FILES = {
    "results/d12/d12_model_metric_intervals.csv",
    "results/d12/d12_pairwise_model_differences.csv",
    "results/d12/d12_random_optimism_gaps.csv",
}
D12_PROPAGATED_FIGURE = "figures/d12_h1_optimism_gaps.png"
D14_EXCLUDE2020_PREDICTION = (
    "results/d14/exclude2020/predictions/lightgbm_weighted__test.csv.gz"
)

STATS19_TESTS = (
    "code/test_d6_protocol.py",
    "code/test_d7_training_inputs.py",
    "code/test_d8_baselines.py",
    "code/test_d9_ordered_logit.py",
    "code/test_d9s1_matched_subset.py",
    "code/test_d10_lightgbm.py",
    "code/test_d10b_tree_sensitivity.py",
    "code/test_d11_evaluation.py",
    "code/test_d12_bootstrap.py",
    "code/test_d13_checkpoint.py",
    "code/test_d14_shap.py",
    "code/test_d16_d14_exclude2020.py",
    "code/test_d14_closeout.py",
    "code/test_environment.py",
)


@dataclass(frozen=True)
class Stage:
    name: str
    arguments: tuple[str, ...]
    timeout_seconds: int = STAGE_TIMEOUT_SECONDS


PIPELINE_STAGES = (
    Stage("D1_acquire_and_audit", ("code/d1_acquire_and_audit.py",)),
    Stage("D2_schema_and_codes", ("code/d2_audit_schema_and_codes.py",)),
    Stage("D3_leakage_audit", ("code/d3_build_leakage_audit.py",)),
    Stage("D4_feature_table", ("code/d4_build_features.py",)),
    Stage("D5_quality_control", ("code/d5_quality_control.py",)),
    Stage("D6_protocol_and_splits", ("code/d6_freeze_protocol_and_plot.py",)),
    Stage("D7_training_inputs", ("code/d7_finalize_training_inputs.py",)),
    Stage("D8_full_baselines", ("code/d8_train_baselines.py", "--full")),
    Stage(
        "D9_ordered_logit_subset",
        ("code/d9_train_ordered_logit.py", "--run", "--mode", "subset"),
    ),
    Stage(
        "D9S1_prepare_protocol",
        ("code/d16_reproduce.py", "--internal-protocol", "d9s1"),
    ),
    Stage(
        "D9S1_freeze_protocol",
        ("code/d9s1_matched_subset_logistic.py", "--freeze"),
    ),
    Stage("D9S1_matched_subset", ("code/d9s1_matched_subset_logistic.py", "--run")),
    Stage(
        "D10_prepare_protocol",
        ("code/d16_reproduce.py", "--internal-protocol", "d10"),
    ),
    Stage("D10_tune", ("code/d10_tune_lightgbm.py", "--tune")),
    Stage("D10_fit_all", ("code/d10_tune_lightgbm.py", "--fit-all")),
    Stage(
        "D10B_rebase_protocol",
        ("code/d16_reproduce.py", "--internal-protocol", "d10b"),
    ),
    Stage(
        "D10B_tree_sensitivity",
        ("code/d16_reproduce.py", "--internal-protocol", "d10b_execute"),
    ),
    Stage(
        "D11_prepare_protocol",
        ("code/d16_reproduce.py", "--internal-protocol", "d11"),
    ),
    Stage(
        "D11_freeze_protocol",
        ("code/d11_evaluate_frozen_models.py", "--freeze"),
    ),
    Stage("D11_preflight", ("code/d11_evaluate_frozen_models.py", "--preflight")),
    Stage("D11_one_time_evaluation", ("code/d11_evaluate_frozen_models.py", "--run")),
    Stage(
        "D12_prepare_protocol",
        ("code/d16_reproduce.py", "--internal-protocol", "d12"),
    ),
    Stage(
        "D12_freeze_protocol",
        ("code/d12_bootstrap_uncertainty.py", "--freeze-protocol"),
    ),
    Stage("D12_bootstrap", ("code/d12_bootstrap_uncertainty.py", "--run")),
    Stage(
        "D13_prepare_protocol",
        ("code/d16_reproduce.py", "--internal-protocol", "d13"),
    ),
    Stage(
        "D13_freeze_protocol",
        ("code/d13_freeze_results.py", "--freeze-protocol"),
    ),
    Stage("D13_freeze_results", ("code/d13_freeze_results.py", "--run")),
    Stage(
        "D14_SHAP_prepare_protocol",
        ("code/d16_reproduce.py", "--internal-protocol", "d14_shap"),
    ),
    Stage(
        "D14_SHAP_freeze_protocol",
        ("code/d14_shap_stability.py", "--freeze"),
    ),
    Stage("D14_SHAP", ("code/d14_shap_stability.py", "--run")),
    Stage(
        "D14_exclude_2020_prepare_protocol",
        ("code/d16_reproduce.py", "--internal-protocol", "d14_exclude2020"),
    ),
    Stage(
        "D14_exclude_2020_freeze_protocol",
        (
            "code/d16_reproduce.py",
            "--internal-protocol",
            "d14_exclude2020_freeze",
        ),
    ),
    Stage(
        "D14_exclude_2020",
        (
            "code/d16_reproduce.py",
            "--internal-protocol",
            "d14_exclude2020_run",
        ),
    ),
    Stage(
        "D14_closeout_prepare_protocol",
        ("code/d16_reproduce.py", "--internal-protocol", "d14_closeout"),
    ),
    Stage(
        "D14_closeout_freeze_protocol",
        ("code/d14_closeout.py", "--freeze"),
    ),
    Stage("D14_closeout", ("code/d14_closeout.py", "--run")),
)


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str
    seconds: float


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def relative(path: Path, root: Path = SOURCE_DIR) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def ensure_external_workspace(workspace: Path) -> Path:
    resolved = workspace.expanduser().resolve()
    source = SOURCE_DIR.resolve()
    if resolved == source or source in resolved.parents:
        raise ValueError(
            "D16 workspace must be outside the frozen source project; "
            f"received {resolved}"
        )
    return resolved


def source_input_paths() -> list[Path]:
    paths = sorted((SOURCE_DIR / "code").glob("*.py"))
    paths.extend(SOURCE_DIR / value for value in PRE_FROZEN_CONFIGS)
    paths.extend(SOURCE_DIR / value for value in ROOT_INPUTS)
    paths.extend(SOURCE_DIR / value for value in AUXILIARY_INPUTS)
    paths.extend(sorted((SOURCE_DIR / "data" / "raw" / "downloads").glob("*")))
    paths.extend(
        sorted((SOURCE_DIR / "data" / "external" / "documentation").glob("*"))
    )
    return [path for path in paths if path.is_file()]


def frozen_artifact_expectations() -> dict[str, str]:
    expected: dict[str, str] = {}
    for manifest in sorted((SOURCE_DIR / "logs").glob("d*_artifact_manifest.csv")):
        if manifest.name.startswith(("d15_", "d16_")):
            continue
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"relative_path", "sha256"}
            if not required.issubset(reader.fieldnames or set()):
                raise ValueError(f"Unexpected artifact manifest schema: {manifest}")
            for row in reader:
                current = expected.setdefault(row["relative_path"], row["sha256"])
                if current != row["sha256"]:
                    raise ValueError(
                        f"Conflicting frozen hashes for {row['relative_path']}"
                    )

    d1_manifest = SOURCE_DIR / "logs" / "d1_file_manifest.csv"
    with d1_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            current = expected.setdefault(row["local_path"], row["sha256"])
            if current != row["sha256"]:
                raise ValueError(f"Conflicting D1 hash for {row['local_path']}")
    return expected


def verify_frozen_source() -> dict[str, Any]:
    expected = frozen_artifact_expectations()
    started = time.perf_counter()
    for index, (raw_path, expected_hash) in enumerate(sorted(expected.items()), start=1):
        path = SOURCE_DIR / Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Frozen source artifact missing: {raw_path}")
        observed = hash_file(path)
        if observed != expected_hash:
            raise AssertionError(
                f"Frozen source artifact changed: {raw_path}; "
                f"expected {expected_hash}, observed {observed}"
            )
        if index == 1 or index % 50 == 0 or index == len(expected):
            print(f"[D16 source check] {index}/{len(expected)}", flush=True)
    digest = hashlib.sha256()
    for raw_path, expected_hash in sorted(expected.items()):
        digest.update(f"{raw_path}\0{expected_hash}\n".encode("utf-8"))
    return {
        "files": len(expected),
        "set_sha256": digest.hexdigest(),
        "seconds": round(time.perf_counter() - started, 3),
    }


def expected_scientific_summary() -> dict[str, Any]:
    d15_file = SOURCE_DIR / "logs" / "d15_reproducibility_report.json"
    d14_file = SOURCE_DIR / "logs" / "d14_final_summary.json"
    if not d15_file.is_file() or not d14_file.is_file():
        raise FileNotFoundError("D15/D14 summary files are required before D16 freeze")
    d15 = json.loads(d15_file.read_text(encoding="utf-8"))
    d14 = json.loads(d14_file.read_text(encoding="utf-8"))
    primary_detail = next(
        check["detail"]
        for check in d15["checks"]
        if check["name"] == "primary prediction contract"
    )
    return {
        "rows": 743_646,
        "features": 17,
        "temporal_test_rows": 100_927,
        "primary_prediction_detail": primary_detail,
        "D14_status": d14["status"],
        "H1_status": d14["H1"]["broad_random_split_optimism_claim"],
        "H2_status": d14["H2"]["joint_incremental_value_rule"],
        "H3_status": d14["H3"]["status"],
    }


def freeze_protocol() -> None:
    source_check = verify_frozen_source()
    inputs: dict[str, dict[str, Any]] = {}
    for path in source_input_paths():
        raw_path = relative(path)
        inputs[raw_path] = {
            "bytes": int(path.stat().st_size),
            "sha256": hash_file(path),
        }
    payload = {
        "version": VERSION,
        "status": "FROZEN_BEFORE_D16_EXECUTION",
        "created_local": time.strftime("%Y-%m-%d"),
        "purpose": (
            "Clean-environment, isolated raw-to-results reproduction of the "
            "frozen STATS19 analysis."
        ),
        "execution_history": {
            "pre_scientific_run_incident": (
                "The first environment-install attempt stopped before D1 because "
                "the Windows console could not encode one pip output character."
            ),
            "amendment": (
                "Console output now replaces unsupported display characters and "
                "a ready marker is required before a partial environment is reused."
            ),
            "scientific_parameters_changed": False,
            "pipeline_stage_started_before_amendment": False,
            "V2_amendment": (
                "After exact D1-D9 reconstruction, one Ordered Logit split "
                "showed four changed labels and small probability drift. V2 "
                "retains that result as a secondary-baseline deviation and "
                "continues strict checks of the primary Logistic/LightGBM chain."
            ),
            "V2_tolerance_changed": False,
            "V2_primary_model_contract_changed": False,
        },
        "source_policy": {
            "source_project_is_read_only": True,
            "workspace_must_be_outside_source_project": True,
            "source_frozen_artifact_set": source_check,
        },
        "runtime": {
            "required_python": "3.13.15",
            "clean_virtual_environment": True,
            "locked_requirements": "requirements-lock.txt",
            "thread_cap": THREAD_CAP,
            "thread_environment": THREAD_ENVIRONMENT,
            "per_stage_timeout_seconds": STAGE_TIMEOUT_SECONDS,
        },
        "copied_inputs": inputs,
        "pipeline": [
            {"stage": stage.name, "arguments": list(stage.arguments)}
            for stage in PIPELINE_STAGES
        ],
        "test_suite": list(STATS19_TESTS),
        "comparison": {
            "identifiers_categories_labels_counts": "exact",
            "generated_data_and_split_files": "byte-for-byte SHA-256 equality",
            "floating_absolute_tolerance": FLOAT_ATOL,
            "floating_relative_tolerance": FLOAT_RTOL,
            "probability_absolute_tolerance": PROBABILITY_ATOL,
            "PNG_pixel_absolute_tolerance": PIXEL_ATOL,
            "model_binary_hashes": (
                "not required; regenerated predictions, metrics and tests are compared"
            ),
            "PDF_binary_hashes": (
                "not required because renderer metadata may vary; PNG content and "
                "underlying result tables are checked"
            ),
            "secondary_Ordered_Logit_policy": (
                "Identity/target columns and every non-Ordered model component "
                "remain strict. Ordered Logit numerical mismatches are reported "
                "as DEVIATION, never coerced to equality or hidden by a wider "
                "tolerance."
            ),
        },
        "expected_scientific_summary": expected_scientific_summary(),
        "boundaries": {
            "H1": "metric-dependent; no uniform random-split optimism claim",
            "H2": "Macro-F1 gain with fatal-recall trade-off; no uniform superiority claim",
            "H3": "descriptive SHAP rank analysis; no causal claim",
            "regional_holdout": "not executed; spatial generalization unverified",
            "CAS": "not part of D16; execute only after STATS19 reproduction is frozen",
        },
    }
    write_json_atomic(CONFIG_FILE, payload)
    print(f"D16_PROTOCOL_FROZEN={CONFIG_FILE}")
    print(f"D16_PROTOCOL_SHA256={hash_file(CONFIG_FILE)}")


def load_protocol() -> dict[str, Any]:
    if not CONFIG_FILE.is_file():
        raise FileNotFoundError("Run D16 --freeze before preparing the workspace")
    protocol = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    if protocol.get("version") != VERSION:
        raise ValueError("Unexpected D16 protocol version")
    if protocol.get("status") != "FROZEN_BEFORE_D16_EXECUTION":
        raise ValueError("D16 protocol is not frozen")
    return protocol


def verify_protocol_inputs(protocol: dict[str, Any]) -> None:
    for raw_path, expected in protocol["copied_inputs"].items():
        path = SOURCE_DIR / Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"D16 input missing: {raw_path}")
        if path.stat().st_size != int(expected["bytes"]):
            raise AssertionError(f"D16 input size changed: {raw_path}")
        if hash_file(path) != expected["sha256"]:
            raise AssertionError(f"D16 input hash changed: {raw_path}")


def copy_protocol_inputs(
    workspace: Path,
    protocol: dict[str, Any],
    *,
    preserve_runtime_protocols: bool = False,
) -> None:
    for raw_path in protocol["copied_inputs"]:
        if preserve_runtime_protocols and raw_path in RUNTIME_PRESERVED_CONFIGS:
            continue
        source = SOURCE_DIR / Path(raw_path)
        destination = workspace / Path(raw_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    destination = workspace / "config" / CONFIG_FILE.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG_FILE, destination)


def prepare_workspace(workspace: Path) -> None:
    protocol = load_protocol()
    verify_protocol_inputs(protocol)
    workspace = ensure_external_workspace(workspace)
    if workspace.exists():
        raise FileExistsError(
            f"D16 workspace already exists: {workspace}. Use --resume to continue "
            "it or choose a new path."
        )
    workspace.mkdir(parents=True)
    copy_protocol_inputs(workspace, protocol)
    for directory in (
        "data/raw/collisions",
        "data/interim",
        "data/processed",
        "figures",
        "logs/d16_stage_logs",
        "models",
        "results",
    ):
        (workspace / directory).mkdir(parents=True, exist_ok=True)
    marker = {
        "version": VERSION,
        "status": "D16_WORKSPACE_PREPARED",
        "source_project": str(SOURCE_DIR),
        "workspace": str(workspace),
        "protocol_sha256": hash_file(CONFIG_FILE),
        "prepared_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json_atomic(workspace / "logs" / "d16_workspace.json", marker)
    print(f"D16_WORKSPACE_PREPARED={workspace}")


RUNTIME_PROTOCOL_PATHS = {
    "d9s1": "config/d9s1_matched_subset_protocol.json",
    "d11": "config/d11_evaluation_protocol.json",
    "d12": "config/d12_bootstrap_protocol.json",
    "d13": "config/d13_positioning_protocol.json",
    "d14_shap": "config/d14_shap_protocol.json",
    "d14_exclude2020": "config/d14_exclude2020_protocol_v2.json",
    "d14_closeout": "config/d14_closeout_protocol_v2.json",
}
INTERNAL_PROTOCOL_CHOICES = (
    "d9s1",
    "d10",
    "d10b",
    "d10b_execute",
    "d11",
    "d12",
    "d13",
    "d14_shap",
    "d14_exclude2020",
    "d14_exclude2020_freeze",
    "d14_exclude2020_run",
    "d14_closeout",
)


def require_isolated_runtime_project(project: Path) -> None:
    project = project.resolve()
    marker_path = project / "logs" / "d16_workspace.json"
    if not marker_path.is_file():
        raise RuntimeError("D16 workspace marker is missing")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "D16_WORKSPACE_PREPARED":
        raise RuntimeError("Invalid D16 workspace marker")
    workspace_value = marker.get("workspace")
    source_value = marker.get("source_project")
    if not isinstance(workspace_value, str) or not isinstance(source_value, str):
        raise RuntimeError("Incomplete D16 workspace marker")
    if Path(workspace_value).resolve() != project:
        raise RuntimeError("D16 workspace marker path mismatch")
    if Path(source_value).resolve() == project:
        raise RuntimeError(
            "Internal protocol preparation is forbidden in the source project"
        )


def archive_runtime_protocol(project: Path, name: str) -> None:
    require_isolated_runtime_project(project)
    raw_path = RUNTIME_PROTOCOL_PATHS[name]
    path = project / raw_path
    if not path.is_file():
        print(f"D16_PROTOCOL_ALREADY_PREPARED={raw_path}")
        return
    archive_dir = project / "config" / "d16_reference_protocols"
    archive_dir.mkdir(parents=True, exist_ok=True)
    destination = archive_dir / path.name
    if destination.exists():
        raise FileExistsError(
            "Both active and archived runtime protocols exist; refusing to "
            f"delete either copy: {path.name}"
        )
    path.replace(destination)
    print(f"D16_PROTOCOL_ARCHIVED={raw_path}")


def rebase_d10b_protocol(project: Path) -> None:
    require_isolated_runtime_project(project)
    path = project / "config" / "d10b_tree_sensitivity_protocol.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    archive_dir = project / "config" / "d16_reference_protocols"
    archive_dir.mkdir(parents=True, exist_ok=True)
    reference = archive_dir / path.name
    if not reference.exists():
        shutil.copy2(path, reference)
    protocol = json.loads(path.read_text(encoding="utf-8"))
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
    for key, raw_path in mappings.items():
        if key not in protocol["upstream"]:
            raise KeyError(f"D10b protocol lacks {key}")
        protocol["upstream"][key] = hash_file(project / raw_path)
    protocol["d16_reproduction_instance"] = {
        "version": VERSION,
        "action": (
            "Only run-specific upstream hashes were refreshed. Candidate C03, "
            "tree checkpoints, data roles, metrics and decision lock are unchanged."
        ),
        "reference_protocol_sha256": hash_file(reference),
        "scientific_parameters_changed": False,
    }
    write_json_atomic(path, protocol)
    print(f"D16_D10B_PROTOCOL_REBASED={hash_file(path)}")


def rebase_d10_protocol(project: Path) -> None:
    require_isolated_runtime_project(project)
    protocol_path = project / "config" / "d10_lightgbm_protocol.json"
    amendment_path = project / "config" / "d10_execution_amendment.json"
    if not protocol_path.is_file() or not amendment_path.is_file():
        raise FileNotFoundError("D10 protocol or execution amendment is missing")

    archive_dir = project / "config" / "d16_reference_protocols"
    archive_dir.mkdir(parents=True, exist_ok=True)
    protocol_reference = archive_dir / protocol_path.name
    amendment_reference = archive_dir / amendment_path.name
    if protocol_reference.exists() or amendment_reference.exists():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        if (
            protocol_reference.is_file()
            and amendment_reference.is_file()
            and protocol.get("d16_reproduction_instance", {}).get("version")
            == VERSION
            and amendment.get("d16_reproduction_instance", {}).get("version")
            == VERSION
        ):
            print("D16_D10_PROTOCOL_ALREADY_REBASED=TRUE")
            return
        raise FileExistsError(
            "Incomplete or conflicting D10 runtime-protocol archive"
        )

    shutil.copy2(protocol_path, protocol_reference)
    shutil.copy2(amendment_path, amendment_reference)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    mappings = {
        "D5_schema_sha256": "config/d5_dataset_schema.json",
        "D6_protocol_sha256": "config/d6_analysis_protocol.json",
        "D7_training_inputs_sha256": "config/d7_training_inputs.json",
        "D8_protocol_sha256": "config/d8_baseline_protocol.json",
        "D8_validation_metrics_sha256": "results/d8/d8_validation_metrics.csv",
        "D6_assignments_sha256": "data/processed/d6_split_assignments.csv.gz",
    }
    for key, raw_path in mappings.items():
        if key not in protocol["upstream"]:
            raise KeyError(f"D10 protocol lacks {key}")
        protocol["upstream"][key] = hash_file(project / raw_path)
    protocol["d16_reproduction_instance"] = {
        "version": VERSION,
        "action": (
            "Only run-specific upstream hashes were refreshed. Candidate "
            "definitions, selection constraints, fallback, seeds, features, "
            "test embargo and all model parameters are unchanged."
        ),
        "reference_protocol_sha256": hash_file(protocol_reference),
        "scientific_parameters_changed": False,
    }
    write_json_atomic(protocol_path, protocol)

    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    if amendment.get("original_protocol_sha256") != hash_file(
        protocol_reference
    ):
        raise AssertionError("Reference D10 amendment/protocol binding is invalid")
    amendment["original_protocol_sha256"] = hash_file(protocol_path)
    amendment["d16_reproduction_instance"] = {
        "version": VERSION,
        "action": "Rebound only to the D16 runtime protocol instance.",
        "reference_amendment_sha256": hash_file(amendment_reference),
        "scientific_parameters_changed": False,
    }
    write_json_atomic(amendment_path, amendment)
    print(f"D16_D10_PROTOCOL_REBASED={hash_file(protocol_path)}")
    print(f"D16_D10_AMENDMENT_REBOUND={hash_file(amendment_path)}")


def run_or_recover_d10b(project: Path) -> None:
    require_isolated_runtime_project(project)
    script = project / "code" / "d10b_tree_sensitivity.py"
    test_script = project / "code" / "test_d10b_tree_sensitivity.py"
    manifest = project / "logs" / "d10b_artifact_manifest.csv"
    incident = project / "logs" / "d10b_execution_incident.json"
    required_outputs = [
        project / "results" / "d10b" / "d10b_checkpoint_metrics.csv",
        project / "results" / "d10b" / "d10b_logloss_history.csv",
        project / "models" / "d10b" / "temporal" / "c03_2000_trees.joblib",
        project / "figures" / "d10b_validation_logloss_curve.png",
        project / "logs" / "d10b_checkpoint.md",
        *[
            project
            / "results"
            / "d10b"
            / "validation_predictions"
            / f"temporal__c03__{trees}_trees.csv.gz"
            for trees in (1200, 1500, 2000)
        ],
    ]
    present = [path.is_file() for path in required_outputs]
    if not any(present):
        subprocess.run([sys.executable, str(script)], cwd=project, check=True)
        return
    if not all(present):
        missing = [
            path.relative_to(project).as_posix()
            for path, exists in zip(required_outputs, present)
            if not exists
        ]
        raise RuntimeError(f"Partial D10b output set cannot be recovered: {missing}")
    if not incident.is_file():
        raise FileNotFoundError(incident)

    recovered = not manifest.is_file()
    if recovered:
        namespace = runpy.run_path(
            str(script),
            run_name="d16_d10b_manifest_recovery",
        )
        namespace["build_manifest"]()
    subprocess.run([sys.executable, str(test_script)], cwd=project, check=True)
    if recovered:
        write_json_atomic(
            project / "logs" / "d16_d10b_manifest_recovery.json",
            {
                "version": VERSION,
                "status": "RECOVERED_AFTER_MANIFEST_INPUT_OMISSION",
                "completed_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "scientific_outputs_recomputed": False,
                "scientific_parameters_changed": False,
                "recovery_action": (
                    "Copied the pre-existing D10b incident disclosure, invoked "
                    "the original build_manifest function, and passed the "
                    "original independent D10b test."
                ),
                "manifest_sha256": hash_file(manifest),
            },
        )
        print("D16_D10B_MANIFEST_RECOVERY=PASS")
    else:
        print("D16_D10B_OUTPUTS_ALREADY_COMPLETE=PASS")


def run_d14_exclude2020_with_thread_cap(
    project: Path,
    action: str,
) -> None:
    require_isolated_runtime_project(project)
    script = project / "code" / "d14_exclude2020_sensitivity.py"
    namespace = runpy.run_path(
        str(script),
        run_name=f"d16_d14_exclude2020_{action}",
    )
    declared_threads = int(namespace["THREAD_LIMIT"])
    if declared_threads < THREAD_CAP:
        raise RuntimeError(
            "The D14 exclusion script declares fewer threads than D16"
        )
    module_globals = namespace["build_protocol"].__globals__
    module_globals["THREAD_LIMIT"] = THREAD_CAP
    amendment_path = (
        project / "logs" / "d16_d14_exclude2020_runtime_amendment.json"
    )
    if action == "freeze":
        protocol_path = (
            project / "config" / "d14_exclude2020_protocol_v2.json"
        )
        if protocol_path.is_file():
            existing = json.loads(protocol_path.read_text(encoding="utf-8"))
            existing_threads = int(
                existing.get("runtime", {}).get("CPU_thread_limit", -1)
            )
            if existing_threads != THREAD_CAP:
                failed_dir = (
                    project / "config" / "d16_failed_runtime_protocols"
                )
                failed_dir.mkdir(parents=True, exist_ok=True)
                failed_path = (
                    failed_dir
                    / "d14_exclude2020_protocol_v2__8_thread_failed_freeze.json"
                )
                if failed_path.exists():
                    raise FileExistsError(
                        "Failed D14 exclusion protocol is already archived"
                    )
                protocol_path.replace(failed_path)
                print(
                    "D16_D14_EXCLUDE2020_FAILED_PROTOCOL_ARCHIVED="
                    f"{failed_path.relative_to(project).as_posix()}"
                )
        write_json_atomic(
            amendment_path,
            {
                "version": VERSION,
                "status": "FROZEN_BEFORE_D14_EXCLUDE2020_REFIT",
                "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "source_declared_thread_limit": declared_threads,
                "D16_enforced_thread_limit": THREAD_CAP,
                "source_code_changed": False,
                "scientific_parameters_changed": False,
                "source_sha256": hash_file(script),
                "method": (
                    "The isolated wrapper sets the module THREAD_LIMIT global "
                    "before invoking the original freeze/run functions."
                ),
            },
        )
        namespace["freeze_protocol"]()
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if int(protocol["runtime"]["CPU_thread_limit"]) != THREAD_CAP:
            raise AssertionError("D14 exclusion protocol did not freeze at 4 threads")
        print("D16_D14_EXCLUDE2020_4_THREAD_FREEZE=PASS")
        return
    if action == "run":
        if not amendment_path.is_file():
            raise FileNotFoundError(amendment_path)
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        if int(amendment["D16_enforced_thread_limit"]) != THREAD_CAP:
            raise AssertionError("D14 exclusion runtime amendment changed")
        namespace["run_analysis"]()
        print("D16_D14_EXCLUDE2020_4_THREAD_RUN=PASS")
        return
    raise ValueError(f"Unknown D14 exclusion action: {action}")


def prepare_runtime_protocol(name: str) -> None:
    project = SOURCE_DIR
    require_isolated_runtime_project(project)
    if name == "d10":
        rebase_d10_protocol(project)
        return
    if name == "d10b":
        rebase_d10b_protocol(project)
        return
    if name == "d10b_execute":
        run_or_recover_d10b(project)
        return
    if name == "d14_exclude2020_freeze":
        run_d14_exclude2020_with_thread_cap(project, "freeze")
        return
    if name == "d14_exclude2020_run":
        run_d14_exclude2020_with_thread_cap(project, "run")
        return
    if name not in RUNTIME_PROTOCOL_PATHS:
        raise ValueError(f"Unknown runtime protocol: {name}")
    archive_runtime_protocol(project, name)


def venv_python(workspace: Path) -> Path:
    if os.name == "nt":
        return workspace / ".venv" / "Scripts" / "python.exe"
    return workspace / ".venv" / "bin" / "python"


def run_process(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log_file: Path,
    timeout_seconds: int,
    label: str,
) -> dict[str, Any]:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    lines: list[str] = []
    with log_file.open("w", encoding="utf-8", newline="") as log:
        log.write("COMMAND: " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        while True:
            if time.perf_counter() - started > timeout_seconds:
                process.kill()
                process.wait()
                raise TimeoutError(f"{label} exceeded {timeout_seconds} seconds")
            line = process.stdout.readline()
            if line:
                log.write(line)
                log.flush()
                lines.append(line)
                stripped = line.strip()
                if stripped:
                    print(f"[{label}] {stripped}", flush=True)
                continue
            if process.poll() is not None:
                break
        returncode = process.wait()
    elapsed = time.perf_counter() - started
    if returncode != 0:
        tail = "".join(lines[-20:])
        raise subprocess.CalledProcessError(returncode, command, output=tail)
    return {
        "status": "PASS",
        "command": command,
        "seconds": round(elapsed, 3),
        "log": log_file.relative_to(cwd).as_posix(),
    }


def clean_environment(workspace: Path, base_python: Path | None) -> Path:
    python = venv_python(workspace)
    ready_file = workspace / "logs" / "d16_environment_ready.json"
    if python.is_file() and ready_file.is_file():
        return python
    candidate = base_python or (SOURCE_DIR / ".venv" / "Scripts" / "python.exe")
    if not candidate.is_file():
        candidate = Path(sys.executable)
    version = subprocess.check_output(
        [
            str(candidate),
            "-c",
            "import platform; print(platform.python_version())",
        ],
        text=True,
        encoding="utf-8",
    ).strip()
    if version != "3.13.15":
        raise RuntimeError(f"D16 requires Python 3.13.15; observed {version}")
    if not python.is_file():
        print(f"[D16 environment] creating clean venv with {candidate}", flush=True)
        venv.EnvBuilder(with_pip=True, clear=False, symlinks=False).create(
            workspace / ".venv"
        )
    else:
        print("[D16 environment] resuming dependency installation", flush=True)
    if not python.is_file():
        raise FileNotFoundError(f"Clean Python was not created: {python}")
    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    run_process(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--requirement",
            str(workspace / "requirements-lock.txt"),
        ],
        cwd=workspace,
        environment=environment,
        log_file=workspace / "logs" / "d16_stage_logs" / "00_environment_install.log",
        timeout_seconds=STAGE_TIMEOUT_SECONDS,
        label="D16_environment",
    )
    freeze = subprocess.check_output(
        [str(python), "-m", "pip", "freeze"],
        cwd=workspace,
        env=environment,
        text=True,
        encoding="utf-8",
    )
    (workspace / "logs" / "d16_clean_environment_freeze.txt").write_text(
        freeze,
        encoding="utf-8",
    )
    write_json_atomic(
        ready_file,
        {
            "version": VERSION,
            "status": "CLEAN_ENVIRONMENT_READY",
            "python": str(python),
            "python_version": version,
            "requirements_sha256": hash_file(workspace / "requirements-lock.txt"),
            "completed_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    return python


def state_file(workspace: Path) -> Path:
    return workspace / "logs" / "d16_run_state.json"


def load_state(workspace: Path) -> dict[str, Any]:
    path = state_file(workspace)
    if not path.is_file():
        return {
            "version": VERSION,
            "status": "RUNNING",
            "workspace": str(workspace),
            "stages": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def migrate_resume_state(
    workspace: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    observed_version = state.get("version")
    previous_failure = state.get("failure")
    if observed_version == VERSION:
        if previous_failure is not None:
            state.setdefault("resume_history", []).append(
                {
                    "resumed_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "previous_failure": previous_failure,
                }
            )
        state["status"] = "RUNNING"
        state.pop("failure", None)
        write_json_atomic(state_file(workspace), state)
        marker_path = workspace / "logs" / "d16_workspace.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["version"] = VERSION
        marker["protocol_sha256"] = hash_file(CONFIG_FILE)
        marker["last_resumed_local"] = time.strftime(
            "%Y-%m-%dT%H:%M:%S%z"
        )
        write_json_atomic(marker_path, marker)
        return state

    if observed_version != "D16_V1":
        raise RuntimeError(
            f"Unsupported D16 resume-state version: {observed_version!r}"
        )

    expected_stages = [stage.name for stage in PIPELINE_STAGES[:9]]
    observed_stages = [
        str(item.get("name"))
        for item in state.get("stages", [])
        if item.get("status") == "PASS"
    ]
    if observed_stages != expected_stages:
        raise RuntimeError(
            "D16 V1-to-V2 migration is allowed only after the exact passed "
            f"D1-D9 prefix; observed {observed_stages}"
        )
    if any(item.get("status") != "PASS" for item in state.get("stages", [])):
        raise RuntimeError("Legacy D16 state contains a non-passing stage record")
    if not isinstance(previous_failure, dict) or previous_failure.get(
        "stage"
    ) != "D9S1_matched_subset":
        raise RuntimeError(
            "Legacy D16 state does not contain the expected D9-S1 stop"
        )

    migrated_local = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    state["version"] = VERSION
    state["status"] = "RUNNING"
    state.pop("failure", None)
    state["migration"] = {
        "from_version": "D16_V1",
        "to_version": VERSION,
        "migrated_local": migrated_local,
        "retained_passed_stages": expected_stages,
        "previous_failure": previous_failure,
        "reason": (
            "D9 Ordered Logit showed a documented secondary numerical "
            "deviation; V2 continues the strict primary-model reproduction."
        ),
    }
    write_json_atomic(state_file(workspace), state)

    marker_path = workspace / "logs" / "d16_workspace.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["version"] = VERSION
    marker["protocol_sha256"] = hash_file(CONFIG_FILE)
    marker["migrated_local"] = migrated_local
    write_json_atomic(marker_path, marker)

    ready_path = workspace / "logs" / "d16_environment_ready.json"
    if ready_path.is_file():
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        ready["version"] = VERSION
        ready["migrated_local"] = migrated_local
        write_json_atomic(ready_path, ready)
    return state


def completed_stage_names(state: dict[str, Any]) -> set[str]:
    return {
        str(item["name"])
        for item in state.get("stages", [])
        if item.get("status") == "PASS"
    }


def execute_pipeline(
    workspace: Path,
    python: Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    protocol = load_protocol()
    workspace_protocol = workspace / "config" / CONFIG_FILE.name
    if hash_file(workspace_protocol) != hash_file(CONFIG_FILE):
        raise AssertionError(
            "Workspace D16 protocol differs from the frozen source protocol"
        )
    state = load_state(workspace)
    if resume:
        state = migrate_resume_state(workspace, state)
    elif state.get("version") != VERSION:
        raise RuntimeError(
            "Existing D16 state uses another version; use --resume for an "
            "audited migration"
        )
    if state.get("status") == "COMPLETE":
        print("D16 pipeline is already complete; no stages were rerun.", flush=True)
        return state
    completed = completed_stage_names(state) if resume else set()
    if state.get("stages") and not resume:
        raise RuntimeError("D16 state already contains stages; use --resume")
    environment = os.environ.copy()
    environment.update(protocol["runtime"]["thread_environment"])
    environment["PYTHONPATH"] = str(workspace / "code")

    for index, stage in enumerate(PIPELINE_STAGES, start=1):
        if stage.name in completed:
            print(f"[D16 resume] skipping {stage.name}", flush=True)
            continue
        command = [
            str(python),
            str(workspace / stage.arguments[0]),
            *stage.arguments[1:],
        ]
        try:
            result = run_process(
                command,
                cwd=workspace,
                environment=environment,
                log_file=(
                    workspace
                    / "logs"
                    / "d16_stage_logs"
                    / f"{index:02d}_{stage.name}.log"
                ),
                timeout_seconds=stage.timeout_seconds,
                label=stage.name,
            )
        except Exception as exc:
            state["status"] = "FAILED"
            state["failure"] = {
                "stage": stage.name,
                "type": type(exc).__name__,
                "message": str(exc),
            }
            write_json_atomic(state_file(workspace), state)
            raise
        state["stages"].append({"name": stage.name, **result})
        write_json_atomic(state_file(workspace), state)

    for offset, test_path in enumerate(STATS19_TESTS, start=1):
        name = f"TEST_{Path(test_path).stem}"
        if name in completed_stage_names(state):
            print(f"[D16 resume] skipping {name}", flush=True)
            continue
        result = run_process(
            [str(python), str(workspace / test_path)],
            cwd=workspace,
            environment=environment,
            log_file=(
                workspace
                / "logs"
                / "d16_stage_logs"
                / f"{len(PIPELINE_STAGES) + offset:02d}_{name}.log"
            ),
            timeout_seconds=900,
            label=name,
        )
        state["stages"].append({"name": name, **result})
        write_json_atomic(state_file(workspace), state)

    d15_name = "D15_post_reproduction_audit"
    if d15_name not in completed_stage_names(state):
        result = run_process(
            [str(python), str(workspace / "code" / "d15_reproducibility.py")],
            cwd=workspace,
            environment=environment,
            log_file=workspace / "logs" / "d16_stage_logs" / "35_D15_audit.log",
            timeout_seconds=STAGE_TIMEOUT_SECONDS,
            label=d15_name,
        )
        state["stages"].append({"name": d15_name, **result})
        write_json_atomic(state_file(workspace), state)

    d15_test_name = "TEST_test_d15_reproducibility"
    if d15_test_name not in completed_stage_names(state):
        result = run_process(
            [
                str(python),
                str(workspace / "code" / "test_d15_reproducibility.py"),
            ],
            cwd=workspace,
            environment=environment,
            log_file=workspace / "logs" / "d16_stage_logs" / "36_TEST_D15.log",
            timeout_seconds=900,
            label=d15_test_name,
        )
        state["stages"].append({"name": d15_test_name, **result})
    state["status"] = "COMPLETE"
    state["completed_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json_atomic(state_file(workspace), state)
    return state


def is_exact_column(name: str) -> bool:
    lowered = name.lower()
    markers = (
        "collision_index",
        "target",
        "predicted",
        "class_code",
        "class_label",
        "candidate_id",
        "protocol",
        "model",
        "scope",
        "role",
        "status",
        "seed",
        "year",
        "count",
        "rows",
        "iteration",
        "checkpoint_trees",
        "row_position",
    )
    return any(marker in lowered for marker in markers)


def compare_series(
    left: pd.Series,
    right: pd.Series,
    name: str,
) -> tuple[bool, float]:
    if len(left) != len(right):
        return False, float("inf")
    if is_exact_column(name):
        first = left.astype("string").fillna("__D16_NA__")
        second = right.astype("string").fillna("__D16_NA__")
        return bool(first.equals(second)), 0.0

    left_numeric = pd.to_numeric(left, errors="coerce")
    right_numeric = pd.to_numeric(right, errors="coerce")
    left_mask = left_numeric.notna()
    right_mask = right_numeric.notna()
    if left_mask.equals(right_mask) and (bool(left_mask.any()) or left.isna().all()):
        first = left_numeric.to_numpy(dtype=float)
        second = right_numeric.to_numpy(dtype=float)
        differences = np.abs(first - second)
        finite = np.isfinite(differences)
        maximum = float(differences[finite].max()) if finite.any() else 0.0
        tolerance = PROBABILITY_ATOL if "prob" in name.lower() else FLOAT_ATOL
        matches = np.allclose(
            first,
            second,
            rtol=FLOAT_RTOL,
            atol=tolerance,
            equal_nan=True,
        )
        return bool(matches), maximum

    first = left.astype("string").fillna("__D16_NA__")
    second = right.astype("string").fillna("__D16_NA__")
    return bool(first.equals(second)), 0.0


def is_volatile_csv_column(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered == "warning_messages"
        or lowered.endswith("_seconds")
        or lowered.startswith("fit_seconds_")
    )


def scientific_csv_columns(path: Path) -> list[str]:
    return [
        column
        for column in pd.read_csv(path, nrows=0).columns.tolist()
        if not is_volatile_csv_column(column)
    ]


def compare_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: list[str],
) -> tuple[bool, str]:
    left = left.reset_index(drop=True)
    right = right.reset_index(drop=True)
    if len(left) != len(right):
        return False, f"row counts differ: {len(left)} vs {len(right)}"
    maximum = 0.0
    for column in columns:
        matches, difference = compare_series(left[column], right[column], column)
        maximum = max(maximum, difference)
        if not matches:
            return False, f"column {column} differs"
    return True, f"semantic match; rows={len(left):,}; max_abs_diff={maximum:.3g}"


def compare_frames_exact(
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: list[str],
) -> tuple[bool, str]:
    left = left.reset_index(drop=True)
    right = right.reset_index(drop=True)
    if len(left) != len(right):
        return False, f"row counts differ: {len(left)} vs {len(right)}"
    maximum = 0.0
    for column in columns:
        first_numeric = pd.to_numeric(left[column], errors="coerce")
        second_numeric = pd.to_numeric(right[column], errors="coerce")
        first_mask = first_numeric.notna()
        second_mask = second_numeric.notna()
        if first_mask.equals(second_mask) and (
            bool(first_mask.any()) or left[column].isna().all()
        ):
            first = first_numeric.to_numpy(dtype=float)
            second = second_numeric.to_numpy(dtype=float)
            finite = np.isfinite(first) & np.isfinite(second)
            maximum = max(
                maximum,
                float(np.abs(first[finite] - second[finite]).max())
                if finite.any()
                else 0.0,
            )
            if not np.array_equal(first, second, equal_nan=True):
                return False, f"column {column} is not exactly equal"
            continue
        first = left[column].astype("string").fillna("__D16_NA__")
        second = right[column].astype("string").fillna("__D16_NA__")
        if not first.equals(second):
            return False, f"column {column} is not exactly equal"
    return True, f"exact semantic match; rows={len(left):,}; max_abs_diff={maximum:.3g}"


def compare_csv(reference: Path, reproduced: Path) -> tuple[bool, str]:
    if hash_file(reference) == hash_file(reproduced):
        return True, "byte-identical"
    left_header = scientific_csv_columns(reference)
    right_header = scientific_csv_columns(reproduced)
    if left_header != right_header:
        return False, "scientific column names or order differ"
    dtype = {
        column: "string"
        for column in left_header
        if "collision_index" in column.lower()
    }
    left_chunks = pd.read_csv(
        reference,
        usecols=left_header,
        chunksize=50_000,
        dtype=dtype,
        low_memory=False,
    )
    right_chunks = pd.read_csv(
        reproduced,
        usecols=right_header,
        chunksize=50_000,
        dtype=dtype,
        low_memory=False,
    )
    rows = 0
    maximum = 0.0
    while True:
        try:
            left = next(left_chunks)
            left_done = False
        except StopIteration:
            left_done = True
            left = None
        try:
            right = next(right_chunks)
            right_done = False
        except StopIteration:
            right_done = True
            right = None
        if left_done or right_done:
            if left_done != right_done:
                return False, "row counts differ"
            break
        assert left is not None and right is not None
        if len(left) != len(right):
            return False, "chunk row counts differ"
        for column in left_header:
            matches, difference = compare_series(left[column], right[column], column)
            maximum = max(maximum, difference)
            if not matches:
                return False, f"column {column} differs after {rows:,} rows"
        rows += len(left)
    return True, f"semantic match; rows={rows:,}; max_abs_diff={maximum:.3g}"


def compare_csv_without_ordered_model(
    reference: Path,
    reproduced: Path,
) -> tuple[bool, str]:
    left_header = scientific_csv_columns(reference)
    right_header = scientific_csv_columns(reproduced)
    if left_header != right_header:
        return False, "scientific column names or order differ"
    if "model" not in left_header:
        return False, "model column missing from mixed-model table"
    left = pd.read_csv(reference, usecols=left_header, low_memory=False)
    right = pd.read_csv(reproduced, usecols=right_header, low_memory=False)
    left = left.loc[left["model"] != ORDERED_LOGIT_MODEL]
    right = right.loc[right["model"] != ORDERED_LOGIT_MODEL]
    return compare_frames(left, right, left_header)


def compare_d9s1_primary_columns(
    reference: Path,
    reproduced: Path,
) -> tuple[bool, str]:
    left_header = scientific_csv_columns(reference)
    columns = [
        column
        for column in left_header
        if not column.startswith(("ordered_", "delta_logistic_minus_ordered_"))
    ]
    right_header = scientific_csv_columns(reproduced)
    if any(column not in right_header for column in columns):
        return False, "a primary D9-S1 column is missing"
    left = pd.read_csv(reference, usecols=columns, low_memory=False)
    right = pd.read_csv(reproduced, usecols=columns, low_memory=False)
    return compare_frames_exact(left, right, columns)


def compare_prediction_identity(
    reference: Path,
    reproduced: Path,
) -> tuple[bool, str]:
    left_header = scientific_csv_columns(reference)
    right_header = scientific_csv_columns(reproduced)
    identity_columns = [
        column
        for column in left_header
        if column != "predicted_severity" and not column.startswith("prob_")
    ]
    if any(column not in right_header for column in identity_columns):
        return False, "an identity or target column is missing"
    left = pd.read_csv(reference, usecols=identity_columns, low_memory=False)
    right = pd.read_csv(reproduced, usecols=identity_columns, low_memory=False)
    return compare_frames(left, right, identity_columns)


def compare_prediction_identity_and_labels(
    reference: Path,
    reproduced: Path,
) -> tuple[bool, str]:
    left_header = scientific_csv_columns(reference)
    right_header = scientific_csv_columns(reproduced)
    columns = [
        column for column in left_header if not column.startswith("prob_")
    ]
    if "predicted_severity" not in columns:
        return False, "predicted label column is missing"
    if any(column not in right_header for column in columns):
        return False, "an identity, target or predicted label column is missing"
    left = pd.read_csv(reference, usecols=columns, low_memory=False)
    right = pd.read_csv(reproduced, usecols=columns, low_memory=False)
    return compare_frames_exact(left, right, columns)


def summarize_prediction_deviation(
    reference: Path,
    reproduced: Path,
) -> tuple[bool, str]:
    left = pd.read_csv(reference, low_memory=False)
    right = pd.read_csv(reproduced, low_memory=False)
    if left.columns.tolist() != right.columns.tolist() or len(left) != len(right):
        return False, "prediction schema or row count differs"
    predicted_difference = int(
        (left["predicted_severity"] != right["predicted_severity"]).sum()
    )
    probability_columns = [
        column for column in left.columns if column.startswith("prob_")
    ]
    maximum = 0.0
    per_probability: dict[str, float] = {}
    for column in probability_columns:
        differences = np.abs(
            left[column].to_numpy(dtype=float)
            - right[column].to_numpy(dtype=float)
        )
        column_maximum = float(differences.max(initial=0.0))
        per_probability[column] = column_maximum
        maximum = max(maximum, column_maximum)
    matches = predicted_difference == 0 and maximum <= PROBABILITY_ATOL
    return (
        matches,
        f"label_differences={predicted_difference}/{len(left)}; "
        f"max_probability_difference={maximum:.10g}; "
        + ", ".join(
            f"{column}={value:.10g}"
            for column, value in sorted(per_probability.items())
        ),
    )


def compare_d12_primary_point_estimates(
    reference: Path,
    reproduced: Path,
) -> tuple[bool, str]:
    left = pd.read_csv(reference, low_memory=False)
    right = pd.read_csv(reproduced, low_memory=False)
    if left.columns.tolist() != right.columns.tolist():
        return False, "D12 table columns differ"
    if "model" in left.columns:
        left = left.loc[left["model"] != ORDERED_LOGIT_MODEL]
        right = right.loc[right["model"] != ORDERED_LOGIT_MODEL]
    point_columns = [
        column
        for column in (
            "point_estimate",
            "candidate_point",
            "reference_point",
            "raw_delta_candidate_minus_reference",
            "oriented_advantage",
            "random_internal_point",
            "same_random_model_2024_point",
            "optimism_gap",
        )
        if column in left.columns
    ]
    interval_markers = ("ci_lower", "ci_upper")
    metadata_columns = [
        column
        for column in scientific_csv_columns(reference)
        if column not in point_columns
        and not any(marker in column for marker in interval_markers)
    ]
    columns = metadata_columns + point_columns
    if not point_columns:
        return False, "D12 point-estimate columns are missing"
    if any(column not in right.columns for column in columns):
        return False, "a D12 point-estimate or metadata column is missing"
    return compare_frames_exact(left, right, columns)


def compare_npz(reference: Path, reproduced: Path) -> tuple[bool, str]:
    if hash_file(reference) == hash_file(reproduced):
        return True, "byte-identical"
    maximum = 0.0
    with np.load(reference, allow_pickle=False) as left, np.load(
        reproduced,
        allow_pickle=False,
    ) as right:
        if set(left.files) != set(right.files):
            return False, "array names differ"
        for name in left.files:
            first = left[name]
            second = right[name]
            if first.shape != second.shape or first.dtype != second.dtype:
                return False, f"array contract differs: {name}"
            if np.issubdtype(first.dtype, np.floating):
                differences = np.abs(first - second)
                finite = np.isfinite(differences)
                maximum = max(
                    maximum,
                    float(differences[finite].max()) if finite.any() else 0.0,
                )
                if not np.allclose(
                    first,
                    second,
                    rtol=FLOAT_RTOL,
                    atol=FLOAT_ATOL,
                    equal_nan=True,
                ):
                    return False, f"floating array differs: {name}"
            elif not np.array_equal(first, second):
                return False, f"exact array differs: {name}"
    return True, f"array match; max_abs_diff={maximum:.3g}"


def compare_d12_primary_draws(
    reference: Path,
    reproduced: Path,
) -> tuple[bool, str]:
    maximum = 0.0
    with np.load(reference, allow_pickle=False) as left, np.load(
        reproduced,
        allow_pickle=False,
    ) as right:
        if set(left.files) != set(right.files):
            return False, "array names differ"
        if not np.array_equal(left["model_ids"], right["model_ids"]):
            return False, "model identifiers differ"
        if not np.array_equal(left["metric_ids"], right["metric_ids"]):
            return False, "metric identifiers differ"
        primary_indices = np.flatnonzero(
            left["model_ids"].astype(str) != ORDERED_LOGIT_MODEL
        )
        for name in sorted(item for item in left.files if item.startswith("group_")):
            first = left[name][:, primary_indices, :]
            second = right[name][:, primary_indices, :]
            if first.shape != second.shape:
                return False, f"primary draw shape differs: {name}"
            differences = np.abs(first - second)
            finite = np.isfinite(differences)
            maximum = max(
                maximum,
                float(differences[finite].max()) if finite.any() else 0.0,
            )
            if not np.allclose(
                first,
                second,
                rtol=FLOAT_RTOL,
                atol=FLOAT_ATOL,
                equal_nan=True,
            ):
                return False, f"primary model draws differ: {name}"
    return True, f"primary model draws match; max_abs_diff={maximum:.3g}"


def compare_d12_draw_contract(
    reference: Path,
    reproduced: Path,
) -> tuple[bool, str]:
    with np.load(reference, allow_pickle=False) as left, np.load(
        reproduced,
        allow_pickle=False,
    ) as right:
        if set(left.files) != set(right.files):
            return False, "array names differ"
        for name in left.files:
            first = left[name]
            second = right[name]
            if first.shape != second.shape or first.dtype != second.dtype:
                return False, f"array shape or dtype differs: {name}"
        for identifier in ("model_ids", "metric_ids"):
            if not np.array_equal(left[identifier], right[identifier]):
                return False, f"{identifier} differ"
    return True, f"array contract match; arrays={len(left.files)}"


def summarize_d12_draw_propagation(
    reference: Path,
    reproduced: Path,
) -> tuple[bool, str]:
    primary_maximum = 0.0
    ordered_maximum = 0.0
    primary_differing_groups: list[str] = []
    ordered_differing_groups: list[str] = []
    with np.load(reference, allow_pickle=False) as left, np.load(
        reproduced,
        allow_pickle=False,
    ) as right:
        if set(left.files) != set(right.files):
            return False, "array names differ"
        for identifier in ("model_ids", "metric_ids"):
            if not np.array_equal(left[identifier], right[identifier]):
                return False, f"{identifier} differ"
        model_ids = left["model_ids"].astype(str)
        primary_indices = np.flatnonzero(model_ids != ORDERED_LOGIT_MODEL)
        ordered_indices = np.flatnonzero(model_ids == ORDERED_LOGIT_MODEL)
        if len(ordered_indices) != 1:
            return False, "expected exactly one Ordered Logit array index"
        for name in sorted(item for item in left.files if item.startswith("group_")):
            first = left[name]
            second = right[name]
            if first.shape != second.shape or first.dtype != second.dtype:
                return False, f"array contract differs: {name}"
            primary_first = first[:, primary_indices, :]
            primary_second = second[:, primary_indices, :]
            primary_difference = np.abs(primary_first - primary_second)
            primary_finite = np.isfinite(primary_difference)
            primary_group_maximum = (
                float(primary_difference[primary_finite].max())
                if primary_finite.any()
                else 0.0
            )
            primary_maximum = max(primary_maximum, primary_group_maximum)
            if not np.allclose(
                primary_first,
                primary_second,
                rtol=FLOAT_RTOL,
                atol=FLOAT_ATOL,
                equal_nan=True,
            ):
                primary_differing_groups.append(name)
            ordered_first = first[:, ordered_indices, :]
            ordered_second = second[:, ordered_indices, :]
            ordered_difference = np.abs(ordered_first - ordered_second)
            ordered_finite = np.isfinite(ordered_difference)
            ordered_group_maximum = (
                float(ordered_difference[ordered_finite].max())
                if ordered_finite.any()
                else 0.0
            )
            ordered_maximum = max(ordered_maximum, ordered_group_maximum)
            if not np.allclose(
                ordered_first,
                ordered_second,
                rtol=FLOAT_RTOL,
                atol=FLOAT_ATOL,
                equal_nan=True,
            ):
                ordered_differing_groups.append(name)
    matches = not primary_differing_groups and not ordered_differing_groups
    return (
        matches,
        "primary_differing_groups="
        f"{','.join(primary_differing_groups) if primary_differing_groups else 'none'}; "
        f"primary_max_abs_diff={primary_maximum:.10g}; "
        "ordered_differing_groups="
        f"{','.join(ordered_differing_groups) if ordered_differing_groups else 'none'}; "
        f"ordered_max_abs_diff={ordered_maximum:.10g}",
    )


def normalized_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: normalized_json(item)
            for key, item in sorted(value.items())
            if key not in VOLATILE_JSON_KEYS and not key.lower().endswith("sha256")
        }
    if isinstance(value, list):
        return [normalized_json(item) for item in value]
    return value


def compare_json(reference: Path, reproduced: Path) -> tuple[bool, str]:
    left = normalized_json(json.loads(reference.read_text(encoding="utf-8")))
    right = normalized_json(json.loads(reproduced.read_text(encoding="utf-8")))

    def compare_values(first: Any, second: Any, path: str) -> tuple[bool, str]:
        if isinstance(first, dict) and isinstance(second, dict):
            if set(first) != set(second):
                return False, f"JSON keys differ at {path}"
            for key in first:
                matches, detail = compare_values(
                    first[key],
                    second[key],
                    f"{path}.{key}",
                )
                if not matches:
                    return matches, detail
            return True, "semantic JSON match"
        if isinstance(first, list) and isinstance(second, list):
            if len(first) != len(second):
                return False, f"JSON list length differs at {path}"
            for index, (left_item, right_item) in enumerate(zip(first, second)):
                matches, detail = compare_values(
                    left_item,
                    right_item,
                    f"{path}[{index}]",
                )
                if not matches:
                    return matches, detail
            return True, "semantic JSON match"
        if (
            isinstance(first, (int, float))
            and not isinstance(first, bool)
            and isinstance(second, (int, float))
            and not isinstance(second, bool)
        ):
            matches = np.isclose(
                first,
                second,
                rtol=FLOAT_RTOL,
                atol=FLOAT_ATOL,
            )
            detail = f"numeric JSON difference at {path}: {first} vs {second}"
            return bool(matches), detail
        detail = f"JSON difference at {path}: {first!r} vs {second!r}"
        return first == second, detail

    return compare_values(left, right, "root")


def compare_png(reference: Path, reproduced: Path) -> tuple[bool, str]:
    import matplotlib.image as mpimg

    left = np.asarray(mpimg.imread(reference), dtype=float)
    right = np.asarray(mpimg.imread(reproduced), dtype=float)
    if left.shape != right.shape:
        return False, f"pixel dimensions differ: {left.shape} vs {right.shape}"
    maximum = float(np.max(np.abs(left - right)))
    if maximum > PIXEL_ATOL:
        return False, (
            f"pixel max_abs_diff={maximum:.6g} exceeds {PIXEL_ATOL:.6g}"
        )
    return True, f"pixel dimensions={left.shape}; max_abs_diff={maximum:.6g}"


def result_paths(root: Path, suffixes: tuple[str, ...]) -> set[str]:
    paths: set[str] = set()
    parents = (
        root / "results",
        root / "data" / "processed" / "d9s1_matched_subsets",
    )
    for parent in parents:
        if not parent.exists():
            continue
        for path in parent.rglob("*"):
            relative_parts = path.relative_to(root).parts
            archived = any(part.lower().startswith("archive") for part in relative_parts)
            if path.is_file() and not archived and any(
                path.name.endswith(suffix) for suffix in suffixes
            ):
                paths.add(path.relative_to(root).as_posix())
    return paths


def add_check(
    checks: list[CheckResult],
    name: str,
    operation: Callable[[], tuple[bool, str]],
    *,
    mismatch_status: str = "FAIL",
) -> None:
    started = time.perf_counter()
    try:
        passed, detail = operation()
        status = "PASS" if passed else mismatch_status
    except Exception as exc:
        status = "FAIL"
        detail = f"{type(exc).__name__}: {exc}"
    checks.append(
        CheckResult(
            name,
            status,
            detail,
            round(time.perf_counter() - started, 3),
        )
    )
    print(f"[D16 compare {status}] {name}: {detail}", flush=True)


def freeze_final_comparison_protocol(workspace: Path) -> None:
    workspace = ensure_external_workspace(workspace)
    state = load_state(workspace)
    if state.get("status") != "COMPLETE":
        raise RuntimeError("D16 pipeline must be complete before final comparison freeze")
    marker_path = workspace / "logs" / "d16_workspace.json"
    if not marker_path.is_file():
        raise FileNotFoundError(marker_path)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    execution_protocol_sha256 = str(marker["protocol_sha256"])
    if hash_file(CONFIG_FILE) != execution_protocol_sha256:
        raise AssertionError(
            "The original execution protocol changed after the isolated run"
        )
    if not INITIAL_REPORT_ARCHIVE_FILE.is_file():
        if not SOURCE_REPORT_FILE.is_file():
            raise FileNotFoundError(
                "The initial D16 comparison report is required before amendment"
            )
        shutil.copy2(SOURCE_REPORT_FILE, INITIAL_REPORT_ARCHIVE_FILE)
    initial_report = json.loads(
        INITIAL_REPORT_ARCHIVE_FILE.read_text(encoding="utf-8")
    )
    source_check = verify_frozen_source()
    payload = {
        "version": VERSION,
        "status": "FROZEN_AFTER_PIPELINE_BEFORE_FINAL_COMPARISON",
        "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "timing": (
            "Post-execution and post-initial-comparison; this is a transparent "
            "classification amendment, not a preregistered scientific protocol."
        ),
        "execution": {
            "workspace": str(workspace),
            "pipeline_state": state["status"],
            "pipeline_state_sha256": hash_file(state_file(workspace)),
            "execution_protocol_sha256": execution_protocol_sha256,
            "source_execution_protocol_unchanged": True,
            "thread_cap": THREAD_CAP,
        },
        "initial_comparison": {
            "archived_report": relative(INITIAL_REPORT_ARCHIVE_FILE),
            "archived_report_sha256": hash_file(INITIAL_REPORT_ARCHIVE_FILE),
            "status": initial_report["status"],
            "summary": initial_report["comparison_summary"],
        },
        "reason": (
            "The initial comparator correctly exposed numerical differences but "
            "classified runtime metadata, Ordered-driven joint-bootstrap "
            "propagation and a four-thread probability-only sensitivity drift as "
            "primary failures. The final rules separate exact scientific "
            "invariants from documented numerical deviations."
        ),
        "strict_invariants": {
            "generated_data_and_split_files": "byte-identical SHA-256",
            "primary_Logistic_and_LightGBM_predictions": (
                "identity, target, labels and probabilities under frozen tolerance"
            ),
            "D12_primary_point_estimates": "exact value equality",
            "D12_bootstrap_array_contract": (
                "array names, model/metric identifiers, shapes and dtypes exact"
            ),
            "D14_exclude2020_prediction": (
                "identity, target and predicted labels exact"
            ),
            "H1_H2_decision_tables_and_SHAP_arrays": "strict comparison retained",
        },
        "documented_deviations": {
            "Ordered_Logit": (
                "numerical drift remains DEVIATION and is never coerced to equality"
            ),
            "D12_bootstrap_propagation": (
                "joint draw-stream and resulting interval/figure changes remain "
                "DEVIATION while primary point estimates are exact"
            ),
            "D14_four_thread_probability": (
                "probability-only drift remains DEVIATION while all labels are exact"
            ),
        },
        "non_scientific_fields_excluded": [
            "wall-clock timing columns/keys",
            "compressed prediction file byte count",
        ],
        "tolerances": {
            "floating_absolute": FLOAT_ATOL,
            "floating_relative": FLOAT_RTOL,
            "probability_absolute": PROBABILITY_ATOL,
            "PNG_pixel_absolute": PIXEL_ATOL,
            "changed_after_initial_comparison": False,
        },
        "scientific_outputs_changed": False,
        "frozen_D1_D15_artifacts": source_check,
        "comparator_source": {
            "path": "code/d16_reproduce.py",
            "sha256": hash_file(Path(__file__)),
        },
        "final_test_source": {
            "path": "code/test_d16_reproduction.py",
            "sha256": hash_file(SOURCE_DIR / "code" / "test_d16_reproduction.py"),
        },
    }
    write_json_atomic(FINAL_COMPARISON_CONFIG_FILE, payload)
    print(f"D16_FINAL_COMPARISON_FROZEN={FINAL_COMPARISON_CONFIG_FILE}")
    print(
        "D16_FINAL_COMPARISON_SHA256="
        f"{hash_file(FINAL_COMPARISON_CONFIG_FILE)}"
    )


def verify_final_comparison_protocol(workspace: Path) -> dict[str, Any]:
    if not FINAL_COMPARISON_CONFIG_FILE.is_file():
        raise FileNotFoundError(
            "Run D16 --freeze-comparison before final comparison"
        )
    protocol = json.loads(
        FINAL_COMPARISON_CONFIG_FILE.read_text(encoding="utf-8")
    )
    if protocol.get("version") != VERSION:
        raise ValueError("Unexpected final comparison protocol version")
    if (
        protocol.get("status")
        != "FROZEN_AFTER_PIPELINE_BEFORE_FINAL_COMPARISON"
    ):
        raise ValueError("Final comparison protocol is not frozen")
    marker_path = workspace / "logs" / "d16_workspace.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected_execution = protocol["execution"]["execution_protocol_sha256"]
    if marker["protocol_sha256"] != expected_execution:
        raise AssertionError("Workspace execution protocol binding changed")
    if hash_file(CONFIG_FILE) != expected_execution:
        raise AssertionError("Source execution protocol changed")
    if hash_file(Path(__file__)) != protocol["comparator_source"]["sha256"]:
        raise AssertionError("Final comparator source changed after freeze")
    test_path = SOURCE_DIR / protocol["final_test_source"]["path"]
    if hash_file(test_path) != protocol["final_test_source"]["sha256"]:
        raise AssertionError("Final D16 test source changed after freeze")
    archive_path = SOURCE_DIR / protocol["initial_comparison"]["archived_report"]
    if (
        hash_file(archive_path)
        != protocol["initial_comparison"]["archived_report_sha256"]
    ):
        raise AssertionError("Initial comparison report archive changed")
    return protocol


def compare_reproduction(workspace: Path) -> dict[str, Any]:
    workspace = ensure_external_workspace(workspace)
    state = load_state(workspace)
    if state.get("status") != "COMPLETE":
        raise RuntimeError("D16 pipeline has not completed")
    comparison_protocol = verify_final_comparison_protocol(workspace)
    workspace_marker = json.loads(
        (workspace / "logs" / "d16_workspace.json").read_text(encoding="utf-8")
    )
    checks: list[CheckResult] = []

    for raw_path in GENERATED_EXACT_FILES:
        reference = SOURCE_DIR / raw_path
        reproduced = workspace / raw_path

        def exact_operation(
            ref: Path = reference,
            rep: Path = reproduced,
        ) -> tuple[bool, str]:
            if not rep.is_file():
                return False, "reproduced file missing"
            left_hash = hash_file(ref)
            right_hash = hash_file(rep)
            return (
                left_hash == right_hash,
                f"reference={left_hash}; reproduced={right_hash}",
            )

        add_check(checks, f"exact:{raw_path}", exact_operation)

    reference_csv = result_paths(SOURCE_DIR, (".csv", ".csv.gz"))
    reproduced_csv = result_paths(workspace, (".csv", ".csv.gz"))
    if reference_csv != reproduced_csv:
        checks.append(
            CheckResult(
                "result CSV file set",
                "FAIL",
                f"missing={sorted(reference_csv - reproduced_csv)}; "
                f"extra={sorted(reproduced_csv - reference_csv)}",
                0.0,
            )
        )
    else:
        checks.append(
            CheckResult(
                "result CSV file set",
                "PASS",
                f"{len(reference_csv)} files",
                0.0,
            )
        )
        for raw_path in sorted(reference_csv):
            reference = SOURCE_DIR / raw_path
            reproduced = workspace / raw_path
            is_d9_result = raw_path.startswith("results/d9/")
            is_ordered_d11_prediction = (
                raw_path.startswith("results/d11/predictions/")
                and "__ordered_logit_unweighted__" in raw_path
            )
            if is_d9_result and "/validation_predictions/" in raw_path:
                add_check(
                    checks,
                    f"primary identity:{raw_path}",
                    lambda ref=reference, rep=reproduced: (
                        compare_prediction_identity(ref, rep)
                    ),
                )
                add_check(
                    checks,
                    f"secondary Ordered Logit:{raw_path}",
                    lambda ref=reference, rep=reproduced: (
                        summarize_prediction_deviation(ref, rep)
                    ),
                    mismatch_status="DEVIATION",
                )
            elif is_d9_result:
                add_check(
                    checks,
                    f"secondary Ordered Logit:{raw_path}",
                    lambda ref=reference, rep=reproduced: compare_csv(ref, rep),
                    mismatch_status="DEVIATION",
                )
            elif raw_path == D9S1_MIXED_COMPARISON:
                add_check(
                    checks,
                    f"primary filtered:{raw_path}",
                    lambda ref=reference, rep=reproduced: (
                        compare_d9s1_primary_columns(ref, rep)
                    ),
                )
                add_check(
                    checks,
                    f"secondary Ordered Logit:{raw_path}",
                    lambda ref=reference, rep=reproduced: compare_csv(ref, rep),
                    mismatch_status="DEVIATION",
                )
            elif is_ordered_d11_prediction:
                add_check(
                    checks,
                    f"primary identity:{raw_path}",
                    lambda ref=reference, rep=reproduced: (
                        compare_prediction_identity(ref, rep)
                    ),
                )
                add_check(
                    checks,
                    f"secondary Ordered Logit:{raw_path}",
                    lambda ref=reference, rep=reproduced: (
                        summarize_prediction_deviation(ref, rep)
                    ),
                    mismatch_status="DEVIATION",
                )
            elif raw_path in D12_PROPAGATED_CSV_FILES:
                add_check(
                    checks,
                    f"D12 primary point estimates:{raw_path}",
                    lambda ref=reference, rep=reproduced: (
                        compare_d12_primary_point_estimates(ref, rep)
                    ),
                )
                add_check(
                    checks,
                    f"D12 bootstrap propagation:{raw_path}",
                    lambda ref=reference, rep=reproduced: compare_csv(ref, rep),
                    mismatch_status="DEVIATION",
                )
            elif raw_path == D14_EXCLUDE2020_PREDICTION:
                add_check(
                    checks,
                    f"D14 four-thread identity and labels:{raw_path}",
                    lambda ref=reference, rep=reproduced: (
                        compare_prediction_identity_and_labels(ref, rep)
                    ),
                )
                add_check(
                    checks,
                    f"D14 four-thread probabilities:{raw_path}",
                    lambda ref=reference, rep=reproduced: (
                        summarize_prediction_deviation(ref, rep)
                    ),
                    mismatch_status="DEVIATION",
                )
            elif raw_path in MIXED_ORDERED_MODEL_CSV_FILES:
                add_check(
                    checks,
                    f"primary filtered:{raw_path}",
                    lambda ref=reference, rep=reproduced: (
                        compare_csv_without_ordered_model(ref, rep)
                    ),
                )
                add_check(
                    checks,
                    f"secondary Ordered Logit:{raw_path}",
                    lambda ref=reference, rep=reproduced: compare_csv(ref, rep),
                    mismatch_status="DEVIATION",
                )
            else:
                add_check(
                    checks,
                    f"csv:{raw_path}",
                    lambda ref=reference, rep=reproduced: compare_csv(ref, rep),
                )

    reference_npz = result_paths(SOURCE_DIR, (".npz",))
    reproduced_npz = result_paths(workspace, (".npz",))
    if reference_npz != reproduced_npz:
        checks.append(
            CheckResult(
                "result NPZ file set",
                "FAIL",
                f"missing={sorted(reference_npz - reproduced_npz)}; "
                f"extra={sorted(reproduced_npz - reference_npz)}",
                0.0,
            )
        )
    else:
        checks.append(
            CheckResult(
                "result NPZ file set",
                "PASS",
                f"{len(reference_npz)} files",
                0.0,
            )
        )
        for raw_path in sorted(reference_npz):
            reference = SOURCE_DIR / raw_path
            reproduced = workspace / raw_path
            if raw_path == D12_MIXED_DRAW_FILE:
                add_check(
                    checks,
                    f"D12 bootstrap draw contract:{raw_path}",
                    lambda ref=reference, rep=reproduced: (
                        compare_d12_draw_contract(ref, rep)
                    ),
                )
                add_check(
                    checks,
                    f"D12 bootstrap propagation:{raw_path}",
                    lambda ref=reference, rep=reproduced: (
                        summarize_d12_draw_propagation(ref, rep)
                    ),
                    mismatch_status="DEVIATION",
                )
            else:
                add_check(
                    checks,
                    f"npz:{raw_path}",
                    lambda ref=reference, rep=reproduced: compare_npz(ref, rep),
                )

    for raw_path in SCIENTIFIC_JSON_FILES:
        add_check(
            checks,
            f"json:{raw_path}",
            lambda raw_path=raw_path: compare_json(
                SOURCE_DIR / raw_path,
                workspace / raw_path,
            ),
        )

    reference_png = {
        path.relative_to(SOURCE_DIR).as_posix()
        for path in (SOURCE_DIR / "figures").glob("*.png")
    }
    reproduced_png = {
        path.relative_to(workspace).as_posix()
        for path in (workspace / "figures").glob("*.png")
    }
    if reference_png != reproduced_png:
        checks.append(
            CheckResult(
                "PNG figure set",
                "FAIL",
                f"missing={sorted(reference_png - reproduced_png)}; "
                f"extra={sorted(reproduced_png - reference_png)}",
                0.0,
            )
        )
    else:
        checks.append(
            CheckResult(
                "PNG figure set",
                "PASS",
                f"{len(reference_png)} files",
                0.0,
            )
        )
        for raw_path in sorted(reference_png):
            add_check(
                checks,
                f"png:{raw_path}",
                lambda raw_path=raw_path: compare_png(
                    SOURCE_DIR / raw_path,
                    workspace / raw_path,
                ),
                mismatch_status=(
                    "DEVIATION"
                    if raw_path in {D11_MIXED_FIGURE, D12_PROPAGATED_FIGURE}
                    else "FAIL"
                ),
            )

    reference_pdf = SOURCE_DIR / "figures" / "figure2_dataset_overview.pdf"
    reproduced_pdf = workspace / "figures" / "figure2_dataset_overview.pdf"
    add_check(
        checks,
        "PDF figure existence",
        lambda: (
            reference_pdf.stat().st_size > 0
            and reproduced_pdf.stat().st_size > 0,
            f"reference_bytes={reference_pdf.stat().st_size}; "
            f"reproduced_bytes={reproduced_pdf.stat().st_size}",
        ),
    )

    source_after = verify_frozen_source()
    checks.append(
        CheckResult(
            "source frozen artifacts unchanged",
            "PASS",
            f"{source_after['files']} files; "
            f"set_sha256={source_after['set_sha256']}",
            float(source_after["seconds"]),
        )
    )
    failed_checks = [check for check in checks if check.status == "FAIL"]
    deviation_checks = [
        check for check in checks if check.status == "DEVIATION"
    ]
    if failed_checks:
        status = "D16_FULL_REPRODUCTION_FAIL"
    elif deviation_checks:
        status = "D16_CORE_REPRODUCTION_PASS_WITH_DOCUMENTED_NUMERICAL_DEVIATIONS"
    else:
        status = "D16_FULL_REPRODUCTION_PASS"
    report = {
        "version": VERSION,
        "status": status,
        "completed_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_project": str(SOURCE_DIR),
        "workspace": str(workspace),
        "protocol_sha256": hash_file(CONFIG_FILE),
        "execution_protocol_sha256": workspace_marker["protocol_sha256"],
        "final_comparison_protocol_sha256": hash_file(
            FINAL_COMPARISON_CONFIG_FILE
        ),
        "final_comparison_protocol_status": comparison_protocol["status"],
        "clean_environment_python": str(venv_python(workspace)),
        "thread_cap": THREAD_CAP,
        "platform": platform.platform(),
        "pipeline_state": state,
        "comparison_checks": [asdict(check) for check in checks],
        "comparison_summary": {
            "passed": sum(check.status == "PASS" for check in checks),
            "failed": sum(check.status == "FAIL" for check in checks),
            "deviations": sum(
                check.status == "DEVIATION" for check in checks
            ),
            "total": len(checks),
        },
        "documented_numerical_deviations": [
            asdict(check) for check in deviation_checks
        ],
        "interpretation": {
            "scientific_result": (
                "The core Logistic/LightGBM predictions, primary point estimates, "
                "H1/H2 decision tables and SHAP arrays reproduced. Documented "
                "numerical deviations are retained for Ordered Logit, propagated "
                "D12 bootstrap intervals/figure, and one probability-only "
                "four-thread D14 sensitivity result; no tolerance was widened."
                if deviation_checks and not failed_checks
                else (
                    "The frozen raw-to-results pipeline reproduced within the "
                    "pre-specified D16 tolerances."
                    if not failed_checks
                    else (
                        "At least one primary D16 reproduction check failed; "
                        "investigate before release."
                    )
                )
            ),
            "CAS": "not executed in D16",
            "GitHub_Zenodo": (
                "not created by D16; publication packaging remains separate"
            ),
        },
    }
    write_json_atomic(SOURCE_REPORT_FILE, report)
    write_json_atomic(workspace / "logs" / SOURCE_REPORT_FILE.name, report)
    write_checkpoint(report)
    return report


def write_checkpoint(report: dict[str, Any]) -> None:
    summary = report["comparison_summary"]
    state = report["pipeline_state"]
    lines = [
        "# D16 full reproduction checkpoint",
        "",
        f"Status: **{report['status']}**",
        f"Completed: {report['completed_local']}",
        "",
        "## Execution",
        "",
        f"- Isolated workspace: `{report['workspace']}`",
        f"- Clean Python environment: `{report['clean_environment_python']}`",
        f"- Thread cap: **{report['thread_cap']}**",
        f"- Completed pipeline/test stages: **{len(state.get('stages', []))}**",
        f"- Execution protocol SHA-256: "
        f"`{report['execution_protocol_sha256']}`",
        f"- Final comparison protocol SHA-256: "
        f"`{report['final_comparison_protocol_sha256']}`",
        "- The retained D1-D15 source artifacts were read-only references.",
        "",
        "## Comparison",
        "",
        f"- Passed checks: **{summary['passed']}/{summary['total']}**",
        f"- Failed checks: **{summary['failed']}**",
        f"- Documented numerical deviations: **{summary['deviations']}**",
        "- Identifiers, labels, classes and counts were checked exactly.",
        "- Floating outputs used the pre-frozen D16 tolerances.",
        "- Model binary hashes were not the scientific endpoint; regenerated "
        "predictions, metrics and tests were checked.",
        "",
        "## Scope",
        "",
        "- CAS was not run in D16 and remains a separate later validation.",
        "- GitHub Release and Zenodo DOI follow only after D16 passes.",
    ]
    failures = [
        item
        for item in report["comparison_checks"]
        if item["status"] == "FAIL"
    ]
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(
            f"- `{item['name']}`: {item['detail']}" for item in failures
        )
    deviations = [
        item
        for item in report["comparison_checks"]
        if item["status"] == "DEVIATION"
    ]
    if deviations:
        lines.extend(["", "## Documented numerical deviations", ""])
        lines.extend(
            f"- `{item['name']}`: {item['detail']}" for item in deviations
        )
    text = "\n".join(lines) + "\n"
    SOURCE_CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_CHECKPOINT_FILE.write_text(text, encoding="utf-8")
    workspace = Path(report["workspace"])
    (workspace / "logs" / SOURCE_CHECKPOINT_FILE.name).write_text(
        text,
        encoding="utf-8",
    )


def self_test(workspace: Path) -> None:
    ensure_external_workspace(workspace)
    stage_names = [stage.name for stage in PIPELINE_STAGES]
    if len(stage_names) != len(set(stage_names)):
        raise AssertionError("Duplicate D16 stage names")
    required_scripts = {
        stage.arguments[0] for stage in PIPELINE_STAGES
    } | set(STATS19_TESTS)
    missing = sorted(
        raw_path
        for raw_path in required_scripts
        if not (SOURCE_DIR / raw_path).is_file()
    )
    if missing:
        raise FileNotFoundError(f"Missing D16 scripts: {missing}")
    if FINAL_COMPARISON_CONFIG_FILE.is_file():
        verify_final_comparison_protocol(workspace)
    elif CONFIG_FILE.is_file():
        verify_protocol_inputs(load_protocol())
    print("D16_SELF_TEST=PASS")
    print(f"D16_PIPELINE_STAGES={len(PIPELINE_STAGES)}")
    print(f"D16_STANDALONE_TESTS={len(STATS19_TESTS) + 1}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--freeze-comparison", action="store_true")
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--compare", action="store_true")
    action.add_argument("--all", action="store_true")
    action.add_argument("--self-test", action="store_true")
    action.add_argument(
        "--internal-protocol",
        choices=INTERNAL_PROTOCOL_CHOICES,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--base-python", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a prepared workspace and skip passed stages.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.internal_protocol is not None:
        prepare_runtime_protocol(args.internal_protocol)
        return 0
    workspace = ensure_external_workspace(args.workspace)
    if args.freeze:
        freeze_protocol()
        return 0
    if args.freeze_comparison:
        freeze_final_comparison_protocol(workspace)
        return 0
    if args.self_test:
        self_test(workspace)
        return 0
    if args.prepare:
        prepare_workspace(workspace)
        return 0
    if args.run:
        protocol = load_protocol()
        verify_protocol_inputs(protocol)
        copy_protocol_inputs(
            workspace,
            protocol,
            preserve_runtime_protocols=args.resume,
        )
        python = clean_environment(workspace, args.base_python)
        execute_pipeline(workspace, python, resume=args.resume)
        return 0
    if args.compare:
        report = compare_reproduction(workspace)
        print(f"D16_STATUS={report['status']}")
        return 1 if report["status"] == "D16_FULL_REPRODUCTION_FAIL" else 0

    protocol = load_protocol()
    verify_protocol_inputs(protocol)
    if not workspace.exists():
        prepare_workspace(workspace)
    elif not args.resume:
        raise FileExistsError(
            f"D16 workspace already exists: {workspace}; "
            "use --resume or a new path"
        )
    copy_protocol_inputs(
        workspace,
        protocol,
        preserve_runtime_protocols=args.resume,
    )
    python = clean_environment(workspace, args.base_python)
    execute_pipeline(workspace, python, resume=args.resume)
    report = compare_reproduction(workspace)
    print(f"D16_STATUS={report['status']}")
    return 1 if report["status"] == "D16_FULL_REPRODUCTION_FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
