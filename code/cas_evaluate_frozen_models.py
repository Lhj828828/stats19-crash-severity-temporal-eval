"""Freeze and execute the one-time CAS test evaluation."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cas_modeling_common import (
    ANALYSIS_PROTOCOL_FILE,
    ASSIGNMENTS_FILE,
    PROJECT_DIR,
    SCHEMA_FILE,
    TARGET_CODES,
    THREADS,
    TRAINING_INPUTS_FILE,
    build_artifact_manifest,
    classification_metrics,
    confusion_rows,
    hash_file,
    load_aligned_data,
    load_contracts,
    predict_with_probabilities,
    prediction_frame,
    prepare_features,
    relative,
    write_csv,
    write_dataframe_gzip,
    write_json_atomic,
)


VERSION = "CAS_EVALUATION_V1"
MODELS = (
    "dummy_most_frequent",
    "logistic_weighted",
    "lightgbm_weighted",
)
BASELINE_FROZEN_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_baseline_models_frozen.json"
)
LIGHTGBM_FROZEN_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_lightgbm_models_frozen.json"
)
LIGHTGBM_SELECTED_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_lightgbm_selected.json"
)
BASELINE_MODEL_DIR = PROJECT_DIR / "models" / "cas_baseline"
LIGHTGBM_MODEL_DIR = PROJECT_DIR / "models" / "cas_lightgbm"
CONFIG_FILE = PROJECT_DIR / "config" / "cas" / "cas_evaluation_protocol.json"
COMPLETE_FILE = PROJECT_DIR / "config" / "cas" / "cas_evaluation_complete.json"
RESULT_DIR = PROJECT_DIR / "results" / "cas_evaluation"
PREDICTION_DIR = RESULT_DIR / "predictions"
METRICS_FILE = RESULT_DIR / "test_metrics.csv"
CONFUSION_FILE = RESULT_DIR / "test_confusion_matrices.csv"
LOG_DIR = PROJECT_DIR / "logs" / "cas"
ACCESS_AUDIT_FILE = LOG_DIR / "cas_evaluation_access_audit.csv"
MANIFEST_FILE = LOG_DIR / "cas_evaluation_artifact_manifest.csv"
CHECKPOINT_FILE = LOG_DIR / "cas_evaluation_checkpoint.md"


def now_local() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def evaluation_groups(analysis: dict[str, object]) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = [
        {
            "evaluation_group": "temporal_2025",
            "evaluation_design": "temporal_test",
            "model_split_slug": "temporal",
            "model_protocol": "temporal",
            "seed": "year_based",
            "role_column": "temporal_role",
            "role_value": "test",
            "expected_rows": int(
                analysis["temporal_protocol"]["row_counts"]["test"]
            ),
            "expected_years": [2025],
            "interpretation": "primary deployment-oriented temporal test",
        }
    ]
    expected_internal = int(
        analysis["random_reference_protocol"]["row_counts_per_seed"]["test"]
    )
    expected_2025 = int(analysis["temporal_protocol"]["row_counts"]["test"])
    for seed in analysis["random_reference_protocol"]["seeds"]:
        groups.append(
            {
                "evaluation_group": f"random_seed_{seed}_internal_test",
                "evaluation_design": "random_internal_test",
                "model_split_slug": f"random_seed_{seed}",
                "model_protocol": "random_reference",
                "seed": str(seed),
                "role_column": f"random_role_seed_{seed}",
                "role_value": "test",
                "expected_rows": expected_internal,
                "expected_years": [2022, 2023, 2024],
                "interpretation": "same-period optimistic reference",
            }
        )
        groups.append(
            {
                "evaluation_group": f"random_seed_{seed}_2025_diagnostic",
                "evaluation_design": "random_2025_diagnostic",
                "model_split_slug": f"random_seed_{seed}",
                "model_protocol": "random_reference",
                "seed": str(seed),
                "role_column": f"random_role_seed_{seed}",
                "role_value": "locked_temporal_test",
                "expected_rows": expected_2025,
                "expected_years": [2025],
                "interpretation": (
                    "same frozen random model applied to 2025 without retuning"
                ),
            }
        )
    return groups


def load_freeze_payloads() -> tuple[dict[str, object], dict[str, object]]:
    baseline = json.loads(BASELINE_FROZEN_FILE.read_text(encoding="utf-8"))
    lightgbm = json.loads(LIGHTGBM_FROZEN_FILE.read_text(encoding="utf-8"))
    if baseline.get("status") != (
        "BASELINE_MODELS_FROZEN_BEFORE_2025_EVALUATION"
    ):
        raise ValueError("CAS baseline models are not frozen")
    if lightgbm.get("status") != (
        "LIGHTGBM_MODELS_FROZEN_BEFORE_2025_EVALUATION"
    ):
        raise ValueError("CAS LightGBM models are not frozen")
    if baseline.get("test_performance_used") is not False:
        raise ValueError("CAS baseline freeze does not confirm test independence")
    if lightgbm.get("test_performance_used") is not False:
        raise ValueError("CAS LightGBM freeze does not confirm test independence")
    for payload in (baseline, lightgbm):
        entries = payload.get("models", payload.get("model_files"))
        for entry in entries:
            path = PROJECT_DIR / entry["file"]
            if hash_file(path) != entry["sha256"]:
                raise ValueError(f"Frozen model artifact changed: {entry['file']}")
    return baseline, lightgbm


def freeze_protocol() -> None:
    _schema, analysis, training = load_contracts()
    baseline, lightgbm = load_freeze_payloads()
    if METRICS_FILE.exists() or COMPLETE_FILE.exists():
        raise RuntimeError("CAS test results already exist; refusing to refreeze")
    groups = evaluation_groups(analysis)
    if len(groups) != 11:
        raise AssertionError("CAS evaluation protocol must enumerate 11 groups")
    write_json_atomic(
        CONFIG_FILE,
        {
            "version": VERSION,
            "status": "FROZEN_BEFORE_FIRST_CAS_TEST_PREDICTION",
            "created_local": now_local(),
            "upstream": {
                "modeling_schema": relative(SCHEMA_FILE),
                "modeling_schema_sha256": hash_file(SCHEMA_FILE),
                "analysis_protocol": relative(ANALYSIS_PROTOCOL_FILE),
                "analysis_protocol_sha256": hash_file(ANALYSIS_PROTOCOL_FILE),
                "training_inputs": relative(TRAINING_INPUTS_FILE),
                "training_inputs_sha256": hash_file(TRAINING_INPUTS_FILE),
                "split_assignments": relative(ASSIGNMENTS_FILE),
                "split_assignments_sha256": hash_file(ASSIGNMENTS_FILE),
                "baseline_models_frozen": relative(BASELINE_FROZEN_FILE),
                "baseline_models_frozen_sha256": hash_file(
                    BASELINE_FROZEN_FILE
                ),
                "lightgbm_models_frozen": relative(LIGHTGBM_FROZEN_FILE),
                "lightgbm_models_frozen_sha256": hash_file(
                    LIGHTGBM_FROZEN_FILE
                ),
                "lightgbm_selected": relative(LIGHTGBM_SELECTED_FILE),
                "lightgbm_selected_sha256": hash_file(LIGHTGBM_SELECTED_FILE),
            },
            "models": list(MODELS),
            "evaluation_groups": groups,
            "group_count": len(groups),
            "metric_rows_expected": len(groups) * len(MODELS),
            "prediction_rule": "frozen argmax; no threshold tuning",
            "preprocessing_rule": (
                "load split-specific frozen preprocessing; never refit"
            ),
            "one_time_rule": (
                "generate each frozen model/cohort prediction once; any rerun "
                "requires explicit reproducibility verification and must not "
                "change model selection"
            ),
            "reporting": analysis["frozen_metrics"],
            "test_boundary_before_freeze": training["test_boundary"],
            "prior_2025_access_disclosure": (
                "schema, target counts, missingness, category coverage and "
                "drift were inspected during feasibility auditing; no model "
                "performance existed before this protocol"
            ),
            "frozen_payload_status": {
                "baseline": baseline["status"],
                "lightgbm": lightgbm["status"],
            },
        },
    )
    print(f"CAS_EVALUATION_PROTOCOL_SHA256={hash_file(CONFIG_FILE)}")
    print("CAS_EVALUATION_PROTOCOL_STATUS=FROZEN_BEFORE_FIRST_TEST_PREDICTION")


def verify_protocol() -> dict[str, object]:
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    if config.get("version") != VERSION:
        raise ValueError("Unexpected CAS evaluation protocol version")
    if config.get("status") != "FROZEN_BEFORE_FIRST_CAS_TEST_PREDICTION":
        raise ValueError("CAS test evaluation protocol is not frozen")
    checks = {
        SCHEMA_FILE: config["upstream"]["modeling_schema_sha256"],
        ANALYSIS_PROTOCOL_FILE: config["upstream"]["analysis_protocol_sha256"],
        TRAINING_INPUTS_FILE: config["upstream"]["training_inputs_sha256"],
        ASSIGNMENTS_FILE: config["upstream"]["split_assignments_sha256"],
        BASELINE_FROZEN_FILE: config["upstream"][
            "baseline_models_frozen_sha256"
        ],
        LIGHTGBM_FROZEN_FILE: config["upstream"][
            "lightgbm_models_frozen_sha256"
        ],
        LIGHTGBM_SELECTED_FILE: config["upstream"]["lightgbm_selected_sha256"],
    }
    for path, expected in checks.items():
        if hash_file(path) != expected:
            raise ValueError(f"CAS evaluation upstream changed: {path.name}")
    load_freeze_payloads()
    if len(config["evaluation_groups"]) != 11:
        raise ValueError("CAS evaluation group enumeration changed")
    return config


def load_model_inputs(
    *,
    split_slug: str,
    model_name: str,
    X_cohort: pd.DataFrame,
) -> tuple[dict[str, object], object, object]:
    if model_name in {"dummy_most_frequent", "logistic_weighted"}:
        split_dir = BASELINE_MODEL_DIR / split_slug
        artifact = joblib.load(split_dir / f"{model_name}.joblib")
        if model_name == "dummy_most_frequent":
            return artifact, artifact["estimator"], np.zeros(
                (len(X_cohort), 1), dtype=np.int8
            )
        bundle = joblib.load(split_dir / "preprocessing_bundle.joblib")
        prepared = prepare_features(
            X_cohort,
            feature_columns=bundle["feature_columns"],
            categorical_columns=bundle["categorical_columns"],
            numeric_columns=bundle["numeric_columns"],
            category_vocabulary=bundle["category_vocabulary"],
        )
        encoded = bundle["preprocessor"].transform(prepared.frame)
        return artifact, artifact["estimator"], encoded

    split_dir = LIGHTGBM_MODEL_DIR / split_slug
    artifact = joblib.load(split_dir / f"{model_name}.joblib")
    bundle = joblib.load(split_dir / "preprocessing_bundle.joblib")
    prepared = prepare_features(
        X_cohort,
        feature_columns=bundle["feature_columns"],
        categorical_columns=bundle["categorical_columns"],
        numeric_columns=bundle["numeric_columns"],
        category_vocabulary=bundle["category_vocabulary"],
    )
    return artifact, artifact["estimator"], prepared.frame


def run_evaluation() -> None:
    config = verify_protocol()
    if METRICS_FILE.exists() or COMPLETE_FILE.exists():
        raise RuntimeError(
            "CAS one-time test outputs already exist; refusing to evaluate again"
        )
    schema, analysis, _training = load_contracts()
    features, target, metadata, assignments = load_aligned_data(schema, analysis)
    metric_rows: list[dict[str, object]] = []
    confusion_output: list[dict[str, object]] = []
    access_rows: list[dict[str, object]] = []

    for group in config["evaluation_groups"]:
        cohort_positions = np.flatnonzero(
            assignments[group["role_column"]]
            .eq(group["role_value"])
            .to_numpy()
        )
        cohort_years = sorted(
            set(
                assignments.iloc[cohort_positions]["meta_crash_year"].astype(
                    int
                )
            )
        )
        if len(cohort_positions) != int(group["expected_rows"]):
            raise AssertionError(
                f"CAS group row count changed: {group['evaluation_group']}"
            )
        if cohort_years != list(group["expected_years"]):
            raise AssertionError(
                f"CAS group years changed: {group['evaluation_group']}"
            )
        X_cohort = features.iloc[cohort_positions]
        y_true = target.iloc[cohort_positions].to_numpy(dtype=np.int8)
        crash_ids = metadata.iloc[cohort_positions]["meta_crash_id"]
        years = assignments.iloc[cohort_positions]["meta_crash_year"]

        for model_name in MODELS:
            artifact, estimator, model_input = load_model_inputs(
                split_slug=group["model_split_slug"],
                model_name=model_name,
                X_cohort=X_cohort,
            )
            if artifact["protocol"] != group["model_protocol"]:
                raise AssertionError("Frozen model protocol differs from group")
            if str(artifact["seed"]) != str(group["seed"]):
                raise AssertionError("Frozen model seed differs from group")
            started = time.perf_counter()
            predicted, probabilities = predict_with_probabilities(
                estimator, model_input
            )
            prediction_seconds = time.perf_counter() - started
            metrics = classification_metrics(y_true, predicted)
            metric_rows.append(
                {
                    "evaluation_group": group["evaluation_group"],
                    "evaluation_design": group["evaluation_design"],
                    "model_protocol": group["model_protocol"],
                    "seed": group["seed"],
                    "cohort_role": group["role_value"],
                    "cohort_rows": len(cohort_positions),
                    "cohort_years": ";".join(map(str, cohort_years)),
                    "model": model_name,
                    "prediction_seconds": prediction_seconds,
                    **metrics,
                }
            )
            confusion_output.extend(
                confusion_rows(
                    y_true,
                    predicted,
                    protocol=group["evaluation_design"],
                    seed=str(group["seed"]),
                    evaluation_role=group["evaluation_group"],
                    model=model_name,
                )
            )
            prediction_path = (
                PREDICTION_DIR
                / f"{group['evaluation_group']}__{model_name}.csv.gz"
            )
            write_dataframe_gzip(
                prediction_path,
                prediction_frame(
                    crash_ids=crash_ids,
                    years=years,
                    y_true=y_true,
                    y_pred=predicted,
                    probabilities=probabilities,
                ),
            )
            access_rows.append(
                {
                    "evaluation_group": group["evaluation_group"],
                    "model": model_name,
                    "source_split": group["model_split_slug"],
                    "role_column": group["role_column"],
                    "role_value": group["role_value"],
                    "rows": len(cohort_positions),
                    "years": ";".join(map(str, cohort_years)),
                    "prediction_file": relative(prediction_path),
                    "prediction_sha256": hash_file(prediction_path),
                    "model_protocol_sha256": artifact["protocol_sha256"],
                    "preprocessing_refit": False,
                    "threshold_tuning": False,
                }
            )
        print(
            f"{group['evaluation_group']}: rows={len(cohort_positions):,}, "
            f"years={cohort_years}",
            flush=True,
        )

    if len(metric_rows) != int(config["metric_rows_expected"]):
        raise AssertionError("CAS test metric row count differs from protocol")
    write_csv(METRICS_FILE, metric_rows)
    write_csv(CONFUSION_FILE, confusion_output)
    write_csv(ACCESS_AUDIT_FILE, access_rows)
    metrics = pd.DataFrame(metric_rows)
    primary = metrics.loc[
        metrics["evaluation_group"].eq("temporal_2025")
    ].sort_values("model")
    lines = [
        "# CAS one-time test evaluation checkpoint",
        "",
        "Status: **PASS - FROZEN_MODELS_EVALUATED_ONCE**",
        "",
        "## Primary temporal test (2025)",
        "",
    ]
    for _, row in primary.iterrows():
        lines.append(
            f"- {row['model']}: Macro-F1 **{row['macro_f1']:.4f}**, "
            f"QWK **{row['qwk']:.4f}**, Fatal recall "
            f"**{row['fatal_recall']:.4f}**."
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- 11 evaluation data groups were enumerated before prediction.",
            "- Three frozen models were evaluated on identical records per group.",
            "- No preprocessing, model, threshold or hyperparameter was refitted.",
            "- Random internal tests and random-model 2025 diagnostics are "
            "reported separately.",
            "",
        ]
    )
    CHECKPOINT_FILE.write_text("\n".join(lines), encoding="utf-8")

    prediction_paths = sorted(PREDICTION_DIR.glob("*.csv.gz"))
    if len(prediction_paths) != 33:
        raise AssertionError("Expected 33 CAS test prediction files")
    artifacts = [
        CONFIG_FILE,
        METRICS_FILE,
        CONFUSION_FILE,
        ACCESS_AUDIT_FILE,
        CHECKPOINT_FILE,
        *prediction_paths,
    ]
    build_artifact_manifest(artifacts, MANIFEST_FILE)
    write_json_atomic(
        COMPLETE_FILE,
        {
            "version": VERSION,
            "status": "FROZEN_MODELS_EVALUATED_ONCE",
            "created_local": now_local(),
            "protocol": relative(CONFIG_FILE),
            "protocol_sha256": hash_file(CONFIG_FILE),
            "metrics": relative(METRICS_FILE),
            "metrics_sha256": hash_file(METRICS_FILE),
            "artifact_manifest": relative(MANIFEST_FILE),
            "artifact_manifest_sha256": hash_file(MANIFEST_FILE),
            "evaluation_groups": 11,
            "models_per_group": 3,
            "metric_rows": len(metric_rows),
            "prediction_files": len(prediction_paths),
            "preprocessing_refit": False,
            "model_selection_after_test": False,
        },
    )
    print(f"CAS_EVALUATION_METRIC_ROWS={len(metric_rows)}")
    print(f"CAS_EVALUATION_PREDICTION_FILES={len(prediction_paths)}")
    print("CAS_EVALUATION_STATUS=FROZEN_MODELS_EVALUATED_ONCE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--evaluate-once", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.freeze:
        freeze_protocol()
    else:
        run_evaluation()
