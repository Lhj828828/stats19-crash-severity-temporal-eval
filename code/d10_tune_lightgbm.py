"""Tune and freeze the D10 class-weighted LightGBM models.

Execution is deliberately staged:

1. ``--freeze`` writes the candidate set and selection rule before tuning.
2. ``--smoke`` checks the native-categorical pipeline on small subsets.
3. ``--tune`` uses only 2018-2022 training and 2023 validation data.
4. ``--fit-all`` refits one temporal and five random-reference models with the
   same selected structure and iteration count, then evaluates validation only.

No test role is selected and no 2024 performance is calculated in D10.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from baseline_modeling import (
    TARGET_CODES,
    classification_metrics,
    confusion_rows,
    fit_category_vocabulary,
    predict_with_probabilities,
    prepare_features,
)
from modeling_data import load_modeling_data


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCHEMA_FILE = PROJECT_DIR / "config" / "d5_dataset_schema.json"
D6_PROTOCOL_FILE = PROJECT_DIR / "config" / "d6_analysis_protocol.json"
D7_CONFIG_FILE = PROJECT_DIR / "config" / "d7_training_inputs.json"
D8_CONFIG_FILE = PROJECT_DIR / "config" / "d8_baseline_protocol.json"
D8_METRICS_FILE = PROJECT_DIR / "results" / "d8" / "d8_validation_metrics.csv"
ASSIGNMENTS_FILE = PROJECT_DIR / "data" / "processed" / "d6_split_assignments.csv.gz"

D10_PROTOCOL_FILE = PROJECT_DIR / "config" / "d10_lightgbm_protocol.json"
D10_EXECUTION_AMENDMENT_FILE = PROJECT_DIR / "config" / "d10_execution_amendment.json"
D10_SELECTED_FILE = PROJECT_DIR / "config" / "d10_selected_lightgbm.json"
RESULT_DIR = PROJECT_DIR / "results" / "d10"
PREDICTION_DIR = RESULT_DIR / "validation_predictions"
MODEL_DIR = PROJECT_DIR / "models" / "d10"
LOG_DIR = PROJECT_DIR / "logs"

TUNING_FILE = RESULT_DIR / "d10_tuning_results.csv"
TUNING_CONFUSION_FILE = RESULT_DIR / "d10_tuning_confusion_matrices.csv"
METRICS_FILE = RESULT_DIR / "d10_validation_metrics.csv"
CONFUSION_FILE = RESULT_DIR / "d10_validation_confusion_matrices.csv"
PREPROCESS_AUDIT_FILE = LOG_DIR / "d10_preprocessing_audit.csv"
TRAINING_AUDIT_FILE = LOG_DIR / "d10_training_audit.csv"
SMOKE_FILE = LOG_DIR / "d10_smoke_test.json"
CHECKPOINT_FILE = LOG_DIR / "d10_checkpoint.md"
MANIFEST_FILE = LOG_DIR / "d10_artifact_manifest.csv"

VERSION = "D10_V1"
CREATED_LOCAL = "2026-08-30"
MODEL_NAME = "lightgbm_weighted"
MODEL_RANDOM_SEED = 20_260_830
SMOKE_RANDOM_SEED = 100_830
SMOKE_TRAIN_ROWS = 20_000
SMOKE_VALIDATION_ROWS = 5_000
SMOKE_ESTIMATORS = 30
MAX_TUNING_ESTIMATORS = 1_200
EARLY_STOPPING_ROUNDS = 75

# These candidates and their order are frozen before D10 validation inspection.
# The set varies tree capacity, leaf support and one regularized subsampling case.
CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "C01",
        "complexity_rank": 1,
        "parameters": {
            "num_leaves": 15,
            "max_depth": -1,
            "min_child_samples": 100,
            "min_split_gain": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "colsample_bytree": 1.0,
            "subsample": 1.0,
            "subsample_freq": 0,
        },
    },
    {
        "candidate_id": "C02",
        "complexity_rank": 3,
        "parameters": {
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 100,
            "min_split_gain": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "colsample_bytree": 1.0,
            "subsample": 1.0,
            "subsample_freq": 0,
        },
    },
    {
        "candidate_id": "C03",
        "complexity_rank": 6,
        "parameters": {
            "num_leaves": 63,
            "max_depth": -1,
            "min_child_samples": 100,
            "min_split_gain": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "colsample_bytree": 1.0,
            "subsample": 1.0,
            "subsample_freq": 0,
        },
    },
    {
        "candidate_id": "C04",
        "complexity_rank": 5,
        "parameters": {
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 50,
            "min_split_gain": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "colsample_bytree": 1.0,
            "subsample": 1.0,
            "subsample_freq": 0,
        },
    },
    {
        "candidate_id": "C05",
        "complexity_rank": 2,
        "parameters": {
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 250,
            "min_split_gain": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "colsample_bytree": 1.0,
            "subsample": 1.0,
            "subsample_freq": 0,
        },
    },
    {
        "candidate_id": "C06",
        "complexity_rank": 4,
        "parameters": {
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 100,
            "min_split_gain": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 5.0,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
            "subsample_freq": 1,
        },
    },
)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows generated for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_prediction(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.gz")
    frame.to_csv(
        temporary,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        lineterminator="\n",
    )
    temporary.replace(path)


def load_contracts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    d6 = json.loads(D6_PROTOCOL_FILE.read_text(encoding="utf-8"))
    d7 = json.loads(D7_CONFIG_FILE.read_text(encoding="utf-8"))
    d8 = json.loads(D8_CONFIG_FILE.read_text(encoding="utf-8"))
    if schema.get("status") != "QC_PASSED_AND_FROZEN":
        raise ValueError("D10 requires the frozen D5 schema")
    if d6.get("version") != "D6_V2" or d6.get("status") != "FROZEN_BEFORE_MODEL_TRAINING":
        raise ValueError("D10 requires the frozen D6_V2 protocol")
    if d7.get("status") != "TRAINING_INPUTS_AUDITED_AND_FROZEN":
        raise ValueError("D10 requires the completed D7 audit")
    if d8.get("version") != "D8_V1":
        raise ValueError("D10 requires the frozen D8 baselines")
    if hash_file(ASSIGNMENTS_FILE) != d7["upstream"]["D6_assignments_sha256"]:
        raise ValueError("D6 assignments changed after D7")
    if hash_file(SCHEMA_FILE) != d7["upstream"]["D5_schema_sha256"]:
        raise ValueError("D5 schema changed after D7")
    if schema["feature_columns"] != d7["feature_contract"]["feature_columns"]:
        raise ValueError("D5 and D7 feature allowlists differ")
    if d8["features"]["allowlist"] != schema["feature_columns"]:
        raise ValueError("D8 and D5 feature allowlists differ")
    return schema, d6, d7, d8


def weighted_logistic_reference() -> dict[str, float]:
    metrics = pd.read_csv(D8_METRICS_FILE)
    row = metrics.loc[
        metrics["protocol"].eq("temporal")
        & metrics["model"].eq("logistic_weighted")
        & metrics["evaluation_role"].eq("validation")
    ]
    if len(row) != 1:
        raise ValueError("Expected exactly one temporal weighted-Logistic validation row")
    values = row.iloc[0]
    return {
        "macro_f1": float(values["macro_f1"]),
        "qwk": float(values["qwk"]),
        "fatal_recall": float(values["fatal_recall"]),
    }


def make_protocol_config(
    schema: dict[str, Any],
    d6: dict[str, Any],
    d7: dict[str, Any],
) -> dict[str, Any]:
    reference = weighted_logistic_reference()
    return {
        "version": VERSION,
        "status": "PROTOCOL_FROZEN_BEFORE_D10_VALIDATION_INSPECTION",
        "created_local": CREATED_LOCAL,
        "purpose": "Small-range tuning of a class-weighted LightGBM comparator without test-set access.",
        "upstream": {
            "D5_schema": SCHEMA_FILE.relative_to(PROJECT_DIR).as_posix(),
            "D5_schema_sha256": hash_file(SCHEMA_FILE),
            "D6_protocol": D6_PROTOCOL_FILE.relative_to(PROJECT_DIR).as_posix(),
            "D6_protocol_sha256": hash_file(D6_PROTOCOL_FILE),
            "D7_training_inputs": D7_CONFIG_FILE.relative_to(PROJECT_DIR).as_posix(),
            "D7_training_inputs_sha256": hash_file(D7_CONFIG_FILE),
            "D8_protocol": D8_CONFIG_FILE.relative_to(PROJECT_DIR).as_posix(),
            "D8_protocol_sha256": hash_file(D8_CONFIG_FILE),
            "D8_validation_metrics": D8_METRICS_FILE.relative_to(PROJECT_DIR).as_posix(),
            "D8_validation_metrics_sha256": hash_file(D8_METRICS_FILE),
            "D6_assignments_sha256": d7["upstream"]["D6_assignments_sha256"],
        },
        "evaluation_scope": {
            "tuning_train_years": d6["temporal_protocol"]["train_years"],
            "tuning_validation_years": d6["temporal_protocol"]["validation_years"],
            "tuning_protocol": "temporal only",
            "post_selection_validation_models": "temporal plus five frozen random-reference splits",
            "random_split_rule": "reuse the temporal-selected candidate and iteration count; no random-specific tuning",
            "forbidden": "all temporal/random test performance and all 2024 performance",
        },
        "features": {
            "allowlist": schema["feature_columns"],
            "categorical": schema["categorical_feature_columns"],
            "numeric": schema["numeric_feature_columns"],
            "metadata_excluded": schema["metadata_columns"],
            "target_excluded": schema["target_column"],
        },
        "preprocessing": {
            "fit_scope": "separately within each training partition",
            "categorical": "LightGBM-native pandas categorical data with training vocabulary and explicit __UNSEEN__ level",
            "numeric": "numeric type retained; NaN handled natively by LightGBM",
            "numeric_training_median": "stored for audit only and never applied",
            "one_hot_encoding": False,
            "scaling": False,
            "rare_pooling": False,
        },
        "class_weighting": {
            "rule": d7["class_weight_rule"],
            "application": "split-specific D7 balanced weights applied to training rows only",
            "validation_metrics": "unweighted natural-prevalence metrics",
        },
        "common_model_parameters": {
            "boosting_type": "gbdt",
            "objective": "multiclass",
            "num_class": 3,
            "learning_rate": 0.05,
            "max_bin": 255,
            "random_state": MODEL_RANDOM_SEED,
            "n_jobs": -1,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        },
        "tuning": {
            "maximum_estimators": MAX_TUNING_ESTIMATORS,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "early_stopping_metric": "unweighted multiclass log loss on 2023 validation",
            "candidates": list(CANDIDATES),
        },
        "selection_rule": {
            "primary_metric": "Macro-F1 on 2023 validation",
            "constraint_reference": {
                "model": "D8 temporal weighted multinomial Logistic",
                **reference,
            },
            "eligibility_constraints": {
                "minimum_qwk": max(0.0, reference["qwk"] - 0.01),
                "minimum_fatal_recall": max(0.0, reference["fatal_recall"] - 0.05),
                "rationale": "avoid selecting a Macro-F1 gain that materially sacrifices ordinal agreement or fatal-case sensitivity relative to the fixed weighted linear comparator",
            },
            "ranking": [
                "eligible candidates only",
                "highest Macro-F1",
                "highest QWK if Macro-F1 is exactly tied",
                "highest fatal recall if still tied",
                "lowest pre-frozen complexity_rank if still tied",
                "lexicographically smallest candidate_id if still tied",
            ],
            "fallback_if_none_eligible": "apply the same ranking to all candidates and explicitly record that constraints were unmet",
        },
        "final_validation_fit": {
            "n_estimators": "the temporal selected candidate's best_iteration",
            "early_stopping": False,
            "same_parameters_for_all_splits": True,
            "prediction_rule": "argmax over class probabilities; no threshold tuning",
        },
        "random_seeds": d6["random_reference_protocol"]["seeds"],
        "test_embargo": d7["test_embargo"],
    }


def freeze_protocol() -> None:
    schema, d6, d7, _ = load_contracts()
    payload = make_protocol_config(schema, d6, d7)
    if D10_PROTOCOL_FILE.exists():
        existing = json.loads(D10_PROTOCOL_FILE.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("Existing D10 protocol differs; refusing to overwrite a frozen protocol")
        print("D10 protocol already exists and matches the code specification.")
    else:
        write_json(D10_PROTOCOL_FILE, payload)
        print("D10 protocol frozen before tuning output was generated.")
    print("D10_PROTOCOL_SHA256=", hash_file(D10_PROTOCOL_FILE))


def require_frozen_protocol() -> dict[str, Any]:
    if not D10_PROTOCOL_FILE.exists():
        raise FileNotFoundError("Run --freeze before any D10 model operation")
    protocol = json.loads(D10_PROTOCOL_FILE.read_text(encoding="utf-8"))
    if protocol.get("version") != VERSION:
        raise ValueError("Unexpected D10 protocol version")
    if protocol.get("status") != "PROTOCOL_FROZEN_BEFORE_D10_VALIDATION_INSPECTION":
        raise ValueError("D10 protocol is not in the frozen state")
    upstream_paths = {
        "D5_schema_sha256": SCHEMA_FILE,
        "D6_protocol_sha256": D6_PROTOCOL_FILE,
        "D7_training_inputs_sha256": D7_CONFIG_FILE,
        "D8_protocol_sha256": D8_CONFIG_FILE,
        "D8_validation_metrics_sha256": D8_METRICS_FILE,
    }
    for key, path in upstream_paths.items():
        if protocol["upstream"][key] != hash_file(path):
            raise ValueError(f"Upstream artifact changed after D10 freeze: {path.name}")
    if protocol["upstream"]["D6_assignments_sha256"] != hash_file(ASSIGNMENTS_FILE):
        raise ValueError("D6 assignments changed after D10 freeze")
    return protocol


def require_execution_amendment() -> dict[str, Any]:
    """Load the documented runtime-only CPU cap."""
    if not D10_EXECUTION_AMENDMENT_FILE.exists():
        raise FileNotFoundError("D10 execution amendment with the CPU cap is missing")
    amendment = json.loads(
        D10_EXECUTION_AMENDMENT_FILE.read_text(encoding="utf-8")
    )
    if amendment.get("version") != "D10_EXECUTION_AMENDMENT_V1":
        raise ValueError("Unexpected D10 execution amendment version")
    if amendment.get("original_protocol_sha256") != hash_file(D10_PROTOCOL_FILE):
        raise ValueError("D10 execution amendment does not match the frozen protocol")
    if amendment.get("scientific_parameters_changed") is not False:
        raise ValueError("Execution amendment must not change scientific parameters")
    n_jobs = int(amendment["amended_n_jobs"])
    if n_jobs <= 0:
        raise ValueError("The amended LightGBM thread limit must be positive")
    return amendment


def load_aligned_data(
    schema: dict[str, Any], d6: dict[str, Any]
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    features, target, metadata = load_modeling_data()
    role_columns = d6["split_assignment_artifact"]["role_columns"]
    assignments = pd.read_csv(
        ASSIGNMENTS_FILE,
        usecols=["meta_collision_index", "meta_collision_year", *role_columns],
        dtype={"meta_collision_index": "string"},
        low_memory=False,
    )
    if len(assignments) != len(features):
        raise AssertionError("Model table and assignment row counts differ")
    model_ids = metadata["meta_collision_index"].astype("string").reset_index(drop=True)
    assignment_ids = assignments["meta_collision_index"].reset_index(drop=True)
    if not model_ids.equals(assignment_ids):
        raise AssertionError("D5 and D6 collision identifiers are not row-aligned")
    if features.columns.tolist() != schema["feature_columns"]:
        raise AssertionError("D10 feature allowlist differs from the frozen schema")
    for column in schema["categorical_feature_columns"]:
        features[column] = features[column].astype("category")
    for column in role_columns:
        assignments[column] = assignments[column].astype("category")
    return features, target.astype("int8"), metadata, assignments


def make_split_specs(d6: dict[str, Any]) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = [
        {
            "protocol": "temporal",
            "seed": "year_based",
            "role_column": "temporal_role",
            "split_slug": "temporal",
        }
    ]
    for seed in d6["random_reference_protocol"]["seeds"]:
        specs.append(
            {
                "protocol": "random_reference",
                "seed": str(seed),
                "role_column": f"random_role_seed_{seed}",
                "split_slug": f"random_seed_{seed}",
            }
        )
    return specs


def class_weights_for_spec(d7: dict[str, Any], spec: dict[str, str]) -> dict[int, float]:
    if spec["protocol"] == "temporal":
        payload = d7["weight_sets"]["temporal"]
    else:
        payload = d7["weight_sets"]["random_reference"][spec["seed"]]
    return {int(code): float(value) for code, value in payload["class_weights"].items()}


def positions_for_spec(
    spec: dict[str, str],
    assignments: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, set[int], set[int]]:
    role = assignments[spec["role_column"]]
    train_positions = np.flatnonzero(role.eq("train").to_numpy())
    validation_positions = np.flatnonzero(role.eq("validation").to_numpy())
    if np.intersect1d(train_positions, validation_positions).size:
        raise AssertionError("Training and validation positions overlap")
    train_years = set(assignments.iloc[train_positions]["meta_collision_year"].astype(int))
    validation_years = set(
        assignments.iloc[validation_positions]["meta_collision_year"].astype(int)
    )
    if 2024 in train_years or 2024 in validation_years:
        raise AssertionError("D10 selected a 2024 row")
    if spec["protocol"] == "temporal":
        if len(train_positions) != 538_461 or train_years != {2018, 2019, 2020, 2021, 2022}:
            raise AssertionError("Unexpected temporal training cohort")
        if len(validation_positions) != 104_258 or validation_years != {2023}:
            raise AssertionError("Unexpected temporal validation cohort")
    else:
        if len(train_positions) != 449_903 or len(validation_positions) != 96_408:
            raise AssertionError("Unexpected random-reference cohort size")
        if not train_years.issubset(set(range(2018, 2024))):
            raise AssertionError("Unexpected random training year")
        if not validation_years.issubset(set(range(2018, 2024))):
            raise AssertionError("Unexpected random validation year")
    return train_positions, validation_positions, train_years, validation_years


def stratified_subset(
    positions: np.ndarray,
    target: pd.Series,
    rows: int,
    seed: int,
) -> np.ndarray:
    if len(positions) <= rows:
        return positions
    selected, _ = train_test_split(
        positions,
        train_size=rows,
        random_state=seed,
        stratify=target.iloc[positions].to_numpy(),
    )
    return np.sort(selected)


def prepare_native_split(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    schema: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    features = list(schema["feature_columns"])
    categorical = list(schema["categorical_feature_columns"])
    numeric = list(schema["numeric_feature_columns"])
    vocabulary = fit_category_vocabulary(X_train, categorical)
    train_prepared = prepare_features(
        X_train,
        feature_columns=features,
        categorical_columns=categorical,
        numeric_columns=numeric,
        category_vocabulary=vocabulary,
    )
    validation_prepared = prepare_features(
        X_validation,
        feature_columns=features,
        categorical_columns=categorical,
        numeric_columns=numeric,
        category_vocabulary=vocabulary,
    )
    if any(train_prepared.unseen_counts.values()):
        raise AssertionError("Training data produced unseen categorical values")
    for column in categorical:
        if not isinstance(train_prepared.frame[column].dtype, pd.CategoricalDtype):
            raise AssertionError(f"Native categorical dtype lost: {column}")
        if list(train_prepared.frame[column].cat.categories) != list(
            validation_prepared.frame[column].cat.categories
        ):
            raise AssertionError(f"Train/validation categories differ: {column}")
    numeric_medians: dict[str, float] = {}
    for column in numeric:
        train_values = train_prepared.frame[column].to_numpy(dtype=float)
        validation_values = validation_prepared.frame[column].to_numpy(dtype=float)
        if np.isinf(train_values).any() or np.isinf(validation_values).any():
            raise AssertionError(f"Infinite numeric values found: {column}")
        median = float(np.nanmedian(train_values))
        if not np.isfinite(median):
            raise AssertionError(f"No finite training median for audit: {column}")
        numeric_medians[column] = median

    bundle = {
        "version": VERSION,
        "feature_columns": features,
        "categorical_columns": categorical,
        "numeric_columns": numeric,
        "category_vocabulary": vocabulary,
        "category_levels_with_unseen": {
            column: [*vocabulary[column], "__UNSEEN__"] for column in categorical
        },
        "numeric_missing_policy": "native_nan",
        "numeric_training_medians_audit_only": numeric_medians,
    }
    audit_rows: list[dict[str, Any]] = []
    for column in categorical:
        audit_rows.append(
            {
                "feature": column,
                "feature_type": "categorical_native",
                "training_levels": len(vocabulary[column]),
                "validation_unseen_count": validation_prepared.unseen_counts[column],
                "training_missing_count": int(X_train[column].isna().sum()),
                "validation_missing_count": int(X_validation[column].isna().sum()),
                "training_median_audit_only": "",
                "value_applied_to_data": "",
            }
        )
    for column in numeric:
        audit_rows.append(
            {
                "feature": column,
                "feature_type": "numeric_native",
                "training_levels": "",
                "validation_unseen_count": "",
                "training_missing_count": int(X_train[column].isna().sum()),
                "validation_missing_count": int(X_validation[column].isna().sum()),
                "training_median_audit_only": numeric_medians[column],
                "value_applied_to_data": "none; NaN retained",
            }
        )
    return train_prepared.frame, validation_prepared.frame, bundle, audit_rows


def make_estimator(
    protocol: dict[str, Any],
    candidate: dict[str, Any],
    class_weights: dict[int, float],
    n_estimators: int,
) -> lgb.LGBMClassifier:
    common = dict(protocol["common_model_parameters"])
    # Runtime-only CPU cap. The frozen scientific protocol remains unchanged.
    common["n_jobs"] = int(require_execution_amendment()["amended_n_jobs"])
    return lgb.LGBMClassifier(
        **common,
        **candidate["parameters"],
        n_estimators=int(n_estimators),
        class_weight=class_weights,
        bagging_seed=MODEL_RANDOM_SEED,
        feature_fraction_seed=MODEL_RANDOM_SEED,
        data_random_seed=MODEL_RANDOM_SEED,
        importance_type="gain",
    )


def prediction_frame(
    collision_ids: pd.Series,
    years: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "meta_collision_index": collision_ids.astype("string").to_numpy(),
            "meta_collision_year": years.astype(int).to_numpy(),
            "target_severity": y_true.astype(np.int8),
            "predicted_severity": y_pred.astype(np.int8),
            "prob_slight": probabilities[:, 0],
            "prob_serious": probabilities[:, 1],
            "prob_fatal": probabilities[:, 2],
        }
    )


def candidate_lookup(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    candidates = protocol["tuning"]["candidates"]
    return {str(candidate["candidate_id"]): candidate for candidate in candidates}


def select_candidate(
    tuning_rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    constraints = protocol["selection_rule"]["eligibility_constraints"]
    minimum_qwk = float(constraints["minimum_qwk"])
    minimum_fatal = float(constraints["minimum_fatal_recall"])
    eligible = [
        row
        for row in tuning_rows
        if float(row["qwk"]) >= minimum_qwk
        and float(row["fatal_recall"]) >= minimum_fatal
    ]
    constraints_met = bool(eligible)
    pool = eligible if constraints_met else list(tuning_rows)
    if not pool:
        raise ValueError("No tuning rows available for model selection")
    selected = sorted(
        pool,
        key=lambda row: (
            -float(row["macro_f1"]),
            -float(row["qwk"]),
            -float(row["fatal_recall"]),
            int(row["complexity_rank"]),
            str(row["candidate_id"]),
        ),
    )[0]
    return selected, constraints_met


def run_smoke() -> None:
    protocol = require_frozen_protocol()
    schema, d6, d7, _ = load_contracts()
    features, target, _, assignments = load_aligned_data(schema, d6)
    spec = make_split_specs(d6)[0]
    train_positions, validation_positions, _, _ = positions_for_spec(spec, assignments)
    train_positions = stratified_subset(
        train_positions, target, SMOKE_TRAIN_ROWS, SMOKE_RANDOM_SEED
    )
    validation_positions = stratified_subset(
        validation_positions, target, SMOKE_VALIDATION_ROWS, SMOKE_RANDOM_SEED + 1
    )
    X_train, X_validation, bundle, _ = prepare_native_split(
        features.iloc[train_positions],
        features.iloc[validation_positions],
        schema,
    )
    y_train = target.iloc[train_positions].to_numpy(dtype=np.int8)
    y_validation = target.iloc[validation_positions].to_numpy(dtype=np.int8)
    candidate = protocol["tuning"]["candidates"][0]
    estimator = make_estimator(
        protocol,
        candidate,
        class_weights_for_spec(d7, spec),
        SMOKE_ESTIMATORS,
    )
    estimator.fit(
        X_train,
        y_train,
        categorical_feature=bundle["categorical_columns"],
        callbacks=[lgb.log_evaluation(period=0)],
    )
    predicted, probabilities = predict_with_probabilities(estimator, X_validation)
    metrics = classification_metrics(y_validation, predicted)
    payload = {
        "version": "D10_SMOKE_V1",
        "status": "PASS",
        "protocol_sha256": hash_file(D10_PROTOCOL_FILE),
        "candidate_id": candidate["candidate_id"],
        "training_rows": len(train_positions),
        "validation_rows": len(validation_positions),
        "feature_count": X_train.shape[1],
        "categorical_feature_count": len(bundle["categorical_columns"]),
        "numeric_missing_policy": bundle["numeric_missing_policy"],
        "probability_rows": probabilities.shape[0],
        "probability_columns": probabilities.shape[1],
        "no_2024": True,
        "metrics_for_pipeline_check_only": metrics,
    }
    write_json(SMOKE_FILE, payload)
    print("D10_SMOKE_ASSERTIONS=PASS")


def run_tuning() -> None:
    protocol = require_frozen_protocol()
    schema, d6, d7, _ = load_contracts()
    features, target, _, assignments = load_aligned_data(schema, d6)
    spec = make_split_specs(d6)[0]
    train_positions, validation_positions, train_years, validation_years = positions_for_spec(
        spec, assignments
    )
    X_train, X_validation, bundle, _ = prepare_native_split(
        features.iloc[train_positions],
        features.iloc[validation_positions],
        schema,
    )
    y_train = target.iloc[train_positions].to_numpy(dtype=np.int8)
    y_validation = target.iloc[validation_positions].to_numpy(dtype=np.int8)
    class_weights = class_weights_for_spec(d7, spec)

    tuning_rows: list[dict[str, Any]] = []
    tuning_confusions: list[dict[str, Any]] = []
    for candidate in protocol["tuning"]["candidates"]:
        candidate_id = str(candidate["candidate_id"])
        print(f"Tuning {candidate_id} ...", flush=True)
        estimator = make_estimator(
            protocol,
            candidate,
            class_weights,
            int(protocol["tuning"]["maximum_estimators"]),
        )
        started = time.perf_counter()
        estimator.fit(
            X_train,
            y_train,
            eval_set=[(X_validation, y_validation)],
            eval_metric="multi_logloss",
            categorical_feature=bundle["categorical_columns"],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=int(protocol["tuning"]["early_stopping_rounds"]),
                    first_metric_only=True,
                    verbose=False,
                ),
                lgb.log_evaluation(period=0),
            ],
        )
        fit_seconds = time.perf_counter() - started
        predicted, _ = predict_with_probabilities(estimator, X_validation)
        metrics = classification_metrics(y_validation, predicted)
        best_iteration = int(estimator.best_iteration_ or estimator.n_estimators_)
        best_logloss = float(estimator.best_score_["valid_0"]["multi_logloss"])
        row = {
            "protocol": "temporal",
            "seed": "year_based",
            "evaluation_role": "validation",
            "training_rows": len(train_positions),
            "validation_rows": len(validation_positions),
            "training_years": ";".join(str(year) for year in sorted(train_years)),
            "validation_years": ";".join(str(year) for year in sorted(validation_years)),
            "candidate_id": candidate_id,
            "complexity_rank": int(candidate["complexity_rank"]),
            **candidate["parameters"],
            "best_iteration": best_iteration,
            "best_validation_multi_logloss": best_logloss,
            "fit_seconds": fit_seconds,
            **metrics,
        }
        tuning_rows.append(row)
        tuning_confusions.extend(
            confusion_rows(
                y_validation,
                predicted,
                protocol="temporal_tuning",
                seed="year_based",
                model=candidate_id,
            )
        )
        print(
            f"{candidate_id}: best_iteration={best_iteration}, "
            f"Macro-F1={metrics['macro_f1']:.5f}, QWK={metrics['qwk']:.5f}, "
            f"fatal_recall={metrics['fatal_recall']:.5f}",
            flush=True,
        )

    selected, constraints_met = select_candidate(tuning_rows, protocol)
    write_csv(TUNING_FILE, tuning_rows)
    write_csv(TUNING_CONFUSION_FILE, tuning_confusions)
    selected_candidate = candidate_lookup(protocol)[str(selected["candidate_id"])]
    selection_payload = {
        "version": VERSION,
        "status": "HYPERPARAMETERS_SELECTED_ON_TEMPORAL_VALIDATION",
        "created_local": CREATED_LOCAL,
        "protocol": D10_PROTOCOL_FILE.relative_to(PROJECT_DIR).as_posix(),
        "protocol_sha256": hash_file(D10_PROTOCOL_FILE),
        "tuning_results": TUNING_FILE.relative_to(PROJECT_DIR).as_posix(),
        "tuning_results_sha256": hash_file(TUNING_FILE),
        "selection_constraints_met": constraints_met,
        "selected_candidate_id": selected_candidate["candidate_id"],
        "selected_complexity_rank": selected_candidate["complexity_rank"],
        "selected_candidate_parameters": selected_candidate["parameters"],
        "selected_n_estimators": int(selected["best_iteration"]),
        "selected_validation_metrics": {
            key: float(selected[key])
            for key in (
                "macro_f1",
                "qwk",
                "fatal_recall",
                "serious_recall",
                "slight_recall",
                "ordinal_mae",
                "mean_asymmetric_cost",
            )
        },
        "selection_rule": protocol["selection_rule"],
        "test_data_used": False,
    }
    if D10_SELECTED_FILE.exists():
        existing = json.loads(D10_SELECTED_FILE.read_text(encoding="utf-8"))
        if existing != selection_payload:
            raise RuntimeError("Existing selected-parameter file differs; refusing overwrite")
    else:
        write_json(D10_SELECTED_FILE, selection_payload)
    print(
        f"SELECTED_D10_CANDIDATE={selected_candidate['candidate_id']} "
        f"N_ESTIMATORS={int(selected['best_iteration'])} "
        f"CONSTRAINTS_MET={constraints_met}"
    )


def require_selected_config(protocol: dict[str, Any]) -> dict[str, Any]:
    if not D10_SELECTED_FILE.exists():
        raise FileNotFoundError("Run --tune before --fit-all")
    selected = json.loads(D10_SELECTED_FILE.read_text(encoding="utf-8"))
    if selected.get("version") != VERSION:
        raise ValueError("Unexpected selected-parameter version")
    if selected.get("protocol_sha256") != hash_file(D10_PROTOCOL_FILE):
        raise ValueError("Selected parameters do not match the frozen protocol")
    if selected.get("tuning_results_sha256") != hash_file(TUNING_FILE):
        raise ValueError("Tuning results changed after parameter selection")
    candidate = candidate_lookup(protocol).get(str(selected["selected_candidate_id"]))
    if candidate is None or candidate["parameters"] != selected["selected_candidate_parameters"]:
        raise ValueError("Selected candidate is not in the frozen candidate set")
    return selected


def write_checkpoint(metrics: pd.DataFrame, selected: dict[str, Any]) -> None:
    temporal = metrics.loc[metrics["protocol"].eq("temporal")].iloc[0]
    random_rows = metrics.loc[metrics["protocol"].eq("random_reference")]
    reference = weighted_logistic_reference()
    lines = [
        "# D10 LightGBM checkpoint",
        "",
        "## Status",
        "",
        "**PASS - temporal-only tuning and six validation fits completed without test access.**",
        "",
        "## Frozen selection",
        "",
        f"- Candidate: **{selected['selected_candidate_id']}**.",
        f"- Fixed trees for every split: **{selected['selected_n_estimators']}**.",
        f"- Eligibility constraints met: **{selected['selection_constraints_met']}**.",
        "- Candidate structure and tree count are identical across the temporal and five random-reference models.",
        "- Split-specific class weights come only from each frozen training partition.",
        "- A documented runtime-only amendment capped LightGBM at **4 CPU threads**; all candidates were restarted under that same cap.",
        "",
        "## Temporal validation (2023)",
        "",
        f"- LightGBM: Macro-F1 **{temporal['macro_f1']:.4f}**, QWK **{temporal['qwk']:.4f}**, Fatal recall **{temporal['fatal_recall']:.4f}**.",
        f"- D8 weighted Logistic reference: Macro-F1 **{reference['macro_f1']:.4f}**, QWK **{reference['qwk']:.4f}**, Fatal recall **{reference['fatal_recall']:.4f}**.",
        f"- Validation-only delta Macro-F1: **{temporal['macro_f1'] - reference['macro_f1']:+.4f}**; this is not the D11 test claim.",
        "",
        "## Random-reference validation",
        "",
        f"- Macro-F1 mean **{random_rows['macro_f1'].mean():.4f}** (SD **{random_rows['macro_f1'].std(ddof=1):.4f}**).",
        f"- QWK mean **{random_rows['qwk'].mean():.4f}** (SD **{random_rows['qwk'].std(ddof=1):.4f}**).",
        f"- Fatal recall mean **{random_rows['fatal_recall'].mean():.4f}** (SD **{random_rows['fatal_recall'].std(ddof=1):.4f}**).",
        "",
        "## Leakage controls and handoff",
        "",
        "- LightGBM used the 17 frozen predictors only; metadata and target were excluded by allowlist.",
        "- Native category vocabularies were fitted separately on training rows; validation-only levels map to `__UNSEEN__`.",
        "- Speed-limit NaN values were retained for native LightGBM missing-value handling.",
        "- No random test record and no 2024 record was scored.",
        "- D11 is the first permitted one-time test evaluation.",
        "",
    ]
    CHECKPOINT_FILE.write_text("\n".join(lines), encoding="utf-8")


def build_artifact_manifest() -> None:
    paths: list[Path] = [
        D10_PROTOCOL_FILE,
        D10_EXECUTION_AMENDMENT_FILE,
        D10_SELECTED_FILE,
        TUNING_FILE,
        TUNING_CONFUSION_FILE,
        METRICS_FILE,
        CONFUSION_FILE,
        PREPROCESS_AUDIT_FILE,
        TRAINING_AUDIT_FILE,
        SMOKE_FILE,
        CHECKPOINT_FILE,
        Path(__file__),
        PROJECT_DIR / "code" / "test_d10_lightgbm.py",
    ]
    paths.extend(sorted(PREDICTION_DIR.glob("*.csv.gz")))
    paths.extend(sorted(MODEL_DIR.glob("**/*.joblib")))
    unique_paths = sorted({path.resolve() for path in paths if path.exists()})
    rows = [
        {
            "relative_path": path.relative_to(PROJECT_DIR.resolve()).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hash_file(path),
        }
        for path in unique_paths
    ]
    write_csv(MANIFEST_FILE, rows)


def run_fit_all() -> None:
    protocol = require_frozen_protocol()
    selected = require_selected_config(protocol)
    schema, d6, d7, _ = load_contracts()
    features, target, metadata, assignments = load_aligned_data(schema, d6)
    selected_candidate = candidate_lookup(protocol)[str(selected["selected_candidate_id"])]
    fixed_estimators = int(selected["selected_n_estimators"])

    metric_rows: list[dict[str, Any]] = []
    confusion_output: list[dict[str, Any]] = []
    preprocessing_output: list[dict[str, Any]] = []
    training_output: list[dict[str, Any]] = []
    for spec in make_split_specs(d6):
        print(f"Fitting frozen model for {spec['split_slug']} ...", flush=True)
        train_positions, validation_positions, train_years, validation_years = positions_for_spec(
            spec, assignments
        )
        X_train, X_validation, bundle, audit_rows = prepare_native_split(
            features.iloc[train_positions],
            features.iloc[validation_positions],
            schema,
        )
        y_train = target.iloc[train_positions].to_numpy(dtype=np.int8)
        y_validation = target.iloc[validation_positions].to_numpy(dtype=np.int8)
        class_weights = class_weights_for_spec(d7, spec)
        estimator = make_estimator(
            protocol,
            selected_candidate,
            class_weights,
            fixed_estimators,
        )
        started = time.perf_counter()
        estimator.fit(
            X_train,
            y_train,
            categorical_feature=bundle["categorical_columns"],
            callbacks=[lgb.log_evaluation(period=0)],
        )
        fit_seconds = time.perf_counter() - started
        prediction_started = time.perf_counter()
        predicted, probabilities = predict_with_probabilities(estimator, X_validation)
        prediction_seconds = time.perf_counter() - prediction_started
        metrics = classification_metrics(y_validation, predicted)
        metric_rows.append(
            {
                "protocol": spec["protocol"],
                "seed": spec["seed"],
                "evaluation_role": "validation",
                "training_rows": len(train_positions),
                "validation_rows": len(validation_positions),
                "training_years": ";".join(str(year) for year in sorted(train_years)),
                "validation_years": ";".join(str(year) for year in sorted(validation_years)),
                "model": MODEL_NAME,
                "class_weighted": True,
                "selected_candidate_id": selected_candidate["candidate_id"],
                "n_estimators": fixed_estimators,
                "fit_seconds": fit_seconds,
                "prediction_seconds": prediction_seconds,
                **metrics,
            }
        )
        confusion_output.extend(
            confusion_rows(
                y_validation,
                predicted,
                protocol=spec["protocol"],
                seed=spec["seed"],
                model=MODEL_NAME,
            )
        )
        for audit in audit_rows:
            preprocessing_output.append(
                {
                    "protocol": spec["protocol"],
                    "seed": spec["seed"],
                    **audit,
                }
            )
        training_output.append(
            {
                "protocol": spec["protocol"],
                "seed": spec["seed"],
                "role_column": spec["role_column"],
                "training_rows": len(train_positions),
                "validation_rows": len(validation_positions),
                "slight_training_rows": int(np.sum(y_train == 0)),
                "serious_training_rows": int(np.sum(y_train == 1)),
                "fatal_training_rows": int(np.sum(y_train == 2)),
                "slight_class_weight": class_weights[0],
                "serious_class_weight": class_weights[1],
                "fatal_class_weight": class_weights[2],
                "selected_candidate_id": selected_candidate["candidate_id"],
                "n_estimators": fixed_estimators,
                "native_categorical_features": len(bundle["categorical_columns"]),
                "numeric_features": len(bundle["numeric_columns"]),
                "fit_seconds": fit_seconds,
            }
        )

        split_dir = MODEL_DIR / spec["split_slug"]
        split_dir.mkdir(parents=True, exist_ok=True)
        bundle.update(
            {
                "protocol": spec["protocol"],
                "seed": spec["seed"],
                "role_column": spec["role_column"],
                "training_rows": len(train_positions),
                "training_years": sorted(train_years),
                "validation_rows": len(validation_positions),
                "validation_years": sorted(validation_years),
                "protocol_sha256": hash_file(D10_PROTOCOL_FILE),
                "execution_amendment_sha256": hash_file(D10_EXECUTION_AMENDMENT_FILE),
            }
        )
        joblib.dump(bundle, split_dir / "preprocessing_bundle.joblib", compress=3)
        joblib.dump(
            {
                "version": VERSION,
                "protocol": spec["protocol"],
                "seed": spec["seed"],
                "role_column": spec["role_column"],
                "model_name": MODEL_NAME,
                "estimator": estimator,
                "class_weight": class_weights,
                "selected_candidate_id": selected_candidate["candidate_id"],
                "selected_candidate_parameters": selected_candidate["parameters"],
                "n_estimators": fixed_estimators,
                "prediction_rule": "argmax over class probabilities",
                "target_codes": list(TARGET_CODES),
                "protocol_sha256": hash_file(D10_PROTOCOL_FILE),
                "execution_amendment_sha256": hash_file(D10_EXECUTION_AMENDMENT_FILE),
                "selected_config_sha256": hash_file(D10_SELECTED_FILE),
            },
            split_dir / f"{MODEL_NAME}.joblib",
            compress=3,
        )
        validation_ids = metadata.iloc[validation_positions]["meta_collision_index"]
        validation_year_series = assignments.iloc[validation_positions]["meta_collision_year"]
        save_prediction(
            PREDICTION_DIR / f"{spec['split_slug']}__{MODEL_NAME}.csv.gz",
            prediction_frame(
                validation_ids,
                validation_year_series,
                y_validation,
                predicted,
                probabilities,
            ),
        )
        print(
            f"{spec['split_slug']}: Macro-F1={metrics['macro_f1']:.5f}, "
            f"QWK={metrics['qwk']:.5f}, fatal_recall={metrics['fatal_recall']:.5f}",
            flush=True,
        )

    write_csv(METRICS_FILE, metric_rows)
    write_csv(CONFUSION_FILE, confusion_output)
    write_csv(PREPROCESS_AUDIT_FILE, preprocessing_output)
    write_csv(TRAINING_AUDIT_FILE, training_output)
    metrics_frame = pd.DataFrame(metric_rows)
    selected_temporal = selected["selected_validation_metrics"]
    temporal_final = metrics_frame.loc[metrics_frame["protocol"].eq("temporal")].iloc[0]
    for metric in ("macro_f1", "qwk", "fatal_recall"):
        if not np.isclose(
            float(temporal_final[metric]), float(selected_temporal[metric]), rtol=0, atol=1e-12
        ):
            raise AssertionError(f"Temporal refit differs from selected tuning model: {metric}")
    write_checkpoint(metrics_frame, selected)
    build_artifact_manifest()
    print("D10 validation metric rows:", len(metric_rows))
    print("D10 prediction files:", len(list(PREDICTION_DIR.glob("*.csv.gz"))))
    print("FINAL_D10_TRAINING_ASSERTIONS=PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true", help="Freeze the D10 protocol")
    mode.add_argument("--smoke", action="store_true", help="Run a small pipeline check")
    mode.add_argument("--tune", action="store_true", help="Tune on temporal validation only")
    mode.add_argument("--fit-all", action="store_true", help="Fit six models with frozen parameters")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.freeze:
        freeze_protocol()
    elif arguments.smoke:
        run_smoke()
    elif arguments.tune:
        run_tuning()
    elif arguments.fit_all:
        run_fit_all()


if __name__ == "__main__":
    main()
