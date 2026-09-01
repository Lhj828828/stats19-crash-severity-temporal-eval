"""Fit the D9 proportional-odds logistic baseline on validation data only.

The model is intentionally unweighted because statsmodels OrderedModel does
not implement observation or class weights. Categorical predictors use a
training-fitted reference-level encoding so the unpenalized ordinal model has
no implicit intercept or full-dummy collinearity. The 2024 test partition is
never used here.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import qr
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.miscmodels.ordinal_model import OrderedModel

from baseline_modeling import (
    TARGET_CODES,
    UNSEEN_TOKEN,
    classification_metrics,
    confusion_rows,
    fit_category_vocabulary,
    prepare_features,
)
from modeling_data import load_modeling_data


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCHEMA_FILE = PROJECT_DIR / "config" / "d5_dataset_schema.json"
D6_PROTOCOL_FILE = PROJECT_DIR / "config" / "d6_analysis_protocol.json"
D7_CONFIG_FILE = PROJECT_DIR / "config" / "d7_training_inputs.json"
D8_CONFIG_FILE = PROJECT_DIR / "config" / "d8_baseline_protocol.json"
ASSIGNMENTS_FILE = PROJECT_DIR / "data" / "processed" / "d6_split_assignments.csv.gz"

D9_CONFIG_FILE = PROJECT_DIR / "config" / "d9_ordered_logit_protocol.json"
MODEL_DIR = PROJECT_DIR / "models" / "d9"
RESULT_DIR = PROJECT_DIR / "results" / "d9"
PREDICTION_DIR = RESULT_DIR / "validation_predictions"
LOG_DIR = PROJECT_DIR / "logs"
METRICS_FILE = RESULT_DIR / "d9_validation_metrics.csv"
CONFUSION_FILE = RESULT_DIR / "d9_validation_confusion_matrices.csv"
PREPROCESS_AUDIT_FILE = LOG_DIR / "d9_preprocessing_audit.csv"
BENCHMARK_FILE = LOG_DIR / "d9_runtime_benchmark.json"
SMOKE_FILE = LOG_DIR / "d9_smoke_test.json"
CHECKPOINT_FILE = LOG_DIR / "d9_checkpoint.md"

MODEL_NAME = "ordered_logit_unweighted"
OPTIMIZER = "lbfgs"
MAX_ITER = 200
PGTOL = 1e-4
FACTR = 1e7
SUBSET_SEED = 20_260_829
SMOKE_TRAIN_ROWS = 10_000
SMOKE_VALIDATION_ROWS = 5_000
BENCHMARK_TRAIN_ROWS = 100_000
BENCHMARK_VALIDATION_ROWS = 25_000
FULL_RUNTIME_GATE_SECONDS = 180.0


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows generated for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_prediction(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        lineterminator="\n",
    )


def load_contracts() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    d6 = json.loads(D6_PROTOCOL_FILE.read_text(encoding="utf-8"))
    d7 = json.loads(D7_CONFIG_FILE.read_text(encoding="utf-8"))
    d8 = json.loads(D8_CONFIG_FILE.read_text(encoding="utf-8"))
    if schema.get("status") != "QC_PASSED_AND_FROZEN":
        raise ValueError("D9 requires the frozen D5 schema")
    if d6.get("version") != "D6_V2" or d6.get("status") != "FROZEN_BEFORE_MODEL_TRAINING":
        raise ValueError("D9 requires the frozen D6_V2 protocol")
    if d7.get("status") != "TRAINING_INPUTS_AUDITED_AND_FROZEN":
        raise ValueError("D9 requires the completed D7 audit")
    if d8.get("version") != "D8_V1":
        raise ValueError("D9 requires the frozen D8 baseline specification")
    if hash_file(ASSIGNMENTS_FILE) != d7["upstream"]["D6_assignments_sha256"]:
        raise ValueError("Frozen D6 assignments changed after D7")
    if schema["feature_columns"] != d8["features"]["allowlist"]:
        raise ValueError("D8 and D9 feature allowlists differ")
    return schema, d6, d7


def make_protocol_config(
    schema: dict[str, object], d6: dict[str, object], d7: dict[str, object]
) -> dict[str, object]:
    return {
        "version": "D9_V3",
        "status": "SPECIFICATION_FROZEN_AFTER_RUNTIME_PREFLIGHT",
        "freeze_date_local": "2026-08-29",
        "upstream": {
            "D5_schema": SCHEMA_FILE.relative_to(PROJECT_DIR).as_posix(),
            "D5_schema_sha256": hash_file(SCHEMA_FILE),
            "D6_protocol": D6_PROTOCOL_FILE.relative_to(PROJECT_DIR).as_posix(),
            "D6_protocol_sha256": hash_file(D6_PROTOCOL_FILE),
            "D7_training_inputs": D7_CONFIG_FILE.relative_to(PROJECT_DIR).as_posix(),
            "D7_training_inputs_sha256": hash_file(D7_CONFIG_FILE),
            "D8_baseline_protocol": D8_CONFIG_FILE.relative_to(PROJECT_DIR).as_posix(),
            "D8_baseline_protocol_sha256": hash_file(D8_CONFIG_FILE),
            "D6_assignments_sha256": d7["upstream"]["D6_assignments_sha256"],
        },
        "evaluation_scope": {
            "allowed": "training and validation roles from 2018-2023 only",
            "forbidden": "all temporal/random test performance and all 2024 performance",
            "temporal_training_years": d6["temporal_protocol"]["train_years"],
            "temporal_validation_years": d6["temporal_protocol"]["validation_years"],
            "random_seeds": d6["random_reference_protocol"]["seeds"],
        },
        "features": {
            "allowlist": schema["feature_columns"],
            "categorical": schema["categorical_feature_columns"],
            "numeric": schema["numeric_feature_columns"],
            "metadata_excluded": schema["metadata_columns"],
            "target_excluded": schema["target_column"],
        },
        "encoding": {
            "fit_scope": "each training partition only",
            "categorical": (
                "training vocabulary with one reference level dropped per feature; "
                "__UNSEEN__ is explicit during mapping but its unidentified all-zero "
                "training column is removed"
            ),
            "numeric": "training median imputation and training-fitted standardization",
            "reason": "OrderedModel is unpenalized and must not receive a constant or full-dummy collinearity",
            "rank_reduction": (
                "training-design QR with column pivoting removes exact linear dependencies; "
                "the target and validation performance are not used"
            ),
        },
        "model": {
            "name": MODEL_NAME,
            "implementation": "statsmodels.miscmodels.ordinal_model.OrderedModel",
            "distribution": "logit",
            "link": "proportional odds",
            "class_weight": None,
            "weight_note": "OrderedModel has no class-weight interface; this is an unweighted explanatory baseline",
            "optimizer": OPTIMIZER,
            "max_iter": MAX_ITER,
            "pgtol": PGTOL,
            "factr": FACTR,
            "prediction_rule": "argmax over three ordered-class probabilities; no threshold tuning",
        },
        "runtime_gate": {
            "benchmark_training_rows": BENCHMARK_TRAIN_ROWS,
            "benchmark_validation_rows": BENCHMARK_VALIDATION_ROWS,
            "benchmark_split": "temporal",
            "full_fit_if": (
                f"converged and fit_seconds <= {FULL_RUNTIME_GATE_SECONDS:.0f} on the fixed benchmark"
            ),
            "fallback": (
                f"fixed stratified training subset of {BENCHMARK_TRAIN_ROWS} rows per split; "
                "evaluate the complete validation partition"
            ),
            "subset_seed": SUBSET_SEED,
            "decision_basis": "runtime and convergence only, never validation performance",
        },
        "role_in_paper": (
            "secondary ordinal baseline for interpretive hierarchy; not the primary "
            "comparison against weighted LightGBM"
        ),
        "runtime_preflight_amendment": {
            "reason": [
                "D9_V1 smoke fitting identified five exact linear dependencies in the reference-coded design (rank 117 of 122; condition number approximately 1.8e17)",
                "After QR reduction, pgtol=1e-6 reached 300 iterations without convergence; a runtime-only trial with pgtol=1e-4 converged in 100 iterations without warnings",
            ],
            "change": (
                "add training-only QR rank reduction and use the pre-specified practical "
                "L-BFGS convergence settings pgtol=1e-4 and max_iter=200"
            ),
            "information_used": "design matrix rank, convergence status and runtime only; no validation metric",
        },
        "test_embargo": d7["test_embargo"],
    }


def freeze_protocol() -> None:
    schema, d6, d7 = load_contracts()
    write_json(D9_CONFIG_FILE, make_protocol_config(schema, d6, d7))
    print("D9 protocol frozen before Ordered Logit fitting.")
    print("D9_FREEZE_ASSERTIONS=PASS")


def require_frozen_protocol() -> dict[str, object]:
    if not D9_CONFIG_FILE.exists():
        raise FileNotFoundError("Run D9 with --freeze before fitting")
    config = json.loads(D9_CONFIG_FILE.read_text(encoding="utf-8"))
    if config.get("version") != "D9_V3":
        raise ValueError("Unexpected D9 protocol version")
    expected = {
        "D5_schema_sha256": hash_file(SCHEMA_FILE),
        "D6_protocol_sha256": hash_file(D6_PROTOCOL_FILE),
        "D7_training_inputs_sha256": hash_file(D7_CONFIG_FILE),
        "D8_baseline_protocol_sha256": hash_file(D8_CONFIG_FILE),
        "D6_assignments_sha256": hash_file(ASSIGNMENTS_FILE),
    }
    for key, value in expected.items():
        if config["upstream"][key] != value:
            raise ValueError(f"Frozen upstream artifact changed: {key}")
    return config


def load_aligned_data(
    schema: dict[str, object], d6: dict[str, object]
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
    if not metadata["meta_collision_index"].astype("string").reset_index(drop=True).equals(
        assignments["meta_collision_index"].reset_index(drop=True)
    ):
        raise AssertionError("D5 and D6 collision identifiers are not row-aligned")
    if features.columns.tolist() != schema["feature_columns"]:
        raise AssertionError("D9 feature allowlist differs from the frozen schema")
    for column in schema["categorical_feature_columns"]:
        features[column] = features[column].astype("category")
    for column in role_columns:
        assignments[column] = assignments[column].astype("category")
    return features, target.astype("int8"), metadata, assignments


def stratified_subset(
    positions: np.ndarray, target: pd.Series, rows: int, seed: int
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


def make_split_specs(d6: dict[str, object]) -> list[dict[str, str]]:
    specs = [
        {
            "protocol": "temporal",
            "seed": "year_based",
            "role_column": "temporal_role",
            "slug": "temporal",
        }
    ]
    for seed in d6["random_reference_protocol"]["seeds"]:
        specs.append(
            {
                "protocol": "random_reference",
                "seed": str(seed),
                "role_column": f"random_role_seed_{seed}",
                "slug": f"random_seed_{seed}",
            }
        )
    return specs


def make_reference_preprocessor(
    *,
    categorical: list[str],
    numeric: list[str],
    vocabulary: dict[str, list[str]],
) -> tuple[ColumnTransformer, dict[str, str]]:
    baselines = {column: vocabulary[column][0] for column in categorical}
    categories = [[*vocabulary[column], UNSEEN_TOKEN] for column in categorical]
    encoder = OneHotEncoder(
        categories=categories,
        drop=[baselines[column] for column in categorical],
        handle_unknown="error",
        sparse_output=True,
        dtype=np.float64,
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", encoder, categorical),
            ("numeric", numeric_pipeline, numeric),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )
    return preprocessor, baselines


def encode_split(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    schema: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, dict[str, object], list[dict[str, object]]]:
    categorical = list(schema["categorical_feature_columns"])
    numeric = list(schema["numeric_feature_columns"])
    feature_columns = list(schema["feature_columns"])
    vocabulary = fit_category_vocabulary(X_train, categorical)
    prepared_train = prepare_features(
        X_train,
        feature_columns=feature_columns,
        categorical_columns=categorical,
        numeric_columns=numeric,
        category_vocabulary=vocabulary,
    )
    prepared_validation = prepare_features(
        X_validation,
        feature_columns=feature_columns,
        categorical_columns=categorical,
        numeric_columns=numeric,
        category_vocabulary=vocabulary,
    )
    if any(prepared_train.unseen_counts.values()):
        raise AssertionError("Training data produced unseen categories")

    preprocessor, baselines = make_reference_preprocessor(
        categorical=categorical,
        numeric=numeric,
        vocabulary=vocabulary,
    )
    encoded_train = preprocessor.fit_transform(prepared_train.frame)
    encoded_validation = preprocessor.transform(prepared_validation.frame)
    if not sparse.issparse(encoded_train) or not sparse.issparse(encoded_validation):
        raise AssertionError("D9 expected sparse intermediate matrices")
    if encoded_train.shape[0] != len(X_train) or encoded_validation.shape[0] != len(X_validation):
        raise AssertionError("D9 preprocessing changed row counts")

    all_names = np.asarray(preprocessor.get_feature_names_out(), dtype=object)
    identified = np.asarray(encoded_train.getnnz(axis=0)).ravel() > 0
    if not identified.any():
        raise AssertionError("No identified D9 columns remain")
    retained_names = all_names[identified].tolist()
    removed_names = all_names[~identified].tolist()
    identified_train = encoded_train[:, identified].toarray()
    identified_validation = encoded_validation[:, identified].toarray()
    if not np.isfinite(identified_train).all() or not np.isfinite(identified_validation).all():
        raise AssertionError("D9 encoded matrices contain non-finite values")
    if np.any(np.ptp(identified_train, axis=0) == 0):
        raise AssertionError("D9 retained a constant encoded column")

    _, upper, pivots = qr(
        identified_train,
        mode="economic",
        pivoting=True,
        check_finite=False,
    )
    diagonal = np.abs(np.diag(upper))
    tolerance = (
        max(identified_train.shape)
        * np.finfo(identified_train.dtype).eps
        * (diagonal[0] if len(diagonal) else 0.0)
    )
    rank = int(np.sum(diagonal > tolerance))
    independent = np.sort(np.asarray(pivots[:rank], dtype=int))
    dependent = np.setdiff1d(
        np.arange(identified_train.shape[1], dtype=int), independent
    )
    train_dense = identified_train[:, independent]
    validation_dense = identified_validation[:, independent]
    independent_names = [retained_names[index] for index in independent]
    dependent_names = [retained_names[index] for index in dependent]
    if rank != train_dense.shape[1]:
        raise AssertionError("D9 QR reduction did not retain the identified rank")

    imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
    medians = {
        column: float(value)
        for column, value in zip(numeric, imputer.statistics_, strict=True)
    }
    audit_rows: list[dict[str, object]] = []
    for column in categorical:
        audit_rows.append(
            {
                "feature": column,
                "feature_type": "categorical",
                "training_levels": len(vocabulary[column]),
                "reference_level": baselines[column],
                "training_missing_count": int(X_train[column].isna().sum()),
                "validation_missing_count": int(X_validation[column].isna().sum()),
                "validation_unseen_count": prepared_validation.unseen_counts[column],
                "training_fitted_imputation_value": "",
            }
        )
    for column in numeric:
        audit_rows.append(
            {
                "feature": column,
                "feature_type": "numeric",
                "training_levels": "",
                "reference_level": "",
                "training_missing_count": int(X_train[column].isna().sum()),
                "validation_missing_count": int(X_validation[column].isna().sum()),
                "validation_unseen_count": "",
                "training_fitted_imputation_value": medians[column],
            }
        )
    bundle = {
        "version": "D9_V3",
        "feature_columns": feature_columns,
        "categorical_columns": categorical,
        "numeric_columns": numeric,
        "category_vocabulary": vocabulary,
        "reference_levels": baselines,
        "preprocessor": preprocessor,
        "identified_column_mask": identified,
        "identified_feature_names_before_rank_reduction": retained_names,
        "independent_column_indices": independent,
        "retained_feature_names": independent_names,
        "removed_unidentified_feature_names": removed_names,
        "removed_linearly_dependent_feature_names": dependent_names,
        "validation_unseen_counts": prepared_validation.unseen_counts,
    }
    return train_dense, validation_dense, bundle, audit_rows


def fit_ordered_logit(
    X_train: np.ndarray, y_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, object], OrderedModel]:
    if set(np.unique(y_train)) != set(TARGET_CODES):
        raise ValueError("D9 training subset must contain all three classes")
    model = OrderedModel(y_train, X_train, distr="logit")
    fit_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(
            method=OPTIMIZER,
            maxiter=MAX_ITER,
            disp=False,
            pgtol=PGTOL,
            factr=FACTR,
        )
    fit_seconds = time.perf_counter() - fit_start
    params = np.asarray(result.params, dtype=float)
    thresholds = np.asarray(model.transform_threshold_params(params), dtype=float)
    retvals = dict(getattr(result, "mle_retvals", {}))
    converged = bool(retvals.get("converged", False))
    iterations = int(retvals.get("iterations", retvals.get("fcalls", 0)))
    warning_messages = sorted({str(item.message) for item in caught})
    if not np.isfinite(params).all():
        raise AssertionError("D9 produced non-finite model parameters")
    if not np.all(np.diff(thresholds[1:-1]) > 0):
        raise AssertionError("D9 cut points are not strictly ordered")
    fit_info = {
        "fit_seconds": fit_seconds,
        "converged": converged,
        "iterations": iterations,
        "warning_count": len(warning_messages),
        "warning_messages": warning_messages,
        "optimizer_return": {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in retvals.items()
            if isinstance(value, (str, int, float, bool, np.generic))
        },
    }
    return params, thresholds, fit_info, model


def compact_probabilities(
    X: np.ndarray, beta: np.ndarray, finite_thresholds: np.ndarray
) -> np.ndarray:
    linear = X @ beta
    cumulative = expit(finite_thresholds[None, :] - linear[:, None])
    probabilities = np.column_stack(
        [cumulative[:, 0], cumulative[:, 1] - cumulative[:, 0], 1.0 - cumulative[:, 1]]
    )
    if not np.isfinite(probabilities).all():
        raise AssertionError("D9 compact probabilities are non-finite")
    if np.min(probabilities) < -1e-12:
        raise AssertionError("D9 produced negative probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-10):
        raise AssertionError("D9 probabilities do not sum to one")
    return probabilities


def positions_for_spec(
    spec: dict[str, str], assignments: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, set[int], set[int]]:
    roles = assignments[spec["role_column"]]
    train_positions = np.flatnonzero(roles.eq("train").to_numpy())
    validation_positions = np.flatnonzero(roles.eq("validation").to_numpy())
    train_years = set(assignments.iloc[train_positions]["meta_collision_year"].astype(int))
    validation_years = set(
        assignments.iloc[validation_positions]["meta_collision_year"].astype(int)
    )
    if 2024 in train_years or 2024 in validation_years:
        raise AssertionError("D9 attempted to use 2024 for fitting or validation")
    if spec["protocol"] == "temporal":
        if train_years != {2018, 2019, 2020, 2021, 2022}:
            raise AssertionError(f"Unexpected temporal training years: {train_years}")
        if validation_years != {2023}:
            raise AssertionError(f"Unexpected temporal validation years: {validation_years}")
    return train_positions, validation_positions, train_years, validation_years


def run_runtime_trial(
    *,
    train_rows: int,
    validation_rows: int,
    output_file: Path,
    output_version: str,
) -> dict[str, object]:
    config = require_frozen_protocol()
    schema, d6, _ = load_contracts()
    features, target, _, assignments = load_aligned_data(schema, d6)
    spec = make_split_specs(d6)[0]
    train_positions, validation_positions, train_years, validation_years = positions_for_spec(
        spec, assignments
    )
    train_positions = stratified_subset(
        train_positions, target, train_rows, SUBSET_SEED
    )
    validation_positions = stratified_subset(
        validation_positions, target, validation_rows, SUBSET_SEED + 1
    )
    preprocess_start = time.perf_counter()
    encoded_train, encoded_validation, bundle, _ = encode_split(
        features.iloc[train_positions], features.iloc[validation_positions], schema
    )
    preprocessing_seconds = time.perf_counter() - preprocess_start
    params, thresholds, fit_info, model = fit_ordered_logit(
        encoded_train, target.iloc[train_positions].to_numpy(dtype=np.int8)
    )
    statsmodels_prob = np.asarray(
        model.predict(params, exog=encoded_validation, which="prob"), dtype=float
    )
    compact_prob = compact_probabilities(
        encoded_validation,
        params[: encoded_train.shape[1]],
        thresholds[1:-1],
    )
    if not np.allclose(statsmodels_prob, compact_prob, rtol=0, atol=1e-11):
        raise AssertionError("Compact D9 prediction formula differs from statsmodels")
    full_decision = bool(
        fit_info["converged"]
        and float(fit_info["fit_seconds"]) <= FULL_RUNTIME_GATE_SECONDS
    )
    payload = {
        "version": output_version,
        "status": "PASS",
        "protocol_sha256": hash_file(D9_CONFIG_FILE),
        "training_rows": len(train_positions),
        "validation_rows": len(validation_positions),
        "training_years": sorted(train_years),
        "validation_years": sorted(validation_years),
        "encoded_features": encoded_train.shape[1],
        "removed_unidentified_columns": len(bundle["removed_unidentified_feature_names"]),
        "removed_linear_dependencies": len(bundle["removed_linearly_dependent_feature_names"]),
        "preprocessing_seconds": preprocessing_seconds,
        **fit_info,
        "no_2024": True,
        "validation_performance_inspected": False,
    }
    if output_version == "D9_BENCHMARK_V1":
        payload["full_runtime_gate_seconds"] = FULL_RUNTIME_GATE_SECONDS
        payload["execution_decision"] = "full" if full_decision else "subset_100000"
        payload["decision_basis"] = "runtime and convergence only"
    write_json(output_file, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"{output_version}_ASSERTIONS=PASS")
    del encoded_train, encoded_validation, model, params, thresholds
    gc.collect()
    return payload


def resolve_training_mode(requested: str) -> str:
    if requested in {"full", "subset"}:
        return requested
    if not BENCHMARK_FILE.exists():
        raise FileNotFoundError("Run D9 --benchmark before --run --mode auto")
    benchmark = json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))
    if benchmark.get("protocol_sha256") != hash_file(D9_CONFIG_FILE):
        raise ValueError("D9 benchmark does not match the frozen protocol")
    return "full" if benchmark.get("execution_decision") == "full" else "subset"


def prediction_frame(
    *,
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


def fit_one_split(
    *,
    spec: dict[str, str],
    training_mode: str,
    schema: dict[str, object],
    features: pd.DataFrame,
    target: pd.Series,
    metadata: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    train_positions, validation_positions, train_years, validation_years = positions_for_spec(
        spec, assignments
    )
    available_training_rows = len(train_positions)
    if training_mode == "subset":
        split_seed = SUBSET_SEED if spec["protocol"] == "temporal" else SUBSET_SEED + int(spec["seed"])
        train_positions = stratified_subset(
            train_positions, target, BENCHMARK_TRAIN_ROWS, split_seed
        )
    X_train = features.iloc[train_positions]
    X_validation = features.iloc[validation_positions]
    y_train = target.iloc[train_positions].to_numpy(dtype=np.int8)
    y_validation = target.iloc[validation_positions].to_numpy(dtype=np.int8)

    preprocess_start = time.perf_counter()
    encoded_train, encoded_validation, bundle, audit_rows = encode_split(
        X_train, X_validation, schema
    )
    preprocessing_seconds = time.perf_counter() - preprocess_start
    params, thresholds, fit_info, model = fit_ordered_logit(encoded_train, y_train)
    if not fit_info["converged"]:
        raise RuntimeError(f"Ordered Logit did not converge for {spec['slug']}")

    prediction_start = time.perf_counter()
    probabilities = compact_probabilities(
        encoded_validation,
        params[: encoded_train.shape[1]],
        thresholds[1:-1],
    )
    statsmodels_prob = np.asarray(
        model.predict(params, exog=encoded_validation, which="prob"), dtype=float
    )
    if not np.allclose(probabilities, statsmodels_prob, rtol=0, atol=1e-11):
        raise AssertionError("Stored compact D9 model would change predictions")
    predicted = np.argmax(probabilities, axis=1).astype(np.int8)
    prediction_seconds = time.perf_counter() - prediction_start
    metrics = classification_metrics(y_validation, predicted)

    split_dir = MODEL_DIR / spec["slug"]
    split_dir.mkdir(parents=True, exist_ok=True)
    bundle.update(
        {
            "protocol": spec["protocol"],
            "seed": spec["seed"],
            "role_column": spec["role_column"],
            "training_mode": training_mode,
            "available_training_rows": available_training_rows,
            "used_training_rows": len(train_positions),
            "training_years": sorted(train_years),
        }
    )
    joblib.dump(bundle, split_dir / "preprocessing_bundle.joblib", compress=3)
    model_artifact = {
        "version": "D9_V3",
        "model_name": MODEL_NAME,
        "distribution": "logit",
        "class_weight": None,
        "feature_count": encoded_train.shape[1],
        "beta": params[: encoded_train.shape[1]],
        "finite_thresholds": thresholds[1:-1],
        "raw_threshold_parameters": params[encoded_train.shape[1] :],
        "converged": fit_info["converged"],
        "iterations": fit_info["iterations"],
        "prediction_rule": "argmax over ordered-class probabilities",
        "training_mode": training_mode,
        "used_training_rows": len(train_positions),
    }
    joblib.dump(model_artifact, split_dir / f"{MODEL_NAME}.joblib", compress=3)

    validation_ids = metadata.iloc[validation_positions]["meta_collision_index"]
    validation_year_series = assignments.iloc[validation_positions]["meta_collision_year"]
    save_prediction(
        PREDICTION_DIR / f"{spec['slug']}__{MODEL_NAME}.csv.gz",
        prediction_frame(
            collision_ids=validation_ids,
            years=validation_year_series,
            y_true=y_validation,
            y_pred=predicted,
            probabilities=probabilities,
        ),
    )
    metric_row = {
        "protocol": spec["protocol"],
        "seed": spec["seed"],
        "evaluation_role": "validation",
        "training_mode": training_mode,
        "available_training_rows": available_training_rows,
        "training_rows": len(train_positions),
        "validation_rows": len(validation_positions),
        "training_years": ";".join(str(year) for year in sorted(train_years)),
        "validation_years": ";".join(str(year) for year in sorted(validation_years)),
        "model": MODEL_NAME,
        "class_weighted": False,
        "encoded_features": encoded_train.shape[1],
        "removed_unidentified_columns": len(bundle["removed_unidentified_feature_names"]),
        "removed_linear_dependencies": len(bundle["removed_linearly_dependent_feature_names"]),
        "preprocessing_seconds": preprocessing_seconds,
        "fit_seconds": fit_info["fit_seconds"],
        "prediction_seconds": prediction_seconds,
        "n_iter": fit_info["iterations"],
        "converged": fit_info["converged"],
        "warning_count": fit_info["warning_count"],
        "warning_messages": json.dumps(
            fit_info["warning_messages"], ensure_ascii=True, separators=(",", ":")
        ),
        **metrics,
    }
    confusion = confusion_rows(
        y_validation,
        predicted,
        protocol=spec["protocol"],
        seed=spec["seed"],
        model=MODEL_NAME,
    )
    for row in audit_rows:
        row.update(
            {
                "protocol": spec["protocol"],
                "seed": spec["seed"],
                "training_mode": training_mode,
                "encoded_features": encoded_train.shape[1],
                "removed_unidentified_columns": len(bundle["removed_unidentified_feature_names"]),
                "removed_linear_dependencies": len(bundle["removed_linearly_dependent_feature_names"]),
            }
        )
    print(
        f"Completed {spec['slug']}: mode={training_mode}, train={len(train_positions):,}, "
        f"validation={len(validation_positions):,}, encoded={encoded_train.shape[1]}, "
        f"fit={fit_info['fit_seconds']:.1f}s"
    )
    del encoded_train, encoded_validation, model, params, thresholds, probabilities
    gc.collect()
    return metric_row, confusion, audit_rows


def write_checkpoint(metrics: pd.DataFrame, training_mode: str) -> None:
    temporal = metrics.loc[metrics["protocol"].eq("temporal")].iloc[0]
    random = metrics.loc[metrics["protocol"].eq("random_reference")]
    warning_records = int(metrics["warning_count"].sum())
    lines = [
        "# D9 ordered-logit checkpoint",
        "",
        "## Status",
        "",
        "**PASS - unweighted proportional-odds logistic baseline completed on validation data only.**",
        "",
        "## Frozen interpretation",
        "",
        "- Ordered Logit is a secondary ordinal baseline, not the primary comparator for weighted LightGBM.",
        "- statsmodels OrderedModel has no class-weight interface; the model is explicitly unweighted.",
        "- Reference-level encoding was fitted separately on each training partition.",
        "- No test metric and no 2024 performance was calculated.",
        f"- Runtime gate selected **{training_mode}** training mode.",
        f"- Optimizer convergence: **{int(metrics['converged'].sum())}/{len(metrics)}**; warning records: **{warning_records}**.",
        "- Ordered-model coefficients and standard errors are not used for inferential claims.",
        "",
        "## Temporal validation (2023)",
        "",
        f"- Macro-F1 **{temporal['macro_f1']:.4f}**, QWK **{temporal['qwk']:.4f}**, "
        f"Fatal recall **{temporal['fatal_recall']:.4f}**, ordinal MAE **{temporal['ordinal_mae']:.4f}**.",
        "",
        "## Random-reference validation",
        "",
        f"- Macro-F1 mean **{random['macro_f1'].mean():.4f}** "
        f"(SD **{random['macro_f1'].std(ddof=1):.4f}**).",
        f"- QWK mean **{random['qwk'].mean():.4f}** "
        f"(SD **{random['qwk'].std(ddof=1):.4f}**).",
        "",
        "## Handoff",
        "",
        "- D10 may tune class-weighted LightGBM using training and validation data only.",
        "- The 2024 temporal test and all random test partitions remain sealed until D11.",
        "",
    ]
    CHECKPOINT_FILE.write_text("\n".join(lines), encoding="utf-8")


def run_all(requested_mode: str) -> None:
    require_frozen_protocol()
    schema, d6, _ = load_contracts()
    training_mode = resolve_training_mode(requested_mode)
    features, target, metadata, assignments = load_aligned_data(schema, d6)
    metrics: list[dict[str, object]] = []
    confusions: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for spec in make_split_specs(d6):
        metric, confusion, audit = fit_one_split(
            spec=spec,
            training_mode=training_mode,
            schema=schema,
            features=features,
            target=target,
            metadata=metadata,
            assignments=assignments,
        )
        metrics.append(metric)
        confusions.extend(confusion)
        audits.extend(audit)
    write_csv(METRICS_FILE, metrics)
    write_csv(CONFUSION_FILE, confusions)
    write_csv(PREPROCESS_AUDIT_FILE, audits)
    write_checkpoint(pd.DataFrame(metrics), training_mode)
    print("D9 validation metric rows:", len(metrics))
    print("D9 prediction files:", len(list(PREDICTION_DIR.glob("*.csv.gz"))))
    print("FINAL_D9_ASSERTIONS=PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--smoke", action="store_true")
    action.add_argument("--benchmark", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("auto", "full", "subset"),
        default="auto",
        help="Training size for --run; auto follows the frozen runtime gate.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.freeze:
        freeze_protocol()
    elif args.smoke:
        run_runtime_trial(
            train_rows=SMOKE_TRAIN_ROWS,
            validation_rows=SMOKE_VALIDATION_ROWS,
            output_file=SMOKE_FILE,
            output_version="D9_SMOKE_V1",
        )
    elif args.benchmark:
        run_runtime_trial(
            train_rows=BENCHMARK_TRAIN_ROWS,
            validation_rows=BENCHMARK_VALIDATION_ROWS,
            output_file=BENCHMARK_FILE,
            output_version="D9_BENCHMARK_V1",
        )
    else:
        run_all(args.mode)
