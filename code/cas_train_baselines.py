"""Freeze and fit CAS Dummy and weighted-Logistic validation baselines.

This script never evaluates random internal-test rows or 2025 rows.
"""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "8"

import argparse
import json
import time
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from cas_modeling_common import (
    ANALYSIS_PROTOCOL_FILE,
    ASSIGNMENTS_FILE,
    ASYMMETRIC_COST_MATRIX,
    PROJECT_DIR,
    SCHEMA_FILE,
    TARGET_CODES,
    THREADS,
    TRAINING_INPUTS_FILE,
    assert_encoded_matrix,
    build_artifact_manifest,
    class_weights_for_spec,
    classification_metrics,
    confusion_rows,
    fit_category_vocabulary,
    hash_file,
    load_aligned_data,
    load_contracts,
    make_linear_preprocessor,
    make_split_specs,
    predict_with_probabilities,
    prediction_frame,
    prepare_features,
    relative,
    write_csv,
    write_dataframe_gzip,
    write_json_atomic,
)


VERSION = "CAS_BASELINE_V1"
MODEL_NAMES = ("dummy_most_frequent", "logistic_weighted")
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 1_000
LOGISTIC_TOL = 1e-4
LOGISTIC_SEED = 20_260_901
SMOKE_SEED = 90_101

CONFIG_FILE = PROJECT_DIR / "config" / "cas" / "cas_baseline_protocol.json"
FROZEN_MODELS_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_baseline_models_frozen.json"
)
MODEL_DIR = PROJECT_DIR / "models" / "cas_baseline"
RESULT_DIR = PROJECT_DIR / "results" / "cas_baseline"
PREDICTION_DIR = RESULT_DIR / "validation_predictions"
METRICS_FILE = RESULT_DIR / "validation_metrics.csv"
CONFUSION_FILE = RESULT_DIR / "validation_confusion_matrices.csv"
LOG_DIR = PROJECT_DIR / "logs" / "cas"
PREPROCESS_AUDIT_FILE = LOG_DIR / "cas_baseline_preprocessing_audit.csv"
FEATURE_AUDIT_FILE = LOG_DIR / "cas_baseline_encoded_features.csv"
ARTIFACT_MANIFEST_FILE = LOG_DIR / "cas_baseline_artifact_manifest.csv"
CHECKPOINT_FILE = LOG_DIR / "cas_baseline_checkpoint.md"
SMOKE_FILE = LOG_DIR / "cas_baseline_smoke.json"


def now_local() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def protocol_payload(
    schema: dict[str, object],
    analysis: dict[str, object],
    training: dict[str, object],
) -> dict[str, object]:
    return {
        "version": VERSION,
        "status": "FROZEN_BEFORE_CAS_VALIDATION_PERFORMANCE_INSPECTION",
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
        },
        "evaluation_scope": {
            "allowed": "training and validation roles only",
            "forbidden": (
                "2025, random internal tests, and locked_temporal_test "
                "model-performance inspection"
            ),
            "temporal_training_years": analysis["temporal_protocol"]["train_years"],
            "temporal_validation_years": analysis["temporal_protocol"][
                "validation_years"
            ],
            "random_seeds": analysis["random_reference_protocol"]["seeds"],
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
            "categorical": (
                "training-only vocabulary, explicit __UNSEEN__, one-hot "
                "encoding without dropping a level"
            ),
            "numeric": (
                "training-only median imputation plus missingness indicators, "
                "then training-fitted standardization"
            ),
            "structural_missingness": (
                "sparse speed fields retain explicit missingness indicators"
            ),
            "rare_pooling": "disabled",
        },
        "models": {
            "dummy_most_frequent": {
                "strategy": "most_frequent",
                "class_weight": None,
            },
            "logistic_weighted": {
                "solver": "lbfgs",
                "penalty": "l2",
                "C": LOGISTIC_C,
                "max_iter": LOGISTIC_MAX_ITER,
                "tol": LOGISTIC_TOL,
                "class_weight": "split-specific frozen balanced weights",
            },
        },
        "execution": {
            "threads": THREADS,
            "scientific_interpretation": (
                "thread count is a compute setting, not a selected model "
                "hyperparameter"
            ),
            "random_state": LOGISTIC_SEED,
        },
        "prediction_rule": "argmax over class probabilities; no threshold tuning",
        "validation_reporting": analysis["frozen_metrics"]["required_reporting"],
        "asymmetric_cost_matrix": {
            "class_order": ["Minor", "Serious", "Fatal"],
            "rows_true_columns_predicted": ASYMMETRIC_COST_MATRIX.tolist(),
            "purpose": "descriptive only; never a tuning target",
        },
        "class_weight_rule": training["class_weight_rule"],
        "test_boundary": training["test_boundary"],
    }


def freeze_protocol() -> None:
    schema, analysis, training = load_contracts()
    if METRICS_FILE.exists() or FROZEN_MODELS_FILE.exists():
        raise RuntimeError("Baseline results exist; refusing to refreeze protocol")
    write_json_atomic(CONFIG_FILE, protocol_payload(schema, analysis, training))
    print(f"CAS_BASELINE_PROTOCOL_SHA256={hash_file(CONFIG_FILE)}")
    print("CAS_BASELINE_PROTOCOL_STATUS=FROZEN_BEFORE_VALIDATION_INSPECTION")


def verify_protocol() -> dict[str, object]:
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    if config.get("version") != VERSION:
        raise ValueError("Unexpected CAS baseline protocol version")
    if (
        config.get("status")
        != "FROZEN_BEFORE_CAS_VALIDATION_PERFORMANCE_INSPECTION"
    ):
        raise ValueError("CAS baseline protocol is not frozen")
    checks = {
        SCHEMA_FILE: config["upstream"]["modeling_schema_sha256"],
        ANALYSIS_PROTOCOL_FILE: config["upstream"]["analysis_protocol_sha256"],
        TRAINING_INPUTS_FILE: config["upstream"]["training_inputs_sha256"],
        ASSIGNMENTS_FILE: config["upstream"]["split_assignments_sha256"],
    }
    for path, expected in checks.items():
        if hash_file(path) != expected:
            raise ValueError(f"Frozen upstream changed: {path.name}")
    if int(config["execution"]["threads"]) != THREADS:
        raise ValueError("CAS baseline thread setting differs from protocol")
    return config


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


def make_logistic(class_weight: dict[int, float]) -> LogisticRegression:
    return LogisticRegression(
        penalty="l2",
        C=LOGISTIC_C,
        class_weight=class_weight,
        solver="lbfgs",
        max_iter=LOGISTIC_MAX_ITER,
        tol=LOGISTIC_TOL,
        fit_intercept=True,
        random_state=LOGISTIC_SEED,
    )


def fit_one_split(
    *,
    spec: dict[str, str],
    schema: dict[str, object],
    training: dict[str, object],
    features: pd.DataFrame,
    target: pd.Series,
    metadata: pd.DataFrame,
    assignments: pd.DataFrame,
    smoke: bool,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    role_column = spec["role_column"]
    train_positions = np.flatnonzero(
        assignments[role_column].eq("train").to_numpy()
    )
    validation_positions = np.flatnonzero(
        assignments[role_column].eq("validation").to_numpy()
    )
    if smoke:
        train_positions = stratified_subset(
            train_positions, target, 5_000, SMOKE_SEED
        )
        validation_positions = stratified_subset(
            validation_positions, target, 2_000, SMOKE_SEED + 1
        )

    train_years = set(
        assignments.iloc[train_positions]["meta_crash_year"].astype(int)
    )
    validation_years = set(
        assignments.iloc[validation_positions]["meta_crash_year"].astype(int)
    )
    if 2025 in train_years or 2025 in validation_years:
        raise AssertionError("CAS baseline attempted to access 2025")
    if spec["protocol"] == "temporal":
        if train_years != {2022, 2023} or validation_years != {2024}:
            raise AssertionError("CAS temporal years differ from frozen protocol")

    X_train = features.iloc[train_positions]
    X_validation = features.iloc[validation_positions]
    y_train = target.iloc[train_positions].to_numpy(dtype=np.int8)
    y_validation = target.iloc[validation_positions].to_numpy(dtype=np.int8)
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
        raise AssertionError("CAS training data produced unseen categories")

    preprocessor = make_linear_preprocessor(
        categorical_columns=categorical,
        numeric_columns=numeric,
        category_vocabulary=vocabulary,
    )
    start = time.perf_counter()
    encoded_train = preprocessor.fit_transform(prepared_train.frame)
    encoded_validation = preprocessor.transform(prepared_validation.frame)
    preprocessing_seconds = time.perf_counter() - start
    assert_encoded_matrix(encoded_train, len(train_positions))
    assert_encoded_matrix(encoded_validation, len(validation_positions))
    encoded_names = preprocessor.get_feature_names_out().tolist()

    class_weights = class_weights_for_spec(training, spec)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(y_train), 1), dtype=np.int8), y_train)

    logistic = make_logistic(class_weights)
    fit_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        with threadpool_limits(limits=THREADS):
            logistic.fit(encoded_train, y_train)
    logistic_fit_seconds = time.perf_counter() - fit_start
    convergence_warning = any(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )

    models: list[tuple[str, object, object, float, int, bool]] = [
        (
            "dummy_most_frequent",
            dummy,
            np.zeros((len(y_validation), 1), dtype=np.int8),
            0.0,
            0,
            False,
        ),
        (
            "logistic_weighted",
            logistic,
            encoded_validation,
            logistic_fit_seconds,
            int(np.max(logistic.n_iter_)),
            convergence_warning,
        ),
    ]
    metric_rows: list[dict[str, object]] = []
    confusion_output: list[dict[str, object]] = []
    validation_ids = metadata.iloc[validation_positions]["meta_crash_id"]
    validation_year_series = assignments.iloc[validation_positions][
        "meta_crash_year"
    ]
    for model_name, estimator, model_input, fit_seconds, n_iter, warning in models:
        prediction_start = time.perf_counter()
        predicted, probabilities = predict_with_probabilities(estimator, model_input)
        prediction_seconds = time.perf_counter() - prediction_start
        metrics = classification_metrics(y_validation, predicted)
        metric_rows.append(
            {
                "protocol": spec["protocol"],
                "seed": spec["seed"],
                "evaluation_role": "validation",
                "training_rows": len(train_positions),
                "validation_rows": len(validation_positions),
                "training_years": ";".join(map(str, sorted(train_years))),
                "validation_years": ";".join(map(str, sorted(validation_years))),
                "model": model_name,
                "class_weighted": model_name == "logistic_weighted",
                "encoded_features": (
                    len(encoded_names) if model_name == "logistic_weighted" else 0
                ),
                "preprocessing_seconds": (
                    preprocessing_seconds
                    if model_name == "logistic_weighted"
                    else 0.0
                ),
                "fit_seconds": fit_seconds,
                "prediction_seconds": prediction_seconds,
                "n_iter": n_iter,
                "convergence_warning": warning,
                **metrics,
            }
        )
        confusion_output.extend(
            confusion_rows(
                y_validation,
                predicted,
                protocol=spec["protocol"],
                seed=spec["seed"],
                evaluation_role="validation",
                model=model_name,
            )
        )
        if not smoke:
            write_dataframe_gzip(
                PREDICTION_DIR / f"{spec['split_slug']}__{model_name}.csv.gz",
                prediction_frame(
                    crash_ids=validation_ids,
                    years=validation_year_series,
                    y_true=y_validation,
                    y_pred=predicted,
                    probabilities=probabilities,
                ),
            )

    numeric_transformer = preprocessor.named_transformers_["numeric"]
    imputer = numeric_transformer.named_steps["imputer"]
    indicator_features = {
        numeric[int(index)] for index in imputer.indicator_.features_
    }
    audit_rows: list[dict[str, object]] = []
    for column, median in zip(numeric, imputer.statistics_, strict=True):
        audit_rows.append(
            {
                "protocol": spec["protocol"],
                "seed": spec["seed"],
                "feature": column,
                "feature_type": "numeric",
                "training_levels": "",
                "validation_unseen_count": "",
                "training_missing_count": int(X_train[column].isna().sum()),
                "validation_missing_count": int(X_validation[column].isna().sum()),
                "training_fitted_imputation_value": float(median),
                "missing_indicator_fitted": column in indicator_features,
            }
        )
    for column in categorical:
        audit_rows.append(
            {
                "protocol": spec["protocol"],
                "seed": spec["seed"],
                "feature": column,
                "feature_type": "categorical",
                "training_levels": len(vocabulary[column]),
                "validation_unseen_count": prepared_validation.unseen_counts[column],
                "training_missing_count": 0,
                "validation_missing_count": 0,
                "training_fitted_imputation_value": "",
                "missing_indicator_fitted": "",
            }
        )
    feature_rows = [
        {
            "protocol": spec["protocol"],
            "seed": spec["seed"],
            "encoded_position": position,
            "encoded_feature": name,
        }
        for position, name in enumerate(encoded_names)
    ]

    if not smoke:
        split_dir = MODEL_DIR / spec["split_slug"]
        split_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "version": VERSION,
                "protocol": spec["protocol"],
                "seed": spec["seed"],
                "role_column": role_column,
                "feature_columns": feature_columns,
                "categorical_columns": categorical,
                "numeric_columns": numeric,
                "category_vocabulary": vocabulary,
                "preprocessor": preprocessor,
                "encoded_feature_names": encoded_names,
                "numeric_missing_indicator_features": sorted(indicator_features),
                "training_rows": len(train_positions),
                "training_years": sorted(train_years),
                "validation_rows": len(validation_positions),
                "validation_years": sorted(validation_years),
                "protocol_sha256": hash_file(CONFIG_FILE),
            },
            split_dir / "preprocessing_bundle.joblib",
            compress=3,
        )
        for model_name, estimator, _input, _seconds, _n_iter, _warning in models:
            joblib.dump(
                {
                    "version": VERSION,
                    "protocol": spec["protocol"],
                    "seed": spec["seed"],
                    "role_column": role_column,
                    "model_name": model_name,
                    "estimator": estimator,
                    "class_weight": (
                        class_weights if model_name == "logistic_weighted" else None
                    ),
                    "prediction_rule": "argmax over class probabilities",
                    "target_codes": list(TARGET_CODES),
                    "protocol_sha256": hash_file(CONFIG_FILE),
                },
                split_dir / f"{model_name}.joblib",
                compress=3,
            )

    print(
        f"{spec['split_slug']}: train={len(train_positions):,}, "
        f"validation={len(validation_positions):,}, encoded={len(encoded_names)}, "
        f"logistic_iter={int(np.max(logistic.n_iter_))}",
        flush=True,
    )
    return metric_rows, confusion_output, audit_rows, feature_rows


def write_checkpoint(metrics: pd.DataFrame) -> None:
    temporal = metrics.loc[metrics["protocol"].eq("temporal")]
    random = metrics.loc[metrics["protocol"].eq("random_reference")]
    lines = [
        "# CAS baseline validation checkpoint",
        "",
        "Status: **PASS - BASELINES_FROZEN_BEFORE_2025_EVALUATION**",
        "",
        "## Access control",
        "",
        "- Preprocessing was fitted separately within each training partition.",
        "- No random internal-test or 2025 model performance was calculated.",
        "- The 2025 cohort remains locked for one-time evaluation.",
        "",
        "## Temporal validation (2024)",
        "",
    ]
    for _, row in temporal.sort_values("model").iterrows():
        lines.append(
            f"- {row['model']}: Macro-F1 **{row['macro_f1']:.4f}**, "
            f"QWK **{row['qwk']:.4f}**, Fatal recall "
            f"**{row['fatal_recall']:.4f}**."
        )
    lines.extend(["", "## Random-reference validation", ""])
    for model_name in MODEL_NAMES:
        subset = random.loc[random["model"].eq(model_name)]
        lines.append(
            f"- {model_name}: Macro-F1 mean **{subset['macro_f1'].mean():.4f}** "
            f"(SD **{subset['macro_f1'].std(ddof=1):.4f}**)."
        )
    lines.extend(
        [
            "",
            "## Execution",
            "",
            f"- Thread limit: **{THREADS}**.",
            f"- Logistic convergence warnings: "
            f"**{int(metrics['convergence_warning'].astype(bool).sum())}**.",
            "- Weighted Logistic is the frozen linear comparator for LightGBM.",
            "",
        ]
    )
    CHECKPOINT_FILE.write_text("\n".join(lines), encoding="utf-8")


def freeze_models(analysis: dict[str, object]) -> None:
    model_paths = sorted(MODEL_DIR.rglob("*.joblib"))
    prediction_paths = sorted(PREDICTION_DIR.glob("*.csv.gz"))
    split_count = len(make_split_specs(analysis))
    if len(model_paths) != split_count * (len(MODEL_NAMES) + 1):
        raise AssertionError("Unexpected CAS baseline model artifact count")
    if len(prediction_paths) != split_count * len(MODEL_NAMES):
        raise AssertionError("Unexpected CAS baseline validation prediction count")
    artifacts = [
        CONFIG_FILE,
        METRICS_FILE,
        CONFUSION_FILE,
        PREPROCESS_AUDIT_FILE,
        FEATURE_AUDIT_FILE,
        CHECKPOINT_FILE,
        *model_paths,
        *prediction_paths,
    ]
    build_artifact_manifest(artifacts, ARTIFACT_MANIFEST_FILE)
    write_json_atomic(
        FROZEN_MODELS_FILE,
        {
            "version": VERSION,
            "status": "BASELINE_MODELS_FROZEN_BEFORE_2025_EVALUATION",
            "created_local": now_local(),
            "protocol": relative(CONFIG_FILE),
            "protocol_sha256": hash_file(CONFIG_FILE),
            "artifact_manifest": relative(ARTIFACT_MANIFEST_FILE),
            "artifact_manifest_sha256": hash_file(ARTIFACT_MANIFEST_FILE),
            "models": [
                {"file": relative(path), "sha256": hash_file(path)}
                for path in model_paths
            ],
            "validation_predictions": [
                {"file": relative(path), "sha256": hash_file(path)}
                for path in prediction_paths
            ],
            "test_performance_used": False,
            "threads": THREADS,
        },
    )


def run(smoke: bool) -> None:
    verify_protocol()
    schema, analysis, training = load_contracts()
    features, target, metadata, assignments = load_aligned_data(schema, analysis)
    specs = make_split_specs(analysis)
    if smoke:
        specs = specs[:1]

    all_metrics: list[dict[str, object]] = []
    all_confusions: list[dict[str, object]] = []
    all_audits: list[dict[str, object]] = []
    all_feature_rows: list[dict[str, object]] = []
    for spec in specs:
        metrics, confusions, audits, feature_rows = fit_one_split(
            spec=spec,
            schema=schema,
            training=training,
            features=features,
            target=target,
            metadata=metadata,
            assignments=assignments,
            smoke=smoke,
        )
        all_metrics.extend(metrics)
        all_confusions.extend(confusions)
        all_audits.extend(audits)
        all_feature_rows.extend(feature_rows)

    if smoke:
        write_json_atomic(
            SMOKE_FILE,
            {
                "version": VERSION,
                "status": "PASS",
                "models": list(MODEL_NAMES),
                "threads": THREADS,
                "test_performance_used": False,
            },
        )
        print("CAS_BASELINE_SMOKE=PASS")
        return

    write_csv(METRICS_FILE, all_metrics)
    write_csv(CONFUSION_FILE, all_confusions)
    write_csv(PREPROCESS_AUDIT_FILE, all_audits)
    write_csv(FEATURE_AUDIT_FILE, all_feature_rows)
    write_checkpoint(pd.DataFrame(all_metrics))
    freeze_models(analysis)
    print(f"CAS_BASELINE_METRIC_ROWS={len(all_metrics)}")
    print(
        "CAS_BASELINE_VALIDATION_PREDICTIONS="
        f"{len(list(PREDICTION_DIR.glob('*.csv.gz')))}"
    )
    print("CAS_BASELINE_STATUS=FROZEN_BEFORE_2025_EVALUATION")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--fit-validation", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.freeze:
        freeze_protocol()
    else:
        run(smoke=arguments.smoke)
