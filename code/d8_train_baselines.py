"""Fit D8 Dummy and multinomial-logistic validation baselines.

The script fits preprocessing on each training partition only. It evaluates
only validation partitions from 2018-2023 and refuses to evaluate 2024. The
locked temporal and random test partitions remain untouched until D11.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from baseline_modeling import (
    ASYMMETRIC_COST_MATRIX,
    TARGET_CODES,
    classification_metrics,
    confusion_rows,
    fit_category_vocabulary,
    make_preprocessor,
    predict_with_probabilities,
    prepare_features,
    assert_encoded_matrix,
)
from modeling_data import load_modeling_data


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCHEMA_FILE = PROJECT_DIR / "config" / "d5_dataset_schema.json"
D6_PROTOCOL_FILE = PROJECT_DIR / "config" / "d6_analysis_protocol.json"
D7_CONFIG_FILE = PROJECT_DIR / "config" / "d7_training_inputs.json"
ASSIGNMENTS_FILE = PROJECT_DIR / "data" / "processed" / "d6_split_assignments.csv.gz"

MODEL_DIR = PROJECT_DIR / "models" / "d8"
RESULT_DIR = PROJECT_DIR / "results" / "d8"
PREDICTION_DIR = RESULT_DIR / "validation_predictions"
LOG_DIR = PROJECT_DIR / "logs"

D8_CONFIG_FILE = PROJECT_DIR / "config" / "d8_baseline_protocol.json"
METRICS_FILE = RESULT_DIR / "d8_validation_metrics.csv"
CONFUSION_FILE = RESULT_DIR / "d8_validation_confusion_matrices.csv"
PREPROCESS_AUDIT_FILE = LOG_DIR / "d8_preprocessing_audit.csv"
FEATURE_AUDIT_FILE = LOG_DIR / "d8_encoded_features.csv"
CHECKPOINT_FILE = LOG_DIR / "d8_checkpoint.md"
SMOKE_FILE = LOG_DIR / "d8_smoke_test.json"

LOGISTIC_SEED = 20_260_828
SMOKE_SEED = 80_801
SMOKE_TRAIN_ROWS = 20_000
SMOKE_VALIDATION_ROWS = 5_000
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 500
LOGISTIC_TOL = 1e-4
MODEL_NAMES = ("dummy_most_frequent", "logistic_unweighted", "logistic_weighted")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows generated for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_contracts() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    d6 = json.loads(D6_PROTOCOL_FILE.read_text(encoding="utf-8"))
    d7 = json.loads(D7_CONFIG_FILE.read_text(encoding="utf-8"))
    if schema.get("status") != "QC_PASSED_AND_FROZEN":
        raise ValueError("D8 requires the frozen D5 schema")
    if d6.get("version") != "D6_V2" or d6.get("status") != "FROZEN_BEFORE_MODEL_TRAINING":
        raise ValueError("D8 requires the frozen D6_V2 protocol")
    if d7.get("status") != "TRAINING_INPUTS_AUDITED_AND_FROZEN":
        raise ValueError("D8 requires the completed D7 training-input audit")
    if hash_file(ASSIGNMENTS_FILE) != d7["upstream"]["D6_assignments_sha256"]:
        raise ValueError("Frozen D6 assignments changed after D7")
    if hash_file(SCHEMA_FILE) != d7["upstream"]["D5_schema_sha256"]:
        raise ValueError("Frozen D5 schema changed after D7")
    return schema, d6, d7


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
    model_ids = metadata["meta_collision_index"].astype("string").reset_index(drop=True)
    assignment_ids = assignments["meta_collision_index"].reset_index(drop=True)
    if not model_ids.equals(assignment_ids):
        raise AssertionError("D5 and D6 collision identifiers are not row-aligned")
    if features.columns.tolist() != schema["feature_columns"]:
        raise AssertionError("D8 feature allowlist differs from the frozen schema")

    # Categories reduce memory without changing values. Training vocabularies are
    # still fitted from observed training rows, never from global category levels.
    for column in schema["categorical_feature_columns"]:
        features[column] = features[column].astype("category")
    for column in role_columns:
        assignments[column] = assignments[column].astype("category")
    return features, target.astype("int8"), metadata, assignments


def stratified_subset(positions: np.ndarray, target: pd.Series, rows: int, seed: int) -> np.ndarray:
    if len(positions) <= rows:
        return positions
    selected, _ = train_test_split(
        positions,
        train_size=rows,
        random_state=seed,
        stratify=target.iloc[positions].to_numpy(),
    )
    return np.sort(selected)


def make_split_specs(d6: dict[str, object]) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = [
        {
            "protocol": "temporal",
            "seed": "year_based",
            "role_column": "temporal_role",
            "weight_key": ("temporal", None),
        }
    ]
    for seed in d6["random_reference_protocol"]["seeds"]:
        specs.append(
            {
                "protocol": "random_reference",
                "seed": str(seed),
                "role_column": f"random_role_seed_{seed}",
                "weight_key": ("random_reference", str(seed)),
            }
        )
    return specs


def class_weights_for_spec(d7: dict[str, object], spec: dict[str, object]) -> dict[int, float]:
    group, seed = spec["weight_key"]
    payload = d7["weight_sets"][group] if seed is None else d7["weight_sets"][group][seed]
    return {int(code): float(weight) for code, weight in payload["class_weights"].items()}


def make_logistic(class_weight: dict[int, float] | None) -> LogisticRegression:
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


def save_prediction(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        lineterminator="\n",
    )


def fit_one_split(
    *,
    spec: dict[str, object],
    schema: dict[str, object],
    d7: dict[str, object],
    features: pd.DataFrame,
    target: pd.Series,
    metadata: pd.DataFrame,
    assignments: pd.DataFrame,
    smoke: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    role_column = str(spec["role_column"])
    train_positions = np.flatnonzero(assignments[role_column].eq("train").to_numpy())
    validation_positions = np.flatnonzero(assignments[role_column].eq("validation").to_numpy())
    if smoke:
        train_positions = stratified_subset(
            train_positions, target, SMOKE_TRAIN_ROWS, SMOKE_SEED
        )
        validation_positions = stratified_subset(
            validation_positions, target, SMOKE_VALIDATION_ROWS, SMOKE_SEED + 1
        )

    train_years = set(assignments.iloc[train_positions]["meta_collision_year"].astype(int))
    validation_years = set(assignments.iloc[validation_positions]["meta_collision_year"].astype(int))
    if 2024 in train_years or 2024 in validation_years:
        raise AssertionError("D8 attempted to access 2024 as training or validation")
    if str(spec["protocol"]) == "temporal":
        if train_years != {2018, 2019, 2020, 2021, 2022}:
            raise AssertionError(f"Unexpected temporal training years: {train_years}")
        if validation_years != {2023}:
            raise AssertionError(f"Unexpected temporal validation years: {validation_years}")

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
        raise AssertionError("Training data produced unseen categories")

    preprocessor = make_preprocessor(
        categorical_columns=categorical,
        numeric_columns=numeric,
        category_vocabulary=vocabulary,
    )
    preprocessing_start = time.perf_counter()
    encoded_train = preprocessor.fit_transform(prepared_train.frame)
    encoded_validation = preprocessor.transform(prepared_validation.frame)
    preprocessing_seconds = time.perf_counter() - preprocessing_start
    assert_encoded_matrix(encoded_train, len(train_positions))
    assert_encoded_matrix(encoded_validation, len(validation_positions))

    feature_names = preprocessor.get_feature_names_out().tolist()
    split_slug = (
        "temporal" if spec["protocol"] == "temporal" else f"random_seed_{spec['seed']}"
    )
    split_model_dir = MODEL_DIR / split_slug
    split_model_dir.mkdir(parents=True, exist_ok=True)
    if not smoke:
        joblib.dump(
            {
                "version": "D8_V1",
                "protocol": spec["protocol"],
                "seed": spec["seed"],
                "role_column": role_column,
                "feature_columns": feature_columns,
                "categorical_columns": categorical,
                "numeric_columns": numeric,
                "category_vocabulary": vocabulary,
                "preprocessor": preprocessor,
                "encoded_feature_names": feature_names,
                "training_rows": len(train_positions),
                "training_years": sorted(train_years),
            },
            split_model_dir / "preprocessing_bundle.joblib",
            compress=3,
        )

    audit_rows: list[dict[str, object]] = []
    imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
    imputation_values = np.asarray(imputer.statistics_, dtype=float)
    for column, median in zip(numeric, imputation_values, strict=True):
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
                "training_fitted_imputation_value": median,
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
                "training_missing_count": int(X_train[column].isna().sum()),
                "validation_missing_count": int(X_validation[column].isna().sum()),
                "training_fitted_imputation_value": "",
            }
        )

    feature_rows = [
        {
            "protocol": spec["protocol"],
            "seed": spec["seed"],
            "encoded_position": position,
            "encoded_feature": name,
        }
        for position, name in enumerate(feature_names)
    ]
    model_estimators: list[tuple[str, object, object]] = []
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(y_train), 1), dtype=np.int8), y_train)
    model_estimators.append(
        (
            "dummy_most_frequent",
            dummy,
            np.zeros((len(y_validation), 1), dtype=np.int8),
        )
    )

    class_weights = class_weights_for_spec(d7, spec)
    convergence_records: dict[str, tuple[int, bool, float]] = {}
    for model_name, weight in (
        ("logistic_unweighted", None),
        ("logistic_weighted", class_weights),
    ):
        estimator = make_logistic(weight)
        fit_start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(encoded_train, y_train)
        fit_seconds = time.perf_counter() - fit_start
        convergence_warning = any(
            issubclass(item.category, ConvergenceWarning) for item in caught
        )
        n_iter = int(np.max(estimator.n_iter_))
        convergence_records[model_name] = (n_iter, convergence_warning, fit_seconds)
        model_estimators.append((model_name, estimator, encoded_validation))

    metric_rows: list[dict[str, object]] = []
    confusion_output: list[dict[str, object]] = []
    validation_ids = metadata.iloc[validation_positions]["meta_collision_index"]
    validation_year_series = assignments.iloc[validation_positions]["meta_collision_year"]
    for model_name, estimator, validation_input in model_estimators:
        prediction_start = time.perf_counter()
        predicted, probabilities = predict_with_probabilities(estimator, validation_input)
        prediction_seconds = time.perf_counter() - prediction_start
        metrics = classification_metrics(y_validation, predicted)
        if model_name in convergence_records:
            n_iter, convergence_warning, fit_seconds = convergence_records[model_name]
        else:
            n_iter, convergence_warning, fit_seconds = 0, False, 0.0
        metric_rows.append(
            {
                "protocol": spec["protocol"],
                "seed": spec["seed"],
                "evaluation_role": "validation",
                "training_rows": len(train_positions),
                "validation_rows": len(validation_positions),
                "training_years": ";".join(str(year) for year in sorted(train_years)),
                "validation_years": ";".join(str(year) for year in sorted(validation_years)),
                "model": model_name,
                "class_weighted": model_name == "logistic_weighted",
                "encoded_features": len(feature_names) if model_name != "dummy_most_frequent" else 0,
                "preprocessing_seconds": preprocessing_seconds if model_name != "dummy_most_frequent" else 0.0,
                "fit_seconds": fit_seconds,
                "prediction_seconds": prediction_seconds,
                "n_iter": n_iter,
                "convergence_warning": convergence_warning,
                **metrics,
            }
        )
        confusion_output.extend(
            confusion_rows(
                y_validation,
                predicted,
                protocol=str(spec["protocol"]),
                seed=str(spec["seed"]),
                model=model_name,
            )
        )
        if not smoke:
            prediction_path = PREDICTION_DIR / f"{split_slug}__{model_name}.csv.gz"
            save_prediction(
                prediction_path,
                prediction_frame(
                    collision_ids=validation_ids,
                    years=validation_year_series,
                    y_true=y_validation,
                    y_pred=predicted,
                    probabilities=probabilities,
                ),
            )
            estimator_path = split_model_dir / f"{model_name}.joblib"
            joblib.dump(
                {
                    "version": "D8_V1",
                    "protocol": spec["protocol"],
                    "seed": spec["seed"],
                    "model_name": model_name,
                    "estimator": estimator,
                    "class_weight": class_weights if model_name == "logistic_weighted" else None,
                    "prediction_rule": "argmax over class probabilities",
                    "target_codes": list(TARGET_CODES),
                },
                estimator_path,
                compress=3,
            )

    print(
        f"Completed {split_slug}: train={len(train_positions):,}, "
        f"validation={len(validation_positions):,}, encoded={len(feature_names)}"
    )
    return metric_rows, confusion_output, audit_rows, feature_rows


def make_protocol_config(
    schema: dict[str, object], d6: dict[str, object], d7: dict[str, object]
) -> dict[str, object]:
    return {
        "version": "D8_V1",
        "status": "BASELINE_SPECIFICATION_FROZEN_BEFORE_VALIDATION_INSPECTION",
        "created_local": "2026-08-28",
        "upstream": {
            "D5_schema": SCHEMA_FILE.relative_to(PROJECT_DIR).as_posix(),
            "D5_schema_sha256": hash_file(SCHEMA_FILE),
            "D6_protocol": D6_PROTOCOL_FILE.relative_to(PROJECT_DIR).as_posix(),
            "D6_protocol_sha256": hash_file(D6_PROTOCOL_FILE),
            "D7_training_inputs": D7_CONFIG_FILE.relative_to(PROJECT_DIR).as_posix(),
            "D7_training_inputs_sha256": hash_file(D7_CONFIG_FILE),
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
        "preprocessing": {
            "fit_scope": "each training partition only",
            "categorical": "training vocabulary, explicit __UNSEEN__, one-hot without dropping a level",
            "numeric": "training median imputation followed by training-fitted standardization",
            "rare_pooling": "disabled",
        },
        "models": {
            "dummy_most_frequent": {
                "role": "minimum-performance floor",
                "strategy": "most_frequent",
                "class_weight": None,
            },
            "logistic_unweighted": {
                "role": "natural-prevalence linear baseline",
                "solver": "lbfgs",
                "penalty": "l2",
                "C": LOGISTIC_C,
                "max_iter": LOGISTIC_MAX_ITER,
                "tol": LOGISTIC_TOL,
                "class_weight": None,
            },
            "logistic_weighted": {
                "role": "primary linear comparator for class-weighted LightGBM",
                "solver": "lbfgs",
                "penalty": "l2",
                "C": LOGISTIC_C,
                "max_iter": LOGISTIC_MAX_ITER,
                "tol": LOGISTIC_TOL,
                "class_weight": "D7 split-specific balanced weights",
            },
        },
        "prediction_rule": "argmax over class probabilities; no threshold tuning",
        "validation_reporting": [
            "Macro-F1",
            "quadratic weighted kappa",
            "ordinal MAE",
            "accuracy",
            "class recalls",
            "serious-or-fatal recall",
            "mean asymmetric cost",
            "confusion matrix",
        ],
        "asymmetric_cost_matrix": {
            "class_order": ["Slight", "Serious", "Fatal"],
            "rows_true_columns_predicted": ASYMMETRIC_COST_MATRIX.tolist(),
            "purpose": "descriptive test metric only; never a tuning target",
            "rationale": "underestimating severe outcomes receives a larger penalty than overestimation",
            "freeze_timing": "fixed in code and config before D8 validation performance inspection",
        },
        "primary_future_comparison": "logistic_weighted versus class-weighted LightGBM",
        "test_embargo": d7["test_embargo"],
    }


def write_checkpoint(metrics: pd.DataFrame, config: dict[str, object]) -> None:
    temporal = metrics.loc[metrics["protocol"].eq("temporal")].copy()
    random_metrics = metrics.loc[metrics["protocol"].eq("random_reference")].copy()
    lines = [
        "# D8 baseline-model checkpoint",
        "",
        "## Status",
        "",
        "**PASS - Dummy and multinomial-logistic baselines completed on validation data only.**",
        "",
        "## Leakage controls",
        "",
        "- Preprocessing was fitted separately on each training partition.",
        "- Validation data did not fit category vocabularies, imputation values, scaling or model coefficients.",
        "- No temporal/random test metric was calculated.",
        "- No 2024 record appears in a D8 prediction file.",
        "",
        "## Temporal validation (2023)",
        "",
    ]
    for _, row in temporal.sort_values("model").iterrows():
        lines.append(
            f"- {row['model']}: Macro-F1 **{row['macro_f1']:.4f}**, "
            f"QWK **{row['qwk']:.4f}**, Fatal recall **{row['fatal_recall']:.4f}**."
        )
    lines.extend(
        [
            "",
            "## Random-reference validation",
            "",
        ]
    )
    for model_name in MODEL_NAMES:
        subset = random_metrics.loc[random_metrics["model"].eq(model_name)]
        lines.append(
            f"- {model_name}: Macro-F1 mean **{subset['macro_f1'].mean():.4f}** "
            f"(SD **{subset['macro_f1'].std(ddof=1):.4f}**), QWK mean "
            f"**{subset['qwk'].mean():.4f}**."
        )
    warning_count = int(metrics["convergence_warning"].astype(bool).sum())
    lines.extend(
        [
            "",
            "## Convergence and handoff",
            "",
            f"- Logistic convergence warnings: **{warning_count}**.",
            "- Weighted Logistic is the fixed linear comparator for weighted LightGBM.",
            "- These are validation results and are not final generalization estimates.",
            "- D9 may start only when explicitly requested; 2024 remains locked until D11.",
            "",
        ]
    )
    CHECKPOINT_FILE.write_text("\n".join(lines), encoding="utf-8")


def run(smoke: bool) -> None:
    schema, d6, d7 = load_contracts()
    protocol_config = make_protocol_config(schema, d6, d7)
    if not smoke:
        write_json(D8_CONFIG_FILE, protocol_config)

    features, target, metadata, assignments = load_aligned_data(schema, d6)
    specs = make_split_specs(d6)
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
            d7=d7,
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

    metrics_frame = pd.DataFrame(all_metrics)
    if smoke:
        payload = {
            "version": "D8_SMOKE_V1",
            "status": "PASS",
            "training_rows": SMOKE_TRAIN_ROWS,
            "validation_rows": SMOKE_VALIDATION_ROWS,
            "models": MODEL_NAMES,
            "encoded_features": int(metrics_frame["encoded_features"].max()),
            "no_2024": True,
            "convergence_warnings": int(metrics_frame["convergence_warning"].astype(bool).sum()),
        }
        write_json(SMOKE_FILE, payload)
        print("D8_SMOKE_ASSERTIONS=PASS")
        return

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(METRICS_FILE, all_metrics)
    write_csv(CONFUSION_FILE, all_confusions)
    write_csv(PREPROCESS_AUDIT_FILE, all_audits)
    write_csv(FEATURE_AUDIT_FILE, all_feature_rows)
    write_checkpoint(metrics_frame, protocol_config)
    print("D8 validation metric rows:", len(all_metrics))
    print("D8 prediction files:", len(list(PREDICTION_DIR.glob("*.csv.gz"))))
    print("FINAL_D8_ASSERTIONS=PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="Run a small temporal pipeline check")
    mode.add_argument("--full", action="store_true", help="Run all frozen validation splits")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(smoke=arguments.smoke)
