"""Run the post-hoc CAS feature-ablation sensitivity analysis.

This analysis was requested in the early CAS feasibility audit but was not
implemented before the primary 2025 results were inspected. It is therefore
strictly post hoc. The script refits only the frozen temporal design and never
changes the original 15-feature models, hyperparameters, or primary results.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import warnings
from datetime import datetime
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "8"

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from cas_bootstrap_uncertainty import (
    metrics_from_confusion,
    paired_confusion_bootstrap,
    percentile_interval,
    point_confusion,
    stratified_confusion_bootstrap,
)
from cas_modeling_common import (
    PROJECT_DIR,
    TARGET_CODES,
    build_artifact_manifest,
    class_weights_for_spec,
    classification_metrics,
    confusion_rows,
    fit_category_vocabulary,
    hash_file,
    load_aligned_data,
    load_contracts,
    make_linear_preprocessor,
    predict_with_probabilities,
    prediction_frame,
    prepare_features,
    relative,
    write_csv,
    write_dataframe_gzip,
    write_json_atomic,
)
from cas_train_baselines import make_logistic
from cas_train_lightgbm import make_estimator


VERSION = "CAS_FEATURE_ABLATION_POSTHOC_V1"
MODEL_SEED = 20_260_901
THREADS = 8
MODELS = ("logistic_weighted", "lightgbm_weighted")
COHORTS = ("validation_2024", "test_2025")

PROTOCOL_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_feature_ablation_protocol.json"
)
LIGHTGBM_PROTOCOL_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_lightgbm_protocol.json"
)
LIGHTGBM_SELECTED_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_lightgbm_selected.json"
)
MAIN_TEST_METRICS_FILE = (
    PROJECT_DIR / "results" / "cas_evaluation" / "test_metrics.csv"
)
MAIN_PREDICTION_DIR = (
    PROJECT_DIR / "results" / "cas_evaluation" / "predictions"
)

MODEL_DIR = PROJECT_DIR / "models" / "cas_feature_ablation"
RESULT_DIR = PROJECT_DIR / "results" / "cas_feature_ablation"
PREDICTION_DIR = RESULT_DIR / "predictions"
VALIDATION_METRICS_FILE = RESULT_DIR / "validation_metrics.csv"
TEST_METRICS_FILE = RESULT_DIR / "test_metrics.csv"
CONFUSION_FILE = RESULT_DIR / "confusion_matrices.csv"
INTERVAL_FILE = RESULT_DIR / "test_metric_intervals.csv"
DIFFERENCE_FILE = RESULT_DIR / "test_paired_differences.csv"
FEATURE_CONTRACT_FILE = RESULT_DIR / "feature_contract.csv"

LOG_DIR = PROJECT_DIR / "logs" / "cas"
SMOKE_FILE = LOG_DIR / "cas_feature_ablation_smoke.json"
PREPROCESS_AUDIT_FILE = LOG_DIR / "cas_feature_ablation_preprocessing.csv"
TRAINING_AUDIT_FILE = LOG_DIR / "cas_feature_ablation_training.csv"
CHECKPOINT_FILE = LOG_DIR / "cas_feature_ablation_checkpoint.md"
MANIFEST_FILE = LOG_DIR / "cas_feature_ablation_artifact_manifest.csv"
COMPLETE_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_feature_ablation_complete.json"
)


def now_local() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    if protocol.get("version") != VERSION:
        raise ValueError("Unexpected CAS feature-ablation protocol version")
    if protocol.get("status") != (
        "FROZEN_AFTER_PRIMARY_RESULTS_BEFORE_POSTHOC_ABLATION_FITS"
    ):
        raise ValueError("CAS feature-ablation protocol is not frozen")
    disclosure = protocol["post_hoc_disclosure"]
    if disclosure.get("primary_2025_results_known_before_freeze") is not True:
        raise ValueError("Protocol does not disclose prior 2025 result access")
    if disclosure.get("eligible_for_primary_model_selection") is not False:
        raise ValueError("Post-hoc analysis cannot be used for model selection")
    for item in protocol["upstream_files"]:
        path = PROJECT_DIR / item["file"]
        if hash_file(path) != item["sha256"]:
            raise ValueError(f"Frozen ablation upstream changed: {item['file']}")
    if int(protocol["execution"]["threads"]) != THREADS:
        raise ValueError("CAS feature-ablation thread setting changed")
    if int(protocol["bootstrap"]["iterations"]) != 2_000:
        raise ValueError("CAS feature-ablation bootstrap count changed")
    return protocol


def load_temporal_data() -> dict[str, Any]:
    schema, analysis, training = load_contracts()
    features, target, metadata, assignments = load_aligned_data(schema, analysis)
    roles = assignments["temporal_role"]
    positions = {
        "train": np.flatnonzero(roles.eq("train").to_numpy()),
        "validation_2024": np.flatnonzero(roles.eq("validation").to_numpy()),
        "test_2025": np.flatnonzero(roles.eq("test").to_numpy()),
    }
    expected = {"train": 21_934, "validation_2024": 10_645, "test_2025": 10_542}
    years = {"train": {2022, 2023}, "validation_2024": {2024}, "test_2025": {2025}}
    for name, cohort_positions in positions.items():
        if len(cohort_positions) != expected[name]:
            raise AssertionError(f"Unexpected {name} row count")
        observed_years = set(
            assignments.iloc[cohort_positions]["meta_crash_year"].astype(int)
        )
        if observed_years != years[name]:
            raise AssertionError(f"Unexpected {name} years: {observed_years}")
    all_positions = np.concatenate(list(positions.values()))
    if len(np.unique(all_positions)) != len(all_positions):
        raise AssertionError("Temporal train/validation/test positions overlap")
    if len(all_positions) != len(features):
        raise AssertionError("Temporal roles do not cover the CAS dataset")
    return {
        "schema": schema,
        "analysis": analysis,
        "training": training,
        "features": features,
        "target": target,
        "metadata": metadata,
        "assignments": assignments,
        "positions": positions,
    }


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


def scenario_columns(
    schema: dict[str, Any], scenario: dict[str, Any]
) -> tuple[list[str], list[str], list[str]]:
    full = list(schema["feature_columns"])
    dropped = list(scenario["drop_features"])
    if len(dropped) != len(set(dropped)) or not set(dropped).issubset(full):
        raise ValueError(f"Invalid drop list for {scenario['scenario_id']}")
    feature_columns = [column for column in full if column not in dropped]
    categorical = [
        column
        for column in schema["categorical_feature_columns"]
        if column in feature_columns
    ]
    numeric = [
        column
        for column in schema["numeric_feature_columns"]
        if column in feature_columns
    ]
    if len(feature_columns) != int(scenario["expected_feature_count"]):
        raise AssertionError("Scenario feature count differs from protocol")
    if feature_columns != categorical + numeric:
        raise AssertionError("Reduced feature order or type partition changed")
    return feature_columns, categorical, numeric


def prepare_scenario_frames(
    *,
    features: pd.DataFrame,
    positions: dict[str, np.ndarray],
    feature_columns: list[str],
    categorical: list[str],
    numeric: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]], dict[str, dict[str, int]]]:
    source = {
        name: features.iloc[cohort_positions].loc[:, feature_columns]
        for name, cohort_positions in positions.items()
    }
    vocabulary = fit_category_vocabulary(source["train"], categorical)
    prepared: dict[str, pd.DataFrame] = {}
    unseen: dict[str, dict[str, int]] = {}
    for name, frame in source.items():
        result = prepare_features(
            frame,
            feature_columns=feature_columns,
            categorical_columns=categorical,
            numeric_columns=numeric,
            category_vocabulary=vocabulary,
        )
        prepared[name] = result.frame
        unseen[name] = result.unseen_counts
    if any(unseen["train"].values()):
        raise AssertionError("Training data produced unseen categories")
    return prepared, vocabulary, unseen


def metric_row(
    *,
    scenario_id: str,
    feature_count: int,
    dropped_features: list[str],
    cohort: str,
    model: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, object]:
    return {
        "analysis_variant": "posthoc_ablation",
        "scenario": scenario_id,
        "feature_count": feature_count,
        "dropped_features": ";".join(dropped_features),
        "cohort": cohort,
        "cohort_rows": len(y_true),
        "model": model,
        **classification_metrics(y_true, y_pred),
    }


def extend_confusion(
    output: list[dict[str, object]],
    *,
    scenario: str,
    cohort: str,
    model: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    for row in confusion_rows(
        y_true,
        y_pred,
        protocol="posthoc_feature_ablation",
        seed="20260901",
        evaluation_role=cohort,
        model=model,
    ):
        output.append({"scenario": scenario, **row})


def persist_predictions(
    *,
    scenario_id: str,
    cohort: str,
    model: str,
    metadata: pd.DataFrame,
    assignments: pd.DataFrame,
    positions: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> Path:
    path = PREDICTION_DIR / f"{scenario_id}__{cohort}__{model}.csv.gz"
    write_dataframe_gzip(
        path,
        prediction_frame(
            crash_ids=metadata.iloc[positions]["meta_crash_id"],
            years=assignments.iloc[positions]["meta_crash_year"],
            y_true=y_true,
            y_pred=y_pred,
            probabilities=probabilities,
        ),
    )
    return path


def fit_scenario(
    *,
    protocol: dict[str, Any],
    data: dict[str, Any],
    scenario: dict[str, Any],
    positions: dict[str, np.ndarray],
    n_estimators: int,
    persist: bool,
) -> dict[str, Any]:
    schema = data["schema"]
    features = data["features"]
    target = data["target"]
    metadata = data["metadata"]
    assignments = data["assignments"]
    feature_columns, categorical, numeric = scenario_columns(schema, scenario)
    prepared, vocabulary, unseen = prepare_scenario_frames(
        features=features,
        positions=positions,
        feature_columns=feature_columns,
        categorical=categorical,
        numeric=numeric,
    )
    y = {
        name: target.iloc[cohort_positions].to_numpy(dtype=np.int8)
        for name, cohort_positions in positions.items()
    }
    temporal_spec = {
        "split_slug": "temporal",
        "protocol": "temporal",
        "seed": "year_based",
        "role_column": "temporal_role",
    }
    class_weights = class_weights_for_spec(data["training"], temporal_spec)
    protocol_hash = hash_file(PROTOCOL_FILE)
    outputs: dict[str, dict[str, Any]] = {}
    preprocessing_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    artifact_paths: list[Path] = []

    preprocessor = make_linear_preprocessor(
        categorical_columns=categorical,
        numeric_columns=numeric,
        category_vocabulary=vocabulary,
    )
    preprocessing_start = time.perf_counter()
    encoded_train = preprocessor.fit_transform(prepared["train"])
    encoded = {
        cohort: preprocessor.transform(prepared[cohort]) for cohort in COHORTS
    }
    preprocessing_seconds = time.perf_counter() - preprocessing_start
    logistic = make_logistic(class_weights)
    fit_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        with threadpool_limits(limits=THREADS):
            logistic.fit(encoded_train, y["train"])
    logistic_fit_seconds = time.perf_counter() - fit_start
    convergence_warning = any(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    logistic_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    prediction_seconds: dict[str, float] = {}
    for cohort in COHORTS:
        prediction_start = time.perf_counter()
        logistic_predictions[cohort] = predict_with_probabilities(
            logistic, encoded[cohort]
        )
        prediction_seconds[cohort] = time.perf_counter() - prediction_start
    outputs["logistic_weighted"] = {
        "predictions": logistic_predictions,
        "fit_seconds": logistic_fit_seconds,
    }

    numeric_transformer = preprocessor.named_transformers_["numeric"]
    imputer = numeric_transformer.named_steps["imputer"]
    indicator_features = {
        numeric[int(index)] for index in imputer.indicator_.features_
    }
    for column, median in zip(numeric, imputer.statistics_, strict=True):
        preprocessing_rows.append(
            {
                "scenario": scenario["scenario_id"],
                "model": "logistic_weighted",
                "feature": column,
                "feature_type": "numeric",
                "training_levels": "",
                "validation_unseen_count": "",
                "test_unseen_count": "",
                "training_missing_count": int(prepared["train"][column].isna().sum()),
                "validation_missing_count": int(
                    prepared["validation_2024"][column].isna().sum()
                ),
                "test_missing_count": int(prepared["test_2025"][column].isna().sum()),
                "training_fitted_imputation_value": float(median),
                "missing_indicator_fitted": column in indicator_features,
                "value_applied_to_data": "training median plus missing indicator",
            }
        )
    for column in categorical:
        preprocessing_rows.append(
            {
                "scenario": scenario["scenario_id"],
                "model": "logistic_weighted",
                "feature": column,
                "feature_type": "categorical_one_hot",
                "training_levels": len(vocabulary[column]),
                "validation_unseen_count": unseen["validation_2024"][column],
                "test_unseen_count": unseen["test_2025"][column],
                "training_missing_count": 0,
                "validation_missing_count": 0,
                "test_missing_count": 0,
                "training_fitted_imputation_value": "",
                "missing_indicator_fitted": "",
                "value_applied_to_data": "training vocabulary plus __UNSEEN__",
            }
        )
    training_rows.append(
        {
            "scenario": scenario["scenario_id"],
            "model": "logistic_weighted",
            "feature_count": len(feature_columns),
            "training_rows": len(positions["train"]),
            "validation_rows": len(positions["validation_2024"]),
            "test_rows": len(positions["test_2025"]),
            "minor_class_weight": class_weights[0],
            "serious_class_weight": class_weights[1],
            "fatal_class_weight": class_weights[2],
            "encoded_features": len(preprocessor.get_feature_names_out()),
            "n_estimators": "",
            "n_iter": int(np.max(logistic.n_iter_)),
            "convergence_warning": convergence_warning,
            "preprocessing_seconds": preprocessing_seconds,
            "fit_seconds": logistic_fit_seconds,
            "validation_prediction_seconds": prediction_seconds["validation_2024"],
            "test_prediction_seconds": prediction_seconds["test_2025"],
            "threads": THREADS,
            "parameters_json": json.dumps(
                protocol["models"]["logistic_weighted"], sort_keys=True
            ),
        }
    )

    lightgbm_config = json.loads(
        LIGHTGBM_PROTOCOL_FILE.read_text(encoding="utf-8")
    )
    selected = json.loads(LIGHTGBM_SELECTED_FILE.read_text(encoding="utf-8"))
    selected_candidate = {
        "candidate_id": selected["selected_candidate_id"],
        "complexity_rank": selected["selected_complexity_rank"],
        "parameters": selected["selected_candidate_parameters"],
    }
    lightgbm = make_estimator(
        lightgbm_config,
        selected_candidate,
        class_weights,
        n_estimators=n_estimators,
    )
    fit_start = time.perf_counter()
    lightgbm.fit(
        prepared["train"],
        y["train"],
        categorical_feature=categorical,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    lightgbm_fit_seconds = time.perf_counter() - fit_start
    lightgbm_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    lgb_prediction_seconds: dict[str, float] = {}
    for cohort in COHORTS:
        prediction_start = time.perf_counter()
        lightgbm_predictions[cohort] = predict_with_probabilities(
            lightgbm, prepared[cohort]
        )
        lgb_prediction_seconds[cohort] = time.perf_counter() - prediction_start
    outputs["lightgbm_weighted"] = {
        "predictions": lightgbm_predictions,
        "fit_seconds": lightgbm_fit_seconds,
    }
    for column in numeric:
        preprocessing_rows.append(
            {
                "scenario": scenario["scenario_id"],
                "model": "lightgbm_weighted",
                "feature": column,
                "feature_type": "numeric_native",
                "training_levels": "",
                "validation_unseen_count": "",
                "test_unseen_count": "",
                "training_missing_count": int(prepared["train"][column].isna().sum()),
                "validation_missing_count": int(
                    prepared["validation_2024"][column].isna().sum()
                ),
                "test_missing_count": int(prepared["test_2025"][column].isna().sum()),
                "training_fitted_imputation_value": "",
                "missing_indicator_fitted": "",
                "value_applied_to_data": "none; NaN retained",
            }
        )
    for column in categorical:
        preprocessing_rows.append(
            {
                "scenario": scenario["scenario_id"],
                "model": "lightgbm_weighted",
                "feature": column,
                "feature_type": "categorical_native",
                "training_levels": len(vocabulary[column]),
                "validation_unseen_count": unseen["validation_2024"][column],
                "test_unseen_count": unseen["test_2025"][column],
                "training_missing_count": 0,
                "validation_missing_count": 0,
                "test_missing_count": 0,
                "training_fitted_imputation_value": "",
                "missing_indicator_fitted": "",
                "value_applied_to_data": "training vocabulary plus __UNSEEN__",
            }
        )
    training_rows.append(
        {
            "scenario": scenario["scenario_id"],
            "model": "lightgbm_weighted",
            "feature_count": len(feature_columns),
            "training_rows": len(positions["train"]),
            "validation_rows": len(positions["validation_2024"]),
            "test_rows": len(positions["test_2025"]),
            "minor_class_weight": class_weights[0],
            "serious_class_weight": class_weights[1],
            "fatal_class_weight": class_weights[2],
            "encoded_features": "",
            "n_estimators": n_estimators,
            "n_iter": "",
            "convergence_warning": "",
            "preprocessing_seconds": "",
            "fit_seconds": lightgbm_fit_seconds,
            "validation_prediction_seconds": lgb_prediction_seconds["validation_2024"],
            "test_prediction_seconds": lgb_prediction_seconds["test_2025"],
            "threads": THREADS,
            "parameters_json": json.dumps(
                protocol["models"]["lightgbm_weighted"], sort_keys=True
            ),
        }
    )

    if persist:
        split_dir = MODEL_DIR / scenario["scenario_id"]
        split_dir.mkdir(parents=True, exist_ok=True)
        logistic_preprocessor_path = (
            split_dir / "logistic_preprocessing_bundle.joblib"
        )
        logistic_model_path = split_dir / "logistic_weighted.joblib"
        lightgbm_preprocessor_path = (
            split_dir / "lightgbm_preprocessing_bundle.joblib"
        )
        lightgbm_model_path = split_dir / "lightgbm_weighted.joblib"
        joblib.dump(
            {
                "version": VERSION,
                "scenario": scenario["scenario_id"],
                "feature_columns": feature_columns,
                "categorical_columns": categorical,
                "numeric_columns": numeric,
                "category_vocabulary": vocabulary,
                "preprocessor": preprocessor,
                "encoded_feature_names": preprocessor.get_feature_names_out().tolist(),
                "numeric_missing_indicator_features": sorted(indicator_features),
                "fit_scope": "2022-2023 training rows only",
                "protocol_sha256": protocol_hash,
            },
            logistic_preprocessor_path,
            compress=3,
        )
        joblib.dump(
            {
                "version": VERSION,
                "scenario": scenario["scenario_id"],
                "model_name": "logistic_weighted",
                "estimator": logistic,
                "class_weight": class_weights,
                "prediction_rule": "argmax over class probabilities",
                "protocol_sha256": protocol_hash,
            },
            logistic_model_path,
            compress=3,
        )
        joblib.dump(
            {
                "version": VERSION,
                "scenario": scenario["scenario_id"],
                "feature_columns": feature_columns,
                "categorical_columns": categorical,
                "numeric_columns": numeric,
                "category_vocabulary": vocabulary,
                "numeric_missing_policy": "native_nan",
                "fit_scope": "2022-2023 training rows only",
                "protocol_sha256": protocol_hash,
            },
            lightgbm_preprocessor_path,
            compress=3,
        )
        joblib.dump(
            {
                "version": VERSION,
                "scenario": scenario["scenario_id"],
                "model_name": "lightgbm_weighted",
                "estimator": lightgbm,
                "class_weight": class_weights,
                "selected_candidate_id": selected["selected_candidate_id"],
                "n_estimators": n_estimators,
                "prediction_rule": "argmax over class probabilities",
                "protocol_sha256": protocol_hash,
            },
            lightgbm_model_path,
            compress=3,
        )
        artifact_paths.extend(
            [
                logistic_preprocessor_path,
                logistic_model_path,
                lightgbm_preprocessor_path,
                lightgbm_model_path,
            ]
        )
        for model in MODELS:
            for cohort in COHORTS:
                predicted, probabilities = outputs[model]["predictions"][cohort]
                artifact_paths.append(
                    persist_predictions(
                        scenario_id=scenario["scenario_id"],
                        cohort=cohort,
                        model=model,
                        metadata=metadata,
                        assignments=assignments,
                        positions=positions[cohort],
                        y_true=y[cohort],
                        y_pred=predicted,
                        probabilities=probabilities,
                    )
                )

    return {
        "scenario": scenario,
        "feature_columns": feature_columns,
        "categorical": categorical,
        "numeric": numeric,
        "y": y,
        "outputs": outputs,
        "preprocessing_rows": preprocessing_rows,
        "training_rows": training_rows,
        "artifact_paths": artifact_paths,
    }


def load_full15_references(data: dict[str, Any]) -> dict[str, pd.DataFrame]:
    positions = data["positions"]["test_2025"]
    expected_ids = (
        data["metadata"].iloc[positions]["meta_crash_id"].astype("string").to_numpy()
    )
    expected_years = (
        data["assignments"].iloc[positions]["meta_crash_year"].to_numpy(dtype=int)
    )
    expected_target = data["target"].iloc[positions].to_numpy(dtype=np.int8)
    result: dict[str, pd.DataFrame] = {}
    main_metrics = pd.read_csv(MAIN_TEST_METRICS_FILE)
    for model in MODELS:
        path = MAIN_PREDICTION_DIR / f"temporal_2025__{model}.csv.gz"
        frame = pd.read_csv(path, dtype={"meta_crash_id": "string"})
        if frame["meta_crash_id"].duplicated().any():
            raise ValueError(f"Duplicate full-model prediction IDs: {model}")
        if not np.array_equal(frame["meta_crash_id"].to_numpy(), expected_ids):
            raise AssertionError(f"Full-model IDs are not aligned: {model}")
        if not np.array_equal(frame["meta_crash_year"].to_numpy(int), expected_years):
            raise AssertionError(f"Full-model years are not aligned: {model}")
        if not np.array_equal(
            frame["target_severity"].to_numpy(np.int8), expected_target
        ):
            raise AssertionError(f"Full-model targets are not aligned: {model}")
        probabilities = frame[["prob_minor", "prob_serious", "prob_fatal"]].to_numpy()
        if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=2e-8):
            raise AssertionError(f"Full-model probabilities are invalid: {model}")
        computed = classification_metrics(
            expected_target,
            frame["predicted_severity"].to_numpy(np.int8),
        )
        row = main_metrics.loc[
            main_metrics["evaluation_group"].eq("temporal_2025")
            & main_metrics["model"].eq(model)
        ]
        if len(row) != 1:
            raise ValueError(f"Expected one main temporal metric row: {model}")
        for metric, value in computed.items():
            if not np.isclose(value, float(row.iloc[0][metric]), rtol=0, atol=1e-14):
                raise AssertionError(f"Main metric mismatch for {model}: {metric}")
        result[model] = frame
    return result


def bootstrap_outputs(
    *,
    protocol: dict[str, Any],
    scenarios: list[dict[str, Any]],
    full_references: dict[str, pd.DataFrame],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    settings = protocol["bootstrap"]
    iterations = int(settings["iterations"])
    base_seed = int(settings["base_seed"])
    metrics = list(settings["metrics"])
    higher = dict(settings["higher_is_better"])
    interval_rows: list[dict[str, object]] = []
    difference_rows: list[dict[str, object]] = []

    for scenario_index, result in enumerate(scenarios):
        scenario_id = result["scenario"]["scenario_id"]
        y_true = result["y"]["test_2025"]
        for model_index, model in enumerate(MODELS):
            predicted = result["outputs"][model]["predictions"]["test_2025"][0]
            point = metrics_from_confusion(point_confusion(y_true, predicted))
            own_seed = base_seed + scenario_index * 1_000 + model_index * 100 + 10
            own_cm = stratified_confusion_bootstrap(
                y_true,
                predicted,
                iterations,
                np.random.default_rng(own_seed),
            )
            own_metrics = metrics_from_confusion(own_cm)
            for metric in metrics:
                lower, upper, valid = percentile_interval(own_metrics[metric])
                interval_rows.append(
                    {
                        "scenario": scenario_id,
                        "model": model,
                        "cohort": "test_2025",
                        "metric": metric,
                        "point_estimate": float(point[metric]),
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "valid_iterations": valid,
                        "bootstrap_design": "stratified",
                        "bootstrap_seed": own_seed,
                        "higher_is_better": bool(higher[metric]),
                    }
                )

            full_predicted = full_references[model][
                "predicted_severity"
            ].to_numpy(np.int8)
            comparison_seed = own_seed + 1
            cm_ablation, cm_full = paired_confusion_bootstrap(
                y_true,
                predicted,
                full_predicted,
                iterations,
                np.random.default_rng(comparison_seed),
            )
            metrics_ablation = metrics_from_confusion(cm_ablation)
            metrics_full = metrics_from_confusion(cm_full)
            point_full = metrics_from_confusion(
                point_confusion(y_true, full_predicted)
            )
            for metric in metrics:
                differences = metrics_ablation[metric] - metrics_full[metric]
                lower, upper, valid = percentile_interval(differences)
                difference_rows.append(
                    {
                        "scenario": scenario_id,
                        "comparison": "ablation_minus_full15",
                        "model_a": model,
                        "model_b": model,
                        "model_a_variant": scenario_id,
                        "model_b_variant": "full15_primary",
                        "metric": metric,
                        "model_a_value": float(point[metric]),
                        "model_b_value": float(point_full[metric]),
                        "point_difference_a_minus_b": float(
                            point[metric] - point_full[metric]
                        ),
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "valid_iterations": valid,
                        "bootstrap_design": "paired_stratified",
                        "bootstrap_seed": comparison_seed,
                        "higher_is_better": bool(higher[metric]),
                    }
                )

        logistic_predicted = result["outputs"]["logistic_weighted"][
            "predictions"
        ]["test_2025"][0]
        lightgbm_predicted = result["outputs"]["lightgbm_weighted"][
            "predictions"
        ]["test_2025"][0]
        within_seed = base_seed + scenario_index * 1_000 + 900
        cm_lightgbm, cm_logistic = paired_confusion_bootstrap(
            y_true,
            lightgbm_predicted,
            logistic_predicted,
            iterations,
            np.random.default_rng(within_seed),
        )
        lightgbm_metrics = metrics_from_confusion(cm_lightgbm)
        logistic_metrics = metrics_from_confusion(cm_logistic)
        point_lightgbm = metrics_from_confusion(
            point_confusion(y_true, lightgbm_predicted)
        )
        point_logistic = metrics_from_confusion(
            point_confusion(y_true, logistic_predicted)
        )
        for metric in metrics:
            differences = lightgbm_metrics[metric] - logistic_metrics[metric]
            lower, upper, valid = percentile_interval(differences)
            difference_rows.append(
                {
                    "scenario": scenario_id,
                    "comparison": "lightgbm_minus_logistic_within_ablation",
                    "model_a": "lightgbm_weighted",
                    "model_b": "logistic_weighted",
                    "model_a_variant": scenario_id,
                    "model_b_variant": scenario_id,
                    "metric": metric,
                    "model_a_value": float(point_lightgbm[metric]),
                    "model_b_value": float(point_logistic[metric]),
                    "point_difference_a_minus_b": float(
                        point_lightgbm[metric] - point_logistic[metric]
                    ),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "valid_iterations": valid,
                    "bootstrap_design": "paired_stratified",
                    "bootstrap_seed": within_seed,
                    "higher_is_better": bool(higher[metric]),
                }
            )
    return interval_rows, difference_rows


def write_checkpoint(
    test_metrics: pd.DataFrame,
    differences: pd.DataFrame,
) -> None:
    lines = [
        "# CAS post-hoc feature-ablation checkpoint",
        "",
        f"Completed: {now_local()}",
        "",
        "## Status and scope",
        "",
        "- The primary 2025 CAS results were known before this protocol was frozen.",
        "- These results are post-hoc sensitivity evidence only.",
        "- The original 15-feature models, tuning decision, and primary results were not changed.",
        "- Both ablations reused the frozen temporal split, class weights, hyperparameters, and argmax rule.",
        "",
        "## 2025 point estimates",
        "",
        "| Variant | Model | Macro-F1 | QWK | Fatal recall | Ordinal MAE | Mean cost |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in test_metrics.iterrows():
        lines.append(
            f"| {row['scenario']} | {row['model']} | "
            f"{row['macro_f1']:.4f} | {row['qwk']:.4f} | "
            f"{row['fatal_recall']:.4f} | {row['ordinal_mae']:.4f} | "
            f"{row['mean_asymmetric_cost']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired differences against the corresponding full 15-feature model",
            "",
            "Differences are ablation minus full model. For ordinal MAE and mean cost, lower is better.",
            "",
            "| Ablation | Model | Metric | Difference | 95% interval |",
            "|---|---|---|---:|---:|",
        ]
    )
    selected_metrics = {"macro_f1", "qwk", "fatal_recall", "mean_asymmetric_cost"}
    subset = differences.loc[
        differences["comparison"].eq("ablation_minus_full15")
        & differences["metric"].isin(selected_metrics)
    ]
    for _, row in subset.iterrows():
        lines.append(
            f"| {row['scenario']} | {row['model_a']} | {row['metric']} | "
            f"{row['point_difference_a_minus_b']:+.4f} | "
            f"[{row['ci_lower']:+.4f}, {row['ci_upper']:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The intervals quantify record-level sampling uncertainty conditional on the fixed data, split, and fitting procedure. They do not convert this post-hoc analysis into a preregistered test and do not justify selecting a replacement primary model.",
            "",
        ]
    )
    CHECKPOINT_FILE.write_text("\n".join(lines), encoding="utf-8")


def run_smoke() -> None:
    protocol = load_protocol()
    data = load_temporal_data()
    settings = protocol["smoke_test"]
    positions = {
        "train": stratified_subset(
            data["positions"]["train"],
            data["target"],
            int(settings["training_rows"]),
            int(settings["seed"]),
        ),
        "validation_2024": stratified_subset(
            data["positions"]["validation_2024"],
            data["target"],
            int(settings["validation_rows"]),
            int(settings["seed"]) + 1,
        ),
        "test_2025": stratified_subset(
            data["positions"]["test_2025"],
            data["target"],
            int(settings["test_rows"]),
            int(settings["seed"]) + 2,
        ),
    }
    rows = []
    for scenario in protocol["scenarios"]:
        result = fit_scenario(
            protocol=protocol,
            data=data,
            scenario=scenario,
            positions=positions,
            n_estimators=int(settings["lightgbm_estimators"]),
            persist=False,
        )
        for model in MODELS:
            prediction = result["outputs"][model]["predictions"]["test_2025"][0]
            rows.append(
                {
                    "scenario": scenario["scenario_id"],
                    "model": model,
                    "test_rows": len(prediction),
                    "predicted_classes": sorted(map(int, np.unique(prediction))),
                }
            )
    write_json_atomic(
        SMOKE_FILE,
        {
            "version": VERSION,
            "status": "PASS",
            "created_local": now_local(),
            "protocol_sha256": hash_file(PROTOCOL_FILE),
            "analytical_result": False,
            "rows": rows,
        },
    )
    print("CAS_FEATURE_ABLATION_SMOKE=PASS")


def run_full() -> None:
    protocol = load_protocol()
    if COMPLETE_FILE.exists():
        raise RuntimeError("Completed CAS feature-ablation outputs already exist")
    if not SMOKE_FILE.exists():
        raise RuntimeError("Run --smoke before the full post-hoc analysis")
    smoke = json.loads(SMOKE_FILE.read_text(encoding="utf-8"))
    if smoke.get("status") != "PASS":
        raise ValueError("CAS feature-ablation smoke test did not pass")
    if smoke.get("protocol_sha256") != hash_file(PROTOCOL_FILE):
        raise ValueError("Smoke test used a different ablation protocol")

    data = load_temporal_data()
    full_references = load_full15_references(data)
    selected = json.loads(LIGHTGBM_SELECTED_FILE.read_text(encoding="utf-8"))
    n_estimators = int(selected["selected_n_estimators"])
    if n_estimators != int(protocol["models"]["lightgbm_weighted"]["n_estimators"]):
        raise AssertionError("Selected LightGBM tree count differs from protocol")

    scenario_results = []
    all_preprocessing_rows: list[dict[str, object]] = []
    all_training_rows: list[dict[str, object]] = []
    artifact_paths: list[Path] = []
    validation_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    confusion_output: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []

    for model in MODELS:
        frame = full_references[model]
        y_true = frame["target_severity"].to_numpy(np.int8)
        y_pred = frame["predicted_severity"].to_numpy(np.int8)
        test_rows.append(
            {
                "analysis_variant": "primary_reference",
                "scenario": "full15_primary",
                "feature_count": 15,
                "dropped_features": "",
                "cohort": "test_2025",
                "cohort_rows": len(frame),
                "model": model,
                **classification_metrics(y_true, y_pred),
            }
        )
        extend_confusion(
            confusion_output,
            scenario="full15_primary",
            cohort="test_2025",
            model=model,
            y_true=y_true,
            y_pred=y_pred,
        )

    for scenario in protocol["scenarios"]:
        result = fit_scenario(
            protocol=protocol,
            data=data,
            scenario=scenario,
            positions=data["positions"],
            n_estimators=n_estimators,
            persist=True,
        )
        scenario_results.append(result)
        all_preprocessing_rows.extend(result["preprocessing_rows"])
        all_training_rows.extend(result["training_rows"])
        artifact_paths.extend(result["artifact_paths"])
        dropped = list(scenario["drop_features"])
        for column in data["schema"]["feature_columns"]:
            feature_rows.append(
                {
                    "scenario": scenario["scenario_id"],
                    "feature": column,
                    "status": "dropped" if column in dropped else "retained",
                    "feature_type": (
                        "categorical"
                        if column in data["schema"]["categorical_feature_columns"]
                        else "numeric"
                    ),
                }
            )
        for model in MODELS:
            for cohort in COHORTS:
                predicted = result["outputs"][model]["predictions"][cohort][0]
                row = metric_row(
                    scenario_id=scenario["scenario_id"],
                    feature_count=len(result["feature_columns"]),
                    dropped_features=dropped,
                    cohort=cohort,
                    model=model,
                    y_true=result["y"][cohort],
                    y_pred=predicted,
                )
                if cohort == "validation_2024":
                    validation_rows.append(row)
                else:
                    test_rows.append(row)
                extend_confusion(
                    confusion_output,
                    scenario=scenario["scenario_id"],
                    cohort=cohort,
                    model=model,
                    y_true=result["y"][cohort],
                    y_pred=predicted,
                )
        print(
            f"{scenario['scenario_id']}: features={len(result['feature_columns'])}, "
            f"Logistic={result['outputs']['logistic_weighted']['fit_seconds']:.1f}s, "
            f"LightGBM={result['outputs']['lightgbm_weighted']['fit_seconds']:.1f}s",
            flush=True,
        )

    interval_rows, difference_rows = bootstrap_outputs(
        protocol=protocol,
        scenarios=scenario_results,
        full_references=full_references,
    )
    write_csv(VALIDATION_METRICS_FILE, validation_rows)
    write_csv(TEST_METRICS_FILE, test_rows)
    write_csv(CONFUSION_FILE, confusion_output)
    write_csv(INTERVAL_FILE, interval_rows)
    write_csv(DIFFERENCE_FILE, difference_rows)
    write_csv(FEATURE_CONTRACT_FILE, feature_rows)
    write_csv(PREPROCESS_AUDIT_FILE, all_preprocessing_rows)
    write_csv(TRAINING_AUDIT_FILE, all_training_rows)
    write_checkpoint(pd.DataFrame(test_rows), pd.DataFrame(difference_rows))

    generated = [
        VALIDATION_METRICS_FILE,
        TEST_METRICS_FILE,
        CONFUSION_FILE,
        INTERVAL_FILE,
        DIFFERENCE_FILE,
        FEATURE_CONTRACT_FILE,
        PREPROCESS_AUDIT_FILE,
        TRAINING_AUDIT_FILE,
        CHECKPOINT_FILE,
        SMOKE_FILE,
        PROTOCOL_FILE,
        Path(__file__),
        *artifact_paths,
    ]
    build_artifact_manifest(generated, MANIFEST_FILE)
    write_json_atomic(
        COMPLETE_FILE,
        {
            "version": VERSION,
            "status": "POST_HOC_FEATURE_ABLATION_COMPLETE",
            "created_local": now_local(),
            "protocol": relative(PROTOCOL_FILE),
            "protocol_sha256": hash_file(PROTOCOL_FILE),
            "manifest": relative(MANIFEST_FILE),
            "manifest_sha256": hash_file(MANIFEST_FILE),
            "scenarios": [
                {
                    "scenario_id": result["scenario"]["scenario_id"],
                    "feature_count": len(result["feature_columns"]),
                    "drop_features": result["scenario"]["drop_features"],
                }
                for result in scenario_results
            ],
            "models_refit": list(MODELS),
            "primary_models_or_results_modified": False,
            "eligible_for_model_selection": False,
            "primary_2025_results_known_before_protocol": True,
            "bootstrap_iterations": int(protocol["bootstrap"]["iterations"]),
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": package_metadata.version("numpy"),
                "pandas": package_metadata.version("pandas"),
                "scikit_learn": package_metadata.version("scikit-learn"),
                "lightgbm": package_metadata.version("lightgbm"),
                "joblib": package_metadata.version("joblib"),
                "threads": THREADS,
            },
        },
    )
    print("CAS_FEATURE_ABLATION_STATUS=POST_HOC_COMPLETE")


def main() -> None:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--smoke", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        run_smoke()
    else:
        run_full()


if __name__ == "__main__":
    main()
