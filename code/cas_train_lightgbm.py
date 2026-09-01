"""Freeze, tune and fit the weighted CAS LightGBM comparator.

Tuning uses only 2022-2023 training and 2024 validation. Test roles and 2025
remain inaccessible until the selected models are frozen.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import joblib
import lightgbm as lgb
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
    class_weights_for_spec,
    classification_metrics,
    confusion_rows,
    fit_category_vocabulary,
    hash_file,
    load_aligned_data,
    load_contracts,
    make_split_specs,
    predict_with_probabilities,
    prediction_frame,
    prepare_features,
    relative,
    write_csv,
    write_dataframe_gzip,
    write_json_atomic,
)


VERSION = "CAS_LIGHTGBM_V1"
MODEL_NAME = "lightgbm_weighted"
MODEL_SEED = 20_260_901
MAX_ESTIMATORS = 1_200
EXTENDED_MAX_ESTIMATORS = 2_000
EARLY_STOPPING_ROUNDS = 75

BASELINE_PROTOCOL_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_baseline_protocol.json"
)
BASELINE_FROZEN_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_baseline_models_frozen.json"
)
BASELINE_METRICS_FILE = (
    PROJECT_DIR / "results" / "cas_baseline" / "validation_metrics.csv"
)
CONFIG_FILE = PROJECT_DIR / "config" / "cas" / "cas_lightgbm_protocol.json"
SELECTED_FILE = PROJECT_DIR / "config" / "cas" / "cas_lightgbm_selected.json"
FROZEN_MODELS_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_lightgbm_models_frozen.json"
)
MODEL_DIR = PROJECT_DIR / "models" / "cas_lightgbm"
RESULT_DIR = PROJECT_DIR / "results" / "cas_lightgbm"
TUNING_FILE = RESULT_DIR / "tuning_results.csv"
METRICS_FILE = RESULT_DIR / "validation_metrics.csv"
CONFUSION_FILE = RESULT_DIR / "validation_confusion_matrices.csv"
PREDICTION_DIR = RESULT_DIR / "validation_predictions"
LOG_DIR = PROJECT_DIR / "logs" / "cas"
PREPROCESS_AUDIT_FILE = LOG_DIR / "cas_lightgbm_preprocessing_audit.csv"
TRAINING_AUDIT_FILE = LOG_DIR / "cas_lightgbm_training_audit.csv"
MANIFEST_FILE = LOG_DIR / "cas_lightgbm_artifact_manifest.csv"
CHECKPOINT_FILE = LOG_DIR / "cas_lightgbm_checkpoint.md"
SMOKE_FILE = LOG_DIR / "cas_lightgbm_smoke.json"

CANDIDATES = [
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
]


def now_local() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def baseline_reference() -> dict[str, float]:
    metrics = pd.read_csv(BASELINE_METRICS_FILE)
    rows = metrics.loc[
        metrics["protocol"].eq("temporal")
        & metrics["model"].eq("logistic_weighted")
    ]
    if len(rows) != 1:
        raise ValueError("Expected one temporal weighted-Logistic baseline")
    row = rows.iloc[0]
    return {
        "macro_f1": float(row["macro_f1"]),
        "qwk": float(row["qwk"]),
        "fatal_recall": float(row["fatal_recall"]),
    }


def freeze_protocol() -> None:
    schema, analysis, training = load_contracts()
    baseline_frozen = json.loads(
        BASELINE_FROZEN_FILE.read_text(encoding="utf-8")
    )
    if baseline_frozen.get("status") != (
        "BASELINE_MODELS_FROZEN_BEFORE_2025_EVALUATION"
    ):
        raise ValueError("CAS baselines are not frozen")
    if SELECTED_FILE.exists() or TUNING_FILE.exists():
        raise RuntimeError("LightGBM results exist; refusing to refreeze protocol")
    reference = baseline_reference()
    payload = {
        "version": VERSION,
        "status": "FROZEN_BEFORE_CAS_LIGHTGBM_VALIDATION_INSPECTION",
        "created_local": now_local(),
        "purpose": (
            "Reuse the STATS19 six-candidate weighted-LightGBM comparison "
            "on CAS without accessing test performance"
        ),
        "upstream": {
            "modeling_schema": relative(SCHEMA_FILE),
            "modeling_schema_sha256": hash_file(SCHEMA_FILE),
            "analysis_protocol": relative(ANALYSIS_PROTOCOL_FILE),
            "analysis_protocol_sha256": hash_file(ANALYSIS_PROTOCOL_FILE),
            "training_inputs": relative(TRAINING_INPUTS_FILE),
            "training_inputs_sha256": hash_file(TRAINING_INPUTS_FILE),
            "split_assignments": relative(ASSIGNMENTS_FILE),
            "split_assignments_sha256": hash_file(ASSIGNMENTS_FILE),
            "baseline_protocol": relative(BASELINE_PROTOCOL_FILE),
            "baseline_protocol_sha256": hash_file(BASELINE_PROTOCOL_FILE),
            "baseline_models_frozen": relative(BASELINE_FROZEN_FILE),
            "baseline_models_frozen_sha256": hash_file(BASELINE_FROZEN_FILE),
            "baseline_validation_metrics": relative(BASELINE_METRICS_FILE),
            "baseline_validation_metrics_sha256": hash_file(
                BASELINE_METRICS_FILE
            ),
        },
        "evaluation_scope": {
            "tuning_protocol": "temporal only",
            "tuning_train_years": [2022, 2023],
            "tuning_validation_years": [2024],
            "post_selection_validation_models": (
                "temporal plus five frozen random-reference splits"
            ),
            "random_split_rule": (
                "reuse temporal-selected candidate and iteration count"
            ),
            "forbidden": "all test roles and all 2025 model performance",
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
            "categorical": (
                "LightGBM-native pandas categorical values with training-only "
                "vocabulary and explicit __UNSEEN__"
            ),
            "numeric": "numeric values retained; NaN handled natively",
            "one_hot_encoding": False,
            "scaling": False,
            "rare_pooling": False,
        },
        "class_weighting": {
            "rule": training["class_weight_rule"],
            "application": "split-specific frozen training weights",
            "validation_metrics": "unweighted natural-prevalence metrics",
        },
        "common_model_parameters": {
            "boosting_type": "gbdt",
            "objective": "multiclass",
            "num_class": 3,
            "learning_rate": 0.05,
            "max_bin": 255,
            "random_state": MODEL_SEED,
            "n_jobs": THREADS,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        },
        "tuning": {
            "maximum_estimators": MAX_ESTIMATORS,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "early_stopping_metric": (
                "unweighted multiclass log loss on 2024 validation"
            ),
            "candidates": CANDIDATES,
            "boundary_rule": {
                "trigger": (
                    "selected candidate best_iteration equals "
                    f"{MAX_ESTIMATORS}"
                ),
                "action": (
                    "rerun only the selected candidate with "
                    f"{EXTENDED_MAX_ESTIMATORS} estimators and the same "
                    "early-stopping rule before final model freeze"
                ),
                "selection": "use the extended run's best_iteration",
            },
        },
        "selection_rule": {
            "primary_metric": "Macro-F1 on 2024 temporal validation",
            "constraint_reference": {
                "model": "temporal weighted multinomial Logistic",
                **reference,
            },
            "eligibility_constraints": {
                "minimum_qwk": reference["qwk"] - 0.01,
                "minimum_fatal_recall": reference["fatal_recall"] - 0.05,
            },
            "ranking": [
                "eligible candidates only",
                "highest Macro-F1",
                "highest QWK if tied",
                "highest fatal recall if tied",
                "lowest pre-frozen complexity_rank if tied",
                "lexicographically smallest candidate_id if tied",
            ],
            "fallback_if_none_eligible": (
                "apply the same ranking to all candidates and explicitly "
                "record that constraints were unmet"
            ),
        },
        "final_validation_fit": {
            "n_estimators": "selected temporal best_iteration",
            "early_stopping": False,
            "same_parameters_for_all_splits": True,
            "prediction_rule": "argmax; no threshold tuning",
        },
        "test_boundary": training["test_boundary"],
    }
    write_json_atomic(CONFIG_FILE, payload)
    print(f"CAS_LIGHTGBM_PROTOCOL_SHA256={hash_file(CONFIG_FILE)}")
    print("CAS_LIGHTGBM_PROTOCOL_STATUS=FROZEN_BEFORE_VALIDATION_INSPECTION")


def verify_protocol() -> dict[str, object]:
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    if config.get("version") != VERSION:
        raise ValueError("Unexpected CAS LightGBM protocol version")
    if config.get("status") != (
        "FROZEN_BEFORE_CAS_LIGHTGBM_VALIDATION_INSPECTION"
    ):
        raise ValueError("CAS LightGBM protocol is not frozen")
    checks = {
        SCHEMA_FILE: config["upstream"]["modeling_schema_sha256"],
        ANALYSIS_PROTOCOL_FILE: config["upstream"]["analysis_protocol_sha256"],
        TRAINING_INPUTS_FILE: config["upstream"]["training_inputs_sha256"],
        ASSIGNMENTS_FILE: config["upstream"]["split_assignments_sha256"],
        BASELINE_PROTOCOL_FILE: config["upstream"]["baseline_protocol_sha256"],
        BASELINE_FROZEN_FILE: config["upstream"][
            "baseline_models_frozen_sha256"
        ],
        BASELINE_METRICS_FILE: config["upstream"][
            "baseline_validation_metrics_sha256"
        ],
    }
    for path, expected in checks.items():
        if hash_file(path) != expected:
            raise ValueError(f"Frozen LightGBM upstream changed: {path.name}")
    common = config["common_model_parameters"]
    if int(common["n_jobs"]) != THREADS:
        raise ValueError("LightGBM thread setting differs from frozen protocol")
    if not common["deterministic"] or not common["force_col_wise"]:
        raise ValueError("LightGBM deterministic controls are not frozen")
    return config


def make_estimator(
    config: dict[str, object],
    candidate: dict[str, object],
    class_weights: dict[int, float],
    *,
    n_estimators: int,
) -> lgb.LGBMClassifier:
    common = dict(config["common_model_parameters"])
    common.update(candidate["parameters"])
    return lgb.LGBMClassifier(
        **common,
        n_estimators=n_estimators,
        class_weight=class_weights,
        bagging_seed=MODEL_SEED,
        feature_fraction_seed=MODEL_SEED,
        data_random_seed=MODEL_SEED,
    )


def temporal_data() -> tuple[
    dict[str, object],
    dict[str, object],
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
]:
    schema, analysis, training = load_contracts()
    features, target, metadata, assignments = load_aligned_data(schema, analysis)
    train_positions = np.flatnonzero(
        assignments["temporal_role"].eq("train").to_numpy()
    )
    validation_positions = np.flatnonzero(
        assignments["temporal_role"].eq("validation").to_numpy()
    )
    if set(assignments.iloc[train_positions]["meta_crash_year"]) != {2022, 2023}:
        raise AssertionError("Unexpected temporal training years")
    if set(assignments.iloc[validation_positions]["meta_crash_year"]) != {2024}:
        raise AssertionError("Unexpected temporal validation year")
    return (
        schema,
        training,
        features,
        target,
        metadata,
        assignments,
        train_positions,
        validation_positions,
    )


def prepare_native_frames(
    schema: dict[str, object],
    features: pd.DataFrame,
    train_positions: np.ndarray,
    validation_positions: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]], dict[str, int]]:
    categorical = list(schema["categorical_feature_columns"])
    numeric = list(schema["numeric_feature_columns"])
    feature_columns = list(schema["feature_columns"])
    X_train = features.iloc[train_positions]
    X_validation = features.iloc[validation_positions]
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
        raise AssertionError("CAS LightGBM training data produced unseen levels")
    return (
        prepared_train.frame,
        prepared_validation.frame,
        vocabulary,
        prepared_validation.unseen_counts,
    )


def fit_candidate(
    *,
    config: dict[str, object],
    candidate: dict[str, object],
    class_weights: dict[int, float],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_validation: pd.DataFrame,
    y_validation: np.ndarray,
    n_estimators: int,
    run_type: str,
) -> tuple[dict[str, object], lgb.LGBMClassifier]:
    estimator = make_estimator(
        config,
        candidate,
        class_weights,
        n_estimators=n_estimators,
    )
    start = time.perf_counter()
    estimator.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        eval_metric="multi_logloss",
        categorical_feature=list(config["features"]["categorical"]),
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    fit_seconds = time.perf_counter() - start
    predicted, _probabilities = predict_with_probabilities(estimator, X_validation)
    metrics = classification_metrics(y_validation, predicted)
    best_iteration = int(estimator.best_iteration_ or n_estimators)
    best_logloss = float(
        estimator.best_score_["valid_0"]["multi_logloss"]
    )
    row = {
        "candidate_id": candidate["candidate_id"],
        "complexity_rank": int(candidate["complexity_rank"]),
        "run_type": run_type,
        "maximum_estimators": n_estimators,
        "best_iteration": best_iteration,
        "boundary_reached": best_iteration == n_estimators,
        "validation_multi_logloss": best_logloss,
        "fit_seconds": fit_seconds,
        **metrics,
    }
    for name, value in candidate["parameters"].items():
        row[name] = value
    return row, estimator


def select_candidate(
    rows: list[dict[str, object]],
    config: dict[str, object],
) -> tuple[dict[str, object], bool]:
    initial = [row for row in rows if row["run_type"] == "initial"]
    constraints = config["selection_rule"]["eligibility_constraints"]
    eligible = [
        row
        for row in initial
        if float(row["qwk"]) >= float(constraints["minimum_qwk"])
        and float(row["fatal_recall"])
        >= float(constraints["minimum_fatal_recall"])
    ]
    pool = eligible if eligible else initial
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
    return selected, bool(eligible)


def run_smoke() -> None:
    config = verify_protocol()
    (
        schema,
        training,
        features,
        target,
        _metadata,
        _assignments,
        train_positions,
        validation_positions,
    ) = temporal_data()
    rng = np.random.default_rng(90_102)
    train_positions = np.sort(rng.choice(train_positions, 5_000, replace=False))
    validation_positions = np.sort(
        rng.choice(validation_positions, 2_000, replace=False)
    )
    X_train, X_validation, _vocab, _unseen = prepare_native_frames(
        schema, features, train_positions, validation_positions
    )
    y_train = target.iloc[train_positions].to_numpy(dtype=np.int8)
    y_validation = target.iloc[validation_positions].to_numpy(dtype=np.int8)
    weights = class_weights_for_spec(
        training,
        {
            "protocol": "temporal",
            "seed": "year_based",
            "role_column": "temporal_role",
            "split_slug": "temporal",
        },
    )
    estimator = make_estimator(
        config,
        CANDIDATES[0],
        weights,
        n_estimators=100,
    )
    estimator.fit(
        X_train,
        y_train,
        categorical_feature=list(config["features"]["categorical"]),
        callbacks=[lgb.log_evaluation(period=0)],
    )
    predicted, probabilities = predict_with_probabilities(estimator, X_validation)
    if len(predicted) != len(validation_positions):
        raise AssertionError("CAS LightGBM smoke prediction rows changed")
    write_json_atomic(
        SMOKE_FILE,
        {
            "version": VERSION,
            "status": "PASS",
            "threads": THREADS,
            "training_rows": len(train_positions),
            "validation_rows": len(validation_positions),
            "probability_rows": len(probabilities),
            "test_performance_used": False,
        },
    )
    print("CAS_LIGHTGBM_SMOKE=PASS")


def run_tuning() -> None:
    config = verify_protocol()
    (
        schema,
        training,
        features,
        target,
        _metadata,
        _assignments,
        train_positions,
        validation_positions,
    ) = temporal_data()
    X_train, X_validation, _vocabulary, _unseen = prepare_native_frames(
        schema, features, train_positions, validation_positions
    )
    y_train = target.iloc[train_positions].to_numpy(dtype=np.int8)
    y_validation = target.iloc[validation_positions].to_numpy(dtype=np.int8)
    temporal_spec = {
        "protocol": "temporal",
        "seed": "year_based",
        "role_column": "temporal_role",
        "split_slug": "temporal",
    }
    class_weights = class_weights_for_spec(training, temporal_spec)

    rows: list[dict[str, object]] = []
    for candidate in config["tuning"]["candidates"]:
        row, _estimator = fit_candidate(
            config=config,
            candidate=candidate,
            class_weights=class_weights,
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            n_estimators=MAX_ESTIMATORS,
            run_type="initial",
        )
        rows.append(row)
        print(
            f"{row['candidate_id']}: best={row['best_iteration']}, "
            f"Macro-F1={row['macro_f1']:.5f}, QWK={row['qwk']:.5f}, "
            f"Fatal recall={row['fatal_recall']:.5f}",
            flush=True,
        )

    selected, constraints_met = select_candidate(rows, config)
    boundary_extension = False
    if int(selected["best_iteration"]) == MAX_ESTIMATORS:
        boundary_extension = True
        candidate = next(
            item
            for item in config["tuning"]["candidates"]
            if item["candidate_id"] == selected["candidate_id"]
        )
        extended, _estimator = fit_candidate(
            config=config,
            candidate=candidate,
            class_weights=class_weights,
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            n_estimators=EXTENDED_MAX_ESTIMATORS,
            run_type="boundary_extension",
        )
        rows.append(extended)
        selected = extended
        constraints = config["selection_rule"]["eligibility_constraints"]
        constraints_met = (
            float(selected["qwk"]) >= float(constraints["minimum_qwk"])
            and float(selected["fatal_recall"])
            >= float(constraints["minimum_fatal_recall"])
        )
        print(
            f"Boundary extension {selected['candidate_id']}: "
            f"best={selected['best_iteration']}",
            flush=True,
        )

    write_csv(TUNING_FILE, rows)
    candidate = next(
        item
        for item in config["tuning"]["candidates"]
        if item["candidate_id"] == selected["candidate_id"]
    )
    selected_metrics = {
        name: float(selected[name])
        for name in (
            "macro_f1",
            "qwk",
            "fatal_recall",
            "serious_recall",
            "minor_recall",
            "ordinal_mae",
            "mean_asymmetric_cost",
        )
    }
    write_json_atomic(
        SELECTED_FILE,
        {
            "version": VERSION,
            "status": "HYPERPARAMETERS_SELECTED_ON_2024_VALIDATION",
            "created_local": now_local(),
            "protocol": relative(CONFIG_FILE),
            "protocol_sha256": hash_file(CONFIG_FILE),
            "tuning_results": relative(TUNING_FILE),
            "tuning_results_sha256": hash_file(TUNING_FILE),
            "selection_constraints_met": constraints_met,
            "selected_candidate_id": selected["candidate_id"],
            "selected_complexity_rank": int(candidate["complexity_rank"]),
            "selected_candidate_parameters": candidate["parameters"],
            "selected_n_estimators": int(selected["best_iteration"]),
            "selected_validation_metrics": selected_metrics,
            "boundary_extension_triggered": boundary_extension,
            "selection_rule": config["selection_rule"],
            "test_data_used": False,
            "threads": THREADS,
        },
    )
    print(f"CAS_LIGHTGBM_SELECTED={selected['candidate_id']}")
    print(f"CAS_LIGHTGBM_SELECTED_TREES={int(selected['best_iteration'])}")
    print(f"CAS_LIGHTGBM_CONSTRAINTS_MET={constraints_met}")
    print("CAS_LIGHTGBM_TUNING_STATUS=SELECTED_WITHOUT_TEST_ACCESS")


def run_fit_all() -> None:
    config = verify_protocol()
    selected = json.loads(SELECTED_FILE.read_text(encoding="utf-8"))
    if selected.get("test_data_used") is not False:
        raise ValueError("CAS LightGBM selection is not test-independent")
    if selected["protocol_sha256"] != hash_file(CONFIG_FILE):
        raise ValueError("CAS selected LightGBM protocol hash differs")
    if selected["tuning_results_sha256"] != hash_file(TUNING_FILE):
        raise ValueError("CAS tuning results changed after selection")

    schema, analysis, training = load_contracts()
    features, target, metadata, assignments = load_aligned_data(schema, analysis)
    selected_candidate = {
        "candidate_id": selected["selected_candidate_id"],
        "complexity_rank": selected["selected_complexity_rank"],
        "parameters": selected["selected_candidate_parameters"],
    }
    n_estimators = int(selected["selected_n_estimators"])
    metric_rows: list[dict[str, object]] = []
    confusion_output: list[dict[str, object]] = []
    preprocessing_output: list[dict[str, object]] = []
    training_output: list[dict[str, object]] = []

    for spec in make_split_specs(analysis):
        role_column = spec["role_column"]
        train_positions = np.flatnonzero(
            assignments[role_column].eq("train").to_numpy()
        )
        validation_positions = np.flatnonzero(
            assignments[role_column].eq("validation").to_numpy()
        )
        train_years = sorted(
            set(assignments.iloc[train_positions]["meta_crash_year"].astype(int))
        )
        validation_years = sorted(
            set(
                assignments.iloc[validation_positions]["meta_crash_year"].astype(
                    int
                )
            )
        )
        if 2025 in train_years or 2025 in validation_years:
            raise AssertionError("CAS LightGBM fit-all attempted 2025 access")
        X_train, X_validation, vocabulary, unseen_counts = prepare_native_frames(
            schema, features, train_positions, validation_positions
        )
        y_train = target.iloc[train_positions].to_numpy(dtype=np.int8)
        y_validation = target.iloc[validation_positions].to_numpy(dtype=np.int8)
        class_weights = class_weights_for_spec(training, spec)
        estimator = make_estimator(
            config,
            selected_candidate,
            class_weights,
            n_estimators=n_estimators,
        )
        start = time.perf_counter()
        estimator.fit(
            X_train,
            y_train,
            categorical_feature=list(config["features"]["categorical"]),
            callbacks=[lgb.log_evaluation(period=0)],
        )
        fit_seconds = time.perf_counter() - start
        predicted, probabilities = predict_with_probabilities(
            estimator, X_validation
        )
        metrics = classification_metrics(y_validation, predicted)
        metric_rows.append(
            {
                "protocol": spec["protocol"],
                "seed": spec["seed"],
                "evaluation_role": "validation",
                "training_rows": len(train_positions),
                "validation_rows": len(validation_positions),
                "training_years": ";".join(map(str, train_years)),
                "validation_years": ";".join(map(str, validation_years)),
                "model": MODEL_NAME,
                "selected_candidate_id": selected["selected_candidate_id"],
                "n_estimators": n_estimators,
                "fit_seconds": fit_seconds,
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
                model=MODEL_NAME,
            )
        )
        write_dataframe_gzip(
            PREDICTION_DIR / f"{spec['split_slug']}__{MODEL_NAME}.csv.gz",
            prediction_frame(
                crash_ids=metadata.iloc[validation_positions]["meta_crash_id"],
                years=assignments.iloc[validation_positions]["meta_crash_year"],
                y_true=y_validation,
                y_pred=predicted,
                probabilities=probabilities,
            ),
        )

        categorical = list(schema["categorical_feature_columns"])
        numeric = list(schema["numeric_feature_columns"])
        for column in categorical:
            preprocessing_output.append(
                {
                    "protocol": spec["protocol"],
                    "seed": spec["seed"],
                    "feature": column,
                    "feature_type": "categorical_native",
                    "training_levels": len(vocabulary[column]),
                    "validation_unseen_count": unseen_counts[column],
                    "training_missing_count": 0,
                    "validation_missing_count": 0,
                    "value_applied_to_data": "training categories plus __UNSEEN__",
                }
            )
        for column in numeric:
            preprocessing_output.append(
                {
                    "protocol": spec["protocol"],
                    "seed": spec["seed"],
                    "feature": column,
                    "feature_type": "numeric_native",
                    "training_levels": "",
                    "validation_unseen_count": "",
                    "training_missing_count": int(
                        features.iloc[train_positions][column].isna().sum()
                    ),
                    "validation_missing_count": int(
                        features.iloc[validation_positions][column].isna().sum()
                    ),
                    "value_applied_to_data": "none; NaN retained",
                }
            )
        training_output.append(
            {
                "protocol": spec["protocol"],
                "seed": spec["seed"],
                "training_rows": len(train_positions),
                "validation_rows": len(validation_positions),
                "minor_training_rows": int(np.sum(y_train == 0)),
                "serious_training_rows": int(np.sum(y_train == 1)),
                "fatal_training_rows": int(np.sum(y_train == 2)),
                "minor_class_weight": class_weights[0],
                "serious_class_weight": class_weights[1],
                "fatal_class_weight": class_weights[2],
                "selected_candidate_id": selected["selected_candidate_id"],
                "n_estimators": n_estimators,
                "threads": THREADS,
                "fit_seconds": fit_seconds,
            }
        )

        split_dir = MODEL_DIR / spec["split_slug"]
        split_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "version": VERSION,
                "protocol": spec["protocol"],
                "seed": spec["seed"],
                "role_column": role_column,
                "feature_columns": list(schema["feature_columns"]),
                "categorical_columns": categorical,
                "numeric_columns": numeric,
                "category_vocabulary": vocabulary,
                "numeric_missing_policy": "native_nan",
                "training_rows": len(train_positions),
                "training_years": train_years,
                "validation_rows": len(validation_positions),
                "validation_years": validation_years,
                "protocol_sha256": hash_file(CONFIG_FILE),
            },
            split_dir / "preprocessing_bundle.joblib",
            compress=3,
        )
        joblib.dump(
            {
                "version": VERSION,
                "protocol": spec["protocol"],
                "seed": spec["seed"],
                "role_column": role_column,
                "model_name": MODEL_NAME,
                "estimator": estimator,
                "class_weight": class_weights,
                "selected_candidate_id": selected["selected_candidate_id"],
                "selected_candidate_parameters": selected[
                    "selected_candidate_parameters"
                ],
                "n_estimators": n_estimators,
                "prediction_rule": "argmax over class probabilities",
                "target_codes": list(TARGET_CODES),
                "protocol_sha256": hash_file(CONFIG_FILE),
                "selected_config_sha256": hash_file(SELECTED_FILE),
                "threads": THREADS,
            },
            split_dir / f"{MODEL_NAME}.joblib",
            compress=3,
        )
        print(
            f"{spec['split_slug']}: Macro-F1={metrics['macro_f1']:.5f}, "
            f"QWK={metrics['qwk']:.5f}, Fatal recall={metrics['fatal_recall']:.5f}",
            flush=True,
        )

    write_csv(METRICS_FILE, metric_rows)
    write_csv(CONFUSION_FILE, confusion_output)
    write_csv(PREPROCESS_AUDIT_FILE, preprocessing_output)
    write_csv(TRAINING_AUDIT_FILE, training_output)
    metrics = pd.DataFrame(metric_rows)
    temporal = metrics.loc[metrics["protocol"].eq("temporal")].iloc[0]
    for metric in ("macro_f1", "qwk", "fatal_recall"):
        if not np.isclose(
            float(temporal[metric]),
            float(selected["selected_validation_metrics"][metric]),
            rtol=0,
            atol=1e-12,
        ):
            raise AssertionError(
                f"CAS temporal refit differs from tuning: {metric}"
            )

    lines = [
        "# CAS LightGBM validation checkpoint",
        "",
        "Status: **PASS - LIGHTGBM_MODELS_FROZEN_BEFORE_2025_EVALUATION**",
        "",
        f"- Selected candidate: **{selected['selected_candidate_id']}**.",
        f"- Trees: **{n_estimators}**.",
        f"- Eligibility constraints met: **{selected['selection_constraints_met']}**.",
        f"- Boundary extension triggered: **{selected['boundary_extension_triggered']}**.",
        f"- Threads: **{THREADS}**.",
        "- No random internal-test or 2025 model performance was calculated.",
        "",
    ]
    CHECKPOINT_FILE.write_text("\n".join(lines), encoding="utf-8")

    model_paths = sorted(MODEL_DIR.rglob("*.joblib"))
    prediction_paths = sorted(PREDICTION_DIR.glob("*.csv.gz"))
    if len(model_paths) != 12 or len(prediction_paths) != 6:
        raise AssertionError("Unexpected CAS LightGBM artifact count")
    artifacts = [
        CONFIG_FILE,
        SELECTED_FILE,
        TUNING_FILE,
        METRICS_FILE,
        CONFUSION_FILE,
        PREPROCESS_AUDIT_FILE,
        TRAINING_AUDIT_FILE,
        CHECKPOINT_FILE,
        *model_paths,
        *prediction_paths,
    ]
    build_artifact_manifest(artifacts, MANIFEST_FILE)
    write_json_atomic(
        FROZEN_MODELS_FILE,
        {
            "version": VERSION,
            "status": "LIGHTGBM_MODELS_FROZEN_BEFORE_2025_EVALUATION",
            "created_local": now_local(),
            "protocol_sha256": hash_file(CONFIG_FILE),
            "selected_sha256": hash_file(SELECTED_FILE),
            "artifact_manifest_sha256": hash_file(MANIFEST_FILE),
            "model_files": [
                {"file": relative(path), "sha256": hash_file(path)}
                for path in model_paths
            ],
            "test_performance_used": False,
            "threads": THREADS,
        },
    )
    print(f"CAS_LIGHTGBM_VALIDATION_METRIC_ROWS={len(metric_rows)}")
    print("CAS_LIGHTGBM_STATUS=FROZEN_BEFORE_2025_EVALUATION")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--tune", action="store_true")
    mode.add_argument("--fit-all", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.freeze:
        freeze_protocol()
    elif arguments.smoke:
        run_smoke()
    elif arguments.tune:
        run_tuning()
    else:
        run_fit_all()
