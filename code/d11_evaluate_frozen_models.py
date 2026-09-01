"""One-time D11 evaluation of frozen STATS19 models."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "4"

import joblib
import lightgbm
import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
from scipy.special import expit
from sklearn.metrics import log_loss, precision_score

matplotlib.use("Agg")

from baseline_modeling import TARGET_CODES, classification_metrics, confusion_rows, prepare_features
from d10_tune_lightgbm import load_aligned_data, load_contracts

PROJECT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_DIR / "code"
CONFIG_DIR = PROJECT_DIR / "config"
RESULT_DIR = PROJECT_DIR / "results" / "d11"
STAGING_DIR = RESULT_DIR / ".staging"
PREDICTION_DIR = RESULT_DIR / "predictions"
LOG_DIR = PROJECT_DIR / "logs"
MODEL_DIR = PROJECT_DIR / "models"
FIGURE_DIR = PROJECT_DIR / "figures"

PROTOCOL_FILE = CONFIG_DIR / "d11_evaluation_protocol.json"
RUN_LOCK_FILE = LOG_DIR / "d11_run_lock.json"
INCIDENT_FILE = LOG_DIR / "d11_execution_incident.json"
METRICS_FILE = RESULT_DIR / "d11_test_metrics.csv"
CONFUSION_FILE = RESULT_DIR / "d11_test_confusion_matrices.csv"
AUDIT_FILE = LOG_DIR / "d11_preprocessing_audit.csv"
SUMMARY_FILE = LOG_DIR / "d11_run_summary.json"
MANIFEST_FILE = LOG_DIR / "d11_artifact_manifest.csv"
FIGURE_FILE = FIGURE_DIR / "d11_test_metric_overview.png"
TEST_SOURCE_FILE = CODE_DIR / "test_d11_evaluation.py"

DATA_FILE = PROJECT_DIR / "data" / "processed" / "stats19_modeling_dataset.csv.gz"
ASSIGNMENTS_FILE = PROJECT_DIR / "data" / "processed" / "d6_split_assignments.csv.gz"
SCHEMA_FILE = CONFIG_DIR / "d5_dataset_schema.json"
D6_FILE = CONFIG_DIR / "d6_analysis_protocol.json"
D7_FILE = CONFIG_DIR / "d7_training_inputs.json"
D8_FILE = CONFIG_DIR / "d8_baseline_protocol.json"
D9_FILE = CONFIG_DIR / "d9_ordered_logit_protocol.json"
D9S1_FILE = CONFIG_DIR / "d9s1_matched_subset_protocol.json"
D10_FILE = CONFIG_DIR / "d10_lightgbm_protocol.json"
D10_AMENDMENT_FILE = CONFIG_DIR / "d10_execution_amendment.json"
D10_SELECTED_FILE = CONFIG_DIR / "d10_selected_lightgbm.json"
D10B_FILE = CONFIG_DIR / "d10b_tree_sensitivity_protocol.json"
D10B_CHECKPOINT_FILE = LOG_DIR / "d10b_checkpoint.md"
D10B_METRICS_FILE = PROJECT_DIR / "results" / "d10b" / "d10b_checkpoint_metrics.csv"

D8_MODEL_DIR = MODEL_DIR / "d8"
D9_MODEL_DIR = MODEL_DIR / "d9"
D10_MODEL_DIR = MODEL_DIR / "d10"

VERSION = "D11_V1"
CREATED_LOCAL = "2026-08-30"
THREAD_LIMIT = 4
TEMPORAL_TEST_ROWS = 100_927
RANDOM_TEST_ROWS = 96_408
RANDOM_SEEDS = (1103, 2207, 3301, 4409, 5501)

MODEL_SPECS: tuple[dict[str, str], ...] = (
    {"model_id": "dummy_most_frequent", "family": "D8", "reporting_role": "primary_floor"},
    {"model_id": "logistic_unweighted", "family": "D8", "reporting_role": "secondary_linear_baseline"},
    {"model_id": "logistic_weighted", "family": "D8", "reporting_role": "primary_linear_comparator"},
    {"model_id": "ordered_logit_unweighted", "family": "D9", "reporting_role": "secondary_ordinal_baseline"},
    {"model_id": "lightgbm_weighted", "family": "D10", "reporting_role": "primary_nonlinear_comparator"},
)

CORE_METRICS = (
    "macro_f1", "qwk", "ordinal_mae", "accuracy", "slight_recall",
    "serious_recall", "fatal_recall", "serious_or_fatal_recall",
    "mean_asymmetric_cost",
)
PRECISION_METRICS = ("slight_precision", "serious_precision", "fatal_precision")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows generated for {path}")
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


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_DIR.resolve()).as_posix()


def package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "statsmodels": statsmodels.__version__,
        "platform": platform.platform(),
    }


def split_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "split_slug": "temporal",
            "protocol": "temporal",
            "seed": "year_based",
            "role_column": "temporal_role",
            "test_role": "test",
            "test_scope": "temporal_test_2024",
        }
    ]
    for seed in RANDOM_SEEDS:
        specs.append(
            {
                "split_slug": f"random_seed_{seed}",
                "protocol": "random_reference",
                "seed": str(seed),
                "role_column": f"random_role_seed_{seed}",
                "test_role": "test",
                "test_scope": "random_reference_test",
            }
        )
    return specs


def artifact_paths() -> list[Path]:
    paths: list[Path] = [
        DATA_FILE, ASSIGNMENTS_FILE, SCHEMA_FILE, D6_FILE, D7_FILE, D8_FILE,
        D9_FILE, D9S1_FILE, D10_FILE, D10_AMENDMENT_FILE, D10_SELECTED_FILE,
        D10B_FILE, D10B_CHECKPOINT_FILE, D10B_METRICS_FILE,
        CODE_DIR / "baseline_modeling.py", CODE_DIR / "modeling_data.py",
        CODE_DIR / "d8_train_baselines.py", CODE_DIR / "d9_train_ordered_logit.py",
        CODE_DIR / "d9s1_matched_subset_logistic.py", CODE_DIR / "d10_tune_lightgbm.py",
        CODE_DIR / "d10b_tree_sensitivity.py", CODE_DIR / "test_d10_lightgbm.py",
        CODE_DIR / "test_d10b_tree_sensitivity.py", Path(__file__),
        TEST_SOURCE_FILE, PROJECT_DIR / "requirements.txt",
    ]
    for spec in split_specs():
        slug = spec["split_slug"]
        for model in ("dummy_most_frequent", "logistic_unweighted", "logistic_weighted"):
            paths.extend(
                [
                    D8_MODEL_DIR / slug / "preprocessing_bundle.joblib",
                    D8_MODEL_DIR / slug / f"{model}.joblib",
                    PROJECT_DIR / "results" / "d8" / "validation_predictions"
                    / f"{slug}__{model}.csv.gz",
                ]
            )
        paths.extend(
            [
                D9_MODEL_DIR / slug / "preprocessing_bundle.joblib",
                D9_MODEL_DIR / slug / "ordered_logit_unweighted.joblib",
                PROJECT_DIR / "results" / "d9" / "validation_predictions"
                / f"{slug}__ordered_logit_unweighted.csv.gz",
                D10_MODEL_DIR / slug / "preprocessing_bundle.joblib",
                D10_MODEL_DIR / slug / "lightgbm_weighted.joblib",
                PROJECT_DIR / "results" / "d10" / "validation_predictions"
                / f"{slug}__lightgbm_weighted.csv.gz",
            ]
        )
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def build_protocol() -> dict[str, Any]:
    import json
    required = artifact_paths()
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("D11 required artifacts are missing: " + "; ".join(missing))
    d6 = json.loads(D6_FILE.read_text(encoding="utf-8"))
    d8 = json.loads(D8_FILE.read_text(encoding="utf-8"))
    d9 = json.loads(D9_FILE.read_text(encoding="utf-8"))
    d10 = json.loads(D10_FILE.read_text(encoding="utf-8"))
    selected = json.loads(D10_SELECTED_FILE.read_text(encoding="utf-8"))
    d10b = json.loads(D10B_FILE.read_text(encoding="utf-8"))
    if d6["version"] != "D6_V2":
        raise ValueError("D11 requires D6_V2")
    if (d8["version"], d9["version"], d10["version"]) != (
        "D8_V1", "D9_V3", "D10_V1"
    ):
        raise ValueError("D11 requires frozen D8, D9 and D10 protocols")
    if selected["selected_candidate_id"] != "C03" or int(
        selected["selected_n_estimators"]
    ) != 1200:
        raise ValueError("D11 requires frozen C03-1200")
    if d10b["decision_lock"]["D11_change_allowed_from_D10b"] is not False:
        raise ValueError("D10b must remain diagnostic-only")

    upstream = {relative(path): hash_file(path) for path in required}
    return {
        "version": VERSION,
        "status": "FROZEN_BEFORE_D11_TEST_ACCESS",
        "created_local": CREATED_LOCAL,
        "purpose": "One-time evaluation of frozen STATS19 models on frozen test partitions.",
        "scientific_lock": {
            "no_model_reselection": True,
            "no_threshold_tuning": True,
            "no_test_based_preprocessing_fit": True,
            "no_test_based_hyperparameter_tuning": True,
            "d10b_does_not_change_d11": True,
            "d9s1_excluded_from_d11": "appendix matched-subset sensitivity only",
        },
        "upstream_sha256": upstream,
        "data_and_splits": {
            "statistical_unit": "one police-reported personal-injury collision",
            "temporal_test": {
                "role_column": "temporal_role",
                "test_role": "test",
                "years": [2024],
                "rows": TEMPORAL_TEST_ROWS,
            },
            "random_reference_tests": {
                "role": "test",
                "years": [2018, 2019, 2020, 2021, 2022, 2023],
                "rows_per_seed": RANDOM_TEST_ROWS,
                "seeds": list(RANDOM_SEEDS),
            },
        },
        "models": list(MODEL_SPECS),
        "evaluation_matrix": {
            "main_rows": 30,
            "secondary_2024_diagnostic": {
                "enabled": True,
                "purpose": "Apply each already-fitted random model to 2024 without refitting or selection.",
                "rows": 25,
                "seeds": list(RANDOM_SEEDS),
            },
        },
        "metrics": {
            "classification": list(CORE_METRICS) + list(PRECISION_METRICS),
            "probability_metric": "multiclass_logloss",
            "confusion_matrix": "3x3; rows true and columns predicted",
            "uncertainty": "D12 uses saved row-level predictions; D11 selects no interval",
        },
        "prediction_rule": "argmax over Slight, Serious, Fatal probabilities",
        "preprocessing_rule": {
            "D8": "load training-fitted vocabulary and preprocessor; transform only",
            "D9": "load training-fitted vocabulary, preprocessor and QR mask; transform only",
            "D10": "load training-fitted native-category vocabulary; transform only",
            "all": "zero fit operations on test rows",
        },
        "runtime": {
            "native_thread_limit": THREAD_LIMIT,
            "package_versions_at_freeze": package_versions(),
        },
        "output_contract": {
            "metrics": relative(METRICS_FILE),
            "confusion": relative(CONFUSION_FILE),
            "predictions": relative(PREDICTION_DIR),
            "preprocessing_audit": relative(AUDIT_FILE),
            "summary": relative(SUMMARY_FILE),
            "manifest": relative(MANIFEST_FILE),
        },
    }


def freeze_protocol() -> None:
    import json
    payload = build_protocol()
    if PROTOCOL_FILE.exists():
        existing = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("Existing D11 protocol differs; refusing overwrite")
        print("D11 protocol already matches the specification.")
    else:
        write_json(PROTOCOL_FILE, payload)
        print("D11 protocol frozen before test access.")
    print("D11_PROTOCOL_SHA256=", hash_file(PROTOCOL_FILE))
    print("D11_TEST_ACCESS=SEALED")


def require_protocol() -> dict[str, Any]:
    import json
    if not PROTOCOL_FILE.exists():
        raise FileNotFoundError("Run --freeze before D11 operations")
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    if protocol.get("version") != VERSION or protocol.get(
        "status"
    ) != "FROZEN_BEFORE_D11_TEST_ACCESS":
        raise ValueError("Unexpected D11 protocol state")
    if build_protocol() != protocol:
        raise ValueError("D11 upstream artifacts changed after freeze")
    return protocol


def load_d11_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    schema, d6, _, _ = load_contracts()
    features, target, metadata, assignments = load_aligned_data(schema, d6)
    if len(features) != 743_646 or len(assignments) != len(features):
        raise AssertionError("Unexpected frozen table row count")
    if not np.array_equal(
        metadata["meta_collision_index"].astype("string").to_numpy(),
        assignments["meta_collision_index"].astype("string").to_numpy(),
    ):
        raise AssertionError("Collision IDs are not row-aligned")
    return features, target.astype("int8"), metadata, assignments


def positions_for_evaluation(
    spec: dict[str, Any], assignments: pd.DataFrame
) -> tuple[np.ndarray, set[int]]:
    positions = np.flatnonzero(
        assignments[spec["role_column"]].astype("string").eq(
            spec["test_role"]
        ).to_numpy()
    )
    years = set(assignments.iloc[positions]["meta_collision_year"].astype(int))
    if spec["protocol"] == "temporal":
        if len(positions) != TEMPORAL_TEST_ROWS or years != {2024}:
            raise AssertionError("Unexpected temporal test partition")
    elif (
        len(positions) != RANDOM_TEST_ROWS
        or 2024 in years
        or not years.issubset(set(range(2018, 2024)))
    ):
        raise AssertionError("Unexpected random test partition")
    return positions, years


def positions_for_2024(assignments: pd.DataFrame) -> np.ndarray:
    positions = np.flatnonzero(
        assignments["meta_collision_year"].astype(int).eq(2024).to_numpy()
    )
    if len(positions) != TEMPORAL_TEST_ROWS:
        raise AssertionError("Unexpected 2024 row count")
    return positions


def artifact_paths_for(
    family: str, split_slug: str, model_id: str
) -> tuple[Path, Path]:
    if family == "D8":
        return (
            D8_MODEL_DIR / split_slug / "preprocessing_bundle.joblib",
            D8_MODEL_DIR / split_slug / f"{model_id}.joblib",
        )
    if family == "D9":
        return (
            D9_MODEL_DIR / split_slug / "preprocessing_bundle.joblib",
            D9_MODEL_DIR / split_slug / f"{model_id}.joblib",
        )
    if family == "D10":
        return (
            D10_MODEL_DIR / split_slug / "preprocessing_bundle.joblib",
            D10_MODEL_DIR / split_slug / f"{model_id}.joblib",
        )
    raise ValueError(f"Unknown model family: {family}")


def validation_prediction_path(
    family: str, split_slug: str, model_id: str
) -> Path:
    family_dir = {"D8": "d8", "D9": "d9", "D10": "d10"}[family]
    return (
        PROJECT_DIR / "results" / family_dir / "validation_predictions"
        / f"{split_slug}__{model_id}.csv.gz"
    )


def validate_artifact_identity(
    family: str,
    model_id: str,
    split: dict[str, Any],
    bundle: dict[str, Any],
    artifact: dict[str, Any],
) -> None:
    if family == "D8":
        assert bundle["version"] == "D8_V1"
        assert artifact["version"] == "D8_V1"
        assert artifact["target_codes"] == list(TARGET_CODES)
    elif family == "D9":
        assert bundle["version"] == "D9_V3"
        assert bundle["training_mode"] == "subset"
        assert bundle["used_training_rows"] == 100_000
        assert artifact["version"] == "D9_V3"
        assert artifact["training_mode"] == "subset"
        assert artifact["used_training_rows"] == 100_000
        assert artifact["class_weight"] is None
        assert artifact["converged"] is True
    elif family == "D10":
        assert bundle["version"] == "D10_V1"
        assert artifact["version"] == "D10_V1"
        assert artifact["selected_candidate_id"] == "C03"
        assert artifact["n_estimators"] == 1200
        assert artifact["target_codes"] == list(TARGET_CODES)
        assert artifact["estimator"].get_params()["n_jobs"] == THREAD_LIMIT
    else:
        raise ValueError(f"Unknown family: {family}")
    assert bundle["protocol"] == split["protocol"]
    assert str(bundle["seed"]) == str(split["seed"])
    if family != "D9":
        assert artifact["protocol"] == split["protocol"]
        assert str(artifact["seed"]) == str(split["seed"])
    assert artifact["model_name"] == model_id
    assert 2024 not in bundle.get("training_years", [])


def prepare_saved_features(
    family: str, X: pd.DataFrame, bundle: dict[str, Any]
) -> tuple[Any, dict[str, int]]:
    prepared = prepare_features(
        X,
        feature_columns=list(bundle["feature_columns"]),
        categorical_columns=list(bundle["categorical_columns"]),
        numeric_columns=list(bundle["numeric_columns"]),
        category_vocabulary=bundle["category_vocabulary"],
    )
    if family in {"D8", "D9"}:
        preprocessor = bundle["preprocessor"]
        if not hasattr(preprocessor, "transformers_"):
            raise AssertionError("Saved preprocessor is not fitted")
        encoded = preprocessor.transform(prepared.frame)
        if family == "D8":
            if encoded.shape[1] != len(bundle["encoded_feature_names"]):
                raise AssertionError("D8 encoded width changed")
            return encoded, prepared.unseen_counts
        mask = np.asarray(bundle["identified_column_mask"], dtype=bool)
        independent = np.asarray(bundle["independent_column_indices"], dtype=int)
        if encoded.shape[1] != len(mask):
            raise AssertionError("D9 mask width changed")
        dense = encoded[:, mask].toarray()[:, independent]
        if dense.shape[1] != len(bundle["retained_feature_names"]):
            raise AssertionError("D9 retained width changed")
        return dense, prepared.unseen_counts
    if family == "D10":
        return prepared.frame, prepared.unseen_counts
    raise ValueError(f"Unknown family: {family}")


def ordered_probabilities(
    encoded: np.ndarray, artifact: dict[str, Any]
) -> np.ndarray:
    beta = np.asarray(artifact["beta"], dtype=float)
    thresholds = np.asarray(artifact["finite_thresholds"], dtype=float)
    if encoded.shape[1] != len(beta) or len(thresholds) != 2:
        raise AssertionError("D9 dimensions changed")
    cumulative = expit(thresholds[None, :] - (encoded @ beta)[:, None])
    probabilities = np.column_stack(
        [cumulative[:, 0], cumulative[:, 1] - cumulative[:, 0], 1 - cumulative[:, 1]]
    )
    if probabilities.min() < -1e-12:
        raise AssertionError("D9 negative probability")
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def predict_saved_model(
    family: str, artifact: dict[str, Any], prepared: Any
) -> tuple[np.ndarray, np.ndarray]:
    if family == "D9":
        probabilities = ordered_probabilities(prepared, artifact)
        return np.argmax(probabilities, axis=1).astype(np.int8), probabilities
    estimator = artifact["estimator"]
    probabilities = np.asarray(estimator.predict_proba(prepared), dtype=float)
    classes = np.asarray(estimator.classes_, dtype=int)
    if classes.tolist() != list(TARGET_CODES):
        raise AssertionError("Class order changed")
    return classes[np.argmax(probabilities, axis=1)].astype(np.int8), probabilities


def validate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray
) -> None:
    if probabilities.shape != (len(y_true), 3) or len(y_pred) != len(y_true):
        raise AssertionError("Prediction shape changed")
    if not np.isfinite(probabilities).all() or probabilities.min() < -1e-10:
        raise AssertionError("Invalid probability")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=2e-8):
        raise AssertionError("Probabilities do not sum to one")
    if set(np.unique(y_true)) != set(TARGET_CODES):
        raise AssertionError("Evaluation cohort lacks a true class")


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    metrics = classification_metrics(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, labels=list(TARGET_CODES), average=None, zero_division=0
    )
    metrics.update(
        {
            "slight_precision": float(precision[0]),
            "serious_precision": float(precision[1]),
            "fatal_precision": float(precision[2]),
            "multiclass_logloss": float(
                log_loss(y_true, probabilities, labels=list(TARGET_CODES))
            ),
        }
    )
    return metrics


def evaluate_one(
    *,
    model_spec: dict[str, str],
    split: dict[str, Any],
    positions: np.ndarray,
    evaluation_scope: str,
    features: pd.DataFrame,
    target: pd.Series,
    metadata: pd.DataFrame,
    assignments: pd.DataFrame,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    model_id = model_spec["model_id"]
    family = model_spec["family"]
    bundle_path, model_path = artifact_paths_for(
        family, split["split_slug"], model_id
    )
    bundle = joblib.load(bundle_path)
    artifact = joblib.load(model_path)
    validate_artifact_identity(family, model_id, split, bundle, artifact)

    X = features.iloc[positions]
    y_true = target.iloc[positions].to_numpy(dtype=np.int8)
    started = time.perf_counter()
    prepared, unseen = prepare_saved_features(family, X, bundle)
    transform_seconds = time.perf_counter() - started
    started = time.perf_counter()
    y_pred, probabilities = predict_saved_model(family, artifact, prepared)
    prediction_seconds = time.perf_counter() - started
    validate_predictions(y_true, y_pred, probabilities)
    metrics = calculate_metrics(y_true, y_pred, probabilities)

    ids = metadata.iloc[positions]["meta_collision_index"].astype("string").to_numpy()
    years = assignments.iloc[positions]["meta_collision_year"].astype(int).to_numpy()
    prediction = pd.DataFrame(
        {
            "meta_collision_index": ids,
            "meta_collision_year": years,
            "evaluation_scope": evaluation_scope,
            "split_slug": split["split_slug"],
            "model": model_id,
            "target_severity": y_true,
            "predicted_severity": y_pred,
            "prob_slight": probabilities[:, 0],
            "prob_serious": probabilities[:, 1],
            "prob_fatal": probabilities[:, 2],
        }
    )
    suffix = "2024_diagnostic" if evaluation_scope == "random_model_on_2024" else "test"
    save_prediction(
        output_dir / f"{split['split_slug']}__{model_id}__{suffix}.csv.gz",
        prediction,
    )

    confusion = confusion_rows(
        y_true,
        y_pred,
        protocol=evaluation_scope,
        seed=split["seed"],
        model=model_id,
    )
    for item in confusion:
        item["split_slug"] = split["split_slug"]
        item["reporting_role"] = model_spec["reporting_role"]
    metric_row: dict[str, Any] = {
        "evaluation_scope": evaluation_scope,
        "split_slug": split["split_slug"],
        "protocol": split["protocol"],
        "seed": split["seed"],
        "model": model_id,
        "family": family,
        "reporting_role": model_spec["reporting_role"],
        "evaluation_role": "test",
        "test_rows": len(positions),
        "test_years": ";".join(str(year) for year in sorted(set(years))),
        "training_mode": bundle.get("training_mode", "full"),
        "training_rows": bundle.get(
            "used_training_rows", bundle.get("training_rows", "")
        ),
        "transform_seconds": transform_seconds,
        "prediction_seconds": prediction_seconds,
        "unseen_categorical_values": int(sum(unseen.values())),
        **metrics,
    }
    audit = {
        "evaluation_scope": evaluation_scope,
        "split_slug": split["split_slug"],
        "protocol": split["protocol"],
        "seed": split["seed"],
        "model": model_id,
        "family": family,
        "reporting_role": model_spec["reporting_role"],
        "test_rows": len(positions),
        "transform_seconds": transform_seconds,
        "prediction_seconds": prediction_seconds,
        "unseen_total": int(sum(unseen.values())),
        "unseen_by_feature": __import__("json").dumps(
            unseen, ensure_ascii=True, sort_keys=True
        ),
        "preprocessor_action": "transform_only",
        "preprocessor_fit_operations": 0,
    }
    del bundle, artifact, prepared, prediction
    gc.collect()
    return metric_row, confusion, audit


def preflight_validation_only(require_frozen: bool = True) -> None:
    if require_frozen:
        require_protocol()
    features, target, metadata, assignments = load_d11_data()
    checks = 0
    for split in split_specs():
        positions = np.flatnonzero(
            assignments[split["role_column"]].astype("string").eq(
                "validation"
            ).to_numpy()
        )
        if split["protocol"] == "temporal":
            assert len(positions) == 104_258
        else:
            assert len(positions) == RANDOM_TEST_ROWS
        sample_parts = []
        partition_target = target.iloc[positions].to_numpy(dtype=np.int8)
        for target_code in TARGET_CODES:
            class_offsets = np.flatnonzero(partition_target == target_code)
            if not len(class_offsets):
                raise AssertionError("Validation preflight lacks a target class")
            sample_parts.append(positions[class_offsets[:64]])
        sample = np.sort(np.concatenate(sample_parts))
        for model_spec in MODEL_SPECS:
            family = model_spec["family"]
            model_id = model_spec["model_id"]
            bundle_path, model_path = artifact_paths_for(
                family, split["split_slug"], model_id
            )
            bundle = joblib.load(bundle_path)
            artifact = joblib.load(model_path)
            validate_artifact_identity(family, model_id, split, bundle, artifact)
            prepared, _ = prepare_saved_features(
                family, features.iloc[sample], bundle
            )
            predicted, probabilities = predict_saved_model(
                family, artifact, prepared
            )
            validate_predictions(
                target.iloc[sample].to_numpy(dtype=np.int8),
                predicted,
                probabilities,
            )
            frozen = pd.read_csv(
                validation_prediction_path(family, split["split_slug"], model_id),
                dtype={"meta_collision_index": "string"},
            ).set_index("meta_collision_index")
            sample_ids = metadata.iloc[sample]["meta_collision_index"].astype("string")
            expected = frozen.loc[sample_ids]
            if not np.array_equal(
                predicted, expected["predicted_severity"].to_numpy(dtype=np.int8)
            ):
                raise AssertionError("D11 preflight labels differ from frozen validation")
            expected_probabilities = expected[
                ["prob_slight", "prob_serious", "prob_fatal"]
            ].to_numpy(dtype=float)
            if not np.allclose(
                probabilities, expected_probabilities, rtol=0, atol=2e-12
            ):
                raise AssertionError(
                    "D11 preflight probabilities differ from frozen validation"
                )
            checks += 1
    print(f"D11_PREFLIGHT_VALIDATION_MODEL_CHECKS={checks}")
    print("D11_TEST_ACCESS=SEALED")
    print("D11_PREFLIGHT=PASS")


def save_overview_figure(metrics: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    order = [spec["model_id"] for spec in MODEL_SPECS]
    labels = ["Dummy", "Logistic\n(unweighted)", "Logistic\n(weighted)",
              "Ordered\nLogit", "LightGBM\n(C03-1200)"]
    temporal = metrics.loc[
        metrics["evaluation_scope"].eq("temporal_test_2024")
    ].set_index("model")
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
    x = np.arange(len(order))
    for axis, metric, title in (
        (axes[0], "macro_f1", "Macro-F1"),
        (axes[1], "fatal_recall", "Fatal recall"),
    ):
        values = [float(temporal.loc[model, metric]) for model in order]
        bars = axis.bar(
            x, values,
            color=["#8C9AA5", "#4C78A8", "#1F77B4", "#F2A541", "#2A9D8F"],
        )
        axis.set_xticks(x, labels)
        axis.set_ylabel(title)
        axis.set_ylim(0, max(1.0, max(values) * 1.18))
        axis.grid(axis="y", color="#D9DEE3", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                fontsize=8.5,
            )
    figure.suptitle("D11 temporal test results (2024)", fontsize=13)
    FIGURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_FILE, dpi=300, facecolor="white")
    plt.close(figure)


def build_manifest(extra_paths: Iterable[Path] = ()) -> None:
    paths = [
        PROTOCOL_FILE, METRICS_FILE, CONFUSION_FILE, AUDIT_FILE, SUMMARY_FILE,
        FIGURE_FILE, Path(__file__), TEST_SOURCE_FILE, RUN_LOCK_FILE,
        *sorted(PREDICTION_DIR.glob("*.csv.gz")), *extra_paths,
    ]
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = raw_path.resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        rows.append(
            {
                "relative_path": relative(path),
                "bytes": path.stat().st_size,
                "sha256": hash_file(path),
            }
        )
    write_csv(MANIFEST_FILE, sorted(rows, key=lambda row: row["relative_path"]))


def publish_staged_outputs(
    staged_metrics: Path,
    staged_confusion: Path,
    staged_audit: Path,
    staged_predictions: Path,
) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if METRICS_FILE.exists() or CONFUSION_FILE.exists() or PREDICTION_DIR.exists():
        raise FileExistsError("D11 outputs already exist")
    staged_metrics.replace(METRICS_FILE)
    staged_confusion.replace(CONFUSION_FILE)
    staged_audit.replace(AUDIT_FILE)
    staged_predictions.replace(PREDICTION_DIR)


def run_once() -> None:
    require_protocol()
    if RUN_LOCK_FILE.exists():
        raise FileExistsError("D11 run lock exists; refusing a second evaluation")
    if METRICS_FILE.exists() or CONFUSION_FILE.exists() or PREDICTION_DIR.exists():
        raise FileExistsError("D11 output exists without a run lock")
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    staging_predictions = STAGING_DIR / "predictions"
    staging_predictions.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    try:
        features, target, metadata, assignments = load_d11_data()
        metric_rows: list[dict[str, Any]] = []
        confusion_rows_output: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        completed = 0
        positions_2024 = positions_for_2024(assignments)

        for split in split_specs():
            positions, _ = positions_for_evaluation(split, assignments)
            for model_spec in MODEL_SPECS:
                row, confusion, audit = evaluate_one(
                    model_spec=model_spec,
                    split=split,
                    positions=positions,
                    evaluation_scope=split["test_scope"],
                    features=features,
                    target=target,
                    metadata=metadata,
                    assignments=assignments,
                    output_dir=staging_predictions,
                )
                metric_rows.append(row)
                confusion_rows_output.extend(confusion)
                audit_rows.append(audit)
                completed += 1
                print(
                    f"D11_MAIN_COMPLETED={completed}/30 "
                    f"split={split['split_slug']} model={model_spec['model_id']}",
                    flush=True,
                )

        # D6-pre-registered diagnostic; never used for model selection.
        for split in split_specs()[1:]:
            for model_spec in MODEL_SPECS:
                row, confusion, audit = evaluate_one(
                    model_spec=model_spec,
                    split=split,
                    positions=positions_2024,
                    evaluation_scope="random_model_on_2024",
                    features=features,
                    target=target,
                    metadata=metadata,
                    assignments=assignments,
                    output_dir=staging_predictions,
                )
                metric_rows.append(row)
                confusion_rows_output.extend(confusion)
                audit_rows.append(audit)
                completed += 1
                print(
                    f"D11_2024_DIAGNOSTIC_COMPLETED={completed}/55 "
                    f"split={split['split_slug']} model={model_spec['model_id']}",
                    flush=True,
                )

        metrics = pd.DataFrame(metric_rows)
        if (len(metrics), len(confusion_rows_output), len(audit_rows)) != (
            55, 55 * 9, 55
        ):
            raise AssertionError("D11 evaluation matrix is incomplete")
        if set(
            metrics.loc[
                metrics["evaluation_scope"].eq("random_model_on_2024"), "test_years"
            ]
        ) != {"2024"}:
            raise AssertionError("Secondary diagnostic years changed")

        staged_metrics = STAGING_DIR / "d11_test_metrics.csv"
        staged_confusion = STAGING_DIR / "d11_test_confusion_matrices.csv"
        staged_audit = STAGING_DIR / "d11_preprocessing_audit.csv"
        write_csv(staged_metrics, metric_rows)
        write_csv(staged_confusion, confusion_rows_output)
        write_csv(staged_audit, audit_rows)
        publish_staged_outputs(
            staged_metrics, staged_confusion, staged_audit, staging_predictions
        )

        summary = {
            "version": VERSION,
            "status": "PASS_ONE_TIME_TEST_EVALUATION_COMPLETED",
            "protocol_sha256": hash_file(PROTOCOL_FILE),
            "completed_local": CREATED_LOCAL,
            "main_evaluations": 30,
            "secondary_2024_diagnostics": 25,
            "prediction_files": len(list(PREDICTION_DIR.glob("*.csv.gz"))),
            "prediction_rows_main": int(
                metrics.loc[
                    metrics["evaluation_scope"] != "random_model_on_2024",
                    "test_rows",
                ].sum()
            ),
            "prediction_rows_secondary_2024": int(
                metrics.loc[
                    metrics["evaluation_scope"].eq("random_model_on_2024"),
                    "test_rows",
                ].sum()
            ),
            "fit_operations": 0,
            "preprocessor_fit_operations": 0,
            "test_based_selection_operations": 0,
            "runtime_seconds": time.perf_counter() - started,
            "D11_primary_model_lock": "D10 C03-1200",
            "D12_input": "results/d11/predictions",
        }
        write_json(SUMMARY_FILE, summary)
        write_json(
            RUN_LOCK_FILE,
            {
                "version": "D11_RUN_LOCK_V1",
                "status": "SEALED_AFTER_SUCCESSFUL_ONE_TIME_EVALUATION",
                "protocol_sha256": hash_file(PROTOCOL_FILE),
                "metrics_sha256": hash_file(METRICS_FILE),
                "confusion_sha256": hash_file(CONFUSION_FILE),
                "prediction_file_count": len(list(PREDICTION_DIR.glob("*.csv.gz"))),
                "created_local": CREATED_LOCAL,
                "rerun_policy": "No second test evaluation under D11_V1.",
            },
        )
        save_overview_figure(metrics)
        build_manifest()
        shutil.rmtree(STAGING_DIR, ignore_errors=True)
        print("D11_ONE_TIME_EVALUATION=PASS")
        print(f"D11_RUNTIME_SECONDS={summary['runtime_seconds']:.2f}")
        print("D11_TEST_ACCESS=SEALED_AFTER_SUCCESS")
    except Exception as error:
        write_json(
            INCIDENT_FILE,
            {
                "version": "D11_EXECUTION_INCIDENT_V1",
                "created_local": CREATED_LOCAL,
                "outcome": "FAILED_BEFORE_PUBLICATION",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "scientific_parameters_changed": False,
                "published_primary_outputs": False,
            },
        )
        shutil.rmtree(STAGING_DIR, ignore_errors=True)
        raise
    finally:
        gc.collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze", action="store_true")
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.freeze:
        freeze_protocol()
    elif args.preflight:
        preflight_validation_only()
    else:
        run_once()
