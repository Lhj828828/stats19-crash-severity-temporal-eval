"""D14 appendix sensitivity analysis excluding 2020 from temporal training."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import time
import warnings
from pathlib import Path
from typing import Any

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "8"

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning

from baseline_modeling import (
    ASYMMETRIC_COST_MATRIX,
    TARGET_CODES,
    assert_encoded_matrix,
    confusion_rows,
    fit_category_vocabulary,
    make_preprocessor,
    predict_with_probabilities,
    prepare_features,
)
from d8_train_baselines import make_logistic, prediction_frame
from d10_tune_lightgbm import load_aligned_data, load_contracts
from d11_evaluate_frozen_models import calculate_metrics


PROJECT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_DIR / "code"
CONFIG_DIR = PROJECT_DIR / "config"
RESULT_DIR = PROJECT_DIR / "results" / "d14" / "exclude2020"
STAGING_DIR = RESULT_DIR / ".staging"
PREDICTION_DIR = RESULT_DIR / "predictions"
MODEL_DIR = PROJECT_DIR / "models" / "d14_exclude2020"
LOG_DIR = PROJECT_DIR / "logs"

OUTLINE_FILE = PROJECT_DIR.parent / "PeerJ_Computer_Science_交通事故严重度预测论文大纲_执行质量完善版.docx"
DATA_FILE = PROJECT_DIR / "data" / "processed" / "stats19_modeling_dataset.csv.gz"
ASSIGNMENTS_FILE = PROJECT_DIR / "data" / "processed" / "d6_split_assignments.csv.gz"
SCHEMA_FILE = CONFIG_DIR / "d5_dataset_schema.json"
D6_FILE = CONFIG_DIR / "d6_analysis_protocol.json"
D10_PROTOCOL_FILE = CONFIG_DIR / "d10_lightgbm_protocol.json"
D10_SELECTED_FILE = CONFIG_DIR / "d10_selected_lightgbm.json"
D11_PROTOCOL_FILE = CONFIG_DIR / "d11_evaluation_protocol.json"
D11_MANIFEST_FILE = LOG_DIR / "d11_artifact_manifest.csv"
D14_SHAP_PROTOCOL_FILE = CONFIG_DIR / "d14_shap_protocol.json"
D14_SHAP_MANIFEST_FILE = LOG_DIR / "d14_shap_artifact_manifest.csv"

PROTOCOL_FILE = CONFIG_DIR / "d14_exclude2020_protocol_v2.json"
PREVIOUS_PROTOCOL_FILE = CONFIG_DIR / "d14_exclude2020_protocol.json"
VALIDATION_METRICS_FILE = RESULT_DIR / "d14_exclude2020_validation_metrics.csv"
TEST_METRICS_FILE = RESULT_DIR / "d14_exclude2020_test_metrics.csv"
CONFUSION_FILE = RESULT_DIR / "d14_exclude2020_confusion_matrices.csv"
COMPARISON_FILE = RESULT_DIR / "d14_exclude2020_comparisons.csv"
BOOTSTRAP_FILE = RESULT_DIR / "d14_exclude2020_bootstrap_draws.npz"
PREPROCESS_AUDIT_FILE = LOG_DIR / "d14_exclude2020_preprocessing_audit.csv"
SUMMARY_FILE = LOG_DIR / "d14_exclude2020_run_summary.json"
CHECKPOINT_FILE = LOG_DIR / "d14_exclude2020_checkpoint.md"
MANIFEST_FILE = LOG_DIR / "d14_exclude2020_artifact_manifest.csv"
TEST_SOURCE_FILE = CODE_DIR / "test_d14_exclude2020.py"

D11_PREDICTION_DIR = PROJECT_DIR / "results" / "d11" / "predictions"
MAIN_PREDICTION_FILES = {
    "main_logistic_weighted": D11_PREDICTION_DIR / "temporal__logistic_weighted__test.csv.gz",
    "main_lightgbm_weighted": D11_PREDICTION_DIR / "temporal__lightgbm_weighted__test.csv.gz",
}

VERSION = "D14_EXCLUDE2020_V2"
CREATED_LOCAL = "2026-08-31"
THREAD_LIMIT = 8
TRAIN_YEARS = (2018, 2019, 2021, 2022)
EXCLUDED_TRAIN_YEAR = 2020
VALIDATION_YEAR = 2023
TEST_YEAR = 2024
EXPECTED_TRAIN_ROWS = 447_262
EXPECTED_VALIDATION_ROWS = 104_258
EXPECTED_TEST_ROWS = 100_927
BOOTSTRAP_ITERATIONS = 2_000
BOOTSTRAP_SEED = 20_260_828
CI_ALPHA = 0.05
MODEL_ORDER = (
    "main_logistic_weighted",
    "main_lightgbm_weighted",
    "exclude2020_logistic_weighted",
    "exclude2020_lightgbm_weighted",
)
METRICS = (
    "macro_f1",
    "qwk",
    "ordinal_mae",
    "accuracy",
    "fatal_recall",
    "serious_or_fatal_recall",
    "mean_asymmetric_cost",
)
HIGHER_IS_BETTER = {
    "macro_f1",
    "qwk",
    "accuracy",
    "fatal_recall",
    "serious_or_fatal_recall",
}
LOWER_IS_BETTER = {"ordinal_mae", "mean_asymmetric_cost"}


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_DIR.resolve()).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError(f"No rows generated for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", lineterminator="\n")
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


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lgb.__version__,
        "platform": platform.platform(),
    }


def upstream_paths() -> list[Path]:
    return [
        OUTLINE_FILE,
        DATA_FILE,
        ASSIGNMENTS_FILE,
        SCHEMA_FILE,
        D6_FILE,
        D10_PROTOCOL_FILE,
        D10_SELECTED_FILE,
        D11_PROTOCOL_FILE,
        D11_MANIFEST_FILE,
        D14_SHAP_PROTOCOL_FILE,
        D14_SHAP_MANIFEST_FILE,
        PREVIOUS_PROTOCOL_FILE,
        *MAIN_PREDICTION_FILES.values(),
        CODE_DIR / "baseline_modeling.py",
        CODE_DIR / "d8_train_baselines.py",
        CODE_DIR / "d10_tune_lightgbm.py",
        CODE_DIR / "d11_evaluate_frozen_models.py",
        Path(__file__),
        TEST_SOURCE_FILE,
        PROJECT_DIR / "requirements.txt",
    ]


def build_protocol() -> dict[str, Any]:
    required = upstream_paths()
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing exclusion-2020 inputs: " + "; ".join(missing))
    d6 = read_json(D6_FILE)
    d10 = read_json(D10_PROTOCOL_FILE)
    selected = read_json(D10_SELECTED_FILE)
    shap_protocol = read_json(D14_SHAP_PROTOCOL_FILE)
    if d6.get("version") != "D6_V2":
        raise ValueError("Sensitivity requires frozen D6_V2")
    if selected.get("selected_candidate_id") != "C03" or int(
        selected.get("selected_n_estimators", -1)
    ) != 1200:
        raise ValueError("Sensitivity requires frozen C03-1200")
    if shap_protocol.get("version") != "D14_SHAP_V1":
        raise ValueError("Sensitivity follows completed D14 SHAP analysis")
    candidate = next(
        item for item in d10["tuning"]["candidates"] if item["candidate_id"] == "C03"
    )
    if candidate["parameters"] != selected["selected_candidate_parameters"]:
        raise ValueError("C03 parameters changed")
    return {
        "version": VERSION,
        "revision": {
            "supersedes_protocol": relative(PREVIOUS_PROTOCOL_FILE),
            "supersedes_protocol_sha256": hash_file(PREVIOUS_PROTOCOL_FILE),
            "reason": (
                "Corrected a non-scientific output-serialization bug found during the first "
                "attempt; no cohort, feature, model or statistical rule changed."
            ),
        },
        "status": "FROZEN_BEFORE_EXCLUDE2020_MODEL_FITTING",
        "created_local": CREATED_LOCAL,
        "timing_disclosure": (
            "Excluding 2020 was specified in the working outline before this run, but was "
            "not externally preregistered. The exact implementation, model set and paired "
            "bootstrap comparisons were frozen after D11-D14 main results were known and "
            "before fitting any exclusion-2020 model."
        ),
        "scientific_scope": {
            "role": "appendix sensitivity analysis",
            "question": (
                "Are strict-temporal 2024 estimates and the primary LightGBM-versus-Logistic "
                "comparison sensitive to retaining 2020 in temporal training?"
            ),
            "not_answered": (
                "This analysis does not identify a causal pandemic effect and cannot determine "
                "whether 2020 drives H1 random-internal-versus-2024 gaps, because random models "
                "are not refitted here."
            ),
            "no_model_selection": True,
            "no_retuning": True,
            "no_threshold_change": True,
        },
        "upstream_sha256": {str(path.resolve()): hash_file(path) for path in required},
        "split": {
            "training_years": list(TRAIN_YEARS),
            "excluded_training_year": EXCLUDED_TRAIN_YEAR,
            "validation_years": [VALIDATION_YEAR],
            "test_years": [TEST_YEAR],
            "expected_rows": {
                "train": EXPECTED_TRAIN_ROWS,
                "validation": EXPECTED_VALIDATION_ROWS,
                "test": EXPECTED_TEST_ROWS,
            },
            "test_rule": "same frozen 2024 records and order as the main temporal evaluation",
        },
        "features_and_preprocessing": {
            "allowlist": d6["frozen_schema"]["feature_columns"],
            "fit_scope": "exclude-2020 training rows only",
            "categorical": "training-fitted vocabulary with __UNSEEN__ transform-only mapping",
            "logistic": "training-fitted one-hot encoding; speed-limit median and scaling",
            "lightgbm": "native categorical features; speed-limit NaN retained",
            "class_weights": "recomputed only from the exclusion-2020 training class counts",
        },
        "models": {
            "logistic_weighted": {
                "parameters": {
                    "penalty": "l2",
                    "C": 1.0,
                    "solver": "lbfgs",
                    "max_iter": 500,
                    "tol": 1e-4,
                    "random_state": 20260828,
                },
                "role": "primary linear comparator",
            },
            "lightgbm_weighted": {
                "selected_candidate": "C03",
                "n_estimators": 1200,
                "parameters": candidate["parameters"],
                "common_parameters": {
                    **d10["common_model_parameters"],
                    "n_jobs": THREAD_LIMIT,
                },
                "role": "primary nonlinear comparator",
            },
        },
        "evaluation": {
            "prediction_rule": "argmax over Slight, Serious and Fatal probabilities",
            "validation": "descriptive pipeline audit only; no selection",
            "test_metrics": list(METRICS),
            "comparisons": [
                "exclude2020 LightGBM versus exclude2020 Logistic on 2024",
                "exclude2020 versus main model for each primary model on 2024",
                "change in the LightGBM-versus-Logistic contrast relative to the main analysis",
            ],
        },
        "bootstrap": {
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "resampling": (
                "paired nonparametric bootstrap over common 2024 collisions, stratified by "
                "true severity; implemented by multinomial resampling of joint prediction patterns"
            ),
            "confidence_level": 0.95,
            "interval_method": "percentile",
            "multiplicity_adjustment": "none; descriptive sensitivity intervals",
        },
        "runtime": {
            "CPU_thread_limit": THREAD_LIMIT,
            "GPU_used": False,
            "package_versions_at_freeze": package_versions(),
        },
    }


def freeze_protocol() -> None:
    payload = build_protocol()
    if PROTOCOL_FILE.exists():
        existing = read_json(PROTOCOL_FILE)
        if existing != payload:
            raise RuntimeError("Existing exclusion-2020 protocol differs; refusing overwrite")
        print("Exclusion-2020 protocol already matches the frozen specification.", flush=True)
    else:
        write_json(PROTOCOL_FILE, payload)
        print("Exclusion-2020 protocol frozen before model fitting.", flush=True)
    print("D14_EXCLUDE2020_PROTOCOL_SHA256=", hash_file(PROTOCOL_FILE), flush=True)


def require_protocol() -> dict[str, Any]:
    if not PROTOCOL_FILE.is_file():
        raise FileNotFoundError("Run --freeze before exclusion-2020 analysis")
    protocol = read_json(PROTOCOL_FILE)
    if protocol.get("version") != VERSION or protocol.get("status") != (
        "FROZEN_BEFORE_EXCLUDE2020_MODEL_FITTING"
    ):
        raise ValueError("Unexpected exclusion-2020 protocol state")
    if build_protocol() != protocol:
        raise ValueError("Exclusion-2020 inputs or implementation changed after freeze")
    return protocol


def balanced_class_weights(y: np.ndarray) -> dict[int, float]:
    y = np.asarray(y, dtype=np.int8)
    counts = np.bincount(y, minlength=len(TARGET_CODES))
    if len(counts) != len(TARGET_CODES) or np.any(counts == 0):
        raise ValueError("Training labels lack a severity class")
    return {
        code: float(len(y) / (len(TARGET_CODES) * counts[code]))
        for code in TARGET_CODES
    }


def make_lightgbm(
    d10: dict[str, Any], selected: dict[str, Any], weights: dict[int, float]
) -> lgb.LGBMClassifier:
    common = dict(d10["common_model_parameters"])
    common["n_jobs"] = THREAD_LIMIT
    return lgb.LGBMClassifier(
        **common,
        **selected["selected_candidate_parameters"],
        n_estimators=int(selected["selected_n_estimators"]),
        class_weight=weights,
        bagging_seed=20260830,
        feature_fraction_seed=20260830,
        data_random_seed=20260830,
        importance_type="gain",
    )


def evaluation_positions(assignments: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    years = assignments["meta_collision_year"].astype(int).to_numpy()
    role = assignments["temporal_role"].astype("string")
    train = np.flatnonzero(role.eq("train").to_numpy() & (years != EXCLUDED_TRAIN_YEAR))
    validation = np.flatnonzero(role.eq("validation").to_numpy())
    test = np.flatnonzero(role.eq("test").to_numpy())
    if len(train) != EXPECTED_TRAIN_ROWS or set(years[train]) != set(TRAIN_YEARS):
        raise AssertionError("Unexpected exclusion-2020 training cohort")
    if len(validation) != EXPECTED_VALIDATION_ROWS or set(years[validation]) != {VALIDATION_YEAR}:
        raise AssertionError("Unexpected validation cohort")
    if len(test) != EXPECTED_TEST_ROWS or set(years[test]) != {TEST_YEAR}:
        raise AssertionError("Unexpected test cohort")
    return train, validation, test


def metric_row(
    *,
    model: str,
    role: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    transform_seconds: float,
    prediction_seconds: float,
    fit_seconds: float,
    unseen_total: int,
    convergence_warning: bool,
    n_iter: int,
) -> dict[str, Any]:
    return {
        "analysis": "exclude2020_temporal_sensitivity",
        "model": model,
        "evaluation_role": role,
        "training_years": ";".join(str(year) for year in TRAIN_YEARS),
        "training_rows": EXPECTED_TRAIN_ROWS,
        "evaluation_year": VALIDATION_YEAR if role == "validation" else TEST_YEAR,
        "evaluation_rows": int(len(y_true)),
        "class_weighted": True,
        "transform_seconds": transform_seconds,
        "prediction_seconds": prediction_seconds,
        "fit_seconds": fit_seconds,
        "unseen_categorical_values": unseen_total,
        "convergence_warning": convergence_warning,
        "n_iter": n_iter,
        **calculate_metrics(y_true, y_pred, probabilities),
    }


def confusion_frame(y_true: np.ndarray, y_pred: np.ndarray, model: str, role: str) -> pd.DataFrame:
    rows = confusion_rows(
        y_true,
        y_pred,
        protocol="exclude2020_temporal_sensitivity",
        seed="year_based",
        model=model,
    )
    frame = pd.DataFrame(rows)
    if "evaluation_role" in frame.columns:
        frame["evaluation_role"] = role
    else:
        frame.insert(2, "evaluation_role", role)
    return frame


def stream_rng(key: str) -> np.random.Generator:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    words = [int.from_bytes(digest[offset:offset + 4], "little") for offset in (0, 4, 8, 12)]
    return np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SEED, *words]))


def confusion_from_vectors(y_true: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    matrices = np.zeros((predictions.shape[1], 3, 3), dtype=np.int64)
    for model_index in range(predictions.shape[1]):
        np.add.at(matrices[model_index], (y_true, predictions[:, model_index]), 1)
    return matrices


def bootstrap_joint_confusions(
    y_true: np.ndarray, predictions: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    model_count = predictions.shape[1]
    pattern_count = 3 ** model_count
    pattern_codes = np.arange(pattern_count, dtype=np.int64)
    powers = 3 ** np.arange(model_count, dtype=np.int64)
    decoded = (pattern_codes[:, None] // powers[None, :]) % 3
    draws = np.zeros((BOOTSTRAP_ITERATIONS, model_count, 3, 3), dtype=np.int32)
    encoded = np.sum(predictions.astype(np.int64) * powers[None, :], axis=1)
    for true_code in TARGET_CODES:
        class_codes = encoded[y_true == true_code]
        n_class = int(len(class_codes))
        empirical = np.bincount(class_codes, minlength=pattern_count)
        sampled = rng.multinomial(n_class, empirical / n_class, size=BOOTSTRAP_ITERATIONS)
        for model_index in range(model_count):
            for predicted_code in TARGET_CODES:
                draws[:, model_index, true_code, predicted_code] = sampled[
                    :, decoded[:, model_index] == predicted_code
                ].sum(axis=1)
    return draws


def metrics_from_confusions(confusions: np.ndarray) -> dict[str, np.ndarray]:
    matrix = np.asarray(confusions, dtype=float)
    total = matrix.sum(axis=(-2, -1))
    true_total = matrix.sum(axis=-1)
    predicted_total = matrix.sum(axis=-2)
    diagonal = np.diagonal(matrix, axis1=-2, axis2=-1)
    recalls = np.divide(diagonal, true_total, out=np.zeros_like(diagonal), where=true_total != 0)
    precisions = np.divide(diagonal, predicted_total, out=np.zeros_like(diagonal), where=predicted_total != 0)
    f1 = np.divide(
        2 * precisions * recalls,
        precisions + recalls,
        out=np.zeros_like(diagonal),
        where=(precisions + recalls) != 0,
    )
    quadratic = (
        np.arange(3, dtype=float)[:, None] - np.arange(3, dtype=float)[None, :]
    ) ** 2
    observed = np.sum(matrix * quadratic, axis=(-2, -1))
    expected = np.einsum("...i,...j->...ij", true_total, predicted_total) / total[..., None, None]
    expected_weighted = np.sum(expected * quadratic, axis=(-2, -1))
    qwk = 1.0 - np.divide(observed, expected_weighted, out=np.zeros_like(observed), where=expected_weighted != 0)
    ordinal = np.abs(
        np.arange(3, dtype=float)[:, None] - np.arange(3, dtype=float)[None, :]
    )
    severe_total = true_total[..., 1] + true_total[..., 2]
    severe_correct = matrix[..., 1:, 1:].sum(axis=(-2, -1))
    return {
        "macro_f1": f1.mean(axis=-1),
        "qwk": qwk,
        "ordinal_mae": np.sum(matrix * ordinal, axis=(-2, -1)) / total,
        "accuracy": diagonal.sum(axis=-1) / total,
        "fatal_recall": recalls[..., 2],
        "serious_or_fatal_recall": severe_correct / severe_total,
        "mean_asymmetric_cost": np.sum(matrix * ASYMMETRIC_COST_MATRIX, axis=(-2, -1)) / total,
    }


def orientation(metric: str) -> float:
    if metric in HIGHER_IS_BETTER:
        return 1.0
    if metric in LOWER_IS_BETTER:
        return -1.0
    raise KeyError(metric)


def interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(values, [CI_ALPHA / 2, 1 - CI_ALPHA / 2], method="linear")
    return float(lower), float(upper)


def comparison_rows(point: dict[str, np.ndarray], draws: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    comparisons = (
        ("sensitivity_H2", 3, 2, "exclude2020 LightGBM minus exclude2020 Logistic"),
        ("exclude2020_effect_logistic", 2, 0, "exclude2020 Logistic minus main Logistic"),
        ("exclude2020_effect_lightgbm", 3, 1, "exclude2020 LightGBM minus main LightGBM"),
    )
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        sign = orientation(metric)
        for name, candidate, reference, definition in comparisons:
            raw_point = float(point[metric][candidate] - point[metric][reference])
            raw_draws = draws[metric][:, candidate] - draws[metric][:, reference]
            oriented_point = sign * raw_point
            oriented_draws = sign * raw_draws
            raw_lower, raw_upper = interval(raw_draws)
            oriented_lower, oriented_upper = interval(oriented_draws)
            rows.append(
                {
                    "comparison": name,
                    "definition": definition,
                    "metric": metric,
                    "direction": "higher_is_better" if sign == 1 else "lower_is_better",
                    "raw_delta": raw_point,
                    "raw_ci_lower": raw_lower,
                    "raw_ci_upper": raw_upper,
                    "oriented_advantage": oriented_point,
                    "oriented_ci_lower": oriented_lower,
                    "oriented_ci_upper": oriented_upper,
                    "positive_oriented_value_favors": "first_named_model_or_exclude2020",
                    "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                }
            )

        main_h2 = point[metric][1] - point[metric][0]
        sensitivity_h2 = point[metric][3] - point[metric][2]
        raw_point = float(sensitivity_h2 - main_h2)
        raw_draws = (
            (draws[metric][:, 3] - draws[metric][:, 2])
            - (draws[metric][:, 1] - draws[metric][:, 0])
        )
        oriented_draws = sign * raw_draws
        raw_lower, raw_upper = interval(raw_draws)
        oriented_lower, oriented_upper = interval(oriented_draws)
        rows.append(
            {
                "comparison": "H2_contrast_change",
                "definition": "exclude2020 H2 contrast minus main H2 contrast",
                "metric": metric,
                "direction": "higher_is_better" if sign == 1 else "lower_is_better",
                "raw_delta": raw_point,
                "raw_ci_lower": raw_lower,
                "raw_ci_upper": raw_upper,
                "oriented_advantage": sign * raw_point,
                "oriented_ci_lower": oriented_lower,
                "oriented_ci_upper": oriented_upper,
                "positive_oriented_value_favors": "larger LightGBM advantage after excluding 2020",
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            }
        )
    return rows


def assert_outputs_absent() -> None:
    paths = [
        VALIDATION_METRICS_FILE,
        TEST_METRICS_FILE,
        CONFUSION_FILE,
        COMPARISON_FILE,
        BOOTSTRAP_FILE,
        PREPROCESS_AUDIT_FILE,
        SUMMARY_FILE,
        CHECKPOINT_FILE,
        MANIFEST_FILE,
    ]
    if any(path.exists() for path in paths) or MODEL_DIR.exists() or PREDICTION_DIR.exists():
        raise RuntimeError("Exclusion-2020 outputs already exist; refusing overwrite")


def run_analysis() -> None:
    protocol = require_protocol()
    assert_outputs_absent()
    started = time.perf_counter()
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    (STAGING_DIR / "predictions").mkdir(parents=True, exist_ok=False)
    (STAGING_DIR / "models" / "logistic").mkdir(parents=True, exist_ok=False)
    (STAGING_DIR / "models" / "lightgbm").mkdir(parents=True, exist_ok=False)

    schema, d6, _, _ = load_contracts()
    d10 = read_json(D10_PROTOCOL_FILE)
    selected = read_json(D10_SELECTED_FILE)
    features, target, metadata, assignments = load_aligned_data(schema, d6)
    train_pos, validation_pos, test_pos = evaluation_positions(assignments)
    y_train = target.iloc[train_pos].to_numpy(dtype=np.int8)
    y_validation = target.iloc[validation_pos].to_numpy(dtype=np.int8)
    y_test = target.iloc[test_pos].to_numpy(dtype=np.int8)
    weights = balanced_class_weights(y_train)
    categorical = list(schema["categorical_feature_columns"])
    numeric = list(schema["numeric_feature_columns"])
    feature_columns = list(schema["feature_columns"])

    print("D14 exclude-2020: fitting training-only categorical vocabulary...", flush=True)
    vocabulary = fit_category_vocabulary(features.iloc[train_pos], categorical)
    prepared: dict[str, Any] = {}
    unseen: dict[str, dict[str, int]] = {}
    for role, positions in (
        ("train", train_pos),
        ("validation", validation_pos),
        ("test", test_pos),
    ):
        output = prepare_features(
            features.iloc[positions],
            feature_columns=feature_columns,
            categorical_columns=categorical,
            numeric_columns=numeric,
            category_vocabulary=vocabulary,
        )
        prepared[role] = output.frame
        unseen[role] = output.unseen_counts
    if any(unseen["train"].values()):
        raise AssertionError("Training data generated unseen categories")

    preprocessor = make_preprocessor(
        categorical_columns=categorical,
        numeric_columns=numeric,
        category_vocabulary=vocabulary,
    )
    transform_started = time.perf_counter()
    encoded_train = preprocessor.fit_transform(prepared["train"])
    encoded_validation = preprocessor.transform(prepared["validation"])
    encoded_test = preprocessor.transform(prepared["test"])
    logistic_transform_seconds = time.perf_counter() - transform_started
    for matrix, expected in (
        (encoded_train, EXPECTED_TRAIN_ROWS),
        (encoded_validation, EXPECTED_VALIDATION_ROWS),
        (encoded_test, EXPECTED_TEST_ROWS),
    ):
        assert_encoded_matrix(matrix, expected)

    print("D14 exclude-2020: fitting weighted multinomial Logistic...", flush=True)
    logistic = make_logistic(weights)
    fit_started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        logistic.fit(encoded_train, y_train)
    logistic_fit_seconds = time.perf_counter() - fit_started
    convergence_warning = any(issubclass(item.category, ConvergenceWarning) for item in caught)
    logistic_n_iter = int(np.max(logistic.n_iter_))

    print("D14 exclude-2020: fitting frozen C03-1200 LightGBM with 8 threads...", flush=True)
    lightgbm = make_lightgbm(d10, selected, weights)
    fit_started = time.perf_counter()
    lightgbm.fit(
        prepared["train"],
        y_train,
        categorical_feature=categorical,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    lightgbm_fit_seconds = time.perf_counter() - fit_started
    if int(lightgbm.n_estimators_) != 1200:
        raise AssertionError("Sensitivity LightGBM tree count changed")

    metrics_by_role: dict[str, list[dict[str, Any]]] = {"validation": [], "test": []}
    confusion_frames: list[pd.DataFrame] = []
    test_predictions: dict[str, np.ndarray] = {}
    for role, positions, y_true in (
        ("validation", validation_pos, y_validation),
        ("test", test_pos, y_test),
    ):
        for model_name, estimator, matrix, transform_seconds, fit_seconds, warning, n_iter in (
            (
                "logistic_weighted",
                logistic,
                encoded_validation if role == "validation" else encoded_test,
                logistic_transform_seconds,
                logistic_fit_seconds,
                convergence_warning,
                logistic_n_iter,
            ),
            (
                "lightgbm_weighted",
                lightgbm,
                prepared[role],
                0.0,
                lightgbm_fit_seconds,
                False,
                1200,
            ),
        ):
            prediction_started = time.perf_counter()
            predicted, probabilities = predict_with_probabilities(estimator, matrix)
            prediction_seconds = time.perf_counter() - prediction_started
            row = metric_row(
                model=model_name,
                role=role,
                y_true=y_true,
                y_pred=predicted,
                probabilities=probabilities,
                transform_seconds=transform_seconds,
                prediction_seconds=prediction_seconds,
                fit_seconds=fit_seconds,
                unseen_total=int(sum(unseen[role].values())),
                convergence_warning=warning,
                n_iter=n_iter,
            )
            metrics_by_role[role].append(row)
            confusion_frames.append(confusion_frame(y_true, predicted, model_name, role))
            save_prediction(
                STAGING_DIR / "predictions" / f"{model_name}__{role}.csv.gz",
                prediction_frame(
                    collision_ids=metadata.iloc[positions]["meta_collision_index"],
                    years=assignments.iloc[positions]["meta_collision_year"],
                    y_true=y_true,
                    y_pred=predicted,
                    probabilities=probabilities,
                ),
            )
            if role == "test":
                test_predictions[f"exclude2020_{model_name}"] = predicted

    print("D14 exclude-2020: paired stratified bootstrap on the common 2024 test...", flush=True)
    base_ids: np.ndarray | None = None
    for key, path in MAIN_PREDICTION_FILES.items():
        frame = pd.read_csv(
            path,
            usecols=["meta_collision_index", "target_severity", "predicted_severity"],
            dtype={"meta_collision_index": "string"},
        )
        ids = frame["meta_collision_index"].astype("string").to_numpy()
        labels = frame["target_severity"].to_numpy(dtype=np.int8)
        if base_ids is None:
            base_ids = ids
        elif not np.array_equal(base_ids, ids):
            raise AssertionError("Main prediction IDs are not aligned")
        if not np.array_equal(labels, y_test):
            raise AssertionError("Main prediction labels differ from sensitivity test labels")
        test_predictions[key] = frame["predicted_severity"].to_numpy(dtype=np.int8)
    test_ids = metadata.iloc[test_pos]["meta_collision_index"].astype("string").to_numpy()
    if base_ids is None or not np.array_equal(base_ids, test_ids):
        raise AssertionError("Sensitivity and main test collision order differs")
    prediction_matrix = np.column_stack([test_predictions[name] for name in MODEL_ORDER])
    point_confusions = confusion_from_vectors(y_test, prediction_matrix)
    bootstrap_confusions = bootstrap_joint_confusions(
        y_test, prediction_matrix, stream_rng("D14_EXCLUDE2020_2024_COMMON")
    )
    point_metrics = metrics_from_confusions(point_confusions)
    bootstrap_metrics = metrics_from_confusions(bootstrap_confusions)
    comparisons = pd.DataFrame(comparison_rows(point_metrics, bootstrap_metrics))

    validation_metrics = pd.DataFrame(metrics_by_role["validation"])
    test_metrics = pd.DataFrame(metrics_by_role["test"])
    confusions = pd.concat(confusion_frames, ignore_index=True)
    audit_rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        audit_rows.append(
            {
                "feature": feature,
                "feature_type": "categorical" if feature in categorical else "numeric",
                "training_levels": len(vocabulary[feature]) if feature in categorical else "",
                "training_missing": int(features.iloc[train_pos][feature].isna().sum()),
                "validation_missing": int(features.iloc[validation_pos][feature].isna().sum()),
                "test_missing": int(features.iloc[test_pos][feature].isna().sum()),
                "validation_unseen": unseen["validation"].get(feature, ""),
                "test_unseen": unseen["test"].get(feature, ""),
                "fit_scope": "exclude2020_training_only",
            }
        )

    write_frame(STAGING_DIR / VALIDATION_METRICS_FILE.name, validation_metrics)
    write_frame(STAGING_DIR / TEST_METRICS_FILE.name, test_metrics)
    write_frame(STAGING_DIR / CONFUSION_FILE.name, confusions)
    write_frame(STAGING_DIR / COMPARISON_FILE.name, comparisons)
    write_npz(
        STAGING_DIR / BOOTSTRAP_FILE.name,
        {
            "model_order": np.asarray(MODEL_ORDER, dtype="U40"),
            "metric_order": np.asarray(METRICS, dtype="U40"),
            "point_confusions": point_confusions,
            "bootstrap_confusions": bootstrap_confusions,
            **{f"metric__{metric}": bootstrap_metrics[metric] for metric in METRICS},
        },
    )
    write_frame(STAGING_DIR / PREPROCESS_AUDIT_FILE.name, pd.DataFrame(audit_rows))
    joblib.dump(
        {
            "version": VERSION,
            "model": "logistic_weighted",
            "estimator": logistic,
            "class_weight": weights,
            "training_years": list(TRAIN_YEARS),
            "protocol_sha256": hash_file(PROTOCOL_FILE),
        },
        STAGING_DIR / "models" / "logistic" / "logistic_weighted.joblib",
        compress=3,
    )
    joblib.dump(
        {
            "version": VERSION,
            "feature_columns": feature_columns,
            "categorical_columns": categorical,
            "numeric_columns": numeric,
            "category_vocabulary": vocabulary,
            "preprocessor": preprocessor,
            "training_years": list(TRAIN_YEARS),
            "protocol_sha256": hash_file(PROTOCOL_FILE),
        },
        STAGING_DIR / "models" / "logistic" / "preprocessing_bundle.joblib",
        compress=3,
    )
    joblib.dump(
        {
            "version": VERSION,
            "model": "lightgbm_weighted",
            "estimator": lightgbm,
            "class_weight": weights,
            "selected_candidate_id": "C03",
            "n_estimators": 1200,
            "training_years": list(TRAIN_YEARS),
            "protocol_sha256": hash_file(PROTOCOL_FILE),
        },
        STAGING_DIR / "models" / "lightgbm" / "lightgbm_weighted.joblib",
        compress=3,
    )
    joblib.dump(
        {
            "version": VERSION,
            "feature_columns": feature_columns,
            "categorical_columns": categorical,
            "numeric_columns": numeric,
            "category_vocabulary": vocabulary,
            "numeric_missing_policy": "native_nan",
            "training_years": list(TRAIN_YEARS),
            "protocol_sha256": hash_file(PROTOCOL_FILE),
        },
        STAGING_DIR / "models" / "lightgbm" / "preprocessing_bundle.joblib",
        compress=3,
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log_destinations = {
        PREPROCESS_AUDIT_FILE.name: PREPROCESS_AUDIT_FILE,
        SUMMARY_FILE.name: SUMMARY_FILE,
        CHECKPOINT_FILE.name: CHECKPOINT_FILE,
    }
    for path in STAGING_DIR.iterdir():
        if path.name == "predictions":
            for child in path.iterdir():
                child.replace(PREDICTION_DIR / child.name)
        elif path.name == "models":
            for family in path.iterdir():
                destination = MODEL_DIR / family.name
                destination.mkdir(parents=True, exist_ok=True)
                for child in family.iterdir():
                    child.replace(destination / child.name)
        else:
            destination = log_destinations.get(path.name, RESULT_DIR / path.name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.replace(destination)
    shutil.rmtree(STAGING_DIR)

    macro = comparisons[
        comparisons["comparison"].eq("sensitivity_H2")
        & comparisons["metric"].eq("macro_f1")
    ].iloc[0]
    fatal = comparisons[
        comparisons["comparison"].eq("sensitivity_H2")
        & comparisons["metric"].eq("fatal_recall")
    ].iloc[0]
    macro_change = comparisons[
        comparisons["comparison"].eq("H2_contrast_change")
        & comparisons["metric"].eq("macro_f1")
    ].iloc[0]
    runtime = time.perf_counter() - started
    summary = {
        "version": VERSION,
        "status": "D14_EXCLUDE2020_SENSITIVITY_COMPLETE",
        "protocol_sha256": hash_file(PROTOCOL_FILE),
        "training_rows": EXPECTED_TRAIN_ROWS,
        "excluded_training_year": EXCLUDED_TRAIN_YEAR,
        "model_fits": 2,
        "test_rows": EXPECTED_TEST_ROWS,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "sensitivity_H2_macro_f1_delta": float(macro.raw_delta),
        "sensitivity_H2_macro_f1_ci": [float(macro.raw_ci_lower), float(macro.raw_ci_upper)],
        "sensitivity_H2_fatal_recall_delta": float(fatal.raw_delta),
        "sensitivity_H2_fatal_recall_ci": [float(fatal.raw_ci_lower), float(fatal.raw_ci_upper)],
        "macro_f1_H2_contrast_change": float(macro_change.raw_delta),
        "macro_f1_H2_contrast_change_ci": [
            float(macro_change.raw_ci_lower),
            float(macro_change.raw_ci_upper),
        ],
        "H1_pandemic_driver_claim_allowed": False,
        "runtime_seconds": float(runtime),
    }
    write_json(SUMMARY_FILE, summary)
    CHECKPOINT_FILE.write_text(
        "# D14 exclusion-2020 sensitivity checkpoint\n\n"
        "Status: **D14_EXCLUDE2020_SENSITIVITY_COMPLETE**\n\n"
        f"- Training years: {', '.join(str(year) for year in TRAIN_YEARS)}; "
        f"2020 excluded; n={EXPECTED_TRAIN_ROWS:,}.\n"
        f"- Sensitivity H2 Macro-F1 delta: {macro.raw_delta:+.6f}, "
        f"95% CI [{macro.raw_ci_lower:+.6f}, {macro.raw_ci_upper:+.6f}].\n"
        f"- Sensitivity H2 fatal-recall delta: {fatal.raw_delta:+.6f}, "
        f"95% CI [{fatal.raw_ci_lower:+.6f}, {fatal.raw_ci_upper:+.6f}].\n"
        f"- Change in the H2 Macro-F1 contrast versus the main analysis: "
        f"{macro_change.raw_delta:+.6f}, 95% CI "
        f"[{macro_change.raw_ci_lower:+.6f}, {macro_change.raw_ci_upper:+.6f}].\n"
        "- This sensitivity does not test whether 2020 caused H1 random-versus-future gaps.\n",
        encoding="utf-8",
    )

    artifacts = [
        PROTOCOL_FILE,
        VALIDATION_METRICS_FILE,
        TEST_METRICS_FILE,
        CONFUSION_FILE,
        COMPARISON_FILE,
        BOOTSTRAP_FILE,
        PREPROCESS_AUDIT_FILE,
        SUMMARY_FILE,
        CHECKPOINT_FILE,
        Path(__file__),
        TEST_SOURCE_FILE,
        *sorted(PREDICTION_DIR.glob("*.csv.gz")),
        *sorted(MODEL_DIR.rglob("*.joblib")),
    ]
    manifest = pd.DataFrame(
        [
            {
                "relative_path": relative(path),
                "bytes": int(path.stat().st_size),
                "sha256": hash_file(path),
            }
            for path in artifacts
        ]
    )
    write_frame(MANIFEST_FILE, manifest)
    print(f"D14 exclusion-2020 sensitivity complete in {runtime / 60:.1f} minutes.", flush=True)
    print(f"D14_EXCLUDE2020_MANIFEST_SHA256={hash_file(MANIFEST_FILE)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.freeze == args.run:
        parser.error("choose exactly one of --freeze or --run")
    if args.freeze:
        freeze_protocol()
    else:
        run_analysis()


if __name__ == "__main__":
    main()
