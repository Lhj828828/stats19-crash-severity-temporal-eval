"""Shared, dataset-specific modeling utilities for the frozen CAS study."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "data" / "processed" / "cas_modeling_dataset.csv.gz"
SCHEMA_FILE = PROJECT_DIR / "config" / "cas" / "cas_modeling_schema.json"
ASSIGNMENTS_FILE = (
    PROJECT_DIR / "data" / "processed" / "cas_split_assignments.csv.gz"
)
ANALYSIS_PROTOCOL_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_analysis_protocol.json"
)
TRAINING_INPUTS_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_training_inputs.json"
)

TARGET_CODES = (0, 1, 2)
TARGET_LABELS = {0: "Minor", 1: "Serious", 2: "Fatal"}
PROBABILITY_COLUMNS = ("prob_minor", "prob_serious", "prob_fatal")
UNSEEN_TOKEN = "__UNSEEN__"
THREADS = 8
RANDOM_SEEDS = (1103, 2207, 3301, 4409, 5501)

ASYMMETRIC_COST_MATRIX = np.array(
    [
        [0.0, 1.0, 2.0],
        [2.0, 0.0, 1.0],
        [5.0, 3.0, 0.0],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class PreparedFeatures:
    frame: pd.DataFrame
    unseen_counts: dict[str, int]


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_DIR.resolve()).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows generated for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_dataframe_gzip(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as binary:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=binary,
            compresslevel=6,
            mtime=0,
        ) as compressed:
            with io.TextIOWrapper(
                compressed,
                encoding="utf-8",
                newline="",
            ) as text:
                frame.to_csv(text, index=False, lineterminator="\n")
    temporary.replace(path)


def load_contracts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    schema = load_json(SCHEMA_FILE)
    analysis = load_json(ANALYSIS_PROTOCOL_FILE)
    training = load_json(TRAINING_INPUTS_FILE)
    if schema.get("status") != "QC_PASSED_AND_FROZEN_BEFORE_MODELING":
        raise ValueError("CAS modeling schema is not frozen")
    if analysis.get("status") != "FROZEN_BEFORE_ANY_CAS_MODEL_TRAINING":
        raise ValueError("CAS analysis protocol is not frozen")
    if (
        training.get("status")
        != "TRAINING_INPUTS_AUDITED_AND_FROZEN_BEFORE_MODELING"
    ):
        raise ValueError("CAS training inputs are not frozen")
    if hash_file(DATA_FILE) != schema["sha256"]:
        raise ValueError("CAS model-ready table changed after freezing")
    if hash_file(ASSIGNMENTS_FILE) != analysis["assignments"]["sha256"]:
        raise ValueError("CAS split assignments changed after freezing")
    if hash_file(ASSIGNMENTS_FILE) != training["split_assignments"]["sha256"]:
        raise ValueError("CAS training-input assignment hash differs")
    return schema, analysis, training


def load_aligned_data(
    schema: dict[str, Any],
    analysis: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    feature_columns = list(schema["feature_columns"])
    categorical = list(schema["categorical_feature_columns"])
    numeric = list(schema["numeric_feature_columns"])
    metadata_columns = list(schema["metadata_columns"])
    target_column = str(schema["target_column"])
    expected = metadata_columns + [target_column] + feature_columns
    header = pd.read_csv(DATA_FILE, nrows=0).columns.tolist()
    if header != expected:
        raise ValueError("CAS model table differs from the frozen column allowlist")

    dtype = {
        "meta_crash_id": "string",
        "meta_region": "string",
        **{column: "string" for column in categorical},
    }
    table = pd.read_csv(
        DATA_FILE,
        dtype=dtype,
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )
    if len(table) != int(schema["row_count"]):
        raise ValueError("CAS model table row count changed")
    for column in numeric:
        table[column] = pd.to_numeric(table[column], errors="raise")
    features = table.loc[:, feature_columns].copy()
    target = table.loc[:, target_column].astype("int8").copy()
    metadata = table.loc[:, metadata_columns].copy()

    role_columns = list(analysis["assignments"]["role_columns"])
    assignments = pd.read_csv(
        ASSIGNMENTS_FILE,
        dtype={"meta_crash_id": "string"},
        keep_default_na=False,
        low_memory=False,
    )
    required_assignment_columns = [
        "meta_crash_id",
        "meta_crash_year",
        "target_severity",
        *role_columns,
    ]
    if assignments.columns.tolist() != required_assignment_columns:
        raise ValueError("CAS assignment columns changed")
    if len(assignments) != len(table):
        raise ValueError("CAS model and assignment row counts differ")
    if not metadata["meta_crash_id"].reset_index(drop=True).equals(
        assignments["meta_crash_id"].reset_index(drop=True)
    ):
        raise ValueError("CAS model table and assignments are not row-aligned")
    if not np.array_equal(
        metadata["meta_crash_year"].to_numpy(dtype=int),
        assignments["meta_crash_year"].to_numpy(dtype=int),
    ):
        raise ValueError("CAS model table and assignment years differ")
    if not np.array_equal(
        target.to_numpy(dtype=int),
        assignments["target_severity"].to_numpy(dtype=int),
    ):
        raise ValueError("CAS model table and assignment targets differ")
    if features.columns.tolist() != feature_columns:
        raise AssertionError("CAS feature order changed")
    if set(features).intersection(metadata_columns + [target_column]):
        raise AssertionError("CAS metadata or target leaked into features")
    return features, target, metadata, assignments


def make_split_specs(analysis: dict[str, Any]) -> list[dict[str, str]]:
    specs = [
        {
            "split_slug": "temporal",
            "protocol": "temporal",
            "seed": "year_based",
            "role_column": "temporal_role",
        }
    ]
    for seed in analysis["random_reference_protocol"]["seeds"]:
        specs.append(
            {
                "split_slug": f"random_seed_{seed}",
                "protocol": "random_reference",
                "seed": str(seed),
                "role_column": f"random_role_seed_{seed}",
            }
        )
    return specs


def class_weights_for_spec(
    training: dict[str, Any],
    spec: dict[str, str],
) -> dict[int, float]:
    if spec["protocol"] == "temporal":
        payload = training["weight_sets"]["temporal"]
    else:
        payload = training["weight_sets"]["random_reference"][spec["seed"]]
    return {
        int(code): float(weight)
        for code, weight in payload["class_weights"].items()
    }


def fit_category_vocabulary(
    training: pd.DataFrame,
    categorical_columns: list[str],
) -> dict[str, list[str]]:
    vocabulary: dict[str, list[str]] = {}
    for column in categorical_columns:
        series = training[column]
        if series.isna().any():
            raise ValueError(f"Categorical training feature is missing: {column}")
        levels = sorted(str(value) for value in series.unique().tolist())
        if not levels:
            raise ValueError(f"No training categories found: {column}")
        if UNSEEN_TOKEN in levels:
            raise ValueError(f"Reserved unseen token occurs in source: {column}")
        vocabulary[column] = levels
    return vocabulary


def prepare_features(
    features: pd.DataFrame,
    *,
    feature_columns: list[str],
    categorical_columns: list[str],
    numeric_columns: list[str],
    category_vocabulary: dict[str, list[str]],
) -> PreparedFeatures:
    if features.columns.tolist() != feature_columns:
        raise ValueError("CAS feature columns or order differ from allowlist")
    prepared = pd.DataFrame(index=features.index)
    unseen_counts: dict[str, int] = {}
    for column in categorical_columns:
        levels = category_vocabulary[column]
        source = features[column].astype("object")
        known = source.isin(levels)
        unseen_counts[column] = int((~known).sum())
        prepared[column] = pd.Categorical(
            source.where(known, UNSEEN_TOKEN),
            categories=[*levels, UNSEEN_TOKEN],
        )
    for column in numeric_columns:
        prepared[column] = pd.to_numeric(features[column], errors="raise")
    return PreparedFeatures(prepared.loc[:, feature_columns], unseen_counts)


def make_linear_preprocessor(
    *,
    categorical_columns: list[str],
    numeric_columns: list[str],
    category_vocabulary: dict[str, list[str]],
) -> ColumnTransformer:
    categories = [
        [*category_vocabulary[column], UNSEEN_TOKEN]
        for column in categorical_columns
    ]
    categorical = OneHotEncoder(
        categories=categories,
        handle_unknown="error",
        sparse_output=True,
        dtype=np.float64,
    )
    numeric = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical, categorical_columns),
            ("numeric", numeric, numeric_columns),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )


def assert_encoded_matrix(matrix: Any, expected_rows: int) -> None:
    if matrix.shape[0] != expected_rows or matrix.shape[1] <= 0:
        raise AssertionError("Encoded CAS matrix has an invalid shape")
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(values).all():
        raise AssertionError("Encoded CAS matrix contains non-finite values")


def predict_with_probabilities(
    estimator: Any,
    model_input: Any,
) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(estimator.predict(model_input), dtype=np.int8)
    probabilities = np.asarray(estimator.predict_proba(model_input), dtype=float)
    classes = np.asarray(estimator.classes_, dtype=int)
    if classes.tolist() != list(TARGET_CODES):
        raise AssertionError(f"Unexpected estimator class order: {classes.tolist()}")
    if probabilities.shape != (len(predicted), len(TARGET_CODES)):
        raise AssertionError("Unexpected CAS probability matrix shape")
    if not np.isfinite(probabilities).all():
        raise AssertionError("CAS probabilities contain non-finite values")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=2e-8):
        raise AssertionError("CAS probabilities do not sum to one")
    return predicted, probabilities


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if set(np.unique(y_true)) != set(TARGET_CODES):
        raise ValueError("CAS metric cohort must contain all target classes")
    if not set(np.unique(y_pred)).issubset(TARGET_CODES):
        raise ValueError("CAS predictions contain an invalid target class")
    recalls = recall_score(
        y_true,
        y_pred,
        labels=list(TARGET_CODES),
        average=None,
        zero_division=0,
    )
    precisions = precision_score(
        y_true,
        y_pred,
        labels=list(TARGET_CODES),
        average=None,
        zero_division=0,
    )
    return {
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(TARGET_CODES),
                average="macro",
                zero_division=0,
            )
        ),
        "qwk": float(
            cohen_kappa_score(
                y_true,
                y_pred,
                labels=list(TARGET_CODES),
                weights="quadratic",
            )
        ),
        "ordinal_mae": float(np.mean(np.abs(y_true - y_pred))),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "minor_recall": float(recalls[0]),
        "serious_recall": float(recalls[1]),
        "fatal_recall": float(recalls[2]),
        "minor_precision": float(precisions[0]),
        "serious_precision": float(precisions[1]),
        "fatal_precision": float(precisions[2]),
        "serious_or_fatal_recall": float(
            recall_score(y_true >= 1, y_pred >= 1, zero_division=0)
        ),
        "mean_asymmetric_cost": float(
            np.mean(ASYMMETRIC_COST_MATRIX[y_true, y_pred])
        ),
    }


def confusion_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    protocol: str,
    seed: str,
    evaluation_role: str,
    model: str,
) -> list[dict[str, object]]:
    matrix = confusion_matrix(y_true, y_pred, labels=list(TARGET_CODES))
    rows: list[dict[str, object]] = []
    for true_code in TARGET_CODES:
        true_total = int(matrix[true_code].sum())
        for predicted_code in TARGET_CODES:
            count = int(matrix[true_code, predicted_code])
            rows.append(
                {
                    "protocol": protocol,
                    "seed": seed,
                    "evaluation_role": evaluation_role,
                    "model": model,
                    "true_code": true_code,
                    "true_label": TARGET_LABELS[true_code],
                    "predicted_code": predicted_code,
                    "predicted_label": TARGET_LABELS[predicted_code],
                    "count": count,
                    "row_pct": count / true_total,
                }
            )
    return rows


def prediction_frame(
    *,
    crash_ids: pd.Series,
    years: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "meta_crash_id": crash_ids.astype("string").to_numpy(),
            "meta_crash_year": years.astype(int).to_numpy(),
            "target_severity": np.asarray(y_true, dtype=np.int8),
            "predicted_severity": np.asarray(y_pred, dtype=np.int8),
            "prob_minor": probabilities[:, 0],
            "prob_serious": probabilities[:, 1],
            "prob_fatal": probabilities[:, 2],
        }
    )


def build_artifact_manifest(paths: list[Path], output: Path) -> None:
    rows = []
    for path in sorted({item.resolve() for item in paths}):
        rows.append(
            {
                "relative_path": path.relative_to(PROJECT_DIR.resolve()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hash_file(path),
            }
        )
    write_csv(output, rows)
