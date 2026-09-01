"""Reusable preprocessing and metrics for the frozen baseline models."""

from __future__ import annotations

from dataclasses import dataclass
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
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_CODES = (0, 1, 2)
TARGET_LABELS = {0: "Slight", 1: "Serious", 2: "Fatal"}
UNSEEN_TOKEN = "__UNSEEN__"

# Rows are true classes and columns are predicted classes. This matrix is
# frozen before performance inspection and is descriptive, not a tuning target.
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


def fit_category_vocabulary(
    training: pd.DataFrame,
    categorical_columns: list[str],
) -> dict[str, list[str]]:
    """Fit an exact categorical vocabulary from one training partition."""
    vocabulary: dict[str, list[str]] = {}
    for column in categorical_columns:
        series = training[column]
        if series.isna().any():
            raise ValueError(f"Categorical training feature contains missing values: {column}")
        levels = sorted(str(value) for value in series.unique().tolist())
        if not levels:
            raise ValueError(f"No training levels found for {column}")
        if UNSEEN_TOKEN in levels:
            raise ValueError(f"Reserved unseen token occurs in frozen data: {column}")
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
    """Apply a training-fitted vocabulary without learning from this frame."""
    if features.columns.tolist() != feature_columns:
        raise ValueError("Feature columns or their order differ from the frozen allowlist")
    prepared = pd.DataFrame(index=features.index)
    unseen_counts: dict[str, int] = {}
    for column in categorical_columns:
        known_levels = category_vocabulary[column]
        source = features[column].astype("object")
        known = source.isin(known_levels)
        unseen_counts[column] = int((~known).sum())
        mapped = source.where(known, UNSEEN_TOKEN)
        prepared[column] = pd.Categorical(
            mapped,
            categories=[*known_levels, UNSEEN_TOKEN],
        )
    for column in numeric_columns:
        prepared[column] = pd.to_numeric(features[column], errors="raise")
    prepared = prepared.loc[:, feature_columns]
    return PreparedFeatures(prepared, unseen_counts)


def make_preprocessor(
    *,
    categorical_columns: list[str],
    numeric_columns: list[str],
    category_vocabulary: dict[str, list[str]],
) -> ColumnTransformer:
    """Build the fixed one-hot/imputation preprocessing specification."""
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
            ("imputer", SimpleImputer(strategy="median")),
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
    if matrix.shape[0] != expected_rows:
        raise AssertionError("Encoded matrix row count changed")
    if matrix.shape[1] <= 0:
        raise AssertionError("Encoded feature matrix is empty")
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(values).all():
        raise AssertionError("Encoded feature matrix contains non-finite values")


def predict_with_probabilities(estimator: Any, encoded: Any) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(estimator.predict(encoded), dtype=np.int8)
    probabilities = np.asarray(estimator.predict_proba(encoded), dtype=float)
    classes = np.asarray(estimator.classes_, dtype=int)
    if classes.tolist() != list(TARGET_CODES):
        raise AssertionError(f"Unexpected estimator class order: {classes.tolist()}")
    if probabilities.shape != (len(predicted), len(TARGET_CODES)):
        raise AssertionError("Unexpected probability matrix shape")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-8):
        raise AssertionError("Predicted probabilities do not sum to one")
    return predicted, probabilities


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if set(np.unique(y_true)) != set(TARGET_CODES):
        raise ValueError("Metric cohort must contain all three true classes")
    if not set(np.unique(y_pred)).issubset(TARGET_CODES):
        raise ValueError("Predictions contain an invalid class")
    recalls = recall_score(
        y_true,
        y_pred,
        labels=list(TARGET_CODES),
        average=None,
        zero_division=0,
    )
    serious_or_fatal_true = y_true >= 1
    serious_or_fatal_pred = y_pred >= 1
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
        "slight_recall": float(recalls[0]),
        "serious_recall": float(recalls[1]),
        "fatal_recall": float(recalls[2]),
        "serious_or_fatal_recall": float(
            recall_score(
                serious_or_fatal_true,
                serious_or_fatal_pred,
                zero_division=0,
            )
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
                    "evaluation_role": "validation",
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
