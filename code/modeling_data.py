"""Load the frozen D5 table with an explicit feature allowlist.

All modelling scripts should use this loader. It returns metadata, target and
features separately, and refuses schema drift or accidental meta-column use.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_FILE = (
    PROJECT_DIR / "data" / "processed" / "stats19_modeling_dataset.csv.gz"
)
DEFAULT_SCHEMA_FILE = PROJECT_DIR / "config" / "d5_dataset_schema.json"


def load_modeling_data(
    data_file: Path = DEFAULT_DATA_FILE,
    schema_file: Path = DEFAULT_SCHEMA_FILE,
    *,
    nrows: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return X, y and metadata using only the frozen schema allowlists."""
    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    if schema.get("status") != "QC_PASSED_AND_FROZEN":
        raise ValueError("Model data must use the D5 QC-passed frozen schema")
    feature_columns = list(schema["feature_columns"])
    metadata_columns = list(schema["metadata_columns"])
    target_column = str(schema["target_column"])

    if len(feature_columns) != 17 or len(set(feature_columns)) != 17:
        raise ValueError("Expected 17 unique frozen feature columns")
    if any(not column.startswith("feature_") for column in feature_columns):
        raise ValueError("Every model feature must use the feature_ prefix")
    if any(not column.startswith("meta_") for column in metadata_columns):
        raise ValueError("Every metadata column must use the meta_ prefix")

    expected_columns = metadata_columns + [target_column] + feature_columns
    actual_columns = pd.read_csv(data_file, nrows=0).columns.tolist()
    if actual_columns != expected_columns:
        raise ValueError(
            "D4 table schema differs from the frozen allowlist: "
            f"expected={expected_columns}, actual={actual_columns}"
        )

    table = pd.read_csv(data_file, usecols=expected_columns, nrows=nrows)
    metadata = table.loc[:, metadata_columns].copy()
    target = table.loc[:, target_column].copy()
    features = table.loc[:, feature_columns].copy()

    if features.columns.tolist() != feature_columns:
        raise AssertionError("Feature order changed during loading")
    if set(features).intersection(metadata_columns + [target_column]):
        raise AssertionError("Metadata or target leaked into the feature matrix")
    return features, target, metadata


if __name__ == "__main__":
    X, y, metadata = load_modeling_data(nrows=1_000)
    print(f"X shape: {X.shape}")
    print(f"y rows: {len(y)}")
    print(f"metadata shape: {metadata.shape}")
    print("Feature allowlist check passed.")
