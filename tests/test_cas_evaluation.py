"""Independent verification of the one-time CAS test evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "code"))

from cas_modeling_common import (  # noqa: E402
    ASSIGNMENTS_FILE,
    PROBABILITY_COLUMNS,
    classification_metrics,
    hash_file,
)


CONFIG_FILE = PROJECT_DIR / "config" / "cas" / "cas_evaluation_protocol.json"
COMPLETE_FILE = PROJECT_DIR / "config" / "cas" / "cas_evaluation_complete.json"
METRICS_FILE = PROJECT_DIR / "results" / "cas_evaluation" / "test_metrics.csv"
CONFUSION_FILE = (
    PROJECT_DIR / "results" / "cas_evaluation" / "test_confusion_matrices.csv"
)
PREDICTION_DIR = PROJECT_DIR / "results" / "cas_evaluation" / "predictions"
ACCESS_FILE = (
    PROJECT_DIR / "logs" / "cas" / "cas_evaluation_access_audit.csv"
)
MANIFEST_FILE = (
    PROJECT_DIR / "logs" / "cas" / "cas_evaluation_artifact_manifest.csv"
)
MODELS = {
    "dummy_most_frequent",
    "logistic_weighted",
    "lightgbm_weighted",
}


def main() -> None:
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    complete = json.loads(COMPLETE_FILE.read_text(encoding="utf-8"))
    assert config["version"] == "CAS_EVALUATION_V1"
    assert config["status"] == "FROZEN_BEFORE_FIRST_CAS_TEST_PREDICTION"
    assert config["group_count"] == 11
    assert config["metric_rows_expected"] == 33
    assert len(config["evaluation_groups"]) == 11
    assert set(config["models"]) == MODELS
    assert complete["status"] == "FROZEN_MODELS_EVALUATED_ONCE"
    assert complete["protocol_sha256"] == hash_file(CONFIG_FILE)
    assert complete["metrics_sha256"] == hash_file(METRICS_FILE)
    assert complete["model_selection_after_test"] is False
    assert complete["preprocessing_refit"] is False

    metrics = pd.read_csv(METRICS_FILE, dtype={"seed": "string"})
    assert len(metrics) == 33
    assert metrics["evaluation_group"].nunique() == 11
    assert set(metrics["model"]) == MODELS
    assert metrics.groupby("evaluation_group")["model"].nunique().eq(3).all()
    primary = metrics.loc[metrics["evaluation_group"].eq("temporal_2025")]
    assert len(primary) == 3
    assert set(primary["cohort_rows"]) == {10_542}
    assert set(primary["cohort_years"].astype(str)) == {"2025"}
    internal = metrics.loc[
        metrics["evaluation_design"].eq("random_internal_test")
    ]
    assert len(internal) == 15
    assert set(internal["cohort_rows"]) == {4_887}
    diagnostic = metrics.loc[
        metrics["evaluation_design"].eq("random_2025_diagnostic")
    ]
    assert len(diagnostic) == 15
    assert set(diagnostic["cohort_rows"]) == {10_542}
    assert set(diagnostic["cohort_years"].astype(str)) == {"2025"}

    group_lookup = {
        group["evaluation_group"]: group
        for group in config["evaluation_groups"]
    }
    assignments = pd.read_csv(
        ASSIGNMENTS_FILE,
        dtype={"meta_crash_id": "string"},
        keep_default_na=False,
        low_memory=False,
    ).set_index("meta_crash_id")
    prediction_files = sorted(PREDICTION_DIR.glob("*.csv.gz"))
    assert len(prediction_files) == 33
    metric_lookup = metrics.set_index(["evaluation_group", "model"])
    total_rows = 0
    for path in prediction_files:
        stem = path.name.removesuffix(".csv.gz")
        group_name, model = stem.split("__", maxsplit=1)
        group = group_lookup[group_name]
        prediction = pd.read_csv(path, dtype={"meta_crash_id": "string"})
        total_rows += len(prediction)
        assert prediction["meta_crash_id"].is_unique
        assert len(prediction) == int(group["expected_rows"])
        assigned = assignments.loc[
            prediction["meta_crash_id"], group["role_column"]
        ]
        assert assigned.eq(group["role_value"]).all()
        assert sorted(prediction["meta_crash_year"].unique().tolist()) == list(
            group["expected_years"]
        )
        probabilities = prediction[list(PROBABILITY_COLUMNS)].to_numpy(float)
        assert np.isfinite(probabilities).all()
        assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=2e-8)
        computed = classification_metrics(
            prediction["target_severity"].to_numpy(int),
            prediction["predicted_severity"].to_numpy(int),
        )
        row = metric_lookup.loc[(group_name, model)]
        for metric, value in computed.items():
            assert np.isclose(value, float(row[metric]), rtol=0, atol=1e-14)

    confusion = pd.read_csv(CONFUSION_FILE)
    assert len(confusion) == 33 * 9
    assert int(confusion["count"].sum()) == total_rows
    access = pd.read_csv(ACCESS_FILE)
    assert len(access) == 33
    assert not access["preprocessing_refit"].astype(bool).any()
    assert not access["threshold_tuning"].astype(bool).any()
    for _, row in access.iterrows():
        path = PROJECT_DIR / str(row["prediction_file"])
        assert hash_file(path) == row["prediction_sha256"]

    manifest = pd.read_csv(MANIFEST_FILE)
    assert manifest["relative_path"].is_unique
    for _, row in manifest.iterrows():
        path = PROJECT_DIR / str(row["relative_path"])
        assert path.stat().st_size == int(row["bytes"])
        assert hash_file(path) == row["sha256"]

    print("CAS evaluation groups:", metrics["evaluation_group"].nunique())
    print("CAS evaluation metric rows:", len(metrics))
    print("CAS evaluation prediction rows:", total_rows)
    print("CAS_EVALUATION_INDEPENDENT_CHECKS=PASS")


if __name__ == "__main__":
    main()
