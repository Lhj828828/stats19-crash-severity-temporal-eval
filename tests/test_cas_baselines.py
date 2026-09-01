"""Independent integrity checks for frozen CAS baseline artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "code"))

from cas_modeling_common import (  # noqa: E402
    ASSIGNMENTS_FILE,
    PROBABILITY_COLUMNS,
    RANDOM_SEEDS,
    THREADS,
    classification_metrics,
    hash_file,
)


CONFIG_FILE = PROJECT_DIR / "config" / "cas" / "cas_baseline_protocol.json"
FROZEN_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_baseline_models_frozen.json"
)
METRICS_FILE = PROJECT_DIR / "results" / "cas_baseline" / "validation_metrics.csv"
CONFUSION_FILE = (
    PROJECT_DIR
    / "results"
    / "cas_baseline"
    / "validation_confusion_matrices.csv"
)
PREDICTION_DIR = (
    PROJECT_DIR / "results" / "cas_baseline" / "validation_predictions"
)
MODEL_DIR = PROJECT_DIR / "models" / "cas_baseline"
MANIFEST_FILE = (
    PROJECT_DIR / "logs" / "cas" / "cas_baseline_artifact_manifest.csv"
)
PREPROCESS_AUDIT = (
    PROJECT_DIR / "logs" / "cas" / "cas_baseline_preprocessing_audit.csv"
)
MODELS = {"dummy_most_frequent", "logistic_weighted"}
NUMERIC = {
    "feature_advisory_speed",
    "feature_speed_limit",
    "feature_temporary_speed_limit",
}


def split_identity(path: Path) -> tuple[str, str, str]:
    stem = path.name.removesuffix(".csv.gz")
    split_slug, model = stem.split("__", maxsplit=1)
    if split_slug == "temporal":
        return "temporal_role", "year_based", model
    seed = split_slug.removeprefix("random_seed_")
    return f"random_role_seed_{seed}", seed, model


def main() -> None:
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN_FILE.read_text(encoding="utf-8"))
    assert config["version"] == "CAS_BASELINE_V1"
    assert config["status"] == (
        "FROZEN_BEFORE_CAS_VALIDATION_PERFORMANCE_INSPECTION"
    )
    assert config["execution"]["threads"] == THREADS == 8
    assert config["features"]["metadata_excluded"] == [
        "meta_crash_id",
        "meta_crash_year",
        "meta_region",
    ]
    assert config["features"]["numeric"] == [
        "feature_advisory_speed",
        "feature_speed_limit",
        "feature_temporary_speed_limit",
    ]
    assert "missingness indicators" in config["preprocessing"]["numeric"]
    assert frozen["status"] == (
        "BASELINE_MODELS_FROZEN_BEFORE_2025_EVALUATION"
    )
    assert frozen["test_performance_used"] is False
    assert frozen["protocol_sha256"] == hash_file(CONFIG_FILE)

    metrics = pd.read_csv(METRICS_FILE, dtype={"seed": "string"})
    assert len(metrics) == 12
    assert set(metrics["model"]) == MODELS
    assert set(metrics["evaluation_role"]) == {"validation"}
    assert not metrics["training_years"].astype(str).str.contains("2025").any()
    assert not metrics["validation_years"].astype(str).str.contains("2025").any()
    assert not metrics["convergence_warning"].astype(bool).any()
    temporal = metrics.loc[metrics["protocol"].eq("temporal")]
    assert set(temporal["training_rows"]) == {21_934}
    assert set(temporal["validation_rows"]) == {10_645}
    random = metrics.loc[metrics["protocol"].eq("random_reference")]
    assert set(random["training_rows"]) == {22_805}
    assert set(random["validation_rows"]) == {4_887}

    assignments = pd.read_csv(
        ASSIGNMENTS_FILE,
        dtype={"meta_crash_id": "string"},
        keep_default_na=False,
        low_memory=False,
    ).set_index("meta_crash_id")
    prediction_files = sorted(PREDICTION_DIR.glob("*.csv.gz"))
    assert len(prediction_files) == 12
    lookup = metrics.set_index(["seed", "model"])
    prediction_rows = 0
    for path in prediction_files:
        role_column, seed, model = split_identity(path)
        prediction = pd.read_csv(path, dtype={"meta_crash_id": "string"})
        prediction_rows += len(prediction)
        assert prediction["meta_crash_id"].is_unique
        assert not prediction["meta_crash_year"].eq(2025).any()
        assert assignments.loc[
            prediction["meta_crash_id"], role_column
        ].eq("validation").all()
        probabilities = prediction[list(PROBABILITY_COLUMNS)].to_numpy(float)
        assert np.isfinite(probabilities).all()
        assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=2e-8)
        computed = classification_metrics(
            prediction["target_severity"].to_numpy(int),
            prediction["predicted_severity"].to_numpy(int),
        )
        row = lookup.loc[(seed, model)]
        for metric, value in computed.items():
            assert np.isclose(value, float(row[metric]), rtol=0, atol=1e-14)

    confusion = pd.read_csv(CONFUSION_FILE)
    assert len(confusion) == 12 * 9
    assert int(confusion["count"].sum()) == prediction_rows

    split_slugs = ["temporal", *[f"random_seed_{s}" for s in RANDOM_SEEDS]]
    assert len(list(MODEL_DIR.rglob("*.joblib"))) == 18
    for split_slug in split_slugs:
        split_dir = MODEL_DIR / split_slug
        bundle = joblib.load(split_dir / "preprocessing_bundle.joblib")
        assert 2025 not in bundle["training_years"]
        assert 2025 not in bundle["validation_years"]
        assert set(bundle["numeric_missing_indicator_features"]) == NUMERIC
        for model in MODELS:
            artifact = joblib.load(split_dir / f"{model}.joblib")
            assert artifact["model_name"] == model
            assert artifact["target_codes"] == [0, 1, 2]
            assert artifact["protocol_sha256"] == hash_file(CONFIG_FILE)

    audit = pd.read_csv(PREPROCESS_AUDIT)
    numeric_audit = audit.loc[audit["feature_type"].eq("numeric")]
    assert len(numeric_audit) == 6 * 3
    assert numeric_audit["missing_indicator_fitted"].astype(bool).all()

    manifest = pd.read_csv(MANIFEST_FILE)
    assert manifest["relative_path"].is_unique
    for _, row in manifest.iterrows():
        path = PROJECT_DIR / str(row["relative_path"])
        assert path.stat().st_size == int(row["bytes"])
        assert hash_file(path) == row["sha256"]

    print("CAS baseline metric rows:", len(metrics))
    print("CAS baseline validation prediction rows:", prediction_rows)
    print("CAS baseline 2025 embargo: PASS")
    print("CAS_BASELINE_INDEPENDENT_CHECKS=PASS")


if __name__ == "__main__":
    main()
