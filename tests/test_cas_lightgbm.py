"""Independent checks for frozen CAS LightGBM artifacts."""

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
from cas_train_lightgbm import select_candidate  # noqa: E402


CONFIG_FILE = PROJECT_DIR / "config" / "cas" / "cas_lightgbm_protocol.json"
SELECTED_FILE = PROJECT_DIR / "config" / "cas" / "cas_lightgbm_selected.json"
FROZEN_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_lightgbm_models_frozen.json"
)
TUNING_FILE = PROJECT_DIR / "results" / "cas_lightgbm" / "tuning_results.csv"
METRICS_FILE = (
    PROJECT_DIR / "results" / "cas_lightgbm" / "validation_metrics.csv"
)
CONFUSION_FILE = (
    PROJECT_DIR
    / "results"
    / "cas_lightgbm"
    / "validation_confusion_matrices.csv"
)
PREDICTION_DIR = (
    PROJECT_DIR / "results" / "cas_lightgbm" / "validation_predictions"
)
MODEL_DIR = PROJECT_DIR / "models" / "cas_lightgbm"
PREPROCESS_AUDIT = (
    PROJECT_DIR / "logs" / "cas" / "cas_lightgbm_preprocessing_audit.csv"
)
MANIFEST_FILE = (
    PROJECT_DIR / "logs" / "cas" / "cas_lightgbm_artifact_manifest.csv"
)
MODEL_NAME = "lightgbm_weighted"


def split_identity(path: Path) -> tuple[str, str]:
    split_slug = path.name.removesuffix(f"__{MODEL_NAME}.csv.gz")
    if split_slug == "temporal":
        return "temporal_role", "year_based"
    seed = split_slug.removeprefix("random_seed_")
    return f"random_role_seed_{seed}", seed


def main() -> None:
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    selected = json.loads(SELECTED_FILE.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN_FILE.read_text(encoding="utf-8"))
    assert config["version"] == "CAS_LIGHTGBM_V1"
    assert config["status"] == (
        "FROZEN_BEFORE_CAS_LIGHTGBM_VALIDATION_INSPECTION"
    )
    common = config["common_model_parameters"]
    assert common["n_jobs"] == THREADS == 8
    assert common["deterministic"] is True
    assert common["force_col_wise"] is True
    assert len(config["tuning"]["candidates"]) == 6
    assert selected["protocol_sha256"] == hash_file(CONFIG_FILE)
    assert selected["tuning_results_sha256"] == hash_file(TUNING_FILE)
    assert selected["test_data_used"] is False
    assert selected["selected_candidate_id"] == "C06"
    assert selected["selected_n_estimators"] == 1_000
    assert selected["selection_constraints_met"] is False
    assert selected["boundary_extension_triggered"] is False
    assert frozen["status"] == (
        "LIGHTGBM_MODELS_FROZEN_BEFORE_2025_EVALUATION"
    )
    assert frozen["test_performance_used"] is False

    tuning = pd.read_csv(TUNING_FILE).to_dict("records")
    recomputed, constraints_met = select_candidate(tuning, config)
    assert recomputed["candidate_id"] == selected["selected_candidate_id"]
    assert constraints_met is False

    metrics = pd.read_csv(METRICS_FILE, dtype={"seed": "string"})
    assert len(metrics) == 6
    assert set(metrics["model"]) == {MODEL_NAME}
    assert set(metrics["selected_candidate_id"]) == {"C06"}
    assert set(metrics["n_estimators"]) == {1_000}
    assert not metrics["training_years"].astype(str).str.contains("2025").any()
    assert not metrics["validation_years"].astype(str).str.contains("2025").any()
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
    predictions = sorted(PREDICTION_DIR.glob("*.csv.gz"))
    assert len(predictions) == 6
    lookup = metrics.set_index("seed")
    total_rows = 0
    for path in predictions:
        role_column, seed = split_identity(path)
        prediction = pd.read_csv(path, dtype={"meta_crash_id": "string"})
        total_rows += len(prediction)
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
        row = lookup.loc[seed]
        for metric, value in computed.items():
            assert np.isclose(value, float(row[metric]), rtol=0, atol=1e-14)

    confusion = pd.read_csv(CONFUSION_FILE)
    assert len(confusion) == 6 * 9
    assert int(confusion["count"].sum()) == total_rows
    assert len(list(MODEL_DIR.rglob("*.joblib"))) == 12
    split_slugs = ["temporal", *[f"random_seed_{s}" for s in RANDOM_SEEDS]]
    for split_slug in split_slugs:
        split_dir = MODEL_DIR / split_slug
        bundle = joblib.load(split_dir / "preprocessing_bundle.joblib")
        artifact = joblib.load(split_dir / f"{MODEL_NAME}.joblib")
        assert bundle["numeric_missing_policy"] == "native_nan"
        assert 2025 not in bundle["training_years"]
        assert 2025 not in bundle["validation_years"]
        estimator = artifact["estimator"]
        parameters = estimator.get_params()
        assert parameters["n_jobs"] == 8
        assert parameters["deterministic"] is True
        assert parameters["force_col_wise"] is True
        assert artifact["n_estimators"] == 1_000
        assert artifact["selected_candidate_id"] == "C06"
        assert artifact["protocol_sha256"] == hash_file(CONFIG_FILE)
        assert artifact["selected_config_sha256"] == hash_file(SELECTED_FILE)

    audit = pd.read_csv(PREPROCESS_AUDIT)
    numeric = audit.loc[audit["feature_type"].eq("numeric_native")]
    assert len(numeric) == 6 * 3
    assert numeric["value_applied_to_data"].eq("none; NaN retained").all()
    manifest = pd.read_csv(MANIFEST_FILE)
    assert manifest["relative_path"].is_unique
    for _, row in manifest.iterrows():
        path = PROJECT_DIR / str(row["relative_path"])
        assert path.stat().st_size == int(row["bytes"])
        assert hash_file(path) == row["sha256"]

    print("CAS LightGBM tuning rows:", len(tuning))
    print("CAS LightGBM validation prediction rows:", total_rows)
    print("CAS LightGBM 2025 embargo: PASS")
    print("CAS_LIGHTGBM_INDEPENDENT_CHECKS=PASS")


if __name__ == "__main__":
    main()
