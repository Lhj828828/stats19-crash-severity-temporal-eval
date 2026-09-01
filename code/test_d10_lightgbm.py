"""Independent integrity checks for all D10 artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from baseline_modeling import classification_metrics
from d10_tune_lightgbm import select_candidate


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_DIR / "config" / "d10_lightgbm_protocol.json"
EXECUTION_AMENDMENT_FILE = PROJECT_DIR / "config" / "d10_execution_amendment.json"
SELECTED_FILE = PROJECT_DIR / "config" / "d10_selected_lightgbm.json"
D7_FILE = PROJECT_DIR / "config" / "d7_training_inputs.json"
ASSIGNMENTS_FILE = PROJECT_DIR / "data" / "processed" / "d6_split_assignments.csv.gz"
TUNING_FILE = PROJECT_DIR / "results" / "d10" / "d10_tuning_results.csv"
METRICS_FILE = PROJECT_DIR / "results" / "d10" / "d10_validation_metrics.csv"
CONFUSION_FILE = PROJECT_DIR / "results" / "d10" / "d10_validation_confusion_matrices.csv"
PREDICTION_DIR = PROJECT_DIR / "results" / "d10" / "validation_predictions"
MODEL_DIR = PROJECT_DIR / "models" / "d10"
PREPROCESS_AUDIT_FILE = PROJECT_DIR / "logs" / "d10_preprocessing_audit.csv"
TRAINING_AUDIT_FILE = PROJECT_DIR / "logs" / "d10_training_audit.csv"
MANIFEST_FILE = PROJECT_DIR / "logs" / "d10_artifact_manifest.csv"
MODEL_NAME = "lightgbm_weighted"
TARGET_CODES = {0, 1, 2}
SEEDS = (1103, 2207, 3301, 4409, 5501)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_identity(path: Path) -> tuple[str, str, str, str]:
    stem = path.name.removesuffix(".csv.gz")
    split_slug, model = stem.split("__", maxsplit=1)
    if split_slug == "temporal":
        return split_slug, "temporal", "year_based", "temporal_role"
    seed = split_slug.removeprefix("random_seed_")
    return split_slug, "random_reference", seed, f"random_role_seed_{seed}"


def main() -> None:
    protocol = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    amendment = json.loads(EXECUTION_AMENDMENT_FILE.read_text(encoding="utf-8"))
    selected = json.loads(SELECTED_FILE.read_text(encoding="utf-8"))
    d7 = json.loads(D7_FILE.read_text(encoding="utf-8"))
    assert protocol["version"] == "D10_V1"
    assert protocol["status"] == "PROTOCOL_FROZEN_BEFORE_D10_VALIDATION_INSPECTION"
    assert len(protocol["features"]["allowlist"]) == 17
    assert len(protocol["features"]["categorical"]) == 16
    assert protocol["features"]["numeric"] == ["feature_speed_limit"]
    assert not set(protocol["features"]["allowlist"]).intersection(
        protocol["features"]["metadata_excluded"]
    )
    assert protocol["preprocessing"]["one_hot_encoding"] is False
    assert "native" in protocol["preprocessing"]["numeric"]
    assert len(protocol["tuning"]["candidates"]) == 6
    assert protocol["evaluation_scope"]["forbidden"].startswith("all temporal/random test")
    assert selected["protocol_sha256"] == hash_file(CONFIG_FILE)
    assert selected["tuning_results_sha256"] == hash_file(TUNING_FILE)
    assert selected["test_data_used"] is False
    assert amendment["version"] == "D10_EXECUTION_AMENDMENT_V1"
    assert amendment["original_protocol_sha256"] == hash_file(CONFIG_FILE)
    assert amendment["original_n_jobs"] == -1
    assert amendment["amended_n_jobs"] == 4
    assert amendment["scientific_parameters_changed"] is False
    assert amendment["restart_all_candidates"] is True

    tuning = pd.read_csv(TUNING_FILE)
    assert len(tuning) == 6
    assert set(tuning["candidate_id"]) == {
        candidate["candidate_id"] for candidate in protocol["tuning"]["candidates"]
    }
    assert set(tuning["protocol"]) == {"temporal"}
    assert set(tuning["evaluation_role"]) == {"validation"}
    assert set(tuning["training_rows"]) == {538_461}
    assert set(tuning["validation_rows"]) == {104_258}
    assert not tuning["training_years"].astype(str).str.contains("2024").any()
    assert not tuning["validation_years"].astype(str).str.contains("2024").any()
    recomputed, constraints_met = select_candidate(tuning.to_dict("records"), protocol)
    assert recomputed["candidate_id"] == selected["selected_candidate_id"]
    assert int(recomputed["best_iteration"]) == int(selected["selected_n_estimators"])
    assert constraints_met == selected["selection_constraints_met"]

    metrics = pd.read_csv(METRICS_FILE, dtype={"seed": "string"})
    assert len(metrics) == 6
    assert set(metrics["model"]) == {MODEL_NAME}
    assert set(metrics["evaluation_role"]) == {"validation"}
    assert set(metrics["selected_candidate_id"]) == {selected["selected_candidate_id"]}
    assert set(metrics["n_estimators"]) == {int(selected["selected_n_estimators"])}
    assert not metrics["training_years"].astype(str).str.contains("2024").any()
    assert not metrics["validation_years"].astype(str).str.contains("2024").any()
    temporal = metrics.loc[metrics["protocol"].eq("temporal")]
    random_rows = metrics.loc[metrics["protocol"].eq("random_reference")]
    assert len(temporal) == 1
    assert len(random_rows) == 5
    assert set(temporal["training_rows"]) == {538_461}
    assert set(temporal["validation_rows"]) == {104_258}
    assert set(random_rows["training_rows"]) == {449_903}
    assert set(random_rows["validation_rows"]) == {96_408}

    assignments = pd.read_csv(
        ASSIGNMENTS_FILE,
        dtype={"meta_collision_index": "string"},
        usecols=[
            "meta_collision_index",
            "meta_collision_year",
            "temporal_role",
            *[f"random_role_seed_{seed}" for seed in SEEDS],
        ],
        low_memory=False,
    ).set_index("meta_collision_index")
    prediction_files = sorted(PREDICTION_DIR.glob("*.csv.gz"))
    assert len(prediction_files) == 6
    metric_lookup = metrics.set_index(["protocol", "seed", "model"])
    total_prediction_rows = 0
    for path in prediction_files:
        split_slug, model_protocol, seed, role_column = split_identity(path)
        assert path.name.endswith(f"__{MODEL_NAME}.csv.gz")
        prediction = pd.read_csv(path, dtype={"meta_collision_index": "string"})
        total_prediction_rows += len(prediction)
        assert prediction["meta_collision_index"].is_unique
        assert prediction["meta_collision_year"].max() <= 2023
        assert set(prediction["target_severity"].unique()) == TARGET_CODES
        assert set(prediction["predicted_severity"].unique()).issubset(TARGET_CODES)
        probabilities = prediction[["prob_slight", "prob_serious", "prob_fatal"]].to_numpy()
        assert np.isfinite(probabilities).all()
        assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=2e-8)
        assigned = assignments.loc[prediction["meta_collision_index"], role_column]
        assert assigned.eq("validation").all(), f"Non-validation row scored in {split_slug}"
        assigned_year = assignments.loc[
            prediction["meta_collision_index"], "meta_collision_year"
        ].to_numpy(dtype=int)
        assert np.array_equal(assigned_year, prediction["meta_collision_year"].to_numpy(dtype=int))

        row = metric_lookup.loc[(model_protocol, seed, MODEL_NAME)]
        computed = classification_metrics(
            prediction["target_severity"].to_numpy(dtype=int),
            prediction["predicted_severity"].to_numpy(dtype=int),
        )
        for metric, value in computed.items():
            assert np.isclose(value, float(row[metric]), rtol=0, atol=1e-14), (
                path.name,
                metric,
            )

    confusion = pd.read_csv(CONFUSION_FILE)
    assert len(confusion) == 6 * 9
    assert int(confusion["count"].sum()) == total_prediction_rows
    preprocess_audit = pd.read_csv(PREPROCESS_AUDIT_FILE)
    assert len(preprocess_audit) == 6 * 17
    assert set(preprocess_audit["feature_type"]) == {
        "categorical_native",
        "numeric_native",
    }
    numeric_audit = preprocess_audit.loc[
        preprocess_audit["feature"].eq("feature_speed_limit")
    ]
    assert len(numeric_audit) == 6
    assert numeric_audit["value_applied_to_data"].eq("none; NaN retained").all()

    training_audit = pd.read_csv(TRAINING_AUDIT_FILE, dtype={"seed": "string"})
    assert len(training_audit) == 6
    assert set(training_audit["selected_candidate_id"]) == {
        selected["selected_candidate_id"]
    }
    assert set(training_audit["n_estimators"]) == {int(selected["selected_n_estimators"])}

    split_specs = [("temporal", "temporal", "year_based")] + [
        (f"random_seed_{seed}", "random_reference", str(seed)) for seed in SEEDS
    ]
    for split_slug, model_protocol, seed in split_specs:
        split_dir = MODEL_DIR / split_slug
        bundle = joblib.load(split_dir / "preprocessing_bundle.joblib")
        artifact = joblib.load(split_dir / f"{MODEL_NAME}.joblib")
        assert bundle["version"] == "D10_V1"
        assert bundle["feature_columns"] == protocol["features"]["allowlist"]
        assert bundle["numeric_missing_policy"] == "native_nan"
        assert bundle["execution_amendment_sha256"] == hash_file(
            EXECUTION_AMENDMENT_FILE
        )
        assert 2024 not in bundle["training_years"]
        assert 2024 not in bundle["validation_years"]
        assert artifact["version"] == "D10_V1"
        assert artifact["protocol"] == model_protocol
        assert str(artifact["seed"]) == seed
        assert artifact["selected_candidate_id"] == selected["selected_candidate_id"]
        assert artifact["selected_candidate_parameters"] == selected["selected_candidate_parameters"]
        assert artifact["n_estimators"] == int(selected["selected_n_estimators"])
        assert artifact["target_codes"] == [0, 1, 2]
        assert artifact["execution_amendment_sha256"] == hash_file(
            EXECUTION_AMENDMENT_FILE
        )
        assert artifact["estimator"].get_params()["n_jobs"] == 4
        expected_weights_payload = (
            d7["weight_sets"]["temporal"]
            if model_protocol == "temporal"
            else d7["weight_sets"]["random_reference"][seed]
        )
        expected_weights = {
            int(code): float(value)
            for code, value in expected_weights_payload["class_weights"].items()
        }
        assert artifact["class_weight"] == expected_weights
        assert artifact["estimator"].get_params()["class_weight"] == expected_weights

    temporal_metric = temporal.iloc[0]
    selected_metrics = selected["selected_validation_metrics"]
    for metric in ("macro_f1", "qwk", "fatal_recall"):
        assert np.isclose(
            float(temporal_metric[metric]),
            float(selected_metrics[metric]),
            rtol=0,
            atol=1e-12,
        )

    manifest = pd.read_csv(MANIFEST_FILE)
    assert manifest["relative_path"].is_unique
    for _, row in manifest.iterrows():
        artifact_path = PROJECT_DIR / str(row["relative_path"])
        assert artifact_path.exists()
        assert artifact_path.stat().st_size == int(row["bytes"])
        assert hash_file(artifact_path) == row["sha256"]

    print("D10 tuning candidates checked:", len(tuning))
    print("D10 validation prediction rows checked:", total_prediction_rows)
    print("D10 split-specific class weights: PASS")
    print("D10 random-test and 2024 embargo: PASS")
    print("INDEPENDENT_D10_ASSERTIONS=PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"D10 TEST FAILURE: {error}", file=sys.stderr)
        raise
