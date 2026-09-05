"""Independent checks for the completed CAS post-hoc feature ablations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "code"))

from cas_bootstrap_uncertainty import (  # noqa: E402
    metrics_from_confusion,
    point_confusion,
)
from cas_modeling_common import classification_metrics, hash_file  # noqa: E402


PROTOCOL_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_feature_ablation_protocol.json"
)
COMPLETE_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_feature_ablation_complete.json"
)
RESULT_DIR = PROJECT_DIR / "results" / "cas_feature_ablation"
PREDICTION_DIR = RESULT_DIR / "predictions"
MODEL_DIR = PROJECT_DIR / "models" / "cas_feature_ablation"
MANIFEST_FILE = (
    PROJECT_DIR / "logs" / "cas" / "cas_feature_ablation_artifact_manifest.csv"
)
METRICS = (
    "macro_f1",
    "qwk",
    "ordinal_mae",
    "accuracy",
    "minor_recall",
    "serious_recall",
    "fatal_recall",
    "serious_or_fatal_recall",
    "mean_asymmetric_cost",
)
SCENARIOS = {"drop_urban": 14, "drop_sparse_speed": 13}
MODELS = {"logistic_weighted", "lightgbm_weighted"}


def main() -> None:
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    complete = json.loads(COMPLETE_FILE.read_text(encoding="utf-8"))
    assert protocol["version"] == "CAS_FEATURE_ABLATION_POSTHOC_V1"
    assert protocol["post_hoc_disclosure"][
        "primary_2025_results_known_before_freeze"
    ] is True
    assert protocol["post_hoc_disclosure"][
        "eligible_for_primary_model_selection"
    ] is False
    assert complete["status"] == "POST_HOC_FEATURE_ABLATION_COMPLETE"
    assert complete["protocol_sha256"] == hash_file(PROTOCOL_FILE)
    assert complete["primary_models_or_results_modified"] is False
    assert complete["eligible_for_model_selection"] is False
    assert complete["bootstrap_iterations"] == 2_000

    validation = pd.read_csv(RESULT_DIR / "validation_metrics.csv")
    test = pd.read_csv(RESULT_DIR / "test_metrics.csv")
    intervals = pd.read_csv(RESULT_DIR / "test_metric_intervals.csv")
    differences = pd.read_csv(RESULT_DIR / "test_paired_differences.csv")
    features = pd.read_csv(RESULT_DIR / "feature_contract.csv")
    confusion = pd.read_csv(RESULT_DIR / "confusion_matrices.csv")
    preprocessing = pd.read_csv(
        PROJECT_DIR / "logs" / "cas" / "cas_feature_ablation_preprocessing.csv"
    )
    training = pd.read_csv(
        PROJECT_DIR / "logs" / "cas" / "cas_feature_ablation_training.csv"
    )

    assert len(validation) == 2 * 2
    assert len(test) == 2 + 2 * 2
    assert len(intervals) == 2 * 2 * len(METRICS)
    assert len(differences) == (2 * 2 + 2) * len(METRICS)
    assert len(confusion) == (2 + 2 * 2 * 2) * 9
    assert set(validation["scenario"]) == set(SCENARIOS)
    assert set(test["scenario"]) == {"full15_primary", *SCENARIOS}
    assert set(intervals["valid_iterations"]) == {2_000}
    assert set(differences["valid_iterations"]) == {2_000}
    assert (intervals["ci_lower"] <= intervals["ci_upper"]).all()
    assert (differences["ci_lower"] <= differences["ci_upper"]).all()
    assert np.isfinite(
        intervals[["point_estimate", "ci_lower", "ci_upper"]].to_numpy(float)
    ).all()
    assert np.isfinite(
        differences[
            ["point_difference_a_minus_b", "ci_lower", "ci_upper"]
        ].to_numpy(float)
    ).all()

    for scenario, count in SCENARIOS.items():
        scenario_features = features.loc[features["scenario"].eq(scenario)]
        assert len(scenario_features) == 15
        assert int(scenario_features["status"].eq("retained").sum()) == count
        assert set(
            validation.loc[validation["scenario"].eq(scenario), "feature_count"]
        ) == {count}
        assert set(
            test.loc[test["scenario"].eq(scenario), "feature_count"]
        ) == {count}
        assert set(
            training.loc[training["scenario"].eq(scenario), "feature_count"]
        ) == {count}

    dropped_urban = features.loc[
        features["scenario"].eq("drop_urban")
        & features["status"].eq("dropped"),
        "feature",
    ]
    dropped_speed = features.loc[
        features["scenario"].eq("drop_sparse_speed")
        & features["status"].eq("dropped"),
        "feature",
    ]
    assert set(dropped_urban) == {"feature_urban"}
    assert set(dropped_speed) == {
        "feature_advisory_speed",
        "feature_temporary_speed_limit",
    }
    assert set(training["model"]) == MODELS
    assert set(training["training_rows"]) == {21_934}
    assert set(training["validation_rows"]) == {10_645}
    assert set(training["test_rows"]) == {10_542}
    assert set(training["threads"]) == {8}
    logistic_warnings = training.loc[
        training["model"].eq("logistic_weighted"), "convergence_warning"
    ]
    assert logistic_warnings.astype(str).str.lower().eq("false").all()
    native_numeric = preprocessing.loc[
        preprocessing["feature_type"].eq("numeric_native")
    ]
    assert native_numeric["value_applied_to_data"].eq("none; NaN retained").all()

    expected_ids: np.ndarray | None = None
    expected_targets: np.ndarray | None = None
    for scenario, feature_count in SCENARIOS.items():
        split_dir = MODEL_DIR / scenario
        logistic_bundle = joblib.load(
            split_dir / "logistic_preprocessing_bundle.joblib"
        )
        lightgbm_bundle = joblib.load(
            split_dir / "lightgbm_preprocessing_bundle.joblib"
        )
        assert len(logistic_bundle["feature_columns"]) == feature_count
        assert logistic_bundle["feature_columns"] == lightgbm_bundle[
            "feature_columns"
        ]
        assert logistic_bundle["fit_scope"] == "2022-2023 training rows only"
        assert lightgbm_bundle["fit_scope"] == "2022-2023 training rows only"
        assert lightgbm_bundle["numeric_missing_policy"] == "native_nan"
        for model in MODELS:
            artifact = joblib.load(split_dir / f"{model}.joblib")
            assert artifact["protocol_sha256"] == hash_file(PROTOCOL_FILE)
            if model == "lightgbm_weighted":
                params = artifact["estimator"].get_params()
                assert artifact["selected_candidate_id"] == "C06"
                assert artifact["n_estimators"] == 1_000
                assert params["n_jobs"] == 8
                assert params["deterministic"] is True
                assert params["force_col_wise"] is True

            prediction = pd.read_csv(
                PREDICTION_DIR / f"{scenario}__test_2025__{model}.csv.gz",
                dtype={"meta_crash_id": "string"},
            )
            assert len(prediction) == 10_542
            assert prediction["meta_crash_id"].is_unique
            assert set(prediction["meta_crash_year"]) == {2025}
            probabilities = prediction[
                ["prob_minor", "prob_serious", "prob_fatal"]
            ].to_numpy(float)
            assert np.isfinite(probabilities).all()
            assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=2e-8)
            ids = prediction["meta_crash_id"].to_numpy()
            targets = prediction["target_severity"].to_numpy(np.int8)
            if expected_ids is None:
                expected_ids, expected_targets = ids, targets
            else:
                assert np.array_equal(ids, expected_ids)
                assert np.array_equal(targets, expected_targets)
            computed = classification_metrics(
                targets, prediction["predicted_severity"].to_numpy(np.int8)
            )
            row = test.loc[
                test["scenario"].eq(scenario) & test["model"].eq(model)
            ].iloc[0]
            for metric, value in computed.items():
                assert np.isclose(value, float(row[metric]), rtol=0, atol=1e-14)

    for _, row in differences.iterrows():
        assert np.isclose(
            float(row["point_difference_a_minus_b"]),
            float(row["model_a_value"]) - float(row["model_b_value"]),
            rtol=0,
            atol=1e-14,
        )
    for _, row in intervals.iterrows():
        prediction = pd.read_csv(
            PREDICTION_DIR
            / f"{row['scenario']}__test_2025__{row['model']}.csv.gz"
        )
        point = metrics_from_confusion(
            point_confusion(
                prediction["target_severity"].to_numpy(np.int8),
                prediction["predicted_severity"].to_numpy(np.int8),
            )
        )
        assert np.isclose(
            float(row["point_estimate"]),
            float(point[str(row["metric"])]),
            rtol=0,
            atol=1e-14,
        )

    manifest = pd.read_csv(MANIFEST_FILE)
    assert manifest["relative_path"].is_unique
    for _, row in manifest.iterrows():
        path = PROJECT_DIR / str(row["relative_path"])
        assert path.stat().st_size == int(row["bytes"])
        assert hash_file(path) == row["sha256"]
    assert complete["manifest_sha256"] == hash_file(MANIFEST_FILE)
    print("CAS feature-ablation scenarios:", len(SCENARIOS))
    print("CAS feature-ablation 2025 predictions:", 2 * 2 * 10_542)
    print("CAS_FEATURE_ABLATION_INDEPENDENT_CHECKS=PASS")


if __name__ == "__main__":
    main()
