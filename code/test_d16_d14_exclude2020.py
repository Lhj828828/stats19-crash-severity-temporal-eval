"""Validate the D14 exclusion-2020 sensitivity under the D16 thread cap."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

import d14_exclude2020_sensitivity as analysis


THREAD_CAP = 4


def main() -> None:
    analysis.THREAD_LIMIT = THREAD_CAP

    weights = analysis.balanced_class_weights(
        np.array([0, 0, 0, 1, 1, 2], dtype=np.int8)
    )
    assert np.isclose(weights[0], 2 / 3)
    assert np.isclose(weights[1], 1.0)
    assert np.isclose(weights[2], 2.0)
    target = np.array([0, 0, 1, 1, 2, 2], dtype=np.int8)
    predictions = np.column_stack([target, np.zeros_like(target)])
    metrics = analysis.metrics_from_confusions(
        analysis.confusion_from_vectors(target, predictions)
    )
    assert np.isclose(metrics["accuracy"][0], 1.0)
    assert np.isclose(metrics["fatal_recall"][1], 0.0)

    protocol = json.loads(
        analysis.PROTOCOL_FILE.read_text(encoding="utf-8")
    )
    assert protocol["version"] == analysis.VERSION
    assert protocol == analysis.build_protocol()
    assert protocol["runtime"]["CPU_thread_limit"] == THREAD_CAP
    assert protocol["scientific_scope"]["no_retuning"] is True
    assert "cannot determine" in protocol["scientific_scope"]["not_answered"]

    validation = pd.read_csv(analysis.VALIDATION_METRICS_FILE)
    test = pd.read_csv(analysis.TEST_METRICS_FILE)
    confusion = pd.read_csv(analysis.CONFUSION_FILE)
    comparisons = pd.read_csv(analysis.COMPARISON_FILE)
    assert len(validation) == 2 and len(test) == 2
    assert set(validation["evaluation_rows"]) == {
        analysis.EXPECTED_VALIDATION_ROWS
    }
    assert set(test["evaluation_rows"]) == {analysis.EXPECTED_TEST_ROWS}
    assert set(test["training_rows"]) == {analysis.EXPECTED_TRAIN_ROWS}
    assert set(test["training_years"]) == {
        ";".join(str(year) for year in analysis.TRAIN_YEARS)
    }
    assert len(confusion) == 2 * 2 * 9
    assert len(comparisons) == len(analysis.METRICS) * 4
    assert set(comparisons["comparison"]) == {
        "sensitivity_H2",
        "exclude2020_effect_logistic",
        "exclude2020_effect_lightgbm",
        "H2_contrast_change",
    }

    for model in ("logistic_weighted", "lightgbm_weighted"):
        for role, expected in (
            ("validation", analysis.EXPECTED_VALIDATION_ROWS),
            ("test", analysis.EXPECTED_TEST_ROWS),
        ):
            path = analysis.PREDICTION_DIR / f"{model}__{role}.csv.gz"
            frame = pd.read_csv(
                path,
                usecols=[
                    "meta_collision_index",
                    "target_severity",
                    "predicted_severity",
                ],
            )
            assert len(frame) == expected
            assert frame["meta_collision_index"].nunique() == expected
            assert set(frame["target_severity"]) == {0, 1, 2}
            assert set(frame["predicted_severity"]).issubset({0, 1, 2})

    logistic = joblib.load(
        analysis.MODEL_DIR / "logistic" / "logistic_weighted.joblib"
    )
    lightgbm = joblib.load(
        analysis.MODEL_DIR / "lightgbm" / "lightgbm_weighted.joblib"
    )
    assert logistic["training_years"] == list(analysis.TRAIN_YEARS)
    assert lightgbm["training_years"] == list(analysis.TRAIN_YEARS)
    assert lightgbm["selected_candidate_id"] == "C03"
    assert lightgbm["n_estimators"] == 1200
    assert lightgbm["estimator"].get_params()["n_jobs"] == THREAD_CAP

    with np.load(analysis.BOOTSTRAP_FILE) as bootstrap:
        assert bootstrap["bootstrap_confusions"].shape == (
            analysis.BOOTSTRAP_ITERATIONS,
            4,
            3,
            3,
        )
        assert bootstrap["point_confusions"].shape == (4, 3, 3)
        assert bootstrap["model_order"].shape == (4,)
        for metric in analysis.METRICS:
            assert bootstrap[f"metric__{metric}"].shape == (
                analysis.BOOTSTRAP_ITERATIONS,
                4,
            )

    summary = json.loads(
        analysis.SUMMARY_FILE.read_text(encoding="utf-8")
    )
    assert summary["status"] == "D14_EXCLUDE2020_SENSITIVITY_COMPLETE"
    assert summary["model_fits"] == 2
    assert summary["H1_pandemic_driver_claim_allowed"] is False
    assert analysis.CHECKPOINT_FILE.is_file()

    manifest = pd.read_csv(analysis.MANIFEST_FILE)
    assert len(manifest) == 19
    for row in manifest.itertuples(index=False):
        path = analysis.PROJECT_DIR / row.relative_path
        assert path.stat().st_size == int(row.bytes)
        assert analysis.hash_file(path) == row.sha256

    print("INDEPENDENT_D16_D14_EXCLUDE2020_ASSERTIONS=PASS")
    print("D16_D14_EXCLUDE2020_THREAD_CAP=4")


if __name__ == "__main__":
    main()
