"""Independent checks for the D14 exclusion-2020 sensitivity analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from d14_exclude2020_sensitivity import (
    BOOTSTRAP_FILE,
    BOOTSTRAP_ITERATIONS,
    CHECKPOINT_FILE,
    COMPARISON_FILE,
    CONFUSION_FILE,
    EXPECTED_TEST_ROWS,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_VALIDATION_ROWS,
    MANIFEST_FILE,
    METRICS,
    MODEL_DIR,
    PREDICTION_DIR,
    PROJECT_DIR,
    PROTOCOL_FILE,
    SUMMARY_FILE,
    TEST_METRICS_FILE,
    TRAIN_YEARS,
    VALIDATION_METRICS_FILE,
    VERSION,
    balanced_class_weights,
    build_protocol,
    confusion_from_vectors,
    metrics_from_confusions,
)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unit_checks() -> None:
    weights = balanced_class_weights(np.array([0, 0, 0, 1, 1, 2], dtype=np.int8))
    assert np.isclose(weights[0], 2 / 3)
    assert np.isclose(weights[1], 1.0)
    assert np.isclose(weights[2], 2.0)
    y = np.array([0, 0, 1, 1, 2, 2], dtype=np.int8)
    predictions = np.column_stack([y, np.zeros_like(y)])
    metrics = metrics_from_confusions(confusion_from_vectors(y, predictions))
    assert np.isclose(metrics["accuracy"][0], 1.0)
    assert np.isclose(metrics["fatal_recall"][1], 0.0)


def main() -> None:
    unit_checks()
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    assert protocol["version"] == VERSION
    assert protocol == build_protocol()
    assert protocol["runtime"]["CPU_thread_limit"] == 8
    assert protocol["scientific_scope"]["no_retuning"] is True
    assert "cannot determine" in protocol["scientific_scope"]["not_answered"]

    validation = pd.read_csv(VALIDATION_METRICS_FILE)
    test = pd.read_csv(TEST_METRICS_FILE)
    confusion = pd.read_csv(CONFUSION_FILE)
    comparisons = pd.read_csv(COMPARISON_FILE)
    assert len(validation) == 2 and len(test) == 2
    assert set(validation["evaluation_rows"]) == {EXPECTED_VALIDATION_ROWS}
    assert set(test["evaluation_rows"]) == {EXPECTED_TEST_ROWS}
    assert set(test["training_rows"]) == {EXPECTED_TRAIN_ROWS}
    assert set(test["training_years"]) == {";".join(str(year) for year in TRAIN_YEARS)}
    assert len(confusion) == 2 * 2 * 9
    assert len(comparisons) == len(METRICS) * 4
    assert set(comparisons["comparison"]) == {
        "sensitivity_H2",
        "exclude2020_effect_logistic",
        "exclude2020_effect_lightgbm",
        "H2_contrast_change",
    }

    for model in ("logistic_weighted", "lightgbm_weighted"):
        for role, expected in (("validation", EXPECTED_VALIDATION_ROWS), ("test", EXPECTED_TEST_ROWS)):
            path = PREDICTION_DIR / f"{model}__{role}.csv.gz"
            frame = pd.read_csv(path, usecols=["meta_collision_index", "target_severity", "predicted_severity"])
            assert len(frame) == expected
            assert frame["meta_collision_index"].nunique() == expected
            assert set(frame["target_severity"]) == {0, 1, 2}
            assert set(frame["predicted_severity"]).issubset({0, 1, 2})

    logistic = joblib.load(MODEL_DIR / "logistic" / "logistic_weighted.joblib")
    lightgbm = joblib.load(MODEL_DIR / "lightgbm" / "lightgbm_weighted.joblib")
    assert logistic["training_years"] == list(TRAIN_YEARS)
    assert lightgbm["training_years"] == list(TRAIN_YEARS)
    assert lightgbm["selected_candidate_id"] == "C03"
    assert lightgbm["n_estimators"] == 1200
    assert lightgbm["estimator"].get_params()["n_jobs"] == 8

    bootstrap = np.load(BOOTSTRAP_FILE)
    assert bootstrap["bootstrap_confusions"].shape == (BOOTSTRAP_ITERATIONS, 4, 3, 3)
    assert bootstrap["point_confusions"].shape == (4, 3, 3)
    assert bootstrap["model_order"].shape == (4,)
    for metric in METRICS:
        assert bootstrap[f"metric__{metric}"].shape == (BOOTSTRAP_ITERATIONS, 4)

    summary = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
    assert summary["status"] == "D14_EXCLUDE2020_SENSITIVITY_COMPLETE"
    assert summary["model_fits"] == 2
    assert summary["H1_pandemic_driver_claim_allowed"] is False
    assert CHECKPOINT_FILE.is_file()

    manifest = pd.read_csv(MANIFEST_FILE)
    assert len(manifest) == 19
    for row in manifest.itertuples(index=False):
        path = PROJECT_DIR / row.relative_path
        assert path.stat().st_size == int(row.bytes)
        assert hash_file(path) == row.sha256
    print("INDEPENDENT_D14_EXCLUDE2020_ASSERTIONS=PASS")
    print("TRAIN_ROWS=447262 VALIDATION_ROWS=104258 TEST_ROWS=100927")
    print("COMPARISON_ROWS=28 BOOTSTRAP_ITERATIONS=2000")


if __name__ == "__main__":
    main()
