"""Independent integrity checks for the sealed D11 evaluation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_modeling import classification_metrics

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROTOCOL_FILE = PROJECT_DIR / "config" / "d11_evaluation_protocol.json"
RUN_LOCK_FILE = PROJECT_DIR / "logs" / "d11_run_lock.json"
SUMMARY_FILE = PROJECT_DIR / "logs" / "d11_run_summary.json"
METRICS_FILE = PROJECT_DIR / "results" / "d11" / "d11_test_metrics.csv"
CONFUSION_FILE = PROJECT_DIR / "results" / "d11" / "d11_test_confusion_matrices.csv"
AUDIT_FILE = PROJECT_DIR / "logs" / "d11_preprocessing_audit.csv"
PREDICTION_DIR = PROJECT_DIR / "results" / "d11" / "predictions"
FIGURE_FILE = PROJECT_DIR / "figures" / "d11_test_metric_overview.png"
MANIFEST_FILE = PROJECT_DIR / "logs" / "d11_artifact_manifest.csv"
ASSIGNMENTS_FILE = PROJECT_DIR / "data" / "processed" / "d6_split_assignments.csv.gz"
D10_SELECTED_FILE = PROJECT_DIR / "config" / "d10_selected_lightgbm.json"

TARGET_CODES = {0, 1, 2}
RANDOM_SEEDS = (1103, 2207, 3301, 4409, 5501)
MODEL_IDS = {
    "dummy_most_frequent", "logistic_unweighted", "logistic_weighted",
    "ordered_logit_unweighted", "lightgbm_weighted",
}


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    lock = json.loads(RUN_LOCK_FILE.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
    selected = json.loads(D10_SELECTED_FILE.read_text(encoding="utf-8"))
    assert protocol["version"] == "D11_V1"
    assert protocol["status"] == "FROZEN_BEFORE_D11_TEST_ACCESS"
    assert protocol["scientific_lock"]["no_model_reselection"] is True
    assert protocol["scientific_lock"]["d10b_does_not_change_d11"] is True
    assert protocol["evaluation_matrix"]["main_rows"] == 30
    assert protocol["evaluation_matrix"]["secondary_2024_diagnostic"]["enabled"] is True
    assert selected["selected_candidate_id"] == "C03"
    assert selected["selected_n_estimators"] == 1200
    assert lock["status"] == "SEALED_AFTER_SUCCESSFUL_ONE_TIME_EVALUATION"
    assert lock["protocol_sha256"] == hash_file(PROTOCOL_FILE)
    assert summary["status"] == "PASS_ONE_TIME_TEST_EVALUATION_COMPLETED"
    assert summary["main_evaluations"] == 30
    assert summary["secondary_2024_diagnostics"] == 25
    assert summary["fit_operations"] == 0
    assert summary["preprocessor_fit_operations"] == 0
    assert summary["test_based_selection_operations"] == 0

    assignments = pd.read_csv(
        ASSIGNMENTS_FILE,
        dtype={"meta_collision_index": "string"},
        usecols=[
            "meta_collision_index", "meta_collision_year", "temporal_role",
            *[f"random_role_seed_{seed}" for seed in RANDOM_SEEDS],
        ],
        low_memory=False,
    ).set_index("meta_collision_index")
    metrics = pd.read_csv(METRICS_FILE, dtype={"seed": "string"})
    assert len(metrics) == 55
    assert set(metrics["model"]) == MODEL_IDS
    assert set(metrics["evaluation_scope"]) == {
        "temporal_test_2024", "random_reference_test", "random_model_on_2024",
    }
    main_rows = metrics.loc[metrics["evaluation_scope"] != "random_model_on_2024"]
    secondary = metrics.loc[metrics["evaluation_scope"].eq("random_model_on_2024")]
    assert len(main_rows) == 30 and len(secondary) == 25
    assert set(main_rows.loc[main_rows["split_slug"].eq("temporal"), "test_rows"]) == {100_927}
    assert set(main_rows.loc[main_rows["split_slug"].str.startswith("random_seed_"), "test_rows"]) == {96_408}
    assert set(secondary["test_rows"]) == {100_927}
    assert set(secondary["test_years"].astype(str)) == {"2024"}
    metric_columns = [
        "macro_f1", "qwk", "ordinal_mae", "accuracy", "slight_recall",
        "serious_recall", "fatal_recall", "serious_or_fatal_recall",
        "mean_asymmetric_cost", "slight_precision", "serious_precision",
        "fatal_precision", "multiclass_logloss",
    ]
    assert np.isfinite(metrics[metric_columns].to_numpy(dtype=float)).all()

    prediction_files = sorted(PREDICTION_DIR.glob("*.csv.gz"))
    assert len(prediction_files) == 55
    lookup = metrics.set_index(["evaluation_scope", "split_slug", "model"])
    total_rows = 0
    for path in prediction_files:
        prediction = pd.read_csv(path, dtype={"meta_collision_index": "string"})
        total_rows += len(prediction)
        assert prediction["meta_collision_index"].is_unique
        assert set(prediction["target_severity"]) == TARGET_CODES
        assert set(prediction["predicted_severity"]).issubset(TARGET_CODES)
        scope = str(prediction["evaluation_scope"].iloc[0])
        slug = str(prediction["split_slug"].iloc[0])
        model = str(prediction["model"].iloc[0])
        assert prediction["evaluation_scope"].nunique() == 1
        assert prediction["split_slug"].nunique() == 1
        assert prediction["model"].nunique() == 1
        row = lookup.loc[(scope, slug, model)]
        assert len(prediction) == int(row["test_rows"])
        probabilities = prediction[["prob_slight", "prob_serious", "prob_fatal"]].to_numpy(float)
        assert np.isfinite(probabilities).all()
        assert probabilities.min() >= -1e-10
        assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=2e-8)
        assigned_years = assignments.loc[
            prediction["meta_collision_index"], "meta_collision_year"
        ].to_numpy(dtype=int)
        assert np.array_equal(assigned_years, prediction["meta_collision_year"].to_numpy(int))
        if scope in {"temporal_test_2024", "random_model_on_2024"}:
            assert set(assigned_years) == {2024}
        else:
            assert set(assigned_years).issubset(set(range(2018, 2024)))
        computed = classification_metrics(
            prediction["target_severity"].to_numpy(int),
            prediction["predicted_severity"].to_numpy(int),
        )
        for metric, value in computed.items():
            assert np.isclose(value, float(row[metric]), rtol=0, atol=1e-14)

    expected_total = 5 * 100_927 + 25 * 96_408 + 25 * 100_927
    assert total_rows == expected_total
    confusion = pd.read_csv(CONFUSION_FILE)
    assert len(confusion) == 55 * 9
    assert int(confusion["count"].sum()) == total_rows
    audit = pd.read_csv(AUDIT_FILE)
    assert len(audit) == 55
    assert set(audit["preprocessor_action"]) == {"transform_only"}
    assert set(audit["preprocessor_fit_operations"]) == {0}
    assert FIGURE_FILE.exists() and FIGURE_FILE.stat().st_size > 10_000
    manifest = pd.read_csv(MANIFEST_FILE)
    assert manifest["relative_path"].is_unique
    for _, row in manifest.iterrows():
        path = PROJECT_DIR / str(row["relative_path"])
        assert path.exists()
        assert path.stat().st_size == int(row["bytes"])
        assert hash_file(path) == row["sha256"]
    assert lock["metrics_sha256"] == hash_file(METRICS_FILE)
    assert lock["confusion_sha256"] == hash_file(CONFUSION_FILE)

    print("D11 main metric rows checked:", len(main_rows))
    print("D11 secondary diagnostic rows checked:", len(secondary))
    print("D11 prediction rows checked:", total_rows)
    print("D11 role/year and transform-only checks: PASS")
    print("D11 one-time run lock: PASS")
    print("INDEPENDENT_D11_ASSERTIONS=PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"D11 TEST FAILURE: {error}", file=sys.stderr)
        raise
