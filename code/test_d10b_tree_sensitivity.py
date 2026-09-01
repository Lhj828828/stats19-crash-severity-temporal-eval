"""Independent integrity checks for the D10b sensitivity artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from baseline_modeling import classification_metrics


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROTOCOL_FILE = PROJECT_DIR / "config" / "d10b_tree_sensitivity_protocol.json"
D10_SELECTED_FILE = PROJECT_DIR / "config" / "d10_selected_lightgbm.json"
D10_METRICS_FILE = PROJECT_DIR / "results" / "d10" / "d10_validation_metrics.csv"
D10_PREDICTION_FILE = (
    PROJECT_DIR
    / "results"
    / "d10"
    / "validation_predictions"
    / "temporal__lightgbm_weighted.csv.gz"
)
ASSIGNMENTS_FILE = PROJECT_DIR / "data" / "processed" / "d6_split_assignments.csv.gz"
METRICS_FILE = PROJECT_DIR / "results" / "d10b" / "d10b_checkpoint_metrics.csv"
HISTORY_FILE = PROJECT_DIR / "results" / "d10b" / "d10b_logloss_history.csv"
PREDICTION_DIR = PROJECT_DIR / "results" / "d10b" / "validation_predictions"
MODEL_FILE = PROJECT_DIR / "models" / "d10b" / "temporal" / "c03_2000_trees.joblib"
FIGURE_FILE = PROJECT_DIR / "figures" / "d10b_validation_logloss_curve.png"
CHECKPOINT_FILE = PROJECT_DIR / "logs" / "d10b_checkpoint.md"
MANIFEST_FILE = PROJECT_DIR / "logs" / "d10b_artifact_manifest.csv"
CHECKPOINTS = (1200, 1500, 2000)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    selected = json.loads(D10_SELECTED_FILE.read_text(encoding="utf-8"))
    assert protocol["version"] == "D10B_V1"
    assert protocol["status"] == "FROZEN_BEFORE_D10B_1500_OR_2000_VALIDATION_INSPECTION"
    assert tuple(protocol["diagnostic"]["checkpoints"]) == CHECKPOINTS
    assert protocol["diagnostic"]["single_fit_estimators"] == 2000
    assert protocol["model_lock"]["candidate_id"] == "C03"
    assert protocol["model_lock"]["n_jobs"] == 4
    assert protocol["model_lock"]["early_stopping"] is False
    assert protocol["decision_lock"]["D11_change_allowed_from_D10b"] is False
    assert selected["selected_candidate_id"] == "C03"
    assert selected["selected_n_estimators"] == 1200

    metrics = pd.read_csv(METRICS_FILE)
    assert len(metrics) == 3
    assert tuple(metrics["checkpoint_trees"].astype(int)) == CHECKPOINTS
    assert set(metrics["protocol"]) == {"temporal"}
    assert set(metrics["evaluation_role"]) == {"validation"}
    assert set(metrics["training_rows"]) == {538_461}
    assert set(metrics["validation_rows"]) == {104_258}
    assert set(metrics["training_years"].astype(str)) == {"2018;2019;2020;2021;2022"}
    assert set(metrics["validation_years"].astype(str)) == {"2023"}

    history = pd.read_csv(HISTORY_FILE)
    assert len(history) == 2000
    assert np.array_equal(history["iteration"].to_numpy(), np.arange(1, 2001))
    assert np.isfinite(history["validation_multi_logloss"].to_numpy(dtype=float)).all()
    assert int(history["is_checkpoint"].sum()) == 3
    metric_lookup = metrics.set_index("checkpoint_trees")
    for checkpoint in CHECKPOINTS:
        recorded = float(history.loc[history["iteration"].eq(checkpoint), "validation_multi_logloss"].iloc[0])
        assert np.isclose(
            recorded,
            float(metric_lookup.loc[checkpoint, "validation_multi_logloss"]),
            rtol=0,
            atol=1e-15,
        )

    assignments = pd.read_csv(
        ASSIGNMENTS_FILE,
        dtype={"meta_collision_index": "string"},
        usecols=["meta_collision_index", "meta_collision_year", "temporal_role"],
        low_memory=False,
    ).set_index("meta_collision_index")
    d10_frozen = pd.read_csv(
        D10_PREDICTION_FILE, dtype={"meta_collision_index": "string"}
    )
    prediction_files = sorted(PREDICTION_DIR.glob("*.csv.gz"))
    assert len(prediction_files) == 3
    for checkpoint in CHECKPOINTS:
        path = PREDICTION_DIR / f"temporal__c03__{checkpoint}_trees.csv.gz"
        prediction = pd.read_csv(path, dtype={"meta_collision_index": "string"})
        assert len(prediction) == 104_258
        assert prediction["meta_collision_index"].is_unique
        assert set(prediction["meta_collision_year"]) == {2023}
        assert set(prediction["checkpoint_trees"]) == {checkpoint}
        assert set(prediction["target_severity"]) == {0, 1, 2}
        assert set(prediction["predicted_severity"]).issubset({0, 1, 2})
        assigned = assignments.loc[prediction["meta_collision_index"]]
        assert assigned["temporal_role"].eq("validation").all()
        assert np.array_equal(
            assigned["meta_collision_year"].to_numpy(dtype=int),
            prediction["meta_collision_year"].to_numpy(dtype=int),
        )
        probabilities = prediction[["prob_slight", "prob_serious", "prob_fatal"]].to_numpy(dtype=float)
        assert np.isfinite(probabilities).all()
        assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=2e-8)
        computed = classification_metrics(
            prediction["target_severity"].to_numpy(dtype=int),
            prediction["predicted_severity"].to_numpy(dtype=int),
        )
        for metric, value in computed.items():
            assert np.isclose(
                value, float(metric_lookup.loc[checkpoint, metric]), rtol=0, atol=1e-14
            ), (checkpoint, metric)

        if checkpoint == 1200:
            assert prediction["meta_collision_index"].equals(d10_frozen["meta_collision_index"])
            for column in ("meta_collision_year", "target_severity", "predicted_severity"):
                assert np.array_equal(prediction[column], d10_frozen[column])
            assert np.allclose(
                probabilities,
                d10_frozen[["prob_slight", "prob_serious", "prob_fatal"]].to_numpy(dtype=float),
                rtol=0,
                atol=2e-14,
            )

    d10_metrics = pd.read_csv(D10_METRICS_FILE, dtype={"seed": "string"})
    d10_temporal = d10_metrics.loc[d10_metrics["protocol"].eq("temporal")].iloc[0]
    d10b_1200 = metric_lookup.loc[1200]
    for metric in (
        "macro_f1",
        "qwk",
        "ordinal_mae",
        "accuracy",
        "slight_recall",
        "serious_recall",
        "fatal_recall",
        "serious_or_fatal_recall",
        "mean_asymmetric_cost",
    ):
        assert np.isclose(float(d10b_1200[metric]), float(d10_temporal[metric]), rtol=0, atol=1e-14)

    artifact = joblib.load(MODEL_FILE)
    assert artifact["version"] == "D10B_V1"
    assert artifact["status"] == "D10B_DIAGNOSTIC_ONLY"
    assert artifact["protocol_sha256"] == hash_file(PROTOCOL_FILE)
    assert artifact["candidate_id"] == "C03"
    assert artifact["n_estimators"] == 2000
    assert tuple(artifact["checkpoints"]) == CHECKPOINTS
    assert artifact["training_rows"] == 538_461
    assert artifact["training_years"] == [2018, 2019, 2020, 2021, 2022]
    assert artifact["validation_rows"] == 104_258
    assert artifact["validation_years"] == [2023]
    assert artifact["estimator"].get_params()["n_jobs"] == 4
    assert artifact["estimator"].get_params()["n_estimators"] == 2000

    assert FIGURE_FILE.exists() and FIGURE_FILE.stat().st_size > 10_000
    checkpoint_text = CHECKPOINT_FILE.read_text(encoding="utf-8")
    assert "D11 remains locked" in checkpoint_text
    assert "No post-hoc numerical threshold" in checkpoint_text
    assert "no 2024 row was predicted or evaluated" in checkpoint_text

    manifest = pd.read_csv(MANIFEST_FILE)
    assert manifest["relative_path"].is_unique
    for _, row in manifest.iterrows():
        path = PROJECT_DIR / str(row["relative_path"])
        assert path.exists()
        assert path.stat().st_size == int(row["bytes"])
        assert hash_file(path) == row["sha256"]

    print("D10b checkpoints checked:", len(metrics))
    print("D10b validation prediction rows checked:", 3 * 104_258)
    print("D10b 1200-tree reproduction: PASS")
    print("D10b 2024 embargo: PASS")
    print("INDEPENDENT_D10B_ASSERTIONS=PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"D10B TEST FAILURE: {error}", file=sys.stderr)
        raise
