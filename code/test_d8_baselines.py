"""Independent validation of D8 artifacts and the 2024 embargo."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_DIR / "config" / "d8_baseline_protocol.json"
D7_CONFIG_FILE = PROJECT_DIR / "config" / "d7_training_inputs.json"
ASSIGNMENTS_FILE = PROJECT_DIR / "data" / "processed" / "d6_split_assignments.csv.gz"
METRICS_FILE = PROJECT_DIR / "results" / "d8" / "d8_validation_metrics.csv"
CONFUSION_FILE = PROJECT_DIR / "results" / "d8" / "d8_validation_confusion_matrices.csv"
PREDICTION_DIR = PROJECT_DIR / "results" / "d8" / "validation_predictions"
MODEL_DIR = PROJECT_DIR / "models" / "d8"
TARGET_CODES = [0, 1, 2]
MODELS = {"dummy_most_frequent", "logistic_unweighted", "logistic_weighted"}


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    d7 = json.loads(D7_CONFIG_FILE.read_text(encoding="utf-8"))
    assert config["version"] == "D8_V1"
    assert config["status"] == "BASELINE_SPECIFICATION_FROZEN_BEFORE_VALIDATION_INSPECTION"
    assert config["upstream"]["D6_assignments_sha256"] == hash_file(ASSIGNMENTS_FILE)
    assert config["upstream"]["D6_assignments_sha256"] == d7["upstream"]["D6_assignments_sha256"]
    assert config["prediction_rule"].startswith("argmax")
    assert config["primary_future_comparison"].startswith("logistic_weighted")
    assert config["features"]["allowlist"] == d7["feature_contract"]["feature_columns"]
    assert not set(config["features"]["allowlist"]).intersection(
        config["features"]["metadata_excluded"]
    )

    metrics = pd.read_csv(METRICS_FILE)
    assert len(metrics) == 18
    assert set(metrics["model"]) == MODELS
    assert set(metrics["evaluation_role"]) == {"validation"}
    assert not metrics["validation_years"].astype(str).str.contains("2024").any()
    assert not metrics["training_years"].astype(str).str.contains("2024").any()
    assert not metrics["convergence_warning"].astype(bool).any()
    assert np.isfinite(
        metrics[
            [
                "macro_f1",
                "qwk",
                "ordinal_mae",
                "fatal_recall",
                "serious_or_fatal_recall",
                "mean_asymmetric_cost",
            ]
        ].to_numpy(dtype=float)
    ).all()
    temporal = metrics.loc[metrics["protocol"].eq("temporal")]
    assert len(temporal) == 3
    assert set(temporal["training_rows"]) == {538_461}
    assert set(temporal["validation_rows"]) == {104_258}
    random_metrics = metrics.loc[metrics["protocol"].eq("random_reference")]
    assert len(random_metrics) == 15
    assert set(random_metrics["training_rows"]) == {449_903}
    assert set(random_metrics["validation_rows"]) == {96_408}

    prediction_files = sorted(PREDICTION_DIR.glob("*.csv.gz"))
    assert len(prediction_files) == 18
    metric_lookup = metrics.set_index(["protocol", "seed", "model"])
    total_prediction_rows = 0
    for path in prediction_files:
        prediction = pd.read_csv(path, dtype={"meta_collision_index": "string"})
        total_prediction_rows += len(prediction)
        assert prediction["meta_collision_index"].is_unique
        assert prediction["meta_collision_year"].max() <= 2023
        assert set(prediction["target_severity"].unique()) == set(TARGET_CODES)
        assert set(prediction["predicted_severity"].unique()).issubset(TARGET_CODES)
        probabilities = prediction[["prob_slight", "prob_serious", "prob_fatal"]].to_numpy()
        assert np.isfinite(probabilities).all()
        assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=2e-8)

        stem = path.name.removesuffix(".csv.gz")
        split_slug, model = stem.split("__", maxsplit=1)
        if split_slug == "temporal":
            protocol, seed = "temporal", "year_based"
        else:
            protocol, seed = "random_reference", split_slug.removeprefix("random_seed_")
        row = metric_lookup.loc[(protocol, str(seed), model)]
        y_true = prediction["target_severity"].to_numpy()
        y_pred = prediction["predicted_severity"].to_numpy()
        macro_f1 = f1_score(y_true, y_pred, labels=TARGET_CODES, average="macro", zero_division=0)
        qwk = cohen_kappa_score(y_true, y_pred, labels=TARGET_CODES, weights="quadratic")
        fatal_recall = recall_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0]
        assert np.isclose(macro_f1, float(row["macro_f1"]), rtol=0, atol=1e-14)
        assert np.isclose(qwk, float(row["qwk"]), rtol=0, atol=1e-14)
        assert np.isclose(fatal_recall, float(row["fatal_recall"]), rtol=0, atol=1e-14)

    confusion = pd.read_csv(CONFUSION_FILE)
    assert len(confusion) == 18 * 9
    assert int(confusion["count"].sum()) == total_prediction_rows

    split_dirs = [MODEL_DIR / "temporal"] + [
        MODEL_DIR / f"random_seed_{seed}" for seed in (1103, 2207, 3301, 4409, 5501)
    ]
    for split_dir in split_dirs:
        bundle = joblib.load(split_dir / "preprocessing_bundle.joblib")
        assert bundle["version"] == "D8_V1"
        assert 2024 not in bundle["training_years"]
        assert len(bundle["feature_columns"]) == 17
        assert len(bundle["encoded_feature_names"]) > 17
        for model in MODELS:
            artifact = joblib.load(split_dir / f"{model}.joblib")
            assert artifact["model_name"] == model
            assert artifact["target_codes"] == TARGET_CODES

    print("D8 validation metric rows:", len(metrics))
    print("D8 prediction rows checked:", total_prediction_rows)
    print("D8 2024 embargo: PASS")
    print("INDEPENDENT_D8_ASSERTIONS=PASS")


if __name__ == "__main__":
    main()
