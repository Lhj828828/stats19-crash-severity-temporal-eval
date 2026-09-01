"""Independent checks for the D9 ordered-logit validation artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_DIR / "config" / "d9_ordered_logit_protocol.json"
BENCHMARK_FILE = PROJECT_DIR / "logs" / "d9_runtime_benchmark.json"
ASSIGNMENTS_FILE = PROJECT_DIR / "data" / "processed" / "d6_split_assignments.csv.gz"
METRICS_FILE = PROJECT_DIR / "results" / "d9" / "d9_validation_metrics.csv"
CONFUSION_FILE = PROJECT_DIR / "results" / "d9" / "d9_validation_confusion_matrices.csv"
PREDICTION_DIR = PROJECT_DIR / "results" / "d9" / "validation_predictions"
MODEL_DIR = PROJECT_DIR / "models" / "d9"
MODEL_NAME = "ordered_logit_unweighted"
TARGET_CODES = [0, 1, 2]


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    assert config["version"] == "D9_V3"
    assert config["status"] == "SPECIFICATION_FROZEN_AFTER_RUNTIME_PREFLIGHT"
    assert config["model"]["class_weight"] is None
    assert config["model"]["distribution"] == "logit"
    assert config["upstream"]["D6_assignments_sha256"] == hash_file(ASSIGNMENTS_FILE)
    assert "2024" in config["evaluation_scope"]["forbidden"]
    benchmark = json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))
    assert benchmark["protocol_sha256"] == hash_file(CONFIG_FILE)
    assert benchmark["converged"] is True
    assert benchmark["execution_decision"] == "subset_100000"
    assert benchmark["validation_performance_inspected"] is False

    metrics = pd.read_csv(METRICS_FILE)
    assert len(metrics) == 6
    assert set(metrics["model"]) == {MODEL_NAME}
    assert set(metrics["evaluation_role"]) == {"validation"}
    assert set(metrics["training_mode"]).issubset({"full", "subset"})
    assert set(metrics["training_mode"]) == {"subset"}
    assert metrics["converged"].astype(bool).all()
    assert not metrics["training_years"].astype(str).str.contains("2024").any()
    assert not metrics["validation_years"].astype(str).str.contains("2024").any()
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
    random = metrics.loc[metrics["protocol"].eq("random_reference")]
    assert len(temporal) == 1 and len(random) == 5
    assert set(temporal["validation_rows"]) == {104_258}
    assert set(random["validation_rows"]) == {96_408}
    if set(metrics["training_mode"]) == {"subset"}:
        assert set(metrics["training_rows"]) == {100_000}
    else:
        assert int(temporal.iloc[0]["training_rows"]) == 538_461
        assert set(random["training_rows"]) == {449_903}

    prediction_files = sorted(PREDICTION_DIR.glob("*.csv.gz"))
    assert len(prediction_files) == 6
    lookup = metrics.set_index(["protocol", "seed", "model"])
    total_rows = 0
    for path in prediction_files:
        prediction = pd.read_csv(path, dtype={"meta_collision_index": "string"})
        total_rows += len(prediction)
        assert prediction["meta_collision_index"].is_unique
        assert prediction["meta_collision_year"].max() <= 2023
        assert set(prediction["target_severity"].unique()) == set(TARGET_CODES)
        probabilities = prediction[["prob_slight", "prob_serious", "prob_fatal"]].to_numpy()
        assert np.isfinite(probabilities).all()
        assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=2e-8)
        assert probabilities.min() >= -1e-12
        split_slug = path.name.removesuffix(".csv.gz").split("__", maxsplit=1)[0]
        if split_slug == "temporal":
            protocol, seed = "temporal", "year_based"
        else:
            protocol, seed = "random_reference", split_slug.removeprefix("random_seed_")
        row = lookup.loc[(protocol, str(seed), MODEL_NAME)]
        y_true = prediction["target_severity"].to_numpy()
        y_pred = prediction["predicted_severity"].to_numpy()
        assert np.isclose(
            f1_score(y_true, y_pred, labels=TARGET_CODES, average="macro", zero_division=0),
            float(row["macro_f1"]),
            rtol=0,
            atol=1e-14,
        )
        assert np.isclose(
            cohen_kappa_score(y_true, y_pred, labels=TARGET_CODES, weights="quadratic"),
            float(row["qwk"]),
            rtol=0,
            atol=1e-14,
        )
        assert np.isclose(
            recall_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0],
            float(row["fatal_recall"]),
            rtol=0,
            atol=1e-14,
        )

    confusion = pd.read_csv(CONFUSION_FILE)
    assert len(confusion) == 6 * 9
    assert int(confusion["count"].sum()) == total_rows
    split_dirs = [MODEL_DIR / "temporal"] + [
        MODEL_DIR / f"random_seed_{seed}" for seed in (1103, 2207, 3301, 4409, 5501)
    ]
    for split_dir in split_dirs:
        preprocessing = joblib.load(split_dir / "preprocessing_bundle.joblib")
        model = joblib.load(split_dir / f"{MODEL_NAME}.joblib")
        assert preprocessing["version"] == "D9_V3"
        assert 2024 not in preprocessing["training_years"]
        assert model["version"] == "D9_V3"
        assert model["class_weight"] is None
        assert model["converged"]
        assert len(model["beta"]) == model["feature_count"]
        assert len(model["finite_thresholds"]) == 2
        assert np.diff(model["finite_thresholds"])[0] > 0

    print("D9 validation metric rows:", len(metrics))
    print("D9 prediction rows checked:", total_rows)
    print("D9 2024 embargo: PASS")
    print("INDEPENDENT_D9_ASSERTIONS=PASS")


if __name__ == "__main__":
    main()
