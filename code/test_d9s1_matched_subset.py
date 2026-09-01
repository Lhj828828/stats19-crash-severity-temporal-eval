"""Independent checks for D9-S1 matched-subset sensitivity artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

from d9_train_ordered_logit import (
    BENCHMARK_TRAIN_ROWS,
    SUBSET_SEED,
    load_aligned_data,
    load_contracts,
    make_split_specs,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROTOCOL_FILE = PROJECT_DIR / "config" / "d9s1_matched_subset_protocol.json"
ASSIGNMENTS_FILE = PROJECT_DIR / "data" / "processed" / "d6_split_assignments.csv.gz"
SUBSET_DIR = PROJECT_DIR / "data" / "processed" / "d9s1_matched_subsets"
MODEL_DIR = PROJECT_DIR / "models" / "d9s1"
RESULT_DIR = PROJECT_DIR / "results" / "d9s1"
METRICS_FILE = RESULT_DIR / "d9s1_validation_metrics.csv"
CONFUSION_FILE = RESULT_DIR / "d9s1_validation_confusion_matrices.csv"
COMPARISON_FILE = RESULT_DIR / "d9s1_matched_comparison.csv"
PREDICTION_DIR = RESULT_DIR / "validation_predictions"
D9_PREDICTION_DIR = PROJECT_DIR / "results" / "d9" / "validation_predictions"
SUBSET_AUDIT_FILE = PROJECT_DIR / "logs" / "d9s1_subset_manifest.csv"
MODEL_NAME = "logistic_unweighted_matched_100k"
TARGET_CODES = [0, 1, 2]


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_row_identity(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.itertuples(index=False):
        digest.update(
            f"{int(row.row_position)}\t{row.meta_collision_index}\t{int(row.target_severity)}\n".encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def slug_for_spec(spec: dict[str, str]) -> str:
    return "temporal" if spec["protocol"] == "temporal" else f"random_seed_{spec['seed']}"


def seed_for_spec(spec: dict[str, str]) -> int:
    return SUBSET_SEED if spec["protocol"] == "temporal" else SUBSET_SEED + int(spec["seed"])


def independently_reconstruct_positions(
    spec: dict[str, str], assignments: pd.DataFrame, target: pd.Series
) -> np.ndarray:
    roles = assignments[spec["role_column"]]
    available = np.flatnonzero(roles.eq("train").to_numpy())
    selected, _ = train_test_split(
        available,
        train_size=BENCHMARK_TRAIN_ROWS,
        random_state=seed_for_spec(spec),
        stratify=target.iloc[available].to_numpy(),
    )
    return np.sort(selected)


def main() -> None:
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    assert protocol["version"] == "D9S1_V1"
    assert protocol["status"] == "FROZEN_BEFORE_MATCHED_SENSITIVITY_RESULTS"
    assert protocol["comparator"]["class_weight"] is None
    assert protocol["matched_subset"]["training_rows_per_split"] == 100_000
    assert protocol["reporting_role"].startswith("appendix")
    assert protocol["upstream"]["D6_assignments_sha256"] == hash_file(ASSIGNMENTS_FILE)

    schema, d6, _ = load_contracts()
    _, target, metadata, assignments = load_aligned_data(schema, d6)
    specs = make_split_specs(d6)
    assert len(specs) == 6
    subset_audit = pd.read_csv(SUBSET_AUDIT_FILE, dtype={"seed": "string"})
    assert len(subset_audit) == 6

    for spec in specs:
        slug = slug_for_spec(spec)
        expected = independently_reconstruct_positions(spec, assignments, target)
        manifest_path = SUBSET_DIR / f"{slug}.csv.gz"
        manifest = pd.read_csv(manifest_path, dtype={"meta_collision_index": "string"})
        assert len(manifest) == 100_000
        assert manifest["meta_collision_index"].is_unique
        assert manifest["meta_collision_year"].max() <= 2023
        assert np.array_equal(manifest["row_position"].to_numpy(), expected)
        expected_ids = metadata.iloc[expected]["meta_collision_index"].astype("string").reset_index(drop=True)
        assert manifest["meta_collision_index"].equals(expected_ids)
        assert np.array_equal(
            manifest["target_severity"].to_numpy(),
            target.iloc[expected].to_numpy(),
        )
        audit = subset_audit.loc[subset_audit["split_slug"].eq(slug)].iloc[0]
        assert int(audit["subset_seed"]) == seed_for_spec(spec)
        assert audit["manifest_file_sha256"] == hash_file(manifest_path)
        assert audit["ordered_row_identity_sha256"] == hash_row_identity(manifest)

    metrics = pd.read_csv(METRICS_FILE, dtype={"seed": "string"})
    assert len(metrics) == 6
    assert set(metrics["model"]) == {MODEL_NAME}
    assert set(metrics["training_rows"]) == {100_000}
    assert set(metrics["training_mode"]) == {"matched_d9_subset"}
    assert not metrics["convergence_warning"].astype(bool).any()
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

    lookup = metrics.set_index(["protocol", "seed", "model"])
    prediction_files = sorted(PREDICTION_DIR.glob("*.csv.gz"))
    assert len(prediction_files) == 6
    total_prediction_rows = 0
    for path in prediction_files:
        prediction = pd.read_csv(path, dtype={"meta_collision_index": "string"})
        total_prediction_rows += len(prediction)
        assert prediction["meta_collision_index"].is_unique
        assert prediction["meta_collision_year"].max() <= 2023
        probabilities = prediction[["prob_slight", "prob_serious", "prob_fatal"]].to_numpy()
        assert np.isfinite(probabilities).all()
        assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=2e-8)
        slug = path.name.removesuffix(".csv.gz").split("__", maxsplit=1)[0]
        if slug == "temporal":
            protocol_name, seed = "temporal", "year_based"
        else:
            protocol_name, seed = "random_reference", slug.removeprefix("random_seed_")
        paired_d9 = pd.read_csv(
            D9_PREDICTION_DIR / f"{slug}__ordered_logit_unweighted.csv.gz",
            usecols=["meta_collision_index", "target_severity"],
            dtype={"meta_collision_index": "string"},
        )
        assert prediction["meta_collision_index"].equals(paired_d9["meta_collision_index"])
        assert np.array_equal(prediction["target_severity"], paired_d9["target_severity"])
        row = lookup.loc[(protocol_name, seed, MODEL_NAME)]
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
    assert len(confusion) == 54
    assert int(confusion["count"].sum()) == total_prediction_rows
    comparison = pd.read_csv(COMPARISON_FILE, dtype={"seed": "string"})
    assert len(comparison) == 6
    assert set(comparison["matched_logistic_training_rows"]) == {100_000}
    assert set(comparison["ordered_logit_training_rows"]) == {100_000}
    assert np.array_equal(
        comparison["validation_rows"].to_numpy(),
        comparison["ordered_validation_rows"].to_numpy(),
    )

    for spec in specs:
        slug = slug_for_spec(spec)
        preprocessing = joblib.load(MODEL_DIR / slug / "preprocessing_bundle.joblib")
        model = joblib.load(MODEL_DIR / slug / f"{MODEL_NAME}.joblib")
        manifest = subset_audit.loc[subset_audit["split_slug"].eq(slug)].iloc[0]
        assert preprocessing["version"] == "D9S1_V1"
        assert model["version"] == "D9S1_V1"
        assert model["class_weight"] is None
        assert model["training_rows"] == 100_000
        assert model["training_subset_identity_sha256"] == manifest["ordered_row_identity_sha256"]
        assert preprocessing["training_subset_identity_sha256"] == manifest["ordered_row_identity_sha256"]

    print("D9-S1 subset identities checked:", 600_000)
    print("D9-S1 validation prediction rows checked:", total_prediction_rows)
    print("D9-S1 2024 embargo: PASS")
    print("INDEPENDENT_D9S1_ASSERTIONS=PASS")


if __name__ == "__main__":
    main()
