"""Independent checks for frozen CAS H1/H2 bootstrap artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "code"))

from cas_modeling_common import hash_file  # noqa: E402


PROTOCOL_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_post_analysis_protocol.json"
)
COMPLETE_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_bootstrap_complete.json"
)
EVALUATION_FILE = (
    PROJECT_DIR / "results" / "cas_evaluation" / "test_metrics.csv"
)
RESULT_DIR = PROJECT_DIR / "results" / "cas_post_analysis"
H1_FILE = RESULT_DIR / "h1_bootstrap.csv"
H1_SUMMARY_FILE = RESULT_DIR / "h1_across_seed_summary.csv"
H2_FILE = RESULT_DIR / "h2_bootstrap.csv"
H2_SUMMARY_FILE = RESULT_DIR / "h2_design_summary.csv"
MANIFEST_FILE = (
    PROJECT_DIR / "logs" / "cas" / "cas_bootstrap_artifact_manifest.csv"
)


def main() -> None:
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    complete = json.loads(COMPLETE_FILE.read_text(encoding="utf-8"))
    assert protocol["status"] == (
        "OPERATIONAL_SUPPLEMENT_FROZEN_BEFORE_BOOTSTRAP_OR_SHAP_RESULTS"
    )
    assert protocol["common"]["iterations"] == 2_000
    assert protocol["common"]["base_seed"] == 20_260_901
    assert complete["status"] == "H1_H2_BOOTSTRAP_COMPLETE"
    assert complete["protocol_sha256"] == hash_file(PROTOCOL_FILE)
    assert complete["post_test_model_selection"] is False

    h1 = pd.read_csv(H1_FILE)
    h1_summary = pd.read_csv(H1_SUMMARY_FILE)
    h2 = pd.read_csv(H2_FILE, dtype={"seed": "string"})
    h2_summary = pd.read_csv(H2_SUMMARY_FILE)
    assert len(h1) == 5 * 3 * 9 == 135
    assert len(h1_summary) == 3 * 9 == 27
    assert len(h2) == 11 * 9 == 99
    assert len(h2_summary) == 3 * 9 == 27
    assert set(h1["bootstrap_design"]) == {"independent_stratified"}
    assert set(h2["bootstrap_design"]) == {"paired_stratified"}
    assert set(h1["valid_iterations"]) == {2_000}
    assert set(h2["valid_iterations"]) == {2_000}
    assert np.isfinite(
        h1[["point_gap", "ci_lower", "ci_upper"]].to_numpy(float)
    ).all()
    assert np.isfinite(
        h2[["point_difference", "ci_lower", "ci_upper"]].to_numpy(float)
    ).all()
    assert (h1["ci_lower"] <= h1["ci_upper"]).all()
    assert (h2["ci_lower"] <= h2["ci_upper"]).all()

    evaluation = pd.read_csv(EVALUATION_FILE, dtype={"seed": "string"}).set_index(
        ["evaluation_group", "model"]
    )
    for _, row in h1.iterrows():
        seed = str(row["seed"])
        model = row["model"]
        metric = row["metric"]
        internal = evaluation.loc[
            (f"random_seed_{seed}_internal_test", model), metric
        ]
        future = evaluation.loc[
            (f"random_seed_{seed}_2025_diagnostic", model), metric
        ]
        assert np.isclose(float(row["internal_value"]), internal, atol=1e-14)
        assert np.isclose(float(row["future_2025_value"]), future, atol=1e-14)
        assert np.isclose(float(row["point_gap"]), internal - future, atol=1e-14)
    for _, row in h2.iterrows():
        group = row["evaluation_group"]
        metric = row["metric"]
        logistic = evaluation.loc[(group, "logistic_weighted"), metric]
        lightgbm = evaluation.loc[(group, "lightgbm_weighted"), metric]
        assert np.isclose(float(row["logistic_value"]), logistic, atol=1e-14)
        assert np.isclose(float(row["lightgbm_value"]), lightgbm, atol=1e-14)
        assert np.isclose(
            float(row["point_difference"]),
            lightgbm - logistic,
            atol=1e-14,
        )

    manifest = pd.read_csv(MANIFEST_FILE)
    assert manifest["relative_path"].is_unique
    for _, row in manifest.iterrows():
        path = PROJECT_DIR / str(row["relative_path"])
        assert path.stat().st_size == int(row["bytes"])
        assert hash_file(path) == row["sha256"]

    print("CAS H1 bootstrap rows:", len(h1))
    print("CAS H2 bootstrap rows:", len(h2))
    print("CAS_BOOTSTRAP_INDEPENDENT_CHECKS=PASS")


if __name__ == "__main__":
    main()
