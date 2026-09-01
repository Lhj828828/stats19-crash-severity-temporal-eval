"""Independent checks for D14 TreeSHAP stability outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from d14_shap_stability import (
    AUDIT_FILE,
    BOOTSTRAP_FILE,
    BOOTSTRAP_ITERATIONS,
    CHECKPOINT_FILE,
    EXPLANATION_ROWS,
    FEATURE_SCOPES,
    FIGURE_FILE,
    IMPORTANCE_FILE,
    MANIFEST_FILE,
    PROJECT_DIR,
    PROTOCOL_FILE,
    RANDOM_SEEDS,
    RUN_SUMMARY_FILE,
    SAMPLE_FILE,
    SHAP_DIR,
    STABILITY_FILE,
    STABILITY_SUMMARY_FILE,
    VERSION,
    build_protocol,
    proportional_allocation,
    quantile_interval,
    spearman_rows,
)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unit_checks() -> None:
    labels = np.array([0] * 70 + [1] * 20 + [2] * 10)
    assert proportional_allocation(labels, 33) == {0: 23, 1: 7, 2: 3}
    first = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
    second = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    observed = spearman_rows(first, second)
    assert np.allclose(observed, [1.0, -0.5])


def main() -> None:
    unit_checks()
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    assert protocol["version"] == VERSION
    assert protocol == build_protocol()
    assert protocol["runtime_preflight"]["importance_or_ranking_inspected"] is False
    assert protocol["scientific_lock"]["new_model_fits_for_H3"] == 0
    assert protocol["regional_holdout"]["status"] == "NOT_PART_OF_H3_AND_NOT_RETROACTIVELY_DEFINED"

    samples = pd.read_csv(SAMPLE_FILE, dtype={"collision_index": "string"})
    assert len(samples) == 6 * EXPLANATION_ROWS
    assert samples["sample_key"].nunique() == 6
    for key, group in samples.groupby("sample_key", sort=False):
        assert len(group) == EXPLANATION_ROWS
        assert group["collision_index"].nunique() == EXPLANATION_ROWS
        assert set(group["target_code"]) == {0, 1, 2}
        if key == "future_2024":
            assert set(group["collision_year"]) == {2024}
        else:
            assert 2024 not in set(group["collision_year"])

    importance = pd.read_csv(IMPORTANCE_FILE)
    stability = pd.read_csv(STABILITY_FILE)
    summary = pd.read_csv(STABILITY_SUMMARY_FILE)
    audit = pd.read_csv(AUDIT_FILE)
    assert len(importance) == 11 * 17 * len(FEATURE_SCOPES)
    assert len(stability) == len(RANDOM_SEEDS) * len(FEATURE_SCOPES)
    assert len(summary) == len(FEATURE_SCOPES)
    assert len(audit) == 11
    assert audit["max_abs_additivity_error"].max() <= 1e-6
    assert set(importance["output_scope"]) == set(FEATURE_SCOPES)
    assert not {"top10_overlap", "top_k_overlap"}.intersection(stability.columns)

    draws = np.load(BOOTSTRAP_FILE)
    assert len(draws.files) == len(RANDOM_SEEDS) * len(FEATURE_SCOPES)
    for row in stability.itertuples(index=False):
        internal = importance[
            importance["split_slug"].eq(f"random_seed_{row.seed}")
            & importance["explanation_cohort"].eq("random_internal_test")
            & importance["output_scope"].eq(row.output_scope)
        ].sort_values("feature")
        future = importance[
            importance["split_slug"].eq(f"random_seed_{row.seed}")
            & importance["explanation_cohort"].eq("future_2024")
            & importance["output_scope"].eq(row.output_scope)
        ].sort_values("feature")
        assert internal["feature"].tolist() == future["feature"].tolist()
        rho = float(spearmanr(internal["mean_abs_shap"], future["mean_abs_shap"]).statistic)
        assert np.isclose(rho, row.spearman_rho, rtol=0, atol=1e-14)
        key = f"seed_{row.seed}__{row.output_scope}"
        values = np.asarray(draws[key], dtype=float)
        assert len(values) == BOOTSTRAP_ITERATIONS
        lower, upper = quantile_interval(values)
        assert np.isclose(lower, row.ci_lower, rtol=0, atol=1e-14)
        assert np.isclose(upper, row.ci_upper, rtol=0, atol=1e-14)

    raw_files = sorted(SHAP_DIR.glob("*.npz"))
    assert len(raw_files) == 11
    for path in raw_files:
        raw = np.load(path)
        assert raw["shap_values"].shape == (EXPLANATION_ROWS, 17, 3)
        assert raw["expected_value"].shape == (3,)
        assert raw["row_positions"].shape == (EXPLANATION_ROWS,)
        assert raw["y_true"].shape == (EXPLANATION_ROWS,)

    run_summary = json.loads(RUN_SUMMARY_FILE.read_text(encoding="utf-8"))
    assert run_summary["status"] == "D14_SHAP_COMPLETE"
    assert run_summary["model_fits"] == 0
    assert run_summary["causal_claim_allowed"] is False
    assert CHECKPOINT_FILE.is_file() and FIGURE_FILE.stat().st_size > 10_000

    manifest = pd.read_csv(MANIFEST_FILE)
    assert len(manifest) == 23
    for row in manifest.itertuples(index=False):
        path = PROJECT_DIR / row.relative_path
        assert path.stat().st_size == int(row.bytes)
        assert hash_file(path) == row.sha256
    print("INDEPENDENT_D14_SHAP_ASSERTIONS=PASS")
    print("D14_IMPORTANCE_ROWS=748 STABILITY_ROWS=20 RAW_SHAP_FILES=11")
    print(f"H3_STATUS={run_summary['H3_status']}")


if __name__ == "__main__":
    main()
