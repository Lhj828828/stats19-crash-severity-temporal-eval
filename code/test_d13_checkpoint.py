"""Independent checks for the D13 frozen tables and positioning record."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from d13_freeze_results import (
    CANDIDATE_MODEL,
    CHECKPOINT_FILE,
    DECISION_FILE,
    D11_MANIFEST_FILE,
    D11_METRICS_FILE,
    D12_INTERVAL_FILE,
    D12_MANIFEST_FILE,
    H1_TABLE_FILE,
    H2_TABLE_FILE,
    MAIN_TABLE_FILE,
    MANIFEST_FILE,
    METRICS,
    MODEL_IDS,
    PRIMARY_MODELS,
    PROJECT_DIR,
    PROTOCOL_FILE,
    REFERENCE_MODEL,
    SUMMARY_FILE,
)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_manifest(path: Path, expected_rows: int) -> None:
    manifest = pd.read_csv(path)
    assert len(manifest) == expected_rows
    for row in manifest.itertuples(index=False):
        artifact = PROJECT_DIR / row.relative_path
        assert artifact.stat().st_size == int(row.bytes)
        assert hash_file(artifact) == row.sha256


def main() -> None:
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    assert protocol["version"] == "D13_V1"
    assert protocol["scientific_lock"]["model_fits"] == 0
    assert protocol["scientific_lock"]["new_bootstrap_draws"] == 0
    assert protocol["decision_rules"]["H3"]["status"] == "PENDING_NOT_EVALUATED"
    for name, expected_hash in protocol["parent_artifacts"].items():
        assert hash_file(PROJECT_DIR / name) == expected_hash, name
    assert_manifest(D11_MANIFEST_FILE, 64)
    assert_manifest(D12_MANIFEST_FILE, 15)

    main_table = pd.read_csv(MAIN_TABLE_FILE)
    h1 = pd.read_csv(H1_TABLE_FILE)
    h2 = pd.read_csv(H2_TABLE_FILE)
    decision = json.loads(DECISION_FILE.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
    d11 = pd.read_csv(D11_METRICS_FILE)
    intervals = pd.read_csv(D12_INTERVAL_FILE)

    assert len(main_table) == len(MODEL_IDS) * len(METRICS) == 35
    assert len(h1) == len(PRIMARY_MODELS) * len(METRICS) == 14
    assert len(h2) == len(METRICS) == 7
    assert main_table.duplicated(["model", "metric"]).sum() == 0
    assert h1.duplicated(["model", "metric"]).sum() == 0
    assert h2.duplicated(["metric"]).sum() == 0

    for row in main_table.itertuples(index=False):
        random_internal = d11[
            d11["evaluation_scope"].eq("random_reference_test")
            & d11["model"].eq(row.model)
        ][row.metric].to_numpy(dtype=float)
        random_future = d11[
            d11["evaluation_scope"].eq("random_model_on_2024")
            & d11["model"].eq(row.model)
        ][row.metric].to_numpy(dtype=float)
        temporal = intervals[
            intervals["group_key"].eq("temporal_test_2024")
            & intervals["model"].eq(row.model)
            & intervals["metric"].eq(row.metric)
        ].iloc[0]
        assert np.isclose(
            row.random_internal_mean_5_seeds,
            random_internal.mean(),
            rtol=0,
            atol=1e-14,
        )
        assert np.isclose(
            row.random_internal_sd_5_seeds,
            random_internal.std(ddof=1),
            rtol=0,
            atol=1e-14,
        )
        assert np.isclose(
            row.same_random_models_2024_mean_5_seeds,
            random_future.mean(),
            rtol=0,
            atol=1e-14,
        )
        assert np.isclose(
            row.strict_temporal_2024_point,
            temporal["point_estimate"],
            rtol=0,
            atol=1e-14,
        )
        assert np.isclose(
            row.strict_temporal_2024_ci_lower,
            temporal["ci_lower"],
            rtol=0,
            atol=1e-14,
        )
        assert np.isclose(
            row.strict_temporal_2024_ci_upper,
            temporal["ci_upper"],
            rtol=0,
            atol=1e-14,
        )

    macro_h1 = h1[
        h1["model"].eq(CANDIDATE_MODEL) & h1["metric"].eq("macro_f1")
    ].iloc[0]
    qwk_h1 = h1[
        h1["model"].eq(CANDIDATE_MODEL) & h1["metric"].eq("qwk")
    ].iloc[0]
    fatal_h1 = h1[
        h1["model"].eq(CANDIDATE_MODEL) & h1["metric"].eq("fatal_recall")
    ].iloc[0]
    assert macro_h1.seed_level_evidence_class == "mixed_or_uncertain"
    assert qwk_h1.seed_level_evidence_class == "consistent_random_optimism"
    assert fatal_h1.seed_level_evidence_class == "consistent_random_optimism"

    macro_h2 = h2[h2["metric"].eq("macro_f1")].iloc[0]
    fatal_h2 = h2[h2["metric"].eq("fatal_recall")].iloc[0]
    assert bool(macro_h2.macro_f1_point_threshold_0_01_pass)
    assert bool(macro_h2.macro_f1_ci_lower_above_zero_pass)
    assert fatal_h2.temporal_interval_evidence_class == "clear_reference_advantage"
    assert macro_h2.candidate == CANDIDATE_MODEL
    assert macro_h2.reference == REFERENCE_MODEL

    assert decision["status"] == "D13_PERFORMANCE_CHECKPOINT_COMPLETE_H3_PENDING"
    assert decision["H1"]["broad_random_split_optimism_claim"] == "NOT_SUPPORTED"
    assert decision["H2"]["macro_f1_practical_and_interval_rule"] == "PASS"
    assert decision["H2"]["joint_incremental_value_rule"] == "FAIL"
    assert decision["H2"]["uniform_lightgbm_superiority_claim"] == "NOT_SUPPORTED"
    assert decision["H3"]["status"] == "PENDING_NOT_EVALUATED"
    assert decision["H3"]["claim_allowed_now"] is False
    assert summary["model_fits"] == 0
    assert summary["new_bootstrap_draws"] == 0
    assert CHECKPOINT_FILE.is_file()

    assert_manifest(MANIFEST_FILE, 9)
    print("INDEPENDENT_D13_ASSERTIONS=PASS")
    print("D13_MAIN_ROWS=35 H1_ROWS=14 H2_ROWS=7")
    print("H3_STATUS=PENDING_NOT_EVALUATED")


if __name__ == "__main__":
    main()
