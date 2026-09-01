"""Independent checks for the CAS replication closeout."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "code"))

from cas_finalize_replication import (  # noqa: E402
    CHECKPOINT_FILE,
    COMPLETE_FILE,
    MANIFEST_FILE,
    PROTOCOL_FILE,
    RESULT_FILE,
    VERSION,
    resolve_protocol_path,
)
from cas_modeling_common import hash_file  # noqa: E402


WORKSPACE_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_workspace_manifest.json"
)


def main() -> None:
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    complete = json.loads(COMPLETE_FILE.read_text(encoding="utf-8"))
    assert protocol["status"] == "POST_RESULT_DETERMINISTIC_REPORTING_CONTRACT"
    assert complete["version"] == VERSION
    assert complete["status"] == "CAS_REPLICATION_ANALYSIS_COMPLETE"
    assert complete["protocol_sha256"] == hash_file(PROTOCOL_FILE)
    assert complete["one_time_2025_evaluation_status"] == (
        "FROZEN_MODELS_EVALUATED_ONCE"
    )
    assert complete["post_test_model_selection"] is False
    assert complete["new_inferential_test"] is False
    assert complete["records_pooled"] is False
    assert complete["cross_national_generalizability_claim_allowed"] is False

    for scope in ("cas_upstream", "stats19_upstream"):
        for item in protocol[scope].values():
            path = resolve_protocol_path(item["path"])
            assert path.exists()
            assert hash_file(path) == item["sha256"]

    comparison = pd.read_csv(RESULT_FILE)
    assert len(comparison) == complete["comparison_rows"] == 8
    assert comparison[["question", "component"]].duplicated().sum() == 0
    expected_labels = {
        ("H1", "broad_metric_pattern"): "broad_pattern_agreement",
        ("H1", "macro_f1"): "different_direction_near_zero_or_mixed",
        ("H1", "qwk"): "direction_same_evidence_weaker_in_cas",
        ("H1", "fatal_recall"): "direction_same_evidence_weaker_in_cas",
        ("H2", "macro_f1"): "not_replicated",
        ("H2", "fatal_recall"): "direction_replicated",
        ("H3", "overall_mean_absolute_shap_rank_rho"): (
            "different_observed_pattern"
        ),
        ("workflow", "second_source_executability"): (
            "executability_demonstrated"
        ),
    }
    observed_labels = {
        (row.question, row.component): row.replication_label
        for row in comparison.itertuples(index=False)
    }
    assert observed_labels == expected_labels
    assert set(comparison["replication_label"]).issubset(
        set(protocol["allowed_labels"])
    )

    stats19 = json.loads(
        resolve_protocol_path(
            protocol["stats19_upstream"]["final_summary"]["path"]
        ).read_text(encoding="utf-8")
    )
    h2 = pd.read_csv(
        resolve_protocol_path(
            protocol["cas_upstream"]["h2_bootstrap"]["path"]
        )
    ).set_index(["evaluation_group", "metric"])
    cas_h3 = json.loads(
        resolve_protocol_path(
            protocol["cas_upstream"]["h3_complete"]["path"]
        ).read_text(encoding="utf-8")
    )
    for metric in ("macro_f1", "fatal_recall"):
        row = comparison.loc[
            comparison["question"].eq("H2")
            & comparison["component"].eq(metric)
        ].iloc[0]
        source = h2.loc[("temporal_2025", metric)]
        assert np.isclose(row["cas_point"], source["point_difference"], atol=1e-14)
        assert np.isclose(row["cas_ci_lower"], source["ci_lower"], atol=1e-14)
        assert np.isclose(row["cas_ci_upper"], source["ci_upper"], atol=1e-14)
        assert np.isclose(row["stats19_point"], stats19["H2"][metric]["delta"])
        assert np.isclose(
            row["stats19_ci_lower"], stats19["H2"][metric]["ci"][0]
        )
        assert np.isclose(
            row["stats19_ci_upper"], stats19["H2"][metric]["ci"][1]
        )
    h3_row = comparison.loc[comparison["question"].eq("H3")].iloc[0]
    assert np.isclose(h3_row["stats19_point"], stats19["H3"]["overall_rho_mean"])
    assert np.isclose(h3_row["cas_point"], cas_h3["rho_mean"])
    assert h3_row["cas_point"] > 0.99
    assert h3_row["stats19_point"] < 0.7

    manifest = pd.read_csv(MANIFEST_FILE)
    assert manifest["relative_path"].is_unique
    for _, row in manifest.iterrows():
        path = PROJECT_DIR / str(row["relative_path"])
        assert path.exists()
        assert path.stat().st_size == int(row["bytes"])
        assert hash_file(path) == row["sha256"]
    assert CHECKPOINT_FILE.exists()
    checkpoint = CHECKPOINT_FILE.read_text(encoding="utf-8")
    assert "not replicated" in checkpoint
    assert "direction replicated" in checkpoint
    assert "do not prove cross-national" in checkpoint

    workspace = json.loads(WORKSPACE_FILE.read_text(encoding="utf-8"))
    assert workspace["status"] == "CAS_REPLICATION_ANALYSIS_COMPLETE"
    assert workspace["modeling_status"]["models_trained"] is True
    assert workspace["modeling_status"]["test_model_evaluated"] is True
    assert workspace["modeling_status"]["one_time_2025_evaluation"] is True
    print("CAS closeout comparison rows:", len(comparison))
    print("CAS_CLOSEOUT_INDEPENDENT_CHECKS=PASS")


if __name__ == "__main__":
    main()
