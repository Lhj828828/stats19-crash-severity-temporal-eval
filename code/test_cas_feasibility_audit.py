"""Regression checks for the frozen CAS feasibility evidence package."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw" / "cas"
LOG_DIR = PROJECT_DIR / "logs" / "cas"
CONFIG_DIR = PROJECT_DIR / "config" / "cas"


def load_protocol() -> dict[str, object]:
    return json.loads(
        (CONFIG_DIR / "cas_feasibility_protocol.json").read_text(encoding="utf-8")
    )


def test_snapshot_shape_and_target() -> None:
    data = pd.read_csv(
        RAW_DIR / "cas_injury_2022_2025_snapshot.csv.gz",
        keep_default_na=False,
        low_memory=False,
    )
    assert data.shape == (43_121, 70)
    assert data["OBJECTID"].nunique() == len(data)
    assert set(data["crashYear"].astype(int)) == {2022, 2023, 2024, 2025}
    assert set(data["crashSeverity"]) == {
        "Minor Crash",
        "Serious Crash",
        "Fatal Crash",
    }


def test_literal_none_and_null_are_preserved() -> None:
    data = pd.read_csv(
        RAW_DIR / "cas_injury_2022_2025_snapshot.csv.gz",
        keep_default_na=False,
        low_memory=False,
    )
    assert (data["streetLight"] == "None").sum() == 14_219
    assert (data["streetLight"] == "Null").sum() == 670
    assert (data["weatherB"] == "None").sum() == 83
    assert (data["weatherB"] == "Null").sum() == 41_122


def test_jsonl_is_lossless_and_complete() -> None:
    path = RAW_DIR / "cas_injury_2022_2025_snapshot.jsonl.gz"
    rows = 0
    null_crash_sh = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows += 1
            null_crash_sh += row["crashSHDescription"] is None
    assert rows == 43_121
    assert null_crash_sh == 3


def test_leakage_audit_covers_live_schema() -> None:
    audit = pd.read_csv(LOG_DIR / "cas_leakage_audit.csv")
    assert len(audit) == 70
    assert audit["field_name"].nunique() == 70
    assert set(audit.loc[audit["decision"] == "EXCLUDE_OUTCOME", "field_name"]) == {
        "fatalCount",
        "minorInjuryCount",
        "seriousInjuryCount",
    }
    assert set(
        audit.loc[audit["decision"] == "EXCLUDE_UNDOCUMENTED", "field_name"]
    ) == {"crashRoadSideRoad", "intersection"}
    assert audit.loc[audit["decision"] == "INCLUDE_CANDIDATE"].shape[0] == 15
    target_row = audit.loc[audit["field_name"] == "crashSeverity"].iloc[0]
    assert target_row["model_handling"] == "map injury crashes to Minor < Serious < Fatal"


def test_protocol_is_separate_and_temporal() -> None:
    protocol = load_protocol()
    temporal = protocol["candidate_temporal_protocol"]
    assert temporal["train_years"] == [2022, 2023]
    assert temporal["validation_years"] == [2024]
    assert temporal["test_years"] == [2025]
    assert protocol["excluded_severity"] == "Non-Injury Crash"
    assert protocol["analysis_target_order"] == ["Minor", "Serious", "Fatal"]
    assert protocol["target_mapping"]["Minor Crash"] == "Minor"
    assert "Do not pool" in protocol["comparison_rule"]
    assert protocol["raw_snapshot"]["row_count"] == 43_121


def test_no_unseen_categories_in_frozen_splits() -> None:
    coverage = pd.read_csv(LOG_DIR / "cas_category_coverage_by_split.csv")
    assert coverage["unseen_row_count"].sum() == 0


def test_crosswalk_prevents_false_external_validation_claim() -> None:
    crosswalk = pd.read_csv(LOG_DIR / "cas_stats19_feature_crosswalk.csv")
    assert len(crosswalk) == 22
    assert set(crosswalk["relationship"]) == {
        "close_concept",
        "partial",
        "none",
        "cas_only",
    }
    assert (crosswalk["relationship"] == "none").sum() == 9
    protocol = load_protocol()
    assert "not an external test set" in protocol["feature_alignment"]["rule"]


def test_snapshot_hash_matches_manifest_protocol_and_report() -> None:
    path = RAW_DIR / "cas_injury_2022_2025_snapshot.csv.gz"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    protocol = load_protocol()
    manifest = pd.read_csv(LOG_DIR / "cas_source_manifest.csv")
    manifest_hash = manifest.loc[
        manifest["artifact_role"] == "official_api_attribute_snapshot", "sha256"
    ].item()
    report = (LOG_DIR / "CAS_feasibility_audit_report.md").read_text(encoding="utf-8")
    assert actual == protocol["raw_snapshot"]["sha256"]
    assert actual == manifest_hash
    assert f"快照SHA-256：`{actual}`" in report


def main() -> None:
    checks = [
        test_snapshot_shape_and_target,
        test_literal_none_and_null_are_preserved,
        test_jsonl_is_lossless_and_complete,
        test_leakage_audit_covers_live_schema,
        test_protocol_is_separate_and_temporal,
        test_no_unseen_categories_in_frozen_splits,
        test_crosswalk_prevents_false_external_validation_claim,
        test_snapshot_hash_matches_manifest_protocol_and_report,
    ]
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")
    print(f"All {len(checks)} CAS feasibility checks passed.")


if __name__ == "__main__":
    main()
