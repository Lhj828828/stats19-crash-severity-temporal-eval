"""Audit a fixed CAS snapshot without querying the live ArcGIS service.

The public CAS workflow starts from the lossless JSONL snapshot supplied by the
researcher. Official metadata are read from the repository, while all audit
tables are rebuilt in the current workspace. The live acquisition script is
kept separate so a public run cannot silently replace the fixed snapshot.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from cas_feasibility_audit import (
    CAS_CONFIG_DIR,
    CAS_DOC_DIR,
    CAS_LOG_DIR,
    CAS_RAW_DIR,
    CAS_SYSTEM_PAGE,
    FIELD_DESCRIPTION_PAGE,
    FIELD_PAGE_ID,
    FIELD_PAGE_URL,
    INCLUDE_CATEGORICAL,
    INCLUDE_FIELDS,
    INCLUDE_NUMERIC,
    ITEM_ID,
    ITEM_URL,
    LANDING_PAGE,
    LAYER_URL,
    NZ_GLOSSARY_PAGE,
    QUERY_URL,
    SERVICE_URL,
    SNAPSHOT_YEARS,
    TARGET_MAP,
    STATS19_FEATURE_CROSSWALK,
    build_categorical_drift,
    build_category_coverage,
    build_documentation_value_checks,
    build_leakage_audit,
    build_quality_outputs,
    md5_file,
    parse_documented_fields,
    relative,
    sha256_file,
    utc_now,
    write_raw_snapshot,
)


RAW_JSONL = CAS_RAW_DIR / "cas_injury_2022_2025_snapshot.jsonl.gz"
RAW_CSV = CAS_RAW_DIR / "cas_injury_2022_2025_snapshot.csv.gz"


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_lossless_snapshot() -> tuple[list[dict[str, object]], list[str]]:
    if not RAW_JSONL.is_file():
        raise FileNotFoundError(
            "Fixed CAS JSONL snapshot is missing: "
            f"{RAW_JSONL}. Supply the snapshot before running the public CAS entry."
        )
    rows: list[dict[str, object]] = []
    with gzip.open(RAW_JSONL, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Snapshot line {line_number} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError("The fixed CAS JSONL snapshot is empty")

    fields = list(rows[0])
    if not fields or len(fields) != len(set(fields)):
        raise ValueError("The fixed CAS snapshot has an invalid first-row schema")
    for row_number, row in enumerate(rows, start=1):
        if list(row) != fields:
            raise ValueError(f"Snapshot field order changed at line {row_number}")
    return rows, fields


def metadata_manifest() -> tuple[dict[str, object], list[dict[str, object]]]:
    metadata = {
        "cas_item_metadata.json": (ITEM_URL + "?f=pjson", "official_metadata_snapshot"),
        "cas_service_metadata.json": (SERVICE_URL + "?f=pjson", "official_metadata_snapshot"),
        "cas_layer_metadata.json": (LAYER_URL + "?f=pjson", "official_metadata_snapshot"),
        "cas_field_page_item.json": (FIELD_PAGE_URL + "?f=pjson", "official_metadata_snapshot"),
        "cas_field_page_data.json": (
            FIELD_PAGE_URL + "/data?f=pjson",
            "official_metadata_snapshot",
        ),
    }
    payloads: dict[str, object] = {}
    manifest_rows: list[dict[str, object]] = []
    for filename, (source_url, artifact_role) in metadata.items():
        path = CAS_DOC_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Tracked CAS metadata is missing: {path}")
        payloads[filename] = read_json(path)
        manifest_rows.append(
            {
                "artifact_role": artifact_role,
                "local_path": relative(path),
                "source_url": source_url,
                "retrieved_at_local": "repository_metadata_snapshot",
                "size_bytes": path.stat().st_size,
                "md5": md5_file(path),
                "sha256": sha256_file(path),
            }
        )
    return payloads, manifest_rows


def main() -> None:
    for directory in (CAS_RAW_DIR, CAS_DOC_DIR, CAS_LOG_DIR, CAS_CONFIG_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    rows, fields = read_lossless_snapshot()
    payloads, manifest_rows = metadata_manifest()
    layer = payloads["cas_layer_metadata.json"]
    field_page_data = payloads["cas_field_page_data.json"]
    if not isinstance(layer, dict) or not isinstance(field_page_data, dict):
        raise TypeError("CAS metadata snapshots must be JSON objects")

    layer_fields = layer.get("fields", [])
    if not isinstance(layer_fields, list):
        raise TypeError("CAS layer metadata does not contain a field list")
    live_fields = [str(field["name"]) for field in layer_fields]
    if live_fields != fields:
        raise ValueError(
            "Fixed CAS JSONL field order differs from the tracked layer metadata"
        )

    documented = parse_documented_fields(field_page_data)
    documented.to_csv(
        CAS_LOG_DIR / "cas_documented_fields.csv",
        index=False,
        encoding="utf-8-sig",
    )
    leakage = build_leakage_audit(layer_fields, documented)
    leakage.to_csv(
        CAS_LOG_DIR / "cas_leakage_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        STATS19_FEATURE_CROSSWALK,
        columns=["stats19_feature", "cas_feature", "relationship", "reason"],
    ).to_csv(
        CAS_LOG_DIR / "cas_stats19_feature_crosswalk.csv",
        index=False,
        encoding="utf-8-sig",
    )

    documented_set = set(documented["field_name"])
    live_set = set(live_fields)
    pd.DataFrame(
        [
            {"comparison": "live_not_documented", "field_name": field}
            for field in sorted(live_set - documented_set)
        ]
        + [
            {"comparison": "documented_not_live", "field_name": field}
            for field in sorted(documented_set - live_set)
        ]
    ).to_csv(
        CAS_LOG_DIR / "cas_schema_documentation_mismatch.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_raw_snapshot(RAW_CSV, rows, fields)
    data = pd.DataFrame(rows, columns=fields)
    if data["OBJECTID"].duplicated().any():
        raise ValueError("Duplicate OBJECTID values found in the fixed CAS snapshot")
    observed_years = tuple(
        sorted(data["crashYear"].dropna().astype(int).unique())
    )
    if observed_years != SNAPSHOT_YEARS:
        raise ValueError(f"Unexpected fixed snapshot years: {observed_years}")
    unexpected_targets = sorted(
        set(data["crashSeverity"].dropna()) - set(TARGET_MAP)
    )
    if unexpected_targets:
        raise ValueError(f"Unexpected injury severity labels: {unexpected_targets}")

    data["mapped_severity"] = data["crashSeverity"].map(TARGET_MAP)
    annual_target = (
        data.groupby(["crashYear", "crashSeverity", "mapped_severity"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    annual_totals = annual_target.groupby("crashYear")["count"].transform("sum")
    annual_target["pct_within_year"] = annual_target["count"] / annual_totals
    annual_target.to_csv(
        CAS_LOG_DIR / "cas_injury_severity_by_year.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # The public input is intentionally limited to the fixed 2022-2025 injury
    # snapshot. This local summary replaces the live all-years API query.
    full_annual = (
        data.groupby(["crashYear", "crashSeverity"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    full_annual.to_csv(
        CAS_LOG_DIR / "cas_all_severity_by_year_2015_present.csv",
        index=False,
        encoding="utf-8-sig",
    )
    build_documentation_value_checks(data, full_annual).to_csv(
        CAS_LOG_DIR / "cas_documentation_value_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )

    quality, categories, numeric = build_quality_outputs(data)
    quality.to_csv(
        CAS_LOG_DIR / "cas_candidate_field_quality_by_year.csv",
        index=False,
        encoding="utf-8-sig",
    )
    categories.to_csv(
        CAS_LOG_DIR / "cas_candidate_categories_by_year.csv",
        index=False,
        encoding="utf-8-sig",
    )
    numeric.to_csv(
        CAS_LOG_DIR / "cas_candidate_numeric_values_by_year.csv",
        index=False,
        encoding="utf-8-sig",
    )
    build_category_coverage(data).to_csv(
        CAS_LOG_DIR / "cas_category_coverage_by_split.csv",
        index=False,
        encoding="utf-8-sig",
    )
    build_categorical_drift(data).to_csv(
        CAS_LOG_DIR / "cas_categorical_drift_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    snapshot_hash = sha256_file(RAW_JSONL)
    manifest_rows.extend(
        [
            {
                "artifact_role": "fixed_lossless_attribute_snapshot",
                "local_path": relative(RAW_JSONL),
                "source_url": QUERY_URL,
                "retrieved_at_local": "provided_fixed_snapshot",
                "size_bytes": RAW_JSONL.stat().st_size,
                "md5": md5_file(RAW_JSONL),
                "sha256": snapshot_hash,
                "note": "Authoritative public input; preserve literal None and Null source categories.",
            },
            {
                "artifact_role": "derived_inspection_snapshot",
                "local_path": relative(RAW_CSV),
                "source_url": QUERY_URL,
                "retrieved_at_local": "derived_from_fixed_jsonl",
                "size_bytes": RAW_CSV.stat().st_size,
                "md5": md5_file(RAW_CSV),
                "sha256": sha256_file(RAW_CSV),
            },
        ]
    )
    pd.DataFrame(manifest_rows).to_csv(
        CAS_LOG_DIR / "cas_source_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    protocol = {
        "version": "CAS_FEASIBILITY_V1",
        "status": "FEASIBILITY_AUDIT_NOT_MODEL_TRAINING",
        "retrieved_at_local": utc_now(),
        "execution_mode": "offline_fixed_snapshot",
        "official_item_id": ITEM_ID,
        "license": "CC BY 4.0 International",
        "statistical_unit": "police-reported crash",
        "matched_population": "police-reported personal-injury crashes only",
        "excluded_severity": "Non-Injury Crash",
        "native_target_order": ["Minor Crash", "Serious Crash", "Fatal Crash"],
        "analysis_target_order": ["Minor", "Serious", "Fatal"],
        "target_mapping": TARGET_MAP,
        "prediction_time": (
            "at collision; no casualty count, participant count, struck-object, "
            "event-trajectory, response, exact-location, or outcome-derived input"
        ),
        "candidate_features": {
            "numeric": INCLUDE_NUMERIC,
            "categorical": INCLUDE_CATEGORICAL,
            "special_handling": {
                "NumberOfLanes": "categorical despite integer storage; retain 0 because it is observed mainly with Off road",
                "advisorySpeed": "structurally sparse; do not apply ordinary mean imputation",
                "temporarySpeedLimit": "structurally sparse; do not apply ordinary mean imputation",
            },
        },
        "candidate_temporal_protocol": {
            "train_years": [2022, 2023],
            "validation_years": [2024],
            "test_years": [2025],
            "excluded_years": {
                "2020-2021": "official CAS open-data warning says these years are incomplete",
                "pre-2020": "two included fields changed recording on 2019-12-17",
                "2026": "current partial year",
            },
        },
        "comparison_rule": (
            "Train and evaluate CAS separately from STATS19. Compare only the "
            "direction of random-versus-temporal gaps, model-gain contraction, "
            "ordinal error patterns, and explanation-rank stability. Do not pool "
            "records or compare absolute metrics as if labels were identical."
        ),
        "feature_alignment": {
            "crosswalk": "logs/cas/cas_stats19_feature_crosswalk.csv",
            "rule": (
                "Use dataset-specific feature configurations. The feature sets are "
                "not one-to-one aligned, so CAS is an independent workflow replication, "
                "not an external test set for a model trained on STATS19."
            ),
        },
        "official_definition_evidence": {
            "source": NZ_GLOSSARY_PAGE,
            "fatal": "injury resulting in death within 30 days of the crash",
            "serious": (
                "fracture, concussion, internal injury, crushing, severe cut or "
                "laceration, severe general shock requiring treatment, or removal "
                "to and detention in hospital"
            ),
            "minor": "minor injury such as sprain or bruise",
        },
        "source_pages": {
            "landing": LANDING_PAGE,
            "field_descriptions": FIELD_DESCRIPTION_PAGE,
            "cas_system": CAS_SYSTEM_PAGE,
            "severity_glossary": NZ_GLOSSARY_PAGE,
        },
        "raw_snapshot": {
            "where_clause": "crashYear BETWEEN 2022 AND 2025 AND crashSeverity <> 'Non-Injury Crash'",
            "row_count": len(data),
            "csv_read_rule": "pd.read_csv(..., keep_default_na=False); convert only empty strings to missing",
            "literal_category_warning": "Do not let pandas default NA parsing convert the literal source value None to NaN",
            "lossless_jsonl": relative(RAW_JSONL),
            "md5": md5_file(RAW_CSV),
            "sha256": sha256_file(RAW_CSV),
            "lossless_jsonl_sha256": snapshot_hash,
        },
    }
    write_json(CAS_CONFIG_DIR / "cas_feasibility_protocol.json", protocol)

    print("CAS_FEASIBILITY_MODE=OFFLINE_FIXED_SNAPSHOT")
    print(f"CAS_FEASIBILITY_ROWS={len(data)}")
    print(f"CAS_FEASIBILITY_FIELDS={len(fields)}")
    print(f"CAS_FEASIBILITY_CANDIDATE_FEATURES={len(INCLUDE_FIELDS)}")
    print("CAS_FEASIBILITY_STATUS=AUDIT_COMPLETE_WITHOUT_LIVE_QUERY")


if __name__ == "__main__":
    main()
