"""Build and freeze the D3 leakage audit for the STATS19 study.

Prediction setting: conditional on a collision having occurred, predict its
reported injury severity using only collision-table information describing
conditions that existed at or before the collision. The script does not alter
raw data and does not perform feature engineering.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_DIR / "logs"
CONFIG_DIR = PROJECT_DIR / "config"


def rule(
    field: str,
    meaning: str,
    availability: str,
    decision: str,
    handling: str,
    risk: str,
    rationale: str,
) -> dict[str, str]:
    return {
        "field_name": field,
        "meaning": meaning,
        "availability_time": availability,
        "decision": decision,
        "model_handling": handling,
        "risk_type": risk,
        "rationale": rationale,
    }


RULES = [
    rule("collision_index", "Unique collision identifier", "record_created_after_collision", "EXCLUDE_IDENTIFIER", "drop", "memorization", "A unique record key has no transferable predictive meaning."),
    rule("collision_year", "Calendar year", "known_at_collision", "SPLIT_ONLY", "temporal split and reporting only", "temporal_shortcut", "Use for temporal partitioning; including it would let the model exploit year-specific prevalence."),
    rule("collision_ref_no", "Police in-year reference", "record_created_after_collision", "EXCLUDE_IDENTIFIER", "drop", "memorization", "Administrative reference is not unique across years and has no transferable meaning."),
    rule("location_easting_osgr", "Exact OSGR easting", "location_exists_before_collision", "EXCLUDE_EXACT_GEOGRAPHY", "drop", "spatial_shortcut", "Exact coordinates encourage location memorization and weaken geographic transportability."),
    rule("location_northing_osgr", "Exact OSGR northing", "location_exists_before_collision", "EXCLUDE_EXACT_GEOGRAPHY", "drop", "spatial_shortcut", "Exact coordinates encourage location memorization and weaken geographic transportability."),
    rule("longitude", "Exact longitude", "location_exists_before_collision", "EXCLUDE_EXACT_GEOGRAPHY", "drop", "spatial_shortcut", "Exact coordinates encourage location memorization and weaken geographic transportability."),
    rule("latitude", "Exact latitude", "location_exists_before_collision", "EXCLUDE_EXACT_GEOGRAPHY", "drop", "spatial_shortcut", "Exact coordinates encourage location memorization and weaken geographic transportability."),
    rule("police_force", "Reporting police force", "known_from_location_and_reporting_system", "GROUP_ONLY", "regional grouping and sensitivity audit", "reporting_and_geography_proxy", "Useful for geographic evaluation, but as a predictor it can encode force-specific reporting practice."),
    rule("collision_severity", "Reported collision severity", "outcome_after_collision", "TARGET", "map 3=Slight to 0, 2=Serious to 1, 1=Fatal to 2", "target", "This is the prespecified ordinal prediction target."),
    rule("number_of_vehicles", "Vehicles involved", "known_after_impact", "EXCLUDE_POST_COLLISION", "drop", "post_collision_information", "The number involved is not fixed before impact under the strict prediction-time definition."),
    rule("number_of_casualties", "Casualties resulting from collision", "known_after_collision", "EXCLUDE_OUTCOME", "drop", "direct_outcome_leakage", "Casualty count is a consequence of the event and directly related to the target."),
    rule("date", "Collision date", "known_at_collision", "DERIVE_THEN_DROP", "derive month; retain only outside feature matrix for chronology", "temporal_shortcut_if_raw", "Month captures seasonality; raw date is reserved for chronological splitting and then dropped."),
    rule("day_of_week", "Day of week", "known_before_collision", "INCLUDE", "categorical", "low", "Calendar condition is available without using post-collision information."),
    rule("time", "Collision time", "known_at_collision", "DERIVE_THEN_DROP", "derive hour as categorical; drop raw HH:MM", "overprecision", "Hour preserves time-of-day information without treating each minute as a separate pattern."),
    rule("local_authority_district", "Historic local-authority district", "location_exists_before_collision", "EXCLUDE_EXACT_GEOGRAPHY", "drop", "high_cardinality_spatial_shortcut", "Administrative geography is high-cardinality and boundaries change over time."),
    rule("local_authority_ons_district", "ONS district", "location_exists_before_collision", "EXCLUDE_EXACT_GEOGRAPHY", "drop", "high_cardinality_spatial_shortcut", "Administrative geography is high-cardinality and boundaries change over time."),
    rule("local_authority_highway", "Historic highway authority", "location_exists_before_collision", "EXCLUDE_EXACT_GEOGRAPHY", "drop", "high_cardinality_spatial_shortcut", "Authority code is mainly a location identifier and has changing boundaries."),
    rule("local_authority_highway_current", "Current harmonized highway authority", "derived_from_location", "EXCLUDE_EXACT_GEOGRAPHY", "drop", "high_cardinality_spatial_shortcut", "Current boundary mapping is useful administratively but can memorize geography."),
    rule("first_road_class", "Primary-road class", "road_attribute_before_collision", "INCLUDE", "categorical", "low", "Road class is a stable pre-existing road attribute."),
    rule("first_road_number", "Primary-road number", "road_attribute_before_collision", "EXCLUDE_HIGH_CARDINALITY", "drop", "route_identifier_shortcut", "Road number is a high-cardinality route identifier rather than a general road characteristic."),
    rule("road_type", "Road layout/type", "road_attribute_before_collision", "INCLUDE", "categorical", "low", "Road type is a pre-existing geometric characteristic."),
    rule("speed_limit", "Posted speed limit", "road_attribute_before_collision", "INCLUDE", "ordered discrete numeric; convert -1 to missing", "low", "The legal speed limit exists before the collision; -1 is not a physical speed."),
    rule("junction_detail_historic", "Historic junction-detail coding", "road_attribute_before_collision", "EXCLUDE_HISTORIC_DUPLICATE", "drop", "redundancy_and_schema_drift", "Use the DfT-harmonized current junction_detail field instead."),
    rule("junction_detail", "Harmonized junction detail", "road_attribute_before_collision", "INCLUDE", "categorical; retain unknown/not-applicable category", "conversion_uncertainty", "Current-format field provides one coding scheme across all study years."),
    rule("junction_control", "Junction control", "road_attribute_before_collision", "INCLUDE", "categorical; retain unknown/not-applicable category", "structural_missingness", "Control type exists before collision; many non-junction records are structurally not applicable."),
    rule("second_road_class", "Secondary-road class", "road_attribute_before_collision", "INCLUDE", "categorical; retain not-applicable category", "structural_missingness", "Provides junction context while explicitly retaining non-junction cases."),
    rule("second_road_number", "Secondary-road number", "road_attribute_before_collision", "EXCLUDE_HIGH_CARDINALITY", "drop", "route_identifier_shortcut", "Road number is a high-cardinality route identifier rather than a general road characteristic."),
    rule("pedestrian_crossing_human_control_historic", "Historic crossing human-control field", "road_attribute_before_collision", "EXCLUDE_HISTORIC_DUPLICATE", "drop", "redundancy_and_schema_drift", "Superseded by the harmonized pedestrian_crossing field."),
    rule("pedestrian_crossing_physical_facilities_historic", "Historic crossing-facility field", "road_attribute_before_collision", "EXCLUDE_HISTORIC_DUPLICATE", "drop", "redundancy_and_schema_drift", "Superseded by the harmonized pedestrian_crossing field."),
    rule("pedestrian_crossing", "Harmonized pedestrian crossing", "road_attribute_before_collision", "INCLUDE", "categorical; retain unknown category", "conversion_uncertainty", "Crossing facility is a pre-existing road attribute and current coding is harmonized."),
    rule("light_conditions", "Lighting at collision", "observable_at_collision", "INCLUDE", "categorical; retain unknown category", "retrospective_recording", "Lighting existed at the event and is not derived from injury severity."),
    rule("weather_conditions", "Weather at collision", "observable_at_collision", "INCLUDE", "categorical; retain unknown category", "retrospective_recording", "Weather existed at the event and is not derived from injury severity."),
    rule("road_surface_conditions", "Road-surface state", "observable_at_collision", "INCLUDE", "categorical; retain unknown category", "retrospective_recording", "Surface state existed at the event and is not derived from injury severity."),
    rule("special_conditions_at_site", "Special site condition", "exists_at_or_before_collision", "INCLUDE", "categorical; retain unknown category", "retrospective_recording", "Recorded conditions such as roadworks or defects pre-exist the injury outcome, but are documented retrospectively."),
    rule("carriageway_hazards_historic", "Historic carriageway-hazard coding", "exists_at_or_before_collision", "EXCLUDE_HISTORIC_DUPLICATE", "drop", "redundancy_and_schema_drift", "Use the DfT-harmonized current carriageway_hazards field instead."),
    rule("carriageway_hazards", "Harmonized carriageway hazard", "exists_at_or_before_collision", "INCLUDE", "categorical; retain unknown category", "retrospective_recording", "Hazard existed at the event; current-format coding is used consistently."),
    rule("urban_or_rural_area", "Urban/rural classification", "location_attribute_before_collision", "INCLUDE", "categorical; retain unknown category", "low", "Broad area type is transferable and avoids exact-location memorization."),
    rule("did_police_officer_attend_scene_of_accident", "Police attendance", "response_after_collision", "EXCLUDE_POST_COLLISION", "drop", "response_leakage", "Attendance is decided after the collision and is likely associated with event severity."),
    rule("trunk_road_flag", "Trunk-road status", "road_attribute_before_collision", "INCLUDE", "categorical; retain unknown/not-applicable category", "coverage_variation", "Trunk-road status is a pre-existing road characteristic."),
    rule("lsoa_of_accident_location", "Lower-layer statistical area", "derived_from_location", "EXCLUDE_EXACT_GEOGRAPHY", "drop", "high_cardinality_spatial_shortcut", "LSOA is a fine geographic identifier and is not available uniformly for Scotland."),
    rule("enhanced_severity_collision", "Enhanced severity category", "alternative_outcome_after_collision", "EXCLUDE_TARGET_DERIVED", "drop", "direct_target_leakage", "This is an alternative severity outcome, not an independent predictor."),
    rule("collision_injury_based", "Injury-based reporting-system flag", "reporting_process_metadata", "REPORTING_AUDIT_ONLY", "exclude from model; use for label-process sensitivity reporting", "label_definition_proxy", "It identifies how severity was recorded and can let the model learn reporting practice rather than injury risk."),
    rule("collision_adjusted_severity_serious", "DfT adjusted-serious probability", "modelled_from_severity_reporting", "EXCLUDE_TARGET_DERIVED", "drop", "target_model_leakage", "Official guidance defines this as a severity-adjustment model probability."),
    rule("collision_adjusted_severity_slight", "DfT adjusted-slight probability", "modelled_from_severity_reporting", "EXCLUDE_TARGET_DERIVED", "drop", "target_model_leakage", "Official guidance defines this as a severity-adjustment model probability."),
]


DIRECT_CATEGORICAL = [
    "day_of_week", "first_road_class", "road_type", "junction_detail",
    "junction_control", "second_road_class", "pedestrian_crossing",
    "light_conditions", "weather_conditions", "road_surface_conditions",
    "special_conditions_at_site", "carriageway_hazards",
    "urban_or_rural_area", "trunk_road_flag",
]
DIRECT_NUMERIC = ["speed_limit"]
DERIVED_FEATURES = [
    {"feature": "month", "source": "date", "type": "categorical", "rule": "calendar month 1-12"},
    {"feature": "hour", "source": "time", "type": "categorical", "rule": "hour 0-23 parsed from valid HH:MM"},
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def main() -> None:
    mapping = read_csv(LOG_DIR / "d2_field_mapping.csv")
    schema = read_csv(LOG_DIR / "d2_schema_by_year.csv")
    codes = read_csv(LOG_DIR / "d2_code_by_year.csv")
    raw_fields = [row["field_name"] for row in mapping]
    rule_fields = [row["field_name"] for row in RULES]
    if rule_fields != raw_fields:
        missing = sorted(set(raw_fields) - set(rule_fields))
        extra = sorted(set(rule_fields) - set(raw_fields))
        raise ValueError(f"D3 rules do not match the 44 raw fields; missing={missing}, extra={extra}")

    total_records = sum(
        int(row["row_count"])
        for row in schema
        if row["field_name"] == raw_fields[0]
    )
    blank_counts: Counter[str] = Counter()
    schema_unknown_counts: Counter[str] = Counter()
    for row in schema:
        blank_counts[row["field_name"]] += int(row["csv_blank_count"])
        schema_unknown_counts[row["field_name"]] += int(row["official_unknown_code_count"])

    code_unknown_counts: Counter[str] = Counter()
    observed_codes: dict[str, set[str]] = defaultdict(set)
    coded_fields: set[str] = set()
    for row in codes:
        field = row["field_name"]
        coded_fields.add(field)
        observed_codes[field].add(row["observed_code"])
        if as_bool(row["unknown_or_missing_label"]):
            code_unknown_counts[field] += int(row["count"])

    mapping_by_field = {row["field_name"]: row for row in mapping}
    audit_rows: list[dict[str, object]] = []
    for position, item in enumerate(RULES, start=1):
        field = item["field_name"]
        unknown_count = (
            code_unknown_counts[field]
            if field in coded_fields
            else schema_unknown_counts[field]
        )
        audit_rows.append(
            {
                "column_position": position,
                "field_name": field,
                "official_field_name": mapping_by_field[field]["official_field_name"],
                "source_table": "collision",
                **item,
                "csv_blank_count": blank_counts[field],
                "csv_blank_pct": blank_counts[field] / total_records,
                "official_unknown_count": unknown_count,
                "official_unknown_pct": unknown_count / total_records,
                "observed_category_count": len(observed_codes[field]) if field in coded_fields else "",
                "audit_status": "FROZEN_D3_V1",
            }
        )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    output = LOG_DIR / "d3_leakage_audit.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    included = DIRECT_CATEGORICAL + DIRECT_NUMERIC
    manifest = {
        "version": "D3_V1",
        "status": "FROZEN",
        "source_table": "collision_only",
        "study_population": "police-reported personal-injury collisions in Great Britain, 2018-2024",
        "prediction_task": "predict reported collision injury severity conditional on a collision",
        "prediction_time": "at collision; no consequence, response, injury-derived, or severity-adjustment inputs",
        "target": {
            "source": "collision_severity",
            "encoding": {"3": 0, "2": 1, "1": 2},
            "ordered_labels": ["Slight", "Serious", "Fatal"],
        },
        "direct_features": {
            "categorical": DIRECT_CATEGORICAL,
            "numeric": DIRECT_NUMERIC,
        },
        "derived_features": DERIVED_FEATURES,
        "split_and_group_only": {
            "collision_year": "temporal split",
            "date": "chronological ordering before month derivation",
            "police_force": "regional grouping/sensitivity analysis only",
            "collision_injury_based": "label-process sensitivity reporting only",
        },
        "unknown_policy": {
            "categorical": "map official unknown and not-applicable codes to separate explicit categories; never treat raw codes as ordered magnitudes",
            "speed_limit": {
                "valid_values": [20, 30, 40, 50, 60, 70],
                "verified_unknown_codes_in_2018_2024_data": [-1],
                "transform": "convert verified unknown codes to missing",
                "schema_guard": "raise an error for any other value until that year's official guide is checked",
                "downstream": "LightGBM uses native missing values; any imputation for other models is fitted within each training fold only",
            },
        },
        "excluded_fields": {
            row["field_name"]: row["decision"]
            for row in audit_rows
            if row["decision"].startswith("EXCLUDE")
        },
    }
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (CONFIG_DIR / "d3_feature_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    decision_counts = Counter(row["decision"] for row in audit_rows)
    included_unknown = sorted(
        (
            (row["field_name"], float(row["official_unknown_pct"]))
            for row in audit_rows
            if row["field_name"] in included and float(row["official_unknown_pct"]) >= 0.10
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    lines = [
        "# D3 leakage audit and frozen feature set",
        "",
        "## Prediction-time definition",
        "",
        "The task is conditional severity prediction after defining that a collision occurs, using only collision-table conditions that existed at or before the event. Post-impact consequences, emergency response, identifiers, exact geography, alternative severity outcomes and DfT severity-adjustment outputs are excluded.",
        "",
        "## Frozen feature set",
        "",
        f"- Raw included fields: **{len(included)}** ({len(DIRECT_CATEGORICAL)} categorical, {len(DIRECT_NUMERIC)} numeric).",
        f"- Derived fields: **{len(DERIVED_FEATURES)}** (`month`, `hour`).",
        f"- Final model feature count before encoding: **{len(included) + len(DERIVED_FEATURES)}**.",
        "- Target order: Slight < Serious < Fatal.",
        "- Source scope: collision table only; vehicle and casualty tables are not joined.",
        "",
        "Direct fields: " + ", ".join(f"`{field}`" for field in included) + ".",
        "",
        "## Decision counts",
        "",
    ]
    lines.extend(f"- {name}: {count}" for name, count in sorted(decision_counts.items()))
    lines.extend(["", "## High unknown/not-applicable rates among included fields", ""])
    if included_unknown:
        lines.extend(f"- `{field}`: {pct:.1%}" for field, pct in included_unknown)
    else:
        lines.append("- None at or above 10%.")
    lines.extend(
        [
            "",
            "These values are not silently deleted. Categorical unknown/not-applicable codes remain explicit; speed-limit -1 is converted to missing inside the modelling pipeline.",
            "",
            "## Non-negotiable leakage exclusions",
            "",
            "`number_of_casualties`, `enhanced_severity_collision`, `collision_adjusted_severity_serious`, and `collision_adjusted_severity_slight` encode outcomes or severity-adjustment results. `did_police_officer_attend_scene_of_accident` is a post-collision response. None may enter training, preprocessing, feature selection or SHAP analysis.",
            "",
            "`collision_injury_based` is reserved for reporting-system sensitivity analysis and must not enter the feature matrix.",
        ]
    )
    (LOG_DIR / "d3_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if len(audit_rows) != 44 or len(included) != 15 or len(DERIVED_FEATURES) != 2:
        raise AssertionError("Frozen D3 feature counts changed unexpectedly")
    print("D3 leakage audit completed.")
    print(f"  audited raw fields: {len(audit_rows)}")
    print(f"  final features before encoding: {len(included) + len(DERIVED_FEATURES)}")
    print(f"  records profiled: {total_records:,}")


if __name__ == "__main__":
    main()
