"""Acquire and audit New Zealand CAS as a possible second data source.

The script freezes official metadata and an attribute-only snapshot of injury
crashes for 2022-2025. It then produces schema, target, missingness, category,
and prediction-time leakage audits. It does not modify any STATS19 artifact.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import html
import json
import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests


PROJECT_DIR = Path(__file__).resolve().parents[1]
CAS_RAW_DIR = PROJECT_DIR / "data" / "raw" / "cas"
CAS_DOC_DIR = PROJECT_DIR / "data" / "external" / "cas"
CAS_LOG_DIR = PROJECT_DIR / "logs" / "cas"
CAS_CONFIG_DIR = PROJECT_DIR / "config" / "cas"

ITEM_ID = "8d684f1841fa4dbea6afaefc8a1ba0fc"
FIELD_PAGE_ID = "087345dce5454c4399af3cae69e0f5c1"
ITEM_URL = f"https://www.arcgis.com/sharing/rest/content/items/{ITEM_ID}"
FIELD_PAGE_URL = (
    f"https://www.arcgis.com/sharing/rest/content/items/{FIELD_PAGE_ID}"
)
SERVICE_URL = (
    "https://services.arcgis.com/CXBb7LAjgIIdcsPt/arcgis/rest/services/"
    "CAS_Data_Public/FeatureServer"
)
LAYER_URL = f"{SERVICE_URL}/0"
QUERY_URL = f"{LAYER_URL}/query"
LANDING_PAGE = (
    "https://opendata-nzta.opendata.arcgis.com/datasets/"
    "NZTA::crash-analysis-system-cas-data-1/about"
)
FIELD_DESCRIPTION_PAGE = (
    "https://opendata-nzta.opendata.arcgis.com/pages/"
    "cas-data-field-descriptions"
)
CAS_SYSTEM_PAGE = (
    "https://www.nzta.govt.nz/partners/data-and-tools/"
    "crash-analysis-system"
)
NZ_GLOSSARY_PAGE = (
    "https://www.transport.govt.nz/statistics-and-insights/"
    "glossary-and-references"
)

SNAPSHOT_YEARS = (2022, 2023, 2024, 2025)
SNAPSHOT_WHERE = (
    "crashYear BETWEEN 2022 AND 2025 "
    "AND crashSeverity <> 'Non-Injury Crash'"
)
TARGET_MAP = {
    "Minor Crash": "Minor",
    "Serious Crash": "Serious",
    "Fatal Crash": "Fatal",
}

INCLUDE_NUMERIC = [
    "advisorySpeed",
    "speedLimit",
    "temporarySpeedLimit",
]
INCLUDE_CATEGORICAL = [
    "NumberOfLanes",
    "crashSHDescription",
    "flatHill",
    "light",
    "roadCharacter",
    "roadLane",
    "roadSurface",
    "streetLight",
    "trafficControl",
    "urban",
    "weatherA",
    "weatherB",
]
INCLUDE_FIELDS = INCLUDE_NUMERIC + INCLUDE_CATEGORICAL
NUMERIC_AUDIT_FIELDS = INCLUDE_NUMERIC + ["NumberOfLanes"]
AUDIT_ONLY_FIELDS = ["holiday"]
QUALITY_FIELDS = INCLUDE_FIELDS + AUDIT_ONLY_FIELDS

STATS19_FEATURE_CROSSWALK = [
    ("feature_day_of_week", "", "none", "CAS public layer has no raw crash date or day-of-week field."),
    ("feature_first_road_class", "crashSHDescription", "partial", "CAS distinguishes State Highway from other roads, not the STATS19 road-class levels."),
    ("feature_road_type", "roadLane|roadCharacter", "partial", "Lane configuration and road character overlap conceptually but use different constructs and labels."),
    ("feature_junction_detail", "", "none", "The live CAS intersection field is undocumented and was excluded."),
    ("feature_junction_control", "trafficControl", "partial", "Both describe traffic control, but category definitions are jurisdiction-specific."),
    ("feature_second_road_class", "", "none", "No documented equivalent is available in the CAS public layer."),
    ("feature_pedestrian_crossing", "", "none", "No pre-collision pedestrian-crossing field is available in the CAS public layer."),
    ("feature_light_conditions", "light|streetLight", "close_concept", "CAS separates ambient light from street lighting; labels require dataset-specific preprocessing."),
    ("feature_weather_conditions", "weatherA|weatherB", "close_concept", "CAS splits weather across two fields and uses different labels."),
    ("feature_road_surface_conditions", "", "none", "CAS roadSurface is surface type, not the wet/dry or contaminant condition represented in STATS19."),
    ("feature_special_conditions_at_site", "", "none", "No documented pre-collision equivalent is available."),
    ("feature_carriageway_hazards", "", "none", "Available struck-object variables are event-derived and were excluded."),
    ("feature_urban_or_rural_area", "urban", "partial", "CAS urban is derived from speedLimit and is not an independently coded geography field."),
    ("feature_trunk_road_flag", "crashSHDescription", "partial", "State Highway and trunk-road status are related but not jurisdictionally identical."),
    ("feature_speed_limit", "speedLimit", "close_concept", "Numeric coding and ranges differ by jurisdiction; values must not be pooled."),
    ("feature_month", "", "none", "CAS public layer has no raw crash date from which month can be derived."),
    ("feature_hour", "", "none", "CAS public layer has no raw crash time from which hour can be derived."),
    ("", "advisorySpeed", "cas_only", "Documented CAS road attribute with no STATS19 counterpart in the frozen feature set."),
    ("", "temporarySpeedLimit", "cas_only", "Documented CAS road attribute with no STATS19 counterpart in the frozen feature set."),
    ("", "NumberOfLanes", "cas_only", "Documented CAS road attribute with no STATS19 counterpart in the frozen feature set."),
    ("", "flatHill", "cas_only", "Documented CAS road attribute with no STATS19 counterpart in the frozen feature set."),
    ("", "roadSurface", "cas_only", "CAS surface type has no equivalent in the frozen STATS19 feature set."),
]

DIRECT_OUTCOME = {
    "fatalCount",
    "minorInjuryCount",
    "seriousInjuryCount",
}
EXACT_GEOGRAPHY = {
    "areaUnitID",
    "crashLocation1",
    "crashLocation2",
    "meshblockId",
    "tlaId",
    "tlaName",
}
AMBIGUOUS_UNDOCUMENTED = {"crashRoadSideRoad", "intersection"}
EVENT_TRAJECTORY = {"crashDirectionDescription", "directionRoleDescription"}
POST_COLLISION_DERIVED = {
    "bicycle",
    "bridge",
    "bus",
    "carStationWagon",
    "cliffBank",
    "debris",
    "ditch",
    "fence",
    "guardRail",
    "houseOrBuilding",
    "kerb",
    "moped",
    "motorcycle",
    "objectThrownOrDropped",
    "otherObject",
    "otherVehicleType",
    "overBank",
    "parkedVehicle",
    "pedestrian",
    "phoneBoxEtc",
    "postOrPole",
    "roadworks",
    "schoolBus",
    "slipOrFlood",
    "strayAnimal",
    "suv",
    "taxi",
    "trafficIsland",
    "trafficSign",
    "train",
    "tree",
    "truck",
    "unknownVehicleType",
    "vanOrUtility",
    "vehicle",
    "waterRiver",
}


def ensure_directories() -> None:
    for path in (CAS_RAW_DIR, CAS_DOC_DIR, CAS_LOG_DIR, CAS_CONFIG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def get_json(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, object] | None = None,
    timeout: int = 180,
) -> dict[str, object]:
    response = session.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and "error" in payload:
        raise RuntimeError(f"ArcGIS API error from {response.url}: {payload['error']}")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object from {response.url}")
    return payload


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(PROJECT_DIR).as_posix()


def iter_markdown(value: object) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "markdown" and isinstance(child, str):
                yield child
            else:
                yield from iter_markdown(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_markdown(child)


def plain_text(fragment: str) -> str:
    fragment = re.sub(r"<sup.*?</sup>", "", fragment, flags=re.I | re.S)
    fragment = re.sub(r"<[^>]+>", " ", fragment)
    return " ".join(html.unescape(fragment).split())


def parse_documented_fields(page_data: dict[str, object]) -> pd.DataFrame:
    markdown = "\n".join(iter_markdown(page_data))
    row_pattern = re.compile(
        r"<tr><td>\s*<p>\s*(.*?)\s*</p>\s*</td>"
        r"<td>\s*<p>\s*(.*?)\s*</p>\s*</td>"
        r"<td>\s*<p>(.*?)</p>\s*</td></tr>",
        flags=re.I | re.S,
    )
    rows = []
    for field_html, alias_html, description_html in row_pattern.findall(markdown):
        rows.append(
            {
                "field_name": plain_text(field_html),
                "alias": plain_text(alias_html),
                "description": plain_text(description_html),
                "changed_2019_12_17": "<sup>1</sup>" in field_html,
            }
        )
    if not rows:
        raise ValueError("No field rows were parsed from the official Hub page")
    return pd.DataFrame(rows)


def fetch_feature_rows(
    session: requests.Session,
    fields: list[str],
    *,
    batch_size: int = 5000,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    offset = 0
    while True:
        payload = get_json(
            session,
            QUERY_URL,
            params={
                "where": SNAPSHOT_WHERE,
                "outFields": ",".join(fields),
                "orderByFields": "OBJECTID",
                "returnGeometry": "false",
                "resultOffset": offset,
                "resultRecordCount": batch_size,
                "f": "json",
            },
        )
        features = payload.get("features", [])
        if not isinstance(features, list):
            raise TypeError("ArcGIS feature response did not contain a feature list")
        batch = [feature["attributes"] for feature in features]
        rows.extend(batch)
        print(f"  CAS rows downloaded: {len(rows):,}")
        if len(batch) < batch_size and not payload.get("exceededTransferLimit", False):
            break
        offset += len(batch)
        if not batch:
            break
    return rows


def fetch_annual_severity(session: requests.Session) -> pd.DataFrame:
    out_statistics = json.dumps(
        [
            {
                "statisticType": "count",
                "onStatisticField": "OBJECTID",
                "outStatisticFieldName": "n",
            }
        ],
        separators=(",", ":"),
    )
    payload = get_json(
        session,
        QUERY_URL,
        params={
            "where": "crashYear >= 2015",
            "outStatistics": out_statistics,
            "groupByFieldsForStatistics": "crashYear,crashSeverity",
            "orderByFields": "crashYear,crashSeverity",
            "returnGeometry": "false",
            "f": "json",
        },
    )
    return pd.DataFrame(
        feature["attributes"] for feature in payload.get("features", [])
    ).rename(columns={"n": "count"})


def write_raw_snapshot(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl_snapshot(path: Path, rows: list[dict[str, object]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def classify_field(field: str) -> dict[str, str]:
    if field == "OBJECTID":
        return {
            "availability_time": "record_identifier_after_entry",
            "decision": "EXCLUDE_IDENTIFIER",
            "model_handling": "drop",
            "risk_type": "memorization",
            "rationale": "System-maintained row identifier has no transferable meaning.",
        }
    if field == "crashSeverity":
        return {
            "availability_time": "outcome_after_collision",
            "decision": "TARGET",
            "model_handling": "map injury crashes to Minor < Serious < Fatal",
            "risk_type": "target",
            "rationale": "Worst injury in the crash is the prespecified ordinal outcome.",
        }
    if field == "crashYear":
        return {
            "availability_time": "known_at_collision",
            "decision": "SPLIT_ONLY",
            "model_handling": "temporal split and reporting only",
            "risk_type": "temporal_shortcut",
            "rationale": "Including year would let the model exploit year-specific prevalence.",
        }
    if field == "crashFinancialYear":
        return {
            "availability_time": "known_at_collision",
            "decision": "EXCLUDE_TEMPORAL_REDUNDANCY",
            "model_handling": "drop",
            "risk_type": "temporal_shortcut",
            "rationale": "Redundant with crash year and unsuitable as a deployable feature.",
        }
    if field == "region":
        return {
            "availability_time": "location_attribute_before_collision",
            "decision": "GROUP_ONLY",
            "model_handling": "regional audit or sensitivity analysis only",
            "risk_type": "geographic_reporting_proxy",
            "rationale": "Useful for grouping, but as a feature it can encode regional practice.",
        }
    if field in EXACT_GEOGRAPHY:
        return {
            "availability_time": "location_attribute_before_collision",
            "decision": "EXCLUDE_EXACT_GEOGRAPHY",
            "model_handling": "drop",
            "risk_type": "spatial_shortcut",
            "rationale": "Fine location identifiers encourage memorization and reduce transportability.",
        }
    if field == "holiday":
        return {
            "availability_time": "known_at_collision",
            "decision": "EXCLUDE_TEMPORAL_DERIVED",
            "model_handling": "drop",
            "risk_type": "calendar_derivation_without_raw_date",
            "rationale": "Calendar-derived holiday status is not needed for the minimum matched framework and the public layer lacks raw crash date/time.",
        }
    if field in INCLUDE_FIELDS:
        handling = "numeric" if field in INCLUDE_NUMERIC else "categorical"
        return {
            "availability_time": "exists_or_observable_at_collision",
            "decision": "INCLUDE_CANDIDATE",
            "model_handling": handling,
            "risk_type": "retrospective_recording",
            "rationale": "Road or environmental condition precedes the injury outcome.",
        }
    if field in DIRECT_OUTCOME:
        return {
            "availability_time": "outcome_after_collision",
            "decision": "EXCLUDE_OUTCOME",
            "model_handling": "drop",
            "risk_type": "direct_target_leakage",
            "rationale": "Casualty counts directly determine or reveal crash severity.",
        }
    if field in AMBIGUOUS_UNDOCUMENTED:
        return {
            "availability_time": "not_verified_in_current_documentation",
            "decision": "EXCLUDE_UNDOCUMENTED",
            "model_handling": "drop unless NZTA supplies a current definition",
            "risk_type": "semantic_uncertainty",
            "rationale": "The live schema contains this field but the official description page does not define it.",
        }
    if field in EVENT_TRAJECTORY:
        return {
            "availability_time": "coded_from_crash_event",
            "decision": "EXCLUDE_EVENT_DERIVED",
            "model_handling": "drop",
            "risk_type": "post_collision_or_event_description",
            "rationale": "Direction is assigned from the recorded crash event or principal vehicle.",
        }
    if field in POST_COLLISION_DERIVED:
        return {
            "availability_time": "known_or_derived_after_impact",
            "decision": "EXCLUDE_POST_COLLISION",
            "model_handling": "drop",
            "risk_type": "event_consequence_or_participant_information",
            "rationale": "Official documentation defines this as an involved/struck-object count or other event-derived variable.",
        }
    raise KeyError(f"No leakage decision was defined for live CAS field: {field}")


def build_leakage_audit(
    layer_fields: list[dict[str, object]],
    documented: pd.DataFrame,
) -> pd.DataFrame:
    documented_by_field = documented.set_index("field_name").to_dict("index")
    rows = []
    for position, field_info in enumerate(layer_fields, start=1):
        field = str(field_info["name"])
        doc = documented_by_field.get(field, {})
        rows.append(
            {
                "field_position": position,
                "field_name": field,
                "alias": field_info.get("alias", ""),
                "arcgis_type": field_info.get("type", ""),
                "nullable": field_info.get("nullable", ""),
                "documented_on_hub_page": bool(doc),
                "official_description": doc.get("description", ""),
                "changed_2019_12_17": doc.get("changed_2019_12_17", False),
                **classify_field(field),
                "audit_status": "CAS_FEASIBILITY_V1",
            }
        )
    return pd.DataFrame(rows)


def build_quality_outputs(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    quality_rows = []
    category_rows = []
    numeric_rows = []
    for year in SNAPSHOT_YEARS:
        year_data = data.loc[data["crashYear"] == year]
        for field in QUALITY_FIELDS:
            series = year_data[field]
            quality_rows.append(
                {
                    "year": year,
                    "field_name": field,
                    "row_count": len(year_data),
                    "missing_count": int(series.isna().sum()),
                    "missing_pct": float(series.isna().mean()),
                    "unique_non_missing": int(series.nunique(dropna=True)),
                }
            )
            counts = series.astype("string").fillna("__MISSING__").value_counts(dropna=False)
            for value, count in counts.items():
                category_rows.append(
                    {
                        "year": year,
                        "field_name": field,
                        "observed_value": value,
                        "count": int(count),
                        "pct": float(count / len(year_data)),
                    }
                )
        for field in NUMERIC_AUDIT_FIELDS:
            numeric = pd.to_numeric(year_data[field], errors="coerce")
            values = sorted(numeric.dropna().unique().tolist())
            numeric_rows.append(
                {
                    "year": year,
                    "field_name": field,
                    "non_missing_count": int(numeric.notna().sum()),
                    "minimum": min(values) if values else None,
                    "maximum": max(values) if values else None,
                    "unique_values": "|".join(str(value) for value in values),
                }
            )
    return (
        pd.DataFrame(quality_rows),
        pd.DataFrame(category_rows),
        pd.DataFrame(numeric_rows),
    )


def normalized_category(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("__MISSING__")


def build_category_coverage(data: pd.DataFrame) -> pd.DataFrame:
    train = data.loc[data["crashYear"].isin([2022, 2023])]
    validation = data.loc[data["crashYear"].eq(2024)]
    test = data.loc[data["crashYear"].eq(2025)]
    rows = []
    for field in INCLUDE_CATEGORICAL:
        train_values = set(normalized_category(train[field]))
        for split_name, split_data in (("validation_2024", validation), ("test_2025", test)):
            split_values = set(normalized_category(split_data[field]))
            unseen = sorted(split_values - train_values)
            unseen_mask = normalized_category(split_data[field]).isin(unseen)
            rows.append(
                {
                    "field_name": field,
                    "split": split_name,
                    "train_category_count": len(train_values),
                    "split_category_count": len(split_values),
                    "unseen_categories": "|".join(unseen),
                    "unseen_row_count": int(unseen_mask.sum()),
                    "unseen_row_pct": float(unseen_mask.mean()),
                }
            )
    return pd.DataFrame(rows)


def build_categorical_drift(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    year_pairs = [
        (left, right)
        for index, left in enumerate(SNAPSHOT_YEARS)
        for right in SNAPSHOT_YEARS[index + 1 :]
    ]
    for field in INCLUDE_CATEGORICAL:
        distributions = {}
        for year in SNAPSHOT_YEARS:
            values = normalized_category(data.loc[data["crashYear"].eq(year), field])
            distributions[year] = values.value_counts(normalize=True)
        pair_results = []
        for left, right in year_pairs:
            support = distributions[left].index.union(distributions[right].index)
            left_dist = distributions[left].reindex(support, fill_value=0.0)
            right_dist = distributions[right].reindex(support, fill_value=0.0)
            absolute = (left_dist - right_dist).abs()
            pair_results.append(
                {
                    "left_year": left,
                    "right_year": right,
                    "total_variation": float(0.5 * absolute.sum()),
                    "max_shift_category": str(absolute.idxmax()),
                    "max_absolute_pct_point_shift": float(absolute.max()),
                }
            )
        worst = max(pair_results, key=lambda row: row["total_variation"])
        rows.append({"field_name": field, **worst})
    return pd.DataFrame(rows).sort_values("total_variation", ascending=False)


def build_documentation_value_checks(
    data: pd.DataFrame,
    full_annual: pd.DataFrame,
) -> pd.DataFrame:
    observed_severity = sorted(full_annual["crashSeverity"].dropna().astype(str).unique())

    def values(field: str) -> str:
        return "|".join(sorted(normalized_category(data[field]).unique()))

    return pd.DataFrame(
        [
            {
                "field_name": "crashSeverity",
                "documented_values": "F|S|M|N",
                "observed_values": "|".join(observed_severity),
                "status": "MISMATCH",
                "handling": "freeze and map the observed full labels; never assume F/S/M/N",
            },
            {
                "field_name": "crashSHDescription",
                "documented_values": "1|2",
                "observed_values": values("crashSHDescription"),
                "status": "MISMATCH",
                "handling": "treat observed Yes/No/Unknown as categorical labels",
            },
            {
                "field_name": "holiday",
                "documented_values": "four named periods; otherwise None",
                "observed_values": values("holiday"),
                "status": "REPRESENTATION_DIFFERENCE",
                "handling": "map API missing to outside listed holiday periods",
            },
            {
                "field_name": "roadSurface",
                "documented_values": "Sealed|Unsealed",
                "observed_values": values("roadSurface"),
                "status": "ADDITIONAL_LIVE_VALUES",
                "handling": "retain End of seal and Null as explicit categories",
            },
            {
                "field_name": "streetLight",
                "documented_values": "On|Off|None|Unknown",
                "observed_values": values("streetLight"),
                "status": "LABEL_DIFFERENCE",
                "handling": "retain None and Null as distinct source labels",
            },
            {
                "field_name": "weatherB",
                "documented_values": "Frost|Strong Wind|Unknown",
                "observed_values": values("weatherB"),
                "status": "LABEL_DIFFERENCE",
                "handling": "retain Frost, Strong wind, None and Null without merging",
            },
        ]
    )


def main() -> None:
    ensure_directories()
    retrieved_at = utc_now()
    session = requests.Session()
    session.headers.update({"User-Agent": "CAS-feasibility-audit/1.0"})

    print("Freezing official CAS metadata...")
    item = get_json(session, ITEM_URL, params={"f": "pjson"})
    service = get_json(session, SERVICE_URL, params={"f": "pjson"})
    layer = get_json(session, LAYER_URL, params={"f": "pjson"})
    field_page_item = get_json(session, FIELD_PAGE_URL, params={"f": "pjson"})
    field_page_data = get_json(session, f"{FIELD_PAGE_URL}/data", params={"f": "pjson"})

    metadata_files = {
        "cas_item_metadata.json": (item, f"{ITEM_URL}?f=pjson"),
        "cas_service_metadata.json": (service, f"{SERVICE_URL}?f=pjson"),
        "cas_layer_metadata.json": (layer, f"{LAYER_URL}?f=pjson"),
        "cas_field_page_item.json": (field_page_item, f"{FIELD_PAGE_URL}?f=pjson"),
        "cas_field_page_data.json": (field_page_data, f"{FIELD_PAGE_URL}/data?f=pjson"),
    }
    manifest_rows = []
    for filename, (payload, url) in metadata_files.items():
        path = CAS_DOC_DIR / filename
        write_json(path, payload)
        manifest_rows.append(
            {
                "artifact_role": "official_metadata_snapshot",
                "local_path": relative(path),
                "source_url": url,
                "retrieved_at_local": retrieved_at,
                "size_bytes": path.stat().st_size,
                "md5": md5_file(path),
                "sha256": sha256_file(path),
            }
        )

    documented = parse_documented_fields(field_page_data)
    documented.to_csv(CAS_LOG_DIR / "cas_documented_fields.csv", index=False, encoding="utf-8-sig")

    layer_fields = layer.get("fields", [])
    if not isinstance(layer_fields, list):
        raise TypeError("Live CAS layer did not contain a field list")
    fields = [str(field["name"]) for field in layer_fields]
    leakage = build_leakage_audit(layer_fields, documented)
    leakage.to_csv(CAS_LOG_DIR / "cas_leakage_audit.csv", index=False, encoding="utf-8-sig")
    feature_crosswalk = pd.DataFrame(
        STATS19_FEATURE_CROSSWALK,
        columns=["stats19_feature", "cas_feature", "relationship", "reason"],
    )
    feature_crosswalk.to_csv(
        CAS_LOG_DIR / "cas_stats19_feature_crosswalk.csv",
        index=False,
        encoding="utf-8-sig",
    )

    live_set = set(fields)
    documented_set = set(documented["field_name"])
    schema_comparison = pd.DataFrame(
        [
            {"comparison": "live_not_documented", "field_name": field}
            for field in sorted(live_set - documented_set)
        ]
        + [
            {"comparison": "documented_not_live", "field_name": field}
            for field in sorted(documented_set - live_set)
        ]
    )
    schema_comparison.to_csv(
        CAS_LOG_DIR / "cas_schema_documentation_mismatch.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Downloading the frozen 2022-2025 injury-crash snapshot...")
    rows = fetch_feature_rows(session, fields)
    raw_path = CAS_RAW_DIR / "cas_injury_2022_2025_snapshot.csv.gz"
    write_raw_snapshot(raw_path, rows, fields)
    jsonl_path = CAS_RAW_DIR / "cas_injury_2022_2025_snapshot.jsonl.gz"
    write_jsonl_snapshot(jsonl_path, rows)
    manifest_rows.append(
        {
            "artifact_role": "official_api_attribute_snapshot",
            "local_path": relative(raw_path),
            "source_url": QUERY_URL,
            "retrieved_at_local": retrieved_at,
            "size_bytes": raw_path.stat().st_size,
            "md5": md5_file(raw_path),
            "sha256": sha256_file(raw_path),
        }
    )
    manifest_rows.append(
        {
            "artifact_role": "lossless_attribute_snapshot",
            "local_path": relative(jsonl_path),
            "source_url": QUERY_URL,
            "retrieved_at_local": retrieved_at,
            "size_bytes": jsonl_path.stat().st_size,
            "md5": md5_file(jsonl_path),
            "sha256": sha256_file(jsonl_path),
            "note": "Use keep_default_na=False for CSV; literal None and Null are source categories.",
        }
    )

    data = pd.DataFrame(rows, columns=fields)
    if data["OBJECTID"].duplicated().any():
        raise ValueError("Duplicate OBJECTID values found in the API snapshot")
    observed_years = tuple(sorted(data["crashYear"].dropna().astype(int).unique()))
    if observed_years != SNAPSHOT_YEARS:
        raise ValueError(f"Unexpected snapshot years: {observed_years}")
    unexpected_targets = sorted(set(data["crashSeverity"].dropna()) - set(TARGET_MAP))
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

    full_annual = fetch_annual_severity(session)
    full_annual.to_csv(
        CAS_LOG_DIR / "cas_all_severity_by_year_2015_present.csv",
        index=False,
        encoding="utf-8-sig",
    )

    documentation_checks = build_documentation_value_checks(data, full_annual)
    documentation_checks.to_csv(
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

    category_coverage = build_category_coverage(data)
    category_coverage.to_csv(
        CAS_LOG_DIR / "cas_category_coverage_by_split.csv",
        index=False,
        encoding="utf-8-sig",
    )
    category_drift = build_categorical_drift(data)
    category_drift.to_csv(
        CAS_LOG_DIR / "cas_categorical_drift_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(CAS_LOG_DIR / "cas_source_manifest.csv", index=False, encoding="utf-8-sig")

    protocol = {
        "version": "CAS_FEASIBILITY_V1",
        "status": "FEASIBILITY_AUDIT_NOT_MODEL_TRAINING",
        "retrieved_at_local": retrieved_at,
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
            "records or compare absolute metric levels as if labels were identical."
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
            "where_clause": SNAPSHOT_WHERE,
            "row_count": len(data),
            "csv_read_rule": "pd.read_csv(..., keep_default_na=False); convert only empty strings to missing",
            "literal_category_warning": "Do not let pandas default NA parsing convert the literal source value None to NaN",
            "lossless_jsonl": relative(jsonl_path),
            "md5": md5_file(raw_path),
            "sha256": sha256_file(raw_path),
        },
    }
    write_json(CAS_CONFIG_DIR / "cas_feasibility_protocol.json", protocol)

    print("CAS feasibility evidence package completed.")
    print(f"  snapshot rows: {len(data):,}")
    print(f"  live fields: {len(fields)}")
    print(f"  candidate features: {len(INCLUDE_FIELDS)}")
    print(f"  output: {relative(CAS_LOG_DIR)}")


if __name__ == "__main__":
    main()
