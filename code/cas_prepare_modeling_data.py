"""Build and freeze the CAS collision-level modeling table.

The lossless JSONL snapshot is the authoritative input. This stage performs
only deterministic semantic mapping and quality control. It does not fit an
encoder, imputer, resampler or predictive model.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_JSONL = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "cas"
    / "cas_injury_2022_2025_snapshot.jsonl.gz"
)
SOURCE_MANIFEST = PROJECT_DIR / "logs" / "cas" / "cas_source_manifest.csv"
FEASIBILITY_PROTOCOL = (
    PROJECT_DIR / "config" / "cas" / "cas_feasibility_protocol.json"
)
INTERIM_FILE = PROJECT_DIR / "data" / "interim" / "cas_modeling_table.csv.gz"
PROCESSED_FILE = (
    PROJECT_DIR / "data" / "processed" / "cas_modeling_dataset.csv.gz"
)
SCHEMA_FILE = PROJECT_DIR / "config" / "cas" / "cas_modeling_schema.json"
CLEANING_FILE = PROJECT_DIR / "config" / "cas" / "cas_cleaning_rules.json"
TARGET_AUDIT_FILE = PROJECT_DIR / "logs" / "cas" / "cas_modeling_target_audit.csv"
FEATURE_AUDIT_FILE = (
    PROJECT_DIR / "logs" / "cas" / "cas_modeling_feature_audit.csv"
)
CATEGORY_AUDIT_FILE = (
    PROJECT_DIR / "logs" / "cas" / "cas_modeling_category_audit.csv"
)
CHECKPOINT_FILE = PROJECT_DIR / "logs" / "cas" / "cas_modeling_checkpoint.md"
ARTIFACT_MANIFEST_FILE = (
    PROJECT_DIR / "logs" / "cas" / "cas_modeling_artifact_manifest.csv"
)

SOURCE_MISSING_TOKEN = "__SOURCE_MISSING__"
TARGET_COLUMN = "target_severity"
TARGET_LABELS = {0: "Minor", 1: "Serious", 2: "Fatal"}

METADATA_COLUMNS = [
    "meta_crash_id",
    "meta_crash_year",
    "meta_region",
]
METADATA_ROLES = {
    "meta_crash_id": "record identity and audit only",
    "meta_crash_year": "temporal splitting only",
    "meta_region": "grouped descriptive audit only; excluded from models",
}

CATEGORICAL_SOURCE_TO_FEATURE = {
    "NumberOfLanes": "feature_number_of_lanes",
    "crashSHDescription": "feature_state_highway",
    "flatHill": "feature_terrain",
    "light": "feature_light",
    "roadCharacter": "feature_road_character",
    "roadLane": "feature_road_lane",
    "roadSurface": "feature_road_surface",
    "streetLight": "feature_street_light",
    "trafficControl": "feature_traffic_control",
    "urban": "feature_urban",
    "weatherA": "feature_weather_primary",
    "weatherB": "feature_weather_secondary",
}
NUMERIC_SOURCE_TO_FEATURE = {
    "advisorySpeed": "feature_advisory_speed",
    "speedLimit": "feature_speed_limit",
    "temporarySpeedLimit": "feature_temporary_speed_limit",
}
ALLOWED_NUMERIC_VALUES = {
    "advisorySpeed": {15, 25, 30, 35, 45, 55, 65, 75, 85, 95},
    "speedLimit": {5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110},
    "temporarySpeedLimit": {10, 20, 30, 40, 50, 60, 70, 80, 90},
}
STRUCTURALLY_SPARSE_NUMERIC = {
    "feature_advisory_speed",
    "feature_temporary_speed_limit",
}

CATEGORICAL_FEATURES = list(CATEGORICAL_SOURCE_TO_FEATURE.values())
NUMERIC_FEATURES = list(NUMERIC_SOURCE_TO_FEATURE.values())
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
OUTPUT_COLUMNS = METADATA_COLUMNS + [TARGET_COLUMN] + FEATURE_COLUMNS


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows generated for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_dataframe_gzip(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as binary:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=binary,
            compresslevel=9,
            mtime=0,
        ) as compressed:
            with io.TextIOWrapper(
                compressed,
                encoding="utf-8",
                newline="",
            ) as text:
                frame.to_csv(
                    text,
                    index=False,
                    lineterminator="\n",
                    na_rep="",
                    float_format="%.12g",
                )
    temporary.replace(path)


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_DIR.resolve()).as_posix()


def load_protocol() -> dict[str, Any]:
    protocol = json.loads(FEASIBILITY_PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("version") != "CAS_FEASIBILITY_V1":
        raise ValueError("Unexpected CAS feasibility protocol version")
    if protocol.get("status") != "FEASIBILITY_AUDIT_NOT_MODEL_TRAINING":
        raise ValueError("CAS feasibility protocol status changed")
    expected_categorical = list(
        protocol["candidate_features"]["categorical"]
    )
    expected_numeric = list(protocol["candidate_features"]["numeric"])
    if expected_categorical != list(CATEGORICAL_SOURCE_TO_FEATURE):
        raise ValueError("Categorical source allowlist differs from the audit")
    if expected_numeric != list(NUMERIC_SOURCE_TO_FEATURE):
        raise ValueError("Numeric source allowlist differs from the audit")
    return protocol


def expected_manifest_hash(local_path: str) -> str:
    manifest = pd.read_csv(SOURCE_MANIFEST, keep_default_na=False)
    rows = manifest.loc[manifest["local_path"].eq(local_path)]
    if len(rows) != 1:
        raise ValueError(f"Source manifest entry missing or duplicated: {local_path}")
    return str(rows.iloc[0]["sha256"])


def load_lossless_snapshot(protocol: dict[str, Any]) -> pd.DataFrame:
    expected_hash = expected_manifest_hash(
        "data/raw/cas/cas_injury_2022_2025_snapshot.jsonl.gz"
    )
    actual_hash = hash_file(RAW_JSONL)
    if actual_hash != expected_hash:
        raise ValueError("Lossless CAS JSONL hash differs from source manifest")

    selected_fields = [
        "OBJECTID",
        "crashYear",
        "crashSeverity",
        "region",
        *CATEGORICAL_SOURCE_TO_FEATURE,
        *NUMERIC_SOURCE_TO_FEATURE,
    ]
    rows: list[dict[str, Any]] = []
    with gzip.open(RAW_JSONL, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            missing = [field for field in selected_fields if field not in record]
            if missing:
                raise ValueError(
                    f"JSONL row {line_number} lacks fields: {missing}"
                )
            rows.append({field: record[field] for field in selected_fields})

    frame = pd.DataFrame(rows, columns=selected_fields)
    expected_rows = int(protocol["raw_snapshot"]["row_count"])
    if len(frame) != expected_rows:
        raise ValueError(
            f"CAS row count differs: expected {expected_rows}, observed {len(frame)}"
        )
    if frame["OBJECTID"].isna().any() or frame["OBJECTID"].duplicated().any():
        raise ValueError("CAS OBJECTID is missing or duplicated")
    frame["OBJECTID"] = pd.to_numeric(frame["OBJECTID"], errors="raise").astype(
        "int64"
    )
    frame["crashYear"] = pd.to_numeric(
        frame["crashYear"], errors="raise"
    ).astype("int16")
    if set(frame["crashYear"].unique()) != {2022, 2023, 2024, 2025}:
        raise ValueError("Unexpected CAS year coverage")
    return frame.sort_values("OBJECTID", kind="stable").reset_index(drop=True)


def canonical_category(value: object) -> str:
    if value is None or value is pd.NA:
        return SOURCE_MISSING_TOKEN
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return SOURCE_MISSING_TOKEN
    if isinstance(value, str):
        if value == "":
            return SOURCE_MISSING_TOKEN
        if value != value.strip():
            raise ValueError(f"Category contains surrounding whitespace: {value!r}")
        if value == SOURCE_MISSING_TOKEN:
            raise ValueError("Reserved missing token occurs in source data")
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    raise TypeError(f"Unsupported categorical source value: {value!r}")


def build_modeling_table(
    raw: pd.DataFrame,
    protocol: dict[str, Any],
) -> pd.DataFrame:
    output = pd.DataFrame(index=raw.index)
    output["meta_crash_id"] = "CAS-" + raw["OBJECTID"].astype(str)
    output["meta_crash_year"] = raw["crashYear"].astype("int16")
    output["meta_region"] = raw["region"].map(canonical_category)

    target_mapping = dict(protocol["target_mapping"])
    analysis_labels = raw["crashSeverity"].map(target_mapping)
    if analysis_labels.isna().any():
        unexpected = sorted(
            set(raw.loc[analysis_labels.isna(), "crashSeverity"].astype(str))
        )
        raise ValueError(f"Unexpected CAS target labels: {unexpected}")
    label_to_code = {label: code for code, label in TARGET_LABELS.items()}
    output[TARGET_COLUMN] = analysis_labels.map(label_to_code).astype("int8")

    for source, feature in CATEGORICAL_SOURCE_TO_FEATURE.items():
        output[feature] = raw[source].map(canonical_category)

    for source, feature in NUMERIC_SOURCE_TO_FEATURE.items():
        source_values = raw[source]
        source_missing = source_values.isna() | source_values.eq("")
        numeric = pd.to_numeric(source_values.where(~source_missing), errors="coerce")
        invalid_conversion = (~source_missing) & numeric.isna()
        if invalid_conversion.any():
            examples = sorted(
                set(source_values.loc[invalid_conversion].astype(str))
            )
            raise ValueError(f"{source} contains nonnumeric values: {examples}")
        observed = set(numeric.dropna().astype(int).unique())
        unexpected = observed - ALLOWED_NUMERIC_VALUES[source]
        if unexpected:
            raise ValueError(f"{source} contains unexpected values: {unexpected}")
        if not np.allclose(
            numeric.dropna().to_numpy(dtype=float),
            numeric.dropna().astype(int).to_numpy(dtype=float),
        ):
            raise ValueError(f"{source} contains noninteger speeds")
        output[feature] = numeric.astype("float64")

    if output.columns.tolist() != OUTPUT_COLUMNS:
        raise AssertionError("CAS output column contract changed")
    if output["meta_crash_id"].duplicated().any():
        raise AssertionError("CAS output identifiers are duplicated")
    if set(output[TARGET_COLUMN].unique()) != set(TARGET_LABELS):
        raise AssertionError("CAS output target codes are incomplete")
    if output[CATEGORICAL_FEATURES].isna().any().any():
        raise AssertionError("Categorical model features contain implicit missing values")
    return output


def make_target_audit(table: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for year in (2022, 2023, 2024, 2025):
        subset = table.loc[table["meta_crash_year"].eq(year)]
        for code, label in TARGET_LABELS.items():
            count = int(subset[TARGET_COLUMN].eq(code).sum())
            rows.append(
                {
                    "year": year,
                    "target_code": code,
                    "target_label": label,
                    "count": count,
                    "pct": count / len(subset),
                    "year_total": len(subset),
                }
            )
    return rows


def make_feature_audit(table: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for year in (2022, 2023, 2024, 2025):
        subset = table.loc[table["meta_crash_year"].eq(year)]
        for feature in FEATURE_COLUMNS:
            if feature in CATEGORICAL_FEATURES:
                missing_count = int(subset[feature].eq(SOURCE_MISSING_TOKEN).sum())
                unique_nonmissing = int(
                    subset.loc[
                        ~subset[feature].eq(SOURCE_MISSING_TOKEN), feature
                    ].nunique()
                )
                missing_representation = SOURCE_MISSING_TOKEN
            else:
                missing_count = int(subset[feature].isna().sum())
                unique_nonmissing = int(subset[feature].nunique(dropna=True))
                missing_representation = "NaN"
            rows.append(
                {
                    "year": year,
                    "feature": feature,
                    "kind": (
                        "categorical"
                        if feature in CATEGORICAL_FEATURES
                        else "numeric"
                    ),
                    "rows": len(subset),
                    "missing_count": missing_count,
                    "missing_pct": missing_count / len(subset),
                    "unique_nonmissing": unique_nonmissing,
                    "missing_representation": missing_representation,
                }
            )
    return rows


def make_category_audit(table: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature in CATEGORICAL_FEATURES:
        counts = table[feature].value_counts(dropna=False).sort_index()
        for category, count in counts.items():
            rows.append(
                {
                    "feature": feature,
                    "category": category,
                    "count": int(count),
                    "pct": int(count) / len(table),
                    "is_source_missing": category == SOURCE_MISSING_TOKEN,
                    "is_literal_none_or_null": category in {"None", "Null"},
                }
            )
    return rows


def validate_round_trip(table: pd.DataFrame) -> None:
    header = pd.read_csv(PROCESSED_FILE, nrows=0).columns.tolist()
    if header != OUTPUT_COLUMNS:
        raise AssertionError("Stored CAS modeling table columns changed")
    stored = pd.read_csv(
        PROCESSED_FILE,
        dtype={
            "meta_crash_id": "string",
            **{feature: "string" for feature in CATEGORICAL_FEATURES},
        },
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )
    if len(stored) != len(table):
        raise AssertionError("Stored CAS modeling table row count changed")
    if stored["meta_crash_id"].duplicated().any():
        raise AssertionError("Stored CAS identifiers are duplicated")
    if stored[CATEGORICAL_FEATURES].isna().any().any():
        raise AssertionError("Stored categorical features lost explicit missing tokens")
    for feature in NUMERIC_FEATURES:
        if int(stored[feature].isna().sum()) != int(table[feature].isna().sum()):
            raise AssertionError(f"Stored numeric missingness changed: {feature}")


def write_artifact_manifest(paths: list[Path]) -> None:
    rows = [
        {
            "relative_path": relative(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": hash_file(path),
        }
        for path in paths
    ]
    write_csv(ARTIFACT_MANIFEST_FILE, rows)


def main() -> None:
    if SCHEMA_FILE.exists() or CLEANING_FILE.exists():
        raise FileExistsError(
            "CAS modeling schema is already frozen; use an isolated workspace "
            "for reproduction rather than overwriting it"
        )
    protocol = load_protocol()
    raw = load_lossless_snapshot(protocol)
    table = build_modeling_table(raw, protocol)

    write_dataframe_gzip(INTERIM_FILE, table)
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(INTERIM_FILE, PROCESSED_FILE)
    if hash_file(INTERIM_FILE) != hash_file(PROCESSED_FILE):
        raise AssertionError("Interim and processed CAS tables differ")
    validate_round_trip(table)

    target_rows = make_target_audit(table)
    feature_rows = make_feature_audit(table)
    category_rows = make_category_audit(table)
    write_csv(TARGET_AUDIT_FILE, target_rows)
    write_csv(FEATURE_AUDIT_FILE, feature_rows)
    write_csv(CATEGORY_AUDIT_FILE, category_rows)

    observed_categories = {
        feature: sorted(table[feature].unique().tolist())
        for feature in CATEGORICAL_FEATURES
    }
    observed_numeric_values = {
        feature: sorted(
            int(value) for value in table[feature].dropna().unique().tolist()
        )
        for feature in NUMERIC_FEATURES
    }
    missing_counts = {
        feature: (
            int(table[feature].eq(SOURCE_MISSING_TOKEN).sum())
            if feature in CATEGORICAL_FEATURES
            else int(table[feature].isna().sum())
        )
        for feature in FEATURE_COLUMNS
    }
    source_hash = hash_file(RAW_JSONL)
    output_hash = hash_file(PROCESSED_FILE)
    created = datetime.now().astimezone().isoformat(timespec="seconds")

    cleaning = {
        "version": "CAS_MODELING_V1",
        "status": "FROZEN_BEFORE_MODELING",
        "created_local": created,
        "source": relative(RAW_JSONL),
        "source_sha256": source_hash,
        "row_policy": "No rows dropped; frozen snapshot already excludes non-injury crashes.",
        "categorical_policy": {
            "actual_null_or_empty": SOURCE_MISSING_TOKEN,
            "literal_None_and_Null": "preserve as source categories",
            "whitespace": "reject surrounding whitespace rather than silently normalize",
            "training_vocabulary": "fit on each training partition only",
            "unseen_category": "map to __UNSEEN__ at model preprocessing time",
        },
        "numeric_policy": {
            "storage": "numeric value or NaN; no imputation in the modeling table",
            "LightGBM": "native missing-value handling",
            "linear_and_ordered_models": (
                "training-only observed-value imputation plus an explicit missing "
                "indicator; no ordinary mean imputation for structurally sparse fields"
            ),
            "structurally_sparse_features": sorted(
                STRUCTURALLY_SPARSE_NUMERIC
            ),
        },
        "target_policy": {
            "order": [TARGET_LABELS[index] for index in sorted(TARGET_LABELS)],
            "codes": {str(code): label for code, label in TARGET_LABELS.items()},
            "source_mapping": protocol["target_mapping"],
        },
        "test_boundary": (
            "2025 schema, target counts and feature coverage were feasibility-"
            "audited, but no preprocessing fit or model performance is permitted "
            "before model/protocol freeze."
        ),
    }
    write_json_atomic(CLEANING_FILE, cleaning)

    schema = {
        "version": "CAS_MODELING_V1",
        "status": "QC_PASSED_AND_FROZEN_BEFORE_MODELING",
        "created_local": created,
        "file": relative(PROCESSED_FILE),
        "sha256": output_hash,
        "row_count": len(table),
        "statistical_unit": "police-reported personal-injury crash",
        "metadata_columns": METADATA_COLUMNS,
        "metadata_roles": METADATA_ROLES,
        "target_column": TARGET_COLUMN,
        "target_labels": {str(code): label for code, label in TARGET_LABELS.items()},
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "categorical_feature_columns": CATEGORICAL_FEATURES,
        "numeric_feature_columns": NUMERIC_FEATURES,
        "source_to_feature": {
            **CATEGORICAL_SOURCE_TO_FEATURE,
            **NUMERIC_SOURCE_TO_FEATURE,
        },
        "feature_selection_policy": (
            "Model code must select exactly feature_columns; metadata and target "
            "must never enter a model through exclusion-based selection."
        ),
        "observed_categories_for_schema_audit_only": observed_categories,
        "observed_numeric_values": observed_numeric_values,
        "missing_counts": missing_counts,
        "category_vocabulary_policy": (
            "Observed full-snapshot categories are audit evidence only and must "
            "not be supplied to an encoder; fit vocabulary on training rows only."
        ),
        "parent": {
            "feasibility_protocol": relative(FEASIBILITY_PROTOCOL),
            "feasibility_protocol_sha256": hash_file(FEASIBILITY_PROTOCOL),
            "lossless_snapshot": relative(RAW_JSONL),
            "lossless_snapshot_sha256": source_hash,
            "interim_table": relative(INTERIM_FILE),
            "interim_table_sha256": hash_file(INTERIM_FILE),
            "cleaning_rules": relative(CLEANING_FILE),
            "cleaning_rules_sha256": hash_file(CLEANING_FILE),
        },
    }
    write_json_atomic(SCHEMA_FILE, schema)

    checkpoint = [
        "# CAS modeling-table checkpoint",
        "",
        "Status: **QC_PASSED_AND_FROZEN_BEFORE_MODELING**",
        "",
        f"- Rows: **{len(table):,}**",
        f"- Features: **{len(FEATURE_COLUMNS)}** "
        f"({len(CATEGORICAL_FEATURES)} categorical, {len(NUMERIC_FEATURES)} numeric)",
        f"- Processed SHA-256: `{output_hash}`",
        "- Rows removed: **0**",
        "- Literal `None` and `Null` categories remain distinct from missing.",
        "- No encoder, imputer, class weight, resampling step or model was fitted.",
        "- Full-snapshot category lists are audit evidence only; model vocabularies "
        "must be learned from training rows.",
        "",
    ]
    CHECKPOINT_FILE.write_text("\n".join(checkpoint), encoding="utf-8")

    write_artifact_manifest(
        [
            Path(__file__),
            FEASIBILITY_PROTOCOL,
            RAW_JSONL,
            INTERIM_FILE,
            PROCESSED_FILE,
            CLEANING_FILE,
            SCHEMA_FILE,
            TARGET_AUDIT_FILE,
            FEATURE_AUDIT_FILE,
            CATEGORY_AUDIT_FILE,
            CHECKPOINT_FILE,
        ]
    )
    print("CAS_MODELING_TABLE_STATUS=QC_PASSED_AND_FROZEN_BEFORE_MODELING")
    print(f"CAS_MODELING_ROWS={len(table)}")
    print(f"CAS_MODELING_FEATURES={len(FEATURE_COLUMNS)}")
    print(f"CAS_MODELING_SHA256={output_hash}")


if __name__ == "__main__":
    main()
