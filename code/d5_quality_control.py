"""Run D5 quality control and freeze the processed STATS19 dataset.

The D4 table is validated without silently deleting observations. If a hard
integrity rule fails, execution stops. When all rules pass, the validated D4
file is copied byte-for-byte to the processed directory and accompanied by
auditable quality-control tables and a frozen cleaning policy.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_SCHEMA_FILE = PROJECT_DIR / "config" / "d4_output_schema.json"
INPUT_DATA_FILE = PROJECT_DIR / "data" / "interim" / "d4_modeling_table.csv.gz"
OUTPUT_DATA_FILE = PROJECT_DIR / "data" / "processed" / "stats19_modeling_dataset.csv.gz"
OUTPUT_SCHEMA_FILE = PROJECT_DIR / "config" / "d5_dataset_schema.json"
CLEANING_RULES_FILE = PROJECT_DIR / "config" / "d5_cleaning_rules.json"
LOG_DIR = PROJECT_DIR / "logs"
D1_AUDIT_FILE = LOG_DIR / "d1_initial_audit.csv"

YEARS = tuple(range(2018, 2025))
LATEST_YEAR_DIAGNOSTIC = 2024
RARE_CATEGORY_AUDIT_COUNT = 100
VALID_TARGETS = {0, 1, 2}
VALID_SPEED_LIMITS = {20.0, 30.0, 40.0, 50.0, 60.0, 70.0}
SEMANTIC_MISSING_PATTERN = re.compile(
    r"unknown|missing|out of range|not reported|not applicable|__csv_missing__",
    re.I,
)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows generated for {path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def column_role(
    column: str,
    metadata_columns: list[str],
    target_column: str,
    categorical_columns: list[str],
) -> str:
    if column in metadata_columns:
        return "metadata"
    if column == target_column:
        return "target"
    if column in categorical_columns:
        return "categorical_feature"
    return "numeric_feature"


def make_cleaning_rules(input_sha256: str) -> dict[str, object]:
    return {
        "version": "D5_V1",
        "status": "FROZEN_BEFORE_MODELING",
        "input_file": INPUT_DATA_FILE.relative_to(PROJECT_DIR).as_posix(),
        "input_sha256": input_sha256,
        "prediction_scope": "collision table only; conditional severity prediction",
        "hard_fail_rules": {
            "schema": "exact D4 schema and column order required",
            "row_count": "must match the frozen D4 row count",
            "collision_index": "must be nonmissing and globally unique",
            "target": "must be nonmissing and restricted to 0, 1, 2",
            "date_year": "meta_date must parse and match meta_collision_year",
            "categorical_features": "must be nonmissing because official unknown/not-applicable values are explicit labels",
            "speed_limit": "nonmissing values must be one of 20, 30, 40, 50, 60, 70",
        },
        "row_retention": {
            "duplicates": "do not auto-delete; fail and investigate if collision identifiers repeat",
            "unknown_categories": "retain all records and explicit official categories",
            "speed_limit_missing": "retain as NaN",
            "rare_categories": "retain at D5; do not pool using the full dataset",
        },
        "split_aware_category_policy": {
            "fit_scope": "fit category vocabulary on each training partition only",
            "unseen_validation_or_test": "map to __UNSEEN__",
            "rare_audit_threshold": RARE_CATEGORY_AUDIT_COUNT,
            "rare_regular_categories": "if pooling is used, map training categories below the threshold to __RARE__ using training counts only",
            "official_unknown_or_not_applicable": "retain the original semantic label in frozen data and exempt it from generic rare pooling when observed in training; if absent from training, the model encoder maps it to __UNSEEN__ while the audit preserves the original label",
            "latest_year_check": "2018-2023 versus 2024 is diagnostic only; D6 freezes the final temporal split",
        },
        "missing_value_policy": {
            "LightGBM": "native missing-value handling for speed_limit",
            "multinomial_logistic": "fit numeric imputation within each training fold only",
            "ordered_logit": "fit numeric imputation within each training fold only",
        },
        "class_imbalance_policy": {
            "D5": "no over-sampling, under-sampling, class weighting or synthetic data",
            "primary_modeling": "class weights are computed from each training partition only",
            "SMOTENC": "optional sensitivity analysis only and training-fold only",
        },
        "feature_selection": "use only the exact feature_columns allowlist in the frozen schema",
    }


def main() -> None:
    schema = json.loads(INPUT_SCHEMA_FILE.read_text(encoding="utf-8"))
    metadata_columns = list(schema["metadata_columns"])
    target_column = str(schema["target_column"])
    feature_columns = list(schema["feature_columns"])
    categorical_columns = list(schema["categorical_feature_columns"])
    numeric_columns = list(schema["numeric_feature_columns"])
    expected_columns = metadata_columns + [target_column] + feature_columns

    input_sha256 = hash_file(INPUT_DATA_FILE)
    if input_sha256 != schema["sha256"]:
        raise ValueError("D4 file hash differs from its frozen schema")
    if len(feature_columns) != 17 or len(set(feature_columns)) != 17:
        raise ValueError("Expected 17 unique frozen feature columns")

    actual_header = pd.read_csv(INPUT_DATA_FILE, nrows=0).columns.tolist()
    if actual_header != expected_columns:
        raise ValueError(
            f"D4 schema mismatch: expected={expected_columns}, actual={actual_header}"
        )

    data = pd.read_csv(INPUT_DATA_FILE, low_memory=False)
    row_count = len(data)
    if row_count != int(schema["row_count"]):
        raise ValueError(
            f"D4 row count mismatch: expected={schema['row_count']}, actual={row_count}"
        )

    duplicate_id_count = int(data["meta_collision_index"].duplicated().sum())
    missing_id_count = int(data["meta_collision_index"].isna().sum())
    if duplicate_id_count or missing_id_count:
        raise ValueError(
            f"Collision identifier failure: duplicates={duplicate_id_count}, "
            f"missing={missing_id_count}"
        )

    target_values = set(data[target_column].dropna().astype(int).unique())
    target_missing = int(data[target_column].isna().sum())
    if target_missing or target_values != VALID_TARGETS:
        raise ValueError(
            f"Target failure: missing={target_missing}, values={sorted(target_values)}"
        )

    years = set(data["meta_collision_year"].dropna().astype(int).unique())
    if years != set(YEARS):
        raise ValueError(f"Unexpected year set: {sorted(years)}")
    parsed_dates = pd.to_datetime(data["meta_date"], format="%Y-%m-%d", errors="coerce")
    invalid_date_count = int(parsed_dates.isna().sum())
    date_year_mismatch = int(
        (~parsed_dates.dt.year.eq(data["meta_collision_year"])).sum()
    )
    if invalid_date_count or date_year_mismatch:
        raise ValueError(
            f"Date failure: invalid={invalid_date_count}, "
            f"year_mismatch={date_year_mismatch}"
        )

    categorical_missing = {
        column: int(data[column].isna().sum()) for column in categorical_columns
    }
    categorical_missing = {
        column: count for column, count in categorical_missing.items() if count
    }
    if categorical_missing:
        raise ValueError(
            f"Unexpected categorical feature missingness: {categorical_missing}"
        )
    if numeric_columns != ["feature_speed_limit"]:
        raise ValueError(f"Unexpected numeric feature columns: {numeric_columns}")
    observed_speed = set(data["feature_speed_limit"].dropna().astype(float).unique())
    invalid_speed = sorted(observed_speed - VALID_SPEED_LIMITS)
    if invalid_speed:
        raise ValueError(f"Invalid speed-limit values: {invalid_speed}")

    feature_metadata_overlap = set(feature_columns).intersection(
        metadata_columns + [target_column]
    )
    if feature_metadata_overlap:
        raise ValueError(
            f"Metadata/target entered feature allowlist: {feature_metadata_overlap}"
        )

    column_rows: list[dict[str, object]] = []
    for column in expected_columns:
        series = data[column]
        nonmissing = series.dropna()
        numeric = pd.api.types.is_numeric_dtype(series)
        column_rows.append(
            {
                "column_name": column,
                "role": column_role(
                    column, metadata_columns, target_column, categorical_columns
                ),
                "pandas_dtype": str(series.dtype),
                "row_count": row_count,
                "missing_count": int(series.isna().sum()),
                "missing_pct": float(series.isna().mean()),
                "unique_nonmissing": int(nonmissing.nunique()),
                "is_constant": int(nonmissing.nunique()) <= 1,
                "minimum": float(nonmissing.min()) if numeric and len(nonmissing) else "",
                "maximum": float(nonmissing.max()) if numeric and len(nonmissing) else "",
            }
        )

    category_rows: list[dict[str, object]] = []
    rare_category_count = 0
    unseen_latest_count = 0
    latest_mask = data["meta_collision_year"].eq(LATEST_YEAR_DIAGNOSTIC)
    for column in categorical_columns:
        grouped = (
            data.groupby([column, "meta_collision_year"], observed=True)
            .size()
            .unstack(fill_value=0)
        )
        for category, counts in grouped.iterrows():
            count_by_year = {year: int(counts.get(year, 0)) for year in YEARS}
            total = sum(count_by_year.values())
            pre_latest = sum(
                count for year, count in count_by_year.items() if year < LATEST_YEAR_DIAGNOSTIC
            )
            latest = count_by_year[LATEST_YEAR_DIAGNOSTIC]
            semantic_missing = bool(SEMANTIC_MISSING_PATTERN.search(str(category)))
            rare_global = total < RARE_CATEGORY_AUDIT_COUNT
            unseen_latest = pre_latest == 0 and latest > 0
            absent_latest = pre_latest > 0 and latest == 0
            rare_category_count += int(rare_global)
            unseen_latest_count += int(unseen_latest)
            if unseen_latest:
                action = "map_to___UNSEEN___when_2024_is_out_of_sample"
            elif semantic_missing:
                action = "retain_explicit_semantic_category"
            elif rare_global:
                action = "audit_flag_only; any pooling must be training-only"
            else:
                action = "retain_category"
            row = {
                "feature_name": column,
                "category": category,
                "total_count": total,
                "total_pct": total / row_count,
                "years_present": ";".join(
                    str(year) for year, count in count_by_year.items() if count
                ),
                "pre_2024_count": pre_latest,
                "year_2024_count": latest,
                "rare_global_lt_100": rare_global,
                "semantic_unknown_or_not_applicable": semantic_missing,
                "unseen_in_2018_2023_but_present_2024": unseen_latest,
                "present_2018_2023_but_absent_2024": absent_latest,
                "D5_action": action,
            }
            row.update({f"count_{year}": count_by_year[year] for year in YEARS})
            category_rows.append(row)

    target_rows: list[dict[str, object]] = []
    target_labels = {0: "Slight", 1: "Serious", 2: "Fatal"}
    for year_label, subset in [
        *[(str(year), data[data["meta_collision_year"].eq(year)]) for year in YEARS],
        ("ALL", data),
    ]:
        counts = subset[target_column].value_counts().to_dict()
        for target in sorted(VALID_TARGETS):
            count = int(counts.get(target, 0))
            target_rows.append(
                {
                    "year": year_label,
                    "target_code": target,
                    "target_label": target_labels[target],
                    "count": count,
                    "pct": count / len(subset),
                    "year_total": len(subset),
                }
            )

    group_rows: list[dict[str, object]] = []
    for metadata_field in ["meta_police_force", "meta_injury_based"]:
        grouped = (
            data.groupby([metadata_field, "meta_collision_year"], dropna=False)
            .size()
            .unstack(fill_value=0)
        )
        for category, counts in grouped.iterrows():
            count_by_year = {year: int(counts.get(year, 0)) for year in YEARS}
            total = sum(count_by_year.values())
            row = {
                "metadata_field": metadata_field,
                "category": category,
                "total_count": total,
                "total_pct": total / row_count,
                "years_present": ";".join(
                    str(year) for year, count in count_by_year.items() if count
                ),
            }
            row.update({f"count_{year}": count_by_year[year] for year in YEARS})
            group_rows.append(row)

    d1_rows = {
        int(row["year"]): int(row["rows"])
        for row in read_csv_dicts(D1_AUDIT_FILE)
    }
    d4_year_counts = (
        data["meta_collision_year"].value_counts().sort_index().astype(int).to_dict()
    )
    sample_flow_rows: list[dict[str, object]] = []
    for year in YEARS:
        d1_count = d1_rows[year]
        d4_count = int(d4_year_counts[year])
        d5_count = d4_count
        sample_flow_rows.append(
            {
                "year": year,
                "D1_raw_rows": d1_count,
                "D4_feature_rows": d4_count,
                "D5_final_rows": d5_count,
                "removed_D1_to_D4": d1_count - d4_count,
                "removed_D4_to_D5": d4_count - d5_count,
                "retained_pct_of_D1": d5_count / d1_count,
            }
        )
    sample_flow_rows.append(
        {
            "year": "ALL",
            "D1_raw_rows": sum(d1_rows.values()),
            "D4_feature_rows": row_count,
            "D5_final_rows": row_count,
            "removed_D1_to_D4": sum(d1_rows.values()) - row_count,
            "removed_D4_to_D5": 0,
            "retained_pct_of_D1": row_count / sum(d1_rows.values()),
        }
    )

    cleaning_rules = make_cleaning_rules(input_sha256)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SCHEMA_FILE.parent.mkdir(parents=True, exist_ok=True)
    CLEANING_RULES_FILE.write_text(
        json.dumps(cleaning_rules, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(LOG_DIR / "d5_column_quality.csv", column_rows)
    write_csv(LOG_DIR / "d5_category_quality.csv", category_rows)
    write_csv(LOG_DIR / "d5_target_distribution.csv", target_rows)
    write_csv(LOG_DIR / "d5_group_quality.csv", group_rows)
    write_csv(LOG_DIR / "d5_sample_flow.csv", sample_flow_rows)

    part_file = OUTPUT_DATA_FILE.with_suffix(OUTPUT_DATA_FILE.suffix + ".part")
    part_file.unlink(missing_ok=True)
    try:
        shutil.copyfile(INPUT_DATA_FILE, part_file)
        copied_sha256 = hash_file(part_file)
        if copied_sha256 != input_sha256:
            raise IOError("Processed copy hash differs from validated D4 input")
        os.replace(part_file, OUTPUT_DATA_FILE)
    except Exception:
        part_file.unlink(missing_ok=True)
        raise

    output_sha256 = hash_file(OUTPUT_DATA_FILE)
    speed_missing = int(data["feature_speed_limit"].isna().sum())
    all_target_counts = Counter(data[target_column].astype(int).tolist())
    output_schema = {
        **schema,
        "version": "D5_V1",
        "status": "QC_PASSED_AND_FROZEN",
        "parent_D4_file": INPUT_DATA_FILE.relative_to(PROJECT_DIR).as_posix(),
        "parent_D4_sha256": input_sha256,
        "file": OUTPUT_DATA_FILE.relative_to(PROJECT_DIR).as_posix(),
        "sha256": output_sha256,
        "row_count": row_count,
        "cleaning_rules": CLEANING_RULES_FILE.relative_to(PROJECT_DIR).as_posix(),
        "rows_removed_at_D5": 0,
    }
    OUTPUT_SCHEMA_FILE.write_text(
        json.dumps(output_schema, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "# D5 data-quality audit and frozen dataset",
        "",
        "## Result",
        "",
        f"- Records audited and retained: **{row_count:,}** (100.000%).",
        f"- Duplicate collision identifiers: **{duplicate_id_count}**.",
        f"- Invalid dates or date/year mismatches: **{invalid_date_count + date_year_mismatch}**.",
        f"- Missing categorical feature values: **{sum(categorical_missing.values())}**.",
        f"- Unknown speed-limit values retained as NaN: **{speed_missing} ({speed_missing / row_count:.3%})**.",
        f"- D5 row deletions: **0**.",
        "",
        "## Target distribution",
        "",
        f"- Slight: {all_target_counts[0]:,} ({all_target_counts[0] / row_count:.3%}).",
        f"- Serious: {all_target_counts[1]:,} ({all_target_counts[1] / row_count:.3%}).",
        f"- Fatal: {all_target_counts[2]:,} ({all_target_counts[2] / row_count:.3%}).",
        "",
        "The target is strongly imbalanced. D5 does not resample the data. Class weights, if used, must be computed inside each training partition; SMOTENC remains an optional training-fold-only sensitivity analysis.",
        "",
        "## Category audit",
        "",
        f"- Categories with fewer than {RARE_CATEGORY_AUDIT_COUNT} records globally: **{rare_category_count}**.",
        f"- Categories present in 2024 but absent in 2018-2023: **{unseen_latest_count}**.",
        "- No category is pooled using the complete dataset. Frozen data preserve official labels. Future encoders fit vocabulary on training data only; a validation/test category absent from training maps to `__UNSEEN__`, even when its official label denotes unknown/not applicable. The audit retains the original label.",
        "- The 2018-2023 versus 2024 comparison is a quality diagnostic only; the final temporal split is frozen at D6.",
        "",
        "## Provenance",
        "",
        f"- Parent D4 SHA-256: `{input_sha256}`.",
        f"- Frozen D5 SHA-256: `{output_sha256}`.",
        "- The hashes are identical because all hard checks passed and no record required deletion or value rewriting at D5.",
    ]
    (LOG_DIR / "d5_summary.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    print("D5 quality control completed.")
    print(f"  rows retained: {row_count:,}")
    print(f"  duplicate identifiers: {duplicate_id_count}")
    print(f"  rare categories flagged: {rare_category_count}")
    print(f"  latest-year unseen categories: {unseen_latest_count}")
    print(f"  output sha256: {output_sha256}")


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
