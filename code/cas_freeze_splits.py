"""Freeze CAS temporal/random split assignments and training-only weights.

This stage reads the frozen model-ready table, creates deterministic assignment
columns and records the evaluation contract. It does not fit or evaluate a
predictive model.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "data" / "processed" / "cas_modeling_dataset.csv.gz"
SCHEMA_FILE = PROJECT_DIR / "config" / "cas" / "cas_modeling_schema.json"
CLEANING_FILE = PROJECT_DIR / "config" / "cas" / "cas_cleaning_rules.json"
FEASIBILITY_PROTOCOL = (
    PROJECT_DIR / "config" / "cas" / "cas_feasibility_protocol.json"
)
PROTOCOL_FILE = PROJECT_DIR / "config" / "cas" / "cas_analysis_protocol.json"
TRAINING_INPUTS_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_training_inputs.json"
)
ASSIGNMENTS_FILE = (
    PROJECT_DIR / "data" / "processed" / "cas_split_assignments.csv.gz"
)
ANNUAL_AUDIT_FILE = PROJECT_DIR / "logs" / "cas" / "cas_annual_severity.csv"
TEMPORAL_AUDIT_FILE = (
    PROJECT_DIR / "logs" / "cas" / "cas_temporal_split_audit.csv"
)
RANDOM_AUDIT_FILE = (
    PROJECT_DIR / "logs" / "cas" / "cas_random_split_audit.csv"
)
CHECKPOINT_FILE = PROJECT_DIR / "logs" / "cas" / "cas_split_checkpoint.md"
ARTIFACT_MANIFEST_FILE = (
    PROJECT_DIR / "logs" / "cas" / "cas_split_artifact_manifest.csv"
)

TARGET_COLUMN = "target_severity"
TARGET_LABELS = {0: "Minor", 1: "Serious", 2: "Fatal"}
ALL_YEARS = (2022, 2023, 2024, 2025)
DEVELOPMENT_YEARS = (2022, 2023, 2024)
TEMPORAL_TRAIN_YEARS = (2022, 2023)
TEMPORAL_VALIDATION_YEAR = 2024
TEMPORAL_TEST_YEAR = 2025

RANDOM_TRAIN_RATIO = 0.70
RANDOM_VALIDATION_RATIO = 0.15
RANDOM_TEST_RATIO = 0.15
RANDOM_SEEDS = (1103, 2207, 3301, 4409, 5501)
RANDOM_SECOND_STAGE_OFFSET = 1_000_003
FATAL_PREFLIGHT_THRESHOLD = 50
BOOTSTRAP_ITERATIONS = 2_000
BOOTSTRAP_SEED = 20_260_901

ASYMMETRIC_COST_MATRIX = [
    [0.0, 1.0, 2.0],
    [2.0, 0.0, 1.0],
    [5.0, 3.0, 0.0],
]


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_DIR.resolve()).as_posix()


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
                frame.to_csv(text, index=False, lineterminator="\n")
    temporary.replace(path)


def load_inputs() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    cleaning = json.loads(CLEANING_FILE.read_text(encoding="utf-8"))
    if schema.get("status") != "QC_PASSED_AND_FROZEN_BEFORE_MODELING":
        raise ValueError("CAS modeling schema is not QC-passed and frozen")
    if cleaning.get("status") != "FROZEN_BEFORE_MODELING":
        raise ValueError("CAS cleaning rules are not frozen")
    if hash_file(DATA_FILE) != schema["sha256"]:
        raise ValueError("CAS modeling dataset hash differs from schema")
    if hash_file(CLEANING_FILE) != schema["parent"]["cleaning_rules_sha256"]:
        raise ValueError("CAS cleaning rules changed after schema freeze")

    required = ["meta_crash_id", "meta_crash_year", TARGET_COLUMN]
    table = pd.read_csv(
        DATA_FILE,
        usecols=required,
        dtype={"meta_crash_id": "string"},
        low_memory=False,
    )
    if len(table) != int(schema["row_count"]):
        raise ValueError("CAS modeling row count changed")
    if table["meta_crash_id"].isna().any() or table["meta_crash_id"].duplicated().any():
        raise ValueError("CAS model identifiers are missing or duplicated")
    observed_years = set(table["meta_crash_year"].astype(int).unique())
    if observed_years != set(ALL_YEARS):
        raise ValueError(f"Unexpected CAS years: {sorted(observed_years)}")
    if set(table[TARGET_COLUMN].astype(int).unique()) != set(TARGET_LABELS):
        raise ValueError("CAS target codes are incomplete")
    return table, schema, cleaning


def role_summary(
    table: pd.DataFrame,
    role_column: str,
    roles: tuple[str, ...],
    *,
    seed: int | str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for role in roles:
        subset = table.loc[table[role_column].eq(role)]
        counts = subset[TARGET_COLUMN].value_counts().to_dict()
        row: dict[str, object] = {
            "seed": seed,
            "split": role,
            "years": ";".join(
                str(year)
                for year in sorted(subset["meta_crash_year"].astype(int).unique())
            ),
            "rows": len(subset),
        }
        for code, label in TARGET_LABELS.items():
            count = int(counts.get(code, 0))
            row[f"{label.lower()}_n"] = count
            row[f"{label.lower()}_pct"] = count / len(subset) if len(subset) else np.nan
        rows.append(row)
    relevant_total = sum(int(row["rows"]) for row in rows)
    for row in rows:
        row["share_of_relevant_pool"] = int(row["rows"]) / relevant_total
    return rows


def annual_summary(table: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for year in ALL_YEARS:
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


def assign_splits(
    table: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    assignments = table.copy()
    year = assignments["meta_crash_year"].astype(int)

    temporal_role = np.full(len(assignments), "", dtype=object)
    temporal_role[year.isin(TEMPORAL_TRAIN_YEARS).to_numpy()] = "train"
    temporal_role[year.eq(TEMPORAL_VALIDATION_YEAR).to_numpy()] = "validation"
    temporal_role[year.eq(TEMPORAL_TEST_YEAR).to_numpy()] = "test"
    if np.any(temporal_role == ""):
        raise AssertionError("A CAS row lacks a temporal role")
    assignments["temporal_role"] = temporal_role
    temporal_rows = role_summary(
        assignments,
        "temporal_role",
        ("train", "validation", "test"),
        seed="year_based",
    )

    development_mask = year.isin(DEVELOPMENT_YEARS).to_numpy()
    development_positions = np.flatnonzero(development_mask)
    development_targets = assignments.loc[
        development_mask, TARGET_COLUMN
    ].astype(int).to_numpy()
    local_positions = np.arange(len(development_positions))
    random_rows: list[dict[str, object]] = []

    for seed in RANDOM_SEEDS:
        local_train, local_remainder = train_test_split(
            local_positions,
            train_size=RANDOM_TRAIN_RATIO,
            random_state=seed,
            stratify=development_targets,
        )
        local_validation, local_test = train_test_split(
            local_remainder,
            test_size=0.5,
            random_state=seed + RANDOM_SECOND_STAGE_OFFSET,
            stratify=development_targets[local_remainder],
        )
        role = np.full(len(assignments), "locked_temporal_test", dtype=object)
        role[development_positions[local_train]] = "train"
        role[development_positions[local_validation]] = "validation"
        role[development_positions[local_test]] = "test"
        role_column = f"random_role_seed_{seed}"
        assignments[role_column] = role
        if np.any(role[~development_mask] != "locked_temporal_test"):
            raise AssertionError(f"2025 was not locked for random seed {seed}")
        random_rows.extend(
            role_summary(
                assignments.loc[development_mask],
                role_column,
                ("train", "validation", "test"),
                seed=seed,
            )
        )
    return assignments, temporal_rows, random_rows


def class_weight_payload(
    assignments: pd.DataFrame,
    role_column: str,
) -> dict[str, object]:
    training = assignments.loc[assignments[role_column].eq("train"), TARGET_COLUMN]
    counts = training.value_counts().sort_index()
    if set(counts.index.astype(int)) != set(TARGET_LABELS):
        raise ValueError(f"Training split lacks a target class: {role_column}")
    n_rows = len(training)
    n_classes = len(TARGET_LABELS)
    weights = {
        str(code): n_rows / (n_classes * int(counts.loc[code]))
        for code in TARGET_LABELS
    }
    return {
        "role_column": role_column,
        "training_rows": n_rows,
        "class_counts": {
            str(code): int(counts.loc[code]) for code in TARGET_LABELS
        },
        "class_weights": weights,
    }


def rows_by_role(rows: list[dict[str, object]]) -> dict[str, int]:
    return {str(row["split"]): int(row["rows"]) for row in rows}


def target_counts_by_role(
    rows: list[dict[str, object]],
) -> dict[str, dict[str, int]]:
    return {
        str(row["split"]): {
            label: int(row[f"{label.lower()}_n"])
            for label in TARGET_LABELS.values()
        }
        for row in rows
    }


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
    if PROTOCOL_FILE.exists() or TRAINING_INPUTS_FILE.exists():
        raise FileExistsError(
            "CAS split protocol is already frozen; do not regenerate in place"
        )
    table, schema, cleaning = load_inputs()
    assignments, temporal_rows, random_rows = assign_splits(table)
    annual_rows = annual_summary(table)

    role_columns = [
        "temporal_role",
        *[f"random_role_seed_{seed}" for seed in RANDOM_SEEDS],
    ]
    expected_columns = [
        "meta_crash_id",
        "meta_crash_year",
        TARGET_COLUMN,
        *role_columns,
    ]
    if assignments.columns.tolist() != expected_columns:
        raise AssertionError("CAS assignment column contract changed")
    write_dataframe_gzip(ASSIGNMENTS_FILE, assignments)
    assignment_hash = hash_file(ASSIGNMENTS_FILE)
    write_csv(ANNUAL_AUDIT_FILE, annual_rows)
    write_csv(TEMPORAL_AUDIT_FILE, temporal_rows)
    write_csv(RANDOM_AUDIT_FILE, random_rows)

    temporal_weights = class_weight_payload(assignments, "temporal_role")
    random_weights = {
        str(seed): class_weight_payload(
            assignments, f"random_role_seed_{seed}"
        )
        for seed in RANDOM_SEEDS
    }
    created = datetime.now().astimezone().isoformat(timespec="seconds")
    training_inputs = {
        "version": "CAS_SPLITS_V1",
        "status": "TRAINING_INPUTS_AUDITED_AND_FROZEN_BEFORE_MODELING",
        "created_local": created,
        "upstream": {
            "modeling_dataset": relative(DATA_FILE),
            "modeling_dataset_sha256": hash_file(DATA_FILE),
            "modeling_schema": relative(SCHEMA_FILE),
            "modeling_schema_sha256": hash_file(SCHEMA_FILE),
            "cleaning_rules": relative(CLEANING_FILE),
            "cleaning_rules_sha256": hash_file(CLEANING_FILE),
        },
        "split_assignments": {
            "file": relative(ASSIGNMENTS_FILE),
            "sha256": assignment_hash,
            "role_columns": role_columns,
        },
        "feature_contract": {
            "feature_count": int(schema["feature_count"]),
            "feature_columns": schema["feature_columns"],
            "metadata_excluded_from_models": schema["metadata_columns"],
            "target_excluded_from_features": TARGET_COLUMN,
        },
        "class_weight_rule": {
            "name": "balanced",
            "formula": "n_training / (n_classes * n_training_in_class)",
            "n_classes": len(TARGET_LABELS),
            "source_rows": "training role only, calculated separately per split",
            "target_labels": {str(code): label for code, label in TARGET_LABELS.items()},
        },
        "weight_sets": {
            "temporal": temporal_weights,
            "random_reference": random_weights,
        },
        "test_boundary": {
            "year": TEMPORAL_TEST_YEAR,
            "prior_access": (
                "Schema, target counts, missingness/category coverage and drift "
                "were inspected during feasibility auditing."
            ),
            "prohibited_before_model_freeze": [
                "preprocessing fit",
                "hyperparameter tuning",
                "threshold choice",
                "model selection",
                "model performance inspection",
            ],
        },
    }
    write_json_atomic(TRAINING_INPUTS_FILE, training_inputs)

    temporal_counts = rows_by_role(temporal_rows)
    temporal_targets = target_counts_by_role(temporal_rows)
    random_seed_rows = [
        row for row in random_rows if int(row["seed"]) == RANDOM_SEEDS[0]
    ]
    random_counts = rows_by_role(random_seed_rows)
    fatal_test_count = temporal_targets["test"]["Fatal"]
    protocol = {
        "version": "CAS_SPLITS_V1",
        "status": "FROZEN_BEFORE_ANY_CAS_MODEL_TRAINING",
        "created_local": created,
        "decision_basis": (
            "Frozen after feasibility/schema auditing and before model fitting or "
            "performance inspection. The 2025 schema and distributions were "
            "previously inspected and are disclosed; no 2025 model result exists."
        ),
        "upstream": {
            "feasibility_protocol": relative(FEASIBILITY_PROTOCOL),
            "feasibility_protocol_sha256": hash_file(FEASIBILITY_PROTOCOL),
            "modeling_dataset": relative(DATA_FILE),
            "modeling_dataset_sha256": hash_file(DATA_FILE),
            "modeling_schema": relative(SCHEMA_FILE),
            "modeling_schema_sha256": hash_file(SCHEMA_FILE),
            "cleaning_rules": relative(CLEANING_FILE),
            "cleaning_rules_sha256": hash_file(CLEANING_FILE),
        },
        "population": {
            "years": list(ALL_YEARS),
            "row_count": len(table),
            "statistical_unit": "police-reported personal-injury crash",
            "target_order": [TARGET_LABELS[code] for code in TARGET_LABELS],
            "feature_count": int(schema["feature_count"]),
        },
        "temporal_protocol": {
            "purpose": "primary deployment-oriented evaluation",
            "train_years": list(TEMPORAL_TRAIN_YEARS),
            "validation_years": [TEMPORAL_VALIDATION_YEAR],
            "test_years": [TEMPORAL_TEST_YEAR],
            "row_counts": temporal_counts,
            "target_counts": temporal_targets,
            "model_selection": "training and validation only",
            "final_test_rule": (
                "Evaluate frozen models once on 2025. Do not use 2025 for "
                "preprocessing fit, tuning, threshold selection or model choice."
            ),
        },
        "random_reference_protocol": {
            "purpose": "same-period random reference, not a deployment estimate",
            "pool_years": list(DEVELOPMENT_YEARS),
            "pool_row_count": int(
                table["meta_crash_year"].isin(DEVELOPMENT_YEARS).sum()
            ),
            "ratios": {
                "train": RANDOM_TRAIN_RATIO,
                "validation": RANDOM_VALIDATION_RATIO,
                "test": RANDOM_TEST_RATIO,
            },
            "row_counts_per_seed": random_counts,
            "stratify_by": TARGET_COLUMN,
            "seeds": list(RANDOM_SEEDS),
            "second_stage_seed_rule": (
                f"seed + {RANDOM_SECOND_STAGE_OFFSET}"
            ),
            "locked_temporal_test_year": TEMPORAL_TEST_YEAR,
            "future_diagnostic": (
                "After each random model is frozen, apply it once to 2025 "
                "without retuning; report separately from its internal test."
            ),
        },
        "class_feasibility_preflight": {
            "year": TEMPORAL_TEST_YEAR,
            "fatal_count": fatal_test_count,
            "screening_threshold": FATAL_PREFLIGHT_THRESHOLD,
            "decision": (
                "proceed"
                if fatal_test_count >= FATAL_PREFLIGHT_THRESHOLD
                else "retain_with_high_uncertainty_warning"
            ),
            "interpretation": (
                "This checks only extreme sparsity and does not guarantee narrow "
                "fatal-recall confidence intervals."
            ),
        },
        "training_only_operations": [
            "categorical vocabulary fitting",
            "unseen-category mapping",
            "numeric imputation and missing-indicator fitting for linear models",
            "class-weight calculation",
            "hyperparameter selection",
        ],
        "planned_model_scope": {
            "main": [
                "dummy_most_frequent",
                "logistic_weighted",
                "lightgbm_weighted",
            ],
            "rule": (
                "Any additional model is appendix-only and must be declared before "
                "its performance is inspected."
            ),
        },
        "frozen_metrics": {
            "selection_metric": "Macro-F1 on validation data",
            "required_reporting": [
                "Macro-F1",
                "quadratic weighted kappa",
                "ordinal MAE",
                "accuracy",
                "class-specific recall",
                "serious-or-fatal recall",
                "pre-specified asymmetric error cost",
            ],
            "asymmetric_cost_matrix": ASYMMETRIC_COST_MATRIX,
            "cost_matrix_note": (
                "Rows are true Minor/Serious/Fatal and columns are predicted "
                "Minor/Serious/Fatal; descriptive, not a tuning target."
            ),
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_unit": "crash record, stratified by true class",
        },
        "directional_replication_questions": {
            "H1": (
                "Compare each metric's random-internal versus same-random-model "
                "2025 gap; do not force a uniform optimism claim."
            ),
            "H2": (
                "Compare weighted LightGBM minus weighted Logistic on random and "
                "temporal tests, including Macro-F1 and fatal-recall trade-offs."
            ),
            "H3": (
                "Describe LightGBM mean-absolute-SHAP feature-rank stability "
                "between internal and 2025 cohorts using Spearman correlation."
            ),
            "cross_dataset_rule": (
                "Report direction and uncertainty separately. Do not pool records, "
                "absolute metrics or claim cross-national generalizability."
            ),
        },
        "assignments": {
            "file": relative(ASSIGNMENTS_FILE),
            "sha256": assignment_hash,
            "role_columns": role_columns,
        },
    }
    write_json_atomic(PROTOCOL_FILE, protocol)

    checkpoint = [
        "# CAS split and training-input checkpoint",
        "",
        "Status: **FROZEN_BEFORE_ANY_CAS_MODEL_TRAINING**",
        "",
        f"- Temporal train (2022-2023): **{temporal_counts['train']:,}**",
        f"- Temporal validation (2024): **{temporal_counts['validation']:,}**",
        f"- Temporal test (2025): **{temporal_counts['test']:,}**",
        f"- 2025 Fatal crashes: **{fatal_test_count:,}**",
        f"- Random pool (2022-2024): **{sum(random_counts.values()):,}**",
        f"- Random rows per seed: train **{random_counts['train']:,}**, "
        f"validation **{random_counts['validation']:,}**, "
        f"test **{random_counts['test']:,}**",
        f"- Seeds: **{', '.join(str(seed) for seed in RANDOM_SEEDS)}**",
        f"- Assignment SHA-256: `{assignment_hash}`",
        "- Class weights use training roles only.",
        "- No model has been fitted or evaluated.",
        "",
    ]
    CHECKPOINT_FILE.write_text("\n".join(checkpoint), encoding="utf-8")

    write_artifact_manifest(
        [
            Path(__file__),
            DATA_FILE,
            SCHEMA_FILE,
            CLEANING_FILE,
            ASSIGNMENTS_FILE,
            PROTOCOL_FILE,
            TRAINING_INPUTS_FILE,
            ANNUAL_AUDIT_FILE,
            TEMPORAL_AUDIT_FILE,
            RANDOM_AUDIT_FILE,
            CHECKPOINT_FILE,
        ]
    )
    print("CAS_SPLIT_STATUS=FROZEN_BEFORE_ANY_CAS_MODEL_TRAINING")
    print(
        "CAS_TEMPORAL_ROWS="
        f"{temporal_counts['train']}/{temporal_counts['validation']}/"
        f"{temporal_counts['test']}"
    )
    print(
        "CAS_RANDOM_ROWS_PER_SEED="
        f"{random_counts['train']}/{random_counts['validation']}/"
        f"{random_counts['test']}"
    )
    print(f"CAS_ASSIGNMENTS_SHA256={assignment_hash}")


if __name__ == "__main__":
    main()

