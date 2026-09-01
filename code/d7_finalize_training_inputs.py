"""Audit frozen D6 splits and save training-only class weights.

D7 does not regenerate any split and does not fit a model. The collision-level
assignments created at D6 remain the single source of truth. Class weights are
computed separately from each training partition only.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROTOCOL_FILE = PROJECT_DIR / "config" / "d6_analysis_protocol.json"
SCHEMA_FILE = PROJECT_DIR / "config" / "d5_dataset_schema.json"
ASSIGNMENTS_FILE = PROJECT_DIR / "data" / "processed" / "d6_split_assignments.csv.gz"

D7_CONFIG_FILE = PROJECT_DIR / "config" / "d7_training_inputs.json"
SPLIT_AUDIT_FILE = PROJECT_DIR / "logs" / "d7_split_integrity.csv"
WEIGHT_TABLE_FILE = PROJECT_DIR / "logs" / "d7_class_weights.csv"
CHECKPOINT_FILE = PROJECT_DIR / "logs" / "d7_checkpoint.md"

TARGET_LABELS = {0: "Slight", 1: "Serious", 2: "Fatal"}
TARGET_CODES = tuple(TARGET_LABELS)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows generated for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_frozen_inputs() -> tuple[dict[str, object], dict[str, object], pd.DataFrame]:
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    if protocol.get("version") != "D6_V2":
        raise ValueError("D7 requires the frozen D6_V2 protocol")
    if protocol.get("status") != "FROZEN_BEFORE_MODEL_TRAINING":
        raise ValueError("D6 protocol is not frozen")
    if schema.get("status") != "QC_PASSED_AND_FROZEN":
        raise ValueError("D5 schema is not frozen")

    expected_hash = protocol["split_assignment_artifact"]["sha256"]
    actual_hash = hash_file(ASSIGNMENTS_FILE)
    if actual_hash != expected_hash:
        raise ValueError("D6 split-assignment hash changed after protocol freeze")

    expected_role_columns = protocol["split_assignment_artifact"]["role_columns"]
    required_columns = [
        "meta_collision_index",
        "meta_collision_year",
        "target_severity",
        *expected_role_columns,
    ]
    assignments = pd.read_csv(
        ASSIGNMENTS_FILE,
        usecols=required_columns,
        dtype={"meta_collision_index": "string"},
        low_memory=False,
    )
    if len(assignments) != int(protocol["frozen_population"]["row_count"]):
        raise ValueError("D6 assignment row count changed")
    if assignments["meta_collision_index"].isna().any():
        raise ValueError("Missing collision identifier in D6 assignments")
    if not assignments["meta_collision_index"].is_unique:
        raise ValueError("Duplicate collision identifier in D6 assignments")
    if set(assignments["target_severity"].unique()) != set(TARGET_CODES):
        raise ValueError("Unexpected target codes in D6 assignments")
    return protocol, schema, assignments


def balanced_class_weights(target: pd.Series) -> tuple[dict[int, int], dict[int, float]]:
    counts = target.astype(int).value_counts().reindex(TARGET_CODES, fill_value=0)
    if (counts <= 0).any():
        raise ValueError("Every class must occur in each training partition")
    total = int(counts.sum())
    class_count = len(TARGET_CODES)
    weights = {
        code: total / (class_count * int(counts.loc[code])) for code in TARGET_CODES
    }
    return {code: int(counts.loc[code]) for code in TARGET_CODES}, weights


def audit_role(
    assignments: pd.DataFrame,
    *,
    protocol_name: str,
    seed: int | str,
    role_column: str,
    role: str,
) -> dict[str, object]:
    subset = assignments.loc[assignments[role_column].eq(role)]
    years = sorted(subset["meta_collision_year"].astype(int).unique())
    return {
        "protocol": protocol_name,
        "seed": seed,
        "role_column": role_column,
        "role": role,
        "rows": len(subset),
        "years": ";".join(str(year) for year in years),
        "unique_collision_ids": int(subset["meta_collision_index"].nunique()),
    }


def build_audits(
    protocol: dict[str, object], assignments: pd.DataFrame
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    split_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    weight_sets: dict[str, object] = {}

    temporal_column = "temporal_role"
    for role in ("train", "validation", "test"):
        split_rows.append(
            audit_role(
                assignments,
                protocol_name="temporal",
                seed="year_based",
                role_column=temporal_column,
                role=role,
            )
        )
    temporal_train = assignments.loc[assignments[temporal_column].eq("train")]
    counts, weights = balanced_class_weights(temporal_train["target_severity"])
    weight_sets["temporal"] = {
        "role_column": temporal_column,
        "training_role": "train",
        "training_rows": len(temporal_train),
        "class_counts": {str(code): counts[code] for code in TARGET_CODES},
        "class_weights": {str(code): weights[code] for code in TARGET_CODES},
    }
    for code in TARGET_CODES:
        weight_rows.append(
            {
                "protocol": "temporal",
                "seed": "year_based",
                "class_code": code,
                "class_label": TARGET_LABELS[code],
                "training_count": counts[code],
                "balanced_weight": weights[code],
            }
        )

    random_sets: dict[str, object] = {}
    for seed in protocol["random_reference_protocol"]["seeds"]:
        role_column = f"random_role_seed_{seed}"
        for role in ("train", "validation", "test", "locked_temporal_test"):
            split_rows.append(
                audit_role(
                    assignments,
                    protocol_name="random_reference",
                    seed=int(seed),
                    role_column=role_column,
                    role=role,
                )
            )
        training = assignments.loc[assignments[role_column].eq("train")]
        counts, weights = balanced_class_weights(training["target_severity"])
        random_sets[str(seed)] = {
            "role_column": role_column,
            "training_role": "train",
            "training_rows": len(training),
            "class_counts": {str(code): counts[code] for code in TARGET_CODES},
            "class_weights": {str(code): weights[code] for code in TARGET_CODES},
        }
        for code in TARGET_CODES:
            weight_rows.append(
                {
                    "protocol": "random_reference",
                    "seed": int(seed),
                    "class_code": code,
                    "class_label": TARGET_LABELS[code],
                    "training_count": counts[code],
                    "balanced_weight": weights[code],
                }
            )
    weight_sets["random_reference"] = random_sets
    return split_rows, weight_rows, weight_sets


def assert_integrity(
    protocol: dict[str, object],
    assignments: pd.DataFrame,
    split_rows: list[dict[str, object]],
    weight_sets: dict[str, object],
) -> None:
    expected_temporal = protocol["temporal_protocol"]["row_counts"]
    observed_temporal = {
        row["role"]: int(row["rows"])
        for row in split_rows
        if row["protocol"] == "temporal"
    }
    if observed_temporal != expected_temporal:
        raise AssertionError(
            f"Temporal counts differ: {observed_temporal} != {expected_temporal}"
        )

    expected_random = protocol["random_reference_protocol"]["row_counts_per_seed"]
    development = assignments["meta_collision_year"].le(2023)
    year_2024 = assignments["meta_collision_year"].eq(2024)
    for seed in protocol["random_reference_protocol"]["seeds"]:
        role_column = f"random_role_seed_{seed}"
        observed = {
            role: int((development & assignments[role_column].eq(role)).sum())
            for role in ("train", "validation", "test")
        }
        if observed != expected_random:
            raise AssertionError(f"Random counts differ for seed {seed}: {observed}")
        if not assignments.loc[year_2024, role_column].eq("locked_temporal_test").all():
            raise AssertionError(f"2024 is not locked for random seed {seed}")

    if weight_sets["temporal"]["training_rows"] != expected_temporal["train"]:
        raise AssertionError("Temporal weights were not computed from training only")
    for seed, payload in weight_sets["random_reference"].items():
        if payload["training_rows"] != expected_random["train"]:
            raise AssertionError(f"Random weights for seed {seed} used non-training rows")


def make_config(
    protocol: dict[str, object],
    schema: dict[str, object],
    weight_sets: dict[str, object],
) -> dict[str, object]:
    return {
        "version": "D7_V1",
        "status": "TRAINING_INPUTS_AUDITED_AND_FROZEN",
        "created_local": "2026-08-28",
        "purpose": (
            "Verify the frozen D6 assignments and save class weights calculated "
            "only from each training partition. No split is regenerated."
        ),
        "upstream": {
            "D6_protocol": PROTOCOL_FILE.relative_to(PROJECT_DIR).as_posix(),
            "D6_protocol_sha256": hash_file(PROTOCOL_FILE),
            "D6_assignments": ASSIGNMENTS_FILE.relative_to(PROJECT_DIR).as_posix(),
            "D6_assignments_sha256": hash_file(ASSIGNMENTS_FILE),
            "D5_schema": SCHEMA_FILE.relative_to(PROJECT_DIR).as_posix(),
            "D5_schema_sha256": hash_file(SCHEMA_FILE),
        },
        "split_storage": {
            "design": "single collision-level table with six frozen role columns",
            "reason": (
                "One combined artifact prevents row drift and avoids five redundant "
                "copies of the same collision identifiers."
            ),
            "role_columns": protocol["split_assignment_artifact"]["role_columns"],
        },
        "feature_contract": {
            "feature_count": len(schema["feature_columns"]),
            "feature_columns": schema["feature_columns"],
            "metadata_excluded_from_models": schema["metadata_columns"],
            "target_excluded_from_features": schema["target_column"],
        },
        "class_weight_rule": {
            "name": "balanced",
            "formula": "n_training / (n_classes * n_training_in_class)",
            "n_classes": len(TARGET_CODES),
            "source_rows": "training role only, calculated separately per split",
            "target_labels": {str(code): TARGET_LABELS[code] for code in TARGET_CODES},
        },
        "weight_sets": weight_sets,
        "test_embargo": {
            "year": 2024,
            "rule": (
                "No preprocessing fit, tuning, threshold choice, model selection or "
                "intermediate performance inspection before the one-time D11 evaluation."
            ),
        },
    }


def write_checkpoint(config: dict[str, object]) -> None:
    temporal = config["weight_sets"]["temporal"]
    random_sets = config["weight_sets"]["random_reference"]
    temporal_weights = temporal["class_weights"]
    lines = [
        "# D7 training-input audit",
        "",
        "## Status",
        "",
        "**PASS - frozen split assignments verified; no split regenerated.**",
        "",
        "## Split storage",
        "",
        "- Temporal roles and all five random-seed roles remain in the single collision-level D6 assignment artifact.",
        "- Temporal counts: train **538,461**, validation **104,258**, locked 2024 test **100,927**.",
        "- Each random seed: train **449,903**, validation **96,408**, internal test **96,408**; all 2024 rows remain `locked_temporal_test`.",
        "- Seeds: **1103, 2207, 3301, 4409, 5501**.",
        "",
        "## Training-only class weights",
        "",
        "Balanced weights use `n_training / (3 * n_training_in_class)` and are computed independently for each training partition.",
        "",
        f"- Temporal Slight (0): **{float(temporal_weights['0']):.8f}**.",
        f"- Temporal Serious (1): **{float(temporal_weights['1']):.8f}**.",
        f"- Temporal Fatal (2): **{float(temporal_weights['2']):.8f}**.",
        f"- Five random training-specific weight sets saved: **{len(random_sets)}**.",
        "",
        "No validation or test observations contribute to any class weight.",
        "",
        "## 2024 embargo",
        "",
        "D8-D10 may use 2018-2022 for fitting and 2023 for validation. The 2024 labels must not be inspected for model performance until every model and decision rule has been frozen for the one-time D11 evaluation.",
        "",
        "## Decision",
        "",
        "D7 is complete. D8 may start only when explicitly requested.",
        "",
    ]
    CHECKPOINT_FILE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    protocol, schema, assignments = load_frozen_inputs()
    split_rows, weight_rows, weight_sets = build_audits(protocol, assignments)
    assert_integrity(protocol, assignments, split_rows, weight_sets)

    write_csv(SPLIT_AUDIT_FILE, split_rows)
    write_csv(WEIGHT_TABLE_FILE, weight_rows)
    config = make_config(protocol, schema, weight_sets)
    write_json(D7_CONFIG_FILE, config)
    write_checkpoint(config)

    print("D7 status:", config["status"])
    print("Temporal counts: 538461 / 104258 / 100927")
    print("Random seeds audited:", len(weight_sets["random_reference"]))
    print("Assignments SHA-256:", hash_file(ASSIGNMENTS_FILE))
    print("FINAL_D7_ASSERTIONS=PASS")


if __name__ == "__main__":
    main()
