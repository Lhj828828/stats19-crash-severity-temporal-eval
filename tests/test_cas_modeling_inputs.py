"""Independent checks for frozen CAS modeling data and split contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "data" / "processed" / "cas_modeling_dataset.csv.gz"
ASSIGNMENTS_FILE = (
    PROJECT_DIR / "data" / "processed" / "cas_split_assignments.csv.gz"
)
SCHEMA_FILE = PROJECT_DIR / "config" / "cas" / "cas_modeling_schema.json"
CLEANING_FILE = PROJECT_DIR / "config" / "cas" / "cas_cleaning_rules.json"
PROTOCOL_FILE = PROJECT_DIR / "config" / "cas" / "cas_analysis_protocol.json"
TRAINING_INPUTS_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_training_inputs.json"
)
LEAKAGE_AUDIT_FILE = PROJECT_DIR / "logs" / "cas" / "cas_leakage_audit.csv"

TARGET_LABELS = {0: "Minor", 1: "Serious", 2: "Fatal"}
SOURCE_MISSING_TOKEN = "__SOURCE_MISSING__"
SEEDS = (1103, 2207, 3301, 4409, 5501)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_table() -> tuple[pd.DataFrame, dict[str, object]]:
    schema = load_json(SCHEMA_FILE)
    categorical = list(schema["categorical_feature_columns"])
    table = pd.read_csv(
        DATA_FILE,
        dtype={
            "meta_crash_id": "string",
            **{feature: "string" for feature in categorical},
        },
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )
    return table, schema


def test_frozen_schema_and_hashes() -> None:
    schema = load_json(SCHEMA_FILE)
    cleaning = load_json(CLEANING_FILE)
    assert schema["status"] == "QC_PASSED_AND_FROZEN_BEFORE_MODELING"
    assert cleaning["status"] == "FROZEN_BEFORE_MODELING"
    assert schema["row_count"] == 43_121
    assert schema["feature_count"] == 15
    assert len(schema["categorical_feature_columns"]) == 12
    assert len(schema["numeric_feature_columns"]) == 3
    assert hash_file(DATA_FILE) == schema["sha256"]
    assert hash_file(CLEANING_FILE) == schema["parent"]["cleaning_rules_sha256"]


def test_modeling_table_contract() -> None:
    table, schema = load_table()
    expected_columns = (
        list(schema["metadata_columns"])
        + [schema["target_column"]]
        + list(schema["feature_columns"])
    )
    assert table.columns.tolist() == expected_columns
    assert len(table) == 43_121
    assert table["meta_crash_id"].nunique() == len(table)
    assert table["meta_crash_id"].str.startswith("CAS-").all()
    assert set(table["meta_crash_year"].astype(int)) == {2022, 2023, 2024, 2025}
    assert set(table["target_severity"].astype(int)) == {0, 1, 2}
    observed = table["target_severity"].value_counts().sort_index().to_dict()
    assert observed == {0: 33_635, 1: 8_339, 2: 1_147}
    assert not set(schema["metadata_columns"]).intersection(
        schema["feature_columns"]
    )
    assert schema["target_column"] not in schema["feature_columns"]


def test_literal_categories_and_missingness() -> None:
    table, schema = load_table()
    assert int(table["feature_street_light"].eq("None").sum()) == 14_219
    assert int(table["feature_street_light"].eq("Null").sum()) == 670
    assert int(table["feature_weather_secondary"].eq("None").sum()) == 83
    assert int(table["feature_weather_secondary"].eq("Null").sum()) == 41_122
    assert int(
        table["feature_state_highway"].eq(SOURCE_MISSING_TOKEN).sum()
    ) == 3
    assert int(
        table["feature_number_of_lanes"].eq(SOURCE_MISSING_TOKEN).sum()
    ) == 695
    assert int(table["feature_advisory_speed"].isna().sum()) == 40_801
    assert int(table["feature_speed_limit"].isna().sum()) == 698
    assert int(table["feature_temporary_speed_limit"].isna().sum()) == 41_627
    assert not table[list(schema["categorical_feature_columns"])].isna().any().any()


def test_allowlist_matches_leakage_audit() -> None:
    schema = load_json(SCHEMA_FILE)
    audit = pd.read_csv(LEAKAGE_AUDIT_FILE)
    included = set(
        audit.loc[audit["decision"].eq("INCLUDE_CANDIDATE"), "field_name"]
    )
    source_fields = set(schema["source_to_feature"])
    assert source_fields == included
    assert {"fatalCount", "minorInjuryCount", "seriousInjuryCount"}.isdisjoint(
        source_fields
    )
    assert "region" not in source_fields
    assert "OBJECTID" not in source_fields
    assert "crashYear" not in source_fields


def test_split_protocol_and_identity() -> None:
    table, _schema = load_table()
    assignments = pd.read_csv(
        ASSIGNMENTS_FILE,
        dtype={"meta_crash_id": "string"},
        low_memory=False,
    )
    protocol = load_json(PROTOCOL_FILE)
    training_inputs = load_json(TRAINING_INPUTS_FILE)
    assert protocol["status"] == "FROZEN_BEFORE_ANY_CAS_MODEL_TRAINING"
    assert (
        training_inputs["status"]
        == "TRAINING_INPUTS_AUDITED_AND_FROZEN_BEFORE_MODELING"
    )
    assert hash_file(ASSIGNMENTS_FILE) == protocol["assignments"]["sha256"]
    assert hash_file(ASSIGNMENTS_FILE) == training_inputs["split_assignments"]["sha256"]
    assert len(assignments) == len(table)
    for column in ("meta_crash_id", "meta_crash_year", "target_severity"):
        assert np.array_equal(
            assignments[column].to_numpy(),
            table[column].to_numpy(),
        )


def test_temporal_roles_are_year_locked() -> None:
    assignments = pd.read_csv(
        ASSIGNMENTS_FILE,
        dtype={"meta_crash_id": "string"},
        low_memory=False,
    )
    expected = {2022: "train", 2023: "train", 2024: "validation", 2025: "test"}
    for year, role in expected.items():
        observed = set(
            assignments.loc[
                assignments["meta_crash_year"].eq(year), "temporal_role"
            ]
        )
        assert observed == {role}
    counts = assignments["temporal_role"].value_counts().to_dict()
    assert counts == {"train": 21_934, "validation": 10_645, "test": 10_542}
    test = assignments.loc[assignments["temporal_role"].eq("test")]
    assert int(test["target_severity"].eq(2).sum()) == 259


def test_random_roles_and_test_lock() -> None:
    assignments = pd.read_csv(
        ASSIGNMENTS_FILE,
        dtype={"meta_crash_id": "string"},
        low_memory=False,
    )
    development = assignments["meta_crash_year"].isin([2022, 2023, 2024])
    future = assignments["meta_crash_year"].eq(2025)
    for seed in SEEDS:
        column = f"random_role_seed_{seed}"
        assert set(assignments.loc[development, column]) == {
            "train",
            "validation",
            "test",
        }
        assert set(assignments.loc[future, column]) == {"locked_temporal_test"}
        counts = assignments.loc[development, column].value_counts().to_dict()
        assert counts == {"train": 22_805, "validation": 4_887, "test": 4_887}
        for role in ("train", "validation", "test"):
            targets = set(
                assignments.loc[
                    development & assignments[column].eq(role),
                    "target_severity",
                ].astype(int)
            )
            assert targets == {0, 1, 2}


def test_training_only_class_weights() -> None:
    assignments = pd.read_csv(ASSIGNMENTS_FILE, low_memory=False)
    payload = load_json(TRAINING_INPUTS_FILE)
    weight_sets = payload["weight_sets"]
    checks = [("temporal_role", weight_sets["temporal"])]
    checks.extend(
        (
            f"random_role_seed_{seed}",
            weight_sets["random_reference"][str(seed)],
        )
        for seed in SEEDS
    )
    for role_column, stored in checks:
        training = assignments.loc[
            assignments[role_column].eq("train"), "target_severity"
        ]
        counts = training.value_counts().sort_index()
        n_rows = len(training)
        for code in TARGET_LABELS:
            expected = n_rows / (3 * int(counts.loc[code]))
            observed = float(stored["class_weights"][str(code)])
            assert np.isclose(observed, expected, rtol=0, atol=1e-15)
            assert int(stored["class_counts"][str(code)]) == int(counts.loc[code])


def main() -> None:
    checks = [
        test_frozen_schema_and_hashes,
        test_modeling_table_contract,
        test_literal_categories_and_missingness,
        test_allowlist_matches_leakage_audit,
        test_split_protocol_and_identity,
        test_temporal_roles_are_year_locked,
        test_random_roles_and_test_lock,
        test_training_only_class_weights,
    ]
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")
    print(f"All {len(checks)} CAS modeling-input checks passed.")


if __name__ == "__main__":
    main()

