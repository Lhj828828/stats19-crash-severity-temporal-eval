"""Independent checks for the D7 training-input freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROTOCOL_FILE = PROJECT_DIR / "config" / "d6_analysis_protocol.json"
ASSIGNMENTS_FILE = PROJECT_DIR / "data" / "processed" / "d6_split_assignments.csv.gz"
D7_CONFIG_FILE = PROJECT_DIR / "config" / "d7_training_inputs.json"
WEIGHT_TABLE_FILE = PROJECT_DIR / "logs" / "d7_class_weights.csv"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_weights(target: pd.Series) -> dict[int, float]:
    counts = target.astype(int).value_counts().sort_index()
    assert counts.index.tolist() == [0, 1, 2]
    return {code: len(target) / (3 * int(counts.loc[code])) for code in (0, 1, 2)}


def main() -> None:
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    config = json.loads(D7_CONFIG_FILE.read_text(encoding="utf-8"))
    assert config["version"] == "D7_V1"
    assert config["status"] == "TRAINING_INPUTS_AUDITED_AND_FROZEN"
    assert config["upstream"]["D6_assignments_sha256"] == hash_file(ASSIGNMENTS_FILE)
    assert config["upstream"]["D6_assignments_sha256"] == protocol["split_assignment_artifact"]["sha256"]
    assert config["class_weight_rule"]["source_rows"].startswith("training role only")
    assert config["feature_contract"]["feature_count"] == 17
    assert not set(config["feature_contract"]["feature_columns"]).intersection(
        config["feature_contract"]["metadata_excluded_from_models"]
    )

    assignments = pd.read_csv(
        ASSIGNMENTS_FILE,
        dtype={"meta_collision_index": "string"},
        low_memory=False,
    )
    assert len(assignments) == 743_646
    assert assignments["meta_collision_index"].is_unique

    temporal_roles = assignments["temporal_role"].value_counts().to_dict()
    assert temporal_roles == {"train": 538_461, "validation": 104_258, "test": 100_927}
    temporal_train = assignments.loc[assignments["temporal_role"].eq("train")]
    calculated = expected_weights(temporal_train["target_severity"])
    saved = config["weight_sets"]["temporal"]["class_weights"]
    for code in (0, 1, 2):
        assert np.isclose(calculated[code], float(saved[str(code)]), rtol=0, atol=1e-14)

    year_2024 = assignments["meta_collision_year"].eq(2024)
    development = assignments["meta_collision_year"].le(2023)
    for seed in (1103, 2207, 3301, 4409, 5501):
        role_column = f"random_role_seed_{seed}"
        assert assignments.loc[year_2024, role_column].eq("locked_temporal_test").all()
        observed = assignments.loc[development, role_column].value_counts().to_dict()
        assert observed == {"train": 449_903, "validation": 96_408, "test": 96_408}
        training = assignments.loc[assignments[role_column].eq("train")]
        calculated = expected_weights(training["target_severity"])
        saved = config["weight_sets"]["random_reference"][str(seed)]["class_weights"]
        for code in (0, 1, 2):
            assert np.isclose(calculated[code], float(saved[str(code)]), rtol=0, atol=1e-14)

    weight_table = pd.read_csv(WEIGHT_TABLE_FILE)
    assert len(weight_table) == 18
    assert set(weight_table["class_code"]) == {0, 1, 2}
    assert not weight_table["training_count"].isna().any()
    assert not weight_table["balanced_weight"].isna().any()

    print("D7 assignment hash:", hash_file(ASSIGNMENTS_FILE))
    print("D7 class-weight rows:", len(weight_table))
    print("INDEPENDENT_D7_ASSERTIONS=PASS")


if __name__ == "__main__":
    main()
