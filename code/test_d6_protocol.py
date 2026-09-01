"""Independent assertions for the frozen D6 protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROTOCOL_FILE = PROJECT_DIR / "config" / "d6_analysis_protocol.json"
ASSIGNMENTS_FILE = PROJECT_DIR / "data" / "processed" / "d6_split_assignments.csv.gz"
FIGURE_FILE = PROJECT_DIR / "figures" / "figure2_dataset_overview.png"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    assert protocol["status"] == "FROZEN_BEFORE_MODEL_TRAINING"
    assert protocol["frozen_population"]["row_count"] == 743_646
    assert len(protocol["frozen_schema"]["feature_columns"]) == 17
    assert protocol["frozen_metrics"]["bootstrap_iterations"] == 2_000
    assert protocol["frozen_metrics"]["bootstrap_seed"] == 20_260_828
    assert not set(protocol["frozen_schema"]["feature_columns"]).intersection(
        protocol["frozen_schema"]["metadata_columns"]
    )
    assert hash_file(ASSIGNMENTS_FILE) == protocol["split_assignment_artifact"]["sha256"]

    assignments = pd.read_csv(
        ASSIGNMENTS_FILE,
        dtype={"meta_collision_index": "string"},
        low_memory=False,
    )
    assert len(assignments) == 743_646
    assert assignments["meta_collision_index"].is_unique
    assert not assignments["meta_collision_index"].isna().any()

    expected_temporal = {
        "train": ({2018, 2019, 2020, 2021, 2022}, 538_461),
        "validation": ({2023}, 104_258),
        "test": ({2024}, 100_927),
    }
    for role, (years, count) in expected_temporal.items():
        subset = assignments.loc[assignments["temporal_role"].eq(role)]
        assert len(subset) == count
        assert set(subset["meta_collision_year"].astype(int)) == years
    temporal_test = assignments.loc[assignments["temporal_role"].eq("test")]
    assert int(temporal_test["target_severity"].eq(2).sum()) == 1_502

    seeds = protocol["random_reference_protocol"]["seeds"]
    expected_random_counts = {"train": 449_903, "validation": 96_408, "test": 96_408}
    development = assignments["meta_collision_year"].le(2023)
    locked_2024 = assignments["meta_collision_year"].eq(2024)
    role_columns: list[str] = []
    for seed in seeds:
        column = f"random_role_seed_{seed}"
        role_columns.append(column)
        assert assignments.loc[locked_2024, column].eq("locked_temporal_test").all()
        assert not assignments.loc[development, column].eq("locked_temporal_test").any()
        for role, count in expected_random_counts.items():
            subset = assignments.loc[development & assignments[column].eq(role)]
            assert len(subset) == count
            assert int(subset["target_severity"].eq(2).sum()) > 50

    for column in role_columns[1:]:
        assert not assignments[column].equals(assignments[role_columns[0]])

    with Image.open(FIGURE_FILE) as image:
        width, height = image.size
        dpi = image.info.get("dpi", (0, 0))
    assert width >= 1800 and height >= 900
    assert min(dpi) >= 295

    print("D6 assignment hash:", hash_file(ASSIGNMENTS_FILE))
    print("D6 figure pixels:", f"{width}x{height}")
    print("D6 figure DPI:", dpi)
    print("INDEPENDENT_D6_ASSERTIONS=PASS")


if __name__ == "__main__":
    main()
