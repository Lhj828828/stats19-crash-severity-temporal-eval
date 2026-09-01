"""Independent verification for D14 closeout artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from d14_closeout import (
    BOUNDARY_FILE,
    CHECKPOINT_FILE,
    DEVIATION_FILE,
    ERROR_STRUCTURE_FILE,
    EXPECTED_TEST_ROWS,
    ILLUSTRATIVE_ERROR_FILE,
    MANIFEST_FILE,
    MODEL_IDS,
    PAIRED_ERROR_FILE,
    PROJECT_DIR,
    PROTOCOL_FILE,
    SUMMARY_FILE,
    VERSION,
    build_protocol,
)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    assert protocol["version"] == VERSION
    assert protocol == build_protocol()
    assert protocol["regional_holdout_deviation"]["concrete_police_force_frozen_before_modeling"] is False
    assert protocol["error_audit"]["cohort"].startswith("all 100,927")

    errors = pd.read_csv(ERROR_STRUCTURE_FILE)
    paired = pd.read_csv(PAIRED_ERROR_FILE)
    examples = pd.read_csv(ILLUSTRATIVE_ERROR_FILE, dtype={"meta_collision_index": "string"})
    assert len(errors) == len(MODEL_IDS) * 6 == 12
    assert errors.duplicated(["model", "true_code", "predicted_code"]).sum() == 0
    assert set(errors["model"]) == set(MODEL_IDS)
    assert set(errors["error_direction"]) == {"underestimate", "overestimate"}
    assert len(paired) == 4 * 5 == 20
    for scope, group in paired.groupby("true_class_scope"):
        assert group["count"].sum() == int(group["scope_rows"].iloc[0])
        assert np.isclose(group["share_within_scope"].sum(), 1.0, rtol=0, atol=1e-12)
        if scope == "All":
            assert int(group["scope_rows"].iloc[0]) == EXPECTED_TEST_ROWS
    assert len(examples) == 2 * 4 * 3 == 24
    assert set(examples["model"]) == set(MODEL_IDS)
    assert examples.groupby(["model", "error_type"]).size().eq(3).all()
    assert examples["appendix_interpretation"].str.contains("not representative").all()
    for _, group in examples.groupby(["model", "error_type"], sort=False):
        confidence = group["assigned_class_probability"].to_numpy(dtype=float)
        assert np.all(confidence[:-1] >= confidence[1:])

    deviation = json.loads(DEVIATION_FILE.read_text(encoding="utf-8"))
    assert deviation["status"] == "ONE_MATERIAL_PLANNED_ANALYSIS_NOT_EXECUTED"
    item = deviation["deviations"][0]
    assert item["analysis"] == "prespecified regional police-force holdout"
    assert item["executed"] is False
    assert item["evidence"]["D2_police_force_d2_role"] == "not_yet_decided"
    assert item["evidence"]["D2_police_force_final_D3_decision"] == "PENDING"

    boundary = BOUNDARY_FILE.read_text(encoding="utf-8")
    for phrase in (
        "do not estimate whether a collision will occur",
        "not causal effects",
        "no empirical spatial-generalization claim",
        "causal pandemic effect",
        "not external preregistration claims",
    ):
        assert phrase in boundary
    summary = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
    assert summary["status"] == "D14_COMPLETE_WITH_REGIONAL_HOLDOUT_DEVIATION"
    assert summary["H2"]["uniform_lightgbm_superiority_claim"] == "NOT_SUPPORTED"
    assert summary["H3"]["causal_claim_allowed"] is False
    assert summary["protocol_deviation"]["regional_holdout_executed"] is False
    assert summary["D15_ready"] is True
    assert CHECKPOINT_FILE.is_file()

    manifest = pd.read_csv(MANIFEST_FILE)
    assert len(manifest) == 10
    for row in manifest.itertuples(index=False):
        path = PROJECT_DIR / row.relative_path
        assert path.stat().st_size == int(row.bytes)
        assert hash_file(path) == row.sha256
    print("INDEPENDENT_D14_CLOSEOUT_ASSERTIONS=PASS")
    print("ERROR_STRUCTURE_ROWS=12 PAIRED_OUTCOME_ROWS=20 ILLUSTRATIVE_ROWS=24")
    print("D14_STATUS=D14_COMPLETE_WITH_REGIONAL_HOLDOUT_DEVIATION")


if __name__ == "__main__":
    main()
