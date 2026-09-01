"""Independent lightweight checks for the D15 reproducibility contract."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from d15_reproducibility import (
    CONFIG_FILE,
    D14_SUMMARY_FILE,
    MANIFEST_FILE,
    REPORT_FILE,
    VERSION,
    check_data_contract,
    check_dependency_contract,
    check_model_smoke,
    check_portability_contract,
    check_prediction_contract,
    check_protocol_contract,
)


def main() -> None:
    assert json.loads(CONFIG_FILE.read_text(encoding="utf-8"))["version"] == VERSION
    assert check_protocol_contract()
    assert check_data_contract()
    assert check_prediction_contract()
    assert check_model_smoke()
    assert check_dependency_contract()
    assert check_portability_contract()

    assert MANIFEST_FILE.is_file()
    manifest = pd.read_csv(MANIFEST_FILE)
    assert len(manifest) > 100
    assert manifest["relative_path"].is_unique
    assert manifest["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert REPORT_FILE.is_file()
    summary = json.loads(D14_SUMMARY_FILE.read_text(encoding="utf-8"))
    assert summary["D15_ready"] is True
    print("INDEPENDENT_D15_ASSERTIONS=PASS")
    print(f"D15_MANIFEST_ROWS={len(manifest)}")


if __name__ == "__main__":
    main()
