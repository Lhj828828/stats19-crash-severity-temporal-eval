"""Assert the completed D16 isolated reproduction contract."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPORT_FILE = PROJECT_DIR / "logs" / "d16_reproduction_report.json"


def main() -> None:
    if not REPORT_FILE.is_file():
        raise FileNotFoundError(REPORT_FILE)
    report = json.loads(REPORT_FILE.read_text(encoding="utf-8"))
    assert report["version"] == "D16_V2"
    assert report["status"] in {
        "D16_FULL_REPRODUCTION_PASS",
        "D16_CORE_REPRODUCTION_PASS_WITH_DOCUMENTED_NUMERICAL_DEVIATIONS",
    }
    assert report["thread_cap"] == 4
    assert (
        report["final_comparison_protocol_status"]
        == "FROZEN_AFTER_PIPELINE_BEFORE_FINAL_COMPARISON"
    )
    summary = report["comparison_summary"]
    assert summary["failed"] == 0
    assert summary["passed"] + summary["deviations"] == summary["total"]
    assert report["pipeline_state"]["status"] == "COMPLETE"
    checks = {item["name"]: item for item in report["comparison_checks"]}
    assert checks["source frozen artifacts unchanged"]["status"] == "PASS"
    assert checks[
        "exact:data/processed/stats19_modeling_dataset.csv.gz"
    ]["status"] == "PASS"
    assert checks[
        "exact:data/processed/d6_split_assignments.csv.gz"
    ]["status"] == "PASS"
    if summary["deviations"]:
        assert report["documented_numerical_deviations"]
        assert all(
            "Ordered Logit" in item["name"]
            or item["name"]
            in {
                "png:figures/d11_test_metric_overview.png",
                "png:figures/d12_h1_optimism_gaps.png",
            }
            or item["name"].startswith("D12 bootstrap propagation:")
            or item["name"].startswith("D14 four-thread probabilities:")
            for item in report["documented_numerical_deviations"]
        )
    for raw_path in (
        "results/d12/d12_model_metric_intervals.csv",
        "results/d12/d12_pairwise_model_differences.csv",
        "results/d12/d12_random_optimism_gaps.csv",
    ):
        assert checks[f"D12 primary point estimates:{raw_path}"]["status"] == "PASS"
    assert checks[
        "D12 bootstrap draw contract:results/d12/d12_bootstrap_draws.npz"
    ]["status"] == "PASS"
    assert checks[
        "D14 four-thread identity and labels:"
        "results/d14/exclude2020/predictions/lightgbm_weighted__test.csv.gz"
    ]["status"] == "PASS"
    assert len(report["execution_protocol_sha256"]) == 64
    assert len(report["final_comparison_protocol_sha256"]) == 64
    workspace_value = report["workspace"]
    environment_value = report["clean_environment_python"]
    if workspace_value == "AUTHOR_SIDE_REPRODUCTION_WORKSPACE_NOT_DISTRIBUTED":
        assert report["source_project"] == "PUBLIC_SOFTWARE_PROJECT"
        assert environment_value.startswith(workspace_value)
    else:
        environment = Path(environment_value).resolve()
        workspace = Path(workspace_value).resolve()
        assert workspace in environment.parents
    print("D16_REPRODUCTION_TEST=PASS")


if __name__ == "__main__":
    main()
