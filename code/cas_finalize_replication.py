"""Create the frozen CAS-to-STATS19 directional reporting summary."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from cas_modeling_common import (
    PROJECT_DIR,
    build_artifact_manifest,
    hash_file,
    relative,
    write_csv,
    write_json_atomic,
)


VERSION = "CAS_REPLICATION_CLOSEOUT_V1"
PROTOCOL_FILE = (
    PROJECT_DIR
    / "config"
    / "cas"
    / "cas_cross_dataset_reporting_protocol.json"
)
RESULT_FILE = (
    PROJECT_DIR
    / "results"
    / "cas_post_analysis"
    / "cross_dataset_directional_comparison.csv"
)
CHECKPOINT_FILE = (
    PROJECT_DIR / "logs" / "cas" / "cas_replication_checkpoint.md"
)
MANIFEST_FILE = (
    PROJECT_DIR
    / "logs"
    / "cas"
    / "cas_replication_artifact_manifest.csv"
)
COMPLETE_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_replication_complete.json"
)
TEST_FILE = PROJECT_DIR / "tests" / "test_cas_closeout.py"


def resolve_protocol_path(path: str) -> Path:
    return (PROJECT_DIR / path).resolve()


def load_protocol() -> dict[str, object]:
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    if protocol.get("version") != "CAS_CROSS_DATASET_REPORTING_V1":
        raise ValueError("Unexpected CAS cross-dataset protocol")
    if protocol.get("status") != "POST_RESULT_DETERMINISTIC_REPORTING_CONTRACT":
        raise ValueError("CAS cross-dataset reporting contract is not frozen")
    for scope in ("cas_upstream", "stats19_upstream"):
        for name, item in protocol[scope].items():
            path = resolve_protocol_path(item["path"])
            if not path.exists():
                raise FileNotFoundError(f"Missing {scope} file: {name}")
            if hash_file(path) != item["sha256"]:
                raise ValueError(f"Changed {scope} file: {name}")
    if protocol.get("model_changes_allowed") is not False:
        raise ValueError("Cross-dataset contract permits model changes")
    if protocol.get("new_inferential_test_allowed") is not False:
        raise ValueError("Cross-dataset contract permits a new inferential test")
    return protocol


def cas_file(protocol: dict[str, object], name: str) -> Path:
    return resolve_protocol_path(protocol["cas_upstream"][name]["path"])


def stats19_file(protocol: dict[str, object], name: str) -> Path:
    return resolve_protocol_path(protocol["stats19_upstream"][name]["path"])


def h1_row(
    *,
    component: str,
    stats19: dict[str, object],
    cas: pd.Series,
    label: str,
    comparison: str,
) -> dict[str, object]:
    return {
        "question": "H1",
        "component": component,
        "contrast": "random_internal_minus_future_same_frozen_model",
        "stats19_point": float(stats19["mean_optimism_gap"]),
        "stats19_ci_lower": "",
        "stats19_ci_upper": "",
        "stats19_range_min": float(stats19["range"][0]),
        "stats19_range_max": float(stats19["range"][1]),
        "cas_point": float(cas["point_gap_mean"]),
        "cas_ci_lower": "",
        "cas_ci_upper": "",
        "cas_range_min": float(cas["point_gap_min"]),
        "cas_range_max": float(cas["point_gap_max"]),
        "replication_label": label,
        "observed_comparison": comparison,
        "interpretation_boundary": (
            "Across-seed point-gap summaries are descriptive; the five "
            "overlapping random splits are not independent studies."
        ),
    }


def main() -> None:
    if any(path.exists() for path in (RESULT_FILE, CHECKPOINT_FILE, COMPLETE_FILE)):
        raise RuntimeError(
            "CAS replication closeout already exists; refusing to overwrite"
        )
    protocol = load_protocol()
    h1 = pd.read_csv(cas_file(protocol, "h1_summary"))
    h2 = pd.read_csv(cas_file(protocol, "h2_bootstrap"))
    cas_h3 = json.loads(cas_file(protocol, "h3_complete").read_text(encoding="utf-8"))
    evaluation = json.loads(
        cas_file(protocol, "evaluation_complete").read_text(encoding="utf-8")
    )
    bootstrap = json.loads(
        cas_file(protocol, "bootstrap_complete").read_text(encoding="utf-8")
    )
    stats19 = json.loads(
        stats19_file(protocol, "final_summary").read_text(encoding="utf-8")
    )

    if evaluation.get("status") != "FROZEN_MODELS_EVALUATED_ONCE":
        raise ValueError("CAS one-time evaluation is incomplete")
    if bootstrap.get("status") != "H1_H2_BOOTSTRAP_COMPLETE":
        raise ValueError("CAS H1/H2 uncertainty is incomplete")
    if cas_h3.get("status") != "H3_SHAP_COMPLETE":
        raise ValueError("CAS H3 is incomplete")
    if stats19.get("status") != "D14_COMPLETE_WITH_REGIONAL_HOLDOUT_DEVIATION":
        raise ValueError("Unexpected STATS19 final result state")

    cas_h1 = h1.loc[h1["model"].eq("lightgbm_weighted")].set_index("metric")
    required_h1 = {"macro_f1", "qwk", "fatal_recall"}
    if not required_h1.issubset(cas_h1.index):
        raise ValueError("CAS H1 summary lacks a required metric")
    temporal_h2 = h2.loc[
        h2["evaluation_group"].eq("temporal_2025")
    ].set_index("metric")
    if not {"macro_f1", "fatal_recall"}.issubset(temporal_h2.index):
        raise ValueError("CAS H2 temporal summary lacks a required metric")

    rows: list[dict[str, object]] = [
        {
            "question": "H1",
            "component": "broad_metric_pattern",
            "contrast": "random_internal_minus_future_same_frozen_model",
            "stats19_point": "",
            "stats19_ci_lower": "",
            "stats19_ci_upper": "",
            "stats19_range_min": "",
            "stats19_range_max": "",
            "cas_point": "",
            "cas_ci_lower": "",
            "cas_ci_upper": "",
            "cas_range_min": "",
            "cas_range_max": "",
            "replication_label": "broad_pattern_agreement",
            "observed_comparison": (
                "Both datasets show metric-dependent gaps and do not support "
                "a claim that random splitting uniformly overestimates every "
                "future-test metric."
            ),
            "interpretation_boundary": (
                "This is an agreement in the broad descriptive pattern, not "
                "equality of gap magnitudes or evidence strength."
            ),
        },
        h1_row(
            component="macro_f1",
            stats19=stats19["H1"]["lightgbm"]["macro_f1"],
            cas=cas_h1.loc["macro_f1"],
            label="different_direction_near_zero_or_mixed",
            comparison=(
                "STATS19 mean gap is slightly negative; CAS mean gap is "
                "slightly positive. Both are small relative to their "
                "across-seed ranges, so no common optimism pattern is shown."
            ),
        ),
        h1_row(
            component="qwk",
            stats19=stats19["H1"]["lightgbm"]["qwk"],
            cas=cas_h1.loc["qwk"],
            label="direction_same_evidence_weaker_in_cas",
            comparison=(
                "Both mean gaps are positive, but CAS spans negative to "
                "positive seed-level point gaps and only two of five seeds "
                "favor internal performance."
            ),
        ),
        h1_row(
            component="fatal_recall",
            stats19=stats19["H1"]["lightgbm"]["fatal_recall"],
            cas=cas_h1.loc["fatal_recall"],
            label="direction_same_evidence_weaker_in_cas",
            comparison=(
                "All five CAS point gaps and the STATS19 mean gap are "
                "positive, but CAS gaps are smaller and most seed-specific "
                "intervals include zero."
            ),
        ),
    ]

    for metric, label, comparison in (
        (
            "macro_f1",
            "not_replicated",
            "STATS19 shows a positive LightGBM advantage with an interval "
            "above zero; CAS has a near-zero negative point difference with "
            "an interval spanning zero.",
        ),
        (
            "fatal_recall",
            "direction_replicated",
            "Both datasets show lower fatal recall for LightGBM than for "
            "weighted Logistic, with both intervals entirely below zero.",
        ),
    ):
        stats_metric = stats19["H2"][metric]
        cas_metric = temporal_h2.loc[metric]
        rows.append(
            {
                "question": "H2",
                "component": metric,
                "contrast": "lightgbm_minus_weighted_logistic_temporal_test",
                "stats19_point": float(stats_metric["delta"]),
                "stats19_ci_lower": float(stats_metric["ci"][0]),
                "stats19_ci_upper": float(stats_metric["ci"][1]),
                "stats19_range_min": "",
                "stats19_range_max": "",
                "cas_point": float(cas_metric["point_difference"]),
                "cas_ci_lower": float(cas_metric["ci_lower"]),
                "cas_ci_upper": float(cas_metric["ci_upper"]),
                "cas_range_min": "",
                "cas_range_max": "",
                "replication_label": label,
                "observed_comparison": comparison,
                "interpretation_boundary": (
                    "Metric-specific directional comparison only; absolute "
                    "performance and effect sizes are not pooled."
                ),
            }
        )

    rows.extend(
        [
            {
                "question": "H3",
                "component": "overall_mean_absolute_shap_rank_rho",
                "contrast": "same_frozen_model_internal_vs_future",
                "stats19_point": float(stats19["H3"]["overall_rho_mean"]),
                "stats19_ci_lower": "",
                "stats19_ci_upper": "",
                "stats19_range_min": float(stats19["H3"]["overall_rho_range"][0]),
                "stats19_range_max": float(stats19["H3"]["overall_rho_range"][1]),
                "cas_point": float(cas_h3["rho_mean"]),
                "cas_ci_lower": "",
                "cas_ci_upper": "",
                "cas_range_min": float(cas_h3["rho_range"][0]),
                "cas_range_max": float(cas_h3["rho_range"][1]),
                "replication_label": "different_observed_pattern",
                "observed_comparison": (
                    "STATS19 shows moderate rank agreement and visible rank "
                    "change; CAS shows near-complete rank agreement for its "
                    "15-feature representation."
                ),
                "interpretation_boundary": (
                    "No cross-dataset test or binary stability threshold is "
                    "used; feature sets and coding systems differ."
                ),
            },
            {
                "question": "workflow",
                "component": "second_source_executability",
                "contrast": "frozen_pipeline_execution",
                "stats19_point": "",
                "stats19_ci_lower": "",
                "stats19_ci_upper": "",
                "stats19_range_min": "",
                "stats19_range_max": "",
                "cas_point": "",
                "cas_ci_lower": "",
                "cas_ci_upper": "",
                "cas_range_min": "",
                "cas_range_max": "",
                "replication_label": "executability_demonstrated",
                "observed_comparison": (
                    "The leakage audit, frozen temporal evaluation, ordered "
                    "metrics, Bootstrap uncertainty and SHAP rank workflow "
                    "executed on CAS without pooling with STATS19."
                ),
                "interpretation_boundary": (
                    "Executability on two sources does not establish "
                    "cross-national or cross-domain generalizability."
                ),
            },
        ]
    )

    allowed = set(protocol["allowed_labels"])
    labels = {row["replication_label"] for row in rows}
    if not labels.issubset(allowed):
        raise AssertionError("A comparison label is outside the reporting contract")
    write_csv(RESULT_FILE, rows)

    checkpoint = [
        "# CAS independent replication checkpoint",
        "",
        "Status: **CAS_REPLICATION_ANALYSIS_COMPLETE**",
        "",
        "## Frozen execution facts",
        "",
        "- CAS contains 43,121 police-reported injury crashes from 2022-2025.",
        "- Dummy, weighted multinomial Logistic and weighted LightGBM were frozen before the authorized one-time 2025 evaluation.",
        "- The one-time evaluation generated 11 groups, 33 metric rows and 33 prediction files without preprocessing refit or post-test selection.",
        "- H1/H2 use 2,000 class-stratified Bootstrap iterations; H3 explains five frozen LightGBM models with 2,000 record-level Bootstrap iterations.",
        "",
        "## Directional comparison",
        "",
        "- H1: broad metric-dependent behavior agrees, but the specific QWK and fatal-recall evidence is weaker in CAS.",
        "- H2 Macro-F1: not replicated. STATS19 delta +0.01390 [0.01087, 0.01696]; CAS delta -0.00072 [-0.00864, 0.00738].",
        "- H2 fatal recall: direction replicated. STATS19 delta -0.10053 [-0.11651, -0.08389]; CAS delta -0.16216 [-0.22394, -0.10425].",
        f"- H3: different observed pattern. STATS19 mean rho {stats19['H3']['overall_rho_mean']:.4f}; CAS mean rho {cas_h3['rho_mean']:.4f}.",
        "- Workflow executability on the second administrative source is demonstrated.",
        "",
        "## Reporting boundary",
        "",
        "- This reporting contract was written after both datasets' results were known; it is not a preregistration.",
        "- Records, absolute metrics, Bootstrap draws and SHAP ranks were not pooled.",
        "- No new inferential test, model change, retuning or threshold adjustment was performed.",
        "- These results do not prove cross-national or cross-domain generalizability.",
        "",
    ]
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text("\n".join(checkpoint), encoding="utf-8")

    internal_artifacts = [
        Path(__file__),
        TEST_FILE,
        PROTOCOL_FILE,
        RESULT_FILE,
        CHECKPOINT_FILE,
        *[
            resolve_protocol_path(item["path"])
            for item in protocol["cas_upstream"].values()
        ],
    ]
    build_artifact_manifest(internal_artifacts, MANIFEST_FILE)
    label_counts = pd.Series(
        [row["replication_label"] for row in rows]
    ).value_counts().sort_index()
    write_json_atomic(
        COMPLETE_FILE,
        {
            "version": VERSION,
            "status": "CAS_REPLICATION_ANALYSIS_COMPLETE",
            "created_local": datetime.now()
            .astimezone()
            .isoformat(timespec="seconds"),
            "protocol": relative(PROTOCOL_FILE),
            "protocol_sha256": hash_file(PROTOCOL_FILE),
            "comparison_rows": len(rows),
            "replication_label_counts": {
                str(label): int(count)
                for label, count in label_counts.items()
            },
            "one_time_2025_evaluation_status": evaluation["status"],
            "post_test_model_selection": False,
            "new_inferential_test": False,
            "records_pooled": False,
            "cross_national_generalizability_claim_allowed": False,
            "artifact_manifest": relative(MANIFEST_FILE),
            "artifact_manifest_sha256": hash_file(MANIFEST_FILE),
            "external_upstream": protocol["stats19_upstream"],
        },
    )
    print(f"CAS_REPLICATION_COMPARISON_ROWS={len(rows)}")
    print("CAS_REPLICATION_STATUS=ANALYSIS_COMPLETE")


if __name__ == "__main__":
    main()
