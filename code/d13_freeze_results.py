"""Freeze D13 result tables and evidence-constrained manuscript positioning."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_DIR / "code"
CONFIG_DIR = PROJECT_DIR / "config"
RESULT_DIR = PROJECT_DIR / "results" / "d13"
LOG_DIR = PROJECT_DIR / "logs"

D11_METRICS_FILE = PROJECT_DIR / "results" / "d11" / "d11_test_metrics.csv"
D11_MANIFEST_FILE = LOG_DIR / "d11_artifact_manifest.csv"
D12_PROTOCOL_FILE = CONFIG_DIR / "d12_bootstrap_protocol.json"
D12_MANIFEST_FILE = LOG_DIR / "d12_artifact_manifest.csv"
D12_INTERVAL_FILE = PROJECT_DIR / "results" / "d12" / "d12_model_metric_intervals.csv"
D12_PAIRWISE_FILE = PROJECT_DIR / "results" / "d12" / "d12_pairwise_model_differences.csv"
D12_OPTIMISM_FILE = PROJECT_DIR / "results" / "d12" / "d12_random_optimism_gaps.csv"
D12_OPTIMISM_SUMMARY_FILE = PROJECT_DIR / "results" / "d12" / "d12_random_optimism_summary.csv"
D12_GAIN_FILE = PROJECT_DIR / "results" / "d12" / "d12_model_gain_summary.csv"

PROTOCOL_FILE = CONFIG_DIR / "d13_positioning_protocol.json"
MAIN_TABLE_FILE = RESULT_DIR / "d13_main_performance_table.csv"
H1_TABLE_FILE = RESULT_DIR / "d13_h1_primary_model_gaps.csv"
H2_TABLE_FILE = RESULT_DIR / "d13_h2_temporal_comparison.csv"
DECISION_FILE = LOG_DIR / "d13_positioning_decision.json"
CHECKPOINT_FILE = LOG_DIR / "d13_checkpoint.md"
SUMMARY_FILE = LOG_DIR / "d13_run_summary.json"
MANIFEST_FILE = LOG_DIR / "d13_artifact_manifest.csv"
TEST_SOURCE_FILE = CODE_DIR / "test_d13_checkpoint.py"

VERSION = "D13_V1"
CREATED_LOCAL = "2026-08-31"
MODEL_IDS = (
    "dummy_most_frequent",
    "logistic_unweighted",
    "logistic_weighted",
    "ordered_logit_unweighted",
    "lightgbm_weighted",
)
PRIMARY_MODELS = ("logistic_weighted", "lightgbm_weighted")
METRICS = (
    "macro_f1",
    "qwk",
    "ordinal_mae",
    "accuracy",
    "fatal_recall",
    "serious_or_fatal_recall",
    "mean_asymmetric_cost",
)
HIGHER_IS_BETTER = {
    "macro_f1",
    "qwk",
    "accuracy",
    "fatal_recall",
    "serious_or_fatal_recall",
}
LOWER_IS_BETTER = {"ordinal_mae", "mean_asymmetric_cost"}
CANDIDATE_MODEL = "lightgbm_weighted"
REFERENCE_MODEL = "logistic_weighted"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_DIR.resolve()).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError(f"No rows generated for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", lineterminator="\n")
    temporary.replace(path)


def direction(metric: str) -> str:
    if metric in HIGHER_IS_BETTER:
        return "higher_is_better"
    if metric in LOWER_IS_BETTER:
        return "lower_is_better"
    raise KeyError(metric)


def verify_manifest(path: Path, expected_rows: int) -> dict[str, Any]:
    manifest = pd.read_csv(path, dtype={"relative_path": str, "sha256": str})
    if len(manifest) != expected_rows:
        raise AssertionError(f"Unexpected manifest row count: {path}")
    total_bytes = 0
    for row in manifest.itertuples(index=False):
        artifact = PROJECT_DIR / row.relative_path
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        if artifact.stat().st_size != int(row.bytes):
            raise AssertionError(f"Artifact size changed: {row.relative_path}")
        if hash_file(artifact) != row.sha256:
            raise AssertionError(f"Artifact hash changed: {row.relative_path}")
        total_bytes += int(row.bytes)
    return {
        "manifest": relative(path),
        "manifest_sha256": hash_file(path),
        "rows": int(len(manifest)),
        "artifact_bytes": int(total_bytes),
    }


def build_protocol() -> dict[str, Any]:
    d11_audit = verify_manifest(D11_MANIFEST_FILE, 64)
    d12_audit = verify_manifest(D12_MANIFEST_FILE, 15)
    parents = [
        D11_METRICS_FILE,
        D11_MANIFEST_FILE,
        D12_PROTOCOL_FILE,
        D12_MANIFEST_FILE,
        D12_INTERVAL_FILE,
        D12_PAIRWISE_FILE,
        D12_OPTIMISM_FILE,
        D12_OPTIMISM_SUMMARY_FILE,
        D12_GAIN_FILE,
        Path(__file__),
        TEST_SOURCE_FILE,
    ]
    return {
        "version": VERSION,
        "status": "FROZEN_POST_D12_POSITIONING_RULES",
        "created_local": CREATED_LOCAL,
        "timing_disclosure": (
            "D13 is intentionally a post-result checkpoint. It translates frozen "
            "D11/D12 estimates into evidence-constrained narrative branches and "
            "performs no model selection."
        ),
        "scientific_lock": {
            "model_fits": 0,
            "new_bootstrap_draws": 0,
            "threshold_changes": 0,
            "hypothesis_rewrites": 0,
            "read_frozen_D11_D12_only": True,
        },
        "parent_artifacts": {relative(path): hash_file(path) for path in parents},
        "manifest_audits": {"D11": d11_audit, "D12": d12_audit},
        "tables": {
            "main_performance": {
                "rows": 35,
                "definition": (
                    "Five models by seven metrics; random results are mean and sample "
                    "SD across five seeds, temporal 2024 values include D12 intervals."
                ),
            },
            "H1_primary_model_gaps": {
                "rows": 14,
                "models": list(PRIMARY_MODELS),
                "positive_gap": "random internal evaluation is more optimistic",
            },
            "H2_temporal_comparison": {
                "rows": 7,
                "candidate": CANDIDATE_MODEL,
                "reference": REFERENCE_MODEL,
            },
        },
        "decision_rules": {
            "H1": {
                "branch": "Use metric-dependent branch when primary metrics conflict.",
                "consistent_random_optimism": (
                    "all five point gaps positive and all five 95% intervals above zero"
                ),
                "consistent_random_pessimism": (
                    "all five point gaps negative and all five 95% intervals below zero"
                ),
                "equivalence": (
                    "No frozen equivalence margin; an interval crossing zero does not "
                    "establish practical equivalence."
                ),
            },
            "H2": {
                "macro_f1_point_threshold": 0.01,
                "macro_f1_interval_rule": "paired 95% interval lower bound exceeds zero",
                "joint_rule": (
                    "Macro-F1 alone is insufficient if QWK, fatal recall or asymmetric "
                    "cost clearly deteriorates. An oriented interval wholly below zero "
                    "is recorded as clear deterioration."
                ),
            },
            "H3": {
                "status": "PENDING_NOT_EVALUATED",
                "reason": "Frozen SHAP ranking results do not yet exist.",
                "prohibition": "Do not infer H3 from predictive performance.",
            },
        },
        "planned_outputs": [
            relative(MAIN_TABLE_FILE),
            relative(H1_TABLE_FILE),
            relative(H2_TABLE_FILE),
            relative(DECISION_FILE),
            relative(CHECKPOINT_FILE),
            relative(SUMMARY_FILE),
        ],
    }


def freeze_protocol() -> None:
    payload = build_protocol()
    if PROTOCOL_FILE.exists():
        if read_json(PROTOCOL_FILE) != payload:
            raise RuntimeError("Existing D13 protocol differs; use a versioned amendment")
        print(f"D13_PROTOCOL_ALREADY_FROZEN={hash_file(PROTOCOL_FILE)}")
        return
    write_json(PROTOCOL_FILE, payload)
    print(f"D13_PROTOCOL_FROZEN={hash_file(PROTOCOL_FILE)}")


def validate_protocol() -> dict[str, Any]:
    if not PROTOCOL_FILE.is_file():
        raise FileNotFoundError("Freeze D13 protocol before running")
    protocol = read_json(PROTOCOL_FILE)
    if protocol["version"] != VERSION:
        raise AssertionError("Unexpected D13 protocol version")
    for name, expected_hash in protocol["parent_artifacts"].items():
        if hash_file(PROJECT_DIR / name) != expected_hash:
            raise AssertionError(f"Frozen D13 parent changed: {name}")
    return protocol


def build_main_table(
    d11: pd.DataFrame,
    intervals: pd.DataFrame,
    optimism_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    temporal = intervals[intervals["group_key"].eq("temporal_test_2024")]
    for model in MODEL_IDS:
        random_internal = d11[
            d11["evaluation_scope"].eq("random_reference_test")
            & d11["model"].eq(model)
        ]
        random_future = d11[
            d11["evaluation_scope"].eq("random_model_on_2024")
            & d11["model"].eq(model)
        ]
        if len(random_internal) != 5 or len(random_future) != 5:
            raise AssertionError(f"Unexpected random rows for {model}")
        for metric in METRICS:
            t = temporal[
                temporal["model"].eq(model) & temporal["metric"].eq(metric)
            ]
            o = optimism_summary[
                optimism_summary["model"].eq(model)
                & optimism_summary["metric"].eq(metric)
            ]
            if len(t) != 1 or len(o) != 1:
                raise AssertionError(f"Missing D12 summary for {model} {metric}")
            internal_values = random_internal[metric].to_numpy(dtype=float)
            future_values = random_future[metric].to_numpy(dtype=float)
            t_row, o_row = t.iloc[0], o.iloc[0]
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "direction": direction(metric),
                    "random_internal_mean_5_seeds": float(internal_values.mean()),
                    "random_internal_sd_5_seeds": float(internal_values.std(ddof=1)),
                    "same_random_models_2024_mean_5_seeds": float(future_values.mean()),
                    "same_random_models_2024_sd_5_seeds": float(future_values.std(ddof=1)),
                    "mean_oriented_optimism_gap_5_seeds": float(o_row["mean_optimism_gap"]),
                    "optimism_gap_sd_5_seeds": float(o_row["sd_across_seeds"]),
                    "optimism_gap_minimum": float(o_row["minimum_seed_gap"]),
                    "optimism_gap_maximum": float(o_row["maximum_seed_gap"]),
                    "strict_temporal_2024_point": float(t_row["point_estimate"]),
                    "strict_temporal_2024_ci_lower": float(t_row["ci_lower"]),
                    "strict_temporal_2024_ci_upper": float(t_row["ci_upper"]),
                    "random_seed_summary_status": "descriptive_overlapping_partitions",
                    "temporal_interval_status": "fixed_model_test_sample_uncertainty",
                }
            )
    return pd.DataFrame(rows)


def interval_evidence_class(group: pd.DataFrame) -> str:
    points = group["optimism_gap"].to_numpy(dtype=float)
    lower = group["optimism_gap_ci_lower"].to_numpy(dtype=float)
    upper = group["optimism_gap_ci_upper"].to_numpy(dtype=float)
    if np.all(points > 0) and np.all(lower > 0):
        return "consistent_random_optimism"
    if np.all(points < 0) and np.all(upper < 0):
        return "consistent_random_pessimism"
    return "mixed_or_uncertain"


def build_h1_table(
    optimism: pd.DataFrame,
    optimism_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in PRIMARY_MODELS:
        for metric in METRICS:
            seed_rows = optimism[
                optimism["model"].eq(model) & optimism["metric"].eq(metric)
            ].sort_values("seed")
            summary = optimism_summary[
                optimism_summary["model"].eq(model)
                & optimism_summary["metric"].eq(metric)
            ]
            if len(seed_rows) != 5 or len(summary) != 1:
                raise AssertionError(f"Unexpected H1 rows for {model} {metric}")
            s = summary.iloc[0]
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "direction": direction(metric),
                    "mean_optimism_gap_5_seeds": float(s["mean_optimism_gap"]),
                    "sd_across_5_seeds": float(s["sd_across_seeds"]),
                    "minimum_seed_gap": float(s["minimum_seed_gap"]),
                    "maximum_seed_gap": float(s["maximum_seed_gap"]),
                    "positive_point_seed_count": int((seed_rows["optimism_gap"] > 0).sum()),
                    "interval_entirely_positive_count": int(
                        (seed_rows["optimism_gap_ci_lower"] > 0).sum()
                    ),
                    "interval_crosses_zero_count": int(
                        (
                            (seed_rows["optimism_gap_ci_lower"] <= 0)
                            & (seed_rows["optimism_gap_ci_upper"] >= 0)
                        ).sum()
                    ),
                    "interval_entirely_negative_count": int(
                        (seed_rows["optimism_gap_ci_upper"] < 0).sum()
                    ),
                    "seed_level_evidence_class": interval_evidence_class(seed_rows),
                    "uncertainty_layer_1": "seed_specific_independent_bootstrap_intervals",
                    "uncertainty_layer_2": "mean_sample_SD_and_range_across_five_seeds",
                }
            )
    return pd.DataFrame(rows)


def build_h2_table(pairwise: pd.DataFrame, gain: pd.DataFrame) -> pd.DataFrame:
    temporal = pairwise[pairwise["group_key"].eq("temporal_test_2024")]
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        comparison = temporal[temporal["metric"].eq(metric)]
        gain_row = gain[gain["metric"].eq(metric)]
        if len(comparison) != 1 or len(gain_row) != 1:
            raise AssertionError(f"Unexpected H2 rows for {metric}")
        c, g = comparison.iloc[0], gain_row.iloc[0]
        lower = float(c["oriented_advantage_ci_lower"])
        upper = float(c["oriented_advantage_ci_upper"])
        if lower > 0:
            interval_class = "clear_candidate_advantage"
        elif upper < 0:
            interval_class = "clear_reference_advantage"
        else:
            interval_class = "direction_uncertain"
        rows.append(
            {
                "metric": metric,
                "direction": direction(metric),
                "candidate": CANDIDATE_MODEL,
                "reference": REFERENCE_MODEL,
                "candidate_point": float(c["candidate_point"]),
                "reference_point": float(c["reference_point"]),
                "raw_delta_candidate_minus_reference": float(
                    c["raw_delta_candidate_minus_reference"]
                ),
                "raw_delta_ci_lower": float(c["raw_delta_ci_lower"]),
                "raw_delta_ci_upper": float(c["raw_delta_ci_upper"]),
                "oriented_advantage": float(c["oriented_advantage"]),
                "oriented_advantage_ci_lower": lower,
                "oriented_advantage_ci_upper": upper,
                "temporal_interval_evidence_class": interval_class,
                "random_internal_mean_oriented_advantage": float(
                    g["random_internal_mean_oriented_advantage"]
                ),
                "random_internal_sd_across_seeds": float(
                    g["random_internal_sd_across_seeds"]
                ),
                "temporal_minus_random_mean_advantage": float(
                    g["temporal_minus_random_mean_advantage"]
                ),
                "cross_protocol_change_status": "descriptive_not_causal",
                "macro_f1_point_threshold_0_01_pass": (
                    bool(c["raw_delta_candidate_minus_reference"] >= 0.01)
                    if metric == "macro_f1"
                    else ""
                ),
                "macro_f1_ci_lower_above_zero_pass": (
                    bool(c["raw_delta_ci_lower"] > 0)
                    if metric == "macro_f1"
                    else ""
                ),
            }
        )
    return pd.DataFrame(rows)


def evidence_record(h1: pd.DataFrame, h2: pd.DataFrame) -> dict[str, Any]:
    lightgbm_h1 = h1[h1["model"].eq(CANDIDATE_MODEL)].set_index("metric")
    logistic_h1 = h1[h1["model"].eq(REFERENCE_MODEL)].set_index("metric")
    indexed = h2.set_index("metric")
    macro = indexed.loc["macro_f1"]
    qwk = indexed.loc["qwk"]
    fatal = indexed.loc["fatal_recall"]
    cost = indexed.loc["mean_asymmetric_cost"]
    macro_pass = bool(
        macro["raw_delta_candidate_minus_reference"] >= 0.01
        and macro["raw_delta_ci_lower"] > 0
    )
    fatal_worse = (
        fatal["temporal_interval_evidence_class"] == "clear_reference_advantage"
    )
    joint_pass = bool(
        macro_pass
        and qwk["temporal_interval_evidence_class"] != "clear_reference_advantage"
        and not fatal_worse
        and cost["temporal_interval_evidence_class"] != "clear_reference_advantage"
    )
    if lightgbm_h1.loc["qwk", "seed_level_evidence_class"] != "consistent_random_optimism":
        raise AssertionError("Expected QWK optimism pattern is absent")
    if lightgbm_h1.loc["fatal_recall", "seed_level_evidence_class"] != "consistent_random_optimism":
        raise AssertionError("Expected fatal-recall optimism pattern is absent")
    if lightgbm_h1.loc["macro_f1", "seed_level_evidence_class"] == "consistent_random_optimism":
        raise AssertionError("Macro-F1 unexpectedly supports broad optimism")
    if joint_pass:
        raise AssertionError("Joint H2 rule unexpectedly passed")
    selected_h1 = (
        "macro_f1",
        "qwk",
        "fatal_recall",
        "serious_or_fatal_recall",
        "mean_asymmetric_cost",
    )
    return {
        "version": VERSION,
        "status": "D13_PERFORMANCE_CHECKPOINT_COMPLETE_H3_PENDING",
        "H1": {
            "broad_random_split_optimism_claim": "NOT_SUPPORTED",
            "narrative_branch": "METRIC_DEPENDENT_MIXED_GENERALIZATION_GAP",
            "lightgbm": {
                metric: {
                    "mean_optimism_gap": float(
                        lightgbm_h1.loc[metric, "mean_optimism_gap_5_seeds"]
                    ),
                    "sd_across_seeds": float(
                        lightgbm_h1.loc[metric, "sd_across_5_seeds"]
                    ),
                    "range": [
                        float(lightgbm_h1.loc[metric, "minimum_seed_gap"]),
                        float(lightgbm_h1.loc[metric, "maximum_seed_gap"]),
                    ],
                    "seed_level_evidence_class": str(
                        lightgbm_h1.loc[metric, "seed_level_evidence_class"]
                    ),
                }
                for metric in selected_h1
            },
            "weighted_logistic_macro_f1_evidence_class": str(
                logistic_h1.loc["macro_f1", "seed_level_evidence_class"]
            ),
            "allowed_claim": (
                "Random internal evaluation was consistently optimistic for QWK and "
                "severe-class recall, but not for Macro-F1 or all error metrics."
            ),
            "prohibited_claims": [
                "Random splitting uniformly overestimates future performance.",
                "An interval crossing zero proves equivalence.",
                "The descriptive gap is the causal effect of splitting.",
            ],
        },
        "H2": {
            "macro_f1_practical_and_interval_rule": "PASS" if macro_pass else "FAIL",
            "joint_incremental_value_rule": "PASS" if joint_pass else "FAIL",
            "uniform_lightgbm_superiority_claim": "NOT_SUPPORTED",
            "macro_f1": {
                "delta": float(macro["raw_delta_candidate_minus_reference"]),
                "ci": [float(macro["raw_delta_ci_lower"]), float(macro["raw_delta_ci_upper"])],
                "random_internal_mean_advantage": float(
                    macro["random_internal_mean_oriented_advantage"]
                ),
                "temporal_advantage": float(macro["oriented_advantage"]),
                "descriptive_advantage_change": float(
                    macro["temporal_minus_random_mean_advantage"]
                ),
            },
            "fatal_recall": {
                "delta": float(fatal["raw_delta_candidate_minus_reference"]),
                "ci": [float(fatal["raw_delta_ci_lower"]), float(fatal["raw_delta_ci_upper"])],
                "clear_worsening": bool(fatal_worse),
            },
            "qwk_interval_class": str(qwk["temporal_interval_evidence_class"]),
            "asymmetric_cost_interval_class": str(
                cost["temporal_interval_evidence_class"]
            ),
            "allowed_claim": (
                "LightGBM improved several metrics and passed the Macro-F1 rule, "
                "but substantially reduced fatal recall relative to weighted Logistic."
            ),
            "prohibited_claims": [
                "LightGBM is uniformly or safely superior to weighted Logistic.",
                "Macro-F1 alone satisfies the joint incremental-value rule.",
            ],
        },
        "H3": {
            "status": "PENDING_NOT_EVALUATED",
            "reason": "No frozen SHAP ranking or SHAP bootstrap interval exists yet.",
            "required_next_evidence": (
                "Spearman rank agreement between frozen random-test and temporal-test "
                "global SHAP importance, with its prespecified interval."
            ),
            "claim_allowed_now": False,
        },
        "manuscript_positioning": {
            "primary_empirical_message": (
                "Evaluation-protocol optimism is metric dependent, with degradation "
                "in ordinal agreement and severe-class recall despite stable Macro-F1."
            ),
            "secondary_empirical_message": (
                "Nonlinear-model gains depend on the metric and include a fatal-recall trade-off."
            ),
            "scope": "STATS19 2018-2024 and the frozen collision-time feature set",
        },
    }


def write_checkpoint(decision: dict[str, Any]) -> None:
    h1 = decision["H1"]["lightgbm"]
    h2 = decision["H2"]
    lines = [
        "# D13 positioning checkpoint",
        "",
        f"Status: **{decision['status']}**",
        "",
        "## H1",
        "",
        f"- Broad random-split optimism: **{decision['H1']['broad_random_split_optimism_claim']}**.",
        f"- Branch: **{decision['H1']['narrative_branch']}**.",
        (
            f"- LightGBM Macro-F1 gap: {h1['macro_f1']['mean_optimism_gap']:+.6f} "
            f"(SD {h1['macro_f1']['sd_across_seeds']:.6f})."
        ),
        (
            f"- LightGBM QWK gap: {h1['qwk']['mean_optimism_gap']:+.6f} "
            f"(SD {h1['qwk']['sd_across_seeds']:.6f})."
        ),
        (
            f"- LightGBM fatal-recall gap: {h1['fatal_recall']['mean_optimism_gap']:+.6f} "
            f"(SD {h1['fatal_recall']['sd_across_seeds']:.6f})."
        ),
        "",
        "## H2",
        "",
        f"- Macro-F1 rule: **{h2['macro_f1_practical_and_interval_rule']}**.",
        f"- Joint incremental-value rule: **{h2['joint_incremental_value_rule']}**.",
        (
            f"- Macro-F1 delta: {h2['macro_f1']['delta']:+.6f}, 95% CI "
            f"[{h2['macro_f1']['ci'][0]:+.6f}, {h2['macro_f1']['ci'][1]:+.6f}]."
        ),
        (
            f"- Fatal-recall delta: {h2['fatal_recall']['delta']:+.6f}, 95% CI "
            f"[{h2['fatal_recall']['ci'][0]:+.6f}, {h2['fatal_recall']['ci'][1]:+.6f}]."
        ),
        "",
        "## H3",
        "",
        "- Status: **PENDING_NOT_EVALUATED**.",
        "- No SHAP-stability claim is allowed before the frozen ranking analysis.",
        "",
        "## Positioning",
        "",
        decision["manuscript_positioning"]["primary_empirical_message"],
        "",
        decision["manuscript_positioning"]["secondary_empirical_message"],
        "",
    ]
    CHECKPOINT_FILE.write_text("\n".join(lines), encoding="utf-8")


def build_manifest() -> None:
    paths = [
        Path(__file__),
        TEST_SOURCE_FILE,
        PROTOCOL_FILE,
        MAIN_TABLE_FILE,
        H1_TABLE_FILE,
        H2_TABLE_FILE,
        DECISION_FILE,
        CHECKPOINT_FILE,
        SUMMARY_FILE,
    ]
    rows = [
        {
            "relative_path": relative(path),
            "bytes": int(path.stat().st_size),
            "sha256": hash_file(path),
        }
        for path in sorted(paths, key=relative)
    ]
    write_frame(MANIFEST_FILE, pd.DataFrame(rows))


def run() -> None:
    protocol = validate_protocol()
    verify_manifest(D11_MANIFEST_FILE, 64)
    verify_manifest(D12_MANIFEST_FILE, 15)
    d11 = pd.read_csv(D11_METRICS_FILE)
    intervals = pd.read_csv(D12_INTERVAL_FILE)
    pairwise = pd.read_csv(D12_PAIRWISE_FILE)
    optimism = pd.read_csv(D12_OPTIMISM_FILE)
    optimism_summary = pd.read_csv(D12_OPTIMISM_SUMMARY_FILE)
    gain = pd.read_csv(D12_GAIN_FILE)
    main_table = build_main_table(d11, intervals, optimism_summary)
    h1_table = build_h1_table(optimism, optimism_summary)
    h2_table = build_h2_table(pairwise, gain)
    decision = evidence_record(h1_table, h2_table)
    if len(main_table) != protocol["tables"]["main_performance"]["rows"]:
        raise AssertionError("Unexpected D13 main table size")
    if len(h1_table) != protocol["tables"]["H1_primary_model_gaps"]["rows"]:
        raise AssertionError("Unexpected D13 H1 table size")
    if len(h2_table) != protocol["tables"]["H2_temporal_comparison"]["rows"]:
        raise AssertionError("Unexpected D13 H2 table size")
    write_frame(MAIN_TABLE_FILE, main_table)
    write_frame(H1_TABLE_FILE, h1_table)
    write_frame(H2_TABLE_FILE, h2_table)
    write_json(DECISION_FILE, decision)
    write_checkpoint(decision)
    write_json(
        SUMMARY_FILE,
        {
            "version": VERSION,
            "completed_local": CREATED_LOCAL,
            "status": decision["status"],
            "main_table_rows": int(len(main_table)),
            "H1_table_rows": int(len(h1_table)),
            "H2_table_rows": int(len(h2_table)),
            "model_fits": 0,
            "new_bootstrap_draws": 0,
            "threshold_changes": 0,
            "hypothesis_rewrites": 0,
            "H1_branch": decision["H1"]["narrative_branch"],
            "H2_joint_rule": decision["H2"]["joint_incremental_value_rule"],
            "H3_status": decision["H3"]["status"],
        },
    )
    build_manifest()
    print(f"D13_PERFORMANCE_CHECKPOINT=PASS status={decision['status']}")
    print(f"D13_MANIFEST_SHA256={hash_file(MANIFEST_FILE)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze-protocol", action="store_true")
    actions.add_argument("--run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.freeze_protocol:
        freeze_protocol()
    else:
        run()


if __name__ == "__main__":
    main()
