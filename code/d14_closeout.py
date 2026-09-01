"""Close D14 with aggregate error audits, deviations and evidence boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baseline_modeling import ASYMMETRIC_COST_MATRIX, TARGET_LABELS


PROJECT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_DIR / "code"
CONFIG_DIR = PROJECT_DIR / "config"
RESULT_DIR = PROJECT_DIR / "results" / "d14"
LOG_DIR = PROJECT_DIR / "logs"
PREDICTION_DIR = PROJECT_DIR / "results" / "d11" / "predictions"

DATA_FILE = PROJECT_DIR / "data" / "processed" / "stats19_modeling_dataset.csv.gz"
D2_FIELD_MAPPING_FILE = LOG_DIR / "d2_field_mapping.csv"
D6_PROTOCOL_FILE = CONFIG_DIR / "d6_analysis_protocol.json"
D11_MANIFEST_FILE = LOG_DIR / "d11_artifact_manifest.csv"
D13_DECISION_FILE = LOG_DIR / "d13_positioning_decision.json"
D14_SHAP_PROTOCOL_FILE = CONFIG_DIR / "d14_shap_protocol.json"
D14_SHAP_MANIFEST_FILE = LOG_DIR / "d14_shap_artifact_manifest.csv"
D14_SHAP_SUMMARY_FILE = LOG_DIR / "d14_shap_run_summary.json"
D14_EXCLUDE_PROTOCOL_FILE = CONFIG_DIR / "d14_exclude2020_protocol_v2.json"
D14_EXCLUDE_MANIFEST_FILE = LOG_DIR / "d14_exclude2020_artifact_manifest.csv"
D14_EXCLUDE_SUMMARY_FILE = LOG_DIR / "d14_exclude2020_run_summary.json"

PREDICTION_FILES = {
    "logistic_weighted": PREDICTION_DIR / "temporal__logistic_weighted__test.csv.gz",
    "lightgbm_weighted": PREDICTION_DIR / "temporal__lightgbm_weighted__test.csv.gz",
}

PROTOCOL_FILE = CONFIG_DIR / "d14_closeout_protocol_v2.json"
PREVIOUS_PROTOCOL_FILE = CONFIG_DIR / "d14_closeout_protocol.json"
ERROR_STRUCTURE_FILE = RESULT_DIR / "d14_error_structure.csv"
PAIRED_ERROR_FILE = RESULT_DIR / "d14_paired_error_outcomes.csv"
ILLUSTRATIVE_ERROR_FILE = RESULT_DIR / "d14_illustrative_high_confidence_errors.csv"
DEVIATION_FILE = LOG_DIR / "d14_protocol_deviations.json"
BOUNDARY_FILE = LOG_DIR / "d14_limitations_and_boundaries.md"
CHECKPOINT_FILE = LOG_DIR / "d14_final_checkpoint.md"
SUMMARY_FILE = LOG_DIR / "d14_final_summary.json"
MANIFEST_FILE = LOG_DIR / "d14_artifact_manifest.csv"
TEST_SOURCE_FILE = CODE_DIR / "test_d14_closeout.py"

VERSION = "D14_CLOSEOUT_V2"
CREATED_LOCAL = "2026-08-31"
EXPECTED_TEST_ROWS = 100_927
MODEL_IDS = ("logistic_weighted", "lightgbm_weighted")
CLASS_CODES = (0, 1, 2)
SAFETY_ERROR_TYPES = ((2, 0), (2, 1), (1, 0), (0, 2))
ILLUSTRATIVE_ROWS_PER_MODEL_ERROR_TYPE = 3
PROBABILITY_COLUMNS = {0: "prob_slight", 1: "prob_serious", 2: "prob_fatal"}


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


def upstream_paths() -> list[Path]:
    return [
        DATA_FILE,
        D2_FIELD_MAPPING_FILE,
        D6_PROTOCOL_FILE,
        D11_MANIFEST_FILE,
        D13_DECISION_FILE,
        D14_SHAP_PROTOCOL_FILE,
        D14_SHAP_MANIFEST_FILE,
        D14_SHAP_SUMMARY_FILE,
        D14_EXCLUDE_PROTOCOL_FILE,
        D14_EXCLUDE_MANIFEST_FILE,
        D14_EXCLUDE_SUMMARY_FILE,
        PREVIOUS_PROTOCOL_FILE,
        *PREDICTION_FILES.values(),
        CODE_DIR / "baseline_modeling.py",
        Path(__file__),
        TEST_SOURCE_FILE,
    ]


def build_protocol() -> dict[str, Any]:
    required = upstream_paths()
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing D14 closeout input: " + "; ".join(missing))
    d13 = read_json(D13_DECISION_FILE)
    shap_summary = read_json(D14_SHAP_SUMMARY_FILE)
    exclude_summary = read_json(D14_EXCLUDE_SUMMARY_FILE)
    if d13.get("H3", {}).get("status") != "PENDING_NOT_EVALUATED":
        raise ValueError("D14 closeout expects immutable D13 H3-pending record")
    if shap_summary.get("status") != "D14_SHAP_COMPLETE":
        raise ValueError("D14 SHAP is incomplete")
    if exclude_summary.get("status") != "D14_EXCLUDE2020_SENSITIVITY_COMPLETE":
        raise ValueError("D14 exclusion-2020 sensitivity is incomplete")
    return {
        "version": VERSION,
        "revision": {
            "supersedes_protocol": relative(PREVIOUS_PROTOCOL_FILE),
            "supersedes_protocol_sha256": hash_file(PREVIOUS_PROTOCOL_FILE),
            "reason": (
                "Corrected a verification assertion that searched for an overly narrow wording. "
                "No data, model, sample, metric, extraction or scientific conclusion changed."
            ),
        },
        "status": "FROZEN_BEFORE_D14_ERROR_CASE_EXTRACTION",
        "created_local": CREATED_LOCAL,
        "timing_disclosure": (
            "D14 closeout is a post-result descriptive audit. Its rules were frozen after "
            "D11-D14 model and SHAP results were known, and it performs no model selection, "
            "threshold tuning or inferential hypothesis test."
        ),
        "upstream_sha256": {relative(path): hash_file(path) for path in required},
        "error_audit": {
            "models": list(MODEL_IDS),
            "cohort": "all 100,927 collisions in the frozen 2024 temporal test",
            "aggregate_error_structure": (
                "all six off-diagonal true-to-predicted transitions, with count, within-true "
                "share, confidence summaries, ordinal distance and frozen asymmetric cost"
            ),
            "paired_outcomes": [
                "both_correct",
                "logistic_only_correct",
                "lightgbm_only_correct",
                "both_wrong_same_prediction",
                "both_wrong_different_predictions",
            ],
            "paired_scopes": ["all", "Slight", "Serious", "Fatal"],
            "illustrative_cases": {
                "purpose": "appendix illustration only; not representative and not inferential",
                "safety_error_types": [
                    "Fatal_to_Slight",
                    "Fatal_to_Serious",
                    "Serious_to_Slight",
                    "Slight_to_Fatal",
                ],
                "rows_per_model_error_type": ILLUSTRATIVE_ROWS_PER_MODEL_ERROR_TYPE,
                "selection": (
                    "highest probability assigned to the incorrect predicted class; descending "
                    "confidence, then ascending collision_index; no manual substitution"
                ),
                "feature_values": "frozen processed feature labels/codes; no causal interpretation",
            },
        },
        "regional_holdout_deviation": {
            "planned": True,
            "concrete_police_force_frozen_before_modeling": False,
            "evidence": (
                "logs/d2_field_mapping.csv records police_force as d2_role=not_yet_decided and "
                "final_D3_decision=PENDING; no frozen config names a police-force combination"
            ),
            "decision": "do not execute or describe a region as prespecified after outcomes are known",
            "impact": "no empirical spatial-generalization claim is allowed",
        },
        "outputs": {
            "error_structure": relative(ERROR_STRUCTURE_FILE),
            "paired_errors": relative(PAIRED_ERROR_FILE),
            "illustrative_errors": relative(ILLUSTRATIVE_ERROR_FILE),
            "protocol_deviations": relative(DEVIATION_FILE),
            "limitations": relative(BOUNDARY_FILE),
            "checkpoint": relative(CHECKPOINT_FILE),
            "summary": relative(SUMMARY_FILE),
            "manifest": relative(MANIFEST_FILE),
        },
    }


def freeze_protocol() -> None:
    payload = build_protocol()
    if PROTOCOL_FILE.exists():
        existing = read_json(PROTOCOL_FILE)
        if existing != payload:
            raise RuntimeError("Existing D14 closeout protocol differs; refusing overwrite")
        print("D14 closeout protocol already matches the frozen specification.")
    else:
        write_json(PROTOCOL_FILE, payload)
        print("D14 closeout protocol frozen before error extraction.")
    print("D14_CLOSEOUT_PROTOCOL_SHA256=", hash_file(PROTOCOL_FILE))


def require_protocol() -> dict[str, Any]:
    if not PROTOCOL_FILE.is_file():
        raise FileNotFoundError("Run --freeze before D14 closeout")
    protocol = read_json(PROTOCOL_FILE)
    if protocol.get("version") != VERSION or protocol.get("status") != (
        "FROZEN_BEFORE_D14_ERROR_CASE_EXTRACTION"
    ):
        raise ValueError("Unexpected D14 closeout protocol state")
    if protocol != build_protocol():
        raise ValueError("D14 closeout inputs or implementation changed after freeze")
    return protocol


def load_predictions() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    base_ids: np.ndarray | None = None
    base_target: np.ndarray | None = None
    for model, path in PREDICTION_FILES.items():
        frame = pd.read_csv(path, dtype={"meta_collision_index": "string"})
        if len(frame) != EXPECTED_TEST_ROWS or frame["meta_collision_index"].duplicated().any():
            raise AssertionError(f"Invalid D11 prediction file: {model}")
        if set(frame["meta_collision_year"].astype(int)) != {2024}:
            raise AssertionError("D14 error audit is restricted to 2024")
        if frame["model"].nunique() != 1 or frame["model"].iloc[0] != model:
            raise AssertionError("Prediction file model identity changed")
        ids = frame["meta_collision_index"].astype("string").to_numpy()
        target = frame["target_severity"].to_numpy(dtype=np.int8)
        if base_ids is None:
            base_ids, base_target = ids, target
        elif not np.array_equal(ids, base_ids) or not np.array_equal(target, base_target):
            raise AssertionError("D11 primary prediction rows are not aligned")
        frames[model] = frame
    return frames


def error_structure(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in MODEL_IDS:
        frame = frames[model]
        true_values = frame["target_severity"].to_numpy(dtype=np.int8)
        predicted_values = frame["predicted_severity"].to_numpy(dtype=np.int8)
        for true_code in CLASS_CODES:
            true_count = int(np.sum(true_values == true_code))
            for predicted_code in CLASS_CODES:
                if true_code == predicted_code:
                    continue
                mask = (true_values == true_code) & (predicted_values == predicted_code)
                confidence = frame.loc[mask, PROBABILITY_COLUMNS[predicted_code]].to_numpy(dtype=float)
                count = int(mask.sum())
                if count == 0:
                    mean_confidence = median_confidence = p90_confidence = np.nan
                else:
                    mean_confidence = float(confidence.mean())
                    median_confidence = float(np.median(confidence))
                    p90_confidence = float(np.quantile(confidence, 0.9, method="linear"))
                rows.append(
                    {
                        "model": model,
                        "true_code": true_code,
                        "true_label": TARGET_LABELS[true_code],
                        "predicted_code": predicted_code,
                        "predicted_label": TARGET_LABELS[predicted_code],
                        "error_direction": "underestimate" if predicted_code < true_code else "overestimate",
                        "ordinal_distance": abs(predicted_code - true_code),
                        "asymmetric_cost_per_case": float(ASYMMETRIC_COST_MATRIX[true_code, predicted_code]),
                        "count": count,
                        "share_within_true_class": count / true_count,
                        "mean_assigned_class_probability": mean_confidence,
                        "median_assigned_class_probability": median_confidence,
                        "p90_assigned_class_probability": p90_confidence,
                    }
                )
    return pd.DataFrame(rows)


def paired_outcomes(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    logistic = frames["logistic_weighted"]
    lightgbm = frames["lightgbm_weighted"]
    target = logistic["target_severity"].to_numpy(dtype=np.int8)
    logistic_pred = logistic["predicted_severity"].to_numpy(dtype=np.int8)
    lightgbm_pred = lightgbm["predicted_severity"].to_numpy(dtype=np.int8)
    logistic_correct = logistic_pred == target
    lightgbm_correct = lightgbm_pred == target
    status = np.full(len(target), "both_wrong_different_predictions", dtype=object)
    status[logistic_correct & lightgbm_correct] = "both_correct"
    status[logistic_correct & ~lightgbm_correct] = "logistic_only_correct"
    status[~logistic_correct & lightgbm_correct] = "lightgbm_only_correct"
    status[(~logistic_correct) & (~lightgbm_correct) & (logistic_pred == lightgbm_pred)] = (
        "both_wrong_same_prediction"
    )
    statuses = (
        "both_correct",
        "logistic_only_correct",
        "lightgbm_only_correct",
        "both_wrong_same_prediction",
        "both_wrong_different_predictions",
    )
    rows: list[dict[str, Any]] = []
    for scope_code, scope_label in ((None, "All"), (0, "Slight"), (1, "Serious"), (2, "Fatal")):
        scope_mask = np.ones(len(target), dtype=bool) if scope_code is None else target == scope_code
        denominator = int(scope_mask.sum())
        for outcome in statuses:
            count = int(np.sum(scope_mask & (status == outcome)))
            rows.append(
                {
                    "true_class_scope": scope_label,
                    "paired_outcome": outcome,
                    "count": count,
                    "share_within_scope": count / denominator,
                    "scope_rows": denominator,
                }
            )
    return pd.DataFrame(rows)


def illustrative_errors(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    model_data = pd.read_csv(DATA_FILE, dtype={"meta_collision_index": "string"}, low_memory=False)
    model_data = model_data.loc[model_data["meta_collision_year"].astype(int).eq(2024)].copy()
    if len(model_data) != EXPECTED_TEST_ROWS or model_data["meta_collision_index"].duplicated().any():
        raise AssertionError("Frozen 2024 feature table is invalid")
    metadata = [
        "meta_collision_index",
        "meta_collision_year",
        "meta_date",
        "meta_police_force",
    ]
    feature_columns = [column for column in model_data.columns if column.startswith("feature_")]
    selected_rows: list[pd.DataFrame] = []
    for model in MODEL_IDS:
        frame = frames[model]
        for true_code, predicted_code in SAFETY_ERROR_TYPES:
            subset = frame.loc[
                frame["target_severity"].eq(true_code)
                & frame["predicted_severity"].eq(predicted_code)
            ].copy()
            probability_column = PROBABILITY_COLUMNS[predicted_code]
            subset["assigned_class_probability"] = subset[probability_column]
            subset = subset.sort_values(
                ["assigned_class_probability", "meta_collision_index"],
                ascending=[False, True],
                kind="mergesort",
            ).head(ILLUSTRATIVE_ROWS_PER_MODEL_ERROR_TYPE)
            subset["selection_rule"] = (
                "highest incorrect assigned-class probability; tie by collision_index"
            )
            subset["true_label"] = TARGET_LABELS[true_code]
            subset["predicted_label"] = TARGET_LABELS[predicted_code]
            subset["error_type"] = f"{TARGET_LABELS[true_code]}_to_{TARGET_LABELS[predicted_code]}"
            subset["appendix_interpretation"] = (
                "illustrative high-confidence error; not representative or causal"
            )
            selected_rows.append(
                subset[
                    [
                        "meta_collision_index",
                        "model",
                        "target_severity",
                        "true_label",
                        "predicted_severity",
                        "predicted_label",
                        "error_type",
                        "assigned_class_probability",
                        "prob_slight",
                        "prob_serious",
                        "prob_fatal",
                        "selection_rule",
                        "appendix_interpretation",
                    ]
                ]
            )
    selected = pd.concat(selected_rows, ignore_index=True)
    selected = selected.merge(
        model_data[metadata + feature_columns],
        on="meta_collision_index",
        how="left",
        validate="many_to_one",
    )
    if selected["meta_date"].isna().any():
        raise AssertionError("Illustrative errors failed feature-table merge")
    return selected


def deviation_payload() -> dict[str, Any]:
    mapping = pd.read_csv(D2_FIELD_MAPPING_FILE, dtype=str, keep_default_na=False)
    row = mapping[mapping["field_name"].eq("police_force")]
    if len(row) != 1:
        raise AssertionError("Expected one police_force field-mapping row")
    record = row.iloc[0]
    config_names = sorted(path.name for path in CONFIG_DIR.glob("*.json"))
    return {
        "version": VERSION,
        "status": "ONE_MATERIAL_PLANNED_ANALYSIS_NOT_EXECUTED",
        "created_local": CREATED_LOCAL,
        "deviations": [
            {
                "analysis": "prespecified regional police-force holdout",
                "planned_in_outline": True,
                "executed": False,
                "discovery_time": "D14 closeout after D11-D13 outcomes were known",
                "evidence": {
                    "D2_police_force_d2_role": record["d2_role"],
                    "D2_police_force_final_D3_decision": record["final_D3_decision"],
                    "concrete_region_in_frozen_configs": False,
                    "frozen_config_inventory": config_names,
                },
                "reason": (
                    "The outline required a concrete police-force combination to be frozen before "
                    "modeling, but no such choice was made. Selecting one after seeing outcomes "
                    "would be post-hoc and cannot be represented as prespecified."
                ),
                "corrective_action": (
                    "Omit the regional-holdout result and state that spatial generalization was not "
                    "empirically evaluated. A future post-hoc analysis may be exploratory only."
                ),
                "affected_claims": [
                    "no claim of prespecified regional generalization",
                    "no claim that performance transfers across police forces",
                ],
            }
        ],
    }


def boundary_text(
    *,
    shap_summary: dict[str, Any],
    exclude_summary: dict[str, Any],
) -> str:
    return f"""# D14 limitations and result boundaries

## What the study estimates

- The target is police-recorded collision severity conditional on a reported personal-injury collision. The models do not estimate whether a collision will occur.
- Predictors are restricted to 17 collision-table fields observable at the stated prediction time. Vehicle, casualty, consequence, response, exact-location and target-derived fields were excluded.
- Results apply to the frozen STATS19 2018-2024 cohort and the evaluated protocols. One country and one administrative data source do not establish cross-country or cross-domain generality.

## Evaluation boundaries

- The strict temporal test is one future year, 2024. It does not establish stability over longer horizons or under later coding changes.
- Random-internal and 2024 samples are different records and arise from different training/evaluation conditions. Their metric gaps are deployment-oriented descriptions, not causal effects of a split method.
- Bootstrap intervals condition on fitted models and observed test cohorts. They do not include full model-refitting uncertainty. The five random partitions overlap and are summarized descriptively, not treated as independent studies.
- Fatal cases total 1,502 in 2024. Fatal recall is therefore essential but still reflects one year's severe-class sample and the frozen argmax decision rule.
- The asymmetric error-cost matrix is a transparent evaluation convention, not a validated monetary or social cost function.
- Ordered Logit was trained on a fixed 100,000-row subset for computational feasibility; it is a secondary baseline and is not a like-for-like full-data model comparison.

## Model and explanation boundaries

- LightGBM improved temporal-test Macro-F1 relative to weighted Logistic, but the joint incremental-value rule failed because fatal recall deteriorated. No claim of uniform nonlinear-model superiority is permitted.
- H3 compared each frozen random model on its internal test and on 2024, holding the fitted model fixed. Overall feature-rank Spearman rho averaged {shap_summary['overall_rho_mean']:.6f} (SD {shap_summary['overall_rho_sd']:.6f}; range {shap_summary['overall_rho_range'][0]:.6f}-{shap_summary['overall_rho_range'][1]:.6f}). This quantifies ranking change under the evaluated temporal shift; no preregistered material-instability threshold supports a binary stable/unstable claim.
- TreeSHAP values describe contributions to fitted raw class scores. They are not causal effects. Correlated or substitutable features may exchange rank without a changed data-generating mechanism.
- SHAP used fixed 10,000-row severity-stratified samples and record-level Bootstrap. Its intervals quantify explanation-sample variability for frozen fits, not training or hyperparameter uncertainty.

## Sensitivity and protocol deviations

- Excluding 2020 from temporal training left the main metric trade-off intact: the LightGBM-versus-Logistic Macro-F1 delta was {exclude_summary['sensitivity_H2_macro_f1_delta']:+.6f}, whereas fatal-recall delta was {exclude_summary['sensitivity_H2_fatal_recall_delta']:+.6f}. This does not identify a causal pandemic effect and does not test whether 2020 drives H1 random-versus-future gaps.
- A concrete regional holdout was never frozen before modeling. It was therefore not executed at D14, and no empirical spatial-generalization claim is allowed.
- The SHAP implementation protocol was frozen after predictive outcomes were known but before any SHAP ranking was inspected. The exact exclusion-2020 implementation was frozen after main results were known. These are transparent sequential-analysis disclosures, not external preregistration claims.

## Reporting language

- Use association, contribution, predictive importance, ranking agreement and sensitivity.
- Do not use cause, determinant, mechanism change, universal generalization, practical equivalence, or uniform model superiority for these results.
"""


def assert_outputs_absent() -> None:
    outputs = [
        ERROR_STRUCTURE_FILE,
        PAIRED_ERROR_FILE,
        ILLUSTRATIVE_ERROR_FILE,
        DEVIATION_FILE,
        BOUNDARY_FILE,
        CHECKPOINT_FILE,
        SUMMARY_FILE,
        MANIFEST_FILE,
    ]
    existing = [relative(path) for path in outputs if path.exists()]
    if existing:
        raise RuntimeError("D14 closeout outputs already exist; refusing overwrite: " + "; ".join(existing))


def run_closeout() -> None:
    require_protocol()
    assert_outputs_absent()
    frames = load_predictions()
    error_frame = error_structure(frames)
    paired_frame = paired_outcomes(frames)
    illustrative_frame = illustrative_errors(frames)
    deviations = deviation_payload()
    shap_summary = read_json(D14_SHAP_SUMMARY_FILE)
    exclude_summary = read_json(D14_EXCLUDE_SUMMARY_FILE)
    boundaries = boundary_text(shap_summary=shap_summary, exclude_summary=exclude_summary)

    write_frame(ERROR_STRUCTURE_FILE, error_frame)
    write_frame(PAIRED_ERROR_FILE, paired_frame)
    write_frame(ILLUSTRATIVE_ERROR_FILE, illustrative_frame)
    write_json(DEVIATION_FILE, deviations)
    BOUNDARY_FILE.write_text(boundaries, encoding="utf-8")

    fatal_to_slight = error_frame[
        error_frame["true_code"].eq(2) & error_frame["predicted_code"].eq(0)
    ].set_index("model")
    fatal_paired = paired_frame[paired_frame["true_class_scope"].eq("Fatal")].set_index(
        "paired_outcome"
    )
    summary = {
        "version": VERSION,
        "status": "D14_COMPLETE_WITH_REGIONAL_HOLDOUT_DEVIATION",
        "protocol_sha256": hash_file(PROTOCOL_FILE),
        "H1": read_json(D13_DECISION_FILE)["H1"],
        "H2": read_json(D13_DECISION_FILE)["H2"],
        "H3": {
            "status": shap_summary["H3_status"],
            "overall_rho_mean": shap_summary["overall_rho_mean"],
            "overall_rho_sd": shap_summary["overall_rho_sd"],
            "overall_rho_range": shap_summary["overall_rho_range"],
            "binary_material_instability_claim_allowed": False,
            "causal_claim_allowed": False,
        },
        "exclude2020": {
            "status": exclude_summary["status"],
            "H2_macro_f1_delta": exclude_summary["sensitivity_H2_macro_f1_delta"],
            "H2_fatal_recall_delta": exclude_summary["sensitivity_H2_fatal_recall_delta"],
            "H1_pandemic_driver_claim_allowed": False,
        },
        "error_audit": {
            "test_rows": EXPECTED_TEST_ROWS,
            "error_structure_rows": len(error_frame),
            "paired_outcome_rows": len(paired_frame),
            "illustrative_error_rows": len(illustrative_frame),
            "fatal_to_slight": {
                model: {
                    "count": int(fatal_to_slight.loc[model, "count"]),
                    "share_within_fatal": float(
                        fatal_to_slight.loc[model, "share_within_true_class"]
                    ),
                }
                for model in MODEL_IDS
            },
            "fatal_cases_logistic_only_correct": int(
                fatal_paired.loc["logistic_only_correct", "count"]
            ),
            "fatal_cases_lightgbm_only_correct": int(
                fatal_paired.loc["lightgbm_only_correct", "count"]
            ),
        },
        "protocol_deviation": {
            "regional_holdout_executed": False,
            "spatial_generalization_claim_allowed": False,
        },
        "D15_ready": True,
    }
    write_json(SUMMARY_FILE, summary)
    checkpoint = f"""# D14 final checkpoint

Status: **D14_COMPLETE_WITH_REGIONAL_HOLDOUT_DEVIATION**

## H1-H3

- H1: broad random-split optimism remains **not supported**; the observed gap is metric dependent.
- H2: LightGBM's Macro-F1 rule passed, but the joint incremental-value rule remains **failed** because severe-class trade-offs remain.
- H3: overall SHAP rank rho across five fixed-model temporal comparisons was mean **{shap_summary['overall_rho_mean']:.6f}**, SD **{shap_summary['overall_rho_sd']:.6f}**, range **[{shap_summary['overall_rho_range'][0]:.6f}, {shap_summary['overall_rho_range'][1]:.6f}]**. Ranking change was quantified; no arbitrary binary materiality threshold was added.

## Excluding 2020

- LightGBM-versus-Logistic Macro-F1 delta: **{exclude_summary['sensitivity_H2_macro_f1_delta']:+.6f}**.
- LightGBM-versus-Logistic fatal-recall delta: **{exclude_summary['sensitivity_H2_fatal_recall_delta']:+.6f}**.
- The metric trade-off persists; this is not a causal pandemic analysis.

## Error structure

- Fatal-to-Slight errors: weighted Logistic **{int(fatal_to_slight.loc['logistic_weighted', 'count'])}** ({float(fatal_to_slight.loc['logistic_weighted', 'share_within_true_class']):.3%} of fatal cases); LightGBM **{int(fatal_to_slight.loc['lightgbm_weighted', 'count'])}** ({float(fatal_to_slight.loc['lightgbm_weighted', 'share_within_true_class']):.3%}).
- Among fatal cases, Logistic alone was correct for **{int(fatal_paired.loc['logistic_only_correct', 'count'])}** records and LightGBM alone for **{int(fatal_paired.loc['lightgbm_only_correct', 'count'])}** records.
- Individual high-confidence errors are deterministic appendix illustrations, not representative cases.

## Protocol deviation

- The planned regional holdout was not executed because no concrete police-force combination was frozen before outcomes were known.
- No spatial-generalization claim is allowed. Selecting a region now would be post-hoc.

## Handoff

- D14 is complete for SHAP, exclusion-2020 sensitivity, error audit and reporting boundaries.
- D15 may begin with software packaging and end-to-end reproducibility work.
"""
    CHECKPOINT_FILE.write_text(checkpoint, encoding="utf-8")

    artifacts = [
        PROTOCOL_FILE,
        ERROR_STRUCTURE_FILE,
        PAIRED_ERROR_FILE,
        ILLUSTRATIVE_ERROR_FILE,
        DEVIATION_FILE,
        BOUNDARY_FILE,
        CHECKPOINT_FILE,
        SUMMARY_FILE,
        Path(__file__),
        TEST_SOURCE_FILE,
    ]
    manifest = pd.DataFrame(
        [
            {
                "relative_path": relative(path),
                "bytes": int(path.stat().st_size),
                "sha256": hash_file(path),
            }
            for path in artifacts
        ]
    )
    write_frame(MANIFEST_FILE, manifest)
    print("D14 closeout complete.")
    print("D14_STATUS=D14_COMPLETE_WITH_REGIONAL_HOLDOUT_DEVIATION")
    print("D14_MANIFEST_SHA256=", hash_file(MANIFEST_FILE))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.freeze == args.run:
        parser.error("choose exactly one of --freeze or --run")
    if args.freeze:
        freeze_protocol()
    else:
        run_closeout()


if __name__ == "__main__":
    main()
