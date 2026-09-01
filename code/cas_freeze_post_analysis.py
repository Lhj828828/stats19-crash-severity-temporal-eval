"""Freeze the operational CAS uncertainty and SHAP analysis supplement."""

from __future__ import annotations

import json
from datetime import datetime

from cas_modeling_common import (
    ANALYSIS_PROTOCOL_FILE,
    PROJECT_DIR,
    THREADS,
    hash_file,
    load_contracts,
    relative,
    write_json_atomic,
)


VERSION = "CAS_POST_ANALYSIS_V1"
EVALUATION_PROTOCOL_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_evaluation_protocol.json"
)
EVALUATION_COMPLETE_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_evaluation_complete.json"
)
EVALUATION_METRICS_FILE = (
    PROJECT_DIR / "results" / "cas_evaluation" / "test_metrics.csv"
)
LIGHTGBM_FROZEN_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_lightgbm_models_frozen.json"
)
OUTPUT_FILE = PROJECT_DIR / "config" / "cas" / "cas_post_analysis_protocol.json"
BOOTSTRAP_RESULT_FILE = (
    PROJECT_DIR / "results" / "cas_post_analysis" / "h1_bootstrap.csv"
)
SHAP_RESULT_FILE = (
    PROJECT_DIR / "results" / "cas_post_analysis" / "h3_shap_stability.csv"
)

METRICS = [
    "macro_f1",
    "qwk",
    "ordinal_mae",
    "accuracy",
    "minor_recall",
    "serious_recall",
    "fatal_recall",
    "serious_or_fatal_recall",
    "mean_asymmetric_cost",
]


def main() -> None:
    _schema, analysis, _training = load_contracts()
    evaluation = json.loads(
        EVALUATION_COMPLETE_FILE.read_text(encoding="utf-8")
    )
    lightgbm = json.loads(LIGHTGBM_FROZEN_FILE.read_text(encoding="utf-8"))
    if evaluation.get("status") != "FROZEN_MODELS_EVALUATED_ONCE":
        raise ValueError("CAS one-time evaluation is not complete")
    if lightgbm.get("status") != (
        "LIGHTGBM_MODELS_FROZEN_BEFORE_2025_EVALUATION"
    ):
        raise ValueError("CAS LightGBM models are not frozen")
    if BOOTSTRAP_RESULT_FILE.exists() or SHAP_RESULT_FILE.exists():
        raise RuntimeError("CAS post-analysis results exist; refusing to refreeze")

    iterations = int(analysis["frozen_metrics"]["bootstrap_iterations"])
    seed = int(analysis["frozen_metrics"]["bootstrap_seed"])
    payload = {
        "version": VERSION,
        "status": "OPERATIONAL_SUPPLEMENT_FROZEN_BEFORE_BOOTSTRAP_OR_SHAP_RESULTS",
        "created_local": datetime.now()
        .astimezone()
        .isoformat(timespec="seconds"),
        "freeze_timing_disclosure": (
            "The parent analysis protocol fixed the bootstrap count, seed, "
            "metrics and H1-H3 questions before modeling. This operational "
            "supplement was frozen after test point estimates existed but "
            "before any CAS bootstrap interval or SHAP result was computed."
        ),
        "upstream": {
            "parent_analysis_protocol": relative(ANALYSIS_PROTOCOL_FILE),
            "parent_analysis_protocol_sha256": hash_file(
                ANALYSIS_PROTOCOL_FILE
            ),
            "evaluation_protocol": relative(EVALUATION_PROTOCOL_FILE),
            "evaluation_protocol_sha256": hash_file(EVALUATION_PROTOCOL_FILE),
            "evaluation_complete": relative(EVALUATION_COMPLETE_FILE),
            "evaluation_complete_sha256": hash_file(EVALUATION_COMPLETE_FILE),
            "evaluation_metrics": relative(EVALUATION_METRICS_FILE),
            "evaluation_metrics_sha256": hash_file(EVALUATION_METRICS_FILE),
            "lightgbm_models_frozen": relative(LIGHTGBM_FROZEN_FILE),
            "lightgbm_models_frozen_sha256": hash_file(LIGHTGBM_FROZEN_FILE),
        },
        "common": {
            "iterations": iterations,
            "confidence_level": 0.95,
            "interval": "percentile",
            "base_seed": seed,
            "stratification": "true severity class",
            "metrics": METRICS,
            "higher_is_better": {
                "macro_f1": True,
                "qwk": True,
                "ordinal_mae": False,
                "accuracy": True,
                "minor_recall": True,
                "serious_recall": True,
                "fatal_recall": True,
                "serious_or_fatal_recall": True,
                "mean_asymmetric_cost": False,
            },
        },
        "H1": {
            "models": [
                "dummy_most_frequent",
                "logistic_weighted",
                "lightgbm_weighted",
            ],
            "seeds": analysis["random_reference_protocol"]["seeds"],
            "contrast": "random internal test minus same frozen model on 2025",
            "bootstrap": (
                "independent class-stratified resampling of the two cohorts"
            ),
            "within_seed_reporting": "point gap and percentile interval",
            "across_seed_reporting": (
                "point-gap mean, sample SD, minimum and maximum; no five-seed "
                "normal-theory significance test"
            ),
        },
        "H2": {
            "groups": 11,
            "contrast": "weighted LightGBM minus weighted Logistic",
            "bootstrap": (
                "paired class-stratified resampling of shared records within "
                "each evaluation group"
            ),
            "reporting": "point difference and percentile interval",
        },
        "H3": {
            "models": "five frozen random-reference LightGBM models",
            "contrast": "random internal test versus same model on 2025",
            "internal_sample": "all 4,887 internal-test rows",
            "future_sample": (
                "4,887 rows sampled without replacement from 2025, exactly "
                "matching that seed's internal-test class counts"
            ),
            "future_sample_seed_rule": "base_seed + random_split_seed",
            "shap_engine": "LightGBM pred_contrib=True on frozen boosters",
            "class_aggregation": (
                "mean absolute SHAP over records and all three output classes"
            ),
            "ranking_unit": "15 frozen input features",
            "stability_metric": "Spearman rank correlation",
            "bootstrap": (
                "independent class-stratified record resampling of precomputed "
                "absolute SHAP contributions"
            ),
            "threads": THREADS,
            "causal_boundary": (
                "SHAP is descriptive model attribution, not causal evidence"
            ),
        },
        "cross_dataset_boundary": (
            "Report direction and uncertainty separately for CAS and STATS19. "
            "Do not pool records or absolute metrics and do not claim "
            "cross-national generalizability."
        ),
    }
    write_json_atomic(OUTPUT_FILE, payload)
    print(f"CAS_POST_ANALYSIS_PROTOCOL_SHA256={hash_file(OUTPUT_FILE)}")
    print("CAS_POST_ANALYSIS_PROTOCOL_STATUS=FROZEN_BEFORE_BOOTSTRAP_OR_SHAP")


if __name__ == "__main__":
    main()
