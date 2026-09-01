"""D12 uncertainty analysis from the frozen D11 row-level predictions."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "4"

import matplotlib
import numpy as np
import pandas as pd
import sklearn

matplotlib.use("Agg")

from baseline_modeling import ASYMMETRIC_COST_MATRIX, TARGET_CODES, classification_metrics

PROJECT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_DIR / "code"
CONFIG_DIR = PROJECT_DIR / "config"
RESULT_DIR = PROJECT_DIR / "results" / "d12"
LOG_DIR = PROJECT_DIR / "logs"
FIGURE_DIR = PROJECT_DIR / "figures"
PREDICTION_DIR = PROJECT_DIR / "results" / "d11" / "predictions"

D6_PROTOCOL_FILE = CONFIG_DIR / "d6_analysis_protocol.json"
D11_PROTOCOL_FILE = CONFIG_DIR / "d11_evaluation_protocol.json"
D11_MANIFEST_FILE = LOG_DIR / "d11_artifact_manifest.csv"
D11_LOCK_FILE = LOG_DIR / "d11_run_lock.json"
D11_METRICS_FILE = PROJECT_DIR / "results" / "d11" / "d11_test_metrics.csv"

PROTOCOL_FILE = CONFIG_DIR / "d12_bootstrap_protocol.json"
MODEL_INTERVAL_FILE = RESULT_DIR / "d12_model_metric_intervals.csv"
PAIRWISE_FILE = RESULT_DIR / "d12_pairwise_model_differences.csv"
OPTIMISM_FILE = RESULT_DIR / "d12_random_optimism_gaps.csv"
OPTIMISM_SUMMARY_FILE = RESULT_DIR / "d12_random_optimism_summary.csv"
GAIN_SUMMARY_FILE = RESULT_DIR / "d12_model_gain_summary.csv"
DRAW_FILE = RESULT_DIR / "d12_bootstrap_draws.npz"
DRAW_METADATA_FILE = RESULT_DIR / "d12_bootstrap_draw_metadata.json"
AUDIT_FILE = LOG_DIR / "d12_bootstrap_audit.csv"
SUMMARY_FILE = LOG_DIR / "d12_run_summary.json"
CHECKPOINT_FILE = LOG_DIR / "d12_checkpoint.md"
MANIFEST_FILE = LOG_DIR / "d12_artifact_manifest.csv"
H1_FIGURE_FILE = FIGURE_DIR / "d12_h1_optimism_gaps.png"
H2_FIGURE_FILE = FIGURE_DIR / "d12_h2_temporal_differences.png"
TEST_SOURCE_FILE = CODE_DIR / "test_d12_bootstrap.py"

VERSION = "D12_V1"
CREATED_LOCAL = "2026-08-31"
BOOTSTRAP_ITERATIONS = 2_000
BOOTSTRAP_SEED = 20_260_828
CI_ALPHA = 0.05
CI_METHOD = "percentile"
QUANTILE_METHOD = "linear"
RANDOM_SEEDS = (1103, 2207, 3301, 4409, 5501)
EXPECTED_TEMPORAL_ROWS = 100_927
EXPECTED_RANDOM_ROWS = 96_408

MODEL_IDS = (
    "dummy_most_frequent",
    "logistic_unweighted",
    "logistic_weighted",
    "ordered_logit_unweighted",
    "lightgbm_weighted",
)
REFERENCE_MODEL = "logistic_weighted"
CANDIDATE_MODEL = "lightgbm_weighted"

METRICS = (
    "macro_f1",
    "qwk",
    "ordinal_mae",
    "accuracy",
    "fatal_recall",
    "serious_or_fatal_recall",
    "mean_asymmetric_cost",
)
FROZEN_REQUIRED_METRICS = (
    "macro_f1",
    "qwk",
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


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def direction(metric: str) -> str:
    if metric in HIGHER_IS_BETTER:
        return "higher_is_better"
    if metric in LOWER_IS_BETTER:
        return "lower_is_better"
    raise KeyError(metric)


def advantage_sign(metric: str) -> float:
    return 1.0 if metric in HIGHER_IS_BETTER else -1.0


def quantile_interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(
        np.asarray(values, dtype=float),
        [CI_ALPHA / 2, 1 - CI_ALPHA / 2],
        method=QUANTILE_METHOD,
    )
    return float(lower), float(upper)


def group_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "group_key": "temporal_test_2024",
            "evaluation_scope": "temporal_test_2024",
            "split_slug": "temporal",
            "sample_role": "strict_temporal_test",
            "file_suffix": "test",
            "expected_rows": EXPECTED_TEMPORAL_ROWS,
            "expected_years": [2024],
            "record_definition": (
                "All 100,927 collisions in the frozen 2024 temporal test set, "
                "evaluated by the model trained on 2018-2022 and selected using 2023."
            ),
        }
    ]
    for seed in RANDOM_SEEDS:
        slug = f"random_seed_{seed}"
        specs.append(
            {
                "group_key": f"{slug}__internal_test",
                "evaluation_scope": "random_reference_test",
                "split_slug": slug,
                "sample_role": "random_internal_test",
                "file_suffix": "test",
                "expected_rows": EXPECTED_RANDOM_ROWS,
                "expected_years": [2018, 2019, 2020, 2021, 2022, 2023],
                "record_definition": (
                    "The fixed severity-stratified 15% internal test partition "
                    "from the 2018-2023 development pool for this random seed."
                ),
                "seed": seed,
            }
        )
        specs.append(
            {
                "group_key": f"{slug}__2024_diagnostic",
                "evaluation_scope": "random_model_on_2024",
                "split_slug": slug,
                "sample_role": "random_model_future_test",
                "file_suffix": "2024_diagnostic",
                "expected_rows": EXPECTED_TEMPORAL_ROWS,
                "expected_years": [2024],
                "record_definition": (
                    "All 100,927 frozen 2024 collisions evaluated once by the "
                    "already-fitted model from this random split, without retuning."
                ),
                "seed": seed,
            }
        )
    return specs


def prediction_path(spec: dict[str, Any], model: str) -> Path:
    return PREDICTION_DIR / (
        f"{spec['split_slug']}__{model}__{spec['file_suffix']}.csv.gz"
    )


def verify_d11_manifest(*, calculate_hashes: bool = True) -> dict[str, Any]:
    manifest = pd.read_csv(D11_MANIFEST_FILE, dtype={"relative_path": str, "sha256": str})
    prediction_rows = manifest[manifest["relative_path"].str.startswith(
        "results/d11/predictions/"
    )]
    if len(manifest) != 64 or len(prediction_rows) != 55:
        raise AssertionError("Unexpected D11 artifact manifest size")
    for row in manifest.itertuples(index=False):
        path = PROJECT_DIR / row.relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        if int(path.stat().st_size) != int(row.bytes):
            raise AssertionError(f"D11 artifact size changed: {row.relative_path}")
        if calculate_hashes and hash_file(path) != row.sha256:
            raise AssertionError(f"D11 artifact hash changed: {row.relative_path}")
    return {
        "manifest_rows": int(len(manifest)),
        "prediction_files": int(len(prediction_rows)),
        "prediction_bytes": int(prediction_rows["bytes"].astype(int).sum()),
        "manifest_sha256": hash_file(D11_MANIFEST_FILE),
        "metrics_sha256": hash_file(D11_METRICS_FILE),
        "run_lock_sha256": hash_file(D11_LOCK_FILE),
    }


def build_protocol() -> dict[str, Any]:
    d6 = read_json(D6_PROTOCOL_FILE)
    d11 = read_json(D11_PROTOCOL_FILE)
    frozen = d6["frozen_metrics"]
    if int(frozen["bootstrap_iterations"]) != BOOTSTRAP_ITERATIONS:
        raise AssertionError("D6 iteration count does not match D12")
    if int(frozen["bootstrap_seed"]) != BOOTSTRAP_SEED:
        raise AssertionError("D6 bootstrap seed does not match D12")
    if d11["status"] != "FROZEN_BEFORE_D11_TEST_ACCESS":
        raise AssertionError("Unexpected D11 protocol status")
    input_audit = verify_d11_manifest(calculate_hashes=True)
    groups = group_specs()
    return {
        "version": VERSION,
        "status": "FROZEN_D12_IMPLEMENTATION_DERIVED_FROM_D6",
        "created_local": CREATED_LOCAL,
        "freeze_timing_disclosure": (
            "D6 fixed class-stratified paired nonparametric bootstrap, 2,000 "
            "iterations and seed 20260828 before model fitting. The current file "
            "was created after D11 outcomes were available and therefore does not "
            "claim that implementation details newly specified here were preregistered."
        ),
        "scientific_lock": {
            "read_frozen_D11_predictions_only": True,
            "model_fits": 0,
            "threshold_changes": 0,
            "model_selection_operations": 0,
            "D11_reruns": 0,
        },
        "parent_artifacts": {
            relative(D6_PROTOCOL_FILE): hash_file(D6_PROTOCOL_FILE),
            relative(D11_PROTOCOL_FILE): hash_file(D11_PROTOCOL_FILE),
            relative(D11_MANIFEST_FILE): input_audit["manifest_sha256"],
            relative(D11_LOCK_FILE): input_audit["run_lock_sha256"],
            relative(D11_METRICS_FILE): input_audit["metrics_sha256"],
            relative(Path(__file__)): hash_file(Path(__file__)),
            relative(TEST_SOURCE_FILE): hash_file(TEST_SOURCE_FILE),
        },
        "input_inventory": input_audit,
        "bootstrap": {
            "statistical_unit": "police-reported personal-injury collision",
            "iterations": BOOTSTRAP_ITERATIONS,
            "master_seed": BOOTSTRAP_SEED,
            "rng": "NumPy default_rng (PCG64); deterministic SHA-256-derived stream per evaluation group",
            "resampling": (
                "Within each evaluation data set, sample collision records with "
                "replacement separately inside each true-severity stratum while "
                "retaining that stratum's observed sample size."
            ),
            "paired_model_rule": (
                "All five model predictions on the same collision records use the "
                "same bootstrap draw, preserving model-prediction dependence."
            ),
            "independent_sample_rule": (
                "Random-internal and 2024 tests contain different records and use "
                "independent bootstrap streams; they are never called paired."
            ),
            "computational_equivalence": (
                "For hard-prediction metrics, resampling uses multinomial counts of "
                "the empirical joint five-model prediction pattern within each true "
                "class. This is distribution-equivalent to record-index bootstrap "
                "and preserves pairing while avoiding materializing 100,000-row draws."
            ),
            "confidence_level": 0.95,
            "interval_method": CI_METHOD,
            "quantile_method": QUANTILE_METHOD,
            "multiplicity_adjustment": "none; effect-size intervals are reported without dichotomous significance screening",
        },
        "models": list(MODEL_IDS),
        "metrics": {
            "reported": list(METRICS),
            "D6_required": list(FROZEN_REQUIRED_METRICS),
            "higher_is_better": sorted(HIGHER_IS_BETTER),
            "lower_is_better": sorted(LOWER_IS_BETTER),
        },
        "comparisons": {
            "H1_random_optimism": {
                "definition_higher_is_better": "metric_random_internal - metric_same_random_model_on_2024",
                "definition_lower_is_better": "metric_same_random_model_on_2024 - metric_random_internal",
                "interpretation": "positive means the random internal evaluation is more optimistic",
                "models": list(MODEL_IDS),
                "random_seeds": list(RANDOM_SEEDS),
                "inference_boundary": (
                    "Seed-specific intervals condition on each fitted model and two "
                    "test samples. Across-seed mean, SD and range are descriptive; "
                    "overlapping random partitions are not treated as independent studies."
                ),
                "two_layer_reporting": {
                    "within_seed": (
                        "For each seed, report the point gap and independent class-stratified "
                        "percentile bootstrap interval for internal test versus 2024."
                    ),
                    "across_seed": (
                        "Across the five fixed seeds, report mean, sample SD and range of "
                        "the five point gaps as training/split sensitivity; do not attach an "
                        "independence-based meta-analytic confidence interval."
                    ),
                },
            },
            "H2_temporal_model_comparison": {
                "candidate": CANDIDATE_MODEL,
                "reference": REFERENCE_MODEL,
                "sample": "same frozen 2024 temporal test records",
                "raw_delta": "candidate - reference",
                "oriented_advantage": "positive always favors candidate",
                "macro_f1_practical_threshold": 0.01,
                "joint_claim_rule": (
                    "A positive Macro-F1 result alone is insufficient; QWK, fatal "
                    "recall and asymmetric cost must also be inspected."
                ),
            },
        },
        "evaluation_groups": groups,
        "limitations": [
            "Intervals condition on fixed fitted models and do not include full retraining variability.",
            "The H1 gap is descriptive and is not a causal effect of the split protocol.",
            "Class-stratified bootstrap conditions on observed class counts and therefore does not model prevalence uncertainty.",
            "Percentile intervals do not establish practical equivalence when they cross zero.",
        ],
        "planned_outputs": [
            relative(MODEL_INTERVAL_FILE),
            relative(PAIRWISE_FILE),
            relative(OPTIMISM_FILE),
            relative(OPTIMISM_SUMMARY_FILE),
            relative(GAIN_SUMMARY_FILE),
            relative(DRAW_FILE),
            relative(DRAW_METADATA_FILE),
            relative(AUDIT_FILE),
            relative(SUMMARY_FILE),
            relative(CHECKPOINT_FILE),
            relative(H1_FIGURE_FILE),
            relative(H2_FIGURE_FILE),
        ],
    }


def freeze_protocol() -> None:
    payload = build_protocol()
    if PROTOCOL_FILE.exists():
        existing = read_json(PROTOCOL_FILE)
        if existing != payload:
            raise RuntimeError(
                "Existing D12 protocol differs. Do not replace it without a versioned amendment."
            )
        print(f"D12_PROTOCOL_ALREADY_FROZEN={hash_file(PROTOCOL_FILE)}")
        return
    write_json(PROTOCOL_FILE, payload)
    print(f"D12_PROTOCOL_FROZEN={hash_file(PROTOCOL_FILE)}")


def validate_protocol() -> dict[str, Any]:
    if not PROTOCOL_FILE.is_file():
        raise FileNotFoundError("Freeze config/d12_bootstrap_protocol.json before D12 execution")
    protocol = read_json(PROTOCOL_FILE)
    if protocol["version"] != VERSION:
        raise AssertionError("Unexpected D12 protocol version")
    if int(protocol["bootstrap"]["iterations"]) != BOOTSTRAP_ITERATIONS:
        raise AssertionError("D12 iteration mismatch")
    if int(protocol["bootstrap"]["master_seed"]) != BOOTSTRAP_SEED:
        raise AssertionError("D12 seed mismatch")
    for name, expected_hash in protocol["parent_artifacts"].items():
        path = PROJECT_DIR / name
        if hash_file(path) != expected_hash:
            raise AssertionError(f"Frozen D12 parent changed: {name}")
    return protocol


def stream_identity(group_key: str) -> tuple[np.random.Generator, dict[str, Any]]:
    digest = hashlib.sha256(group_key.encode("utf-8")).digest()
    words = [int.from_bytes(digest[offset:offset + 4], "little") for offset in (0, 4, 8, 12)]
    entropy = [BOOTSTRAP_SEED, *words]
    sequence = np.random.SeedSequence(entropy)
    return np.random.default_rng(sequence), {
        "group_key": group_key,
        "entropy": entropy,
        "spawn_key": list(sequence.spawn_key),
        "bit_generator": "PCG64",
    }


def load_group_predictions(
    spec: dict[str, Any],
    d11_metrics: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    usecols = [
        "meta_collision_index",
        "meta_collision_year",
        "evaluation_scope",
        "split_slug",
        "model",
        "target_severity",
        "predicted_severity",
    ]
    base_ids: np.ndarray | None = None
    base_years: np.ndarray | None = None
    base_target: np.ndarray | None = None
    predictions: list[np.ndarray] = []
    audit_rows: list[dict[str, Any]] = []
    for model in MODEL_IDS:
        path = prediction_path(spec, model)
        frame = pd.read_csv(
            path,
            usecols=usecols,
            dtype={"meta_collision_index": "string"},
        )
        if len(frame) != int(spec["expected_rows"]):
            raise AssertionError(f"Unexpected row count: {path}")
        if frame["meta_collision_index"].duplicated().any():
            raise AssertionError(f"Duplicate collision ID: {path}")
        for column, expected in (
            ("evaluation_scope", spec["evaluation_scope"]),
            ("split_slug", spec["split_slug"]),
            ("model", model),
        ):
            if frame[column].nunique(dropna=False) != 1 or frame[column].iloc[0] != expected:
                raise AssertionError(f"Unexpected {column}: {path}")
        ids = frame["meta_collision_index"].astype(str).to_numpy()
        years = frame["meta_collision_year"].to_numpy(dtype=np.int16)
        target = frame["target_severity"].to_numpy(dtype=np.int8)
        predicted = frame["predicted_severity"].to_numpy(dtype=np.int8)
        if sorted(np.unique(years).tolist()) != spec["expected_years"]:
            raise AssertionError(f"Unexpected years: {path}")
        if set(np.unique(target)) != set(TARGET_CODES):
            raise AssertionError(f"Missing target class: {path}")
        if not set(np.unique(predicted)).issubset(TARGET_CODES):
            raise AssertionError(f"Invalid predicted class: {path}")
        if base_ids is None:
            base_ids, base_years, base_target = ids, years, target
        else:
            if not np.array_equal(ids, base_ids):
                raise AssertionError(f"Collision order differs within group: {path}")
            if not np.array_equal(years, base_years) or not np.array_equal(target, base_target):
                raise AssertionError(f"Cohort labels differ within group: {path}")
        observed = classification_metrics(target, predicted)
        metric_row = d11_metrics.loc[
            d11_metrics["evaluation_scope"].eq(spec["evaluation_scope"])
            & d11_metrics["split_slug"].eq(spec["split_slug"])
            & d11_metrics["model"].eq(model)
        ]
        if len(metric_row) != 1:
            raise AssertionError(f"D11 metric row missing or duplicated: {spec['group_key']} {model}")
        for metric in METRICS:
            if not np.isclose(
                observed[metric], float(metric_row.iloc[0][metric]), rtol=0, atol=1e-13
            ):
                raise AssertionError(
                    f"D11 metric mismatch: {spec['group_key']} {model} {metric}"
                )
        predictions.append(predicted)
        audit_rows.append(
            {
                "group_key": spec["group_key"],
                "evaluation_scope": spec["evaluation_scope"],
                "split_slug": spec["split_slug"],
                "sample_role": spec["sample_role"],
                "model": model,
                "prediction_file": relative(path),
                "rows": int(len(frame)),
                "slight_rows": int(np.sum(target == 0)),
                "serious_rows": int(np.sum(target == 1)),
                "fatal_rows": int(np.sum(target == 2)),
                "collision_ids_unique": True,
                "aligned_with_group": True,
                "D11_metrics_reproduced": True,
            }
        )
        del frame, ids, years, target, predicted
        gc.collect()
    assert base_target is not None
    return base_target, np.column_stack(predictions), audit_rows


def confusion_from_vectors(y_true: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    if predictions.ndim != 2 or predictions.shape[0] != len(y_true):
        raise ValueError("Predictions must have shape (rows, models)")
    model_count = predictions.shape[1]
    matrices = np.zeros((model_count, 3, 3), dtype=np.int64)
    for model_index in range(model_count):
        np.add.at(matrices[model_index], (y_true, predictions[:, model_index]), 1)
    return matrices


def bootstrap_joint_confusions(
    y_true: np.ndarray,
    predictions: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    model_count = len(MODEL_IDS)
    pattern_count = 3 ** model_count
    pattern_codes = np.arange(pattern_count, dtype=np.int64)
    powers = 3 ** np.arange(model_count, dtype=np.int64)
    decoded = (pattern_codes[:, None] // powers[None, :]) % 3
    draws = np.zeros(
        (BOOTSTRAP_ITERATIONS, model_count, 3, 3),
        dtype=np.int32,
    )
    encoded = np.sum(predictions.astype(np.int64) * powers[None, :], axis=1)
    for true_code in TARGET_CODES:
        class_codes = encoded[y_true == true_code]
        n_class = int(len(class_codes))
        empirical = np.bincount(class_codes, minlength=pattern_count)
        sampled = rng.multinomial(
            n_class,
            empirical / n_class,
            size=BOOTSTRAP_ITERATIONS,
        )
        for model_index in range(model_count):
            for predicted_code in TARGET_CODES:
                draws[:, model_index, true_code, predicted_code] = sampled[
                    :, decoded[:, model_index] == predicted_code
                ].sum(axis=1)
        del sampled
    return draws


def metrics_from_confusions(confusions: np.ndarray) -> dict[str, np.ndarray]:
    matrix = np.asarray(confusions, dtype=float)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("Confusion array must end in 3x3")
    total = matrix.sum(axis=(-2, -1))
    true_total = matrix.sum(axis=-1)
    predicted_total = matrix.sum(axis=-2)
    diagonal = np.diagonal(matrix, axis1=-2, axis2=-1)
    recalls = np.divide(
        diagonal,
        true_total,
        out=np.zeros_like(diagonal),
        where=true_total != 0,
    )
    precisions = np.divide(
        diagonal,
        predicted_total,
        out=np.zeros_like(diagonal),
        where=predicted_total != 0,
    )
    f1 = np.divide(
        2 * precisions * recalls,
        precisions + recalls,
        out=np.zeros_like(diagonal),
        where=(precisions + recalls) != 0,
    )
    weights = (
        np.arange(3, dtype=float)[:, None] - np.arange(3, dtype=float)[None, :]
    ) ** 2
    observed_weighted = np.sum(matrix * weights, axis=(-2, -1))
    expected = np.einsum("...i,...j->...ij", true_total, predicted_total) / total[..., None, None]
    expected_weighted = np.sum(expected * weights, axis=(-2, -1))
    qwk = 1.0 - np.divide(
        observed_weighted,
        expected_weighted,
        out=np.zeros_like(observed_weighted),
        where=expected_weighted != 0,
    )
    ordinal_weights = np.abs(
        np.arange(3, dtype=float)[:, None] - np.arange(3, dtype=float)[None, :]
    )
    severe_true = true_total[..., 1] + true_total[..., 2]
    severe_correct_binary = matrix[..., 1:, 1:].sum(axis=(-2, -1))
    return {
        "macro_f1": f1.mean(axis=-1),
        "qwk": qwk,
        "ordinal_mae": np.sum(matrix * ordinal_weights, axis=(-2, -1)) / total,
        "accuracy": diagonal.sum(axis=-1) / total,
        "fatal_recall": recalls[..., 2],
        "serious_or_fatal_recall": severe_correct_binary / severe_true,
        "mean_asymmetric_cost": np.sum(
            matrix * ASYMMETRIC_COST_MATRIX, axis=(-2, -1)
        ) / total,
    }


def stack_metric_draws(metric_values: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([metric_values[metric] for metric in METRICS], axis=-1)


def model_interval_rows(
    spec: dict[str, Any],
    point_values: np.ndarray,
    draws: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_index, model in enumerate(MODEL_IDS):
        for metric_index, metric in enumerate(METRICS):
            lower, upper = quantile_interval(draws[:, model_index, metric_index])
            rows.append(
                {
                    "group_key": spec["group_key"],
                    "evaluation_scope": spec["evaluation_scope"],
                    "split_slug": spec["split_slug"],
                    "sample_role": spec["sample_role"],
                    "seed": spec.get("seed", ""),
                    "model": model,
                    "metric": metric,
                    "direction": direction(metric),
                    "point_estimate": float(point_values[model_index, metric_index]),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "confidence_level": 1 - CI_ALPHA,
                    "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                    "interval_method": CI_METHOD,
                    "resampling_relation": "paired_across_models_within_group",
                }
            )
    return rows


def pairwise_rows(
    spec: dict[str, Any],
    point_values: np.ndarray,
    draws: np.ndarray,
) -> list[dict[str, Any]]:
    candidate_index = MODEL_IDS.index(CANDIDATE_MODEL)
    reference_index = MODEL_IDS.index(REFERENCE_MODEL)
    rows: list[dict[str, Any]] = []
    for metric_index, metric in enumerate(METRICS):
        raw_point = float(
            point_values[candidate_index, metric_index]
            - point_values[reference_index, metric_index]
        )
        raw_draw = draws[:, candidate_index, metric_index] - draws[:, reference_index, metric_index]
        raw_lower, raw_upper = quantile_interval(raw_draw)
        sign = advantage_sign(metric)
        advantage_draw = sign * raw_draw
        advantage_lower, advantage_upper = quantile_interval(advantage_draw)
        rows.append(
            {
                "group_key": spec["group_key"],
                "evaluation_scope": spec["evaluation_scope"],
                "split_slug": spec["split_slug"],
                "sample_role": spec["sample_role"],
                "seed": spec.get("seed", ""),
                "candidate": CANDIDATE_MODEL,
                "reference": REFERENCE_MODEL,
                "metric": metric,
                "direction": direction(metric),
                "candidate_point": float(point_values[candidate_index, metric_index]),
                "reference_point": float(point_values[reference_index, metric_index]),
                "raw_delta_candidate_minus_reference": raw_point,
                "raw_delta_ci_lower": raw_lower,
                "raw_delta_ci_upper": raw_upper,
                "oriented_advantage": sign * raw_point,
                "oriented_advantage_ci_lower": advantage_lower,
                "oriented_advantage_ci_upper": advantage_upper,
                "positive_oriented_advantage_favors": CANDIDATE_MODEL,
                "resampling_relation": "paired_same_collision_records",
                "confidence_level": 1 - CI_ALPHA,
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            }
        )
    return rows


def optimism_rows(
    all_points: dict[str, np.ndarray],
    all_draws: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in RANDOM_SEEDS:
        internal_key = f"random_seed_{seed}__internal_test"
        future_key = f"random_seed_{seed}__2024_diagnostic"
        internal_points = all_points[internal_key]
        future_points = all_points[future_key]
        internal_draws = all_draws[internal_key]
        future_draws = all_draws[future_key]
        for model_index, model in enumerate(MODEL_IDS):
            for metric_index, metric in enumerate(METRICS):
                sign = advantage_sign(metric)
                point = sign * (
                    internal_points[model_index, metric_index]
                    - future_points[model_index, metric_index]
                )
                draw = sign * (
                    internal_draws[:, model_index, metric_index]
                    - future_draws[:, model_index, metric_index]
                )
                lower, upper = quantile_interval(draw)
                rows.append(
                    {
                        "seed": seed,
                        "model": model,
                        "metric": metric,
                        "direction": direction(metric),
                        "random_internal_point": float(internal_points[model_index, metric_index]),
                        "same_random_model_2024_point": float(future_points[model_index, metric_index]),
                        "optimism_gap": float(point),
                        "optimism_gap_ci_lower": lower,
                        "optimism_gap_ci_upper": upper,
                        "positive_gap_means": "random_internal_more_optimistic",
                        "resampling_relation": "independent_test_samples",
                        "confidence_level": 1 - CI_ALPHA,
                        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                    }
                )
    return rows


def summarize_optimism(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, metric), group in frame.groupby(["model", "metric"], sort=False):
        values = group["optimism_gap"].to_numpy(dtype=float)
        rows.append(
            {
                "model": model,
                "metric": metric,
                "direction": direction(metric),
                "seed_count": int(len(values)),
                "mean_optimism_gap": float(values.mean()),
                "sd_across_seeds": float(values.std(ddof=1)),
                "minimum_seed_gap": float(values.min()),
                "maximum_seed_gap": float(values.max()),
                "positive_seed_count": int(np.sum(values > 0)),
                "zero_crossing_seed_interval_count": int(
                    np.sum(
                        (group["optimism_gap_ci_lower"].to_numpy(dtype=float) <= 0)
                        & (group["optimism_gap_ci_upper"].to_numpy(dtype=float) >= 0)
                    )
                ),
                "across_seed_status": "descriptive_only_overlapping_random_partitions",
            }
        )
    return pd.DataFrame(rows)


def summarize_model_gain(pairwise: pd.DataFrame) -> pd.DataFrame:
    random_rows = pairwise[pairwise["sample_role"].eq("random_internal_test")]
    temporal_rows = pairwise[pairwise["sample_role"].eq("strict_temporal_test")]
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        random_metric = random_rows[random_rows["metric"].eq(metric)]
        temporal_metric = temporal_rows[temporal_rows["metric"].eq(metric)]
        if len(random_metric) != len(RANDOM_SEEDS) or len(temporal_metric) != 1:
            raise AssertionError(f"Unexpected gain rows for {metric}")
        random_values = random_metric["oriented_advantage"].to_numpy(dtype=float)
        temporal_value = float(temporal_metric.iloc[0]["oriented_advantage"])
        rows.append(
            {
                "metric": metric,
                "direction": direction(metric),
                "random_internal_mean_oriented_advantage": float(random_values.mean()),
                "random_internal_sd_across_seeds": float(random_values.std(ddof=1)),
                "random_internal_minimum": float(random_values.min()),
                "random_internal_maximum": float(random_values.max()),
                "temporal_2024_oriented_advantage": temporal_value,
                "temporal_minus_random_mean_advantage": float(
                    temporal_value - random_values.mean()
                ),
                "interpretation": "descriptive_protocol_comparison_not_causal",
            }
        )
    return pd.DataFrame(rows)


def save_figures(pairwise: pd.DataFrame, optimism: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    temporal = pairwise[pairwise["group_key"].eq("temporal_test_2024")].copy()
    temporal["label"] = temporal["metric"].map(
        {
            "macro_f1": "Macro-F1",
            "qwk": "QWK",
            "ordinal_mae": "Ordinal MAE",
            "accuracy": "Accuracy",
            "fatal_recall": "Fatal recall",
            "serious_or_fatal_recall": "Serious/Fatal recall",
            "mean_asymmetric_cost": "Asymmetric cost",
        }
    )
    y = np.arange(len(temporal))
    values = temporal["oriented_advantage"].to_numpy(dtype=float)
    lower = temporal["oriented_advantage_ci_lower"].to_numpy(dtype=float)
    upper = temporal["oriented_advantage_ci_upper"].to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    axis.errorbar(
        values,
        y,
        xerr=np.vstack([values - lower, upper - values]),
        fmt="o",
        color="#2166AC",
        ecolor="#7A8791",
        capsize=3,
    )
    axis.axvline(0, color="#333333", linewidth=1)
    axis.set_yticks(y, temporal["label"])
    axis.invert_yaxis()
    axis.set_xlabel("Oriented advantage (positive favors LightGBM)")
    axis.set_title("D12 paired bootstrap: 2024 temporal test")
    axis.grid(axis="x", color="#D9DEE3", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    H2_FIGURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(H2_FIGURE_FILE, dpi=300, facecolor="white")
    plt.close(figure)

    selected = optimism[
        optimism["model"].eq(CANDIDATE_MODEL)
        & optimism["metric"].isin(["macro_f1", "qwk", "fatal_recall"])
    ].copy()
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 4.1), constrained_layout=True)
    labels = {"macro_f1": "Macro-F1", "qwk": "QWK", "fatal_recall": "Fatal recall"}
    for axis, metric in zip(axes, ["macro_f1", "qwk", "fatal_recall"], strict=True):
        subset = selected[selected["metric"].eq(metric)].sort_values("seed")
        x = np.arange(len(subset))
        values = subset["optimism_gap"].to_numpy(dtype=float)
        lower = subset["optimism_gap_ci_lower"].to_numpy(dtype=float)
        upper = subset["optimism_gap_ci_upper"].to_numpy(dtype=float)
        axis.errorbar(
            x,
            values,
            yerr=np.vstack([values - lower, upper - values]),
            fmt="o",
            color="#B2182B",
            ecolor="#7A8791",
            capsize=3,
        )
        axis.axhline(0, color="#333333", linewidth=1)
        axis.set_xticks(x, subset["seed"].astype(str), rotation=35)
        axis.set_title(labels[metric])
        axis.grid(axis="y", color="#D9DEE3", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Optimism gap (positive = random internal is better)")
    figure.suptitle("D12 independent bootstrap: random LightGBM internal test vs 2024")
    H1_FIGURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(H1_FIGURE_FILE, dpi=300, facecolor="white")
    plt.close(figure)


def write_checkpoint(
    pairwise: pd.DataFrame,
    optimism_summary: pd.DataFrame,
    elapsed_seconds: float,
) -> None:
    temporal = pairwise[pairwise["group_key"].eq("temporal_test_2024")].set_index("metric")
    lightgbm_h1 = optimism_summary[
        optimism_summary["model"].eq(CANDIDATE_MODEL)
    ].set_index("metric")
    macro = temporal.loc["macro_f1"]
    fatal = temporal.loc["fatal_recall"]
    qwk = temporal.loc["qwk"]
    cost = temporal.loc["mean_asymmetric_cost"]
    lines = [
        "# D12 Bootstrap checkpoint",
        "",
        f"Completed: {CREATED_LOCAL}",
        f"Runtime: {elapsed_seconds:.2f} seconds",
        f"Iterations: {BOOTSTRAP_ITERATIONS}; master seed: {BOOTSTRAP_SEED}",
        "",
        "## H2: LightGBM versus weighted Logistic on the same 2024 test records",
        "",
        (
            f"- Macro-F1 raw delta: {macro['raw_delta_candidate_minus_reference']:+.6f} "
            f"(95% CI {macro['raw_delta_ci_lower']:+.6f} to {macro['raw_delta_ci_upper']:+.6f})."
        ),
        (
            f"- QWK raw delta: {qwk['raw_delta_candidate_minus_reference']:+.6f} "
            f"(95% CI {qwk['raw_delta_ci_lower']:+.6f} to {qwk['raw_delta_ci_upper']:+.6f})."
        ),
        (
            f"- Fatal recall raw delta: {fatal['raw_delta_candidate_minus_reference']:+.6f} "
            f"(95% CI {fatal['raw_delta_ci_lower']:+.6f} to {fatal['raw_delta_ci_upper']:+.6f})."
        ),
        (
            f"- Asymmetric cost raw delta: {cost['raw_delta_candidate_minus_reference']:+.6f} "
            f"(95% CI {cost['raw_delta_ci_lower']:+.6f} to {cost['raw_delta_ci_upper']:+.6f}); "
            "negative favors LightGBM."
        ),
        "- Do not call the nonlinear model uniformly superior unless the joint rule is satisfied.",
        "",
        "## H1: random LightGBM internal test versus the same fitted model on 2024",
        "",
    ]
    for metric in ("macro_f1", "qwk", "fatal_recall", "mean_asymmetric_cost"):
        row = lightgbm_h1.loc[metric]
        lines.append(
            f"- {metric}: mean optimism gap {row['mean_optimism_gap']:+.6f}, "
            f"SD across seeds {row['sd_across_seeds']:.6f}, range "
            f"[{row['minimum_seed_gap']:+.6f}, {row['maximum_seed_gap']:+.6f}]."
        )
    lines.extend(
        [
            "",
            "Seed summaries are descriptive because random test partitions overlap. "
            "Seed-specific intervals condition on fixed fitted models and do not include retraining uncertainty.",
            "",
        ]
    )
    CHECKPOINT_FILE.write_text("\n".join(lines), encoding="utf-8")


def build_manifest() -> None:
    paths = [
        Path(__file__),
        TEST_SOURCE_FILE,
        PROTOCOL_FILE,
        MODEL_INTERVAL_FILE,
        PAIRWISE_FILE,
        OPTIMISM_FILE,
        OPTIMISM_SUMMARY_FILE,
        GAIN_SUMMARY_FILE,
        DRAW_FILE,
        DRAW_METADATA_FILE,
        AUDIT_FILE,
        SUMMARY_FILE,
        CHECKPOINT_FILE,
        H1_FIGURE_FILE,
        H2_FIGURE_FILE,
    ]
    rows = [
        {
            "relative_path": relative(path),
            "bytes": int(path.stat().st_size),
            "sha256": hash_file(path),
        }
        for path in sorted(paths, key=lambda item: relative(item))
    ]
    write_frame(MANIFEST_FILE, pd.DataFrame(rows))


def run_analysis() -> None:
    started = time.perf_counter()
    protocol = validate_protocol()
    input_audit = verify_d11_manifest(calculate_hashes=True)
    d11_metrics = pd.read_csv(D11_METRICS_FILE)
    all_points: dict[str, np.ndarray] = {}
    all_draws: dict[str, np.ndarray] = {}
    interval_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    stream_rows: list[dict[str, Any]] = []

    for spec in protocol["evaluation_groups"]:
        group_started = time.perf_counter()
        target, predictions, group_audit = load_group_predictions(spec, d11_metrics)
        point_confusions = confusion_from_vectors(target, predictions)
        point_values = stack_metric_draws(metrics_from_confusions(point_confusions))
        rng, stream = stream_identity(spec["group_key"])
        confusion_draws = bootstrap_joint_confusions(target, predictions, rng)
        metric_draws = stack_metric_draws(metrics_from_confusions(confusion_draws))
        if metric_draws.shape != (BOOTSTRAP_ITERATIONS, len(MODEL_IDS), len(METRICS)):
            raise AssertionError("Unexpected metric draw shape")
        if not np.isfinite(metric_draws).all():
            raise AssertionError(f"Non-finite bootstrap metric: {spec['group_key']}")
        all_points[spec["group_key"]] = point_values
        all_draws[spec["group_key"]] = metric_draws
        interval_rows.extend(model_interval_rows(spec, point_values, metric_draws))
        pair_rows.extend(pairwise_rows(spec, point_values, metric_draws))
        seconds = time.perf_counter() - group_started
        for row in group_audit:
            row["bootstrap_seconds_for_group"] = seconds
            row["bootstrap_stream_key"] = spec["group_key"]
        audit_rows.extend(group_audit)
        stream["sample_role"] = spec["sample_role"]
        stream["rows"] = int(len(target))
        stream["class_counts"] = [int(np.sum(target == code)) for code in TARGET_CODES]
        stream_rows.append(stream)
        print(
            f"BOOTSTRAP_GROUP={spec['group_key']} rows={len(target)} "
            f"seconds={seconds:.2f}",
            flush=True,
        )
        del target, predictions, confusion_draws
        gc.collect()

    interval_frame = pd.DataFrame(interval_rows)
    pairwise_frame = pd.DataFrame(pair_rows)
    optimism_frame = pd.DataFrame(optimism_rows(all_points, all_draws))
    optimism_summary = summarize_optimism(optimism_frame)
    gain_summary = summarize_model_gain(pairwise_frame)

    write_frame(MODEL_INTERVAL_FILE, interval_frame)
    write_frame(PAIRWISE_FILE, pairwise_frame)
    write_frame(OPTIMISM_FILE, optimism_frame)
    write_frame(OPTIMISM_SUMMARY_FILE, optimism_summary)
    write_frame(GAIN_SUMMARY_FILE, gain_summary)
    write_frame(AUDIT_FILE, pd.DataFrame(audit_rows))

    draw_arrays: dict[str, np.ndarray] = {
        "model_ids": np.asarray(MODEL_IDS, dtype="U40"),
        "metric_ids": np.asarray(METRICS, dtype="U40"),
    }
    draw_key_map: dict[str, str] = {}
    for index, spec in enumerate(protocol["evaluation_groups"]):
        key = f"group_{index:02d}"
        draw_key_map[key] = spec["group_key"]
        draw_arrays[key] = all_draws[spec["group_key"]]
    write_npz(DRAW_FILE, draw_arrays)
    write_json(
        DRAW_METADATA_FILE,
        {
            "version": VERSION,
            "draw_file": relative(DRAW_FILE),
            "array_shape_per_group": [
                BOOTSTRAP_ITERATIONS,
                len(MODEL_IDS),
                len(METRICS),
            ],
            "axis_order": ["bootstrap_iteration", "model", "metric"],
            "draw_key_map": draw_key_map,
            "streams": stream_rows,
            "pairing_note": "Models within a group share draws; different groups use independent streams.",
        },
    )
    save_figures(pairwise_frame, optimism_frame)
    elapsed = time.perf_counter() - started
    write_checkpoint(pairwise_frame, optimism_summary, elapsed)

    temporal = pairwise_frame[pairwise_frame["group_key"].eq("temporal_test_2024")]
    macro = temporal[temporal["metric"].eq("macro_f1")].iloc[0]
    fatal = temporal[temporal["metric"].eq("fatal_recall")].iloc[0]
    lightgbm_h1 = optimism_summary[
        optimism_summary["model"].eq(CANDIDATE_MODEL)
        & optimism_summary["metric"].isin(["macro_f1", "qwk", "fatal_recall"])
    ]
    summary = {
        "version": VERSION,
        "completed_local": CREATED_LOCAL,
        "runtime_seconds": elapsed,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "master_seed": BOOTSTRAP_SEED,
        "evaluation_groups": len(protocol["evaluation_groups"]),
        "model_metric_interval_rows": int(len(interval_frame)),
        "paired_model_difference_rows": int(len(pairwise_frame)),
        "random_optimism_gap_rows": int(len(optimism_frame)),
        "fit_operations": 0,
        "threshold_changes": 0,
        "model_selection_operations": 0,
        "D11_input_audit": input_audit,
        "H2_temporal_macro_f1": {
            "raw_delta": float(macro["raw_delta_candidate_minus_reference"]),
            "ci_lower": float(macro["raw_delta_ci_lower"]),
            "ci_upper": float(macro["raw_delta_ci_upper"]),
            "point_exceeds_0_01": bool(macro["raw_delta_candidate_minus_reference"] >= 0.01),
            "ci_lower_above_zero": bool(macro["raw_delta_ci_lower"] > 0),
        },
        "H2_temporal_fatal_recall": {
            "raw_delta": float(fatal["raw_delta_candidate_minus_reference"]),
            "ci_lower": float(fatal["raw_delta_ci_lower"]),
            "ci_upper": float(fatal["raw_delta_ci_upper"]),
        },
        "H1_lightgbm_mean_optimism_gaps": {
            row.metric: float(row.mean_optimism_gap)
            for row in lightgbm_h1.itertuples(index=False)
        },
        "inference_boundary": (
            "Intervals condition on fixed fitted models. H1 across-seed summaries "
            "are descriptive because random partitions overlap."
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "thread_limit": 4,
        },
    }
    write_json(SUMMARY_FILE, summary)
    build_manifest()
    print(f"D12_BOOTSTRAP=PASS elapsed_seconds={elapsed:.2f}")
    print(f"D12_MANIFEST_SHA256={hash_file(MANIFEST_FILE)}")


def kernel_self_test() -> None:
    rng = np.random.default_rng(104729)
    target = np.repeat(np.arange(3), [1_000, 250, 50]).astype(np.int8)
    predicted = rng.integers(0, 3, size=len(target), dtype=np.int8)
    matrix = confusion_from_vectors(target, predicted[:, None])
    observed = classification_metrics(target, predicted)
    calculated = metrics_from_confusions(matrix)
    for metric in METRICS:
        if not np.isclose(observed[metric], calculated[metric][0], rtol=0, atol=1e-14):
            raise AssertionError(f"Metric kernel mismatch: {metric}")
    print("D12_KERNEL_SELF_TEST=PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze-protocol", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.freeze_protocol:
        freeze_protocol()
    elif args.run:
        run_analysis()
    else:
        kernel_self_test()


if __name__ == "__main__":
    main()
