"""D14 TreeSHAP feature-ranking stability analysis for frozen LightGBM models."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
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

import joblib
import lightgbm
import matplotlib
import numpy as np
import pandas as pd
import scipy
import shap
import sklearn
from scipy.stats import rankdata, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baseline_modeling import TARGET_CODES, TARGET_LABELS
from d10_tune_lightgbm import load_aligned_data, load_contracts
from d11_evaluate_frozen_models import (
    prepare_saved_features,
    validate_artifact_identity,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_DIR / "code"
CONFIG_DIR = PROJECT_DIR / "config"
RESULT_DIR = PROJECT_DIR / "results" / "d14"
STAGING_DIR = RESULT_DIR / ".staging"
SHAP_DIR = RESULT_DIR / "shap_values"
LOG_DIR = PROJECT_DIR / "logs"
FIGURE_DIR = PROJECT_DIR / "figures"
MODEL_DIR = PROJECT_DIR / "models" / "d10"

DATA_FILE = PROJECT_DIR / "data" / "processed" / "stats19_modeling_dataset.csv.gz"
ASSIGNMENTS_FILE = PROJECT_DIR / "data" / "processed" / "d6_split_assignments.csv.gz"
SCHEMA_FILE = CONFIG_DIR / "d5_dataset_schema.json"
D6_FILE = CONFIG_DIR / "d6_analysis_protocol.json"
D10_SELECTED_FILE = CONFIG_DIR / "d10_selected_lightgbm.json"
D13_PROTOCOL_FILE = CONFIG_DIR / "d13_positioning_protocol.json"
D13_DECISION_FILE = LOG_DIR / "d13_positioning_decision.json"
D13_MANIFEST_FILE = LOG_DIR / "d13_artifact_manifest.csv"

PROTOCOL_FILE = CONFIG_DIR / "d14_shap_protocol.json"
IMPORTANCE_FILE = RESULT_DIR / "d14_shap_importance.csv"
STABILITY_FILE = RESULT_DIR / "d14_rank_stability.csv"
STABILITY_SUMMARY_FILE = RESULT_DIR / "d14_rank_stability_summary.csv"
SAMPLE_FILE = RESULT_DIR / "d14_explanation_samples.csv.gz"
BOOTSTRAP_FILE = RESULT_DIR / "d14_rank_bootstrap_draws.npz"
AUDIT_FILE = RESULT_DIR / "d14_shap_additivity_audit.csv"
FIGURE_FILE = FIGURE_DIR / "d14_shap_rank_stability.png"
RUN_SUMMARY_FILE = LOG_DIR / "d14_shap_run_summary.json"
CHECKPOINT_FILE = LOG_DIR / "d14_shap_checkpoint.md"
MANIFEST_FILE = LOG_DIR / "d14_shap_artifact_manifest.csv"
TEST_SOURCE_FILE = CODE_DIR / "test_d14_shap.py"

VERSION = "D14_SHAP_V1"
CREATED_LOCAL = "2026-08-31"
THREAD_LIMIT = 4
EXPLANATION_ROWS = 10_000
SAMPLE_MASTER_SEED = 20_260_831
BOOTSTRAP_ITERATIONS = 2_000
BOOTSTRAP_MASTER_SEED = 20_260_828
BOOTSTRAP_BATCH_ROWS = 50
CI_ALPHA = 0.05
RANDOM_SEEDS = (1103, 2207, 3301, 4409, 5501)
FEATURE_SCOPES = (
    "overall_equal_class_mean",
    "Slight",
    "Serious",
    "Fatal",
)


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


def write_gzip_frame(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError(f"No rows generated for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.gz")
    frame.to_csv(
        temporary,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        lineterminator="\n",
    )
    temporary.replace(path)


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "shap": shap.__version__,
        "platform": platform.platform(),
    }


def derive_seed(master_seed: int, stream_key: str) -> int:
    payload = f"{master_seed}|{stream_key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def model_paths(split_slug: str) -> tuple[Path, Path]:
    return (
        MODEL_DIR / split_slug / "preprocessing_bundle.joblib",
        MODEL_DIR / split_slug / "lightgbm_weighted.joblib",
    )


def upstream_paths() -> list[Path]:
    paths = [
        DATA_FILE,
        ASSIGNMENTS_FILE,
        SCHEMA_FILE,
        D6_FILE,
        D10_SELECTED_FILE,
        D13_PROTOCOL_FILE,
        D13_DECISION_FILE,
        D13_MANIFEST_FILE,
        CODE_DIR / "baseline_modeling.py",
        CODE_DIR / "d10_tune_lightgbm.py",
        CODE_DIR / "d11_evaluate_frozen_models.py",
        Path(__file__),
        TEST_SOURCE_FILE,
        PROJECT_DIR / "requirements.txt",
    ]
    for split_slug in ["temporal", *(f"random_seed_{seed}" for seed in RANDOM_SEEDS)]:
        paths.extend(model_paths(split_slug))
    return paths


def build_protocol() -> dict[str, Any]:
    required = upstream_paths()
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing D14 inputs: " + "; ".join(missing))
    d6 = read_json(D6_FILE)
    selected = read_json(D10_SELECTED_FILE)
    d13 = read_json(D13_PROTOCOL_FILE)
    d13_decision = read_json(D13_DECISION_FILE)
    if d6.get("version") != "D6_V2":
        raise ValueError("D14 requires frozen D6_V2")
    if selected.get("selected_candidate_id") != "C03" or int(
        selected.get("selected_n_estimators", -1)
    ) != 1200:
        raise ValueError("D14 requires the frozen C03-1200 LightGBM")
    if d13.get("decision_rules", {}).get("H3", {}).get("status") != "PENDING_NOT_EVALUATED":
        raise ValueError("D14 requires the D13 H3-pending checkpoint")
    if d13_decision.get("H3", {}).get("status") != "PENDING_NOT_EVALUATED":
        raise ValueError("D13 decision record no longer has H3 pending")

    return {
        "version": VERSION,
        "status": "FROZEN_BEFORE_ANY_SHAP_IMPORTANCE_OR_RANKING_INSPECTION",
        "created_local": CREATED_LOCAL,
        "timing_disclosure": (
            "This protocol was frozen after D11-D13 predictive results were known, "
            "but before any SHAP importance values or feature rankings were inspected."
        ),
        "runtime_preflight": {
            "purpose": "select a feasible explanation sample size without inspecting rankings",
            "rows": 1000,
            "model": "random_seed_1103 frozen LightGBM",
            "cohort": "first 1000 rows from 2024 used only for shape/runtime checks",
            "treeexplainer_output_shape": [1000, 17, 3],
            "treeexplainer_elapsed_seconds": 8.63,
            "importance_or_ranking_inspected": False,
            "decision": "use a fixed 10,000-row class-stratified explanation sample per cohort",
        },
        "scientific_lock": {
            "frozen_model_fits_only": True,
            "new_model_fits_for_H3": 0,
            "test_based_model_selection": 0,
            "threshold_tuning": 0,
            "native_thread_limit": THREAD_LIMIT,
        },
        "upstream_sha256": {relative(path): hash_file(path) for path in required},
        "H3": {
            "statement": "Feature-importance rankings change under temporal distribution shift.",
            "primary_design": (
                "For each of five frozen random-reference LightGBM fits, compare its "
                "ranking on its own internal test cohort with its ranking on 2024. "
                "The fitted model is held fixed within each comparison."
            ),
            "primary_scope": "overall_equal_class_mean",
            "secondary_scopes": ["Slight", "Serious", "Fatal"],
            "ranking_metric": "Spearman correlation across the 17 frozen input features",
            "decision_boundary": (
                "No universal or preregistered practical margin exists for SHAP-rank "
                "correlation. Therefore D14 reports rho and its interval rather than "
                "declaring material instability from an arbitrary cutoff. rho=1 denotes "
                "identical ranks; rho<1 denotes an observed rank difference whose magnitude "
                "must be interpreted quantitatively."
            ),
            "across_seed_reporting": (
                "Report mean, sample SD and range of five point correlations descriptively; "
                "overlapping partitions are not treated as independent studies."
            ),
        },
        "explanation": {
            "algorithm": "TreeSHAP via shap.TreeExplainer for LightGBM raw multiclass scores",
            "feature_count": 17,
            "class_order": ["Slight", "Serious", "Fatal"],
            "class_specific_importance": "mean absolute SHAP over all sampled records for each output class",
            "overall_importance": "equal arithmetic mean of the three class-specific mean absolute SHAP values",
            "causal_boundary": "SHAP is interpreted as model contribution/association, never causation",
            "correlated_feature_boundary": (
                "Correlated or substitutable features may exchange rank; a rank shift alone "
                "does not establish a changed data-generating mechanism."
            ),
            "additivity_check": "SHAP values plus expected values must reconstruct raw class scores",
        },
        "sampling": {
            "rows_per_cohort": EXPLANATION_ROWS,
            "master_seed": SAMPLE_MASTER_SEED,
            "method": (
                "without-replacement stratified sample by true severity, with proportional "
                "integer allocation by largest remainder"
            ),
            "future_sample_rule": "one common 2024 sample is reused for all six fitted LightGBM models",
            "internal_sample_rule": "one deterministic sample from each frozen random internal test partition",
            "sample_keys": [
                "future_2024",
                *(f"random_seed_{seed}_internal" for seed in RANDOM_SEEDS),
            ],
        },
        "bootstrap": {
            "iterations": BOOTSTRAP_ITERATIONS,
            "master_seed": BOOTSTRAP_MASTER_SEED,
            "seed_timing_disclosure": (
                "The D6/D12 seed and iteration count are intentionally reused for consistency; "
                "their application to SHAP ranking was fixed here after D13 but before SHAP inspection."
            ),
            "resampling": (
                "sample explanation records with replacement within each observed true-severity "
                "stratum, retaining stratum counts"
            ),
            "comparison_rule": (
                "internal and 2024 cohorts are resampled independently; all frozen models on "
                "the common 2024 sample share the same 2024 bootstrap draws"
            ),
            "confidence_level": 0.95,
            "interval_method": "percentile",
            "quantile_method": "linear",
            "multiplicity_adjustment": "none; descriptive effect-size intervals",
        },
        "regional_holdout": {
            "status": "NOT_PART_OF_H3_AND_NOT_RETROACTIVELY_DEFINED",
            "reason": (
                "No concrete police-force holdout was frozen before model outcomes were known; "
                "any later regional analysis must be labelled post-hoc exploratory."
            ),
        },
        "outputs": {
            "importance": relative(IMPORTANCE_FILE),
            "rank_stability": relative(STABILITY_FILE),
            "rank_summary": relative(STABILITY_SUMMARY_FILE),
            "samples": relative(SAMPLE_FILE),
            "bootstrap_draws": relative(BOOTSTRAP_FILE),
            "additivity_audit": relative(AUDIT_FILE),
            "figure": relative(FIGURE_FILE),
            "run_summary": relative(RUN_SUMMARY_FILE),
            "checkpoint": relative(CHECKPOINT_FILE),
            "manifest": relative(MANIFEST_FILE),
        },
        "runtime": {"package_versions_at_freeze": package_versions()},
    }


def freeze_protocol() -> None:
    payload = build_protocol()
    if PROTOCOL_FILE.exists():
        existing = read_json(PROTOCOL_FILE)
        if existing != payload:
            raise RuntimeError("Existing D14 SHAP protocol differs; refusing overwrite")
        print("D14 SHAP protocol already matches the frozen specification.", flush=True)
    else:
        write_json(PROTOCOL_FILE, payload)
        print("D14 SHAP protocol frozen before SHAP ranking inspection.", flush=True)
    print("D14_SHAP_PROTOCOL_SHA256=", hash_file(PROTOCOL_FILE), flush=True)


def require_protocol() -> dict[str, Any]:
    if not PROTOCOL_FILE.is_file():
        raise FileNotFoundError("Run --freeze before D14 SHAP analysis")
    protocol = read_json(PROTOCOL_FILE)
    if protocol.get("version") != VERSION or protocol.get("status") != (
        "FROZEN_BEFORE_ANY_SHAP_IMPORTANCE_OR_RANKING_INSPECTION"
    ):
        raise ValueError("Unexpected D14 SHAP protocol state")
    if build_protocol() != protocol:
        raise ValueError("D14 SHAP inputs or implementation changed after protocol freeze")
    return protocol


def proportional_allocation(labels: np.ndarray, rows: int) -> dict[int, int]:
    labels = np.asarray(labels, dtype=int)
    if rows <= 0 or rows > len(labels):
        raise ValueError("Invalid sample size")
    counts = {code: int(np.sum(labels == code)) for code in TARGET_CODES}
    if any(counts[code] == 0 for code in TARGET_CODES):
        raise ValueError("Every severity class must be represented")
    exact = {code: rows * counts[code] / len(labels) for code in TARGET_CODES}
    allocation = {code: int(np.floor(exact[code])) for code in TARGET_CODES}
    remainder = rows - sum(allocation.values())
    order = sorted(TARGET_CODES, key=lambda code: (-(exact[code] - allocation[code]), code))
    for code in order[:remainder]:
        allocation[code] += 1
    if sum(allocation.values()) != rows or any(allocation[c] > counts[c] for c in TARGET_CODES):
        raise AssertionError("Invalid proportional allocation")
    return allocation


def stratified_sample_positions(
    positions: np.ndarray,
    target: pd.Series,
    rows: int,
    seed: int,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.int64)
    labels = target.iloc[positions].to_numpy(dtype=np.int8)
    allocation = proportional_allocation(labels, rows)
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for code in TARGET_CODES:
        candidates = positions[labels == code]
        chosen = rng.choice(candidates, size=allocation[code], replace=False)
        selected.append(np.asarray(chosen, dtype=np.int64))
    output = np.sort(np.concatenate(selected))
    if len(output) != rows or len(np.unique(output)) != rows:
        raise AssertionError("Explanation sample contains duplicate rows")
    return output


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    schema, d6, _, _ = load_contracts()
    features, target, metadata, assignments = load_aligned_data(schema, d6)
    if len(features) != 743_646:
        raise AssertionError("Unexpected frozen data row count")
    model_ids = metadata["meta_collision_index"].astype("string").reset_index(drop=True)
    split_ids = assignments["meta_collision_index"].astype("string").reset_index(drop=True)
    if not model_ids.equals(split_ids):
        raise AssertionError("Model data and split assignments are not aligned")
    return features, target.astype("int8"), metadata, assignments


def build_samples(
    target: pd.Series,
    metadata: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    sample_positions: dict[str, np.ndarray] = {}
    future_positions = np.flatnonzero(
        assignments["meta_collision_year"].astype(int).eq(2024).to_numpy()
    )
    if len(future_positions) != 100_927:
        raise AssertionError("Unexpected 2024 population")
    sample_positions["future_2024"] = stratified_sample_positions(
        future_positions,
        target,
        EXPLANATION_ROWS,
        derive_seed(SAMPLE_MASTER_SEED, "future_2024"),
    )
    for seed in RANDOM_SEEDS:
        key = f"random_seed_{seed}_internal"
        role = assignments[f"random_role_seed_{seed}"].astype("string")
        positions = np.flatnonzero(role.eq("test").to_numpy())
        if len(positions) != 96_408:
            raise AssertionError(f"Unexpected random test population: {seed}")
        if 2024 in set(assignments.iloc[positions]["meta_collision_year"].astype(int)):
            raise AssertionError("Random internal sample contains 2024")
        sample_positions[key] = stratified_sample_positions(
            positions,
            target,
            EXPLANATION_ROWS,
            derive_seed(SAMPLE_MASTER_SEED, key),
        )

    rows: list[dict[str, Any]] = []
    years = assignments["meta_collision_year"].astype(int).to_numpy()
    collision_ids = metadata["meta_collision_index"].astype("string").to_numpy()
    target_values = target.to_numpy(dtype=np.int8)
    for key, positions in sample_positions.items():
        stream_seed = derive_seed(SAMPLE_MASTER_SEED, key)
        for position in positions:
            code = int(target_values[position])
            rows.append(
                {
                    "sample_key": key,
                    "row_position": int(position),
                    "collision_index": str(collision_ids[position]),
                    "collision_year": int(years[position]),
                    "target_code": code,
                    "target_label": TARGET_LABELS[code],
                    "selection_seed": stream_seed,
                }
            )
    frame = pd.DataFrame(rows)
    if len(frame) != EXPLANATION_ROWS * 6:
        raise AssertionError("Unexpected explanation sample manifest size")
    return sample_positions, frame


def load_frozen_model(split_slug: str, seed: str) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle_path, model_path = model_paths(split_slug)
    bundle = joblib.load(bundle_path)
    artifact = joblib.load(model_path)
    protocol = "temporal" if split_slug == "temporal" else "random_reference"
    validate_artifact_identity(
        "D10",
        "lightgbm_weighted",
        {"protocol": protocol, "seed": seed},
        bundle,
        artifact,
    )
    return bundle, artifact


def explain(
    features: pd.DataFrame,
    positions: np.ndarray,
    bundle: dict[str, Any],
    artifact: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    prepared, unseen_counts = prepare_saved_features(
        "D10", features.iloc[positions].copy(), bundle
    )
    estimator = artifact["estimator"]
    started = time.perf_counter()
    explainer = shap.TreeExplainer(
        estimator,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )
    values = np.asarray(
        explainer.shap_values(prepared, check_additivity=False), dtype=np.float64
    )
    elapsed = time.perf_counter() - started
    expected_value = np.asarray(explainer.expected_value, dtype=np.float64)
    expected_shape = (len(positions), len(bundle["feature_columns"]), len(TARGET_CODES))
    if values.shape != expected_shape:
        raise AssertionError(f"Unexpected TreeSHAP shape: {values.shape}")
    if expected_value.shape != (len(TARGET_CODES),):
        raise AssertionError("Unexpected TreeSHAP expected-value shape")
    if not np.isfinite(values).all() or not np.isfinite(expected_value).all():
        raise AssertionError("TreeSHAP produced non-finite values")

    raw_scores = np.asarray(
        estimator.booster_.predict(prepared, raw_score=True), dtype=np.float64
    )
    reconstructed = values.sum(axis=1) + expected_value[None, :]
    max_abs_error = float(np.max(np.abs(reconstructed - raw_scores)))
    if max_abs_error > 1e-6:
        raise AssertionError(f"TreeSHAP additivity error too large: {max_abs_error}")
    audit = {
        "rows": int(len(positions)),
        "features": int(values.shape[1]),
        "classes": int(values.shape[2]),
        "runtime_seconds": float(elapsed),
        "max_abs_additivity_error": max_abs_error,
        "unseen_category_total": int(sum(unseen_counts.values())),
    }
    return values, expected_value, audit


def importance_scopes(values: np.ndarray) -> dict[str, np.ndarray]:
    class_importance = np.mean(np.abs(values), axis=0)
    return {
        "overall_equal_class_mean": class_importance.mean(axis=1),
        "Slight": class_importance[:, 0],
        "Serious": class_importance[:, 1],
        "Fatal": class_importance[:, 2],
    }


def importance_rows(
    *,
    values: np.ndarray,
    feature_names: list[str],
    split_slug: str,
    seed: str,
    explanation_cohort: str,
    sample_key: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope, importance in importance_scopes(values).items():
        ranks = rankdata(-importance, method="average")
        for index, feature in enumerate(feature_names):
            rows.append(
                {
                    "split_slug": split_slug,
                    "seed": seed,
                    "model": "lightgbm_weighted",
                    "explanation_cohort": explanation_cohort,
                    "sample_key": sample_key,
                    "output_scope": scope,
                    "feature": feature,
                    "mean_abs_shap": float(importance[index]),
                    "importance_rank": float(ranks[index]),
                    "sample_rows": int(values.shape[0]),
                }
            )
    return rows


def bootstrap_mean_values(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    stream_key: str,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if values.shape[0] != len(labels):
        raise ValueError("Bootstrap labels and SHAP rows differ")
    original_shape = values.shape[1:]
    flattened = values.reshape(len(values), -1)
    output = np.empty((BOOTSTRAP_ITERATIONS, flattened.shape[1]), dtype=np.float64)
    rng = np.random.default_rng(derive_seed(BOOTSTRAP_MASTER_SEED, stream_key))
    strata = {code: np.flatnonzero(labels == code) for code in TARGET_CODES}
    if any(len(indices) == 0 for indices in strata.values()):
        raise ValueError("Bootstrap cohort lacks a severity class")

    for start in range(0, BOOTSTRAP_ITERATIONS, BOOTSTRAP_BATCH_ROWS):
        stop = min(start + BOOTSTRAP_BATCH_ROWS, BOOTSTRAP_ITERATIONS)
        batch = stop - start
        sums = np.zeros((batch, flattened.shape[1]), dtype=np.float64)
        for code in TARGET_CODES:
            indices = strata[code]
            n_stratum = len(indices)
            draws = rng.integers(
                0,
                n_stratum,
                size=(batch, n_stratum),
                dtype=np.int32,
            )
            counts = np.empty((batch, n_stratum), dtype=np.float64)
            for row_index in range(batch):
                counts[row_index] = np.bincount(
                    draws[row_index], minlength=n_stratum
                )
            sums += counts @ flattened[indices]
        output[start:stop] = sums / len(values)
    return output.reshape((BOOTSTRAP_ITERATIONS, *original_shape))


def scope_from_bootstrap(means: np.ndarray, scope: str) -> np.ndarray:
    if scope == "overall_equal_class_mean":
        return means.mean(axis=-1)
    class_index = {"Slight": 0, "Serious": 1, "Fatal": 2}[scope]
    return means[..., class_index]


def spearman_rows(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Spearman input arrays must have equal 2D shape")
    first_ranks = rankdata(first, axis=1, method="average")
    second_ranks = rankdata(second, axis=1, method="average")
    first_centered = first_ranks - first_ranks.mean(axis=1, keepdims=True)
    second_centered = second_ranks - second_ranks.mean(axis=1, keepdims=True)
    numerator = np.sum(first_centered * second_centered, axis=1)
    denominator = np.sqrt(
        np.sum(first_centered**2, axis=1) * np.sum(second_centered**2, axis=1)
    )
    if np.any(denominator == 0):
        raise AssertionError("A bootstrap feature ranking is constant")
    return numerator / denominator


def quantile_interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(
        values,
        [CI_ALPHA / 2, 1 - CI_ALPHA / 2],
        method="linear",
    )
    return float(lower), float(upper)


def plot_stability(frame: pd.DataFrame, path: Path) -> None:
    colors = {
        "overall_equal_class_mean": "#214761",
        "Slight": "#2A7F62",
        "Serious": "#C17D11",
        "Fatal": "#A63D40",
    }
    titles = {
        "overall_equal_class_mean": "Overall",
        "Slight": "Slight output",
        "Serious": "Serious output",
        "Fatal": "Fatal output",
    }
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 4.3), sharey=True)
    y = np.arange(len(RANDOM_SEEDS))
    for axis, scope in zip(axes, FEATURE_SCOPES, strict=True):
        subset = frame[frame["output_scope"].eq(scope)].sort_values("seed")
        point = subset["spearman_rho"].to_numpy(dtype=float)
        lower = subset["ci_lower"].to_numpy(dtype=float)
        upper = subset["ci_upper"].to_numpy(dtype=float)
        axis.errorbar(
            point,
            y,
            xerr=np.vstack([point - lower, upper - point]),
            fmt="o",
            color=colors[scope],
            ecolor="#65727A",
            elinewidth=1.1,
            capsize=2.5,
            markersize=5,
        )
        axis.axvline(1.0, color="#8A8A8A", linewidth=0.8, linestyle="--")
        axis.axvline(0.0, color="#D0D0D0", linewidth=0.7)
        axis.set_xlim(-1.02, 1.02)
        axis.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        axis.grid(axis="x", color="#E5E8EA", linewidth=0.7)
        axis.set_title(titles[scope], fontsize=10.5)
        axis.set_xlabel("Spearman rho", fontsize=9.5)
        axis.tick_params(labelsize=8.5)
    axes[0].set_yticks(y, labels=[str(seed) for seed in RANDOM_SEEDS])
    axes[0].set_ylabel("Frozen random-split seed", fontsize=9.5)
    fig.suptitle(
        "Feature-ranking agreement: internal test versus 2024\n(same fitted LightGBM within each comparison)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", format="png")
    plt.close(fig)


def assert_outputs_absent() -> None:
    outputs = [
        IMPORTANCE_FILE,
        STABILITY_FILE,
        STABILITY_SUMMARY_FILE,
        SAMPLE_FILE,
        BOOTSTRAP_FILE,
        AUDIT_FILE,
        FIGURE_FILE,
        RUN_SUMMARY_FILE,
        CHECKPOINT_FILE,
        MANIFEST_FILE,
    ]
    existing = [relative(path) for path in outputs if path.exists()]
    if SHAP_DIR.exists() and any(SHAP_DIR.iterdir()):
        existing.append(relative(SHAP_DIR))
    if existing:
        raise RuntimeError("D14 SHAP outputs already exist; refusing overwrite: " + "; ".join(existing))


def run_analysis() -> None:
    protocol = require_protocol()
    assert_outputs_absent()
    started = time.perf_counter()
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    (STAGING_DIR / "shap_values").mkdir(parents=True, exist_ok=False)
    print("D14 SHAP: loading frozen data and deterministic samples...", flush=True)
    features, target, metadata, assignments = load_data()
    sample_positions, sample_frame = build_samples(target, metadata, assignments)
    feature_names = features.columns.tolist()
    target_values = target.to_numpy(dtype=np.int8)

    all_importance_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    internal_values: dict[int, np.ndarray] = {}
    future_values: dict[int, np.ndarray] = {}

    for seed in RANDOM_SEEDS:
        split_slug = f"random_seed_{seed}"
        bundle, artifact = load_frozen_model(split_slug, str(seed))
        for cohort, sample_key in (
            ("random_internal_test", f"random_seed_{seed}_internal"),
            ("future_2024", "future_2024"),
        ):
            positions = sample_positions[sample_key]
            print(f"D14 SHAP: explaining {split_slug} on {cohort} ({len(positions):,} rows)...", flush=True)
            values, expected_value, audit = explain(
                features, positions, bundle, artifact
            )
            if cohort == "random_internal_test":
                internal_values[seed] = values
            else:
                future_values[seed] = values
            all_importance_rows.extend(
                importance_rows(
                    values=values,
                    feature_names=feature_names,
                    split_slug=split_slug,
                    seed=str(seed),
                    explanation_cohort=cohort,
                    sample_key=sample_key,
                )
            )
            audit_rows.append(
                {
                    "split_slug": split_slug,
                    "seed": str(seed),
                    "explanation_cohort": cohort,
                    "sample_key": sample_key,
                    **audit,
                }
            )
            write_npz(
                STAGING_DIR / "shap_values" / f"{split_slug}__{cohort}.npz",
                {
                    "shap_values": values,
                    "expected_value": expected_value,
                    "row_positions": positions.astype(np.int32),
                    "y_true": target_values[positions],
                    "feature_names": np.asarray(feature_names, dtype="U64"),
                },
            )
            del values
            gc.collect()
        del artifact, bundle
        gc.collect()

    temporal_bundle, temporal_artifact = load_frozen_model("temporal", "year_based")
    future_positions = sample_positions["future_2024"]
    print("D14 SHAP: explaining strict temporal model on 2024 (descriptive appendix ranking)...", flush=True)
    temporal_values, temporal_expected, temporal_audit = explain(
        features, future_positions, temporal_bundle, temporal_artifact
    )
    all_importance_rows.extend(
        importance_rows(
            values=temporal_values,
            feature_names=feature_names,
            split_slug="temporal",
            seed="year_based",
            explanation_cohort="future_2024",
            sample_key="future_2024",
        )
    )
    audit_rows.append(
        {
            "split_slug": "temporal",
            "seed": "year_based",
            "explanation_cohort": "future_2024",
            "sample_key": "future_2024",
            **temporal_audit,
        }
    )
    write_npz(
        STAGING_DIR / "shap_values" / "temporal__future_2024.npz",
        {
            "shap_values": temporal_values,
            "expected_value": temporal_expected,
            "row_positions": future_positions.astype(np.int32),
            "y_true": target_values[future_positions],
            "feature_names": np.asarray(feature_names, dtype="U64"),
        },
    )
    del temporal_values, temporal_bundle, temporal_artifact
    gc.collect()

    print("D14 SHAP: running 2,000 class-stratified bootstrap iterations...", flush=True)
    future_stack = np.stack([np.abs(future_values[seed]) for seed in RANDOM_SEEDS], axis=1)
    future_labels = target_values[future_positions]
    future_bootstrap = bootstrap_mean_values(
        future_stack,
        future_labels,
        stream_key="future_2024_common_five_models",
    )
    stability_rows: list[dict[str, Any]] = []
    draw_arrays: dict[str, np.ndarray] = {}
    for model_index, seed in enumerate(RANDOM_SEEDS):
        internal_positions = sample_positions[f"random_seed_{seed}_internal"]
        internal_labels = target_values[internal_positions]
        internal_bootstrap = bootstrap_mean_values(
            np.abs(internal_values[seed]),
            internal_labels,
            stream_key=f"random_seed_{seed}_internal",
        )
        point_internal = importance_scopes(internal_values[seed])
        point_future = importance_scopes(future_values[seed])
        for scope in FEATURE_SCOPES:
            point_rho = float(
                spearmanr(point_internal[scope], point_future[scope]).statistic
            )
            internal_scope = scope_from_bootstrap(internal_bootstrap, scope)
            future_scope = scope_from_bootstrap(
                future_bootstrap[:, model_index, :, :], scope
            )
            draws = spearman_rows(internal_scope, future_scope)
            lower, upper = quantile_interval(draws)
            key = f"seed_{seed}__{scope}"
            draw_arrays[key] = draws.astype(np.float64)
            stability_rows.append(
                {
                    "seed": seed,
                    "model": "lightgbm_weighted",
                    "comparison": "same_model_random_internal_vs_2024",
                    "output_scope": scope,
                    "features_ranked": len(feature_names),
                    "internal_sample_rows": EXPLANATION_ROWS,
                    "future_sample_rows": EXPLANATION_ROWS,
                    "spearman_rho": point_rho,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                }
            )
        print(f"D14 SHAP: bootstrap complete for seed {seed}.", flush=True)
        del internal_bootstrap
        gc.collect()

    importance_frame = pd.DataFrame(all_importance_rows)
    stability_frame = pd.DataFrame(stability_rows)
    summary_rows: list[dict[str, Any]] = []
    for scope in FEATURE_SCOPES:
        points = stability_frame.loc[
            stability_frame["output_scope"].eq(scope), "spearman_rho"
        ].to_numpy(dtype=float)
        summary_rows.append(
            {
                "output_scope": scope,
                "seed_count": len(points),
                "mean_spearman_rho": float(points.mean()),
                "sd_spearman_rho": float(points.std(ddof=1)),
                "min_spearman_rho": float(points.min()),
                "max_spearman_rho": float(points.max()),
                "inference": "descriptive_across_overlapping_random_splits",
            }
        )
    summary_frame = pd.DataFrame(summary_rows)
    audit_frame = pd.DataFrame(audit_rows)

    write_frame(STAGING_DIR / IMPORTANCE_FILE.name, importance_frame)
    write_frame(STAGING_DIR / STABILITY_FILE.name, stability_frame)
    write_frame(STAGING_DIR / STABILITY_SUMMARY_FILE.name, summary_frame)
    write_gzip_frame(STAGING_DIR / SAMPLE_FILE.name, sample_frame)
    write_npz(STAGING_DIR / BOOTSTRAP_FILE.name, draw_arrays)
    write_frame(STAGING_DIR / AUDIT_FILE.name, audit_frame)
    plot_stability(stability_frame, STAGING_DIR / FIGURE_FILE.name)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    for path in STAGING_DIR.iterdir():
        if path.name == "shap_values":
            for raw_file in path.iterdir():
                raw_file.replace(SHAP_DIR / raw_file.name)
        else:
            destination = FIGURE_FILE if path.name == FIGURE_FILE.name else RESULT_DIR / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.replace(destination)
    shutil.rmtree(STAGING_DIR)

    overall = stability_frame[
        stability_frame["output_scope"].eq("overall_equal_class_mean")
    ]
    all_identical = bool(np.allclose(overall["spearman_rho"], 1.0, rtol=0, atol=1e-15))
    h3_status = (
        "NO_OBSERVED_OVERALL_RANK_CHANGE"
        if all_identical
        else "RANK_CHANGE_QUANTIFIED_NO_BINARY_MATERIALITY_THRESHOLD"
    )
    runtime = time.perf_counter() - started
    run_summary = {
        "version": VERSION,
        "status": "D14_SHAP_COMPLETE",
        "protocol_sha256": hash_file(PROTOCOL_FILE),
        "completed_local": CREATED_LOCAL,
        "model_fits": 0,
        "frozen_models_explained": 6,
        "model_cohort_explanations": 11,
        "explanation_rows_per_cohort": EXPLANATION_ROWS,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "H3_status": h3_status,
        "overall_rho_mean": float(overall["spearman_rho"].mean()),
        "overall_rho_sd": float(overall["spearman_rho"].std(ddof=1)),
        "overall_rho_range": [
            float(overall["spearman_rho"].min()),
            float(overall["spearman_rho"].max()),
        ],
        "causal_claim_allowed": False,
        "runtime_seconds": float(runtime),
    }
    write_json(RUN_SUMMARY_FILE, run_summary)
    checkpoint = (
        "# D14 SHAP stability checkpoint\n\n"
        "Status: **D14_SHAP_COMPLETE**\n\n"
        f"- H3 status: **{h3_status}**.\n"
        f"- Overall ranking rho across five frozen fits: mean {run_summary['overall_rho_mean']:.6f}, "
        f"SD {run_summary['overall_rho_sd']:.6f}, range "
        f"[{run_summary['overall_rho_range'][0]:.6f}, {run_summary['overall_rho_range'][1]:.6f}].\n"
        "- Seed-specific percentile intervals are reported in results/d14/d14_rank_stability.csv.\n"
        "- No arbitrary material-instability cutoff was introduced after observing predictive results.\n"
        "- SHAP values describe fitted-model contributions/associations, not causal effects.\n"
        "- Regional holdout remains unexecuted because no concrete region was frozen before outcomes were known.\n"
    )
    CHECKPOINT_FILE.write_text(checkpoint, encoding="utf-8")

    artifact_paths = [
        PROTOCOL_FILE,
        IMPORTANCE_FILE,
        STABILITY_FILE,
        STABILITY_SUMMARY_FILE,
        SAMPLE_FILE,
        BOOTSTRAP_FILE,
        AUDIT_FILE,
        FIGURE_FILE,
        RUN_SUMMARY_FILE,
        CHECKPOINT_FILE,
        Path(__file__),
        TEST_SOURCE_FILE,
        *sorted(SHAP_DIR.glob("*.npz")),
    ]
    manifest = pd.DataFrame(
        [
            {
                "relative_path": relative(path),
                "bytes": int(path.stat().st_size),
                "sha256": hash_file(path),
            }
            for path in artifact_paths
        ]
    )
    write_frame(MANIFEST_FILE, manifest)
    print(f"D14 SHAP complete in {runtime / 60:.1f} minutes.", flush=True)
    print(f"H3_STATUS={h3_status}", flush=True)
    print(f"D14_SHAP_MANIFEST_SHA256={hash_file(MANIFEST_FILE)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", help="freeze D14 protocol")
    parser.add_argument("--run", action="store_true", help="run frozen D14 SHAP analysis")
    args = parser.parse_args()
    if args.freeze == args.run:
        parser.error("choose exactly one of --freeze or --run")
    if args.freeze:
        freeze_protocol()
    else:
        run_analysis()


if __name__ == "__main__":
    main()
