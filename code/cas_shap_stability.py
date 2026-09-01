"""Compute frozen CAS H3 LightGBM SHAP rank stability."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from cas_modeling_common import (
    PROJECT_DIR,
    RANDOM_SEEDS,
    TARGET_CODES,
    THREADS,
    build_artifact_manifest,
    hash_file,
    load_aligned_data,
    load_contracts,
    prepare_features,
    relative,
    write_csv,
    write_dataframe_gzip,
    write_json_atomic,
)


VERSION = "CAS_SHAP_V1"
PROTOCOL_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_post_analysis_protocol.json"
)
LIGHTGBM_FROZEN_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_lightgbm_models_frozen.json"
)
MODEL_DIR = PROJECT_DIR / "models" / "cas_lightgbm"
RESULT_DIR = PROJECT_DIR / "results" / "cas_post_analysis"
STABILITY_FILE = RESULT_DIR / "h3_shap_stability.csv"
IMPORTANCE_FILE = RESULT_DIR / "h3_shap_importance.csv"
CLASS_DIRECTION_FILE = RESULT_DIR / "h3_shap_class_direction.csv"
SAMPLE_FILE = RESULT_DIR / "h3_sample_assignments.csv.gz"
LOG_DIR = PROJECT_DIR / "logs" / "cas"
SAMPLE_AUDIT_FILE = LOG_DIR / "cas_h3_sample_audit.csv"
CHECKPOINT_FILE = LOG_DIR / "cas_h3_shap_checkpoint.md"
MANIFEST_FILE = LOG_DIR / "cas_h3_shap_artifact_manifest.csv"
COMPLETE_FILE = PROJECT_DIR / "config" / "cas" / "cas_h3_complete.json"
RAW_SHAP_FILE = RESULT_DIR / "h3_shap_contributions.npy"
BOOTSTRAP_DRAWS_FILE = RESULT_DIR / "h3_shap_bootstrap_draws.csv.gz"
TEST_FILE = PROJECT_DIR / "tests" / "test_cas_shap.py"
MODEL_NAME = "lightgbm_weighted"
CLASS_LABELS = {0: "Minor", 1: "Serious", 2: "Fatal"}


def load_protocol() -> dict[str, object]:
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    if protocol.get("version") != "CAS_POST_ANALYSIS_V1":
        raise ValueError("Unexpected CAS post-analysis protocol")
    if protocol.get("status") != (
        "OPERATIONAL_SUPPLEMENT_FROZEN_BEFORE_BOOTSTRAP_OR_SHAP_RESULTS"
    ):
        raise ValueError("CAS H3 protocol is not frozen")
    checks = {
        LIGHTGBM_FROZEN_FILE: protocol["upstream"][
            "lightgbm_models_frozen_sha256"
        ]
    }
    for path, expected in checks.items():
        if hash_file(path) != expected:
            raise ValueError(f"CAS H3 upstream changed: {path.name}")
    return protocol


def matched_positions(
    *,
    assignments: pd.DataFrame,
    target: pd.Series,
    seed: int,
    base_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    role_column = f"random_role_seed_{seed}"
    internal = np.flatnonzero(assignments[role_column].eq("test").to_numpy())
    future_pool = np.flatnonzero(
        assignments[role_column].eq("locked_temporal_test").to_numpy()
    )
    if len(internal) != 4_887 or len(future_pool) != 10_542:
        raise AssertionError("CAS H3 cohort size changed")
    if set(assignments.iloc[future_pool]["meta_crash_year"].astype(int)) != {2025}:
        raise AssertionError("CAS H3 future pool is not 2025")
    internal_counts = {
        code: int(np.sum(target.iloc[internal].to_numpy(dtype=int) == code))
        for code in TARGET_CODES
    }
    rng = np.random.default_rng(base_seed + seed)
    selected: list[np.ndarray] = []
    future_targets = target.iloc[future_pool].to_numpy(dtype=int)
    for code in TARGET_CODES:
        candidates = future_pool[future_targets == code]
        required = internal_counts[code]
        if len(candidates) < required:
            raise ValueError(
                f"CAS H3 future class {code} has fewer than {required} rows"
            )
        selected.append(
            np.sort(rng.choice(candidates, size=required, replace=False))
        )
    future = np.sort(np.concatenate(selected))
    if len(future) != len(internal):
        raise AssertionError("CAS H3 matched sample size differs")
    future_counts = {
        code: int(np.sum(target.iloc[future].to_numpy(dtype=int) == code))
        for code in TARGET_CODES
    }
    if future_counts != internal_counts:
        raise AssertionError("CAS H3 target counts were not exactly matched")
    return internal, future, internal_counts


def native_frame(
    features: pd.DataFrame,
    positions: np.ndarray,
    bundle: dict[str, object],
) -> pd.DataFrame:
    prepared = prepare_features(
        features.iloc[positions],
        feature_columns=bundle["feature_columns"],
        categorical_columns=bundle["categorical_columns"],
        numeric_columns=bundle["numeric_columns"],
        category_vocabulary=bundle["category_vocabulary"],
    )
    return prepared.frame


def shap_contributions(
    estimator: object,
    frame: pd.DataFrame,
    feature_count: int,
) -> np.ndarray:
    booster = estimator.booster_
    raw = np.asarray(
        booster.predict(
            frame,
            pred_contrib=True,
            num_threads=THREADS,
        ),
        dtype=float,
    )
    raw_scores = np.asarray(
        booster.predict(
            frame,
            raw_score=True,
            num_threads=THREADS,
        ),
        dtype=float,
    )
    if raw.ndim == 3:
        if raw.shape == (len(frame), 3, feature_count + 1):
            shaped = raw
        elif raw.shape == (len(frame), feature_count + 1, 3):
            shaped = raw.transpose(0, 2, 1)
        else:
            raise ValueError(f"Unexpected three-dimensional SHAP shape: {raw.shape}")
    elif raw.ndim == 2 and raw.shape[1] == 3 * (feature_count + 1):
        class_major = raw.reshape(len(frame), 3, feature_count + 1)
        feature_major = raw.reshape(len(frame), feature_count + 1, 3).transpose(
            0, 2, 1
        )
        if np.allclose(
            class_major.sum(axis=2), raw_scores, rtol=1e-6, atol=1e-6
        ):
            shaped = class_major
        elif np.allclose(
            feature_major.sum(axis=2), raw_scores, rtol=1e-6, atol=1e-6
        ):
            shaped = feature_major
        else:
            raise AssertionError("CAS SHAP contributions do not reconstruct logits")
    else:
        raise ValueError(f"Unexpected CAS SHAP output shape: {raw.shape}")
    if not np.allclose(
        shaped.sum(axis=2), raw_scores, rtol=1e-6, atol=1e-6
    ):
        raise AssertionError("CAS SHAP values fail raw-score reconstruction")
    contributions = shaped[:, :, :feature_count]
    if not np.isfinite(contributions).all():
        raise AssertionError("CAS SHAP contributions contain non-finite values")
    return contributions


def hash_positions(positions: np.ndarray) -> str:
    canonical = np.asarray(positions, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def load_frozen_model(
    seed: int,
    frozen: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], str, str]:
    split_dir = MODEL_DIR / f"random_seed_{seed}"
    model_path = split_dir / f"{MODEL_NAME}.joblib"
    bundle_path = split_dir / "preprocessing_bundle.joblib"
    expected = {
        entry["file"]: entry["sha256"]
        for entry in frozen["model_files"]
    }
    model_key = relative(model_path)
    bundle_key = relative(bundle_path)
    if model_key not in expected or bundle_key not in expected:
        raise ValueError(f"Frozen CAS model manifest lacks seed {seed}")
    model_sha = hash_file(model_path)
    bundle_sha = hash_file(bundle_path)
    if model_sha != expected[model_key] or bundle_sha != expected[bundle_key]:
        raise ValueError(f"Frozen CAS model artifacts changed for seed {seed}")

    artifact = joblib.load(model_path)
    bundle = joblib.load(bundle_path)
    if artifact.get("model_name") != MODEL_NAME:
        raise ValueError(f"Unexpected CAS model for seed {seed}")
    if str(artifact.get("seed")) != str(seed):
        raise ValueError(f"CAS model seed metadata differs for seed {seed}")
    if str(bundle.get("seed")) != str(seed):
        raise ValueError(f"CAS bundle seed metadata differs for seed {seed}")
    if artifact.get("target_codes") != list(TARGET_CODES):
        raise ValueError(f"CAS model target order differs for seed {seed}")
    return artifact, bundle, model_sha, bundle_sha


def write_numpy_atomic(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    temporary.replace(path)


def bootstrap_mean_importance(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    iterations: int,
    rng: np.random.Generator,
    batch_size: int = 50,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if values.ndim != 2 or values.shape[0] != len(labels):
        raise ValueError("CAS SHAP bootstrap input dimensions differ")
    output = np.empty((iterations, values.shape[1]), dtype=np.float64)
    strata = {code: np.flatnonzero(labels == code) for code in TARGET_CODES}
    if any(len(indices) == 0 for indices in strata.values()):
        raise ValueError("CAS SHAP cohort lacks a severity class")

    for start in range(0, iterations, batch_size):
        stop = min(start + batch_size, iterations)
        batch = stop - start
        sums = np.zeros((batch, values.shape[1]), dtype=np.float64)
        for code in TARGET_CODES:
            indices = strata[code]
            count = len(indices)
            weights = rng.multinomial(
                count,
                np.full(count, 1.0 / count, dtype=np.float64),
                size=batch,
            )
            sums += weights @ values[indices]
        output[start:stop] = sums / len(values)
    return output


def spearman_rows(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("CAS SHAP Spearman arrays must have equal 2D shape")
    first_ranks = rankdata(first, axis=1, method="average")
    second_ranks = rankdata(second, axis=1, method="average")
    first_centered = first_ranks - first_ranks.mean(axis=1, keepdims=True)
    second_centered = second_ranks - second_ranks.mean(axis=1, keepdims=True)
    numerator = np.sum(first_centered * second_centered, axis=1)
    denominator = np.sqrt(
        np.sum(first_centered**2, axis=1)
        * np.sum(second_centered**2, axis=1)
    )
    if np.any(denominator == 0):
        raise AssertionError("A CAS SHAP bootstrap feature ranking is constant")
    return numerator / denominator


def importance_rows(
    *,
    contributions: np.ndarray,
    feature_names: list[str],
    seed: int,
    cohort: str,
) -> list[dict[str, object]]:
    importance = np.abs(contributions).mean(axis=(0, 1))
    ranks = rankdata(-importance, method="average")
    return [
        {
            "seed": seed,
            "model": MODEL_NAME,
            "explanation_cohort": cohort,
            "aggregation": "mean_absolute_over_records_and_three_classes",
            "feature": feature,
            "mean_abs_shap": float(importance[index]),
            "importance_rank": float(ranks[index]),
            "sample_rows": int(contributions.shape[0]),
        }
        for index, feature in enumerate(feature_names)
    ]


def class_direction_rows(
    *,
    contributions: np.ndarray,
    feature_names: list[str],
    seed: int,
    cohort: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for class_code in TARGET_CODES:
        values = contributions[:, class_code, :]
        mean_abs = np.abs(values).mean(axis=0)
        ranks = rankdata(-mean_abs, method="average")
        for index, feature in enumerate(feature_names):
            feature_values = values[:, index]
            rows.append(
                {
                    "seed": seed,
                    "model": MODEL_NAME,
                    "explanation_cohort": cohort,
                    "output_class_code": class_code,
                    "output_class_label": CLASS_LABELS[class_code],
                    "feature": feature,
                    "mean_signed_shap": float(feature_values.mean()),
                    "median_signed_shap": float(np.median(feature_values)),
                    "mean_abs_shap": float(mean_abs[index]),
                    "class_specific_importance_rank": float(ranks[index]),
                    "positive_share": float(np.mean(feature_values > 0)),
                    "negative_share": float(np.mean(feature_values < 0)),
                    "zero_share": float(np.mean(feature_values == 0)),
                    "sample_rows": int(contributions.shape[0]),
                    "interpretation": "raw_class_score_attribution_not_causal",
                }
            )
    return rows


def sample_rows(
    *,
    positions: np.ndarray,
    metadata: pd.DataFrame,
    target: pd.Series,
    seed: int,
    cohort: str,
    selection_seed: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    selected_metadata = metadata.iloc[positions]
    selected_target = target.iloc[positions].to_numpy(dtype=int)
    for order, (position, crash_id, year, target_code) in enumerate(
        zip(
            positions,
            selected_metadata["meta_crash_id"],
            selected_metadata["meta_crash_year"],
            selected_target,
            strict=True,
        )
    ):
        rows.append(
            {
                "seed": seed,
                "explanation_cohort": cohort,
                "sample_order": order,
                "row_position": int(position),
                "meta_crash_id": str(crash_id),
                "meta_crash_year": int(year),
                "target_severity": int(target_code),
                "target_label": CLASS_LABELS[int(target_code)],
                "selection_seed": selection_seed,
            }
        )
    return rows


def audit_rows(
    *,
    positions: np.ndarray,
    target: pd.Series,
    seed: int,
    cohort: str,
    selection_seed: str,
    model_sha: str,
    bundle_sha: str,
) -> list[dict[str, object]]:
    labels = target.iloc[positions].to_numpy(dtype=int)
    position_sha = hash_positions(positions)
    return [
        {
            "seed": seed,
            "explanation_cohort": cohort,
            "selection_seed": selection_seed,
            "sample_rows": len(positions),
            "target_code": code,
            "target_label": CLASS_LABELS[code],
            "target_rows": int(np.sum(labels == code)),
            "row_positions_sha256": position_sha,
            "model_sha256": model_sha,
            "preprocessing_bundle_sha256": bundle_sha,
        }
        for code in TARGET_CODES
    ]


def main() -> None:
    protocol = load_protocol()
    outputs = [
        STABILITY_FILE,
        IMPORTANCE_FILE,
        CLASS_DIRECTION_FILE,
        SAMPLE_FILE,
        RAW_SHAP_FILE,
        BOOTSTRAP_DRAWS_FILE,
        SAMPLE_AUDIT_FILE,
        CHECKPOINT_FILE,
        MANIFEST_FILE,
        COMPLETE_FILE,
    ]
    existing = [relative(path) for path in outputs if path.exists()]
    if existing:
        raise RuntimeError(
            "CAS H3 outputs already exist; refusing to rerun: "
            + "; ".join(existing)
        )

    started = time.perf_counter()
    schema, analysis, _training = load_contracts()
    features, target, metadata, assignments = load_aligned_data(schema, analysis)
    feature_names = list(schema["feature_columns"])
    if len(feature_names) != 15:
        raise AssertionError("CAS H3 frozen feature count changed")
    frozen = json.loads(LIGHTGBM_FROZEN_FILE.read_text(encoding="utf-8"))
    if (
        frozen.get("status")
        != "LIGHTGBM_MODELS_FROZEN_BEFORE_2025_EVALUATION"
    ):
        raise ValueError("CAS LightGBM models are not frozen")

    iterations = int(protocol["common"]["iterations"])
    base_seed = int(protocol["common"]["base_seed"])
    stability_output: list[dict[str, object]] = []
    importance_output: list[dict[str, object]] = []
    direction_output: list[dict[str, object]] = []
    sample_output: list[dict[str, object]] = []
    audit_output: list[dict[str, object]] = []
    bootstrap_output: list[dict[str, object]] = []
    raw_output: list[np.ndarray] = []

    for seed in RANDOM_SEEDS:
        artifact, bundle, model_sha, bundle_sha = load_frozen_model(seed, frozen)
        internal, future, matched_counts = matched_positions(
            assignments=assignments,
            target=target,
            seed=seed,
            base_seed=base_seed,
        )
        cohort_values: dict[str, np.ndarray] = {}
        seed_raw: list[np.ndarray] = []
        cohort_positions = {
            "random_internal_test": internal,
            "matched_2025": future,
        }
        for cohort, positions in cohort_positions.items():
            selection_seed = (
                "all_internal_test_rows"
                if cohort == "random_internal_test"
                else str(base_seed + seed)
            )
            print(
                f"CAS H3: explaining seed {seed} on {cohort} "
                f"({len(positions):,} rows)...",
                flush=True,
            )
            frame = native_frame(features, positions, bundle)
            contributions = shap_contributions(
                artifact["estimator"], frame, len(feature_names)
            )
            if contributions.shape != (len(positions), 3, len(feature_names)):
                raise AssertionError("CAS H3 SHAP contribution shape changed")
            seed_raw.append(contributions)
            cohort_values[cohort] = np.abs(contributions).mean(axis=1)
            importance_output.extend(
                importance_rows(
                    contributions=contributions,
                    feature_names=feature_names,
                    seed=seed,
                    cohort=cohort,
                )
            )
            direction_output.extend(
                class_direction_rows(
                    contributions=contributions,
                    feature_names=feature_names,
                    seed=seed,
                    cohort=cohort,
                )
            )
            sample_output.extend(
                sample_rows(
                    positions=positions,
                    metadata=metadata,
                    target=target,
                    seed=seed,
                    cohort=cohort,
                    selection_seed=selection_seed,
                )
            )
            audit_output.extend(
                audit_rows(
                    positions=positions,
                    target=target,
                    seed=seed,
                    cohort=cohort,
                    selection_seed=selection_seed,
                    model_sha=model_sha,
                    bundle_sha=bundle_sha,
                )
            )
        raw_output.append(np.stack(seed_raw, axis=0))

        internal_values = cohort_values["random_internal_test"]
        future_values = cohort_values["matched_2025"]
        internal_labels = target.iloc[internal].to_numpy(dtype=np.int8)
        future_labels = target.iloc[future].to_numpy(dtype=np.int8)
        point_rho = float(
            spearmanr(
                internal_values.mean(axis=0),
                future_values.mean(axis=0),
            ).statistic
        )
        if not np.isfinite(point_rho):
            raise AssertionError(f"CAS H3 point rho is not finite for seed {seed}")

        internal_bootstrap_seed = base_seed + 10_000_000 + seed * 10 + 1
        future_bootstrap_seed = base_seed + 10_000_000 + seed * 10 + 2
        internal_bootstrap = bootstrap_mean_importance(
            internal_values,
            internal_labels,
            iterations=iterations,
            rng=np.random.default_rng(internal_bootstrap_seed),
        )
        future_bootstrap = bootstrap_mean_importance(
            future_values,
            future_labels,
            iterations=iterations,
            rng=np.random.default_rng(future_bootstrap_seed),
        )
        draws = spearman_rows(internal_bootstrap, future_bootstrap)
        if len(draws) != iterations or not np.isfinite(draws).all():
            raise AssertionError("CAS H3 Bootstrap produced invalid draws")
        lower, upper = np.quantile(draws, [0.025, 0.975])
        stability_output.append(
            {
                "seed": seed,
                "model": MODEL_NAME,
                "comparison": (
                    "same_frozen_model_internal_vs_matched_2025"
                ),
                "features_ranked": len(feature_names),
                "internal_sample_rows": len(internal),
                "future_sample_rows": len(future),
                "minor_rows_each": matched_counts[0],
                "serious_rows_each": matched_counts[1],
                "fatal_rows_each": matched_counts[2],
                "spearman_rho": point_rho,
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "bootstrap_iterations": iterations,
                "valid_iterations": int(np.isfinite(draws).sum()),
                "bootstrap_design": "independent_class_stratified",
                "internal_bootstrap_seed": internal_bootstrap_seed,
                "future_bootstrap_seed": future_bootstrap_seed,
            }
        )
        bootstrap_output.extend(
            {
                "seed": seed,
                "iteration": iteration,
                "spearman_rho": float(value),
            }
            for iteration, value in enumerate(draws, start=1)
        )
        print(
            f"CAS H3: seed {seed} rho={point_rho:.4f}, "
            f"95% interval=[{lower:.4f}, {upper:.4f}]",
            flush=True,
        )

    raw_array = np.stack(raw_output, axis=0)
    expected_raw_shape = (len(RANDOM_SEEDS), 2, 4_887, 3, len(feature_names))
    if raw_array.shape != expected_raw_shape:
        raise AssertionError(
            f"CAS H3 raw SHAP array differs: {raw_array.shape}"
        )
    write_csv(STABILITY_FILE, stability_output)
    write_csv(IMPORTANCE_FILE, importance_output)
    write_csv(CLASS_DIRECTION_FILE, direction_output)
    write_dataframe_gzip(SAMPLE_FILE, pd.DataFrame(sample_output))
    write_numpy_atomic(RAW_SHAP_FILE, raw_array)
    write_dataframe_gzip(
        BOOTSTRAP_DRAWS_FILE,
        pd.DataFrame(bootstrap_output),
    )
    write_csv(SAMPLE_AUDIT_FILE, audit_output)

    stability = pd.DataFrame(stability_output)
    rhos = stability["spearman_rho"].to_numpy(dtype=float)
    all_identical = bool(np.allclose(rhos, 1.0, rtol=0, atol=1e-15))
    h3_status = (
        "NO_OBSERVED_RANK_CHANGE"
        if all_identical
        else "RANK_CHANGE_QUANTIFIED_NO_BINARY_MATERIALITY_THRESHOLD"
    )
    checkpoint_lines = [
        "# CAS H3 SHAP stability checkpoint",
        "",
        "Status: **PASS - H3_SHAP_COMPLETE**",
        "",
        "## Method",
        "",
        "- Five frozen random-reference LightGBM models were explained; no model was refit.",
        "- Each comparison uses all 4,887 internal-test rows and a 4,887-row 2025 sample with exactly matched true-severity counts.",
        "- Importance is mean absolute raw-score SHAP over records and the three output classes.",
        f"- Uncertainty uses {iterations:,} independent class-stratified record Bootstrap iterations on precomputed SHAP values.",
        "",
        "## Result",
        "",
        f"- H3 status: **{h3_status}**.",
        f"- Spearman rho across five frozen models: mean **{rhos.mean():.4f}**, SD **{rhos.std(ddof=1):.4f}**, range **[{rhos.min():.4f}, {rhos.max():.4f}]**.",
    ]
    for row in stability_output:
        checkpoint_lines.append(
            f"- Seed {row['seed']}: rho **{row['spearman_rho']:.4f}**, "
            f"95% interval **[{row['ci_lower']:.4f}, "
            f"{row['ci_upper']:.4f}]**."
        )
    checkpoint_lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- SHAP values describe fitted-model attribution, not causal effects.",
            "- No post-test model selection, retuning, threshold adjustment or binary stability cutoff was introduced.",
            "- CAS and STATS19 remain separate; only directional agreement or disagreement may be discussed.",
            "",
        ]
    )
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text("\n".join(checkpoint_lines), encoding="utf-8")

    artifacts = [
        Path(__file__),
        TEST_FILE,
        PROTOCOL_FILE,
        LIGHTGBM_FROZEN_FILE,
        STABILITY_FILE,
        IMPORTANCE_FILE,
        CLASS_DIRECTION_FILE,
        SAMPLE_FILE,
        RAW_SHAP_FILE,
        BOOTSTRAP_DRAWS_FILE,
        SAMPLE_AUDIT_FILE,
        CHECKPOINT_FILE,
    ]
    build_artifact_manifest(artifacts, MANIFEST_FILE)
    runtime = time.perf_counter() - started
    write_json_atomic(
        COMPLETE_FILE,
        {
            "version": VERSION,
            "status": "H3_SHAP_COMPLETE",
            "created_local": datetime.now()
            .astimezone()
            .isoformat(timespec="seconds"),
            "protocol": relative(PROTOCOL_FILE),
            "protocol_sha256": hash_file(PROTOCOL_FILE),
            "frozen_models_explained": len(RANDOM_SEEDS),
            "model_fits": 0,
            "explanation_cohorts": len(RANDOM_SEEDS) * 2,
            "rows_per_cohort": 4_887,
            "feature_count": len(feature_names),
            "raw_shap_shape": list(raw_array.shape),
            "raw_shap_axis_order": [
                "random_seed",
                "cohort",
                "record",
                "output_class",
                "feature",
            ],
            "raw_shap_seed_order": list(RANDOM_SEEDS),
            "raw_shap_cohort_order": [
                "random_internal_test",
                "matched_2025",
            ],
            "raw_shap_class_order": [
                CLASS_LABELS[code] for code in TARGET_CODES
            ],
            "raw_shap_feature_order": feature_names,
            "bootstrap_iterations": iterations,
            "threads": THREADS,
            "h3_status": h3_status,
            "rho_mean": float(rhos.mean()),
            "rho_sd": float(rhos.std(ddof=1)),
            "rho_range": [float(rhos.min()), float(rhos.max())],
            "causal_claim_allowed": False,
            "post_test_model_selection": False,
            "artifact_manifest": relative(MANIFEST_FILE),
            "artifact_manifest_sha256": hash_file(MANIFEST_FILE),
            "runtime_seconds": runtime,
        },
    )
    print(f"CAS_H3_RUNTIME_MINUTES={runtime / 60:.2f}")
    print(f"CAS_H3_RHO_MEAN={rhos.mean():.6f}")
    print("CAS_H3_STATUS=H3_SHAP_COMPLETE")


if __name__ == "__main__":
    main()
