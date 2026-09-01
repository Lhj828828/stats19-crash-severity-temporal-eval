"""Run the D9-S1 matched-subset multinomial-logistic sensitivity analysis.

This analysis reconstructs the exact deterministic 100,000-row training
subsets used by D9 and fits the frozen D8 unweighted multinomial Logistic
Regression on those rows. Every model is evaluated on the same complete
validation partition used by its paired D9 Ordered Logit model. The 2024 test
partition remains embargoed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from baseline_modeling import (
    TARGET_CODES,
    assert_encoded_matrix,
    classification_metrics,
    confusion_rows,
    fit_category_vocabulary,
    make_preprocessor,
    predict_with_probabilities,
    prepare_features,
)
from d8_train_baselines import make_logistic, prediction_frame, save_prediction
from d9_train_ordered_logit import (
    BENCHMARK_TRAIN_ROWS,
    SUBSET_SEED,
    load_aligned_data,
    load_contracts,
    make_split_specs,
    positions_for_spec,
    stratified_subset,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROTOCOL_FILE = PROJECT_DIR / "config" / "d9s1_matched_subset_protocol.json"
D8_PROTOCOL_FILE = PROJECT_DIR / "config" / "d8_baseline_protocol.json"
D9_PROTOCOL_FILE = PROJECT_DIR / "config" / "d9_ordered_logit_protocol.json"
D9_BENCHMARK_FILE = PROJECT_DIR / "logs" / "d9_runtime_benchmark.json"
D9_METRICS_FILE = PROJECT_DIR / "results" / "d9" / "d9_validation_metrics.csv"
D9_PREDICTION_DIR = PROJECT_DIR / "results" / "d9" / "validation_predictions"
D9_SCRIPT_FILE = PROJECT_DIR / "code" / "d9_train_ordered_logit.py"
D8_SCRIPT_FILE = PROJECT_DIR / "code" / "d8_train_baselines.py"
BASELINE_SCRIPT_FILE = PROJECT_DIR / "code" / "baseline_modeling.py"
ASSIGNMENTS_FILE = PROJECT_DIR / "data" / "processed" / "d6_split_assignments.csv.gz"

SUBSET_DIR = PROJECT_DIR / "data" / "processed" / "d9s1_matched_subsets"
MODEL_DIR = PROJECT_DIR / "models" / "d9s1"
RESULT_DIR = PROJECT_DIR / "results" / "d9s1"
PREDICTION_DIR = RESULT_DIR / "validation_predictions"
METRICS_FILE = RESULT_DIR / "d9s1_validation_metrics.csv"
CONFUSION_FILE = RESULT_DIR / "d9s1_validation_confusion_matrices.csv"
COMPARISON_FILE = RESULT_DIR / "d9s1_matched_comparison.csv"
LOG_DIR = PROJECT_DIR / "logs"
SUBSET_AUDIT_FILE = LOG_DIR / "d9s1_subset_manifest.csv"
PREPROCESS_AUDIT_FILE = LOG_DIR / "d9s1_preprocessing_audit.csv"
CHECKPOINT_FILE = LOG_DIR / "d9s1_checkpoint.md"

VERSION = "D9S1_V1"
MODEL_NAME = "logistic_unweighted_matched_100k"
EXPECTED_SPLITS = 6


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_row_identity(
    positions: np.ndarray, collision_ids: pd.Series, target: pd.Series
) -> str:
    digest = hashlib.sha256()
    for position, collision_id, severity in zip(
        positions,
        collision_ids.astype("string"),
        target.astype(int),
        strict=True,
    ):
        digest.update(f"{int(position)}\t{collision_id}\t{int(severity)}\n".encode("utf-8"))
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows generated for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def split_seed(spec: dict[str, str]) -> int:
    if spec["protocol"] == "temporal":
        return SUBSET_SEED
    return SUBSET_SEED + int(spec["seed"])


def split_slug(spec: dict[str, str]) -> str:
    if spec["protocol"] == "temporal":
        return "temporal"
    return f"random_seed_{spec['seed']}"


def results_already_exist() -> bool:
    markers = [
        METRICS_FILE,
        CONFUSION_FILE,
        COMPARISON_FILE,
        SUBSET_AUDIT_FILE,
        CHECKPOINT_FILE,
    ]
    return any(path.exists() for path in markers)


def make_protocol() -> dict[str, object]:
    d8 = json.loads(D8_PROTOCOL_FILE.read_text(encoding="utf-8"))
    d9 = json.loads(D9_PROTOCOL_FILE.read_text(encoding="utf-8"))
    benchmark = json.loads(D9_BENCHMARK_FILE.read_text(encoding="utf-8"))
    if d8.get("version") != "D8_V1":
        raise ValueError("D9-S1 requires the frozen D8_V1 protocol")
    if d9.get("version") != "D9_V3":
        raise ValueError("D9-S1 requires the frozen D9_V3 protocol")
    if benchmark.get("execution_decision") != "subset_100000":
        raise ValueError("D9-S1 is only justified after the D9 100,000-row fallback")
    if benchmark.get("protocol_sha256") != hash_file(D9_PROTOCOL_FILE):
        raise ValueError("D9 benchmark no longer matches its frozen protocol")
    return {
        "version": VERSION,
        "status": "FROZEN_BEFORE_MATCHED_SENSITIVITY_RESULTS",
        "freeze_date_local": "2026-08-29",
        "analysis_origin": (
            "Post-hoc sensitivity analysis specified after D9 selected its "
            "runtime-only 100,000-row fallback and before D10 LightGBM fitting"
        ),
        "purpose": (
            "Reduce training-sample-size confounding when interpreting the "
            "D9 Ordered Logit validation results"
        ),
        "upstream": {
            "D8_protocol": D8_PROTOCOL_FILE.relative_to(PROJECT_DIR).as_posix(),
            "D8_protocol_sha256": hash_file(D8_PROTOCOL_FILE),
            "D9_protocol": D9_PROTOCOL_FILE.relative_to(PROJECT_DIR).as_posix(),
            "D9_protocol_sha256": hash_file(D9_PROTOCOL_FILE),
            "D9_benchmark": D9_BENCHMARK_FILE.relative_to(PROJECT_DIR).as_posix(),
            "D9_benchmark_sha256": hash_file(D9_BENCHMARK_FILE),
            "D9_metrics": D9_METRICS_FILE.relative_to(PROJECT_DIR).as_posix(),
            "D9_metrics_sha256": hash_file(D9_METRICS_FILE),
            "D6_assignments": ASSIGNMENTS_FILE.relative_to(PROJECT_DIR).as_posix(),
            "D6_assignments_sha256": hash_file(ASSIGNMENTS_FILE),
            "D9_script_sha256": hash_file(D9_SCRIPT_FILE),
            "D8_script_sha256": hash_file(D8_SCRIPT_FILE),
            "baseline_modeling_sha256": hash_file(BASELINE_SCRIPT_FILE),
        },
        "matched_subset": {
            "training_rows_per_split": BENCHMARK_TRAIN_ROWS,
            "splits": "temporal plus all five frozen random-reference splits",
            "base_seed": SUBSET_SEED,
            "temporal_seed": SUBSET_SEED,
            "random_seed_rule": "base_seed + frozen random-reference split seed",
            "sampling": (
                "the exact D9 stratified train_test_split algorithm, followed "
                "by ascending row-position sorting"
            ),
            "identity_control": (
                "persist row position, collision identifier and target for every "
                "selected row; record ordered-row SHA-256 and deterministic file SHA-256"
            ),
            "historical_note": (
                "D9 did not persist training identifiers at fit time. D9-S1 "
                "reconstructs them deterministically from the unchanged hashed "
                "assignment artifact and the D9 source algorithm. This is reported "
                "transparently and is not described as a contemporaneous D9 manifest."
            ),
        },
        "comparator": {
            "model": MODEL_NAME,
            "base_specification": "D8 logistic_unweighted",
            "class_weight": None,
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 500,
            "tol": 0.0001,
            "prediction_rule": "argmax over class probabilities; no threshold tuning",
            "preprocessing": (
                "fit category vocabulary, one-hot encoding, numeric median and "
                "standardization separately on each matched 100,000-row subset"
            ),
        },
        "evaluation": {
            "validation": "the complete validation partition paired with each D9 model",
            "metrics": [
                "Macro-F1",
                "quadratic weighted kappa",
                "ordinal MAE",
                "accuracy",
                "class recalls",
                "serious-or-fatal recall",
                "mean asymmetric cost",
                "confusion matrix",
            ],
            "all_six_splits_required": True,
            "no_result_based_tuning": True,
        },
        "interpretation_boundary": (
            "Matching controls training-row count but does not isolate model "
            "structure: proportional-odds constraints, regularization and encoding "
            "still differ. Report as a sensitivity analysis, not a pure causal "
            "decomposition of structure versus sample size."
        ),
        "reporting_role": "appendix sensitivity table; not a primary result table",
        "test_embargo": {
            "year": 2024,
            "rule": "No 2024 record may be fitted, predicted or evaluated in D9-S1.",
        },
    }


def freeze_protocol() -> None:
    if results_already_exist():
        raise RuntimeError("D9-S1 results already exist; protocol cannot be back-filled")
    if PROTOCOL_FILE.exists():
        existing = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
        if existing.get("status") != "FROZEN_BEFORE_MATCHED_SENSITIVITY_RESULTS":
            raise ValueError("Existing D9-S1 protocol has an unexpected status")
        print("D9-S1 protocol already frozen:", hash_file(PROTOCOL_FILE))
        return
    write_json(PROTOCOL_FILE, make_protocol())
    print("D9-S1 protocol frozen:", hash_file(PROTOCOL_FILE))
    print("D9S1_FREEZE_ASSERTIONS=PASS")


def require_frozen_protocol() -> dict[str, object]:
    if not PROTOCOL_FILE.exists():
        raise FileNotFoundError("Run D9-S1 --freeze before --run")
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    if protocol.get("version") != VERSION:
        raise ValueError("Unexpected D9-S1 protocol version")
    if protocol.get("status") != "FROZEN_BEFORE_MATCHED_SENSITIVITY_RESULTS":
        raise ValueError("D9-S1 protocol is not frozen")
    checks = {
        "D8_protocol_sha256": D8_PROTOCOL_FILE,
        "D9_protocol_sha256": D9_PROTOCOL_FILE,
        "D9_benchmark_sha256": D9_BENCHMARK_FILE,
        "D9_metrics_sha256": D9_METRICS_FILE,
        "D6_assignments_sha256": ASSIGNMENTS_FILE,
        "D9_script_sha256": D9_SCRIPT_FILE,
        "D8_script_sha256": D8_SCRIPT_FILE,
        "baseline_modeling_sha256": BASELINE_SCRIPT_FILE,
    }
    for key, path in checks.items():
        if protocol["upstream"][key] != hash_file(path):
            raise ValueError(f"Frozen D9-S1 upstream changed: {path.name}")
    return protocol


def save_subset_manifest(
    *,
    spec: dict[str, str],
    positions: np.ndarray,
    target: pd.Series,
    metadata: pd.DataFrame,
    assignments: pd.DataFrame,
) -> dict[str, object]:
    slug = split_slug(spec)
    collision_ids = metadata.iloc[positions]["meta_collision_index"].astype("string")
    years = assignments.iloc[positions]["meta_collision_year"].astype(int)
    severities = target.iloc[positions].astype(np.int8)
    if collision_ids.duplicated().any():
        raise AssertionError(f"Duplicate collision identifiers in {slug} subset")
    frame = pd.DataFrame(
        {
            "row_position": positions.astype(np.int64),
            "meta_collision_index": collision_ids.to_numpy(),
            "meta_collision_year": years.to_numpy(),
            "target_severity": severities.to_numpy(),
        }
    )
    path = SUBSET_DIR / f"{slug}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        lineterminator="\n",
    )
    reloaded = pd.read_csv(path, dtype={"meta_collision_index": "string"})
    if not np.array_equal(reloaded["row_position"].to_numpy(), positions):
        raise AssertionError(f"Persisted {slug} positions differ from reconstruction")
    if not reloaded["meta_collision_index"].equals(collision_ids.reset_index(drop=True)):
        raise AssertionError(f"Persisted {slug} identifiers differ from reconstruction")
    counts = severities.value_counts().reindex(TARGET_CODES, fill_value=0)
    return {
        "protocol": spec["protocol"],
        "seed": spec["seed"],
        "split_slug": slug,
        "subset_seed": split_seed(spec),
        "training_rows": len(positions),
        "slight_rows": int(counts.loc[0]),
        "serious_rows": int(counts.loc[1]),
        "fatal_rows": int(counts.loc[2]),
        "minimum_year": int(years.min()),
        "maximum_year": int(years.max()),
        "ordered_row_identity_sha256": hash_row_identity(
            positions, collision_ids.reset_index(drop=True), severities.reset_index(drop=True)
        ),
        "manifest_file": path.relative_to(PROJECT_DIR).as_posix(),
        "manifest_file_sha256": hash_file(path),
        "d9_reconstruction_basis": "frozen deterministic D9_V3 algorithm",
    }


def assert_same_validation_as_d9(
    *,
    slug: str,
    validation_ids: pd.Series,
    y_validation: np.ndarray,
) -> None:
    d9_path = D9_PREDICTION_DIR / f"{slug}__ordered_logit_unweighted.csv.gz"
    d9_prediction = pd.read_csv(
        d9_path,
        usecols=["meta_collision_index", "target_severity", "meta_collision_year"],
        dtype={"meta_collision_index": "string"},
    )
    if not d9_prediction["meta_collision_index"].equals(
        validation_ids.astype("string").reset_index(drop=True)
    ):
        raise AssertionError(f"D9-S1 and D9 validation identifiers differ for {slug}")
    if not np.array_equal(d9_prediction["target_severity"].to_numpy(), y_validation):
        raise AssertionError(f"D9-S1 and D9 validation targets differ for {slug}")
    if int(d9_prediction["meta_collision_year"].max()) > 2023:
        raise AssertionError("D9 validation artifact violates the 2024 embargo")


def fit_one_split(
    *,
    spec: dict[str, str],
    schema: dict[str, object],
    features: pd.DataFrame,
    target: pd.Series,
    metadata: pd.DataFrame,
    assignments: pd.DataFrame,
    protocol_sha256: str,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    all_train_positions, validation_positions, train_years, validation_years = positions_for_spec(
        spec, assignments
    )
    positions = stratified_subset(
        all_train_positions,
        target,
        BENCHMARK_TRAIN_ROWS,
        split_seed(spec),
    )
    if len(positions) != BENCHMARK_TRAIN_ROWS:
        raise AssertionError("D9-S1 failed to reconstruct a 100,000-row subset")
    subset_row = save_subset_manifest(
        spec=spec,
        positions=positions,
        target=target,
        metadata=metadata,
        assignments=assignments,
    )
    slug = split_slug(spec)
    X_train = features.iloc[positions]
    X_validation = features.iloc[validation_positions]
    y_train = target.iloc[positions].to_numpy(dtype=np.int8)
    y_validation = target.iloc[validation_positions].to_numpy(dtype=np.int8)
    validation_ids = metadata.iloc[validation_positions]["meta_collision_index"]
    assert_same_validation_as_d9(
        slug=slug,
        validation_ids=validation_ids,
        y_validation=y_validation,
    )

    categorical = list(schema["categorical_feature_columns"])
    numeric = list(schema["numeric_feature_columns"])
    feature_columns = list(schema["feature_columns"])
    vocabulary = fit_category_vocabulary(X_train, categorical)
    prepared_train = prepare_features(
        X_train,
        feature_columns=feature_columns,
        categorical_columns=categorical,
        numeric_columns=numeric,
        category_vocabulary=vocabulary,
    )
    prepared_validation = prepare_features(
        X_validation,
        feature_columns=feature_columns,
        categorical_columns=categorical,
        numeric_columns=numeric,
        category_vocabulary=vocabulary,
    )
    if any(prepared_train.unseen_counts.values()):
        raise AssertionError("Matched training data produced unseen categories")
    preprocessor = make_preprocessor(
        categorical_columns=categorical,
        numeric_columns=numeric,
        category_vocabulary=vocabulary,
    )
    preprocess_start = time.perf_counter()
    encoded_train = preprocessor.fit_transform(prepared_train.frame)
    encoded_validation = preprocessor.transform(prepared_validation.frame)
    preprocessing_seconds = time.perf_counter() - preprocess_start
    assert_encoded_matrix(encoded_train, len(positions))
    assert_encoded_matrix(encoded_validation, len(validation_positions))

    estimator = make_logistic(None)
    fit_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(encoded_train, y_train)
    fit_seconds = time.perf_counter() - fit_start
    warning_messages = sorted(
        {
            str(item.message)
            for item in caught
            if issubclass(item.category, ConvergenceWarning)
        }
    )
    if warning_messages:
        raise RuntimeError(f"Matched Logistic did not converge for {slug}: {warning_messages}")

    prediction_start = time.perf_counter()
    predicted, probabilities = predict_with_probabilities(estimator, encoded_validation)
    prediction_seconds = time.perf_counter() - prediction_start
    metrics = classification_metrics(y_validation, predicted)
    feature_names = preprocessor.get_feature_names_out().tolist()
    model_dir = MODEL_DIR / slug
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "version": VERSION,
            "protocol_sha256": protocol_sha256,
            "protocol": spec["protocol"],
            "seed": spec["seed"],
            "training_rows": len(positions),
            "training_subset_identity_sha256": subset_row["ordered_row_identity_sha256"],
            "training_years": sorted(train_years),
            "feature_columns": feature_columns,
            "categorical_columns": categorical,
            "numeric_columns": numeric,
            "category_vocabulary": vocabulary,
            "preprocessor": preprocessor,
            "encoded_feature_names": feature_names,
        },
        model_dir / "preprocessing_bundle.joblib",
        compress=3,
    )
    joblib.dump(
        {
            "version": VERSION,
            "protocol_sha256": protocol_sha256,
            "model_name": MODEL_NAME,
            "class_weight": None,
            "estimator": estimator,
            "target_codes": list(TARGET_CODES),
            "training_rows": len(positions),
            "training_subset_identity_sha256": subset_row["ordered_row_identity_sha256"],
            "prediction_rule": "argmax over class probabilities",
        },
        model_dir / f"{MODEL_NAME}.joblib",
        compress=3,
    )

    validation_year_series = assignments.iloc[validation_positions]["meta_collision_year"]
    save_prediction(
        PREDICTION_DIR / f"{slug}__{MODEL_NAME}.csv.gz",
        prediction_frame(
            collision_ids=validation_ids,
            years=validation_year_series,
            y_true=y_validation,
            y_pred=predicted,
            probabilities=probabilities,
        ),
    )
    metric_row = {
        "protocol": spec["protocol"],
        "seed": spec["seed"],
        "evaluation_role": "validation",
        "training_mode": "matched_d9_subset",
        "available_training_rows": len(all_train_positions),
        "training_rows": len(positions),
        "validation_rows": len(validation_positions),
        "training_years": ";".join(str(year) for year in sorted(train_years)),
        "validation_years": ";".join(str(year) for year in sorted(validation_years)),
        "model": MODEL_NAME,
        "class_weighted": False,
        "training_subset_identity_sha256": subset_row["ordered_row_identity_sha256"],
        "encoded_features": len(feature_names),
        "preprocessing_seconds": preprocessing_seconds,
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "n_iter": int(np.max(estimator.n_iter_)),
        "convergence_warning": False,
        **metrics,
    }
    confusions = confusion_rows(
        y_validation,
        predicted,
        protocol=str(spec["protocol"]),
        seed=str(spec["seed"]),
        model=MODEL_NAME,
    )
    audits: list[dict[str, object]] = []
    imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
    medians = dict(zip(numeric, imputer.statistics_, strict=True))
    for column in feature_columns:
        is_categorical = column in categorical
        audits.append(
            {
                "protocol": spec["protocol"],
                "seed": spec["seed"],
                "feature": column,
                "feature_type": "categorical" if is_categorical else "numeric",
                "training_levels": len(vocabulary[column]) if is_categorical else "",
                "training_missing_count": int(X_train[column].isna().sum()),
                "validation_missing_count": int(X_validation[column].isna().sum()),
                "validation_unseen_count": (
                    prepared_validation.unseen_counts[column] if is_categorical else ""
                ),
                "training_fitted_imputation_value": (
                    "" if is_categorical else float(medians[column])
                ),
            }
        )
    print(
        f"Completed {slug}: matched train={len(positions):,}, "
        f"validation={len(validation_positions):,}, encoded={len(feature_names)}"
    )
    return metric_row, confusions, audits, subset_row


def build_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    d9 = pd.read_csv(D9_METRICS_FILE, dtype={"seed": "string"})
    current = metrics.copy()
    current["seed"] = current["seed"].astype("string")
    d9["seed"] = d9["seed"].astype("string")
    metric_columns = [
        "macro_f1",
        "qwk",
        "ordinal_mae",
        "accuracy",
        "slight_recall",
        "serious_recall",
        "fatal_recall",
        "serious_or_fatal_recall",
        "mean_asymmetric_cost",
    ]
    left = current[
        ["protocol", "seed", "training_rows", "validation_rows", *metric_columns]
    ].rename(
        columns={
            "training_rows": "matched_logistic_training_rows",
            "validation_rows": "validation_rows",
            **{column: f"matched_logistic_{column}" for column in metric_columns},
        }
    )
    right = d9[
        ["protocol", "seed", "training_rows", "validation_rows", *metric_columns]
    ].rename(
        columns={
            "training_rows": "ordered_logit_training_rows",
            "validation_rows": "ordered_validation_rows",
            **{column: f"ordered_logit_{column}" for column in metric_columns},
        }
    )
    comparison = left.merge(right, on=["protocol", "seed"], how="outer", validate="one_to_one")
    if len(comparison) != EXPECTED_SPLITS or comparison.isna().any().any():
        raise AssertionError("D9-S1 could not form six complete matched pairs")
    if not np.array_equal(
        comparison["validation_rows"].to_numpy(),
        comparison["ordered_validation_rows"].to_numpy(),
    ):
        raise AssertionError("Matched models used different validation row counts")
    for column in metric_columns:
        comparison[f"delta_logistic_minus_ordered_{column}"] = (
            comparison[f"matched_logistic_{column}"]
            - comparison[f"ordered_logit_{column}"]
        )
    return comparison.sort_values(["protocol", "seed"]).reset_index(drop=True)


def write_checkpoint(metrics: pd.DataFrame, comparison: pd.DataFrame) -> None:
    temporal = metrics.loc[metrics["protocol"].eq("temporal")].iloc[0]
    random = metrics.loc[metrics["protocol"].eq("random_reference")]
    temporal_comparison = comparison.loc[comparison["protocol"].eq("temporal")].iloc[0]
    lines = [
        "# D9-S1 matched-subset sensitivity checkpoint",
        "",
        "## Status",
        "",
        "**PASS - six matched 100,000-row unweighted Multinomial Logistic models completed.**",
        "",
        "## Design controls",
        "",
        "- Each training subset was reconstructed with the exact deterministic D9_V3 selection algorithm.",
        "- Row positions, collision identifiers, targets and SHA-256 identities were persisted for all six subsets.",
        "- Preprocessing was fitted only on each matched 100,000-row training subset.",
        "- Each complete validation partition exactly matched the paired D9 Ordered Logit artifact.",
        "- No 2024 record was fitted, predicted or evaluated.",
        "",
        "## Validation summary",
        "",
        f"- Temporal matched Logistic: Macro-F1 **{temporal['macro_f1']:.4f}**, QWK **{temporal['qwk']:.4f}**, Fatal recall **{temporal['fatal_recall']:.4f}**.",
        f"- Temporal Logistic minus Ordered Logit Macro-F1 difference: **{temporal_comparison['delta_logistic_minus_ordered_macro_f1']:+.4f}**.",
        f"- Random-reference matched Logistic Macro-F1 mean **{random['macro_f1'].mean():.4f}** (SD **{random['macro_f1'].std(ddof=1):.4f}**).",
        "",
        "## Interpretation boundary",
        "",
        "- This analysis controls training-row count but does not isolate model structure.",
        "- Ordered and multinomial models still differ in constraints, regularization and encoding.",
        "- Results belong in an appendix sensitivity table, not the primary model table.",
        "- The reconstructed D9 subsets are deterministic, but D9 did not persist contemporaneous training identifiers; this provenance limitation must be disclosed.",
        "",
        "## Handoff",
        "",
        "- D10 may tune full-data class-weighted LightGBM using training and validation data only.",
        "- The 2024 test remains sealed until D11.",
        "",
    ]
    CHECKPOINT_FILE.write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    protocol = require_frozen_protocol()
    protocol_sha256 = hash_file(PROTOCOL_FILE)
    schema, d6, _ = load_contracts()
    features, target, metadata, assignments = load_aligned_data(schema, d6)
    specs = make_split_specs(d6)
    if len(specs) != EXPECTED_SPLITS:
        raise AssertionError(f"Expected six frozen splits, found {len(specs)}")

    all_metrics: list[dict[str, object]] = []
    all_confusions: list[dict[str, object]] = []
    all_audits: list[dict[str, object]] = []
    all_subset_rows: list[dict[str, object]] = []
    for spec in specs:
        metric, confusions, audits, subset_row = fit_one_split(
            spec=spec,
            schema=schema,
            features=features,
            target=target,
            metadata=metadata,
            assignments=assignments,
            protocol_sha256=protocol_sha256,
        )
        all_metrics.append(metric)
        all_confusions.extend(confusions)
        all_audits.extend(audits)
        all_subset_rows.append(subset_row)

    metrics = pd.DataFrame(all_metrics)
    comparison = build_comparison(metrics)
    write_csv(METRICS_FILE, all_metrics)
    write_csv(CONFUSION_FILE, all_confusions)
    write_csv(PREPROCESS_AUDIT_FILE, all_audits)
    write_csv(SUBSET_AUDIT_FILE, all_subset_rows)
    COMPARISON_FILE.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_FILE, index=False, encoding="utf-8-sig")
    write_checkpoint(metrics, comparison)
    if protocol["reporting_role"] != "appendix sensitivity table; not a primary result table":
        raise AssertionError("D9-S1 reporting role changed during execution")
    print("D9-S1 metric rows:", len(metrics))
    print("D9-S1 subset rows persisted:", sum(row["training_rows"] for row in all_subset_rows))
    print("D9-S1 prediction files:", len(list(PREDICTION_DIR.glob("*.csv.gz"))))
    print("FINAL_D9S1_ASSERTIONS=PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true", help="Freeze the sensitivity protocol")
    mode.add_argument("--run", action="store_true", help="Run all six matched validation analyses")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.freeze:
        freeze_protocol()
    else:
        run()
