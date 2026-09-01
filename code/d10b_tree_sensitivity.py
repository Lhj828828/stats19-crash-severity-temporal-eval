"""Run the frozen D10b LightGBM tree-count sensitivity diagnostic.

D10b extends only the already-selected temporal C03 fit to 2000 trees and
reads validation predictions at 1200, 1500 and 2000 trees. It does not tune,
replace the D10 model, inspect a random test split, or score 2024.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

# Set native thread caps before importing numerical libraries.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "4"

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baseline_modeling import TARGET_CODES, classification_metrics
from d10_tune_lightgbm import (
    candidate_lookup,
    class_weights_for_spec,
    load_aligned_data,
    load_contracts,
    make_estimator,
    make_split_specs,
    positions_for_spec,
    prepare_native_split,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROTOCOL_FILE = PROJECT_DIR / "config" / "d10b_tree_sensitivity_protocol.json"
D10_PROTOCOL_FILE = PROJECT_DIR / "config" / "d10_lightgbm_protocol.json"
D10_AMENDMENT_FILE = PROJECT_DIR / "config" / "d10_execution_amendment.json"
D10_SELECTED_FILE = PROJECT_DIR / "config" / "d10_selected_lightgbm.json"
D10_TUNING_FILE = PROJECT_DIR / "results" / "d10" / "d10_tuning_results.csv"
D10_PREDICTION_FILE = (
    PROJECT_DIR
    / "results"
    / "d10"
    / "validation_predictions"
    / "temporal__lightgbm_weighted.csv.gz"
)
ASSIGNMENTS_FILE = PROJECT_DIR / "data" / "processed" / "d6_split_assignments.csv.gz"

RESULT_DIR = PROJECT_DIR / "results" / "d10b"
PREDICTION_DIR = RESULT_DIR / "validation_predictions"
MODEL_DIR = PROJECT_DIR / "models" / "d10b" / "temporal"
FIGURE_FILE = PROJECT_DIR / "figures" / "d10b_validation_logloss_curve.png"
LOG_DIR = PROJECT_DIR / "logs"
METRICS_FILE = RESULT_DIR / "d10b_checkpoint_metrics.csv"
HISTORY_FILE = RESULT_DIR / "d10b_logloss_history.csv"
MODEL_FILE = MODEL_DIR / "c03_2000_trees.joblib"
CHECKPOINT_FILE = LOG_DIR / "d10b_checkpoint.md"
MANIFEST_FILE = LOG_DIR / "d10b_artifact_manifest.csv"
INCIDENT_FILE = LOG_DIR / "d10b_execution_incident.json"

VERSION = "D10B_V1"
CHECKPOINTS = (1200, 1500, 2000)
MAX_ESTIMATORS = 2000
MODEL_NAME = "c03_tree_sensitivity"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows generated for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_prediction(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.gz")
    frame.to_csv(
        temporary,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        lineterminator="\n",
    )
    temporary.replace(path)


def require_protocol() -> dict[str, Any]:
    if not PROTOCOL_FILE.exists():
        raise FileNotFoundError("The pre-result D10b protocol is missing")
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    if protocol.get("version") != VERSION:
        raise ValueError("Unexpected D10b protocol version")
    if protocol.get("status") != "FROZEN_BEFORE_D10B_1500_OR_2000_VALIDATION_INSPECTION":
        raise ValueError("D10b protocol is not frozen")
    path_keys = {
        "D5_schema_sha256": PROJECT_DIR / "config" / "d5_dataset_schema.json",
        "D6_protocol_sha256": PROJECT_DIR / "config" / "d6_analysis_protocol.json",
        "D6_assignments_sha256": ASSIGNMENTS_FILE,
        "D7_training_inputs_sha256": PROJECT_DIR / "config" / "d7_training_inputs.json",
        "D10_protocol_sha256": D10_PROTOCOL_FILE,
        "D10_execution_amendment_sha256": D10_AMENDMENT_FILE,
        "D10_selected_model_sha256": D10_SELECTED_FILE,
        "D10_tuning_results_sha256": D10_TUNING_FILE,
        "D10_source_sha256": PROJECT_DIR / "code" / "d10_tune_lightgbm.py",
        "baseline_modeling_source_sha256": PROJECT_DIR / "code" / "baseline_modeling.py",
        "modeling_data_source_sha256": PROJECT_DIR / "code" / "modeling_data.py",
    }
    for key, path in path_keys.items():
        if hash_file(path) != protocol["upstream"][key]:
            raise ValueError(f"Upstream artifact changed after D10b freeze: {path.name}")
    if tuple(protocol["diagnostic"]["checkpoints"]) != CHECKPOINTS:
        raise ValueError("D10b checkpoint set differs from the frozen design")
    if int(protocol["diagnostic"]["single_fit_estimators"]) != MAX_ESTIMATORS:
        raise ValueError("D10b maximum tree count differs from the frozen design")
    if protocol["decision_lock"]["D11_change_allowed_from_D10b"] is not False:
        raise ValueError("D10b must not be allowed to change D11")
    return protocol


def probabilities_at_iteration(
    estimator: lgb.LGBMClassifier,
    features: pd.DataFrame,
    iteration: int,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(
        estimator.predict_proba(features, num_iteration=iteration), dtype=float
    )
    classes = np.asarray(estimator.classes_, dtype=int)
    if classes.tolist() != list(TARGET_CODES):
        raise AssertionError(f"Unexpected class order: {classes.tolist()}")
    if probabilities.shape != (len(features), len(TARGET_CODES)):
        raise AssertionError("Unexpected probability matrix shape")
    if not np.isfinite(probabilities).all():
        raise AssertionError("Non-finite validation probability")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-8):
        raise AssertionError("Validation probabilities do not sum to one")
    predicted = classes[np.argmax(probabilities, axis=1)].astype(np.int8)
    return predicted, probabilities


def prediction_frame(
    collision_ids: pd.Series,
    years: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    iteration: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "meta_collision_index": collision_ids.astype("string").to_numpy(),
            "meta_collision_year": years.astype(int).to_numpy(),
            "checkpoint_trees": iteration,
            "target_severity": y_true.astype(np.int8),
            "predicted_severity": y_pred.astype(np.int8),
            "prob_slight": probabilities[:, 0],
            "prob_serious": probabilities[:, 1],
            "prob_fatal": probabilities[:, 2],
        }
    )


def verify_d10_reproduction(prediction: pd.DataFrame, metrics: dict[str, float]) -> None:
    frozen = pd.read_csv(D10_PREDICTION_FILE, dtype={"meta_collision_index": "string"})
    if not np.array_equal(
        prediction["meta_collision_index"].astype("string").to_numpy(),
        frozen["meta_collision_index"].astype("string").to_numpy(),
    ):
        raise AssertionError("D10b 1200-tree validation IDs differ from frozen D10")
    for column in ("meta_collision_year", "target_severity", "predicted_severity"):
        if not np.array_equal(prediction[column].to_numpy(), frozen[column].to_numpy()):
            raise AssertionError(f"D10b 1200-tree {column} differs from frozen D10")
    probability_columns = ["prob_slight", "prob_serious", "prob_fatal"]
    if not np.allclose(
        prediction[probability_columns].to_numpy(dtype=float),
        frozen[probability_columns].to_numpy(dtype=float),
        rtol=0,
        atol=2e-14,
    ):
        raise AssertionError("D10b 1200-tree probabilities differ from frozen D10")
    selected = json.loads(D10_SELECTED_FILE.read_text(encoding="utf-8"))
    for metric, expected in selected["selected_validation_metrics"].items():
        if not np.isclose(metrics[metric], float(expected), rtol=0, atol=1e-14):
            raise AssertionError(f"D10b failed to reproduce D10 metric: {metric}")


def save_figure(history: list[float]) -> None:
    FIGURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    iterations = np.arange(1, len(history) + 1)
    figure, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    axis.plot(iterations, history, color="#176B87", linewidth=1.4)
    colors = {1200: "#B23A48", 1500: "#D17A22", 2000: "#3A7D44"}
    for checkpoint in CHECKPOINTS:
        value = history[checkpoint - 1]
        axis.axvline(checkpoint, color=colors[checkpoint], linewidth=0.9, alpha=0.75)
        axis.scatter(checkpoint, value, color=colors[checkpoint], s=28, zorder=3)
        axis.annotate(
            f"{checkpoint}: {value:.6f}",
            (checkpoint, value),
            xytext=(-8, 10 if checkpoint != 1500 else -16),
            textcoords="offset points",
            ha="right",
            fontsize=8.5,
            color=colors[checkpoint],
        )
    axis.set_xlabel("Boosting iteration")
    axis.set_ylabel("2023 validation multiclass log loss")
    axis.set_title("D10b tree-count boundary sensitivity (C03)")
    axis.grid(axis="y", color="#D9DEE3", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(FIGURE_FILE, dpi=300, facecolor="white")
    plt.close(figure)


def write_checkpoint(metrics: pd.DataFrame, history: list[float], fit_seconds: float) -> None:
    values = metrics.set_index("checkpoint_trees")
    loss_1200 = float(history[1199])
    loss_1500 = float(history[1499])
    loss_2000 = float(history[1999])
    lines = [
        "# D10b tree-count sensitivity checkpoint",
        "",
        "## Status",
        "",
        "**PASS - the pre-frozen temporal boundary diagnostic completed.**",
        "",
        "## Scope",
        "",
        "- One fixed C03 model was trained on 2018-2022 and evaluated on 2023 only.",
        "- The fit used 4 CPU threads, no early stopping and checkpoints at 1200, 1500 and 2000 trees.",
        "- No random-reference test row and no 2024 row was predicted or evaluated.",
        "- D10b is diagnostic only. D11 remains locked to the D10 C03 model with 1200 trees.",
        "",
        "## Reproduction control",
        "",
        "- The 1200-tree checkpoint reproduced the frozen D10 temporal validation IDs, labels, probabilities and reported metrics within the prespecified numerical tolerance.",
        "",
        "## Observed validation results",
        "",
        f"- 1200 trees: log loss **{loss_1200:.8f}**, Macro-F1 **{values.loc[1200, 'macro_f1']:.5f}**, QWK **{values.loc[1200, 'qwk']:.5f}**, Fatal recall **{values.loc[1200, 'fatal_recall']:.5f}**.",
        f"- 1500 trees: log loss **{loss_1500:.8f}**, Macro-F1 **{values.loc[1500, 'macro_f1']:.5f}**, QWK **{values.loc[1500, 'qwk']:.5f}**, Fatal recall **{values.loc[1500, 'fatal_recall']:.5f}**.",
        f"- 2000 trees: log loss **{loss_2000:.8f}**, Macro-F1 **{values.loc[2000, 'macro_f1']:.5f}**, QWK **{values.loc[2000, 'qwk']:.5f}**, Fatal recall **{values.loc[2000, 'fatal_recall']:.5f}**.",
        f"- Log-loss change, 1200 to 1500: **{loss_1500 - loss_1200:+.8f}**.",
        f"- Log-loss change, 1500 to 2000: **{loss_2000 - loss_1500:+.8f}**.",
        "- No post-hoc numerical threshold was used to label the curve as plateaued or not plateaued.",
        f"- Full 2000-tree fit time: **{fit_seconds:.2f} s**.",
        "",
    ]
    CHECKPOINT_FILE.write_text("\n".join(lines), encoding="utf-8")


def build_manifest() -> None:
    paths = [
        PROTOCOL_FILE,
        METRICS_FILE,
        HISTORY_FILE,
        MODEL_FILE,
        FIGURE_FILE,
        CHECKPOINT_FILE,
        INCIDENT_FILE,
        Path(__file__),
        PROJECT_DIR / "code" / "test_d10b_tree_sensitivity.py",
        *sorted(PREDICTION_DIR.glob("*.csv.gz")),
    ]
    rows = []
    for path in sorted({item.resolve() for item in paths}):
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(
            {
                "relative_path": path.relative_to(PROJECT_DIR.resolve()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hash_file(path),
            }
        )
    write_csv(MANIFEST_FILE, rows)


def main() -> None:
    protocol = require_protocol()
    if any(path.exists() for path in (METRICS_FILE, HISTORY_FILE, MODEL_FILE)):
        raise FileExistsError("D10b primary outputs already exist; refusing to overwrite")

    d10_protocol = json.loads(D10_PROTOCOL_FILE.read_text(encoding="utf-8"))
    selected = json.loads(D10_SELECTED_FILE.read_text(encoding="utf-8"))
    if selected["selected_candidate_id"] != protocol["model_lock"]["candidate_id"]:
        raise AssertionError("Frozen D10 selection is not the D10b locked candidate")
    if int(selected["selected_n_estimators"]) != CHECKPOINTS[0]:
        raise AssertionError("Frozen D10 tree count is not the first D10b checkpoint")

    schema, d6, d7, _ = load_contracts()
    features, target, metadata, assignments = load_aligned_data(schema, d6)
    spec = make_split_specs(d6)[0]
    train_positions, validation_positions, train_years, validation_years = positions_for_spec(
        spec, assignments
    )
    if train_years != {2018, 2019, 2020, 2021, 2022} or validation_years != {2023}:
        raise AssertionError("Unexpected D10b temporal year cohorts")

    X_train, X_validation, bundle, _ = prepare_native_split(
        features.iloc[train_positions], features.iloc[validation_positions], schema
    )
    y_train = target.iloc[train_positions].to_numpy(dtype=np.int8)
    y_validation = target.iloc[validation_positions].to_numpy(dtype=np.int8)
    class_weights = class_weights_for_spec(d7, spec)
    selected_candidate = candidate_lookup(d10_protocol)[selected["selected_candidate_id"]]
    if selected_candidate["parameters"] != selected["selected_candidate_parameters"]:
        raise AssertionError("D10 selected parameters changed")

    estimator = make_estimator(
        d10_protocol,
        selected_candidate,
        class_weights,
        MAX_ESTIMATORS,
    )
    if estimator.get_params()["n_jobs"] != 4:
        raise AssertionError("D10b LightGBM thread limit is not four")
    evaluation_result: dict[str, dict[str, list[float]]] = {}
    started = time.perf_counter()
    estimator.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        eval_metric="multi_logloss",
        categorical_feature=bundle["categorical_columns"],
        callbacks=[lgb.record_evaluation(evaluation_result), lgb.log_evaluation(period=0)],
    )
    fit_seconds = time.perf_counter() - started

    history = evaluation_result.get("valid_0", {}).get("multi_logloss", [])
    if len(history) != MAX_ESTIMATORS:
        raise AssertionError(f"Expected 2000 validation-loss values, found {len(history)}")
    if not np.isfinite(np.asarray(history, dtype=float)).all():
        raise AssertionError("Non-finite value in validation-loss history")

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: dict[int, pd.DataFrame] = {}
    validation_ids = metadata.iloc[validation_positions]["meta_collision_index"]
    validation_year_series = metadata.iloc[validation_positions]["meta_collision_year"]
    for checkpoint in CHECKPOINTS:
        predicted, probabilities = probabilities_at_iteration(
            estimator, X_validation, checkpoint
        )
        metrics = classification_metrics(y_validation, predicted)
        frame = prediction_frame(
            validation_ids,
            validation_year_series,
            y_validation,
            predicted,
            probabilities,
            checkpoint,
        )
        if checkpoint == CHECKPOINTS[0]:
            verify_d10_reproduction(frame, metrics)
        prediction_frames[checkpoint] = frame
        metric_rows.append(
            {
                "protocol": "temporal",
                "seed": "year_based",
                "evaluation_role": "validation",
                "training_rows": len(train_positions),
                "validation_rows": len(validation_positions),
                "training_years": ";".join(str(year) for year in sorted(train_years)),
                "validation_years": ";".join(str(year) for year in sorted(validation_years)),
                "candidate_id": selected["selected_candidate_id"],
                "checkpoint_trees": checkpoint,
                "validation_multi_logloss": float(history[checkpoint - 1]),
                "fit_seconds_full_2000": fit_seconds,
                **metrics,
            }
        )

    history_rows = [
        {
            "iteration": iteration,
            "validation_multi_logloss": float(value),
            "change_from_previous_iteration": (
                "" if iteration == 1 else float(value - history[iteration - 2])
            ),
            "is_checkpoint": iteration in CHECKPOINTS,
        }
        for iteration, value in enumerate(history, start=1)
    ]

    write_csv(METRICS_FILE, metric_rows)
    write_csv(HISTORY_FILE, history_rows)
    for checkpoint, frame in prediction_frames.items():
        save_prediction(
            PREDICTION_DIR / f"temporal__c03__{checkpoint}_trees.csv.gz", frame
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_temporary = MODEL_FILE.with_suffix(MODEL_FILE.suffix + ".tmp")
    joblib.dump(
        {
            "version": VERSION,
            "status": "D10B_DIAGNOSTIC_ONLY",
            "protocol_sha256": hash_file(PROTOCOL_FILE),
            "D10_protocol_sha256": hash_file(D10_PROTOCOL_FILE),
            "D10_selected_model_sha256": hash_file(D10_SELECTED_FILE),
            "candidate_id": selected["selected_candidate_id"],
            "candidate_parameters": selected["selected_candidate_parameters"],
            "n_estimators": MAX_ESTIMATORS,
            "checkpoints": list(CHECKPOINTS),
            "class_weight": class_weights,
            "training_rows": len(train_positions),
            "training_years": sorted(train_years),
            "validation_rows": len(validation_positions),
            "validation_years": sorted(validation_years),
            "preprocessing_bundle": bundle,
            "estimator": estimator,
        },
        model_temporary,
        compress=3,
    )
    model_temporary.replace(MODEL_FILE)

    metrics_frame = pd.DataFrame(metric_rows)
    save_figure(history)
    write_checkpoint(metrics_frame, history, fit_seconds)
    build_manifest()
    print("D10B_1200_REPRODUCTION=PASS")
    print(f"D10B_FIT_SECONDS={fit_seconds:.2f}")
    for row in metric_rows:
        print(
            f"D10B_{row['checkpoint_trees']}: "
            f"logloss={row['validation_multi_logloss']:.8f}, "
            f"Macro-F1={row['macro_f1']:.6f}, "
            f"QWK={row['qwk']:.6f}, "
            f"FatalRecall={row['fatal_recall']:.6f}"
        )
    print("D10B_RUN=PASS")


if __name__ == "__main__":
    main()
