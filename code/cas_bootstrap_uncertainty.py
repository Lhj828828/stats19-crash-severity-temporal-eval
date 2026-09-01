"""Compute frozen CAS H1/H2 stratified bootstrap uncertainty."""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from cas_modeling_common import (
    ASYMMETRIC_COST_MATRIX,
    PROJECT_DIR,
    TARGET_CODES,
    build_artifact_manifest,
    hash_file,
    relative,
    write_csv,
    write_json_atomic,
)


VERSION = "CAS_BOOTSTRAP_V1"
PROTOCOL_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_post_analysis_protocol.json"
)
EVALUATION_COMPLETE_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_evaluation_complete.json"
)
EVALUATION_METRICS_FILE = (
    PROJECT_DIR / "results" / "cas_evaluation" / "test_metrics.csv"
)
PREDICTION_DIR = PROJECT_DIR / "results" / "cas_evaluation" / "predictions"
RESULT_DIR = PROJECT_DIR / "results" / "cas_post_analysis"
H1_FILE = RESULT_DIR / "h1_bootstrap.csv"
H1_SUMMARY_FILE = RESULT_DIR / "h1_across_seed_summary.csv"
H2_FILE = RESULT_DIR / "h2_bootstrap.csv"
H2_SUMMARY_FILE = RESULT_DIR / "h2_design_summary.csv"
LOG_DIR = PROJECT_DIR / "logs" / "cas"
CHECKPOINT_FILE = LOG_DIR / "cas_bootstrap_checkpoint.md"
MANIFEST_FILE = LOG_DIR / "cas_bootstrap_artifact_manifest.csv"
COMPLETE_FILE = PROJECT_DIR / "config" / "cas" / "cas_bootstrap_complete.json"

CLASS_COUNT = len(TARGET_CODES)
ORDINAL_DISTANCE = np.abs(
    np.arange(CLASS_COUNT)[:, None] - np.arange(CLASS_COUNT)[None, :]
).astype(float)
QWK_WEIGHT = (
    ORDINAL_DISTANCE / float(CLASS_COUNT - 1)
) ** 2


def load_protocol() -> dict[str, object]:
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    if protocol.get("version") != "CAS_POST_ANALYSIS_V1":
        raise ValueError("Unexpected CAS post-analysis protocol")
    if protocol.get("status") != (
        "OPERATIONAL_SUPPLEMENT_FROZEN_BEFORE_BOOTSTRAP_OR_SHAP_RESULTS"
    ):
        raise ValueError("CAS post-analysis protocol is not frozen")
    checks = {
        EVALUATION_COMPLETE_FILE: protocol["upstream"][
            "evaluation_complete_sha256"
        ],
        EVALUATION_METRICS_FILE: protocol["upstream"][
            "evaluation_metrics_sha256"
        ],
    }
    for path, expected in checks.items():
        if hash_file(path) != expected:
            raise ValueError(f"CAS bootstrap upstream changed: {path.name}")
    return protocol


def point_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    codes = y_true.astype(int) * CLASS_COUNT + y_pred.astype(int)
    return np.bincount(codes, minlength=CLASS_COUNT**2).reshape(
        CLASS_COUNT, CLASS_COUNT
    )


def metrics_from_confusion(
    confusion: np.ndarray,
) -> dict[str, np.ndarray]:
    cm = np.asarray(confusion, dtype=float)
    single = cm.ndim == 2
    if single:
        cm = cm[None, :, :]
    total = cm.sum(axis=(1, 2))
    true_counts = cm.sum(axis=2)
    predicted_counts = cm.sum(axis=1)
    diagonal = np.diagonal(cm, axis1=1, axis2=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.divide(
            diagonal,
            true_counts,
            out=np.zeros_like(diagonal),
            where=true_counts != 0,
        )
        precision = np.divide(
            diagonal,
            predicted_counts,
            out=np.zeros_like(diagonal),
            where=predicted_counts != 0,
        )
        f1 = np.divide(
            2.0 * diagonal,
            true_counts + predicted_counts,
            out=np.zeros_like(diagonal),
            where=(true_counts + predicted_counts) != 0,
        )
    observed_weight = np.einsum("bij,ij->b", cm, QWK_WEIGHT)
    expected_count = (
        true_counts[:, :, None] * predicted_counts[:, None, :] / total[:, None, None]
    )
    expected_weight = np.einsum("bij,ij->b", expected_count, QWK_WEIGHT)
    qwk = np.divide(
        observed_weight,
        expected_weight,
        out=np.zeros_like(observed_weight),
        where=expected_weight != 0,
    )
    qwk = 1.0 - qwk
    result = {
        "macro_f1": f1.mean(axis=1),
        "qwk": qwk,
        "ordinal_mae": np.einsum("bij,ij->b", cm, ORDINAL_DISTANCE) / total,
        "accuracy": diagonal.sum(axis=1) / total,
        "minor_recall": recall[:, 0],
        "serious_recall": recall[:, 1],
        "fatal_recall": recall[:, 2],
        "serious_or_fatal_recall": cm[:, 1:, 1:].sum(axis=(1, 2))
        / true_counts[:, 1:].sum(axis=1),
        "mean_asymmetric_cost": np.einsum(
            "bij,ij->b", cm, ASYMMETRIC_COST_MATRIX
        )
        / total,
    }
    if single:
        return {name: values[0] for name, values in result.items()}
    return result


def stratified_confusion_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    output = np.zeros((iterations, CLASS_COUNT, CLASS_COUNT), dtype=np.int32)
    for true_code in TARGET_CODES:
        observed = np.bincount(
            y_pred[y_true == true_code].astype(int),
            minlength=CLASS_COUNT,
        )
        n = int(observed.sum())
        output[:, true_code, :] = rng.multinomial(
            n,
            observed / n,
            size=iterations,
        )
    return output


def paired_confusion_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    output_a = np.zeros((iterations, CLASS_COUNT, CLASS_COUNT), dtype=np.int32)
    output_b = np.zeros((iterations, CLASS_COUNT, CLASS_COUNT), dtype=np.int32)
    for true_code in TARGET_CODES:
        mask = y_true == true_code
        joint_code = (
            pred_a[mask].astype(int) * CLASS_COUNT + pred_b[mask].astype(int)
        )
        observed = np.bincount(
            joint_code,
            minlength=CLASS_COUNT**2,
        ).reshape(CLASS_COUNT, CLASS_COUNT)
        n = int(observed.sum())
        sampled = rng.multinomial(
            n,
            observed.ravel() / n,
            size=iterations,
        ).reshape(iterations, CLASS_COUNT, CLASS_COUNT)
        output_a[:, true_code, :] = sampled.sum(axis=2)
        output_b[:, true_code, :] = sampled.sum(axis=1)
    return output_a, output_b


def percentile_interval(values: np.ndarray) -> tuple[float, float, int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        raise ValueError("No finite CAS bootstrap estimates")
    lower, upper = np.quantile(finite, [0.025, 0.975])
    return float(lower), float(upper), len(finite)


def load_prediction(group: str, model: str) -> pd.DataFrame:
    path = PREDICTION_DIR / f"{group}__{model}.csv.gz"
    frame = pd.read_csv(path, dtype={"meta_crash_id": "string"})
    if frame["meta_crash_id"].duplicated().any():
        raise ValueError(f"Duplicated CAS predictions: {path.name}")
    return frame


def run_h1(
    protocol: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    iterations = int(protocol["common"]["iterations"])
    base_seed = int(protocol["common"]["base_seed"])
    metrics = list(protocol["common"]["metrics"])
    higher = dict(protocol["common"]["higher_is_better"])
    rows: list[dict[str, object]] = []
    models = list(protocol["H1"]["models"])
    for seed_index, seed in enumerate(protocol["H1"]["seeds"]):
        internal_group = f"random_seed_{seed}_internal_test"
        future_group = f"random_seed_{seed}_2025_diagnostic"
        for model_index, model in enumerate(models):
            internal = load_prediction(internal_group, model)
            future = load_prediction(future_group, model)
            y_internal = internal["target_severity"].to_numpy(dtype=np.int8)
            p_internal = internal["predicted_severity"].to_numpy(dtype=np.int8)
            y_future = future["target_severity"].to_numpy(dtype=np.int8)
            p_future = future["predicted_severity"].to_numpy(dtype=np.int8)
            rng = np.random.default_rng(
                base_seed + 100_000 + seed_index * 10_000 + model_index * 1_000
            )
            cm_internal = stratified_confusion_bootstrap(
                y_internal, p_internal, iterations, rng
            )
            cm_future = stratified_confusion_bootstrap(
                y_future, p_future, iterations, rng
            )
            boot_internal = metrics_from_confusion(cm_internal)
            boot_future = metrics_from_confusion(cm_future)
            point_internal = metrics_from_confusion(
                point_confusion(y_internal, p_internal)
            )
            point_future = metrics_from_confusion(
                point_confusion(y_future, p_future)
            )
            for metric in metrics:
                differences = boot_internal[metric] - boot_future[metric]
                lower, upper, valid = percentile_interval(differences)
                rows.append(
                    {
                        "seed": seed,
                        "model": model,
                        "metric": metric,
                        "higher_is_better": bool(higher[metric]),
                        "contrast": "random_internal_minus_same_model_2025",
                        "internal_value": float(point_internal[metric]),
                        "future_2025_value": float(point_future[metric]),
                        "point_gap": float(
                            point_internal[metric] - point_future[metric]
                        ),
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "iterations": iterations,
                        "valid_iterations": valid,
                        "bootstrap_design": "independent_stratified",
                    }
                )

    frame = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for (model, metric), subset in frame.groupby(
        ["model", "metric"], sort=True
    ):
        values = subset["point_gap"].to_numpy(dtype=float)
        higher_is_better = bool(subset["higher_is_better"].iloc[0])
        favorable = values > 0 if higher_is_better else values < 0
        summary_rows.append(
            {
                "model": model,
                "metric": metric,
                "higher_is_better": higher_is_better,
                "seed_count": len(values),
                "point_gap_mean": float(np.mean(values)),
                "point_gap_sd": float(np.std(values, ddof=1)),
                "point_gap_min": float(np.min(values)),
                "point_gap_max": float(np.max(values)),
                "seeds_favoring_internal_performance": int(np.sum(favorable)),
                "note": (
                    "Across-seed dispersion of point gaps; not a "
                    "normal-theory confidence interval"
                ),
            }
        )
    return rows, summary_rows


def evaluation_groups() -> list[tuple[str, str, str]]:
    groups = [("temporal_2025", "temporal_test", "year_based")]
    for seed in (1103, 2207, 3301, 4409, 5501):
        groups.append(
            (
                f"random_seed_{seed}_internal_test",
                "random_internal_test",
                str(seed),
            )
        )
        groups.append(
            (
                f"random_seed_{seed}_2025_diagnostic",
                "random_2025_diagnostic",
                str(seed),
            )
        )
    return groups


def run_h2(
    protocol: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    iterations = int(protocol["common"]["iterations"])
    base_seed = int(protocol["common"]["base_seed"])
    metrics = list(protocol["common"]["metrics"])
    higher = dict(protocol["common"]["higher_is_better"])
    rows: list[dict[str, object]] = []
    for group_index, (group, design, seed) in enumerate(evaluation_groups()):
        logistic = load_prediction(group, "logistic_weighted")
        lightgbm = load_prediction(group, "lightgbm_weighted")
        if not logistic["meta_crash_id"].equals(lightgbm["meta_crash_id"]):
            raise AssertionError(f"CAS H2 model rows differ: {group}")
        if not logistic["target_severity"].equals(lightgbm["target_severity"]):
            raise AssertionError(f"CAS H2 targets differ: {group}")
        y_true = logistic["target_severity"].to_numpy(dtype=np.int8)
        pred_logistic = logistic["predicted_severity"].to_numpy(dtype=np.int8)
        pred_lightgbm = lightgbm["predicted_severity"].to_numpy(dtype=np.int8)
        rng = np.random.default_rng(base_seed + 500_000 + group_index * 10_000)
        cm_logistic, cm_lightgbm = paired_confusion_bootstrap(
            y_true,
            pred_logistic,
            pred_lightgbm,
            iterations,
            rng,
        )
        boot_logistic = metrics_from_confusion(cm_logistic)
        boot_lightgbm = metrics_from_confusion(cm_lightgbm)
        point_logistic = metrics_from_confusion(
            point_confusion(y_true, pred_logistic)
        )
        point_lightgbm = metrics_from_confusion(
            point_confusion(y_true, pred_lightgbm)
        )
        for metric in metrics:
            differences = boot_lightgbm[metric] - boot_logistic[metric]
            lower, upper, valid = percentile_interval(differences)
            rows.append(
                {
                    "evaluation_group": group,
                    "evaluation_design": design,
                    "seed": seed,
                    "metric": metric,
                    "higher_is_better": bool(higher[metric]),
                    "contrast": "lightgbm_minus_logistic",
                    "logistic_value": float(point_logistic[metric]),
                    "lightgbm_value": float(point_lightgbm[metric]),
                    "point_difference": float(
                        point_lightgbm[metric] - point_logistic[metric]
                    ),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "iterations": iterations,
                    "valid_iterations": valid,
                    "bootstrap_design": "paired_stratified",
                }
            )
    frame = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for (design, metric), subset in frame.groupby(
        ["evaluation_design", "metric"], sort=True
    ):
        values = subset["point_difference"].to_numpy(dtype=float)
        higher_is_better = bool(subset["higher_is_better"].iloc[0])
        favorable = values > 0 if higher_is_better else values < 0
        summary_rows.append(
            {
                "evaluation_design": design,
                "metric": metric,
                "higher_is_better": higher_is_better,
                "group_count": len(values),
                "point_difference_mean": float(np.mean(values)),
                "point_difference_sd": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else ""
                ),
                "point_difference_min": float(np.min(values)),
                "point_difference_max": float(np.max(values)),
                "groups_favoring_lightgbm": int(np.sum(favorable)),
            }
        )
    return rows, summary_rows


def validate_point_metrics(
    h2_rows: list[dict[str, object]],
) -> None:
    frozen = pd.read_csv(EVALUATION_METRICS_FILE, dtype={"seed": "string"})
    lookup = frozen.set_index(["evaluation_group", "model"])
    for row in h2_rows:
        metric = str(row["metric"])
        group = str(row["evaluation_group"])
        expected_logistic = float(lookup.loc[(group, "logistic_weighted"), metric])
        expected_lightgbm = float(lookup.loc[(group, "lightgbm_weighted"), metric])
        if not np.isclose(
            float(row["logistic_value"]), expected_logistic, rtol=0, atol=1e-14
        ):
            raise AssertionError(f"CAS H2 Logistic point metric differs: {group}/{metric}")
        if not np.isclose(
            float(row["lightgbm_value"]), expected_lightgbm, rtol=0, atol=1e-14
        ):
            raise AssertionError(f"CAS H2 LightGBM point metric differs: {group}/{metric}")


def main() -> None:
    protocol = load_protocol()
    if H1_FILE.exists() or H2_FILE.exists() or COMPLETE_FILE.exists():
        raise RuntimeError("CAS bootstrap outputs already exist; refusing to rerun")
    h1_rows, h1_summary = run_h1(protocol)
    h2_rows, h2_summary = run_h2(protocol)
    validate_point_metrics(h2_rows)
    write_csv(H1_FILE, h1_rows)
    write_csv(H1_SUMMARY_FILE, h1_summary)
    write_csv(H2_FILE, h2_rows)
    write_csv(H2_SUMMARY_FILE, h2_summary)

    h1 = pd.DataFrame(h1_summary)
    h2 = pd.DataFrame(h2_rows)
    lines = [
        "# CAS bootstrap uncertainty checkpoint",
        "",
        "Status: **PASS - H1_H2_BOOTSTRAP_COMPLETE**",
        "",
        "## Method",
        "",
        f"- Iterations: **{protocol['common']['iterations']}**.",
        "- H1 uses independent class-stratified resampling of internal and 2025 cohorts.",
        "- H2 uses paired class-stratified resampling of shared model-evaluation rows.",
        "- Intervals are percentile 95% intervals; no p-value threshold is used.",
        "",
        "## Selected point summaries",
        "",
    ]
    for model in ("logistic_weighted", "lightgbm_weighted"):
        row = h1.loc[
            h1["model"].eq(model) & h1["metric"].eq("macro_f1")
        ].iloc[0]
        lines.append(
            f"- H1 {model} Macro-F1 internal-minus-2025 mean: "
            f"**{row['point_gap_mean']:.4f}** "
            f"(SD **{row['point_gap_sd']:.4f}**)."
        )
    temporal = h2.loc[
        h2["evaluation_group"].eq("temporal_2025")
        & h2["metric"].isin(["macro_f1", "fatal_recall"])
    ]
    for _, row in temporal.iterrows():
        lines.append(
            f"- H2 temporal 2025 {row['metric']} LightGBM-minus-Logistic: "
            f"**{row['point_difference']:.4f}** "
            f"(95% interval **[{row['ci_lower']:.4f}, "
            f"{row['ci_upper']:.4f}]**)."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- A direction is described from the point estimate and interval; "
            "absence of interval exclusion of zero is not called equivalence.",
            "- CAS and STATS19 estimates remain separate and are not pooled.",
            "",
        ]
    )
    CHECKPOINT_FILE.write_text("\n".join(lines), encoding="utf-8")
    artifacts = [
        PROTOCOL_FILE,
        H1_FILE,
        H1_SUMMARY_FILE,
        H2_FILE,
        H2_SUMMARY_FILE,
        CHECKPOINT_FILE,
    ]
    build_artifact_manifest(artifacts, MANIFEST_FILE)
    write_json_atomic(
        COMPLETE_FILE,
        {
            "version": VERSION,
            "status": "H1_H2_BOOTSTRAP_COMPLETE",
            "created_local": datetime.now()
            .astimezone()
            .isoformat(timespec="seconds"),
            "protocol": relative(PROTOCOL_FILE),
            "protocol_sha256": hash_file(PROTOCOL_FILE),
            "h1_rows": len(h1_rows),
            "h1_summary_rows": len(h1_summary),
            "h2_rows": len(h2_rows),
            "h2_summary_rows": len(h2_summary),
            "artifact_manifest": relative(MANIFEST_FILE),
            "artifact_manifest_sha256": hash_file(MANIFEST_FILE),
            "post_test_model_selection": False,
        },
    )
    print(f"CAS_H1_BOOTSTRAP_ROWS={len(h1_rows)}")
    print(f"CAS_H2_BOOTSTRAP_ROWS={len(h2_rows)}")
    print("CAS_BOOTSTRAP_STATUS=H1_H2_COMPLETE")


if __name__ == "__main__":
    main()
