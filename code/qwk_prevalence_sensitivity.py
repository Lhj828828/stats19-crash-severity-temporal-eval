"""Conditional, post hoc prevalence standardization for Appendix A.14.6."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from threadpoolctl import threadpool_limits

from d12_bootstrap_uncertainty import confusion_from_vectors, metrics_from_confusions
from random_reference_revision import joint_bootstrap, stream_seed
import stats19_feature_revision as prep
import stats19_feature_revision_full as full

NAME = "qwk_prevalence_sensitivity"
VERSION = "POST_HOC_QWK_PREVALENCE_V1"
MODELS = ("logistic_weighted", "lightgbm_weighted")
SOURCE_FILES = (
    "run_qwk_prevalence_sensitivity.py", "code/qwk_prevalence_sensitivity.py",
    "tests/test_qwk_prevalence_sensitivity.py", "code/d12_bootstrap_uncertainty.py",
    "code/random_reference_revision.py", "code/stats19_feature_revision.py",
    "code/stats19_feature_revision_full.py",
)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def protocol_path(root):
    return root / "config" / NAME / "protocol.json"


def standardize_confusions(matrix, reference):
    """Row-weight counts to fixed true-class proportions; preserve row conditionals."""
    matrix = np.asarray(matrix, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if matrix.shape[-2:] != (3, 3) or reference.shape != (3,):
        raise ValueError("Expected 3x3 confusion matrices and three reference proportions")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Confusion counts must be finite and nonnegative")
    if not np.isfinite(reference).all() or (reference <= 0).any() or not np.isclose(reference.sum(), 1):
        raise ValueError("Reference must have three positive proportions summing to one")
    rows = matrix.sum(axis=-1)
    if (rows <= 0).any():
        raise ValueError("All true classes must be represented")
    proportions = rows / rows.sum(axis=-1, keepdims=True)
    return matrix * (reference / proportions)[..., :, None]


def ci(values):
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite bootstrap statistic")
    return np.quantile(values, [0.025, 0.975], method="linear").tolist()


def freeze(root):
    destination = protocol_path(root)
    if destination.exists():
        verify_protocol(root)
        print("Existing post hoc protocol verified; not rewritten.", flush=True)
        return
    evaluation = read_json(root / "logs/stats19_feature_revision/full/evaluation_complete.json")
    bootstrap = read_json(root / "logs/stats19_feature_revision/full/bootstrap_complete.json")
    if evaluation["version"] != full.VERSION or bootstrap["version"] != full.VERSION:
        raise ValueError("Requires final 15-feature evidence")
    input_files = [prep.ASSIGNMENTS, "config/stats19_feature_revision/full/execution_protocol.json",
                   "logs/stats19_feature_revision/full/evaluation_complete.json",
                   "logs/stats19_feature_revision/full/bootstrap_complete.json"]
    base = "results/stats19_feature_revision/full/"
    for name in ("test_metrics.csv", "confusion_matrices.csv", "h1_gaps.csv", "h1_summary.csv"):
        rel = base + name
        expected = evaluation["files_sha256"].get(rel, bootstrap["files_sha256"].get(rel))
        if expected is None or digest(root / rel) != expected:
            raise ValueError("Frozen result mismatch: " + rel)
        input_files.append(rel)
    for seed in prep.SEEDS:
        for cohort in ("internal", "2024"):
            group = f"random_seed_{seed}_{cohort}"
            for part, seal in (("evaluation", evaluation), ("bootstrap", bootstrap)):
                rel = base + part + "/" + group + ".npz"
                if digest(root / rel) != seal["files_sha256"][rel]:
                    raise ValueError("Frozen input mismatch: " + rel)
                input_files.append(rel)
    rules = {
        "version": VERSION, "created_local": datetime.now().astimezone().isoformat(),
        "status": "RULES_FIXED_BEFORE_NEW_STANDARDIZED_RESULTS",
        "timing": "Post hoc sensitivity motivated by known QWK results; not preregistered.",
        "models": list(MODELS), "random_seeds": list(prep.SEEDS),
        "scope": "Same fitted random model: internal 2018-2023 test versus 2024 diagnostic.",
        "labels": ["Slight", "Serious", "Fatal"], "ordinal_codes": [0, 1, 2],
        "reference": "Each seed's observed internal-test true-class proportions, fixed in all replicates.",
        "weights": "For true class k, w_k = p_internal(k) / p_2024(k).",
        "qwk": "Recompute QWK from the weighted full confusion matrix, including both marginals.",
        "gap": "QWK_internal - QWK_2024; positive means higher internal agreement.",
        "bootstrap": {
            "iterations": full.ITERATIONS, "master_seed": full.BOOTSTRAP_SEED,
            "streams": "Replay the final 15-feature group-specific streams exactly.",
            "strata": "True class; preserve observed class counts within each test set.",
            "independence": "Independent streams for internal and 2024 collision cohorts.",
            "pairing": "Same record resamples for models and for raw/standardized metrics in each cohort.",
            "implementation": "Existing joint-prediction-pattern multinomial resampling; record-bootstrap equivalent.",
            "replay": "Retain all four original prediction columns solely to reproduce existing draws; analyze two weighted models.",
            "interval": "95% percentile, linear quantiles, no multiplicity adjustment.",
            "conditional": "Conditions on fitted models and observed class proportions; excludes prevalence-estimation and refitting uncertainty.",
        },
        "across_seed": "Report mean, sample SD and range; no pooled confidence interval or independent-study interpretation.",
        "interpretation": "Standardization sensitivity is descriptive, not a causal decomposition or removal of all temporal shift.",
        "locks": {"model_fits": 0, "retuning": 0, "threshold_changes": 0, "overwrite_primary_results": False},
        "inputs_sha256": {name: digest(root / name) for name in input_files},
        "source_sha256": {name: digest(root / name) for name in SOURCE_FILES},
    }
    write_json(destination, rules)
    print("Post hoc rules saved before standardized-result calculation.", flush=True)


def verify_protocol(root):
    p = read_json(protocol_path(root))
    if p["version"] != VERSION:
        raise ValueError("Unknown sensitivity protocol")
    for name, expected in {**p["inputs_sha256"], **p["source_sha256"]}.items():
        if digest(root / name) != expected:
            raise ValueError("Protocol-bound file changed: " + name)
    return p


def analyze(root):
    protocol = verify_protocol(root)
    output = root / "results" / NAME
    completion = root / "logs" / NAME / "complete.json"
    if completion.exists():
        saved = read_json(completion)
        if saved["protocol_sha256"] != digest(protocol_path(root)):
            raise ValueError("Completed result belongs to another protocol")
        for name, expected in saved["outputs_sha256"].items():
            if digest(root / name) != expected:
                raise ValueError("Completed sensitivity output changed: " + name)
        print("Completed sensitivity verified; no recomputation needed.", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    base = root / "results/stats19_feature_revision/full"
    assignments = pd.read_csv(root / prep.ASSIGNMENTS, dtype={prep.ID: "string"}, keep_default_na=False)
    metrics = pd.read_csv(base / "test_metrics.csv", float_precision="round_trip")
    old_gaps = pd.read_csv(base / "h1_gaps.csv", float_precision="round_trip")
    old_confusions = pd.read_csv(base / "confusion_matrices.csv")
    rows, class_rows, audits, all_draws = [], [], [], {}
    for seed in prep.SEEDS:
        roles = prep.split_positions(assignments, f"random_seed_{seed}")
        matrices, boot, counts = {}, {}, {}
        future_y = future_predictions = None
        for cohort, role in (("internal", "internal"), ("2024", "future")):
            group = f"random_seed_{seed}_{cohort}"
            expected_rows = 96408 if cohort == "internal" else 100927
            with np.load(base / "evaluation" / (group + ".npz"), allow_pickle=False) as stored:
                positions, y, ids = [stored[n] for n in ("positions", "target", "ids")]
                np.testing.assert_array_equal(positions, roles[role])
                np.testing.assert_array_equal(y, assignments.iloc[positions][prep.TARGET])
                np.testing.assert_array_equal(ids, assignments.iloc[positions][prep.ID].to_numpy(dtype=str))
                if len(y) != expected_rows or len(np.unique(ids)) != expected_rows:
                    raise ValueError("Unexpected or duplicate cohort records")
                probabilities = [prep.check_probabilities(stored[name], len(y)) for name in full.MODELS]
                predictions = np.column_stack([p.argmax(axis=1) for p in probabilities])
            matrices[cohort] = confusion_from_vectors(y, predictions)
            counts[cohort] = np.bincount(y, minlength=3)
            rng = np.random.default_rng(stream_seed(full.BOOTSTRAP_SEED, full.VERSION + ":" + group))
            boot[cohort] = joint_bootstrap(y, predictions, rng, iterations=full.ITERATIONS)
            computed_draws = metrics_from_confusions(boot[cohort])["qwk"]
            with np.load(base / "bootstrap" / (group + ".npz"), allow_pickle=False) as original:
                np.testing.assert_allclose(computed_draws, original["qwk"], rtol=0, atol=1e-14)
            for index, name in enumerate(full.MODELS):
                reference = old_confusions.loc[old_confusions.group.eq(group) & old_confusions.model.eq(name)]
                reference = reference.sort_values(["true", "predicted"])["count"].to_numpy().reshape(3, 3)
                np.testing.assert_array_equal(matrices[cohort][index], reference)
                qwk = metrics_from_confusions(matrices[cohort][index])["qwk"]
                frozen_qwk = metrics.loc[metrics.group.eq(group) & metrics.model.eq(name), "qwk"].item()
                np.testing.assert_allclose(qwk, frozen_qwk, rtol=0, atol=1e-14)
            audits.append({"group": group, "rows": len(y), "class_counts": counts[cohort].tolist(),
                           "identifiers_targets_split_positions_match": True,
                           "all_original_qwk_draws_replayed": True, "confusions_match_frozen": True})
            if cohort == "2024":
                future_y, future_predictions = y, predictions
        reference = counts["internal"] / counts["internal"].sum()
        future_proportions = counts["2024"] / counts["2024"].sum()
        weights = reference / future_proportions
        adjusted = standardize_confusions(matrices["2024"], reference)
        adjusted_boot = standardize_confusions(boot["2024"], reference)
        np.testing.assert_allclose(adjusted.sum(axis=-1) / adjusted.sum(axis=(-2, -1))[:, None],
                                   np.tile(reference, (len(full.MODELS), 1)), atol=1e-14)
        for k, label in enumerate(protocol["labels"]):
            class_rows.append({"seed": seed, "class": label, "internal_count": int(counts["internal"][k]),
                               "future_count": int(counts["2024"][k]), "internal_proportion": reference[k],
                               "future_proportion": future_proportions[k], "future_record_weight": weights[k]})
        internal_qwk = metrics_from_confusions(matrices["internal"])["qwk"]
        future_qwk = metrics_from_confusions(matrices["2024"])["qwk"]
        standardized_qwk = metrics_from_confusions(adjusted)["qwk"]
        internal_draws = metrics_from_confusions(boot["internal"])["qwk"]
        raw_gap_draws = internal_draws - metrics_from_confusions(boot["2024"])["qwk"]
        standardized_gap_draws = internal_draws - metrics_from_confusions(adjusted_boot)["qwk"]
        for name in MODELS:
            index = list(full.MODELS).index(name)
            check_qwk = cohen_kappa_score(future_y, future_predictions[:, index], labels=[0, 1, 2],
                                          weights="quadratic", sample_weight=weights[future_y])
            np.testing.assert_allclose(check_qwk, standardized_qwk[index], rtol=0, atol=1e-12)
            raw_gap = float(internal_qwk[index] - future_qwk[index])
            adjusted_gap = float(internal_qwk[index] - standardized_qwk[index])
            raw_ci, adjusted_ci = ci(raw_gap_draws[:, index]), ci(standardized_gap_draws[:, index])
            original = old_gaps.loc[old_gaps.seed.eq(seed) & old_gaps.model.eq(name) & old_gaps.metric.eq("qwk")].iloc[0]
            np.testing.assert_allclose([raw_gap, *raw_ci], original[["gap", "ci_lower", "ci_upper"]].to_numpy(float),
                                       rtol=0, atol=1e-14)
            shift_ci = ci(standardized_gap_draws[:, index] - raw_gap_draws[:, index])
            rows.append({"seed": seed, "model": name, "internal_qwk": float(internal_qwk[index]),
                         "future_qwk": float(future_qwk[index]), "standardized_future_qwk": float(standardized_qwk[index]),
                         "raw_gap": raw_gap, "raw_ci_lower": raw_ci[0], "raw_ci_upper": raw_ci[1],
                         "standardized_gap": adjusted_gap, "standardized_ci_lower": adjusted_ci[0],
                         "standardized_ci_upper": adjusted_ci[1], "gap_change": adjusted_gap - raw_gap,
                         "gap_change_ci_lower": shift_ci[0], "gap_change_ci_upper": shift_ci[1]})
            all_draws[f"{seed}_{name}_raw"] = raw_gap_draws[:, index]
            all_draws[f"{seed}_{name}_standardized"] = standardized_gap_draws[:, index]
        print(f"Verified and standardized seed {seed}: no fitting or prediction calls.", flush=True)
    frame = pd.DataFrame(rows)
    summaries = []
    for name in MODELS:
        model_frame = frame.loc[frame.model.eq(name)]
        for metric in ("raw_gap", "standardized_gap", "gap_change"):
            values = model_frame[metric]
            summaries.append({"model": name, "metric": metric, "mean": values.mean(), "sample_sd": values.std(ddof=1),
                              "minimum": values.min(), "maximum": values.max(), "seeds": len(values)})
    for filename, table in (("qwk_gaps.csv", frame), ("class_weights.csv", pd.DataFrame(class_rows)),
                            ("seed_summary.csv", pd.DataFrame(summaries))):
        table.to_csv(output / filename, index=False, float_format="%.17g")
    np.savez_compressed(output / "bootstrap_draws.npz", **all_draws)
    write_json(output / "input_audit.json", audits)
    verify_protocol(root)
    result = {"status": "PASS", "completed_local": datetime.now().astimezone().isoformat(),
              "protocol_sha256": digest(protocol_path(root)), "post_hoc": True, "model_fits": 0,
              "prediction_calls": 0, "groups_verified": 10, "model_seed_comparisons": len(rows),
              "bootstrap_iterations": full.ITERATIONS, "original_intervals_reproduced": True,
              "independent_sklearn_weighted_qwk_check": True, "frozen_inputs_unchanged": True,
              "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn")},
              "outputs_sha256": {p.relative_to(root).as_posix(): digest(p) for p in sorted(output.iterdir()) if p.is_file()}}
    write_json(completion, result)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--stage", choices=("freeze", "analyze"), required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=8):
        (freeze if args.stage == "freeze" else analyze)(args.root.resolve())


if __name__ == "__main__":
    main()
