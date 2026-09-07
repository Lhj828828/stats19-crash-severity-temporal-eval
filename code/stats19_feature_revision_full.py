"""Post-review analytical run; never overwrites the preparation or legacy run.

Training and selection reuse the frozen preparation implementation. All six
model sets must be sealed before any revised test-performance calculation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import importlib.metadata
import io
from pathlib import Path
import platform
import threading
import time
import unittest

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score, log_loss
from threadpoolctl import threadpool_limits

import stats19_feature_revision as prep
from d12_bootstrap_uncertainty import confusion_from_vectors, metrics_from_confusions
from d14_shap_stability import (
    build_samples, explain, bootstrap_mean_values, spearman_rows,
    BOOTSTRAP_ITERATIONS, BOOTSTRAP_MASTER_SEED, SAMPLE_MASTER_SEED,
)
from random_reference_revision import joint_bootstrap, interval, stream_seed

ROOT = prep.ROOT
VERSION = "STATS19_15_FEATURE_FULL_V1"
THREADS = 8
ITERATIONS = 2000
BOOTSTRAP_SEED = 20260828
SLUGS = ("temporal", *(f"random_seed_{s}" for s in prep.SEEDS))
MODELS = prep.MODEL_NAMES
LOWER = {"ordinal_mae", "mean_asymmetric_cost"}
SCOPES = ("overall", "Slight", "Serious", "Fatal")
PARENT_HASH = "d0090f1b292cb30e784eef7ff5792e507033e40f252207254e0fee71fb67a730"
EXECUTION_SOURCES = ("run_stats19_feature_revision_full.py", "code/stats19_feature_revision_full.py",
                     "tests/test_stats19_feature_revision_full.py")


def path(root, family, *parts):
    return prep.output_path(root, family, "full", *parts)


def protocol_path(root):
    return path(root, "config", "execution_protocol.json")


def write_csv(destination, rows):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(destination)
    return destination


def save_npz(destination, **arrays):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(destination)
    return destination


def hashes(root, files):
    return {p.relative_to(root).as_posix(): prep.digest(p) for p in files}


def groups():
    out = [{"group": "temporal_2024", "split": "temporal", "role": "future", "rows": 100927}]
    for seed in prep.SEEDS:
        slug = f"random_seed_{seed}"
        out.extend([{"group": slug + "_internal", "split": slug, "role": "internal", "rows": 96408},
                    {"group": slug + "_2024", "split": slug, "role": "future", "rows": 100927}])
    return out


def preflight(root):
    if protocol_path(root).exists():
        require_protocol(root)
        print("Frozen preflight evidence verified; not overwritten.", flush=True)
        return
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_stats19_feature_revision*.py")
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {"version": VERSION, "created_local": prep.now(), "passed": result.wasSuccessful(),
              "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "source_sha256": hashes(ROOT, [ROOT / name for name in EXECUTION_SOURCES]), "output": stream.getvalue()}
    prep.write_json(path(root, "logs", "preflight_tests.json"), report)
    if not result.wasSuccessful():
        raise RuntimeError("Preflight tests failed; see preflight_tests.json")
    print(f"PREFLIGHT passed: {result.testsRun} tests", flush=True)


def freeze(root):
    if protocol_path(root).exists():
        require_protocol(root)
        print("Existing full-execution protocol verified; not overwritten.", flush=True)
        return
    parent = prep.require_protocol(root)
    if prep.digest(prep.output_path(root, "config", "protocol.json")) != PARENT_HASH:
        raise ValueError("Unexpected preparation freeze")
    ready_path = prep.output_path(root, "logs", "preparation_complete.json")
    ready = prep.read_json(ready_path)
    if ready["status"] != prep.PREPARATION_STATUS or ready["protocol_sha256"] != PARENT_HASH:
        raise ValueError("Preparation has not passed")
    preflight_path = path(root, "logs", "preflight_tests.json")
    tests = prep.read_json(preflight_path)
    if not tests["passed"] or tests["tests_run"] < 54:
        raise ValueError("Full execution preflight tests have not passed")
    prep.check_hashes(ROOT, tests["source_sha256"])
    prep.check_hashes(root, prep.read_json(root / parent["historical_manifest"]))
    smoke = prep.read_json(prep.output_path(root, "logs", "smoke_complete.json"))
    prep.check_hashes(root, smoke["artifact_sha256"])
    source = sorted((ROOT / "code").glob("*.py")) + [
        ROOT / "run_stats19_feature_revision_full.py", ROOT / "tests/test_stats19_feature_revision_full.py",
        ROOT / "requirements-lock.txt",
    ]
    if BOOTSTRAP_ITERATIONS != ITERATIONS or BOOTSTRAP_MASTER_SEED != BOOTSTRAP_SEED:
        raise ValueError("Imported SHAP bootstrap contract differs")
    report = {
        "version": VERSION, "created_local": prep.now(), "parent_sha256": PARENT_HASH,
        "post_review_correction": True, "old_test_results_known": True,
        "preregistered": False, "new_unseen_test_claim": False,
        "source_sha256": hashes(ROOT, source), "preparation_evidence_sha256": hashes(root, [ready_path, preflight_path]),
        "threads": THREADS, "models": list(MODELS), "splits": list(SLUGS), "evaluation_groups": groups(),
        "training": "Frozen preparation.fit_validation, complete training partitions, six candidates per split, cap 1200, training-only refit",
        "constraints": "new same-split weighted Logistic QWK minus 0.01 and fatal recall minus 0.05; original deterministic fallback",
        "logistic_nonconvergence": "Stop before sealing; retain diagnostics, require documented amendment rather than silent iteration increase",
        "checkpoint": "Completed splits immutable and hash-validated; unfinished split restarts; SHAP resumes verified 1000-row chunks",
        "test_gate": "All six model sets and validation reports sealed before evaluation; no retuning from tests",
        "decision": "three-class argmax, no threshold tuning",
        "bootstrap": {"iterations": ITERATIONS, "seed": BOOTSTRAP_SEED,
                      "within_group": "paired record bootstrap, fixed true-class counts, exact joint-prediction-pattern multinomial kernel",
                      "H1": "independent internal and future streams per seed; error metrics reversed; mean/sample-SD/range across five seeds",
                      "H2": "temporal weighted LightGBM minus weighted Logistic; report all paired intervals",
                      "H2_descriptive_gate": "Macro-F1 delta >=0.01 and lower CI>0, QWK/fatal upper CI>=0 and cost lower CI<=0; NOT evidence of noninferiority"},
        "shap": {"rows": 10000, "sampling_seed": SAMPLE_MASTER_SEED, "features": list(prep.FEATURES),
                 "design": "reuse original position sampling; same random model on internal/future; temporal future descriptive only",
                 "output": "raw margin, tree_path_dependent; mean absolute over records and equal average of three output classes",
                 "scope_caution": "Fatal-output importance uses all sampled records, not a fatal-record-only estimate",
                 "bootstrap": "2000 stratified cached-contribution record draws, no refitting; shared future positions and stream across models",
                 "direction": "signed summaries for separate class outputs only; no pooled signed risk direction or causal claim",
                 "correlated_groups": "not computed; individual-feature rankings only, no correlation-group stability claim"},
        "secondary": "Not rerun here: Ordered Logit/matched subset, exclusion-2020, tree-cap extension. Historical results must not enter corrected main tables.",
        "CAS": "Unchanged; not included in this execution",
        "runtime": {"python": platform.python_version(), **{p: importlib.metadata.version(p) for p in
                    ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "shap", "joblib", "threadpoolctl")}},
    }
    prep.write_json(protocol_path(root), report)
    print("Full-execution protocol frozen before analytical fits.", flush=True)


def require_protocol(root):
    prep.require_protocol(root)
    protocol = prep.read_json(protocol_path(root))
    if (protocol["version"] != VERSION or protocol["parent_sha256"] != PARENT_HASH
            or protocol["evaluation_groups"] != groups() or protocol["threads"] != THREADS):
        raise ValueError("Full execution contract changed")
    if prep.digest(prep.output_path(root, "config", "protocol.json")) != PARENT_HASH:
        raise ValueError("Preparation contract changed")
    prep.check_hashes(ROOT, protocol["source_sha256"])
    prep.check_hashes(root, protocol["preparation_evidence_sha256"])
    for package, version in protocol["runtime"].items():
        current = platform.python_version() if package == "python" else importlib.metadata.version(package)
        if current != version:
            raise ValueError(f"Runtime changed: {package}")
    return protocol


def seal(root, name, files, **details):
    destination = path(root, "logs", name + ".json")
    if destination.exists():
        raise FileExistsError("Refusing to overwrite completed checkpoint: " + name)
    payload = {"status": "COMPLETE", "version": VERSION, "created_local": prep.now(),
               "execution_sha256": prep.digest(protocol_path(root)), "files_sha256": hashes(root, files), **details}
    prep.write_json(destination, payload)
    return destination


def checked_seal(root, name, required=True):
    destination = path(root, "logs", name + ".json")
    if not destination.exists() and not required:
        return None
    payload = prep.read_json(destination)
    if (payload.get("status") != "COMPLETE" or payload.get("version") != VERSION
            or payload.get("execution_sha256") != prep.digest(protocol_path(root))):
        raise ValueError("Stale or incomplete checkpoint: " + name)
    prep.check_hashes(root, payload["files_sha256"])
    return payload


def require_all_models(root):
    complete = checked_seal(root, "training_complete")
    if complete["splits"] != list(SLUGS) or complete["test_performance_evaluated"]:
        raise ValueError("Test gate requires all six validation-only model sets")
    for slug in SLUGS:
        record = checked_seal(root, "training/" + slug)
        if record["split"] != slug or record["test_performance_evaluated"]:
            raise ValueError("Invalid split training seal")
    return complete


@contextmanager
def progress(label):
    started = time.perf_counter()
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(30):
            print(f"RUNNING {label}; elapsed {(time.perf_counter()-started)/60:.1f} min; thread cap {THREADS}", flush=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def load_model(root, slug, name):
    artifact = joblib.load(path(root, "models", slug, name + ".joblib"))
    prep.require_analytical_artifact(artifact)
    if (artifact.get("smoke_only") is not False or artifact.get("split") != slug
            or artifact.get("model_name") != name
            or artifact.get("execution_sha256") != prep.digest(protocol_path(root))):
        raise ValueError("Model identity mismatch")
    return artifact


def assert_metrics(expected, actual):
    for key, value in expected.items():
        np.testing.assert_allclose(value, actual[key], rtol=0, atol=1e-12, err_msg=key)


def train(root):
    require_protocol(root)
    if checked_seal(root, "training_complete", required=False):
        require_all_models(root)
        return
    schema, X, y, metadata, assignments = prep.load_data(root)
    rules = prep.read_json(prep.output_path(root, "config", "protocol.json"))["rules"]
    split_seals = []
    for slug in SLUGS:
        checkpoint = "training/" + slug
        if checked_seal(root, checkpoint, required=False):
            split_seals.append(path(root, "logs", checkpoint + ".json"))
            print("Verified completed model set: " + slug, flush=True)
            continue
        roles = prep.split_positions(assignments, slug)
        data = prep.selection_data(X, y, roles)
        print(f"TRAIN {slug}: {len(data.train)} train / {len(data.validation)} validation; 6 candidates", flush=True)
        started = time.perf_counter()
        with progress("training " + slug):
            result = prep.fit_validation(data, schema, rules, 1200, THREADS)
        diagnostic = path(root, "logs", "validation", slug + ".json")
        details = {key: value for key, value in result.items() if key not in {"artifacts", "probabilities"}}
        details.update(split=slug, train_rows=len(data.train), validation_rows=len(data.validation),
                       seconds=time.perf_counter()-started, test_performance_evaluated=False,
                       role_hashes={role: prep.position_hash(pos) for role, pos in roles.items()})
        prep.write_json(diagnostic, details)
        if any(v["convergence_warning"] for v in result["runtime"].values()):
            raise RuntimeError("Full Logistic nonconvergence; diagnostics retained, no model seal or test evaluation")
        files = [diagnostic]
        for name, artifact in result["artifacts"].items():
            artifact.update(feature_version=prep.VERSION, analytical_result=True, smoke_only=False,
                            split=slug, model_name=name, execution_sha256=prep.digest(protocol_path(root)),
                            train_positions_sha256=prep.position_hash(roles["train"]),
                            validation_positions_sha256=prep.position_hash(roles["validation"]))
            destination = path(root, "models", slug, name + ".joblib")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".joblib.tmp")
            joblib.dump(artifact, temporary, compress=3)
            temporary.replace(destination)
            restored = load_model(root, slug, name)
            np.testing.assert_allclose(prep.predict_artifact(restored, data.validation), result["probabilities"][name],
                                       rtol=1e-12, atol=1e-12)
            files.append(destination)
        files.append(save_npz(path(root, "results", slug, "validation_predictions.npz"),
                              positions=roles["validation"], ids=metadata.iloc[roles["validation"]][prep.ID].to_numpy(dtype=str),
                              target=data.validation_target, **result["probabilities"]))
        files.append(write_csv(path(root, "results", slug, "candidate_metrics.csv"), result["candidate_metrics"]))
        files.append(write_csv(path(root, "results", slug, "validation_metrics.csv"),
                               [{"model": name, **values} for name, values in result["metrics"].items()]))
        split_seals.append(seal(root, checkpoint, files, split=slug, train_rows=len(data.train),
                                validation_rows=len(data.validation), test_performance_evaluated=False))
        print(f"SEALED {slug}: {result['selected']['candidate_id']}, iteration {result['selected']['best_iteration']}, "
              f"constraints met={result['constraints_met']}, {details['seconds']:.1f}s", flush=True)
        del data, result
        gc.collect()
    seal(root, "training_complete", split_seals, splits=list(SLUGS), test_performance_evaluated=False)


def full_metrics(y, probabilities):
    p = prep.check_probabilities(probabilities, len(y))
    predicted = p.argmax(axis=1)
    result = prep.classification_metrics(y, predicted)
    matrix = confusion_matrix(y, predicted, labels=[0, 1, 2])
    for c in range(3):
        result[f"class{c}_precision"] = float(matrix[c, c] / matrix[:, c].sum()) if matrix[:, c].sum() else 0.0
        result[f"class{c}_recall"] = float(matrix[c, c] / matrix[c].sum())
    result["log_loss"] = float(log_loss(y, p, labels=[0, 1, 2]))
    return result, matrix


def prediction_path(root, group):
    return path(root, "results", "evaluation", group + ".npz")


def aligned_predictions(root, group, positions, target, ids):
    with np.load(prediction_path(root, group)) as stored:
        np.testing.assert_array_equal(stored["positions"], positions)
        np.testing.assert_array_equal(stored["target"], target)
        np.testing.assert_array_equal(stored["ids"], np.asarray(ids, dtype=str))
        return {name: prep.check_probabilities(stored[name], len(target)).copy() for name in MODELS}


def evaluate(root):
    require_protocol(root)
    require_all_models(root)
    if checked_seal(root, "evaluation_complete", required=False):
        return
    _, X, y, metadata, assignments = prep.load_data(root)
    metrics, confusions, files = [], [], []
    for group in groups():
        slug, key = group["split"], group["group"]
        roles = prep.split_positions(assignments, slug)
        pos = roles[group["role"]]
        target = y.iloc[pos].to_numpy()
        probability = {}
        for name in MODELS:
            artifact = load_model(root, slug, name)
            p = prep.predict_artifact(artifact, X.iloc[pos])
            probability[name] = p
            values, matrix = full_metrics(target, p)
            metrics.append({**group, "model": name, **values})
            for true in range(3):
                for pred in range(3):
                    confusions.append({"group": key, "model": name, "true": true, "predicted": pred,
                                       "count": int(matrix[true, pred])})
        files.append(save_npz(prediction_path(root, key), positions=pos, target=target,
                              ids=metadata.iloc[pos][prep.ID].to_numpy(dtype=str), **probability))
        print("EVALUATED " + key, flush=True)
    files.append(write_csv(path(root, "results", "test_metrics.csv"), metrics))
    files.append(write_csv(path(root, "results", "confusion_matrices.csv"), confusions))
    files.append(path(root, "logs", "training_complete.json"))
    seal(root, "evaluation_complete", files, evaluation_groups=groups(), model_group_count=44,
         no_test_driven_selection=True, historical_test_results_already_known=True)


def oriented_gap(a, b, metric):
    return (-1 if metric in LOWER else 1) * (a - b)


def descriptive_h2(rows):
    selected = {r["metric"]: r for r in rows if r["group"] == "temporal_2024"
                and r["reference"] == "logistic_weighted"}
    macro, qwk, fatal, cost = [selected[k] for k in ("macro_f1", "qwk", "fatal_recall", "mean_asymmetric_cost")]
    conditions = {"macro_gain_at_least_001_and_lower_ci_positive": macro["delta"] >= .01 and macro["ci_lower"] > 0,
                  "qwk_interval_not_wholly_negative": qwk["ci_upper"] >= 0,
                  "fatal_interval_not_wholly_negative": fatal["ci_upper"] >= 0,
                  "cost_interval_not_wholly_positive": cost["ci_lower"] <= 0}
    return {"conditions": conditions, "joint_descriptive_gate": all(conditions.values()),
            "not_noninferiority_evidence": True, "all_temporal_comparator_intervals": list(selected.values())}


def bootstrap(root):
    require_protocol(root)
    require_all_models(root)
    checked_seal(root, "evaluation_complete")
    if checked_seal(root, "bootstrap_complete", required=False):
        return
    _, _, y, metadata, assignments = prep.load_data(root)
    estimates, draws_by_group, points_by_group, pairs, files = [], {}, {}, [], []
    for group in groups():
        key = group["group"]
        pos = prep.split_positions(assignments, group["split"])[group["role"]]
        target = y.iloc[pos].to_numpy()
        probabilities = aligned_predictions(root, key, pos, target, metadata.iloc[pos][prep.ID])
        predictions = np.column_stack([probabilities[name].argmax(axis=1) for name in MODELS])
        points = metrics_from_confusions(confusion_from_vectors(target, predictions))
        rng = np.random.default_rng(stream_seed(BOOTSTRAP_SEED, VERSION + ":" + key))
        draws = metrics_from_confusions(joint_bootstrap(target, predictions, rng, iterations=ITERATIONS))
        draws_by_group[key], points_by_group[key] = draws, points
        files.append(save_npz(path(root, "results", "bootstrap", key + ".npz"), **draws))
        for metric in points:
            for index, name in enumerate(MODELS):
                estimates.append({"group": key, "model": name, "metric": metric,
                                  "estimate": float(points[metric][index]), **interval(draws[metric][:, index])})
            for reference in range(3):
                differences = draws[metric][:, 3] - draws[metric][:, reference]
                pairs.append({"group": key, "candidate": MODELS[3], "reference": MODELS[reference], "metric": metric,
                              "delta": float(points[metric][3] - points[metric][reference]), **interval(differences),
                              "design": "paired_true_class_stratified", "iterations": ITERATIONS})
    gaps = []
    for seed in prep.SEEDS:
        a, b = f"random_seed_{seed}_internal", f"random_seed_{seed}_2024"
        for metric in points_by_group[a]:
            for index, name in enumerate(MODELS):
                samples = oriented_gap(draws_by_group[a][metric][:, index], draws_by_group[b][metric][:, index], metric)
                point = oriented_gap(points_by_group[a][metric][index], points_by_group[b][metric][index], metric)
                gaps.append({"seed": seed, "model": name, "metric": metric, "gap": float(point), **interval(samples),
                             "design": "independent_true_class_stratified", "iterations": ITERATIONS})
    summary = pd.DataFrame(gaps).groupby(["model", "metric"], sort=False).gap.agg(["mean", "std", "min", "max"]).reset_index()
    for name, rows in (("metric_intervals", estimates), ("paired_differences", pairs), ("h1_gaps", gaps), ("h1_summary", summary)):
        files.append(write_csv(path(root, "results", name + ".csv"), rows))
    h2 = path(root, "results", "h2_descriptive_decision.json")
    prep.write_json(h2, descriptive_h2(pairs))
    files.extend([h2, path(root, "logs", "evaluation_complete.json")])
    seal(root, "bootstrap_complete", files, groups=11, iterations=ITERATIONS,
         uncertainty="within-seed test-sampling intervals separate from across-seed descriptive sample SD/range")
    print("BOOTSTRAP complete: 11 groups; paired differences and H1 independent-cohort intervals.", flush=True)


def scope_values(array, scope):
    if array.shape[-2:] != (15, 3):
        raise ValueError("Expected 15 features and three output classes")
    return array.mean(axis=-1) if scope == "overall" else array[..., SCOPES.index(scope) - 1]


def shap_analysis(root):
    require_protocol(root)
    require_all_models(root)
    checked_seal(root, "evaluation_complete")
    if checked_seal(root, "shap_complete", required=False):
        return
    _, X, y, metadata, assignments = prep.load_data(root)
    samples, sample_frame = build_samples(y, metadata, assignments)
    sample_file = path(root, "results", "shap", "explanation_samples.csv.gz")
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    sample_frame.to_csv(sample_file, index=False, compression={"method": "gzip", "mtime": 0})
    files, importance, directions, stability, audits, counts = [sample_file], [], [], [], [], []
    for slug in SLUGS:
        artifact = load_model(root, slug, "lightgbm_weighted")
        model_sha = prep.digest(path(root, "models", slug, "lightgbm_weighted.joblib"))
        cohort_keys = {"future": "future_2024"} if slug == "temporal" else {
            "internal": slug + "_internal", "future": "future_2024"}
        means, boot = {}, {}
        for cohort, sample_key in cohort_keys.items():
            pos = samples[sample_key]
            labels = y.iloc[pos].to_numpy()
            roles = prep.split_positions(assignments, slug)
            if not np.isin(pos, roles[cohort]).all():
                raise ValueError("Explanation positions outside held-out cohort")
            chunks = []
            for start in range(0, len(pos), 1000):
                selected = pos[start:start+1000]
                destination = path(root, "results", "shap", slug, f"{cohort}_{start:05d}.npz")
                sidecar = destination.with_suffix(".json")
                if destination.exists() and sidecar.exists():
                    checkpoint = prep.read_json(sidecar)
                    if checkpoint != {"sha256": prep.digest(destination), "model_sha256": model_sha,
                                      "execution_sha256": prep.digest(protocol_path(root))}:
                        raise ValueError("SHAP checkpoint mismatch")
                    with np.load(destination) as saved:
                        np.testing.assert_array_equal(saved["positions"], selected)
                        values, error = saved["values"].copy(), float(saved["additivity_error"])
                else:
                    print(f"SHAP {slug}/{cohort} {start+1}-{start+len(selected)}/{len(pos)}", flush=True)
                    with progress("SHAP " + slug + "/" + cohort):
                        values, expected, audit = explain(X, selected, artifact["bundle"], artifact)
                    error = audit["max_abs_additivity_error"]
                    save_npz(destination, positions=selected, values=values, expected=expected, additivity_error=error)
                    prep.write_json(sidecar, {"sha256": prep.digest(destination), "model_sha256": model_sha,
                                              "execution_sha256": prep.digest(protocol_path(root))})
                if values.shape != (len(selected), 15, 3) or not np.isfinite(values).all() or not 0 <= error <= 1e-6:
                    raise ValueError("Invalid SHAP contribution checkpoint")
                chunks.append(values)
                files.extend([destination, sidecar])
                audits.append({"split": slug, "cohort": cohort, "start": start, "rows": len(selected), "max_additivity_error": error})
            values = np.concatenate(chunks)
            means[cohort] = np.abs(values).mean(axis=0)
            boot_path = path(root, "results", "shap", slug, cohort + "_bootstrap.npz")
            boot[cohort] = bootstrap_mean_values(np.abs(values), labels, stream_key=VERSION + ":shap:" + sample_key)
            files.append(save_npz(boot_path, means=boot[cohort], positions=pos, target=labels))
            counts.append({"split": slug, "cohort": cohort, "rows": len(pos),
                           **{f"class{c}_count": int(np.sum(labels == c)) for c in range(3)}})
            for scope in SCOPES:
                values_scope = scope_values(means[cohort], scope)
                ranks = rankdata(-values_scope, method="average")
                for index, feature in enumerate(prep.FEATURES):
                    importance.append({"split": slug, "cohort": cohort, "scope": scope, "feature": feature,
                                       "mean_abs_shap": float(values_scope[index]), "rank": float(ranks[index])})
            for class_index, scope in enumerate(SCOPES[1:]):
                for index, feature in enumerate(prep.FEATURES):
                    signed = values[:, index, class_index]
                    directions.append({"split": slug, "cohort": cohort, "output_class": scope, "feature": feature,
                                       "mean_signed_shap": float(signed.mean()), "positive_fraction": float((signed > 0).mean()),
                                       "negative_fraction": float((signed < 0).mean())})
        if slug != "temporal":
            for scope in SCOPES:
                a, b = [scope_values(means[c], scope) for c in ("internal", "future")]
                rho = float(spearman_rows(a[None, :], b[None, :])[0])
                draws = spearman_rows(scope_values(boot["internal"], scope), scope_values(boot["future"], scope))
                stability.append({"seed": int(slug.rsplit("_", 1)[1]), "scope": scope, "rho": rho, **interval(draws),
                                  "internal_rows": 10000, "future_rows": 10000, "iterations": ITERATIONS})
                files.append(save_npz(path(root, "results", "shap", slug, scope + "_rank_draws.npz"), rho=draws))
        print("SHAP model complete: " + slug, flush=True)
    summary = pd.DataFrame(stability).groupby("scope", sort=False).rho.agg(["mean", "std", "min", "max"]).reset_index()
    for name, rows in (("shap_importance", importance), ("shap_class_directions", directions), ("shap_stability", stability),
                       ("shap_summary", summary), ("shap_sample_counts", counts), ("shap_additivity_audit", audits)):
        files.append(write_csv(path(root, "results", name + ".csv"), rows))
    files.append(path(root, "logs", "evaluation_complete.json"))
    seal(root, "shap_complete", files, models=6, cohorts=11, paired_stability_models=5, iterations=ITERATIONS)


def create_summary(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    h1 = pd.read_csv(path(root, "results", "h1_gaps.csv"), float_precision="round_trip")
    h3 = pd.read_csv(path(root, "results", "shap_stability.csv"), float_precision="round_trip")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    for ax, metric, title in zip(axes, ("macro_f1", "qwk", "fatal_recall"), ("Macro-F1 gap", "QWK gap", "Fatal recall gap")):
        frame = h1.loc[h1.model.eq("lightgbm_weighted") & h1.metric.eq(metric)].sort_values("seed")
        positions = np.arange(5)
        ax.hlines(positions, frame.ci_lower, frame.ci_upper, color="#007C91")
        ax.scatter(frame.gap, positions, color="#007C91", zorder=3)
        ax.axvline(0, color="#888888", linewidth=.8)
        ax.set(yticks=positions, yticklabels=frame.seed.astype(str), title=title, xlabel="Internal minus 2024")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        destination = path(root, "figures", "h1_same_model_gaps." + extension)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=220)
        files.append(destination)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    frame = h3.loc[h3.scope.eq("overall")].sort_values("seed")
    positions = np.arange(5)
    ax.hlines(positions, frame.ci_lower, frame.ci_upper, color="#007C91")
    ax.scatter(frame.rho, positions, color="#007C91", zorder=3)
    ax.set(yticks=positions, yticklabels=frame.seed.astype(str), xlim=(-1.02, 1.02),
           xlabel="Spearman rank correlation", ylabel="Random seed", title="15-feature SHAP rank stability")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        destination = path(root, "figures", "shap_rank_stability." + extension)
        fig.savefig(destination, dpi=220)
        files.append(destination)
    plt.close(fig)
    h2 = prep.read_json(path(root, "results", "h2_descriptive_decision.json"))
    metrics = pd.read_csv(path(root, "results", "test_metrics.csv"), float_precision="round_trip")
    lines = ["# Post-review 15-feature analytical run", "", "Legacy outcomes were known before this correction. This is not a newly unseen test or a preregistration.",
             "", "## Temporal 2024 evaluation", "", "Model | Macro-F1 | QWK | Fatal recall | Mean cost", "--- | ---: | ---: | ---: | ---:"]
    for row in metrics.loc[metrics.group.eq("temporal_2024")].to_dict("records"):
        lines.append(f"{row['model']} | {row['macro_f1']:.6f} | {row['qwk']:.6f} | {row['fatal_recall']:.6f} | {row['mean_asymmetric_cost']:.6f}")
    lines.extend(["", "## Interpretation boundaries", "",
                  f"H2 joint descriptive gate: {h2['joint_descriptive_gate']}. This is not a noninferiority finding.",
                  "H1 intervals represent test-sampling uncertainty within each seed. Across-seed mean, sample SD and range are separate descriptive quantities.",
                  "H3 correlates 15 feature mean-absolute raw-margin SHAP importance values, using the same fitted random model on internal and future cohorts.",
                  "Fatal-output SHAP uses all explanation records, not only fatal records. Output-specific signs are associative, not causal.",
                  "No spatial generalization or cross-country universality is established by this run.", "",
                  "## Not completed by this run", "",
                  "Ordered Logit, matched-subset, exclusion-2020 and tree-count sensitivity analyses have not been rerun on 15 fields.",
                  "Those historical 17-feature results cannot be merged into corrected main results. CAS is unchanged.",
                  "Repository publication, software DOI update and clean-room public reproduction of this revision are not performed here.",
                  "Two previously recorded legacy D14 protocol-hash test failures remain separate; this run does not erase or repair historical freezes.", ""])
    destination = path(root, "results", "SUMMARY.md")
    destination.write_text("\n".join(lines), encoding="utf-8")
    files.append(destination)
    return files


def verify(root):
    require_protocol(root)
    require_all_models(root)
    for name in ("evaluation_complete", "bootstrap_complete", "shap_complete"):
        checked_seal(root, name)
    _, X, y, metadata, assignments = prep.load_data(root)
    checks = []
    for slug in SLUGS:
        roles = prep.split_positions(assignments, slug)
        report = prep.read_json(path(root, "logs", "validation", slug + ".json"))
        selected, eligible, constraints = prep.local_selection(report["candidate_metrics"], report["metrics"]["logistic_weighted"])
        if selected != report["selected"] or eligible != report["constraints_met"] or constraints != report["constraints"]:
            raise ValueError("Split-local candidate selection differs")
        for role, pos in roles.items():
            if report["role_hashes"][role] != prep.position_hash(pos):
                raise ValueError("Training/validation/test identities differ")
        with np.load(path(root, "results", slug, "validation_predictions.npz")) as saved:
            np.testing.assert_array_equal(saved["positions"], roles["validation"])
            np.testing.assert_array_equal(saved["target"], y.iloc[roles["validation"]])
            for name in MODELS:
                assert_metrics(report["metrics"][name], prep.classification_metrics(saved["target"], saved[name].argmax(axis=1)))
                artifact = load_model(root, slug, name)
                if artifact["train_positions_sha256"] != prep.position_hash(roles["train"]):
                    raise ValueError("Model training membership differs")
                for col, vocabulary in artifact["bundle"]["category_vocabulary"].items():
                    if vocabulary != sorted(str(v) for v in X.iloc[roles["train"]][col].unique()):
                        raise ValueError("Categorical vocabulary not training-only")
                weights = artifact["estimator"].get_params().get("class_weight")
                expected = prep.weights_from_training(y.iloc[roles["train"]]) if name in MODELS[2:] else None
                if weights != expected:
                    raise ValueError("Model class weights differ from training counts")
                if "preprocessor" in artifact:
                    median = artifact["preprocessor"].named_transformers_["numeric"].named_steps["imputer"].statistics_[0]
                    np.testing.assert_allclose(median, np.nanmedian(X.iloc[roles["train"]].feature_speed_limit), rtol=0, atol=0)
                if name == "lightgbm_weighted" and artifact["estimator"].get_params()["n_estimators"] != selected["best_iteration"]:
                    raise ValueError("Model iteration differs from selection")
                offsets = np.unique(np.linspace(0, len(roles["validation"])-1, 73, dtype=int))
                np.testing.assert_allclose(prep.predict_artifact(artifact, X.iloc[roles["validation"][offsets]]),
                                           saved[name][offsets], rtol=1e-12, atol=1e-12)
        checks.append({"split": slug, "selection_overlap": 0, "local_constraint_verified": True,
                       "training_preprocessor_and_weights_verified": True})
    metrics = pd.read_csv(path(root, "results", "test_metrics.csv"), float_precision="round_trip")
    intervals = pd.read_csv(path(root, "results", "metric_intervals.csv"), float_precision="round_trip")
    pairs = pd.read_csv(path(root, "results", "paired_differences.csv"), float_precision="round_trip")
    confusion_rows = pd.read_csv(path(root, "results", "confusion_matrices.csv"))
    if len(metrics) != 44 or len(confusion_rows) != 396 or len(intervals) != 308 or len(pairs) != 231:
        raise ValueError("Incomplete analytical output tables")
    cached_draws, point_groups = {}, {}
    for group in groups():
        key, slug = group["group"], group["split"]
        pos = prep.split_positions(assignments, slug)[group["role"]]
        target = y.iloc[pos].to_numpy()
        probabilities = aligned_predictions(root, key, pos, target, metadata.iloc[pos][prep.ID])
        predictions = np.column_stack([probabilities[name].argmax(axis=1) for name in MODELS])
        points = metrics_from_confusions(confusion_from_vectors(target, predictions))
        regenerated = metrics_from_confusions(joint_bootstrap(target, predictions,
            np.random.default_rng(stream_seed(BOOTSTRAP_SEED, VERSION + ":" + key)), iterations=ITERATIONS))
        with np.load(path(root, "results", "bootstrap", key + ".npz")) as saved:
            cached_draws[key] = {metric: saved[metric].copy() for metric in points}
        point_groups[key] = points
        for index, name in enumerate(MODELS):
            p = probabilities[name]
            expected, matrix = full_metrics(target, p)
            actual = metrics.loc[metrics.group.eq(key) & metrics.model.eq(name)].iloc[0]
            assert_metrics(expected, actual)
            np.testing.assert_allclose(actual.macro_f1, f1_score(target, predictions[:, index], average="macro"), atol=1e-12, rtol=0)
            np.testing.assert_allclose(actual.qwk, cohen_kappa_score(target, predictions[:, index], weights="quadratic"), atol=1e-12, rtol=0)
            counts = confusion_rows.loc[confusion_rows.group.eq(key) & confusion_rows.model.eq(name)].sort_values(["true", "predicted"])
            np.testing.assert_array_equal(counts["count"].to_numpy().reshape(3, 3), matrix)
            artifact = load_model(root, slug, name)
            offsets = np.unique(np.linspace(0, len(pos)-1, 73, dtype=int))
            np.testing.assert_allclose(prep.predict_artifact(artifact, X.iloc[pos[offsets]]), p[offsets], rtol=1e-12, atol=1e-12)
            for metric in points:
                np.testing.assert_array_equal(cached_draws[key][metric], regenerated[metric])
                row = intervals.loc[intervals.group.eq(key) & intervals.model.eq(name) & intervals.metric.eq(metric)].iloc[0]
                assert_metrics({"estimate": points[metric][index], **interval(regenerated[metric][:, index])}, row)
        for reference in range(3):
            for metric in points:
                row = pairs.loc[pairs.group.eq(key) & pairs.reference.eq(MODELS[reference]) & pairs.metric.eq(metric)].iloc[0]
                assert_metrics({"delta": points[metric][3]-points[metric][reference],
                                **interval(regenerated[metric][:, 3]-regenerated[metric][:, reference])}, row)
    gaps = pd.read_csv(path(root, "results", "h1_gaps.csv"), float_precision="round_trip")
    if len(gaps) != 140:
        raise ValueError("Incomplete H1 rows")
    for row in gaps.to_dict("records"):
        a, b = f"random_seed_{row['seed']}_internal", f"random_seed_{row['seed']}_2024"
        metric, index = row["metric"], MODELS.index(row["model"])
        assert_metrics({"gap": oriented_gap(point_groups[a][metric][index], point_groups[b][metric][index], metric),
                        **interval(oriented_gap(cached_draws[a][metric][:, index], cached_draws[b][metric][:, index], metric))}, row)
    expected_h1 = gaps.groupby(["model", "metric"], sort=False).gap.agg(["mean", "std", "min", "max"]).reset_index()
    pd.testing.assert_frame_equal(expected_h1, pd.read_csv(path(root, "results", "h1_summary.csv"), float_precision="round_trip"))
    if descriptive_h2(pairs.to_dict("records")) != prep.read_json(path(root, "results", "h2_descriptive_decision.json")):
        raise ValueError("H2 decision differs")
    verify_shap(root, X, y, metadata, assignments)
    parent = prep.read_json(prep.output_path(root, "config", "protocol.json"))
    historical = prep.read_json(root / parent["historical_manifest"])
    prep.check_hashes(root, historical)
    if checked_seal(root, "verification_complete", required=False) is None:
        files = create_summary(root)
        files.extend(path(root, "logs", name + ".json") for name in
                     ("training_complete", "evaluation_complete", "bootstrap_complete", "shap_complete"))
        seal(root, "verification_complete", files, feature_count=15, training_splits=6, evaluation_groups=11,
             model_group_evaluations=44, bootstrap_iterations=2000, shap_cohorts=11,
             historical_files_unchanged=len(historical), training_checks=checks,
             secondary_analyses_rerun=False, public_clean_room_reproduction_performed=False)
    print("VERIFIED revised primary chain; secondary reruns and public release are NOT included.", flush=True)


def verify_shap(root, X, y, metadata, assignments):
    samples, _ = build_samples(y, metadata, assignments)
    importance = pd.read_csv(path(root, "results", "shap_importance.csv"), float_precision="round_trip")
    stability = pd.read_csv(path(root, "results", "shap_stability.csv"), float_precision="round_trip")
    counts = pd.read_csv(path(root, "results", "shap_sample_counts.csv"))
    directions = pd.read_csv(path(root, "results", "shap_class_directions.csv"), float_precision="round_trip")
    if len(importance) != 660 or len(stability) != 20 or len(counts) != 11 or len(directions) != 495:
        raise ValueError("Incomplete SHAP outputs")
    for slug in SLUGS:
        means, boot = {}, {}
        for cohort in (("future",) if slug == "temporal" else ("internal", "future")):
            sample_key = "future_2024" if cohort == "future" else slug + "_internal"
            pos = samples[sample_key]
            chunks = []
            for start in range(0, len(pos), 1000):
                with np.load(path(root, "results", "shap", slug, f"{cohort}_{start:05d}.npz")) as saved:
                    np.testing.assert_array_equal(saved["positions"], pos[start:start+1000])
                    chunks.append(saved["values"].copy())
            values = np.concatenate(chunks)
            artifact = load_model(root, slug, "lightgbm_weighted")
            independent, _, _ = explain(X, pos[:3], artifact["bundle"], artifact)
            np.testing.assert_allclose(independent, values[:3], rtol=1e-12, atol=1e-12)
            means[cohort] = np.abs(values).mean(axis=0)
            with np.load(path(root, "results", "shap", slug, cohort + "_bootstrap.npz")) as saved:
                np.testing.assert_array_equal(saved["positions"], pos)
                np.testing.assert_array_equal(saved["target"], y.iloc[pos])
                boot[cohort] = saved["means"].copy()
            regenerated = bootstrap_mean_values(np.abs(values), y.iloc[pos].to_numpy(), stream_key=VERSION + ":shap:" + sample_key)
            np.testing.assert_allclose(boot[cohort], regenerated, rtol=1e-12, atol=1e-12)
            sample_count = counts.loc[counts.split.eq(slug) & counts.cohort.eq(cohort)].iloc[0]
            for c in range(3):
                if sample_count[f"class{c}_count"] != int(np.sum(y.iloc[pos] == c)):
                    raise ValueError("SHAP sample class counts differ")
            for scope in SCOPES:
                scoped = scope_values(means[cohort], scope)
                subset = importance.loc[importance.split.eq(slug) & importance.cohort.eq(cohort) & importance.scope.eq(scope)].set_index("feature").loc[list(prep.FEATURES)]
                np.testing.assert_allclose(subset.mean_abs_shap, scoped, rtol=1e-12, atol=1e-12)
                np.testing.assert_array_equal(subset["rank"], rankdata(-scoped, method="average"))
            for c, scope in enumerate(SCOPES[1:]):
                subset = directions.loc[directions.split.eq(slug) & directions.cohort.eq(cohort) & directions.output_class.eq(scope)].set_index("feature").loc[list(prep.FEATURES)]
                for key, actual in (("mean_signed_shap", values[:, :, c].mean(axis=0)),
                                    ("positive_fraction", (values[:, :, c] > 0).mean(axis=0)),
                                    ("negative_fraction", (values[:, :, c] < 0).mean(axis=0))):
                    np.testing.assert_allclose(subset[key], actual, atol=1e-12, rtol=0)
        if slug != "temporal":
            for scope in SCOPES:
                a, b = [scope_values(means[c], scope) for c in ("internal", "future")]
                rho = spearman_rows(a[None, :], b[None, :])[0]
                draws = spearman_rows(scope_values(boot["internal"], scope), scope_values(boot["future"], scope))
                row = stability.loc[stability.seed.eq(int(slug.rsplit("_", 1)[1])) & stability.scope.eq(scope)].iloc[0]
                assert_metrics({"rho": rho, **interval(draws)}, row)
                with np.load(path(root, "results", "shap", slug, scope + "_rank_draws.npz")) as saved:
                    np.testing.assert_allclose(saved["rho"], draws, rtol=0, atol=1e-12)
    summary = stability.groupby("scope", sort=False).rho.agg(["mean", "std", "min", "max"]).reset_index()
    pd.testing.assert_frame_equal(summary, pd.read_csv(path(root, "results", "shap_summary.csv"), float_precision="round_trip"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--stage", required=True, choices=("preflight", "freeze", "train", "evaluate", "bootstrap", "shap", "verify", "all"))
    args = parser.parse_args(argv)
    root = args.project_root.expanduser().resolve()
    actions = {"preflight": preflight, "freeze": freeze, "train": train, "evaluate": evaluate, "bootstrap": bootstrap, "shap": shap_analysis, "verify": verify}
    with prep.execution_lock(root), threadpool_limits(limits=THREADS):
        for stage in (("freeze", "train", "evaluate", "bootstrap", "shap", "verify") if args.stage == "all" else (args.stage,)):
            print(f"START {stage} {prep.now()}", flush=True)
            actions[stage](root)
            print(f"END {stage} {prep.now()}", flush=True)


if __name__ == "__main__":
    main()
