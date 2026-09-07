"""Fixed-parameter, temporal-only sensitivity after the corrected main run.

Only the two weighted comparators are refitted. Historical and corrected main
outputs are read-only; no model-selection or early-stopping search is run here.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path
import time
import unittest
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits

import stats19_feature_revision as prep
import stats19_feature_revision_full as full

ROOT = prep.ROOT
VERSION = "STATS19_15_FEATURE_EXCLUDE2020_V1"
MAIN_HASH = "0f3e3f99cdc2338d4704d2279ced3ae45bc9cbf27288c9196bdaa5ecc05c3aed"
THREADS = 8
YEARS = (2018, 2019, 2021, 2022)
MODELS = ("logistic_weighted", "lightgbm_weighted")
JOINT_ORDER = ("main_logistic_weighted", "main_lightgbm_weighted",
               "exclude2020_logistic_weighted", "exclude2020_lightgbm_weighted")
CONTRASTS = {
    "main_LightGBM_minus_Logistic": [-1, 1, 0, 0],
    "exclude2020_LightGBM_minus_Logistic": [0, 0, -1, 1],
    "exclude2020_minus_main_Logistic": [-1, 0, 1, 0],
    "exclude2020_minus_main_LightGBM": [0, -1, 0, 1],
    "change_in_model_contrast": [1, -1, -1, 1],
}
SOURCE_FILES = ("run_stats19_exclude2020_revision.py", "code/stats19_exclude2020_revision.py",
                "tests/test_stats19_exclude2020_revision.py", "docs/STATS19_EXCLUDE2020_REVISION.md")


def path(root, family, *parts):
    return prep.output_path(root, family, "exclude2020", *parts)


def protocol_path(root):
    return path(root, "config", "protocol.json")


def require_main(root):
    full.require_protocol(root)
    if prep.digest(full.protocol_path(root)) != MAIN_HASH:
        raise ValueError("Unexpected corrected main execution freeze")
    full.require_all_models(root)
    for stage in ("evaluation_complete", "bootstrap_complete", "shap_complete", "verification_complete"):
        full.checked_seal(root, stage)


def fixed_parameters(root):
    parameters = {}
    for name in MODELS:
        estimator = full.load_model(root, "temporal", name)["estimator"]
        parameters[name] = {key: value for key, value in estimator.get_params().items() if key != "class_weight"}
    if parameters["lightgbm_weighted"]["n_jobs"] != THREADS:
        raise ValueError("Main model thread cap differs")
    return parameters


def positions(assignments, strict=True):
    main = prep.split_positions(assignments, "temporal", strict=strict)
    years = assignments[prep.YEAR].to_numpy(dtype=int)
    kept = main["train"][years[main["train"]] != 2020]
    removed = main["train"][years[main["train"]] == 2020]
    result = {"train": kept, "removed_2020": removed, "validation": main["validation"], "future": main["future"]}
    prep.assert_disjoint(result)
    if set(years[kept]) != set(YEARS) or set(years[removed]) != {2020}:
        raise ValueError("Unexpected kept or excluded training years")
    if set(years[result["validation"]]) != {2023} or set(years[result["future"]]) != {2024}:
        raise ValueError("Validation/test years changed")
    np.testing.assert_array_equal(np.sort(np.r_[kept, removed]), main["train"])
    if strict and [len(result[k]) for k in result] != [447262, 91199, 104258, 100927]:
        raise ValueError("Unexpected exclusion-2020 cohort sizes")
    return result


def protected_files(root):
    original = prep.read_json(prep.output_path(root, "logs", "historical_files.json"))
    protected = dict(original)
    prep.check_hashes(root, protected)
    # Preserve every completed preparation/main artifact, not only this run's inputs.
    for family in ("config", "results", "models", "logs", "figures"):
        base = root / family / prep.NAME
        for file in base.rglob("*"):
            relative = file.relative_to(base)
            if (file.is_file() and relative.parts[0] != "exclude2020"
                    and file.suffix not in {".tmp", ".lock", ".log"}):
                protected[file.relative_to(root).as_posix()] = prep.digest(file)
    return protected


def preflight(root):
    if protocol_path(root).exists():
        require_protocol(root)
        print("Frozen sensitivity preflight verified; not overwritten.", flush=True)
        return
    suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader
    for pattern in ("test_stats19_exclude2020_revision.py", "test_stats19_feature_revision.py", "test_stats19_feature_revision_full.py"):
        suite.addTests(loader.discover(str(ROOT / "tests"), pattern=pattern))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {"version": VERSION, "created_local": prep.now(), "passed": result.wasSuccessful(),
              "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "source_sha256": full.hashes(ROOT, [ROOT / p for p in SOURCE_FILES]), "output": stream.getvalue()}
    prep.write_json(path(root, "logs", "preflight.json"), report)
    if not result.wasSuccessful():
        raise RuntimeError("Sensitivity preflight failed; see preflight.json")
    print(f"PREFLIGHT passed: {result.testsRun} tests", flush=True)


def freeze(root):
    if protocol_path(root).exists():
        require_protocol(root)
        print("Existing sensitivity protocol verified; not overwritten.", flush=True)
        return
    require_main(root)
    tests_path = path(root, "logs", "preflight.json")
    tests = prep.read_json(tests_path)
    if not tests["passed"] or tests["tests_run"] < 70:
        raise ValueError("Sensitivity preflight has not passed")
    prep.check_hashes(ROOT, tests["source_sha256"])
    schema, _, y, _, assignments = prep.load_data(root)
    roles = positions(assignments)
    source = {**full.hashes(ROOT, [ROOT / p for p in SOURCE_FILES]),
              **prep.read_json(full.protocol_path(root))["source_sha256"]}
    main_report = prep.read_json(full.path(root, "logs", "validation", "temporal.json"))
    payload = {
        "version": VERSION, "created_local": prep.now(), "status": "FROZEN_BEFORE_CORRECTED_SENSITIVITY_FITS",
        "main_execution_sha256": MAIN_HASH, "legacy_and_corrected_main_outcomes_known": True,
        "preregistered": False, "new_unseen_test_claim": False,
        "disclosure": "Post-review fixed-parameter sensitivity, fully specified after corrected main results and before these refits",
        "purpose": "Assess temporal test estimates and the LightGBM-versus-Logistic contrast after removing 2020 from training",
        "scope": {"dataset": "STATS19", "split": "temporal_only", "features": schema,
                  "training_years": list(YEARS), "validation_year": 2023, "test_year": 2024,
                  "models": list(MODELS), "no_retuning": True, "no_early_stopping": True,
                  "no_threshold_optimization": True, "validation": "descriptive audit only, not parameter or model selection",
                  "test_gate": "both sensitivity models and validation predictions sealed before sensitivity 2024 evaluation",
                  "class_weights": "recompute n/(3*n_class) from reduced training records only",
                  "preprocessing": "fit reduced-training vocabulary/Logistic imputation and scaling; LGBM retains speed NaN",
                  "not_answered": "No causal pandemic effect, no H1 random-split sensitivity, no SHAP stability sensitivity"},
        "role_counts_and_hashes": {k: {"rows": len(v), "sha256": prep.position_hash(v)} for k, v in roles.items()},
        "training_class_counts": np.bincount(y.iloc[roles["train"]], minlength=3).tolist(),
        "model_parameters": fixed_parameters(root),
        "main_candidate": {"id": main_report["selected"]["candidate_id"], "iterations": main_report["selected"]["best_iteration"],
                           "main_validation_constraints_met": main_report["constraints_met"]},
        "convergence": "Stop before sealing on Logistic ConvergenceWarning; do not increase iterations silently",
        "bootstrap": {"iterations": full.ITERATIONS, "seed": full.BOOTSTRAP_SEED, "stream": VERSION + ":common_2024",
                      "design": "paired across all four models on identical 2024 rows; true-class-stratified record bootstrap via joint-pattern multinomial counts",
                      "model_order": list(JOINT_ORDER), "contrasts": CONTRASTS,
                      "interval": "95% percentile, 2000 draws; no multiplicity adjustment, descriptive only",
                      "main_interval_note": "Main contrasts recomputed with this shared four-model draw stream for pairing; main published intervals are not overwritten",
                      "H2": "same main descriptive joint gate; an interval crossing zero does not establish noninferiority"},
        "scope_decision": {"retain": "corrected main and this exclusion-2020 sensitivity",
                           "omit_from_revised_manuscript": ["Ordered Logit", "matched-subset Ordered Logit comparison", "tree-count extension sensitivity"],
                           "history": "preserved; never delete original audit or rewrite a previous freeze",
                           "reason": "User chose narrower main line after main results were known; not a pre-result design decision",
                           "tree_cap_disclosure": "Still disclose the 1200-round search boundary and untested larger budgets",
                           "CAS": "unchanged", "later": "final paper-consistent clean reproduction, then repository and software DOI release"},
        "protected_sha256": protected_files(root), "source_sha256": source,
        "evidence_sha256": full.hashes(root, [tests_path]),
        "runtime": prep.read_json(full.protocol_path(root))["runtime"], "threads": THREADS,
    }
    prep.write_json(protocol_path(root), payload)
    print("Sensitivity protocol frozen: 447262 train, 104258 validation, 100927 unchanged test; fixed main parameters.", flush=True)


def require_protocol(root):
    full.require_protocol(root)
    protocol = prep.read_json(protocol_path(root))
    if (protocol["version"] != VERSION or protocol["main_execution_sha256"] != MAIN_HASH
            or prep.digest(full.protocol_path(root)) != MAIN_HASH or protocol["threads"] != THREADS):
        raise ValueError("Sensitivity execution identity differs")
    prep.check_hashes(ROOT, protocol["source_sha256"])
    prep.check_hashes(root, protocol["evidence_sha256"])
    prep.check_hashes(root, protocol["protected_sha256"])
    return protocol


def seal(root, stage, files, **details):
    destination = path(root, "logs", stage + ".json")
    if destination.exists():
        raise FileExistsError("Completed sensitivity stage is immutable")
    prep.write_json(destination, {"status": "COMPLETE", "version": VERSION, "created_local": prep.now(),
                                 "protocol_sha256": prep.digest(protocol_path(root)), "files_sha256": full.hashes(root, files), **details})


def checked_seal(root, stage, required=True):
    destination = path(root, "logs", stage + ".json")
    if not destination.exists() and not required:
        return None
    record = prep.read_json(destination)
    if (record["status"] != "COMPLETE" or record["version"] != VERSION
            or record["protocol_sha256"] != prep.digest(protocol_path(root))):
        raise ValueError("Stale sensitivity checkpoint")
    prep.check_hashes(root, record["files_sha256"])
    return record


def check_roles(protocol, roles):
    actual = {k: {"rows": len(v), "sha256": prep.position_hash(v)} for k, v in roles.items()}
    if actual != protocol["role_counts_and_hashes"]:
        raise ValueError("Sensitivity row membership differs from freeze")


def fit_models(data, schema, parameters, threads=THREADS):
    prep.validate_features(data.train)
    prep.validate_features(data.validation)
    weights = prep.weights_from_training(data.train_target)
    train, val, bundle, audit = prep.prepare_native_split(data.train, data.validation, schema)
    preprocessor = prep.make_preprocessor(categorical_columns=schema["categorical_feature_columns"],
                                        numeric_columns=schema["numeric_feature_columns"],
                                        category_vocabulary=bundle["category_vocabulary"])
    train_encoded, val_encoded = preprocessor.fit_transform(train), preprocessor.transform(val)
    prep.assert_encoded_matrix(train_encoded, len(train))
    prep.assert_encoded_matrix(val_encoded, len(val))
    artifacts, probability, runtime = {}, {}, {}
    with threadpool_limits(limits=threads):
        for name in MODELS:
            estimator = (LogisticRegression(**parameters[name], class_weight=weights) if name == "logistic_weighted"
                         else lgb.LGBMClassifier(**parameters[name], class_weight=weights))
            print("FIT fixed parameters: " + name, flush=True)
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always", ConvergenceWarning)
                if name == "logistic_weighted":
                    estimator.fit(train_encoded, data.train_target)
                else:
                    # No eval data/callback: validation cannot alter the fixed tree count.
                    estimator.fit(train, data.train_target, categorical_feature=schema["categorical_feature_columns"])
            runtime[name] = {"seconds": time.perf_counter()-started,
                             "iterations": np.asarray(getattr(estimator, "n_iter_", [0])).tolist(),
                             "convergence_warning": any(issubclass(w.category, ConvergenceWarning) for w in seen)}
            np.testing.assert_array_equal(estimator.classes_, [0, 1, 2])
            artifacts[name] = {"estimator": estimator, "bundle": bundle}
            if name == "logistic_weighted":
                artifacts[name]["preprocessor"] = preprocessor
            probability[name] = prep.predict_artifact(artifacts[name], data.validation)
    return artifacts, probability, {"runtime": runtime, "class_weights": weights, "preprocessing_audit": audit}


def model_path(root, name):
    return path(root, "models", name + ".joblib")


def load_model(root, name):
    artifact = joblib.load(model_path(root, name))
    prep.require_analytical_artifact(artifact)
    if (artifact.get("sensitivity_version") != VERSION or artifact.get("model_name") != name
            or artifact.get("protocol_sha256") != prep.digest(protocol_path(root)) or artifact.get("smoke_only") is not False):
        raise ValueError("Smoke, legacy, main or incompatible sensitivity model")
    return artifact


def train(root):
    protocol = require_protocol(root)
    if checked_seal(root, "training_complete", required=False):
        return
    schema, X, y, metadata, assignments = prep.load_data(root)
    roles = positions(assignments)
    check_roles(protocol, roles)
    data = prep.selection_data(X, y, roles)
    started = time.perf_counter()
    with full.progress("15-feature exclude-2020 training"):
        artifacts, probabilities, diagnostic = fit_models(data, schema, protocol["model_parameters"])
    diagnostic.update(train_rows=len(data.train), validation_rows=len(data.validation), seconds=time.perf_counter()-started,
                      role_hashes={k: prep.position_hash(v) for k, v in roles.items()}, test_evaluated=False, retuned=False)
    diagnostic_path = path(root, "logs", "training_diagnostics.json")
    prep.write_json(diagnostic_path, diagnostic)
    if any(v["convergence_warning"] for v in diagnostic["runtime"].values()):
        raise RuntimeError("Sensitivity Logistic nonconvergence; diagnostics saved, test gate remains closed")
    files, metric_rows = [diagnostic_path], []
    for name, artifact in artifacts.items():
        artifact.update(analytical_result=True, feature_version=prep.VERSION, sensitivity_version=VERSION, smoke_only=False,
                        model_name=name, protocol_sha256=prep.digest(protocol_path(root)),
                        train_positions_sha256=prep.position_hash(roles["train"]))
        destination = model_path(root, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".joblib.tmp")
        joblib.dump(artifact, temporary, compress=3)
        temporary.replace(destination)
        restored = load_model(root, name)
        np.testing.assert_allclose(prep.predict_artifact(restored, data.validation), probabilities[name], rtol=1e-12, atol=1e-12)
        values, _ = full.full_metrics(data.validation_target, probabilities[name])
        metric_rows.append({"model": name, "rows": len(data.validation), "role": "validation_descriptive_only", **values})
        files.append(destination)
    files.append(full.save_npz(path(root, "results", "validation_predictions.npz"), positions=roles["validation"],
                               ids=metadata.iloc[roles["validation"]][prep.ID].to_numpy(dtype=str), target=data.validation_target, **probabilities))
    files.append(full.write_csv(path(root, "results", "validation_metrics.csv"), metric_rows))
    seal(root, "training_complete", files, models=list(MODELS), test_performance_evaluated=False,
         validation_used_for_selection=False, training_rows=len(data.train))
    print("Both sensitivity models sealed; validation descriptive only.", flush=True)


def require_test_gate(root):
    record = checked_seal(root, "training_complete")
    if record["models"] != list(MODELS) or record["test_performance_evaluated"] or record["validation_used_for_selection"]:
        raise ValueError("Both fixed-parameter sensitivity models must be sealed before test evaluation")


def evaluate(root):
    protocol = require_protocol(root)
    require_test_gate(root)
    if checked_seal(root, "evaluation_complete", required=False):
        return
    _, X, y, metadata, assignments = prep.load_data(root)
    roles = positions(assignments)
    check_roles(protocol, roles)
    pos = roles["future"]
    target, ids = y.iloc[pos].to_numpy(), metadata.iloc[pos][prep.ID].to_numpy(dtype=str)
    main = full.aligned_predictions(root, "temporal_2024", pos, target, ids)
    probabilities = {"main_" + name: main[name] for name in MODELS}
    for name in MODELS:
        probabilities["exclude2020_" + name] = prep.predict_artifact(load_model(root, name), X.iloc[pos])
    rows, confusions = [], []
    for name in JOINT_ORDER:
        values, matrix = full.full_metrics(target, probabilities[name])
        rows.append({"model": name, "test_year": 2024, "rows": len(pos), **values})
        for c in range(3):
            for p in range(3):
                confusions.append({"model": name, "true": c, "predicted": p, "count": int(matrix[c, p])})
    files = [full.save_npz(path(root, "results", "common_2024_predictions.npz"), positions=pos, ids=ids, target=target, **probabilities),
             full.write_csv(path(root, "results", "test_metrics.csv"), rows),
             full.write_csv(path(root, "results", "confusion_matrices.csv"), confusions),
             path(root, "logs", "training_complete.json")]
    seal(root, "evaluation_complete", files, test_rows=len(pos), joint_order=list(JOINT_ORDER), unchanged_main_predictions=True)
    print("2024 evaluation complete on identical records, including corrected-main reference predictions.", flush=True)


def comparison_rows(points, draws):
    rows = []
    for metric in points:
        for name, weights in CONTRASTS.items():
            values = np.asarray(draws[metric]) @ weights
            rows.append({"contrast": name, "metric": metric, "delta": float(np.asarray(points[metric]) @ weights),
                         **full.interval(values), "iterations": full.ITERATIONS, "design": "paired_common_2024_true_class_stratified"})
    return rows


def h2_decision(rows, contrast):
    converted = [{"group": "temporal_2024", "reference": "logistic_weighted", **r} for r in rows if r["contrast"] == contrast]
    result = full.descriptive_h2(converted)
    result["contrast"] = contrast
    return result


def bootstrap_arrays(root, target, predictions):
    rng = np.random.default_rng(full.stream_seed(full.BOOTSTRAP_SEED, VERSION + ":common_2024"))
    points = full.metrics_from_confusions(full.confusion_from_vectors(target, predictions))
    draws = full.metrics_from_confusions(full.joint_bootstrap(target, predictions, rng, iterations=full.ITERATIONS))
    return points, draws


def bootstrap(root):
    require_protocol(root)
    require_test_gate(root)
    checked_seal(root, "evaluation_complete")
    if checked_seal(root, "bootstrap_complete", required=False):
        return
    with np.load(path(root, "results", "common_2024_predictions.npz")) as saved:
        target = saved["target"]
        predictions = np.column_stack([saved[n].argmax(axis=1) for n in JOINT_ORDER])
    points, draws = bootstrap_arrays(root, target, predictions)
    rows = comparison_rows(points, draws)
    estimates = [{"model": name, "metric": metric, "estimate": float(points[metric][index]), **full.interval(draws[metric][:, index])}
                 for metric in points for index, name in enumerate(JOINT_ORDER)]
    h2 = {"main_on_joint_draws": h2_decision(rows, "main_LightGBM_minus_Logistic"),
          "exclude2020": h2_decision(rows, "exclude2020_LightGBM_minus_Logistic"),
          "not_noninferiority_or_equivalence_test": True, "not_a_causal_pandemic_test": True}
    h2_path = path(root, "results", "h2_sensitivity_decision.json")
    prep.write_json(h2_path, h2)
    files = [full.save_npz(path(root, "results", "bootstrap_draws.npz"), **draws),
             full.write_csv(path(root, "results", "paired_comparisons.csv"), rows),
             full.write_csv(path(root, "results", "metric_intervals.csv"), estimates), h2_path,
             path(root, "logs", "evaluation_complete.json")]
    seal(root, "bootstrap_complete", files, iterations=full.ITERATIONS, model_count=4, contrasts_per_metric=5)
    print("Paired Bootstrap complete: 2000 draws across all four predictions.", flush=True)


def summary(root, protocol, comparisons):
    metrics = pd.read_csv(path(root, "results", "test_metrics.csv"), float_precision="round_trip")
    decision = prep.read_json(path(root, "results", "h2_sensitivity_decision.json"))
    lines = ["# 15-feature exclusion-2020 sensitivity", "", "Status: complete and verified.", "",
             "Post-review sensitivity specified after main results were known. Not preregistered and not a newly unseen test.",
             "Only 2020 is removed from temporal training. This also reduces training sample size; it does not isolate a causal pandemic effect.", "",
             "Training: 2018, 2019, 2021, 2022 (447,262 records); validation: 2023 (104,258); identical test: 2024 (100,927).",
             f"Fixed corrected-main LightGBM: {protocol['main_candidate']['id']}, {protocol['main_candidate']['iterations']} rounds. No retuning or early stopping.",
             "Both comparators use reduced-training fitted preprocessing and recomputed balanced class weights.", "",
             "Model | Macro-F1 | QWK | Fatal recall | Mean cost", "--- | ---: | ---: | ---: | ---:"]
    for row in metrics.to_dict("records"):
        lines.append(f"{row['model']} | {row['macro_f1']:.6f} | {row['qwk']:.6f} | {row['fatal_recall']:.6f} | {row['mean_asymmetric_cost']:.6f}")
    lines.extend(["", "Contrast | Metric | Difference | Paired 95% interval", "--- | --- | ---: | ---"])
    for row in comparisons:
        if row["metric"] in {"macro_f1", "qwk", "fatal_recall", "mean_asymmetric_cost"}:
            lines.append(f"{row['contrast']} | {row['metric']} | {row['delta']:+.6f} | [{row['ci_lower']:+.6f}, {row['ci_upper']:+.6f}]")
    lines.extend(["", f"Main joint descriptive H2 gate (common draws): {decision['main_on_joint_draws']['joint_descriptive_gate']}.",
                  f"Exclusion-2020 joint descriptive H2 gate: {decision['exclude2020']['joint_descriptive_gate']}.",
                  "Intervals containing zero do not establish noninferiority. All contrasts use paired true-class-stratified record Bootstrap (2000 draws).",
                  "The change-in-model-contrast interval is calculated on common draws, not by subtracting separate interval endpoints.",
                  "Main reference intervals are recomputed only to preserve four-model pairing; previously frozen main intervals remain unchanged.", "",
                  "## Revised scope", "",
                  "Keep the corrected main experiments and this exclusion-2020 sensitivity regardless of outcome.",
                  "Omit Ordered Logit/matched-subset and tree-count sensitivity from the revised manuscript; retain their historical artifacts for audit.",
                  "Still report the 1200-round search boundary. Larger tree budgets were not tested in the corrected feature run.",
                  "This does not test robustness of random-split H1 gaps or SHAP H3 stability. CAS is unchanged.",
                  "Final manuscript-consistent clean-room reproduction and repository/software DOI updates remain separate later tasks.", ""])
    destination = path(root, "results", "SUMMARY.md")
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def verify(root):
    protocol = require_protocol(root)
    require_test_gate(root)
    checked_seal(root, "evaluation_complete")
    checked_seal(root, "bootstrap_complete")
    schema, X, y, metadata, assignments = prep.load_data(root)
    roles = positions(assignments)
    check_roles(protocol, roles)
    weights = prep.weights_from_training(y.iloc[roles["train"]])
    validation = pd.read_csv(path(root, "results", "validation_metrics.csv"), float_precision="round_trip")
    diagnostic = prep.read_json(path(root, "logs", "training_diagnostics.json"))
    if any(v["convergence_warning"] for v in diagnostic["runtime"].values()):
        raise ValueError("Nonconvergent model was sealed")
    models = {}
    for name in MODELS:
        artifact = load_model(root, name)
        models[name] = artifact
        actual = artifact["estimator"].get_params()
        if actual.pop("class_weight") != weights or actual != protocol["model_parameters"][name]:
            raise ValueError("Fixed hyperparameters or reduced-training weights changed")
        if artifact["train_positions_sha256"] != prep.position_hash(roles["train"]):
            raise ValueError("Model training membership differs")
        for col, vocabulary in artifact["bundle"]["category_vocabulary"].items():
            if vocabulary != sorted(str(v) for v in X.iloc[roles["train"]][col].unique()):
                raise ValueError("Vocabulary is not reduced-training-only")
        if "preprocessor" in artifact:
            median = artifact["preprocessor"].named_transformers_["numeric"].named_steps["imputer"].statistics_[0]
            np.testing.assert_allclose(median, np.nanmedian(X.iloc[roles["train"]].feature_speed_limit), rtol=0, atol=0)
        p = prep.predict_artifact(artifact, X.iloc[roles["validation"]])
        with np.load(path(root, "results", "validation_predictions.npz")) as saved:
            np.testing.assert_array_equal(saved["positions"], roles["validation"])
            np.testing.assert_array_equal(saved["target"], y.iloc[roles["validation"]])
            np.testing.assert_array_equal(saved["ids"], metadata.iloc[roles["validation"]][prep.ID].to_numpy(dtype=str))
            np.testing.assert_allclose(saved[name], p, rtol=1e-12, atol=1e-12)
        values, _ = full.full_metrics(y.iloc[roles["validation"]], p)
        full.assert_metrics(values, validation.loc[validation.model.eq(name)].iloc[0])
    metrics = pd.read_csv(path(root, "results", "test_metrics.csv"), float_precision="round_trip")
    confusions = pd.read_csv(path(root, "results", "confusion_matrices.csv"))
    if len(metrics) != 4 or len(confusions) != 36:
        raise ValueError("Incomplete test outputs")
    pos = roles["future"]
    target, ids = y.iloc[pos].to_numpy(), metadata.iloc[pos][prep.ID].to_numpy(dtype=str)
    main = full.aligned_predictions(root, "temporal_2024", pos, target, ids)
    saved_arrays = {}
    with np.load(path(root, "results", "common_2024_predictions.npz")) as saved:
        np.testing.assert_array_equal(saved["positions"], pos)
        np.testing.assert_array_equal(saved["ids"], ids)
        np.testing.assert_array_equal(saved["target"], target)
        for name in JOINT_ORDER:
            saved_arrays[name] = saved[name].copy()
    for name in MODELS:
        np.testing.assert_array_equal(saved_arrays["main_" + name], main[name])
        p = prep.predict_artifact(models[name], X.iloc[pos])
        np.testing.assert_allclose(saved_arrays["exclude2020_" + name], p, rtol=1e-12, atol=1e-12)
    for name, p in saved_arrays.items():
        values, matrix = full.full_metrics(target, p)
        pred = p.argmax(axis=1)
        np.testing.assert_allclose(values["macro_f1"], f1_score(target, pred, average="macro"), rtol=0, atol=1e-12)
        np.testing.assert_allclose(values["qwk"], cohen_kappa_score(target, pred, weights="quadratic"), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(matrix, confusion_matrix(target, pred, labels=[0,1,2]))
        full.assert_metrics(values, metrics.loc[metrics.model.eq(name)].iloc[0])
        observed = confusions.loc[confusions.model.eq(name)].sort_values(["true", "predicted"])["count"].to_numpy().reshape(3,3)
        np.testing.assert_array_equal(observed, matrix)
    predictions = np.column_stack([saved_arrays[n].argmax(axis=1) for n in JOINT_ORDER])
    points, draws = bootstrap_arrays(root, target, predictions)
    with np.load(path(root, "results", "bootstrap_draws.npz")) as saved:
        for metric, values in draws.items():
            if values.shape != (full.ITERATIONS, 4):
                raise ValueError("Bootstrap count/order differs")
            np.testing.assert_array_equal(saved[metric], values)
    comparisons = comparison_rows(points, draws)
    pd.testing.assert_frame_equal(pd.DataFrame(comparisons), pd.read_csv(path(root, "results", "paired_comparisons.csv"), float_precision="round_trip"))
    estimates = pd.read_csv(path(root, "results", "metric_intervals.csv"), float_precision="round_trip")
    if len(estimates) != len(points) * 4:
        raise ValueError("Incomplete metric intervals")
    for metric in points:
        for index, name in enumerate(JOINT_ORDER):
            full.assert_metrics({"estimate": points[metric][index], **full.interval(draws[metric][:, index])},
                                estimates.loc[estimates.model.eq(name) & estimates.metric.eq(metric)].iloc[0])
    decision = prep.read_json(path(root, "results", "h2_sensitivity_decision.json"))
    for key, contrast in (("main_on_joint_draws", "main_LightGBM_minus_Logistic"), ("exclude2020", "exclude2020_LightGBM_minus_Logistic")):
        if decision[key] != h2_decision(comparisons, contrast):
            raise ValueError("Sensitivity interpretation gate differs")
    prep.check_hashes(root, protocol["protected_sha256"])
    if checked_seal(root, "verification_complete", required=False) is None:
        destination = summary(root, protocol, comparisons)
        seal(root, "verification_complete", [destination, *[path(root, "logs", stage + ".json") for stage in
             ("training_complete", "evaluation_complete", "bootstrap_complete")]],
             feature_count=15, training_rows=len(roles["train"]), excluded_training_rows=len(roles["removed_2020"]),
             validation_rows=len(roles["validation"]), test_rows=len(pos), bootstrap_iterations=full.ITERATIONS,
             full_serialized_validation_and_test_predictions_checked=True, main_results_unchanged=True,
             protected_files_unchanged=len(protocol["protected_sha256"]), retuned=False, public_reproduction_performed=False)
    print("VERIFIED corrected exclusion-2020 sensitivity; main and historical results unchanged.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--stage", required=True, choices=("preflight", "freeze", "train", "evaluate", "bootstrap", "verify", "all"))
    args = parser.parse_args(argv)
    root = args.project_root.expanduser().resolve()
    actions = {"preflight": preflight, "freeze": freeze, "train": train, "evaluate": evaluate, "bootstrap": bootstrap, "verify": verify}
    with prep.execution_lock(root), threadpool_limits(limits=THREADS):
        stages = ("preflight", "freeze", "train", "evaluate", "bootstrap", "verify") if args.stage == "all" else (args.stage,)
        for stage in stages:
            print(f"START {stage} {prep.now()}", flush=True)
            actions[stage](root)
            print(f"END {stage} {prep.now()}", flush=True)


if __name__ == "__main__":
    main()
