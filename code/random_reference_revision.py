"""Split-local LightGBM selection without altering historical artifacts.

The original outcomes are already known. This is a transparent post-review
correction, not a new untouched-test or preregistered experiment.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from threadpoolctl import threadpool_limits

from baseline_modeling import classification_metrics, prepare_features
from d10_tune_lightgbm import prepare_native_split, select_candidate
from d12_bootstrap_uncertainty import confusion_from_vectors, metrics_from_confusions
from d14_shap_stability import (
    bootstrap_mean_values, build_samples, explain, spearman_rows,
)
from cas_shap_stability import matched_positions, shap_contributions, bootstrap_mean_importance

SOURCE_ROOT = Path(__file__).resolve().parents[1]
NAME = "random_reference_revision"
VERSION = "POST_REVIEW_SPLIT_LOCAL_SELECTION_V1"
SEEDS = (1103, 2207, 3301, 4409, 5501)
THREADS = 8
ITERATIONS = 2000
MODELS = ("dummy_most_frequent", "logistic_weighted", "lightgbm_legacy", "lightgbm_revised")
LOWER = {"ordinal_mae", "mean_asymmetric_cost"}
SPECS = {
    "stats19": {
        "data": "data/processed/stats19_modeling_dataset.csv.gz",
        "assignments": "data/processed/d6_split_assignments.csv.gz",
        "schema": "config/d5_dataset_schema.json",
        "parent": "config/d10_lightgbm_protocol.json",
        "baseline_metrics": "results/d8/d8_validation_metrics.csv",
        "baseline_validation": "results/d8/validation_predictions",
        "predictions": "results/d11/predictions",
        "id": "meta_collision_index", "year": "meta_collision_year",
        "future_year": 2024, "rows": [449903, 96408, 96408, 100927],
        "bootstrap_seed": 20260828, "model_seed": 20260830,
        "labels": ["Slight", "Serious", "Fatal"],
    },
    "cas": {
        "data": "data/processed/cas_modeling_dataset.csv.gz",
        "assignments": "data/processed/cas_split_assignments.csv.gz",
        "schema": "config/cas/cas_modeling_schema.json",
        "parent": "config/cas/cas_lightgbm_protocol.json",
        "baseline_metrics": "results/cas_baseline/validation_metrics.csv",
        "baseline_validation": "results/cas_baseline/validation_predictions",
        "predictions": "results/cas_evaluation/predictions",
        "id": "meta_crash_id", "year": "meta_crash_year",
        "future_year": 2025, "rows": [22805, 4887, 4887, 10542],
        "bootstrap_seed": 20260901, "model_seed": 20260901,
        "labels": ["Minor", "Serious", "Fatal"],
    },
}
SOURCE_FILES = [
    "run_random_reference_revision.py", "code/random_reference_revision.py",
    "code/baseline_modeling.py", "code/d10_tune_lightgbm.py",
    "code/d12_bootstrap_uncertainty.py", "code/d14_shap_stability.py",
    "code/cas_shap_stability.py", "code/cas_modeling_common.py",
    "code/d11_evaluate_frozen_models.py", "code/modeling_data.py",
    "tests/test_random_reference_revision.py",
]


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def row_hash(positions):
    return hashlib.sha256(np.asarray(positions, dtype="<i8").tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def stream_seed(base, key):
    return int.from_bytes(hashlib.sha256(f"{base}:{key}".encode()).digest()[:8], "little")


def output_dirs(root, dataset):
    return tuple(root / parent / NAME / dataset for parent in ("config", "results", "models", "logs", "figures"))


@contextmanager
def execution_lock(root):
    path = root / "logs" / NAME / ".execution.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def load_data(root, dataset):
    spec = SPECS[dataset]
    schema = read_json(root / spec["schema"])
    cats = schema["categorical_feature_columns"]
    table = pd.read_csv(root / spec["data"], dtype={spec["id"]: "string", **{c: "string" for c in cats}},
                        keep_default_na=False, na_values=[""], low_memory=False)
    assignments = pd.read_csv(root / spec["assignments"], dtype={spec["id"]: "string"}, keep_default_na=False)
    if not table[spec["id"]].equals(assignments[spec["id"]]) or table[spec["id"]].duplicated().any():
        raise AssertionError("Identifiers must be unique and identically aligned")
    np.testing.assert_array_equal(table[spec["year"]], assignments[spec["year"]])
    features = table[schema["feature_columns"]].copy()
    for c in schema["numeric_feature_columns"]:
        features[c] = pd.to_numeric(features[c], errors="raise")
    if any(c.startswith("meta_") or c == "target_severity" for c in features):
        raise AssertionError("Metadata/target leaked into feature contract")
    target = table.target_severity.astype("int8")
    if "target_severity" in assignments:
        np.testing.assert_array_equal(target, assignments.target_severity)
    return schema, features, target, table[[spec["id"], spec["year"]]], assignments


def split_positions(assignments, seed, dataset=None):
    role = assignments[f"random_role_seed_{seed}"]
    names = ("train", "validation", "test", "locked_temporal_test")
    result = {name: np.flatnonzero(role.eq(name).to_numpy()) for name in names}
    combined = np.concatenate(list(result.values()))
    if len(combined) != len(assignments) or len(np.unique(combined)) != len(assignments):
        raise AssertionError("Invalid, duplicate or missing split roles")
    assert_disjoint(result["train"], result["validation"], result["test"], result["locked_temporal_test"])
    if dataset:
        spec = SPECS[dataset]
        if [len(result[n]) for n in names] != spec["rows"]:
            raise AssertionError("Frozen split sizes changed")
        future = assignments[spec["year"]].eq(spec["future_year"]).to_numpy()
        np.testing.assert_array_equal(np.flatnonzero(future), result["locked_temporal_test"])
    return result


def assert_disjoint(train, validation, internal, future):
    arrays = [np.asarray(a) for a in (train, validation, internal, future)]
    for i, values in enumerate(arrays):
        if len(values) != len(np.unique(values)):
            raise AssertionError("Duplicate positions")
        for other in arrays[i+1:]:
            if np.intersect1d(values, other).size:
                raise AssertionError("Selection and test records overlap")


def aligned_predictions(path, ids, target, id_column):
    frame = pd.read_csv(path, dtype={id_column: "string"}, keep_default_na=False)
    if frame[id_column].duplicated().any() or len(frame) != len(ids) or set(frame[id_column]) != set(ids.astype(str)):
        raise AssertionError(f"Prediction identity mismatch: {path}")
    frame = frame.set_index(id_column).loc[ids.astype(str)]
    np.testing.assert_array_equal(frame.target_severity.to_numpy(), np.asarray(target))
    return frame.predicted_severity.to_numpy(dtype=np.int8)


def legacy_prediction_path(root, dataset, seed, cohort, model):
    spec = SPECS[dataset]
    if dataset == "stats19":
        suffix = "test" if cohort == "internal" else "2024_diagnostic"
        name = f"random_seed_{seed}__{model}__{suffix}.csv.gz"
    else:
        suffix = "internal_test" if cohort == "internal" else "2025_diagnostic"
        name = f"random_seed_{seed}_{suffix}__{model}.csv.gz"
    return root / spec["predictions"] / name


def baseline_reference(root, dataset, seed, positions, metadata, target):
    spec = SPECS[dataset]
    path = root / spec["baseline_validation"] / f"random_seed_{seed}__logistic_weighted.csv.gz"
    y = target.iloc[positions].to_numpy()
    pred = aligned_predictions(path, metadata.iloc[positions][spec["id"]], y, spec["id"])
    metrics = classification_metrics(y, pred)
    recorded = pd.read_csv(root / spec["baseline_metrics"], dtype={"seed": "string"})
    row = recorded.loc[recorded.seed.eq(str(seed)) & recorded.model.eq("logistic_weighted")]
    if len(row) != 1:
        raise AssertionError("Missing split-specific Logistic validation baseline")
    for metric in ("macro_f1", "qwk", "fatal_recall"):
        np.testing.assert_allclose(metrics[metric], row.iloc[0][metric], rtol=0, atol=1e-12)
    return {metric: metrics[metric] for metric in ("macro_f1", "qwk", "fatal_recall")}


def protocol_file(root, dataset):
    return output_dirs(root, dataset)[0] / "protocol.json"


def freeze(root, dataset):
    path = protocol_file(root, dataset)
    if path.exists():
        require_protocol(root, dataset)
        print(f"{dataset}: existing revision protocol verified", flush=True)
        return
    spec = SPECS[dataset]
    parent = read_json(root / spec["parent"])
    schema, features, target, metadata, assignments = load_data(root, dataset)
    upstream = {spec[k] for k in ("data", "assignments", "schema", "parent", "baseline_metrics")}
    split_records = []
    for seed in SEEDS:
        positions = split_positions(assignments, seed, dataset)
        reference = baseline_reference(root, dataset, seed, positions["validation"], metadata, target)
        train_labels = target.iloc[positions["train"]].to_numpy()
        counts = np.bincount(train_labels, minlength=3)
        split_records.append({"seed": seed, "roles": {k: {"rows": len(v), "positions_sha256": row_hash(v)} for k, v in positions.items()},
                              "baseline_reference": reference, "minimum_qwk": reference["qwk"] - 0.01,
                              "minimum_fatal_recall": reference["fatal_recall"] - 0.05,
                              "class_counts": counts.tolist(), "class_weights": (len(train_labels) / (3 * counts)).tolist(),
                              "selection_test_intersection": 0})
        upstream.add(f"{spec['baseline_validation']}/random_seed_{seed}__logistic_weighted.csv.gz")
        for cohort in ("internal", "future"):
            for model in ("dummy_most_frequent", "logistic_weighted", "lightgbm_weighted"):
                upstream.add(legacy_prediction_path(root, dataset, seed, cohort, model).relative_to(root).as_posix())
    # Protect all historical result/configuration/model files against mutation.
    protected = {}
    for folder in ("config", "results", "models", "logs"):
        for file in sorted((root / folder).rglob("*")):
            if file.is_file() and NAME not in file.relative_to(root).parts and not file.name.endswith((".tmp", ".log")):
                protected[file.relative_to(root).as_posix()] = digest(file)
    logdir = output_dirs(root, dataset)[3]
    protected_path = logdir / "historical_artifacts.json"
    write_json(protected_path, protected)
    protocol = {
        "version": VERSION, "dataset": dataset, "created_local": now(),
        "status": "FROZEN_BEFORE_REVISED_FITS_AFTER_LEGACY_OUTCOMES",
        "disclosure": "Review-driven correction; original tests and SHAP results already known. Not preregistration or a newly untouched test.",
        "scope": "Only five random LightGBM branches and dependent evaluations; temporal models and H2 unchanged.",
        "selection_boundary": "Each fit, vocabulary, weight and baseline constraint uses only that seed's train/validation records. No temporal-selected hyperparameters are reused.",
        "reporting_rule": "Report revised random results regardless of direction; preserve and label legacy outputs; no outcome-driven choice between versions.",
        "seeds": list(SEEDS), "features": schema["feature_columns"], "splits": split_records,
        "candidates": parent["tuning"]["candidates"], "common_parameters": {**parent["common_model_parameters"], "n_jobs": THREADS},
        "maximum_estimators": 1200, "early_stopping_rounds": 75, "early_stopping_metric": "unweighted multi_logloss on own validation",
        "boundary_rule": parent["tuning"].get("boundary_rule", "Keep 1200 cap; no random-branch extension or D10b retuning"),
        "ranking": parent["selection_rule"]["ranking"], "fallback": parent["selection_rule"]["fallback_if_none_eligible"],
        "refit": "Train-only refit at selected best_iteration; validation not added to training; verify validation predictions match selection run.",
        "test_gate": "All five revised models for this data source must be sealed before any revised internal/future evaluation.",
        "bootstrap": {"iterations": ITERATIONS, "base_seed": spec["bootstrap_seed"], "stream_rule": "first eight little-endian SHA256 bytes of base:key",
                      "design": "true-class-stratified; fixed counts; paired within a cohort; independent internal/future",
                      "interval": "percentile 2.5 and 97.5; no multiplicity adjustment; fixed models; seeds described by sample SD and range"},
        "shap": {"samples": "Recreate original deterministic STATS19 10000-per-cohort and CAS 4887 class-matched samples",
                 "algorithm": "Existing STATS19 TreeExplainer and CAS native TreeSHAP, raw multiclass scores, additivity verified",
                 "aggregation": "Mean absolute by output class, equal output-class mean, Spearman across original fields",
                 "intervals": "2000 record-stratified draws of precomputed SHAP, not 2000 TreeSHAP runs; shared STATS19 future draws",
                 "threads": THREADS, "causal_claim": False},
        "upstream_sha256": {k: digest(root / k) for k in sorted(upstream)},
        "source_sha256": {k: digest(SOURCE_ROOT / k) for k in SOURCE_FILES},
        "historical_manifest": protected_path.relative_to(root).as_posix(), "historical_manifest_sha256": digest(protected_path),
        "runtime": {"python": platform.python_version(), **{k: importlib.metadata.version(k) for k in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "shap", "joblib")}},
    }
    write_json(path, protocol)
    print(f"{dataset}: revision protocol frozen; {len(protected)} historical files protected", flush=True)


def require_protocol(root, dataset):
    p = read_json(protocol_file(root, dataset))
    if p["version"] != VERSION or p["dataset"] != dataset:
        raise AssertionError("Unexpected revision protocol")
    for name, expected in p["source_sha256"].items():
        observed = digest(SOURCE_ROOT / name)
        if observed != expected:
            amendment_path = root / "config" / NAME / "implementation_amendment_01.json"
            if not amendment_path.exists():
                raise AssertionError(f"Source changed after revision freeze: {name}; document amendment before proceeding")
            amendment = read_json(amendment_path)
            change = amendment["source_changes"].get(name, {})
            if (amendment["parent_protocol_sha256"].get(dataset) != digest(protocol_file(root, dataset))
                    or change.get("before") != expected or change.get("after") != observed):
                raise AssertionError(f"Source change is not covered by the recorded amendment: {name}")
            for protected_name, protected_hash in amendment["frozen_models_sha256"].items():
                if digest(root / protected_name) != protected_hash:
                    raise AssertionError("Frozen model seal changed during implementation amendment")
    for name, expected in p["upstream_sha256"].items():
        if digest(root / name) != expected:
            raise AssertionError(f"Upstream changed: {name}")
    return p


def make_model(protocol, candidate, weights, iterations):
    seed = int(protocol["common_parameters"]["random_state"])
    return lgb.LGBMClassifier(**protocol["common_parameters"], **candidate["parameters"], n_estimators=int(iterations),
                             class_weight=weights, bagging_seed=seed, feature_fraction_seed=seed, data_random_seed=seed,
                             importance_type="gain")


def full_metrics(y, pred):
    metrics = classification_metrics(y, pred)
    matrix = confusion_from_vectors(np.asarray(y), np.asarray(pred)[:, None])[0]
    for c in range(3):
        metrics[f"class{c}_recall"] = float(matrix[c, c] / matrix[c].sum())
        metrics[f"class{c}_precision"] = float(matrix[c, c] / matrix[:, c].sum()) if matrix[:, c].sum() else 0.0
    metrics.pop("slight_recall")
    metrics.pop("serious_recall")
    return metrics


def fit_candidate(protocol, candidate, weights, X_train, y_train, X_validation, y_validation, cap):
    estimator = make_model(protocol, candidate, weights, cap)
    history = {}
    started = time.perf_counter()
    estimator.fit(X_train, y_train, eval_set=[(X_validation, y_validation)], eval_metric="multi_logloss",
                  categorical_feature=[c for c in X_train if isinstance(X_train[c].dtype, pd.CategoricalDtype)],
                  callbacks=[lgb.early_stopping(protocol["early_stopping_rounds"], first_metric_only=True, verbose=False),
                             lgb.record_evaluation(history)])
    probability = estimator.predict_proba(X_validation)
    row = {"candidate_id": candidate["candidate_id"], "complexity_rank": candidate["complexity_rank"],
           "best_iteration": int(estimator.best_iteration_ or cap), "maximum_estimators": cap,
           "fit_seconds": time.perf_counter() - started, **full_metrics(y_validation, probability.argmax(axis=1))}
    return row, history, probability


def tune(root, dataset, smoke=False):
    protocol = require_protocol(root, dataset)
    config, results, models, logs, _ = output_dirs(root, dataset)
    if not smoke and (config / "all_models_frozen.json").exists():
        require_all_models(root, dataset)
        print(f"{dataset}: all five sealed models verified, tuning skipped", flush=True)
        return
    schema, features, target, metadata, assignments = load_data(root, dataset)
    if smoke:
        record = protocol["splits"][0]
        pos = split_positions(assignments, record["seed"], dataset)
        from sklearn.model_selection import train_test_split
        rng_seed = 20260906
        train = train_test_split(pos["train"], train_size=3000, stratify=target.iloc[pos["train"]], random_state=rng_seed)[0]
        val = train_test_split(pos["validation"], train_size=1200, stratify=target.iloc[pos["validation"]], random_state=rng_seed)[0]
        a, b, _, _ = prepare_native_split(features.iloc[train], features.iloc[val], schema)
        weights = {i: len(train)/(3*sum(target.iloc[train] == i)) for i in range(3)}
        row, _, _ = fit_candidate(protocol, protocol["candidates"][0], weights, a, target.iloc[train], b, target.iloc[val], 30)
        write_json(logs / "smoke.json", {"status": "PASS", "analytical_result": False, "training_rows": 3000, "validation_rows": 1200,
                                         "finite_metrics": all(np.isfinite(v) for v in full_metrics(target.iloc[val], np.zeros(len(val), dtype=int)).values()), "seconds": row["fit_seconds"]})
        print(f"{dataset}: smoke PASS", flush=True)
        return
    for record in protocol["splits"]:
        seed = record["seed"]
        directory = results / str(seed)
        selected_path = config / f"selected_{seed}.json"
        if selected_path.exists():
            validate_selected(root, dataset, seed)
            print(f"{dataset}/{seed}: sealed model verified, skipped", flush=True)
            continue
        positions = split_positions(assignments, seed, dataset)
        for role, values in positions.items():
            if row_hash(values) != record["roles"][role]["positions_sha256"]:
                raise AssertionError("Frozen positions changed")
        train, validation = positions["train"], positions["validation"]
        a, b, bundle, prep_audit = prepare_native_split(features.iloc[train], features.iloc[validation], schema)
        y_train, y_val = target.iloc[train].to_numpy(), target.iloc[validation].to_numpy()
        counts = np.bincount(y_train, minlength=3)
        weights = {i: len(train)/(3*counts[i]) for i in range(3)}
        np.testing.assert_array_equal(counts, record["class_counts"])
        rows = []
        for candidate in protocol["candidates"]:
            checkpoint = directory / f"tuning_{candidate['candidate_id']}.json"
            if checkpoint.exists():
                cached = read_json(checkpoint)
                if cached["protocol_sha256"] != digest(protocol_file(root, dataset)):
                    raise AssertionError("Candidate checkpoint protocol mismatch")
                row = cached["metrics"]
            else:
                print(f"{dataset}/{seed}: fitting {candidate['candidate_id']} (8 threads)", flush=True)
                row, history, _ = fit_candidate(protocol, candidate, weights, a, y_train, b, y_val, 1200)
                write_json(checkpoint, {"protocol_sha256": digest(protocol_file(root, dataset)), "metrics": row, "history": history,
                                        "train_positions_sha256": row_hash(train), "validation_positions_sha256": row_hash(validation)})
                print(f"  {candidate['candidate_id']}: Macro-F1={row['macro_f1']:.6f}, best={row['best_iteration']}, {row['fit_seconds']:.1f}s", flush=True)
            rows.append(row)
        local_rule = {"selection_rule": {"eligibility_constraints": {"minimum_qwk": record["minimum_qwk"], "minimum_fatal_recall": record["minimum_fatal_recall"]}}}
        selected, eligible = select_candidate(rows, local_rule)
        candidate = next(c for c in protocol["candidates"] if c["candidate_id"] == selected["candidate_id"])
        if dataset == "cas" and selected["best_iteration"] == 1200:
            extended = directory / "boundary_extension.json"
            if extended.exists():
                selected = read_json(extended)["metrics"]
            else:
                selected, history, _ = fit_candidate(protocol, candidate, weights, a, y_train, b, y_val, 2000)
                write_json(extended, {"metrics": selected, "history": history, "protocol_sha256": digest(protocol_file(root, dataset))})
        final = make_model(protocol, candidate, weights, selected["best_iteration"])
        final.fit(a, y_train, categorical_feature=schema["categorical_feature_columns"])
        val_probability = final.predict_proba(b)
        final_metrics = full_metrics(y_val, val_probability.argmax(axis=1))
        for metric, value in final_metrics.items():
            np.testing.assert_allclose(value, selected[metric], rtol=0, atol=1e-12)
        model_path = models / str(seed) / "model.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"estimator": final, "bundle": bundle, "seed": seed, "dataset": dataset, "protocol_sha256": digest(protocol_file(root, dataset))}, model_path, compress=3)
        write_csv(directory / "candidate_metrics.csv", rows)
        write_csv(directory / "preprocessing_audit.csv", prep_audit)
        save_npz(directory / "validation_predictions.npz", positions=validation, target=y_val, probabilities=val_probability)
        write_csv(directory / "selection_roles.csv", [{"role": k, "rows": len(v), "positions_sha256": row_hash(v),
                   "used_for_fit": k == "train", "used_for_selection": k == "validation"} for k, v in positions.items()])
        files = [model_path, directory / "candidate_metrics.csv", directory / "preprocessing_audit.csv", directory / "validation_predictions.npz", directory / "selection_roles.csv"]
        write_json(selected_path, {"status": "MODEL_SEALED_BEFORE_REVISED_TEST_EVALUATION", "created_local": now(), "seed": seed,
                                  "dataset": dataset, "protocol_sha256": digest(protocol_file(root, dataset)),
                                  "candidate_id": candidate["candidate_id"], "best_iteration": selected["best_iteration"],
                                  "initial_selection_constraints_met": eligible,
                                  "final_constraints_met": selected["qwk"] >= record["minimum_qwk"] and selected["fatal_recall"] >= record["minimum_fatal_recall"],
                                  "validation_metrics": final_metrics, "class_weights": record["class_weights"],
                                  "files": {f.relative_to(root).as_posix(): digest(f) for f in files}})
        print(f"{dataset}/{seed}: sealed {candidate['candidate_id']} at {selected['best_iteration']} rounds", flush=True)
    write_json(config / "all_models_frozen.json", {"status": "ALL_FIVE_RANDOM_MODELS_SEALED", "created_local": now(),
              "protocol_sha256": digest(protocol_file(root, dataset)), "selected_files": {str(s): digest(config / f"selected_{s}.json") for s in SEEDS}})


def validate_selected(root, dataset, seed):
    config = output_dirs(root, dataset)[0]
    selected = read_json(config / f"selected_{seed}.json")
    if selected["protocol_sha256"] != digest(protocol_file(root, dataset)) or selected["seed"] != seed:
        raise AssertionError("Selected artifact identity mismatch")
    for name, expected in selected["files"].items():
        if digest(root / name) != expected:
            raise AssertionError(f"Selected artifact changed: {name}")
    return selected


def require_all_models(root, dataset):
    config = output_dirs(root, dataset)[0]
    seal = read_json(config / "all_models_frozen.json")
    for seed in SEEDS:
        validate_selected(root, dataset, seed)
        if digest(config / f"selected_{seed}.json") != seal["selected_files"][str(seed)]:
            raise AssertionError("All-model seal changed")


def prepare_saved(features, bundle):
    return prepare_features(features, feature_columns=bundle["feature_columns"], categorical_columns=bundle["categorical_columns"],
                            numeric_columns=bundle["numeric_columns"], category_vocabulary=bundle["category_vocabulary"]).frame


def evaluate(root, dataset):
    require_protocol(root, dataset)
    require_all_models(root, dataset)
    config, results, models, logs, _ = output_dirs(root, dataset)
    schema, features, target, metadata, assignments = load_data(root, dataset)
    spec = SPECS[dataset]
    metrics, confusions = [], []
    for seed in SEEDS:
        artifact = joblib.load(models / str(seed) / "model.joblib")
        positions = split_positions(assignments, seed, dataset)
        for cohort, role in (("internal", "test"), ("future", "locked_temporal_test")):
            pos = positions[role]
            path = results / str(seed) / f"{cohort}_predictions.npz"
            model_sha = digest(models / str(seed) / "model.joblib")
            if path.exists():
                with np.load(path) as cached:
                    if cached["model_sha256"].item() != model_sha:
                        raise AssertionError("Prediction cache belongs to another model")
                    np.testing.assert_array_equal(cached["positions"], pos)
                    np.testing.assert_array_equal(cached["target"], target.iloc[pos])
                    probabilities = cached["probabilities"].copy()
            else:
                frame = prepare_saved(features.iloc[pos], artifact["bundle"])
                probabilities = artifact["estimator"].predict_proba(frame)
                np.testing.assert_allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-10)
                save_npz(path, positions=pos, target=target.iloc[pos].to_numpy(), probabilities=probabilities, model_sha256=model_sha)
            y = target.iloc[pos].to_numpy()
            pred = probabilities.argmax(axis=1)
            metrics.append({"dataset": dataset, "seed": seed, "cohort": cohort, "rows": len(pos), **full_metrics(y, pred)})
            matrix = confusion_from_vectors(y, pred[:, None])[0]
            confusions.extend({"dataset": dataset, "seed": seed, "cohort": cohort, "true_class": i, "predicted_class": j, "count": int(matrix[i,j])} for i in range(3) for j in range(3))
            print(f"{dataset}/{seed}/{cohort}: revised evaluation saved", flush=True)
    write_csv(results / "revised_test_metrics.csv", metrics)
    write_csv(results / "revised_confusion_matrices.csv", confusions)
    write_json(logs / "evaluation_complete.json", {"status": "COMPLETE", "created_local": now(), "groups": 10,
                                                  "all_models_seal_sha256": digest(config / "all_models_frozen.json")})


def joint_bootstrap(y, predictions, rng, iterations=ITERATIONS):
    """Exact record-bootstrap distribution, retaining cross-model pairing."""
    predictions = np.asarray(predictions, dtype=int)
    if predictions.shape[0] != len(y) or predictions.ndim != 2:
        raise ValueError("Predictions must be rows by models")
    out = np.zeros((iterations, predictions.shape[1], 3, 3), dtype=np.int64)
    for true_class in range(3):
        subset = predictions[np.asarray(y) == true_class]
        if not len(subset):
            raise ValueError("Missing true severity class")
        patterns, counts = np.unique(subset, axis=0, return_counts=True)
        draws = rng.multinomial(len(subset), counts / len(subset), size=iterations)
        for m in range(predictions.shape[1]):
            for predicted in range(3):
                out[:, m, true_class, predicted] = draws[:, patterns[:, m] == predicted].sum(axis=1)
    return out


def interval(values):
    if not np.isfinite(values).all():
        raise AssertionError("Nonfinite bootstrap draws")
    a, b = np.quantile(values, [0.025, 0.975])
    return {"ci_lower": float(a), "ci_upper": float(b)}


def bootstrap(root, dataset):
    protocol = require_protocol(root, dataset)
    require_all_models(root, dataset)
    _, results, _, logs, _ = output_dirs(root, dataset)
    _, _, target, metadata, assignments = load_data(root, dataset)
    spec = SPECS[dataset]
    intervals, pairs, gaps = [], [], []
    for seed in SEEDS:
        samples = split_positions(assignments, seed, dataset)
        cohort_points, cohort_draws = {}, {}
        for cohort, role in (("internal", "test"), ("future", "locked_temporal_test")):
            pos = samples[role]
            y = target.iloc[pos].to_numpy()
            ids = metadata.iloc[pos][spec["id"]]
            original = [aligned_predictions(legacy_prediction_path(root, dataset, seed, cohort, m), ids, y, spec["id"])
                        for m in ("dummy_most_frequent", "logistic_weighted", "lightgbm_weighted")]
            with np.load(results / str(seed) / f"{cohort}_predictions.npz") as revised:
                np.testing.assert_array_equal(revised["positions"], pos)
                np.testing.assert_array_equal(revised["target"], y)
                predictions = np.column_stack([*original, revised["probabilities"].argmax(axis=1)])
            point = metrics_from_confusions(confusion_from_vectors(y, predictions))
            rng = np.random.default_rng(stream_seed(spec["bootstrap_seed"], f"revision:{dataset}:{seed}:{cohort}"))
            draws = metrics_from_confusions(joint_bootstrap(y, predictions, rng))
            cohort_points[cohort], cohort_draws[cohort] = point, draws
            save_npz(results / str(seed) / f"{cohort}_bootstrap.npz", **draws)
            for metric in point:
                for index, model in enumerate(MODELS):
                    intervals.append({"dataset": dataset, "seed": seed, "cohort": cohort, "model": model, "metric": metric,
                                      "estimate": float(point[metric][index]), **interval(draws[metric][:, index])})
                for reference in (0, 1, 2):
                    values = draws[metric][:, 3] - draws[metric][:, reference]
                    pairs.append({"dataset": dataset, "seed": seed, "cohort": cohort, "candidate": MODELS[3], "reference": MODELS[reference],
                                  "metric": metric, "delta": float(point[metric][3] - point[metric][reference]), **interval(values), "design": "paired_true_class_stratified"})
        for metric in cohort_points["internal"]:
            sign = -1 if metric in LOWER else 1
            for index, model in enumerate(MODELS):
                values = sign*(cohort_draws["internal"][metric][:, index] - cohort_draws["future"][metric][:, index])
                gaps.append({"dataset": dataset, "seed": seed, "model": model, "metric": metric,
                             "gap": float(sign*(cohort_points["internal"][metric][index] - cohort_points["future"][metric][index])),
                             **interval(values), "design": "independent_true_class_stratified", "iterations": ITERATIONS})
    write_csv(results / "metric_intervals.csv", intervals)
    write_csv(results / "paired_differences.csv", pairs)
    write_csv(results / "h1_gaps.csv", gaps)
    frame = pd.DataFrame(gaps)
    summary = frame.groupby(["dataset", "model", "metric"], sort=False).gap.agg(["mean", "std", "min", "max"]).reset_index()
    write_csv(results / "h1_summary.csv", summary)
    write_json(logs / "bootstrap_complete.json", {"status": "COMPLETE", "groups": 10, "iterations": ITERATIONS, "created_local": now()})
    print(f"{dataset}: paired and H1 bootstrap complete", flush=True)


def shap_analysis(root, dataset):
    require_protocol(root, dataset)
    require_all_models(root, dataset)
    _, results, models, logs, _ = output_dirs(root, dataset)
    schema, features, target, metadata, assignments = load_data(root, dataset)
    spec = SPECS[dataset]
    if dataset == "stats19":
        original_samples, sample_frame = build_samples(target, metadata, assignments)
        sample_frame.to_csv(results / "explanation_samples.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    importance, stability, audits = [], [], []
    for seed in SEEDS:
        artifact = joblib.load(models / str(seed) / "model.joblib")
        model_sha = digest(models / str(seed) / "model.joblib")
        if dataset == "stats19":
            positions = {"internal": original_samples[f"random_seed_{seed}_internal"], "future": original_samples["future_2024"]}
        else:
            internal, future, _ = matched_positions(assignments=assignments, target=target, seed=seed, base_seed=20260901)
            positions = {"internal": internal, "future": future}
        values_by_cohort, boot_by_cohort = {}, {}
        for cohort, pos in positions.items():
            chunks = []
            for start in range(0, len(pos), 1000):
                selected = pos[start:start+1000]
                chunk_path = results / str(seed) / "shap_values" / f"{cohort}_{start:05d}.npz"
                if chunk_path.exists():
                    with np.load(chunk_path) as stored:
                        if stored["model_sha256"].item() != model_sha:
                            raise AssertionError("SHAP cache model mismatch")
                        np.testing.assert_array_equal(stored["positions"], selected)
                        values = stored["values"].copy()
                        error = float(stored["additivity_error"])
                else:
                    print(f"{dataset}/{seed}: SHAP {cohort} {start+1}-{start+len(selected)} / {len(pos)}", flush=True)
                    if dataset == "stats19":
                        values, expected, audit = explain(features, selected, artifact["bundle"], artifact)
                        error = audit["max_abs_additivity_error"]
                    else:
                        frame = prepare_saved(features.iloc[selected], artifact["bundle"])
                        values = shap_contributions(artifact["estimator"], frame, len(schema["feature_columns"])).transpose(0, 2, 1)
                        raw = artifact["estimator"].booster_.predict(frame, raw_score=True, num_threads=THREADS)
                        baseline = artifact["estimator"].booster_.predict(frame.iloc[:1], pred_contrib=True, num_threads=THREADS).reshape(1, 3, -1)[:, :, -1]
                        error = float(np.max(np.abs(values.sum(axis=1) + baseline - raw)))
                    if error > 1e-6 or not np.isfinite(values).all():
                        raise AssertionError("Invalid SHAP additivity/values")
                    save_npz(chunk_path, positions=selected, values=values, model_sha256=model_sha, additivity_error=error)
                chunks.append(values)
                audits.append({"dataset": dataset, "seed": seed, "cohort": cohort, "start": start, "rows": len(selected), "max_additivity_error": error})
            values = np.concatenate(chunks)
            values_by_cohort[cohort] = values
            sample_path = results / str(seed) / f"{cohort}_explanation_positions.npz"
            save_npz(sample_path, positions=pos, labels=target.iloc[pos].to_numpy())
            labels = target.iloc[pos].to_numpy()
            if dataset == "stats19":
                key = "revision:future_2024" if cohort == "future" else f"revision:{seed}:internal"
                boots = bootstrap_mean_values(np.abs(values), labels, stream_key=key)
            else:
                rng = np.random.default_rng(20260901 + 10000000 + seed*10 + (1 if cohort == "internal" else 2))
                boots = bootstrap_mean_importance(np.abs(values).reshape(len(values), -1), labels, iterations=ITERATIONS, rng=rng)
                boots = boots.reshape(ITERATIONS, len(schema["feature_columns"]), 3)
            boot_by_cohort[cohort] = boots
            for class_index, scope in [(None, "overall"), *enumerate(spec["labels"])]:
                scoped = values.mean(axis=2) if class_index is None else values[:, :, class_index]
                mean_abs = np.abs(values).mean(axis=(0,2)) if class_index is None else np.abs(scoped).mean(axis=0)
                ranks = rankdata(-mean_abs, method="average")
                for index, feature in enumerate(schema["feature_columns"]):
                    importance.append({"dataset": dataset, "seed": seed, "cohort": cohort, "scope": scope, "feature": feature,
                                       "mean_abs_shap": float(mean_abs[index]), "rank": float(ranks[index]), "mean_signed_shap": float(scoped[:,index].mean()),
                                       "positive_fraction": float(np.mean(scoped[:,index] > 0)), "negative_fraction": float(np.mean(scoped[:,index] < 0))})
        for class_index, scope in [(None, "overall"), *enumerate(spec["labels"])]:
            a = np.abs(values_by_cohort["internal"]).mean(axis=0)
            b = np.abs(values_by_cohort["future"]).mean(axis=0)
            ba, bb = boot_by_cohort["internal"], boot_by_cohort["future"]
            if class_index is None:
                a, b, ba, bb = a.mean(axis=1), b.mean(axis=1), ba.mean(axis=2), bb.mean(axis=2)
            else:
                a, b, ba, bb = a[:,class_index], b[:,class_index], ba[:,:,class_index], bb[:,:,class_index]
            rho = float(spearmanr(a,b).statistic)
            draws = spearman_rows(ba,bb)
            stability.append({"dataset": dataset, "seed": seed, "scope": scope, "rho": rho, **interval(draws),
                              "internal_rows": len(positions["internal"]), "future_rows": len(positions["future"]), "iterations": ITERATIONS})
            save_npz(results / str(seed) / f"shap_rank_draws_{scope}.npz", rho=draws)
        print(f"{dataset}/{seed}: SHAP ranks and intervals complete", flush=True)
    write_csv(results / "shap_importance.csv", importance)
    write_csv(results / "shap_stability.csv", stability)
    write_csv(results / "shap_additivity_audit.csv", audits)
    frame = pd.DataFrame(stability)
    write_csv(results / "shap_summary.csv", frame.groupby(["dataset","scope"],sort=False).rho.agg(["mean","std","min","max"]).reset_index())
    write_json(logs / "shap_complete.json", {"status": "COMPLETE", "created_local": now(), "models": 5, "cohorts": 10, "iterations": ITERATIONS})


def summarize(root, dataset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, results, _, logs, figures = output_dirs(root, dataset)
    figures.mkdir(parents=True, exist_ok=True)
    gaps = pd.read_csv(results / "h1_gaps.csv")
    ranks = pd.read_csv(results / "shap_stability.csv")
    metrics = ["macro_f1", "qwk", "fatal_recall", "mean_asymmetric_cost"]
    fig, axes = plt.subplots(2,2,figsize=(9,6), layout="constrained")
    for ax, metric in zip(axes.flat,metrics):
        for offset, model, color, label in [(-.12,"lightgbm_legacy","#777777","Legacy"),(.12,"lightgbm_revised","#007C91","Split-local revision")]:
            frame = gaps.loc[gaps.metric.eq(metric) & gaps.model.eq(model)].sort_values("seed")
            x = np.arange(len(frame)) + offset
            ax.vlines(x, frame.ci_lower, frame.ci_upper, color=color)
            ax.scatter(x, frame.gap, color=color,s=18,label=label)
        ax.axhline(0,color="black",lw=.7)
        ax.set_xticks(range(5), [str(s) for s in SEEDS],fontsize=8)
        ax.set_title(metric.replace("_"," "),fontsize=11)
        ax.set_ylabel("Oriented internal - future gap",fontsize=9)
    axes[0,0].legend(fontsize=8)
    fig.savefig(figures / "h1_revision.png",dpi=220)
    fig.savefig(figures / "h1_revision.pdf")
    plt.close(fig)
    frame = ranks.loc[ranks.scope.eq("overall")].sort_values("seed")
    fig, ax = plt.subplots(figsize=(6,3.5),layout="constrained")
    ax.vlines(range(5),frame.ci_lower,frame.ci_upper,color="#007C91")
    ax.scatter(range(5),frame.rho,color="#007C91")
    ax.set_xticks(range(5),[str(s) for s in SEEDS])
    ax.set_ylim(-1,1.02)
    ax.set_ylabel("SHAP rank Spearman correlation")
    ax.set_xlabel("Random split seed")
    fig.savefig(figures / "shap_revision.png",dpi=220)
    fig.savefig(figures / "shap_revision.pdf")
    plt.close(fig)
    h1 = pd.read_csv(results / "h1_summary.csv")
    h3 = pd.read_csv(results / "shap_summary.csv")
    lines = [f"# {dataset.upper()} random-reference post-review correction", "", "All original outcomes were known before this correction. No preregistration or newly untouched test is claimed.",
             "", "Only split-local random LightGBM selection and dependent outputs are revised. Historical temporal models and H2 are unchanged.", "", "## Split-local selection"]
    for seed in SEEDS:
        selected = validate_selected(root,dataset,seed)
        lines.append(f"- {seed}: {selected['candidate_id']}, {selected['best_iteration']} iterations; constraints met: {selected['final_constraints_met']}.")
    lines.extend(["", "## H1 oriented gaps (mean and sample SD across five seeds)"])
    for row in h1.loc[h1.model.eq("lightgbm_revised")].to_dict("records"):
        lines.append(f"- {row['metric']}: {row['mean']:.6f} +/- {row['std']:.6f}; range [{row['min']:.6f}, {row['max']:.6f}].")
    row = h3.loc[h3.scope.eq("overall")].iloc[0]
    lines.extend(["", "## H3", f"Mean rho {row['mean']:.6f}, sample SD {row['std']:.6f}, range [{row['min']:.6f}, {row['max']:.6f}].",
                  "", "Results must be interpreted metric by metric. No causal claim, equivalence claim or outcome-based choice between legacy/revised results is supported.",
                  "The two sources are not pooled. Bootstrap conditions on fixed fitted models and observed class counts.", ""])
    (logs / "checkpoint.md").write_text("\n".join(lines),encoding="utf-8")


def verify(root,dataset):
    protocol = require_protocol(root,dataset)
    require_all_models(root,dataset)
    config,results,models,logs,figures = output_dirs(root,dataset)
    schema, features, target, metadata, assignments = load_data(root,dataset)
    checks = []
    for record in protocol["splits"]:
        seed = record["seed"]
        selected = validate_selected(root,dataset,seed)
        pos = split_positions(assignments,seed,dataset)
        for role, values in pos.items():
            if row_hash(values) != record["roles"][role]["positions_sha256"]:
                raise AssertionError("Selection role mismatch")
        rows = pd.read_csv(results / str(seed) / "candidate_metrics.csv").to_dict("records")
        rule = {"selection_rule":{"eligibility_constraints":{"minimum_qwk":record["minimum_qwk"],"minimum_fatal_recall":record["minimum_fatal_recall"]}}}
        expected,_ = select_candidate(rows,rule)
        if expected["candidate_id"] != selected["candidate_id"] or len(rows) != 6:
            raise AssertionError("Candidate selection mismatch")
        artifact = joblib.load(models / str(seed) / "model.joblib")
        train = pos["train"]
        for col, levels in artifact["bundle"]["category_vocabulary"].items():
            if levels != sorted(str(v) for v in features.iloc[train][col].unique()):
                raise AssertionError("Vocabulary not training-only")
        for cohort,role in (("internal","test"),("future","locked_temporal_test")):
            with np.load(results / str(seed) / f"{cohort}_predictions.npz") as prediction:
                np.testing.assert_array_equal(prediction["positions"],pos[role])
                np.testing.assert_array_equal(prediction["target"],target.iloc[pos[role]])
                model_sha = digest(models / str(seed) / "model.joblib")
                if prediction["model_sha256"].item() != model_sha:
                    raise AssertionError("Test prediction model hash mismatch")
                # Independent sample re-prediction verifies serialized model/column order.
                offsets = np.unique(np.linspace(0,len(pos[role])-1,101,dtype=int))
                frame = prepare_saved(features.iloc[pos[role][offsets]],artifact["bundle"])
                np.testing.assert_allclose(artifact["estimator"].predict_proba(frame),prediction["probabilities"][offsets],rtol=1e-12,atol=1e-12)
                values = full_metrics(prediction["target"],prediction["probabilities"].argmax(axis=1))
            metrics = pd.read_csv(results / "revised_test_metrics.csv")
            row = metrics.loc[metrics.seed.eq(seed) & metrics.cohort.eq(cohort)].iloc[0]
            for key,value in values.items():
                np.testing.assert_allclose(row[key],value,rtol=0,atol=1e-12)
        checks.append({"seed":seed,"selection_test_overlap":0,"serialized_predictions_checked":True,"metrics_recomputed":True})
    for file, key, expected in [("h1_gaps.csv","gap",140),("shap_stability.csv","rho",20),("revised_test_metrics.csv","macro_f1",10)]:
        frame = pd.read_csv(results / file)
        if len(frame) != expected or not np.isfinite(frame[key]).all():
            raise AssertionError(f"Incomplete result: {file}")
    audit = pd.read_csv(results / "shap_additivity_audit.csv")
    if audit.max_additivity_error.max() > 1e-6:
        raise AssertionError("SHAP additivity failed")
    historical = read_json(root / protocol["historical_manifest"])
    if digest(root / protocol["historical_manifest"]) != protocol["historical_manifest_sha256"]:
        raise AssertionError("Historical protection manifest changed")
    for file, expected in historical.items():
        if digest(root / file) != expected:
            raise AssertionError(f"Historical artifact modified: {file}")
    if (logs / "complete.json").exists():
        complete = read_json(logs / "complete.json")
        if digest(logs / "artifact_manifest.json") != complete["manifest_sha256"]:
            raise AssertionError("Completed revision manifest changed")
        for file, expected in read_json(logs / "artifact_manifest.json").items():
            if digest(root / file) != expected:
                raise AssertionError(f"Completed revision artifact changed: {file}")
        print(f"{dataset}: existing completed revision reverified",flush=True)
        return
    files = [p for directory in (config,results,models,logs,figures) for p in directory.rglob("*") if p.is_file()
             and p.name not in {"artifact_manifest.json","complete.json"} and not p.name.endswith(".tmp")]
    write_json(logs / "artifact_manifest.json",{p.relative_to(root).as_posix():digest(p) for p in sorted(files)})
    write_json(logs / "complete.json",{"status":"POST_REVIEW_RANDOM_REFERENCE_CORRECTION_VERIFIED","created_local":now(),"dataset":dataset,
               "checks":checks,"historical_files_unchanged":len(historical),"protocol_sha256":digest(protocol_file(root,dataset)),
               "manifest_sha256":digest(logs / "artifact_manifest.json"),"untouched_temporal_models_and_H2":True,
               "legacy_outcomes_known":True,"new_preregistration_claim":False})
    print(f"{dataset}: CORRECTION VERIFIED; {len(historical)} historical files unchanged",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--dataset", choices=["stats19","cas","all"], default="all")
    parser.add_argument("--stage", choices=["freeze","smoke","tune","evaluate","bootstrap","shap","summarize","verify"])
    parser.add_argument("--all",action="store_true")
    args = parser.parse_args()
    if not args.all and not args.stage:
        parser.error("Choose --all or --stage")
    root = args.project_root.resolve()
    datasets = list(SPECS) if args.dataset == "all" else [args.dataset]
    stages = ["freeze","smoke","tune","evaluate","bootstrap","shap","summarize","verify"] if args.all else [args.stage]
    for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(THREADS)
    with execution_lock(root), threadpool_limits(limits=THREADS):
        for stage in stages:
            for dataset in datasets:
                if stage != "verify" and (output_dirs(root,dataset)[3] / "complete.json").exists():
                    print(f"{dataset}: completed revision retained; skipping {stage}",flush=True)
                    continue
                print(f"STAGE={stage} DATASET={dataset} TIME={now()}",flush=True)
                if stage == "smoke":
                    tune(root,dataset,smoke=True)
                elif stage == "shap":
                    shap_analysis(root,dataset)
                else:
                    globals()[stage](root,dataset)


if __name__ == "__main__":
    main()
