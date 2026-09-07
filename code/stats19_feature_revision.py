"""Prepare and smoke-test a separate, post-review STATS19 feature revision.

This entry point deliberately has no full-training or test-evaluation command.
All scientific inputs and historical outputs are read-only; smoke artifacts
are not eligible for manuscript reporting or subsequent model selection.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from baseline_modeling import (
    ASYMMETRIC_COST_MATRIX, assert_encoded_matrix, classification_metrics,
    make_preprocessor, prepare_features,
)
from d10_tune_lightgbm import prepare_native_split, select_candidate

ROOT = Path(__file__).resolve().parents[1]
NAME = "stats19_feature_revision"
VERSION = "STATS19_15_FEATURE_POST_REVIEW_V1"
PREPARATION_STATUS = "PREPARATION_VERIFIED_FULL_EXPERIMENT_NOT_RUN"
SEEDS = (1103, 2207, 3301, 4409, 5501)
SMOKE_SEED = 20260907
SMOKE_TRAIN = 3000
SMOKE_VALIDATION = 1200
SMOKE_TREES = 30
SMOKE_THREADS = 4
ID = "meta_collision_index"
YEAR = "meta_collision_year"
TARGET = "target_severity"
DATA = "data/processed/stats19_modeling_dataset.csv.gz"
ASSIGNMENTS = "data/processed/d6_split_assignments.csv.gz"
SCHEMA = "config/d5_dataset_schema.json"
EXCLUDED = ("feature_special_conditions_at_site", "feature_carriageway_hazards")
FEATURES = (
    "feature_day_of_week", "feature_first_road_class", "feature_road_type",
    "feature_junction_detail", "feature_junction_control", "feature_second_road_class",
    "feature_pedestrian_crossing", "feature_light_conditions", "feature_weather_conditions",
    "feature_road_surface_conditions", "feature_urban_or_rural_area", "feature_trunk_road_flag",
    "feature_speed_limit", "feature_month", "feature_hour",
)
MODEL_NAMES = ("dummy_most_frequent", "logistic_unweighted", "logistic_weighted", "lightgbm_weighted")
PARENTS = (
    SCHEMA, "config/d3_feature_manifest.json", "config/d5_cleaning_rules.json",
    "config/d6_analysis_protocol.json", "config/d8_baseline_protocol.json",
    "config/d10_lightgbm_protocol.json", "config/d14_shap_protocol.json",
)
SOURCE_FILES = (
    "run_stats19_feature_revision.py", "code/stats19_feature_revision.py",
    "tests/test_stats19_feature_revision.py", "code/baseline_modeling.py",
    "code/d10_tune_lightgbm.py", "code/modeling_data.py", "requirements-lock.txt",
)


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def position_hash(positions):
    return hashlib.sha256(np.asarray(positions, dtype="<i8").tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def output_path(root, folder, *parts):
    if folder not in {"config", "results", "models", "logs", "figures"}:
        raise ValueError("Unexpected output family")
    base = (root / folder / NAME).resolve()
    if not base.is_relative_to(root.resolve()):
        raise ValueError("Revision output base escaped project root")
    path = base.joinpath(*parts).resolve()
    if not path.is_relative_to(base):
        raise ValueError("Output escaped revision directory")
    return path


@contextmanager
def execution_lock(root):
    path = output_path(root, "logs", ".execution.lock")
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


def revision_schema(original):
    original_features = original["feature_columns"]
    if len(original_features) != 17 or len(set(original_features)) != 17:
        raise ValueError("Original schema must have 17 unique features")
    if not set(EXCLUDED).issubset(original_features):
        raise ValueError("The two reviewed fields are absent from the original schema")
    if [c for c in original_features if c not in EXCLUDED] != list(FEATURES):
        raise ValueError("Unexpected original feature order or membership")
    cats = [c for c in original["categorical_feature_columns"] if c not in EXCLUDED]
    numeric = list(original["numeric_feature_columns"])
    if set(cats) != set(FEATURES) - {"feature_speed_limit"} or numeric != ["feature_speed_limit"]:
        raise ValueError("Unexpected feature type assignment")
    return {
        "version": VERSION, "feature_columns": list(FEATURES),
        "categorical_feature_columns": cats, "numeric_feature_columns": numeric,
        "excluded_feature_columns": list(EXCLUDED), "metadata_columns": original["metadata_columns"],
        "target_column": TARGET, "feature_count": 15,
    }


def validate_features(frame):
    if frame.columns.tolist() != list(FEATURES):
        raise ValueError("Expected exactly the ordered 15-feature allowlist; legacy or metadata columns forbidden")


def load_data(root):
    original = read_json(root / SCHEMA)
    schema = revision_schema(original)
    expected = original["metadata_columns"] + [TARGET] + original["feature_columns"]
    if pd.read_csv(root / DATA, nrows=0).columns.tolist() != expected:
        raise ValueError("Input table schema changed")
    # Dropped columns are not loaded into the model-facing data at all.
    usecols = original["metadata_columns"] + [TARGET] + list(FEATURES)
    table = pd.read_csv(root / DATA, usecols=usecols, keep_default_na=False, na_values=[""],
                        dtype={ID: "string", **{c: "string" for c in schema["categorical_feature_columns"]}})
    assignments = pd.read_csv(root / ASSIGNMENTS, dtype={ID: "string"}, keep_default_na=False)
    if len(table) != original["row_count"] or table[ID].isna().any() or table[ID].duplicated().any():
        raise ValueError("Input record count or identifiers changed")
    if not table[ID].equals(assignments[ID]):
        raise ValueError("Input and split identifiers are not aligned")
    for col in (YEAR, TARGET):
        np.testing.assert_array_equal(table[col], assignments[col])
    y = table[TARGET].astype("int8")
    if sorted(y.unique()) != [0, 1, 2]:
        raise ValueError("Unexpected severity classes")
    X = table.loc[:, list(FEATURES)].copy()
    X["feature_speed_limit"] = pd.to_numeric(X["feature_speed_limit"], errors="raise")
    if not X.feature_speed_limit.dropna().isin([20, 30, 40, 50, 60, 70]).all():
        raise ValueError("Unreviewed numeric speed code")
    if X[schema["categorical_feature_columns"]].isna().any().any():
        raise ValueError("Categorical unknowns must remain explicit categories")
    validate_features(X)
    return schema, X, y, table[original["metadata_columns"]].copy(), assignments


def assert_disjoint(roles):
    values = [np.asarray(v, dtype=int) for v in roles.values()]
    for i, a in enumerate(values):
        if len(a) != len(np.unique(a)):
            raise ValueError("Duplicate role positions")
        for b in values[i + 1:]:
            if np.intersect1d(a, b).size:
                raise ValueError("Training, selection and test positions overlap")


def split_positions(assignments, slug, strict=True):
    if slug == "temporal":
        column = "temporal_role"
        mapping = {"train": "train", "validation": "validation", "future": "test"}
        counts = [538461, 104258, 100927]
    elif slug in {f"random_seed_{seed}" for seed in SEEDS}:
        column = f"random_role_seed_{slug.rsplit('_', 1)[1]}"
        mapping = {"train": "train", "validation": "validation", "internal": "test", "future": "locked_temporal_test"}
        counts = [449903, 96408, 96408, 100927]
    else:
        raise ValueError("Unknown split")
    roles = {key: np.flatnonzero(assignments[column].eq(value)) for key, value in mapping.items()}
    assert_disjoint(roles)
    if sum(map(len, roles.values())) != len(assignments):
        raise ValueError("Unknown or omitted assignment role")
    if strict:
        if [len(v) for v in roles.values()] != counts:
            raise ValueError("Frozen split sizes changed")
        years = assignments[YEAR].to_numpy()
        np.testing.assert_array_equal(roles["future"], np.flatnonzero(years == 2024))
        if slug == "temporal":
            np.testing.assert_array_equal(roles["train"], np.flatnonzero(np.isin(years, range(2018, 2023))))
            np.testing.assert_array_equal(roles["validation"], np.flatnonzero(years == 2023))
    return roles


def weights_from_training(y):
    counts = np.bincount(np.asarray(y, dtype=int), minlength=3)
    if len(counts) != 3 or (counts == 0).any():
        raise ValueError("Training requires each of three severity classes")
    return {i: float(len(y) / (3 * counts[i])) for i in range(3)}


def stratified_subset(positions, target, count, seed):
    positions = np.asarray(positions, dtype=int)
    if count >= len(positions) or count < 3:
        raise ValueError("Smoke sample must be strictly smaller than its source partition")
    chosen, _ = train_test_split(positions, train_size=count, stratify=np.asarray(target)[positions], random_state=seed)
    return np.sort(chosen)


@dataclass
class SelectionData:
    train: pd.DataFrame
    train_target: np.ndarray
    validation: pd.DataFrame
    validation_target: np.ndarray
    train_positions: np.ndarray
    validation_positions: np.ndarray


def selection_data(X, target, roles, train_positions=None, validation_positions=None):
    validate_features(X)
    assert_disjoint(roles)
    train = roles["train"] if train_positions is None else np.asarray(train_positions)
    val = roles["validation"] if validation_positions is None else np.asarray(validation_positions)
    if not np.isin(train, roles["train"]).all() or not np.isin(val, roles["validation"]).all():
        raise ValueError("Selection data must come from the same split's training and validation roles")
    assert_disjoint({"train": train, "validation": val})
    y = np.asarray(target)
    return SelectionData(X.iloc[train].copy(), y[train].copy(), X.iloc[val].copy(), y[val].copy(), train, val)


def check_probabilities(probability, rows):
    p = np.asarray(probability, dtype=float)
    if p.shape != (rows, 3) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Invalid three-class probabilities")
    np.testing.assert_allclose(p.sum(axis=1), 1, rtol=0, atol=1e-8)
    return p


def local_selection(rows, baseline):
    constraints = {"minimum_qwk": float(baseline["qwk"]) - 0.01,
                   "minimum_fatal_recall": float(baseline["fatal_recall"]) - 0.05}
    for row in rows:
        if not all(np.isfinite(row[key]) for key in ("macro_f1", "qwk", "fatal_recall")):
            raise ValueError("Nonfinite candidate metrics")
    selected, eligible = select_candidate(rows, {"selection_rule": {"eligibility_constraints": constraints}})
    return selected, eligible, constraints


def fit_validation(data, schema, rules, tree_cap, threads):
    """Fit on the supplied training data; only validation labels enter selection."""
    validate_features(data.train)
    validate_features(data.validation)
    if tree_cap < 1 or tree_cap > 1200 or threads not in range(1, 9):
        raise ValueError("Out-of-protocol tree/thread limit")
    weights = weights_from_training(data.train_target)
    native_train, native_val, bundle, audit = prepare_native_split(data.train, data.validation, schema)
    preprocessor = make_preprocessor(categorical_columns=schema["categorical_feature_columns"],
                                    numeric_columns=schema["numeric_feature_columns"],
                                    category_vocabulary=bundle["category_vocabulary"])
    encoded_train = preprocessor.fit_transform(native_train)
    encoded_val = preprocessor.transform(native_val)
    assert_encoded_matrix(encoded_train, len(data.train))
    assert_encoded_matrix(encoded_val, len(data.validation))
    artifacts, probabilities, metrics, runtime = {}, {}, {}, {}
    with threadpool_limits(limits=threads):
        for name in MODEL_NAMES[:-1]:
            if name == "dummy_most_frequent":
                estimator = DummyClassifier(strategy="most_frequent")
            else:
                parameters = rules["baseline_models"][name]
                estimator = LogisticRegression(**parameters, class_weight=weights if name == "logistic_weighted" else None)
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always", ConvergenceWarning)
                estimator.fit(encoded_train, data.train_target)
            runtime[name] = {"seconds": time.perf_counter() - started,
                             "convergence_warning": any(issubclass(w.category, ConvergenceWarning) for w in seen),
                             "iterations": np.asarray(getattr(estimator, "n_iter_", [0])).tolist()}
            np.testing.assert_array_equal(estimator.classes_, [0, 1, 2])
            probability = check_probabilities(estimator.predict_proba(encoded_val), len(data.validation))
            metrics[name] = classification_metrics(data.validation_target, probability.argmax(axis=1))
            probabilities[name] = probability
            artifacts[name] = {"estimator": estimator, "preprocessor": preprocessor, "bundle": bundle}
        rows, histories, candidate_probabilities = [], {}, {}
        for candidate in rules["candidates"]:
            common = {**rules["common_parameters"], "n_jobs": threads}
            seed = common["random_state"]
            estimator = lgb.LGBMClassifier(**common, **candidate["parameters"], n_estimators=tree_cap,
                                          class_weight=weights, bagging_seed=seed,
                                          feature_fraction_seed=seed, data_random_seed=seed, importance_type="gain")
            history = {}
            started = time.perf_counter()
            estimator.fit(native_train, data.train_target, eval_X=native_val, eval_y=data.validation_target,
                          eval_metric="multi_logloss", categorical_feature=schema["categorical_feature_columns"],
                          callbacks=[lgb.early_stopping(75, first_metric_only=True, verbose=False), lgb.record_evaluation(history)])
            np.testing.assert_array_equal(estimator.classes_, [0, 1, 2])
            p = check_probabilities(estimator.predict_proba(native_val), len(data.validation))
            cid = candidate["candidate_id"]
            rows.append({"candidate_id": cid, "complexity_rank": candidate["complexity_rank"],
                         "best_iteration": int(estimator.best_iteration_ or tree_cap),
                         "tree_cap": tree_cap, "seconds": time.perf_counter() - started,
                         **classification_metrics(data.validation_target, p.argmax(axis=1))})
            candidate_probabilities[cid], histories[cid] = p, history
        selected, eligible, constraints = local_selection(rows, metrics["logistic_weighted"])
        candidate = next(c for c in rules["candidates"] if c["candidate_id"] == selected["candidate_id"])
        common = {**rules["common_parameters"], "n_jobs": threads}
        seed = common["random_state"]
        final = lgb.LGBMClassifier(**common, **candidate["parameters"], n_estimators=selected["best_iteration"],
                                  class_weight=weights, bagging_seed=seed, feature_fraction_seed=seed,
                                  data_random_seed=seed, importance_type="gain")
        final.fit(native_train, data.train_target, categorical_feature=schema["categorical_feature_columns"])
        p = check_probabilities(final.predict_proba(native_val), len(data.validation))
        np.testing.assert_allclose(p, candidate_probabilities[selected["candidate_id"]], rtol=1e-12, atol=1e-12)
        artifacts["lightgbm_weighted"] = {"estimator": final, "bundle": bundle}
        probabilities["lightgbm_weighted"] = p
        metrics["lightgbm_weighted"] = classification_metrics(data.validation_target, p.argmax(axis=1))
    return {"artifacts": artifacts, "probabilities": probabilities, "metrics": metrics,
            "candidate_metrics": rows, "histories": histories, "selected": selected,
            "constraints_met": eligible, "constraints": constraints, "class_weights": weights,
            "runtime": runtime, "preprocessing_audit": audit}


def predict_artifact(artifact, X):
    validate_features(X)
    bundle = artifact["bundle"]
    if bundle["feature_columns"] != list(FEATURES):
        raise ValueError("Legacy or incompatible model artifact")
    prepared = prepare_features(X, feature_columns=bundle["feature_columns"],
                                categorical_columns=bundle["categorical_columns"],
                                numeric_columns=bundle["numeric_columns"],
                                category_vocabulary=bundle["category_vocabulary"]).frame
    encoded = artifact["preprocessor"].transform(prepared) if "preprocessor" in artifact else prepared
    np.testing.assert_array_equal(artifact["estimator"].classes_, [0, 1, 2])
    return check_probabilities(artifact["estimator"].predict_proba(encoded), len(X))


def rules_from_parents(root):
    d8 = read_json(root / "config/d8_baseline_protocol.json")
    d10 = read_json(root / "config/d10_lightgbm_protocol.json")
    baseline = {}
    for name in ("logistic_unweighted", "logistic_weighted"):
        baseline[name] = {k: d8["models"][name][k] for k in ("penalty", "C", "solver", "max_iter", "tol")}
        baseline[name].update(fit_intercept=True, random_state=20260828)
    if [c["candidate_id"] for c in d10["tuning"]["candidates"]] != [f"C{i:02}" for i in range(1, 7)]:
        raise ValueError("Expected the original six-candidate search")
    return {"baseline_models": baseline, "candidates": d10["tuning"]["candidates"],
            "common_parameters": {**d10["common_model_parameters"], "n_jobs": 8},
            "maximum_estimators": 1200, "early_stopping_rounds": 75,
            "early_stopping_metric": "unweighted multiclass log loss on that split's own validation",
            "ranking": d10["selection_rule"]["ranking"], "fallback": d10["selection_rule"]["fallback_if_none_eligible"],
            "constraint_reference": "new 15-feature weighted Logistic in the SAME split; QWK minus 0.01, fatal recall minus 0.05",
            "refit": "training-only at selected iteration; no train-plus-validation refit; compare validation probabilities",
            "prediction": "three-class argmax; no threshold optimization", "cost_matrix": ASYMMETRIC_COST_MATRIX.tolist()}


def historical_files(root):
    protected = {}
    for folder in ("config", "code", "tests", "models", "results", "logs", "figures"):
        for path in sorted((root / folder).rglob("*")):
            relative = path.relative_to(root)
            if (path.is_file() and NAME not in relative.parts and path.name not in {"stats19_feature_revision.py", "test_stats19_feature_revision.py"}
                    and "__pycache__" not in relative.parts and not path.name.endswith((".tmp", ".log", ".lock", ".pyc"))):
                protected[relative.as_posix()] = digest(path)
    return protected


def check_hashes(root, hashes):
    for name, expected in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or digest(path) != expected:
            raise ValueError(f"Protected file changed or missing: {name}")


def prepare(root):
    protocol_path = output_path(root, "config", "protocol.json")
    if protocol_path.exists():
        require_protocol(root)
        print("Existing preparation protocol verified; not overwritten.", flush=True)
        return
    original = read_json(root / SCHEMA)
    if digest(root / DATA) != original["sha256"]:
        raise ValueError("Data differs from the frozen D5 snapshot")
    d7 = read_json(root / "config/d7_training_inputs.json")
    if digest(root / ASSIGNMENTS) != d7["upstream"]["D6_assignments_sha256"]:
        raise ValueError("Assignment bytes differ from the frozen D7 reference")
    schema, X, target, metadata, assignments = load_data(root)
    splits = []
    for slug in ["temporal", *[f"random_seed_{s}" for s in SEEDS]]:
        roles = split_positions(assignments, slug)
        y_train = target.iloc[roles["train"]].to_numpy()
        splits.append({"slug": slug, "roles": {key: {"rows": len(v), "positions_sha256": position_hash(v)} for key, v in roles.items()},
                       "training_class_counts": np.bincount(y_train, minlength=3).tolist(),
                       "training_class_weights": weights_from_training(y_train)})
    rules = rules_from_parents(root)
    guide = "data/external/documentation/dft-road-casualty-statistics-road-safety-open-dataset-data-guide-2025.xlsx"
    if not (root / guide).is_file():
        raise FileNotFoundError("The reviewed official 2025 dictionary is required for this author-side preparation")
    upstream = [DATA, ASSIGNMENTS, *PARENTS, "config/d7_training_inputs.json", guide]
    protected = historical_files(root)
    manifest = output_path(root, "logs", "historical_files.json")
    write_json(manifest, protected)
    audit = [
        {"field": EXCLUDED[0], "action": "EXCLUDE_SCHEMA_TRANSITION", "reason": "Legacy special conditions mapped into harmonized carriageway hazards; marked 2024 availability shift.",
         "official_evidence": "2025 guide, 2011_to_2024_conversion, rows 31-39", "claim": "Observed recording-process association; performance contribution not yet quantified"},
        {"field": EXCLUDED[1], "action": "EXCLUDE_OUTCOME_QUALIFIED_CATEGORY", "reason": "Category 19 specifies pedestrian not injured; inconsistent with fixed outcome-excluded prediction time.",
         "official_evidence": "2025 guide, 2024_code_list, row 1394", "claim": "Conservative feature removal; not a demonstrated deterministic target copy"},
    ]
    schema_path = output_path(root, "config", "feature_schema.json")
    audit_path = output_path(root, "config", "field_audit_amendment.json")
    write_json(schema_path, schema)
    write_json(audit_path, {"original_audit_preserved": True, "prediction_time_unchanged": True, "changes": audit})
    protocol = {
        "version": VERSION, "created_local": now(), "status": "METHOD_RULES_FIXED_BEFORE_REVISION_SMOKE_AND_FULL_FITS",
        "dataset": "STATS19", "legacy_outcomes_known": True, "post_hoc": True, "preregistered": False,
        "new_unseen_test_claim": False, "full_run_performed": False,
        "reason": "Review-driven removal of two semantically problematic fields; not performance-based feature selection",
        "reporting": "Report the corrected 15-feature run regardless of direction; preserve legacy 17-feature results, never select a preferred version by outcome",
        "schema": schema, "rules": rules, "splits": splits,
        "prediction_time": "at collision; no consequence, response, injury-derived, or severity-adjustment inputs",
        "future_scope": {
            "primary": "Dummy, full unweighted/weighted Logistic and weighted LightGBM in temporal plus five random splits",
            "test_gate": "all six split model sets must be sealed before corrected future/internal test evaluation",
            "evaluation_groups": ["temporal_2024", *[f"random_seed_{s}_{c}" for s in SEEDS for c in ("internal", "2024")]],
            "H1": "same fitted random model internal-minus-future; reverse sign for error metrics; no causal split-method claim",
            "H2": "corrected temporal LightGBM vs corrected weighted Logistic, all metrics; never infer noninferiority from an interval including zero",
            "bootstrap": "2000 true-class-stratified record draws; paired within cohort; independent across disjoint cohorts; descriptive five-seed summary",
            "bootstrap_seed": 20260828,
            "shap": "existing fixed STATS19 10000-record sampling design, raw multiclass TreeSHAP, equal-output mean absolute ranks, 2000 cached-contribution bootstrap draws",
            "recording_diagnostics": "year by injury-based recording flag by severity counts; descriptive only, metadata never features",
            "secondary": "Ordered Logit/matched subset, tree-count and exclusion-2020 analyses must be explicitly rerun on 15 fields or omitted/labeled historical; no silent reuse",
            "not_claimed": "No spatial, causal, cross-country, calibrated-probability or newly untouched-test validation",
            "CAS": "unchanged; no CAS retraining in this revision",
        },
        "preparation_scope": {"commands": ["prepare", "smoke", "verify"], "full_training_cli_available": False,
                              "bootstrap_executed": False, "shap_executed": False,
                              "smoke": {"train_rows": SMOKE_TRAIN, "validation_rows": SMOKE_VALIDATION, "tree_cap": SMOKE_TREES,
                                        "threads": SMOKE_THREADS, "seed": SMOKE_SEED, "all_six_splits": True,
                                        "test_performance_access": False, "eligible_for_reporting_or_model_selection": False}},
        "upstream_sha256": {p: digest(root / p) for p in sorted(set(upstream))},
        "source_sha256": {p: digest(ROOT / p) for p in SOURCE_FILES},
        "preparation_files_sha256": {p.relative_to(root).as_posix(): digest(p) for p in (schema_path, audit_path, manifest)},
        "historical_manifest": manifest.relative_to(root).as_posix(),
        "runtime": {"python": platform.python_version(), **{p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "joblib")}},
    }
    write_json(protocol_path, protocol)
    print(f"Prepared 15-feature protocol; {len(protected)} historical files protected; no models fitted.", flush=True)


def require_protocol(root):
    protocol = read_json(output_path(root, "config", "protocol.json"))
    if protocol["version"] != VERSION or protocol["schema"]["feature_columns"] != list(FEATURES):
        raise ValueError("Wrong feature revision protocol")
    check_hashes(root, protocol["upstream_sha256"])
    check_hashes(ROOT, protocol["source_sha256"])
    check_hashes(root, protocol["preparation_files_sha256"])
    return protocol


def require_analytical_artifact(artifact):
    if artifact.get("analytical_result") is not True or artifact.get("feature_version") != VERSION:
        raise ValueError("Smoke or legacy artifacts cannot be used as corrected analytical models")


def smoke(root):
    protocol = require_protocol(root)
    summary_path = output_path(root, "logs", "smoke_complete.json")
    if summary_path.exists():
        verify(root)
        print("Existing smoke run verified; no models refitted.", flush=True)
        return
    schema, X, target, _, assignments = load_data(root)
    records, all_files = [], {}
    for split in protocol["splits"]:
        slug = split["slug"]
        print(f"SMOKE {slug}: train={SMOKE_TRAIN}, validation={SMOKE_VALIDATION}, 6 candidates x {SMOKE_TREES} rounds", flush=True)
        roles = split_positions(assignments, slug)
        for role, positions in roles.items():
            if position_hash(positions) != split["roles"][role]["positions_sha256"]:
                raise ValueError("Split positions changed")
        train = stratified_subset(roles["train"], target, SMOKE_TRAIN, SMOKE_SEED)
        val = stratified_subset(roles["validation"], target, SMOKE_VALIDATION, SMOKE_SEED + 1)
        data = selection_data(X, target, roles, train, val)
        started = time.perf_counter()
        result = fit_validation(data, schema, protocol["rules"], SMOKE_TREES, SMOKE_THREADS)
        files = []
        for name, artifact in result["artifacts"].items():
            artifact.update(feature_version=VERSION, analytical_result=False, smoke_only=True,
                            split=slug, protocol_sha256=digest(output_path(root, "config", "protocol.json")))
            path = output_path(root, "models", "smoke", slug, name + ".joblib")
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(artifact, path, compress=3)
            restored = joblib.load(path)
            p = predict_artifact(restored, data.validation)
            np.testing.assert_allclose(p, result["probabilities"][name], rtol=1e-12, atol=1e-12)
            if any(v["convergence_warning"] for v in result["runtime"].values()):
                raise RuntimeError("Smoke Logistic convergence warning; do not mark preparation ready")
            files.append(path)
        prediction_path = output_path(root, "results", "smoke", slug, "validation_predictions.npz")
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(prediction_path, train_positions=train, validation_positions=val,
                            target=data.validation_target, **result["probabilities"])
        files.append(prediction_path)
        report_path = output_path(root, "logs", "smoke", slug + ".json")
        write_json(report_path, {"status": "SMOKE_ONLY_PASS", "analytical_result": False, "split": slug,
                                "test_performance_evaluated": False, "candidate_count": len(result["candidate_metrics"]),
                                "train_rows": len(train), "validation_rows": len(val),
                                "class_weights": result["class_weights"], "preprocessing_audit": result["preprocessing_audit"],
                                "engineering_validation_metrics": result["metrics"], "candidate_diagnostics": result["candidate_metrics"],
                                "selection_kernel_simulation_only": {"candidate_id": result["selected"]["candidate_id"],
                                                                       "constraints_met": result["constraints_met"], "constraints": result["constraints"]},
                                "runtime": result["runtime"], "seconds": time.perf_counter() - started})
        files.append(report_path)
        all_files.update({p.relative_to(root).as_posix(): digest(p) for p in files})
        records.append({"split": slug, "status": "PASS", "train_positions_sha256": position_hash(train),
                        "validation_positions_sha256": position_hash(val), "seconds": time.perf_counter() - started})
    write_json(summary_path, {"status": "ALL_SIX_SMOKE_SPLITS_PASS", "created_local": now(), "analytical_result": False,
                              "protocol_sha256": digest(output_path(root, "config", "protocol.json")), "records": records,
                              "artifact_sha256": all_files, "full_training_started": False, "test_performance_evaluated": False})
    print("All smoke branches passed. Full training, test evaluation, Bootstrap and SHAP remain unexecuted.", flush=True)


def verify(root):
    protocol = require_protocol(root)
    check_hashes(root, read_json(root / protocol["historical_manifest"]))
    summary = read_json(output_path(root, "logs", "smoke_complete.json"))
    if summary["protocol_sha256"] != digest(output_path(root, "config", "protocol.json")):
        raise ValueError("Smoke summary belongs to another protocol")
    if summary["analytical_result"] or summary["full_training_started"] or summary["test_performance_evaluated"]:
        raise ValueError("Preparation boundary was violated")
    if [r["split"] for r in summary["records"]] != [s["slug"] for s in protocol["splits"]]:
        raise ValueError("Smoke coverage is incomplete")
    check_hashes(root, summary["artifact_sha256"])
    _, X, target, _, assignments = load_data(root)
    for record in summary["records"]:
        slug = record["split"]
        roles = split_positions(assignments, slug)
        path = output_path(root, "results", "smoke", slug, "validation_predictions.npz")
        with np.load(path) as prediction:
            train, val = prediction["train_positions"], prediction["validation_positions"]
            selection_data(X, target, roles, train, val)
            np.testing.assert_array_equal(train, stratified_subset(roles["train"], target, SMOKE_TRAIN, SMOKE_SEED))
            np.testing.assert_array_equal(val, stratified_subset(roles["validation"], target, SMOKE_VALIDATION, SMOKE_SEED + 1))
            np.testing.assert_array_equal(prediction["target"], target.iloc[val])
            report = read_json(output_path(root, "logs", "smoke", slug + ".json"))
            for name in MODEL_NAMES:
                probabilities = check_probabilities(prediction[name], len(val))
                metrics = classification_metrics(prediction["target"], probabilities.argmax(axis=1))
                for metric, value in metrics.items():
                    np.testing.assert_allclose(value, report["engineering_validation_metrics"][name][metric], atol=1e-12, rtol=0)
                artifact = joblib.load(output_path(root, "models", "smoke", slug, name + ".joblib"))
                if artifact["analytical_result"] or not artifact["smoke_only"] or artifact["feature_version"] != VERSION:
                    raise ValueError("Artifact missing smoke-only identity")
                if artifact["split"] != slug or artifact["protocol_sha256"] != summary["protocol_sha256"]:
                    raise ValueError("Artifact belongs to a different split or protocol")
                bundle = artifact["bundle"]
                for column, levels in bundle["category_vocabulary"].items():
                    if levels != sorted(str(v) for v in X.iloc[train][column].unique()):
                        raise ValueError("Vocabulary is not fitted on the smoke training records")
                expected_weights = weights_from_training(target.iloc[train].to_numpy())
                actual_weights = artifact["estimator"].get_params().get("class_weight")
                if name in ("logistic_weighted", "lightgbm_weighted") and actual_weights != expected_weights:
                    raise ValueError("Weights differ from the smoke training class counts")
                if "preprocessor" in artifact:
                    median = artifact["preprocessor"].named_transformers_["numeric"].named_steps["imputer"].statistics_[0]
                    np.testing.assert_allclose(median, np.nanmedian(X.iloc[train].feature_speed_limit), rtol=0, atol=0)
                offset = np.unique(np.linspace(0, len(val) - 1, 61, dtype=int))
                np.testing.assert_allclose(predict_artifact(artifact, X.iloc[val[offset]]), probabilities[offset], atol=1e-12, rtol=1e-12)
            selected, eligible, constraints = local_selection(report["candidate_diagnostics"], report["engineering_validation_metrics"]["logistic_weighted"])
            if report["selection_kernel_simulation_only"] != {"candidate_id": selected["candidate_id"], "constraints_met": eligible, "constraints": constraints}:
                raise ValueError("Smoke local-selection diagnostics differ")
    done = {"status": PREPARATION_STATUS, "verified_local": now(), "smoke_splits": 6, "feature_count": 15,
            "historical_files_unchanged": len(read_json(root / protocol["historical_manifest"])),
            "protocol_sha256": digest(output_path(root, "config", "protocol.json")),
            "smoke_summary_sha256": digest(output_path(root, "logs", "smoke_complete.json")),
            "full_training_run": False, "test_evaluation_run": False, "bootstrap_run": False, "shap_run": False,
            "next_action": "Separate authorization and implementation freeze required for full-training/evaluation orchestration"}
    write_json(output_path(root, "logs", "preparation_complete.json"), done)
    print(PREPARATION_STATUS, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--stage", required=True, choices=("prepare", "smoke", "verify"))
    args = parser.parse_args(argv)
    root = args.project_root.expanduser().resolve()
    with execution_lock(root), threadpool_limits(limits=SMOKE_THREADS):
        {"prepare": prepare, "smoke": smoke, "verify": verify}[args.stage](root)


if __name__ == "__main__":
    main()
