"""Independent guards for the preparation-only 15-feature revision."""
import io
from contextlib import redirect_stderr
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import stats19_feature_revision as revision


def original_schema():
    features = list(revision.FEATURES)
    features[10:10] = list(revision.EXCLUDED)
    return {"feature_columns": features,
            "categorical_feature_columns": [c for c in features if c != "feature_speed_limit"],
            "numeric_feature_columns": ["feature_speed_limit"], "metadata_columns": [revision.ID, revision.YEAR]}


def fixture():
    n = 480
    rng = np.random.default_rng(22)
    frame = pd.DataFrame({c: rng.choice(["0:None", "1:Present", "9:Unknown"], n) for c in revision.FEATURES})
    frame["feature_speed_limit"] = rng.choice([20., 30., 40., 60., np.nan], n)
    y = np.arange(n) % 3
    assignment = pd.DataFrame({"temporal_role": ["train"] * 240 + ["validation"] * 120 + ["test"] * 120,
                               "random_role_seed_1103": ["train"] * 240 + ["validation"] * 120 + ["test"] * 60 + ["locked_temporal_test"] * 60})
    return frame, y, assignment


class FeatureRevisionTests(unittest.TestCase):
    def test_exact_15_feature_allowlist(self):
        schema = revision.revision_schema(original_schema())
        self.assertEqual(schema["feature_columns"], list(revision.FEATURES))
        self.assertEqual(len(schema["categorical_feature_columns"]), 14)
        self.assertEqual(schema["numeric_feature_columns"], ["feature_speed_limit"])

    def test_missing_reviewed_column_rejected(self):
        schema = original_schema()
        schema["feature_columns"][10] = "feature_wrong"
        with self.assertRaises(ValueError):
            revision.revision_schema(schema)

    def test_schema_reordering_rejected(self):
        schema = original_schema()
        schema["feature_columns"][0:2] = reversed(schema["feature_columns"][0:2])
        with self.assertRaises(ValueError):
            revision.revision_schema(schema)

    def test_schema_duplicate_rejected(self):
        schema = original_schema()
        schema["feature_columns"][1] = schema["feature_columns"][0]
        with self.assertRaises(ValueError):
            revision.revision_schema(schema)

    def test_numeric_misclassification_rejected(self):
        schema = original_schema()
        schema["numeric_feature_columns"] = ["feature_hour"]
        with self.assertRaises(ValueError):
            revision.revision_schema(schema)

    def test_metadata_injection_rejected(self):
        X, _, _ = fixture()
        X["meta_injury_based"] = 1
        with self.assertRaises(ValueError):
            revision.validate_features(X)

    def test_removed_feature_injection_rejected(self):
        for column in revision.EXCLUDED:
            X, _, _ = fixture()
            X[column] = "19:not injured"
            with self.assertRaises(ValueError):
                revision.validate_features(X)

    def test_target_injection_rejected(self):
        X, y, _ = fixture()
        X[revision.TARGET] = y
        with self.assertRaises(ValueError):
            revision.validate_features(X)

    def test_column_reordering_rejected(self):
        X, _, _ = fixture()
        with self.assertRaises(ValueError):
            revision.validate_features(X.iloc[:, ::-1])

    def test_temporal_and_random_roles(self):
        _, _, assignments = fixture()
        temporal = revision.split_positions(assignments, "temporal", strict=False)
        random = revision.split_positions(assignments, "random_seed_1103", strict=False)
        self.assertEqual([len(v) for v in temporal.values()], [240, 120, 120])
        self.assertEqual([len(v) for v in random.values()], [240, 120, 60, 60])

    def test_unknown_role_rejected(self):
        _, _, assignments = fixture()
        assignments.loc[0, "temporal_role"] = "other"
        with self.assertRaises(ValueError):
            revision.split_positions(assignments, "temporal", strict=False)

    def test_invalid_split_rejected(self):
        with self.assertRaises(ValueError):
            revision.split_positions(pd.DataFrame(), "random_seed_1234")

    def test_strict_counts_reject_toy_dataset(self):
        _, _, assignments = fixture()
        with self.assertRaises(ValueError):
            revision.split_positions(assignments, "temporal")

    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):
            revision.assert_disjoint({"train": [1, 2], "validation": [2, 3]})

    def test_duplicate_positions_rejected(self):
        with self.assertRaises(ValueError):
            revision.assert_disjoint({"train": [1, 1], "validation": [2]})

    def test_test_rows_cannot_be_supplied_to_selection(self):
        X, y, assignments = fixture()
        roles = revision.split_positions(assignments, "random_seed_1103", strict=False)
        with self.assertRaises(ValueError):
            revision.selection_data(X, y, roles, roles["train"], roles["internal"])
        with self.assertRaises(ValueError):
            revision.selection_data(X, y, roles, roles["future"], roles["validation"])

    def test_test_label_mutation_does_not_reach_fit_inputs(self):
        X, y, assignments = fixture()
        roles = revision.split_positions(assignments, "random_seed_1103", strict=False)
        first = revision.selection_data(X, y, roles)
        changed = y.copy()
        changed[roles["internal"]] = 2
        changed[roles["future"]] = 0
        second = revision.selection_data(X, changed, roles)
        np.testing.assert_array_equal(first.train_target, second.train_target)
        np.testing.assert_array_equal(first.validation_target, second.validation_target)
        pd.testing.assert_frame_equal(first.train, second.train)

    def test_smoke_sampling_is_repeatable_and_partition_local(self):
        _, y, assignments = fixture()
        roles = revision.split_positions(assignments, "temporal", strict=False)
        a = revision.stratified_subset(roles["train"], y, 90, 11)
        b = revision.stratified_subset(roles["train"], y, 90, 11)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(np.bincount(y[a]), [30, 30, 30])
        self.assertTrue(np.isin(a, roles["train"]).all())

    def test_smoke_cannot_accidentally_select_full_partition(self):
        with self.assertRaises(ValueError):
            revision.stratified_subset(np.arange(12), np.arange(12) % 3, 12, 1)

    def test_training_weights_use_only_training_labels(self):
        self.assertEqual(revision.weights_from_training([0, 0, 0, 0, 1, 2]), {0: .5, 1: 2., 2: 2.})
        with self.assertRaises(ValueError):
            revision.weights_from_training([0, 1])

    def test_training_vocabulary_and_nan_preserved(self):
        X, y, assignments = fixture()
        schema = revision.revision_schema(original_schema())
        roles = revision.split_positions(assignments, "temporal", strict=False)
        data = revision.selection_data(X, y, roles)
        data.validation.loc[:, "feature_weather_conditions"] = "only_validation"
        a, b, bundle, _ = revision.prepare_native_split(data.train, data.validation, schema)
        self.assertNotIn("only_validation", bundle["category_vocabulary"]["feature_weather_conditions"])
        self.assertTrue(b.feature_weather_conditions.eq("__UNSEEN__").all())
        self.assertEqual(a.feature_speed_limit.isna().sum(), data.train.feature_speed_limit.isna().sum())

    def test_local_constraints_override_better_ineligible_candidate(self):
        rows = [dict(candidate_id="C01", complexity_rank=1, macro_f1=.5, qwk=.1, fatal_recall=.1),
                dict(candidate_id="C02", complexity_rank=2, macro_f1=.4, qwk=.4, fatal_recall=.7)]
        chosen, eligible, constraints = revision.local_selection(rows, {"qwk": .3, "fatal_recall": .6})
        self.assertEqual(chosen["candidate_id"], "C02")
        self.assertTrue(eligible)
        self.assertAlmostEqual(constraints["minimum_qwk"], .29)
        self.assertAlmostEqual(constraints["minimum_fatal_recall"], .55)

    def test_fallback_uses_same_lexicographic_order(self):
        rows = [dict(candidate_id="C02", complexity_rank=1, macro_f1=.5, qwk=.1, fatal_recall=.1),
                dict(candidate_id="C01", complexity_rank=1, macro_f1=.5, qwk=.1, fatal_recall=.1)]
        chosen, eligible, _ = revision.local_selection(rows, {"qwk": .9, "fatal_recall": .9})
        self.assertEqual(chosen["candidate_id"], "C01")
        self.assertFalse(eligible)

    def test_nonfinite_candidate_rejected(self):
        with self.assertRaises(ValueError):
            revision.local_selection([dict(macro_f1=np.nan, qwk=.1, fatal_recall=.2)], {"qwk": .1, "fatal_recall": .2})

    def test_smoke_models_cannot_be_analytical(self):
        with self.assertRaises(ValueError):
            revision.require_analytical_artifact({"analytical_result": False, "feature_version": revision.VERSION})
        with self.assertRaises(ValueError):
            revision.require_analytical_artifact({"analytical_result": True, "feature_version": "D10_V1"})

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                revision.output_path(Path(directory), "models", "..", "legacy.joblib")
            with self.assertRaises(ValueError):
                revision.output_path(Path(directory), "data", "test")

    def test_source_hash_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "source.txt"
            path.write_text("original", encoding="utf-8")
            hashes = {"source.txt": revision.digest(path)}
            revision.check_hashes(root, hashes)
            path.write_text("modified", encoding="utf-8")
            with self.assertRaises(ValueError):
                revision.check_hashes(root, hashes)

    def test_full_run_cli_not_available(self):
        for args in (["--stage", "train"], ["--stage", "evaluate"], ["--all"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                revision.main(args)
            self.assertEqual(caught.exception.code, 2)

    def test_no_implicit_run_without_stage(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            revision.main([])

    def test_prepare_dispatch_does_not_fit(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(revision, "prepare") as prepare, patch.object(revision, "fit_validation") as fit:
            revision.main(["--project-root", directory, "--stage", "prepare"])
            prepare.assert_called_once_with(Path(directory).resolve())
            fit.assert_not_called()

    def test_smoke_dispatch_is_not_full_training(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(revision, "smoke") as smoke:
            revision.main(["--project-root", directory, "--stage", "smoke"])
            smoke.assert_called_once_with(Path(directory).resolve())

    def test_probability_guard(self):
        for value in (np.zeros((3, 3)), np.array([[np.nan, 0, 1]]), np.array([[-.1, .1, 1]])):
            with self.assertRaises((ValueError, AssertionError)):
                revision.check_probabilities(value, len(value))

    def test_real_toy_fit_roundtrip_and_metrics(self):
        X, y, assignments = fixture()
        schema = revision.revision_schema(original_schema())
        roles = revision.split_positions(assignments, "temporal", strict=False)
        data = revision.selection_data(X, y, roles)
        rules = revision.rules_from_parents(revision.ROOT)
        first = revision.fit_validation(data, schema, rules, tree_cap=5, threads=1)
        self.assertEqual(len(first["candidate_metrics"]), 6)
        self.assertEqual(set(first["artifacts"]), set(revision.MODEL_NAMES))
        weights = revision.weights_from_training(data.train_target)
        self.assertEqual(first["class_weights"], weights)
        preprocessor = first["artifacts"]["logistic_weighted"]["preprocessor"]
        median = preprocessor.named_transformers_["numeric"].named_steps["imputer"].statistics_[0]
        self.assertEqual(median, np.nanmedian(data.train.feature_speed_limit))
        with tempfile.TemporaryDirectory() as directory:
            for name, artifact in first["artifacts"].items():
                path = Path(directory) / (name + ".joblib")
                joblib.dump(artifact, path)
                restored = joblib.load(path)
                probability = revision.predict_artifact(restored, data.validation)
                np.testing.assert_allclose(probability, first["probabilities"][name], atol=1e-12)
                prediction = probability.argmax(axis=1)
                metrics = first["metrics"][name]
                self.assertAlmostEqual(metrics["macro_f1"], f1_score(data.validation_target, prediction, average="macro"), places=12)
                self.assertAlmostEqual(metrics["qwk"], cohen_kappa_score(data.validation_target, prediction, weights="quadratic"), places=12)
                self.assertAlmostEqual(metrics["fatal_recall"], recall_score(data.validation_target, prediction, average=None, labels=[0, 1, 2], zero_division=0)[2], places=12)
        changed = y.copy()
        changed[roles["future"]] = 2
        second = revision.fit_validation(revision.selection_data(X, changed, roles), schema, rules, tree_cap=5, threads=1)
        for name in revision.MODEL_NAMES:
            np.testing.assert_array_equal(first["probabilities"][name], second["probabilities"][name])
        self.assertEqual(first["selected"]["candidate_id"], second["selected"]["candidate_id"])

    def test_fit_rejects_unbounded_threads_and_tree_count(self):
        X, y, assignments = fixture()
        data = revision.selection_data(X, y, revision.split_positions(assignments, "temporal", strict=False))
        schema = revision.revision_schema(original_schema())
        for cap, threads in ((1201, 4), (30, -1), (30, 9)):
            with self.assertRaises(ValueError):
                revision.fit_validation(data, schema, {}, cap, threads)


if __name__ == "__main__":
    unittest.main(verbosity=2)
