"""Execution gates, pairing and numerical checks for the full revision."""
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import stats19_feature_revision_full as run
import stats19_feature_revision as prep


def fixture():
    rng = np.random.default_rng(19)
    n = 480
    X = pd.DataFrame({c: rng.choice(["0:None", "1:Present", "9:Unknown"], n) for c in prep.FEATURES})
    X["feature_speed_limit"] = rng.choice([20., 30., 40., 60., np.nan], n)
    y = pd.Series(np.arange(n) % 3, dtype="int8")
    schema = {"feature_columns": list(prep.FEATURES),
              "categorical_feature_columns": [c for c in prep.FEATURES if c != "feature_speed_limit"],
              "numeric_feature_columns": ["feature_speed_limit"]}
    assignments = pd.DataFrame({"temporal_role": ["train"]*240 + ["validation"]*120 + ["test"]*120,
                               "random_role_seed_1103": ["train"]*240 + ["validation"]*120 + ["test"]*60 + ["locked_temporal_test"]*60})
    metadata = pd.DataFrame({prep.ID: [f"toy-{i}" for i in range(n)]})
    return schema, X, y, metadata, assignments


class FullRevisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        prep.write_json(run.protocol_path(self.root), {"test_only": True})

    def test_eleven_explicit_groups(self):
        groups = run.groups()
        self.assertEqual(len(groups), 11)
        self.assertEqual(len({r["group"] for r in groups}), 11)
        self.assertEqual(sum(r["rows"] for r in groups), 1087602)
        self.assertEqual(groups[0]["split"], "temporal")
        self.assertEqual(sum(r["role"] == "future" for r in groups), 6)

    def test_safe_output_paths(self):
        with self.assertRaises(ValueError):
            run.path(self.root, "results", "../../../../escape")

    def test_completed_seal_roundtrip(self):
        file = run.write_csv(run.path(self.root, "results", "small.csv"), [{"a": 1}])
        run.seal(self.root, "test", [file])
        self.assertEqual(run.checked_seal(self.root, "test")["status"], "COMPLETE")

    def test_checkpoint_tamper_rejected(self):
        file = run.write_csv(run.path(self.root, "results", "small.csv"), [{"a": 1}])
        run.seal(self.root, "test", [file])
        run.write_csv(file, [{"a": 2}])
        with self.assertRaises(ValueError):
            run.checked_seal(self.root, "test")

    def test_protocol_tamper_rejects_checkpoint(self):
        run.seal(self.root, "test", [])
        prep.write_json(run.protocol_path(self.root), {"different": True})
        with self.assertRaises(ValueError):
            run.checked_seal(self.root, "test")

    def test_completed_checkpoint_not_overwritten(self):
        run.seal(self.root, "test", [])
        with self.assertRaises(FileExistsError):
            run.seal(self.root, "test", [])

    def test_missing_model_sets_block_evaluation(self):
        with patch.object(run, "require_protocol"), patch.object(prep, "load_data") as data:
            with self.assertRaises(FileNotFoundError):
                run.evaluate(self.root)
            data.assert_not_called()

    def test_incomplete_model_gate_rejected(self):
        run.seal(self.root, "training_complete", [], splits=["temporal"], test_performance_evaluated=False)
        with self.assertRaises(ValueError):
            run.require_all_models(self.root)

    def test_test_evaluated_training_seal_rejected(self):
        run.seal(self.root, "training_complete", [], splits=list(run.SLUGS), test_performance_evaluated=True)
        with self.assertRaises(ValueError):
            run.require_all_models(self.root)

    def test_smoke_artifact_rejected(self):
        destination = run.path(self.root, "models", "temporal", "dummy_most_frequent.joblib")
        destination.parent.mkdir(parents=True)
        joblib.dump({"analytical_result": False, "feature_version": prep.VERSION}, destination)
        with self.assertRaises(ValueError):
            run.load_model(self.root, "temporal", "dummy_most_frequent")

    def test_prediction_ids_and_positions_aligned(self):
        pos, y, ids = np.arange(6), np.arange(6) % 3, np.array(list("abcdef"))
        arrays = {name: np.eye(3)[y] for name in run.MODELS}
        run.save_npz(run.prediction_path(self.root, "toy"), positions=pos, target=y, ids=ids, **arrays)
        run.aligned_predictions(self.root, "toy", pos, y, ids)
        with self.assertRaises(AssertionError):
            run.aligned_predictions(self.root, "toy", pos[::-1], y, ids)
        with self.assertRaises(AssertionError):
            run.aligned_predictions(self.root, "toy", pos, y, ids[::-1])

    def test_metrics_against_sklearn(self):
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        pred = np.array([0, 1, 2, 0, 1, 2, 0, 0, 2])
        result, matrix = run.full_metrics(y, np.eye(3)[pred])
        self.assertAlmostEqual(result["macro_f1"], f1_score(y, pred, average="macro"))
        self.assertAlmostEqual(result["qwk"], cohen_kappa_score(y, pred, weights="quadratic"))
        np.testing.assert_allclose([result[f"class{c}_recall"] for c in range(3)], recall_score(y, pred, average=None))
        self.assertEqual(matrix[2, 0], 2)
        self.assertAlmostEqual(result["mean_asymmetric_cost"], np.array([[0,1,2],[2,0,1],[5,3,0]])[y, pred].mean())

    def test_bootstrap_preserves_pairing_and_true_counts(self):
        y = np.repeat([0, 1, 2], [30, 20, 10])
        p = np.column_stack([y, y])
        draws = run.joint_bootstrap(y, p, np.random.default_rng(9), iterations=2000)
        self.assertEqual(draws.shape, (2000, 2, 3, 3))
        np.testing.assert_array_equal(draws[:, 0], draws[:, 1])
        np.testing.assert_array_equal(draws.sum(axis=-1), np.broadcast_to([30,20,10], (2000,2,3)))

    def test_joint_pattern_bootstrap_matches_record_sampling_expectation(self):
        y = np.repeat([0, 1, 2], 20)
        p = np.column_stack([np.arange(60) % 3, (np.arange(60) + 1) % 3])
        draws = run.joint_bootstrap(y, p, np.random.default_rng(19), iterations=20000)
        expected = run.confusion_from_vectors(y, p)
        np.testing.assert_allclose(draws.mean(axis=0), expected, atol=.06, rtol=0)

    def test_internal_future_use_distinct_streams(self):
        a = run.stream_seed(run.BOOTSTRAP_SEED, run.VERSION + ":random_seed_1103_internal")
        b = run.stream_seed(run.BOOTSTRAP_SEED, run.VERSION + ":random_seed_1103_2024")
        self.assertNotEqual(a, b)

    def test_h1_sign_conventions(self):
        self.assertAlmostEqual(run.oriented_gap(.6, .4, "macro_f1"), .2)
        self.assertAlmostEqual(run.oriented_gap(.6, .4, "mean_asymmetric_cost"), -.2)
        self.assertAlmostEqual(run.oriented_gap(.6, .4, "ordinal_mae"), -.2)

    def test_descriptive_h2_does_not_assert_noninferiority(self):
        rows = [{"group": "temporal_2024", "reference": "logistic_weighted", "metric": metric,
                 "delta": .02, "ci_lower": -.02, "ci_upper": .05} for metric in
                ("macro_f1", "qwk", "fatal_recall", "mean_asymmetric_cost")]
        rows[0]["ci_lower"] = .01
        result = run.descriptive_h2(rows)
        self.assertTrue(result["joint_descriptive_gate"])
        self.assertTrue(result["not_noninferiority_evidence"])
        rows[2]["ci_upper"] = -.01
        self.assertFalse(run.descriptive_h2(rows)["joint_descriptive_gate"])

    def test_shap_scopes_are_equal_output_average(self):
        values = np.arange(90).reshape(2, 15, 3)
        np.testing.assert_array_equal(run.scope_values(values, "overall"), values.mean(axis=-1))
        np.testing.assert_array_equal(run.scope_values(values, "Fatal"), values[:, :, 2])
        with self.assertRaises(ValueError):
            run.scope_values(np.ones((2, 17, 3)), "overall")

    def test_rank_correlation_uses_15_features(self):
        a = np.arange(15, dtype=float)
        b = np.roll(a, 4)
        self.assertAlmostEqual(run.spearman_rows(a[None], b[None])[0], spearmanr(a, b).statistic)

    def test_frozen_fit_to_serialization_evaluation_and_bootstrap_toy(self):
        dataset = fixture()
        schema, X, y, metadata, assignments = dataset
        rules = prep.rules_from_parents(prep.ROOT)
        prep.write_json(prep.output_path(self.root, "config", "protocol.json"), {"rules": rules})
        slugs = ("temporal", "random_seed_1103")
        test_groups = [{"group": "temporal_2024", "split": "temporal", "role": "future", "rows": 120},
                       {"group": "random_seed_1103_internal", "split": "random_seed_1103", "role": "internal", "rows": 60},
                       {"group": "random_seed_1103_2024", "split": "random_seed_1103", "role": "future", "rows": 60}]
        original_split, original_fit = prep.split_positions, prep.fit_validation
        with ExitStack() as stack:
            stack.enter_context(patch.object(run, "require_protocol"))
            stack.enter_context(patch.object(run, "SLUGS", slugs))
            stack.enter_context(patch.object(run, "groups", return_value=test_groups))
            stack.enter_context(patch.object(prep, "SEEDS", (1103,)))
            stack.enter_context(patch.object(prep, "load_data", return_value=dataset))
            stack.enter_context(patch.object(prep, "split_positions", side_effect=lambda frame, slug: original_split(frame, slug, strict=False)))
            fitted = stack.enter_context(patch.object(prep, "fit_validation", side_effect=lambda d,s,r,cap,t: original_fit(d,s,r,3,2)))
            run.train(self.root)
            self.assertEqual(fitted.call_count, 2)
            run.train(self.root)
            self.assertEqual(fitted.call_count, 2)
            run.evaluate(self.root)
            run.bootstrap(self.root)
            self.assertEqual(len(pd.read_csv(run.path(self.root, "results", "test_metrics.csv"))), 12)
            self.assertEqual(len(pd.read_csv(run.path(self.root, "results", "h1_gaps.csv"))), 28)
            artifact = run.load_model(self.root, "temporal", "lightgbm_weighted")
            values, expected, audit = run.explain(X, np.arange(360, 366), artifact["bundle"], artifact)
            self.assertEqual(values.shape, (6, 15, 3))
            self.assertLess(audit["max_abs_additivity_error"], 1e-6)
            self.assertEqual(expected.shape, (3,))


if __name__ == "__main__":
    unittest.main()
