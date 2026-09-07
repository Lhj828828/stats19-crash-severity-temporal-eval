"""Independent invariants for the post-review split-local correction."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "code"))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix

import random_reference_revision as revision


class RevisionTests(unittest.TestCase):
    def test_all_roles_disjoint(self):
        revision.assert_disjoint([0,1],[2],[3],[4,5])

    def test_validation_test_overlap_rejected(self):
        with self.assertRaises(AssertionError):
            revision.assert_disjoint([0],[1,2],[2],[3])

    def test_training_test_overlap_rejected(self):
        with self.assertRaises(AssertionError):
            revision.assert_disjoint([0,3],[1],[2],[3])

    def test_duplicate_record_rejected(self):
        with self.assertRaises(AssertionError):
            revision.assert_disjoint([0,0],[1],[2],[3])

    def test_invalid_role_rejected(self):
        frame = pd.DataFrame({"random_role_seed_1103":["train","validation","test","unknown"]})
        with self.assertRaises(AssertionError):
            revision.split_positions(frame,1103)

    def test_training_only_category_and_nan(self):
        schema = {"feature_columns":["feature_cat","feature_num"],"categorical_feature_columns":["feature_cat"],"numeric_feature_columns":["feature_num"]}
        train = pd.DataFrame({"feature_cat":["A","B","A"],"feature_num":[1.,2.,np.nan]})
        validation = pd.DataFrame({"feature_cat":["ONLY_VALIDATION"],"feature_num":[np.nan]})
        a,b,bundle,_ = revision.prepare_native_split(train,validation,schema)
        self.assertEqual(bundle["category_vocabulary"]["feature_cat"],["A","B"])
        self.assertEqual(b.feature_cat.iloc[0],"__UNSEEN__")
        self.assertTrue(np.isnan(b.feature_num.iloc[0]))
        self.assertTrue(np.isnan(a.feature_num.iloc[2]))

    def test_fallback_and_local_constraint(self):
        rows = [dict(candidate_id="C01",complexity_rank=1,macro_f1=.5,qwk=.1,fatal_recall=.2),
                dict(candidate_id="C02",complexity_rank=2,macro_f1=.4,qwk=.3,fatal_recall=.7)]
        rule = {"selection_rule":{"eligibility_constraints":{"minimum_qwk":.2,"minimum_fatal_recall":.6}}}
        selected,eligible = revision.select_candidate(rows,rule)
        self.assertEqual(selected["candidate_id"],"C02")
        self.assertTrue(eligible)
        rule["selection_rule"]["eligibility_constraints"]["minimum_fatal_recall"] = .9
        selected,eligible = revision.select_candidate(rows,rule)
        self.assertEqual(selected["candidate_id"],"C01")
        self.assertFalse(eligible)

    def test_paired_bootstrap_identity_and_stratum_counts(self):
        y = np.repeat([0,1,2], [30,15,5])
        pred = np.arange(len(y)) % 3
        output = revision.joint_bootstrap(y,np.column_stack([pred,pred]),np.random.default_rng(10),100)
        np.testing.assert_array_equal(output[:,0],output[:,1])
        np.testing.assert_array_equal(output.sum(axis=-1),np.broadcast_to([30,15,5],(100,2,3)))
        values = revision.metrics_from_confusions(output)
        for metric in values:
            np.testing.assert_array_equal(values[metric][:,0],values[metric][:,1])

    def test_confusion_kernel_matches_sklearn(self):
        rng = np.random.default_rng(9)
        y = np.repeat([0,1,2],100)
        predictions = rng.integers(0,3,size=(300,4))
        matrices = revision.confusion_from_vectors(y,predictions)
        values = revision.metrics_from_confusions(matrices)
        for m in range(4):
            np.testing.assert_array_equal(matrices[m],confusion_matrix(y,predictions[:,m]))
            independent = revision.classification_metrics(y,predictions[:,m])
            for key in values:
                self.assertAlmostEqual(values[key][m],independent[key],places=12)

    def test_bootstrap_reproducibility_and_independent_streams(self):
        y = np.repeat([0,1,2],20)
        preds = np.column_stack([np.arange(60)%3,np.arange(60)%2])
        a = revision.joint_bootstrap(y,preds,np.random.default_rng(10),30)
        b = revision.joint_bootstrap(y,preds,np.random.default_rng(10),30)
        np.testing.assert_array_equal(a,b)
        self.assertNotEqual(revision.stream_seed(10,"internal"),revision.stream_seed(10,"future"))

    def test_prediction_id_alignment_and_target_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.csv"
            pd.DataFrame({"id":["b","a","c"],"target_severity":[1,0,2],"predicted_severity":[0,1,2]}).to_csv(path,index=False)
            output = revision.aligned_predictions(path,pd.Series(["a","b","c"]),[0,1,2],"id")
            np.testing.assert_array_equal(output,[1,0,2])
            with self.assertRaises(AssertionError):
                revision.aligned_predictions(path,pd.Series(["a","b","c"]),[2,1,2],"id")

    def test_shap_rank_kernel(self):
        a = np.array([[1.,3.,2.,4.],[1.,1.,3.,2.]])
        b = np.array([[4.,2.,1.,3.],[1.,3.,2.,2.]])
        actual = revision.spearman_rows(a,b)
        expected = [spearmanr(x,y).statistic for x,y in zip(a,b)]
        np.testing.assert_allclose(actual,expected,atol=1e-12)

    def test_constant_or_invalid_bootstrap_rejected(self):
        with self.assertRaises(ValueError):
            revision.joint_bootstrap(np.array([0,1]),np.array([[1],[0]]),np.random.default_rng(1),10)
        with self.assertRaises(AssertionError):
            revision.interval(np.array([1,np.nan]))

    def test_cost_direction(self):
        self.assertIn("mean_asymmetric_cost",revision.LOWER)
        self.assertIn("ordinal_mae",revision.LOWER)
        self.assertNotIn("fatal_recall",revision.LOWER)

    def test_shap_cli_dispatches_to_analysis_function(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = ["revision", "--stage", "shap", "--dataset", "stats19", "--project-root", directory]
            with patch.object(sys, "argv", argv), patch.object(revision, "shap_analysis") as analysis:
                revision.main()
                analysis.assert_called_once_with(Path(directory).resolve(), "stats19")

    def test_test_label_perturbation_does_not_change_selection_fit(self):
        roles = ["train"]*90 + ["validation"]*30 + ["test"]*30 + ["locked_temporal_test"]*30
        assignments = pd.DataFrame({"random_role_seed_1103": roles})
        positions = revision.split_positions(assignments,1103)
        features = pd.DataFrame({"feature_num":np.random.default_rng(9).normal(size=180)})
        y = np.arange(180)%3
        changed = y.copy()
        changed[positions["test"]] = 2
        changed[positions["locked_temporal_test"]] = 0
        protocol = {"common_parameters":{"objective":"multiclass","num_class":3,"random_state":11,"verbosity":-1,"n_jobs":1},"early_stopping_rounds":3}
        candidate = {"candidate_id":"T01","complexity_rank":1,"parameters":{"num_leaves":3,"min_child_samples":5}}
        kwargs = dict(protocol=protocol,candidate=candidate,weights={0:1,1:1,2:1},
                      X_train=features.iloc[positions["train"]],X_validation=features.iloc[positions["validation"]],cap=5)
        first,_,p1 = revision.fit_candidate(**kwargs,y_train=y[positions["train"]],y_validation=y[positions["validation"]])
        second,_,p2 = revision.fit_candidate(**kwargs,y_train=changed[positions["train"]],y_validation=changed[positions["validation"]])
        np.testing.assert_array_equal(p1,p2)
        for key in first:
            if key != "fit_seconds":
                self.assertEqual(first[key],second[key])


if __name__ == "__main__":
    unittest.main(verbosity=2)
