"""Independent guards for corrected fixed-parameter exclusion sensitivity."""
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import stats19_exclude2020_revision as run
import stats19_feature_revision as prep
import stats19_feature_revision_full as full


def fixture():
    rng = np.random.default_rng(21)
    years = np.repeat(np.arange(2018, 2025), 60)
    n = len(years)
    X = pd.DataFrame({c: rng.choice(["0:no", "1:yes", "9:unknown"], n) for c in prep.FEATURES})
    X["feature_speed_limit"] = rng.choice([20.,30.,40.,60.,np.nan], n)
    X.loc[years == 2020, "feature_day_of_week"] = "2020-only"
    X.loc[years == 2023, "feature_month"] = "validation-only"
    y = pd.Series(np.arange(n) % 3, dtype="int8")
    metadata = pd.DataFrame({prep.ID: [f"toy-{i}" for i in range(n)], prep.YEAR: years})
    assignments = pd.DataFrame({prep.YEAR: years, "temporal_role": np.where(years < 2023, "train", np.where(years == 2023, "validation", "test"))})
    schema = {"feature_columns": list(prep.FEATURES), "categorical_feature_columns": [c for c in prep.FEATURES if c != "feature_speed_limit"],
              "numeric_feature_columns": ["feature_speed_limit"]}
    parameters = {"logistic_weighted": LogisticRegression(max_iter=500, random_state=20260828).get_params(),
                  "lightgbm_weighted": lgb.LGBMClassifier(n_estimators=3, min_child_samples=5, verbosity=-1,
                      n_jobs=2, random_state=20260830, deterministic=True, force_col_wise=True).get_params()}
    for model in parameters.values():
        model.pop("class_weight")
    return (schema, X, y, metadata, assignments), parameters


class ExclusionRevisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        prep.write_json(run.protocol_path(self.root), {"toy_test_only": True})

    def test_exact_years_and_unchanged_evaluation_membership(self):
        dataset, _ = fixture()
        roles = run.positions(dataset[-1], strict=False)
        original = prep.split_positions(dataset[-1], "temporal", strict=False)
        self.assertEqual([len(v) for v in roles.values()], [240,60,60,60])
        np.testing.assert_array_equal(roles["validation"], original["validation"])
        np.testing.assert_array_equal(roles["future"], original["future"])
        self.assertEqual(set(dataset[-1].iloc[roles["train"]][prep.YEAR]), set(run.YEARS))

    def test_strict_counts_reject_toy(self):
        with self.assertRaises(ValueError):
            run.positions(fixture()[0][-1])

    def test_incorrect_validation_year_rejected(self):
        assignments = fixture()[0][-1]
        assignments.loc[assignments.temporal_role.eq("validation"), prep.YEAR] = 2022
        with self.assertRaises(ValueError):
            run.positions(assignments, strict=False)

    def test_missing_excluded_year_rejected(self):
        assignments = fixture()[0][-1]
        assignments.loc[assignments[prep.YEAR].eq(2020), prep.YEAR] = 2019
        with self.assertRaises(ValueError):
            run.positions(assignments, strict=False)

    def test_held_out_rows_cannot_be_passed_to_fit(self):
        (schema, X, y, metadata, assignments), _ = fixture()
        roles = run.positions(assignments, strict=False)
        for forbidden in ("future", "validation", "removed_2020"):
            with self.assertRaises(ValueError):
                prep.selection_data(X, y, roles, roles[forbidden], roles["validation"])

    def test_exact_two_comparators(self):
        self.assertEqual(run.MODELS, ("logistic_weighted", "lightgbm_weighted"))
        self.assertEqual(len(run.JOINT_ORDER), 4)

    def test_training_class_weights_recomputed(self):
        y = np.repeat([0,1,2], [60,30,10])
        weights = prep.weights_from_training(y)
        self.assertEqual(weights, {0:100/180,1:100/90,2:100/30})

    def test_seal_integrity(self):
        file = full.write_csv(run.path(self.root, "results", "toy.csv"), [{"x":1}])
        run.seal(self.root, "toy", [file])
        self.assertEqual(run.checked_seal(self.root, "toy")["status"], "COMPLETE")
        full.write_csv(file, [{"x":2}])
        with self.assertRaises(ValueError):
            run.checked_seal(self.root, "toy")

    def test_duplicate_completed_checkpoint_rejected(self):
        run.seal(self.root, "toy", [])
        with self.assertRaises(FileExistsError):
            run.seal(self.root, "toy", [])

    def test_protocol_mismatch_rejects_checkpoint(self):
        run.seal(self.root, "toy", [])
        prep.write_json(run.protocol_path(self.root), {"different":True})
        with self.assertRaises(ValueError):
            run.checked_seal(self.root, "toy")

    def test_missing_training_seal_blocks_data_loading(self):
        with patch.object(run, "require_protocol"), patch.object(prep, "load_data") as data:
            with self.assertRaises(FileNotFoundError):
                run.evaluate(self.root)
            data.assert_not_called()

    def test_only_one_sealed_model_rejected(self):
        run.seal(self.root, "training_complete", [], models=["lightgbm_weighted"], test_performance_evaluated=False, validation_used_for_selection=False)
        with self.assertRaises(ValueError):
            run.require_test_gate(self.root)

    def test_validation_selection_flag_rejected(self):
        run.seal(self.root, "training_complete", [], models=list(run.MODELS), test_performance_evaluated=False, validation_used_for_selection=True)
        with self.assertRaises(ValueError):
            run.require_test_gate(self.root)

    def test_main_artifact_cannot_masquerade_as_sensitivity(self):
        destination = run.model_path(self.root, "lightgbm_weighted")
        destination.parent.mkdir(parents=True)
        joblib.dump({"analytical_result":True,"feature_version":prep.VERSION,"model_name":"lightgbm_weighted","smoke_only":False}, destination)
        with self.assertRaises(ValueError):
            run.load_model(self.root, "lightgbm_weighted")

    def test_safe_output_directory(self):
        with self.assertRaises(ValueError):
            run.path(self.root, "results", "../../../../escape")

    def test_membership_hash_change_rejected(self):
        roles = {"train":np.array([1,2]),"future":np.array([3,4])}
        protocol = {"role_counts_and_hashes": {k:{"rows":len(v),"sha256":prep.position_hash(v)} for k,v in roles.items()}}
        run.check_roles(protocol, roles)
        roles["train"] = np.array([2,1])
        with self.assertRaises(ValueError):
            run.check_roles(protocol, roles)

    def test_shared_draws_keep_identical_predictions_identical(self):
        y = np.repeat([0,1,2], [30,20,10])
        pred = np.arange(60) % 3
        points, draws = run.bootstrap_arrays(self.root, y, np.column_stack([pred]*4))
        rows = run.comparison_rows(points, draws)
        self.assertEqual(len(rows), 35)
        for row in rows:
            self.assertEqual(row["delta"], 0)
            self.assertEqual(row["ci_lower"], 0)
            self.assertEqual(row["ci_upper"], 0)

    def test_change_in_contrast_formula(self):
        point = {"macro_f1":np.array([.3,.4,.35,.46])}
        draws = {"macro_f1":np.broadcast_to(point["macro_f1"],(2000,4))}
        rows = run.comparison_rows(point, draws)
        change = next(r for r in rows if r["contrast"] == "change_in_model_contrast")
        self.assertAlmostEqual(change["delta"], .01)
        self.assertAlmostEqual(change["ci_lower"], .01)

    def test_cost_difference_not_sign_flipped(self):
        point = {"mean_asymmetric_cost":np.array([.7,.6,.75,.62])}
        draws = {"mean_asymmetric_cost":np.broadcast_to(point["mean_asymmetric_cost"],(2000,4))}
        row = next(r for r in run.comparison_rows(point,draws) if r["contrast"] == "exclude2020_LightGBM_minus_Logistic")
        self.assertAlmostEqual(row["delta"], -.13)

    def test_h2_negative_fatal_interval_fails_even_with_macro_gain(self):
        rows = [{"contrast":"exclude2020_LightGBM_minus_Logistic","metric":metric,"delta":.02,"ci_lower":.01,"ci_upper":.03}
                for metric in ("macro_f1","qwk","fatal_recall","mean_asymmetric_cost")]
        rows[2].update(delta=-.1,ci_lower=-.15,ci_upper=-.05)
        rows[3].update(delta=-.1,ci_lower=-.15,ci_upper=-.05)
        result = run.h2_decision(rows,"exclude2020_LightGBM_minus_Logistic")
        self.assertFalse(result["joint_descriptive_gate"])
        self.assertTrue(result["not_noninferiority_evidence"])

    def test_fixed_fit_and_entire_output_verification_toy(self):
        dataset, parameters = fixture()
        schema, X, y, metadata, assignments = dataset
        roles = run.positions(assignments, strict=False)
        protocol = {"model_parameters":parameters,"protected_sha256":{},"main_candidate":{"id":"TOY","iterations":3},
                    "role_counts_and_hashes":{k:{"rows":len(v),"sha256":prep.position_hash(v)} for k,v in roles.items()}}
        original_positions = run.positions
        def reference(root, key, pos, target, ids):
            return {name:np.full((len(pos),3),1/3) for name in run.MODELS}
        with ExitStack() as stack:
            stack.enter_context(patch.object(run,"require_protocol",return_value=protocol))
            stack.enter_context(patch.object(prep,"load_data",return_value=dataset))
            stack.enter_context(patch.object(run,"positions",side_effect=lambda frame:original_positions(frame,strict=False)))
            stack.enter_context(patch.object(full,"aligned_predictions",side_effect=reference))
            fitter = stack.enter_context(patch.object(run,"fit_models",wraps=run.fit_models))
            original_lgb_fit = lgb.LGBMClassifier.fit
            kwargs_seen = []
            def fit_spy(estimator, *args, **kwargs):
                kwargs_seen.append(kwargs)
                return original_lgb_fit(estimator,*args,**kwargs)
            stack.enter_context(patch.object(lgb.LGBMClassifier,"fit",new=fit_spy))
            run.train(self.root)
            run.train(self.root)
            self.assertEqual(fitter.call_count,1)
            self.assertEqual(len(kwargs_seen),1)
            self.assertEqual(set(kwargs_seen[0]),{"categorical_feature"})
            artifact = run.load_model(self.root,"lightgbm_weighted")
            self.assertNotIn("2020-only",artifact["bundle"]["category_vocabulary"]["feature_day_of_week"])
            self.assertNotIn("validation-only",artifact["bundle"]["category_vocabulary"]["feature_month"])
            self.assertEqual(artifact["estimator"].get_params()["n_estimators"],3)
            run.evaluate(self.root)
            run.bootstrap(self.root)
            run.verify(self.root)
            done = run.checked_seal(self.root,"verification_complete")
            self.assertTrue(done["full_serialized_validation_and_test_predictions_checked"])
            self.assertFalse(done["retuned"])


if __name__ == "__main__":
    unittest.main()
