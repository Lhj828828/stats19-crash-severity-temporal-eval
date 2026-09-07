"""Final routing, historical boundaries and no-training entry checks."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
import final_reproduction as runner
import final_result_catalog as catalog
import historical_d14_validation as history
import final_reproduction_binding as binding
import verify_final_reproduction as acceptance


class FinalAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_default_is_plan_without_file_writes_or_training(self):
        with patch.object(runner,"prepare") as prepare,patch.object(runner,"reproduce") as reproduce,redirect_stdout(io.StringIO()) as output:
            runner.main([])
        prepare.assert_not_called()
        reproduce.assert_not_called()
        self.assertIn("PLAN_ONLY",output.getvalue())

    def test_stats19_plan_excludes_retired_stages(self):
        plan=runner.stages("stats19")
        names=[n for n,_ in plan]
        self.assertEqual(len(names),22)
        self.assertEqual(len(names),len(set(names)))
        for name in ("ordered","d9s1","tree_sensitivity","d10b"):
            self.assertNotIn(name,str(plan).lower())
        self.assertLess(names.index("CORRECTED_main_freeze"),names.index("CORRECTED_main_train"))
        self.assertLess(names.index("CORRECTED_main_train"),names.index("CORRECTED_main_evaluate"))
        self.assertLess(names.index("CORRECTED_main_verify"),names.index("CORRECTED_exclude_train"))

    def test_cas_plan_finishes_with_corrected_verification(self):
        plan=runner.stages("cas")
        names=[n for n,_ in plan]
        self.assertEqual(names[-1],"CAS_RANDOM_CORRECTED_verify")
        self.assertIn("CAS_POSTHOC_FEATURE_ABLATION_SENSITIVITY",names)
        self.assertNotIn("CAS_PUBLIC_CLOSEOUT",names)
        self.assertNotIn("CAS_D8_SHAP_STABILITY",names)
        self.assertLess(names.index("CAS_RANDOM_CORRECTED_tune"),names.index("CAS_RANDOM_CORRECTED_evaluate"))

    def test_all_planned_scripts_exist(self):
        for dataset in ("stats19","cas"):
            for name,args in runner.stages(dataset):
                self.assertTrue((runner.ROOT/args[0]).is_file(),name)

    def test_workspace_cannot_overlap_source(self):
        source=self.root/"source"
        source.mkdir()
        for bad in (source,source/"child",self.root):
            with self.assertRaises(ValueError):
                runner.safe_workspace(bad,source)
        self.assertEqual(runner.safe_workspace(self.root/"separate",source),self.root/"separate")

    def test_reproduce_requires_explicit_workspace(self):
        with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()),patch.object(runner,"reproduce") as reproduce:
            with self.assertRaises(SystemExit):
                runner.main(["--stage","reproduce"])
        reproduce.assert_not_called()

    def test_stats19_copies_required_historical_templates(self):
        with patch.object(runner,"supplementary_inputs",return_value={}),patch.object(runner.legacy_stats19,"verify_source_stats19",return_value=[]):
            mapping=runner.stats19_input_map(self.root)
        for name in ("config/d8_baseline_protocol.json","config/d10_lightgbm_protocol.json","config/d14_shap_protocol.json"):
            self.assertEqual(mapping[name],self.root/name)

    def test_reproduction_binding_rejects_author_root(self):
        with self.assertRaises(FileNotFoundError):
            binding.require_workspace(self.root)

    def test_reproduction_binding_rejects_wrong_dataset(self):
        runner.write_json(self.root/"logs/final_workspace.json",{"version":runner.VERSION,"dataset":"cas","copied_sha256":{}})
        with self.assertRaises(ValueError):
            binding.require_workspace(self.root)

    def test_modified_workspace_input_rejected(self):
        target=self.root/"input.json"
        runner.write_json(target,{"x":1})
        marker={"version":runner.VERSION,"dataset":"stats19","copied_sha256":{"input.json":runner.digest(target)},
                "plan":[{"name":n,"arguments":list(a)} for n,a in runner.stages("stats19")]}
        runner.write_json(self.root/"logs/final_workspace.json",marker)
        runner.validate_workspace(self.root,"stats19")
        runner.write_json(target,{"x":2})
        with self.assertRaises(ValueError):
            runner.validate_workspace(self.root,"stats19")

    def test_changed_stage_plan_rejected(self):
        runner.write_json(self.root/"logs/final_workspace.json",{"version":runner.VERSION,"dataset":"stats19","copied_sha256":{},"plan":[]})
        with self.assertRaises(ValueError):
            runner.validate_workspace(self.root,"stats19")

    def test_catalog_has_current_sources_only(self):
        self.assertEqual(len(catalog.OUTPUTS),20)
        for item in catalog.OUTPUTS:
            if item.dataset=="stats19":
                self.assertTrue(item.source.startswith("results/stats19_feature_revision/"))
            self.assertNotIn("ordered",item.source.lower())
            self.assertNotIn("d10b",item.source.lower())

    def test_cas_legacy_random_lightgbm_is_not_exported(self):
        source=self.root/"input.csv"
        pd.DataFrame({"evaluation_group":["temporal_2025","random_1","random_1","random_1"],
                      "model":["lightgbm_weighted","lightgbm_weighted","logistic_weighted","dummy_most_frequent"],
                      "macro_f1":[.1,.2,.3,.4]}).to_csv(source,index=False)
        spec=catalog.Output("toy","cas","input.csv",("evaluation_group","model"),"toy","cas_random_baselines")
        frame=catalog.selected_frame(self.root,spec)
        self.assertEqual(set(frame.model),{"logistic_weighted","dummy_most_frequent"})
        self.assertEqual(len(frame),2)

    def test_legacy_h1_model_is_filtered(self):
        pd.DataFrame({"model":["lightgbm_legacy","lightgbm_revised","logistic_weighted"],"gap":[.1,.2,.3]}).to_csv(self.root/"input.csv",index=False)
        spec=catalog.Output("toy","cas","input.csv",("model",),"toy","exclude_legacy_model")
        frame=catalog.selected_frame(self.root,spec)
        self.assertNotIn("lightgbm_legacy",set(frame.model))

    def test_cas_h2_exports_temporal_only(self):
        spec=next(s for s in catalog.OUTPUTS if s.name=="cas_h2")
        self.assertEqual(spec.selection,"cas_temporal")
        source=self.root/spec.source
        source.parent.mkdir(parents=True)
        pd.DataFrame({"evaluation_group":["temporal_2025","random_seed_1103_internal_test"],
                      "metric":["macro_f1"]*2,"contrast":["lgbm_minus_logistic"]*2,
                      "point_difference":[.02,.03]}).to_csv(source,index=False)
        self.assertEqual(catalog.selected_frame(self.root,spec).evaluation_group.tolist(),["temporal_2025"])

    def test_v1_catalog_archive_preserves_bytes_and_old_exports(self):
        source=self.root/"source.csv"
        pd.DataFrame({"value":[1,2]}).to_csv(source,index=False)
        existing={"version":catalog.PREVIOUS_VERSION,"tables":[{"source":"source.csv","output":"source.csv",
                  "source_sha256":runner.digest(source),"output_sha256":runner.digest(source)}]}
        runner.write_json(self.root/catalog.CATALOG,existing)
        original=(self.root/catalog.CATALOG).read_bytes()
        amendment=catalog.archive_v1_catalog(self.root,existing)
        self.assertEqual((self.root/amendment["previous_catalog"]).read_bytes(),original)
        self.assertEqual(runner.digest(source),existing["tables"][0]["output_sha256"])
        self.assertFalse(amendment["scientific_artifacts_changed"])
        catalog.archive_v1_catalog(self.root,existing)

    def test_v1_catalog_migration_requires_explicit_request(self):
        runner.write_json(self.root/catalog.CATALOG,{"version":catalog.PREVIOUS_VERSION,"tables":[]})
        with patch.object(catalog,"verify_sources",return_value={}):
            with self.assertRaises(ValueError):
                catalog.assemble(self.root)

    def test_v1_archive_rejects_unknown_version(self):
        with self.assertRaises(ValueError):
            catalog.archive_v1_catalog(self.root,{"version":"UNKNOWN","tables":[]})

    def test_duplicate_keys_rejected(self):
        pd.DataFrame({"model":["x","x"],"f1":[.1,.2]}).to_csv(self.root/"input.csv",index=False)
        spec=catalog.Output("toy","stats19","input.csv",("model",),"toy")
        with self.assertRaises(ValueError):
            catalog.selected_frame(self.root,spec)

    def test_runtime_seconds_not_in_scientific_reference(self):
        pd.DataFrame({"model":["x"],"prediction_seconds":[20.],"macro_f1":[.3]}).to_csv(self.root/"input.csv",index=False)
        spec=catalog.Output("toy","stats19","input.csv",("model",),"toy")
        self.assertNotIn("prediction_seconds",catalog.selected_frame(self.root,spec))

    def test_filtered_numeric_looking_identifiers_remain_text(self):
        pd.DataFrame({"evaluation_group":["temporal_2025","random"],"model":["logistic_weighted"]*2,
                      "seed":["year_based","1103"],"cohort_years":["2025","2022;2023;2024"]}).to_csv(self.root/"input.csv",index=False)
        spec=catalog.Output("toy","cas","input.csv",("evaluation_group","model"),"toy","cas_temporal")
        selected=catalog.selected_frame(self.root,spec)
        selected.to_csv(self.root/"output.csv",index=False)
        pd.testing.assert_frame_equal(selected,catalog.read_table(self.root/"output.csv"))

    def test_result_comparison_reorders_keys_but_rejects_changed_numbers_or_reference(self):
        candidate=self.root/"candidate"
        reference=self.root/"reference"
        candidate.mkdir()
        reference.mkdir()
        frame=pd.DataFrame({"id":[2,1],"value":[.2,.1]})
        frame.to_csv(candidate/"input.csv",index=False)
        frame.sort_values("id").to_csv(reference/"expected.csv",index=False)
        spec=catalog.Output("toy","stats19","input.csv",("id",),"toy")
        runner.write_json(reference/catalog.CATALOG,{"version":catalog.VERSION,"tables":[{
            "name":"toy","output":"expected.csv","output_sha256":runner.digest(reference/"expected.csv")} ]})
        with patch.object(catalog,"OUTPUTS",(spec,)):
            self.assertEqual(catalog.compare(candidate,reference,"stats19")["status"],"PASS")
            frame.loc[0,"value"] = .21
            frame.to_csv(candidate/"input.csv",index=False)
            with self.assertRaises(AssertionError):
                catalog.compare(candidate,reference,"stats19")
            frame.to_csv(reference/"expected.csv",index=False)
            with self.assertRaises(ValueError):
                catalog.compare(candidate,reference,"stats19")

    def test_known_migration_is_explicit(self):
        name=next(iter(history.ALIASES))
        before,after=history.ALIASES[name]
        self.assertEqual(history.accept_digest(name,before,after),"DOCUMENTED_PROVENANCE_MIGRATION")
        self.assertEqual(history.accept_digest("same","sha","sha"),"EXACT")

    def test_unknown_migration_not_whitelisted(self):
        with self.assertRaises(ValueError):
            history.accept_digest("unknown","before","after")
        name=next(iter(history.ALIASES))
        with self.assertRaises(ValueError):
            history.accept_digest(name,history.ALIASES[name][0],"a"*64)

    def test_scientific_protocol_change_rejected(self):
        original={"upstream_sha256":{},"models":{"trees":1200}}
        rebuilt={"upstream_sha256":{},"models":{"trees":2000}}
        with self.assertRaises(ValueError):
            history.compare_protocols(original,rebuilt,{})

    def test_protocol_upstream_addition_rejected(self):
        with self.assertRaises(ValueError):
            history.compare_protocols({"upstream_sha256":{}},{"upstream_sha256":{"new":"x"}},{})

    def test_migrated_file_does_not_allow_other_setting_changes(self):
        name=next(iter(history.ALIASES))
        before,after=history.ALIASES[name]
        original={"upstream_sha256":{name:before},"target":"police_recorded"}
        rebuilt={"upstream_sha256":{name:after},"target":"changed"}
        with self.assertRaises(ValueError):
            history.compare_protocols(original,rebuilt,{})

    def test_known_migration_keeps_input_objects_unmodified(self):
        name=next(iter(history.ALIASES))
        before,after=history.ALIASES[name]
        original={"upstream_sha256":{name:before},"target":"police_recorded"}
        rebuilt={"upstream_sha256":{name:after},"target":"police_recorded"}
        saved=deepcopy(original)
        changes=history.compare_protocols(original,rebuilt,{})
        self.assertEqual(original,saved)
        self.assertEqual(len(changes),1)

    def test_acceptance_rejects_unfinished_reconstruction(self):
        runner.write_json(self.root/"logs/final_reproduction_state.log",{"status":"RUNNING","completed":[]})
        with patch.object(runner,"validate_workspace",return_value={}):
            with self.assertRaises(ValueError):
                acceptance.check_completed_workspace(self.root,"stats19")

    def test_nested_seal_detects_changed_artifact(self):
        artifact=self.root/"values.json"
        runner.write_json(artifact,{"value":1})
        runner.write_json(self.root/"seal.json",{"files_sha256":{"values.json":runner.digest(artifact)}})
        runner.write_json(self.root/"complete.json",{"files_sha256":{"seal.json":runner.digest(self.root/"seal.json")}})
        acceptance.verify_seal_tree(self.root,"complete.json")
        runner.write_json(artifact,{"value":2})
        with self.assertRaises(ValueError):
            acceptance.verify_seal_tree(self.root,"complete.json")

    def test_acceptance_rejects_blank_figure(self):
        folder=self.root/"figures/stats19_feature_revision/full"
        folder.mkdir(parents=True)
        acceptance.Image.new("RGB",(400,300),"white").save(folder/"blank.png")
        with self.assertRaises(ValueError):
            acceptance.check_figure_files(self.root,"stats19")

    def test_acceptance_rejects_unreviewed_executable_change(self):
        for name in ("kernel.py", "code/final_result_catalog.py"):
            runner.write_json(self.root/name,{"version":2})
            with patch.object(acceptance,"ROOT",self.root):
                with self.assertRaises(ValueError):
                    acceptance.verify_source_snapshot(self.root,{"copied_sha256":{name:"old_hash"}})


if __name__=="__main__":
    unittest.main()
