"""Validate a normalized historical snapshot without rewriting its freeze.

This is NOT a replacement claim that the two original exact-equality tests
pass. It separately checks known provenance migrations and scientific outputs.
Fresh corrected runs must use their own execution-bound validators instead.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

import stats19_feature_revision as prep
import d14_exclude2020_sensitivity as exclusion
import d14_closeout as closeout

ROOT = prep.ROOT
VERSION = "HISTORICAL_D14_NORMALIZED_SNAPSHOT_VALIDATION_V1"
OUTLINE = "AUTHOR_SIDE_WORKING_OUTLINE_NOT_DISTRIBUTED"
EVIDENCE = "config/d14_exclude2020_planning_evidence.json"
ALIASES = {
    "config/d14_exclude2020_protocol.json": (
        "76f81303b9caee50789d04551c6b09abd0b64830c8ed52f4e84cccc72743233b",
        "f31f9b4436a151bd68a92785368409c3d00999d19787c347915eea9a66d9f378"),
    "config/d14_exclude2020_protocol_v2.json": (
        "2db671ca62f91fb0b061f06c471ef0d625f8770563d12c9f1fe3374d81ee6cba",
        "3addb070f4771fd64ee47309d7f58743a2ae1e4bf8ceafdc17ea8de97a79a88a"),
    "code/d14_exclude2020_sensitivity.py": (
        "d4dcb3d0a79d71add182f8e96a4c81762ac7df8b1749f4ff213bab809d68e6d3",
        "53fcf70c017de5cf1e5e7cf8f6a07e98452732b37895f5a564259bb8384e95e4"),
}
NORMALIZATION_COMMITS = ("b365744cfe6f1a5273abf4aa4af7339d67583606", "5238d0d")


def accept_digest(name, before, observed):
    if before == observed:
        return "EXACT"
    if name not in ALIASES or (before,observed) != ALIASES[name]:
        raise ValueError("Unreviewed historical file change: " + name)
    return "DOCUMENTED_PROVENANCE_MIGRATION"


def compare_protocols(historical, rebuilt, evidence):
    old, new = deepcopy(historical), deepcopy(rebuilt)
    changes = []
    before, after = old.pop("upstream_sha256"), new.pop("upstream_sha256")
    if OUTLINE in before:
        sha = before.pop(OUTLINE)
        if (sha != "bbbf6b60a799a23c3afab0b856f215b34d27c3f726197efbe1d997de8394c77b"
                or evidence["original_author_file"]["sha256"] != sha
                or evidence["original_author_file"]["redistributed"] is not False):
            raise ValueError("Original outline provenance identity differs")
        planning = new.pop("planning_evidence")
        if planning != {"record":EVIDENCE,"sha256":after[EVIDENCE],
                        "boundary":"Local working-outline provenance only; not an external preregistration or an independently timestamped record."}:
            raise ValueError("Planning provenance boundary changed")
        after.pop(EVIDENCE)
        changes.append({"field":OUTLINE,"status":"IDENTITY_RECORDED_ORIGINAL_BYTES_NOT_DISTRIBUTED"})
    if set(before) != set(after):
        raise ValueError("Unexpected historical upstream membership change")
    for name in before:
        status = accept_digest(name,before[name],after[name])
        if status != "EXACT":
            changes.append({"field":name,"status":status,"before":before[name],"after":after[name]})
    if old.get("revision") != new.get("revision"):
        revision = old["revision"]
        status = accept_digest(revision["supersedes_protocol"],revision["supersedes_protocol_sha256"],new["revision"]["supersedes_protocol_sha256"])
        if status == "EXACT":
            raise ValueError("Non-hash revision difference")
        revision["supersedes_protocol_sha256"] = new["revision"]["supersedes_protocol_sha256"]
    if old != new:
        raise ValueError("Historical scientific settings or interpretation changed")
    return changes


def check_manifest(root, file):
    records=[]
    for row in pd.read_csv(file).itertuples(index=False):
        target=(root/row.relative_path).resolve()
        if not target.is_relative_to(root.resolve()):
            raise ValueError("Historical manifest escapes root")
        observed=prep.digest(target)
        status=accept_digest(row.relative_path,row.sha256,observed)
        if status == "EXACT" and target.stat().st_size != int(row.bytes):
            raise ValueError("Historical size mismatch")
        records.append({"file":row.relative_path,"status":status,"archived_sha256":row.sha256,"current_sha256":observed})
    return records


def compare_frame(expected, actual):
    pd.testing.assert_frame_equal(expected.reset_index(drop=True),actual.reset_index(drop=True),
                                  check_dtype=False,check_exact=False,atol=1e-12,rtol=1e-12)


def verify_exclusion_results():
    targets, frames = {}, {}
    validation=pd.read_csv(exclusion.VALIDATION_METRICS_FILE)
    test=pd.read_csv(exclusion.TEST_METRICS_FILE)
    if len(validation)!=2 or len(test)!=2:
        raise ValueError("Historical model count differs")
    for name in ("logistic_weighted","lightgbm_weighted"):
        for role in ("validation","test"):
            frame=pd.read_csv(exclusion.PREDICTION_DIR/f"{name}__{role}.csv.gz",dtype={"meta_collision_index":"string"})
            expected=104258 if role=="validation" else 100927
            if len(frame)!=expected or frame.meta_collision_index.nunique()!=expected:
                raise ValueError("Historical prediction membership differs")
            probabilities=frame[["prob_slight","prob_serious","prob_fatal"]].to_numpy()
            np.testing.assert_array_equal(probabilities.argmax(axis=1),frame.predicted_severity)
            values=exclusion.calculate_metrics(frame.target_severity.to_numpy(),frame.predicted_severity.to_numpy(),probabilities)
            row=(validation if role=="validation" else test).loc[lambda x:x.model.eq(name)].iloc[0]
            for metric,value in values.items():
                np.testing.assert_allclose(value,row[metric],atol=1e-12,rtol=0)
            if role=="test":
                frames["exclude2020_"+name]=frame
    for name,file in exclusion.MAIN_PREDICTION_FILES.items():
        frames[name]=pd.read_csv(file,dtype={"meta_collision_index":"string"})
    reference=frames[exclusion.MODEL_ORDER[0]]
    for frame in frames.values():
        np.testing.assert_array_equal(frame.meta_collision_index,reference.meta_collision_index)
        np.testing.assert_array_equal(frame.target_severity,reference.target_severity)
    pred=np.column_stack([frames[n].predicted_severity for n in exclusion.MODEL_ORDER])
    point_confusions=exclusion.confusion_from_vectors(reference.target_severity.to_numpy(),pred)
    with np.load(exclusion.BOOTSTRAP_FILE) as saved:
        np.testing.assert_array_equal(saved["point_confusions"],point_confusions)
        if saved["bootstrap_confusions"].shape!=(2000,4,3,3):
            raise ValueError("Historical Bootstrap dimensions differ")
        point=exclusion.metrics_from_confusions(point_confusions)
        draws=exclusion.metrics_from_confusions(saved["bootstrap_confusions"])
        for metric,value in draws.items():
            np.testing.assert_allclose(value,saved["metric__"+metric],atol=1e-12,rtol=0)
    expected=pd.DataFrame(exclusion.comparison_rows(point,draws))
    compare_frame(expected,pd.read_csv(exclusion.COMPARISON_FILE))
    return {"validation_and_test_predictions_checked":4,"paired_comparisons_recomputed":len(expected),"bootstrap_iterations":2000}


def verify_closeout_results():
    frames=closeout.load_predictions()
    for function,file in ((closeout.error_structure,closeout.ERROR_STRUCTURE_FILE),
                          (closeout.paired_outcomes,closeout.PAIRED_ERROR_FILE),
                          (closeout.illustrative_errors,closeout.ILLUSTRATIVE_ERROR_FILE)):
        actual=pd.read_csv(file,dtype={"meta_collision_index":"string"})
        expected=function(frames)
        if "meta_collision_index" in expected:
            expected["meta_collision_index"]=expected.meta_collision_index.astype("string")
        compare_frame(expected,actual)
    historical=closeout.read_json(closeout.DEVIATION_FILE)
    rebuilt=closeout.deviation_payload()
    old_inventory=historical["deviations"][0]["evidence"].pop("frozen_config_inventory")
    new_inventory=rebuilt["deviations"][0]["evidence"].pop("frozen_config_inventory")
    if not set(old_inventory).issubset(new_inventory) or historical != rebuilt:
        raise ValueError("Historical deviation record differs")
    return {"error_structure_rows":12,"paired_error_rows":20,"illustrative_rows":24,"regional_holdout_not_executed":True,
            "inventory_boundary":"Historical inventory is checked as a preserved subset, not compared to later-added filenames"}


def verify(root=ROOT):
    if root.resolve()!=ROOT.resolve():
        raise ValueError("Historical adapter is for the author snapshot; fresh runs use newly bound protocols")
    evidence=prep.read_json(root/EVIDENCE)
    records=[]
    for analysis in (exclusion,closeout):
        old=analysis.read_json(analysis.PROTOCOL_FILE)
        new=analysis.build_protocol()
        changes=compare_protocols(old,new,evidence)
        for relative,sha in new["upstream_sha256"].items():
            if prep.digest(root/relative)!=sha:
                raise ValueError("Rebuilt upstream integrity failure")
        manifest=check_manifest(root,analysis.MANIFEST_FILE)
        science=verify_exclusion_results() if analysis is exclusion else verify_closeout_results()
        records.append({"version":analysis.VERSION,"raw_legacy_exact_protocol_equal":old==new,
                        "scientific_protocol_settings_equal":True,"reviewed_provenance_differences":changes,
                        "manifest":manifest,"scientific_output_checks":science})
    report={"version":VERSION,"created_local":prep.now(),"status":"PASS_WITH_EXPLICIT_HISTORICAL_PROVENANCE_MIGRATIONS",
            "records":records,"normalization_commits":list(NORMALIZATION_COMMITS),
            "raw_legacy_test_failures_not_relabelled":True,"old_hashes_or_files_overwritten":False,
            "original_unshared_outline_bytes_verified":False,"fresh_reproduction_performed":False}
    prep.write_json(root/"logs/final_analysis/historical_d14_validation.json",report)
    print("HISTORICAL_D14_SCIENTIFIC_AND_NORMALIZED_SNAPSHOT_CHECK=PASS",flush=True)
    print("RAW_LEGACY_EXACT_HASH_TESTS=KNOWN_INAPPLICABLE_TO_NORMALIZED_SNAPSHOT; original files preserved",flush=True)
    return report


if __name__ == "__main__":
    verify()
