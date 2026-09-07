"""Curate corrected manuscript outputs without modifying scientific artifacts."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import stats19_feature_revision as prep
import stats19_feature_revision_full as full
import stats19_exclude2020_revision as sensitivity
import random_reference_revision as random_revision

ROOT = prep.ROOT
VERSION = "FINAL_ANALYSIS_SCOPE_20260907_V2"
PREVIOUS_VERSION = "FINAL_ANALYSIS_SCOPE_20260907_V1"
CATALOG = "config/final_analysis/result_catalog.json"
EXPORT_DIR = "results/final_analysis/v2"
BASE = "results/stats19_feature_revision/full/"
SENS = "results/stats19_feature_revision/exclude2020/"
CAS_RANDOM = "results/random_reference_revision/cas/"


@dataclass(frozen=True)
class Output:
    name: str
    dataset: str
    source: str
    keys: tuple[str, ...]
    purpose: str
    selection: str = "all"


OUTPUTS = (
    Output("stats19_main_metrics", "stats19", BASE+"test_metrics.csv", ("group","model"), "main performance table"),
    Output("stats19_confusions", "stats19", BASE+"confusion_matrices.csv", ("group","model","true","predicted"), "error structure"),
    Output("stats19_intervals", "stats19", BASE+"metric_intervals.csv", ("group","model","metric"), "main uncertainty"),
    Output("stats19_paired_comparisons", "stats19", BASE+"paired_differences.csv", ("group","candidate","reference","metric"), "H2 and comparator intervals"),
    Output("stats19_h1", "stats19", BASE+"h1_gaps.csv", ("seed","model","metric"), "same-model H1 gaps"),
    Output("stats19_h1_seed_summary", "stats19", BASE+"h1_summary.csv", ("model","metric"), "descriptive seed variability"),
    Output("stats19_shap_stability", "stats19", BASE+"shap_stability.csv", ("seed","scope"), "H3 rank stability"),
    Output("stats19_shap_importance", "stats19", BASE+"shap_importance.csv", ("split","cohort","scope","feature"), "15-feature mean absolute importance"),
    Output("stats19_shap_samples", "stats19", BASE+"shap_sample_counts.csv", ("split","cohort"), "explanation sample class counts"),
    Output("stats19_exclude2020_metrics", "stats19", SENS+"test_metrics.csv", ("model",), "appendix sensitivity"),
    Output("stats19_exclude2020_comparisons", "stats19", SENS+"paired_comparisons.csv", ("contrast","metric"), "appendix paired contrasts"),
    Output("cas_temporal_metrics", "cas", "results/cas_evaluation/test_metrics.csv", ("evaluation_group","model"), "CAS temporal replication", "cas_temporal"),
    Output("cas_random_baselines", "cas", "results/cas_evaluation/test_metrics.csv", ("evaluation_group","model"), "unchanged CAS random baselines", "cas_random_baselines"),
    Output("cas_random_lightgbm", "cas", CAS_RANDOM+"revised_test_metrics.csv", ("seed","cohort"), "corrected CAS random LightGBM"),
    Output("cas_h1", "cas", CAS_RANDOM+"h1_gaps.csv", ("seed","model","metric"), "corrected CAS within-dataset H1", "exclude_legacy_model"),
    Output("cas_h2", "cas", "results/cas_post_analysis/h2_bootstrap.csv", ("evaluation_group","metric","contrast"), "unchanged CAS temporal H2", "cas_temporal"),
    Output("cas_shap_stability", "cas", CAS_RANDOM+"shap_stability.csv", ("seed","scope"), "corrected CAS within-dataset H3"),
    Output("cas_shap_importance", "cas", CAS_RANDOM+"shap_importance.csv", ("seed","cohort","scope","feature"), "corrected CAS importance"),
    Output("cas_ablation_metrics", "cas", "results/cas_feature_ablation/test_metrics.csv", ("scenario","model"), "CAS post-hoc feature sensitivity"),
    Output("cas_ablation_comparisons", "cas", "results/cas_feature_ablation/test_paired_differences.csv", ("scenario","comparison","model_a","model_b","metric"), "CAS post-hoc paired contrasts"),
)


def read_table(file):
    return pd.read_csv(file,keep_default_na=False,float_precision="round_trip",dtype={"seed":"string","cohort_years":"string"})


def selected_frame(root, spec):
    frame = read_table(root/spec.source)
    if spec.selection == "cas_temporal":
        frame = frame.loc[frame.evaluation_group.eq("temporal_2025")]
    elif spec.selection == "cas_random_baselines":
        frame = frame.loc[frame.evaluation_group.ne("temporal_2025") & frame.model.isin(["dummy_most_frequent","logistic_weighted"])]
    elif spec.selection == "exclude_legacy_model":
        frame = frame.loc[frame.model.ne("lightgbm_legacy")]
    elif spec.selection != "all":
        raise ValueError("Unknown result selection")
    frame = frame.drop(columns=[c for c in frame if c.endswith("seconds")], errors="ignore")
    if frame.empty or frame.duplicated(list(spec.keys)).any():
        raise ValueError("Empty or nonunique final result table: " + spec.name)
    return frame.sort_values(list(spec.keys)).reset_index(drop=True)


def manifest_check(root, relative):
    frame = pd.read_csv(root/relative)
    for row in frame.itertuples(index=False):
        name = getattr(row,"relative_path",None) or getattr(row,"path",None)
        file = (root/name).resolve()
        if not file.is_relative_to(root.resolve()) or prep.digest(file) != row.sha256:
            raise ValueError("Artifact manifest mismatch: " + str(name))
        size = getattr(row,"bytes",None)
        if size is not None and file.stat().st_size != int(size):
            raise ValueError("Artifact size mismatch: " + name)
    return len(frame)


def verify_sources(root):
    sensitivity.require_protocol(root)
    sensitivity.require_main(root)
    for name in ("training_complete","evaluation_complete","bootstrap_complete","verification_complete"):
        sensitivity.checked_seal(root,name)
    random_revision.require_protocol(root,"cas")
    random_revision.require_all_models(root,"cas")
    cas_done = prep.read_json(root/"logs/random_reference_revision/cas/complete.json")
    if cas_done.get("status") != "POST_REVIEW_RANDOM_REFERENCE_CORRECTION_VERIFIED":
        raise ValueError("CAS correction not verified")
    manifest = root/"logs/random_reference_revision/cas/artifact_manifest.json"
    if prep.digest(manifest) != cas_done["manifest_sha256"]:
        raise ValueError("CAS corrected manifest differs")
    stored = prep.read_json(manifest)
    prep.check_hashes(root, stored)
    ablation = prep.read_json(root/"config/cas/cas_feature_ablation_complete.json")
    if ablation["status"] != "POST_HOC_FEATURE_ABLATION_COMPLETE":
        raise ValueError("CAS sensitivity incomplete")
    prep.check_hashes(root,{ablation["protocol"]:ablation["protocol_sha256"],ablation["manifest"]:ablation["manifest_sha256"]})
    manifest_check(root,ablation["manifest"])
    return {"stats19_primary": "verified", "stats19_exclude2020": "verified", "cas_random_revision": "verified",
            "cas_feature_ablation": "verified", "models_refitted": False, "fresh_reproduction_performed": False}


def archive_v1_catalog(root, existing):
    if existing["version"] != PREVIOUS_VERSION:
        raise ValueError("Only the documented V1 CAS H2 scope correction is allowed")
    for record in existing["tables"]:
        prep.check_hashes(root,{record["source"]:record["source_sha256"],record["output"]:record["output_sha256"]})
    source = root/CATALOG
    archive = root/"config/final_analysis/history/result_catalog_v1.json"
    original_hash = prep.digest(source)
    archive.parent.mkdir(parents=True,exist_ok=True)
    if archive.exists():
        if prep.digest(archive) != original_hash:
            raise ValueError("Historical V1 catalog archive differs; refusing overwrite")
    else:
        shutil.copy2(source,archive)
    return {"previous_version":PREVIOUS_VERSION,"previous_catalog":archive.relative_to(root).as_posix(),
            "previous_catalog_sha256":original_hash,
            "reason":"CAS H2 export must contain temporal_2025 only; V1 also included 90 superseded random-reference contrast rows",
            "scientific_artifacts_changed":False,"old_export_files_preserved":True,
            "scope_change_only":True}


def verify_catalog_amendment(root, payload):
    amendment = payload.get("catalog_amendment")
    if not amendment:
        return
    archive = root/amendment["previous_catalog"]
    if prep.digest(archive) != amendment["previous_catalog_sha256"]:
        raise ValueError("Archived V1 catalog changed")
    previous = {r["name"]:r for r in prep.read_json(archive)["tables"]}
    current = {r["name"]:r for r in payload["tables"]}
    if set(previous) != set(current):
        raise ValueError("CAS H2 correction unexpectedly changed table membership")
    for name, record in current.items():
        old = previous[name]
        for key in ("source","source_sha256","keys","columns"):
            if old[key] != record[key]:
                raise ValueError("Scientific source or schema changed during catalog correction: "+name)
        prep.check_hashes(root,{old["output"]:old["output_sha256"],record["output"]:record["output_sha256"]})
        if name != "cas_h2":
            if (old["output_sha256"],old["selection"]) != (record["output_sha256"],record["selection"]):
                raise ValueError("Unrelated table changed during CAS H2 correction: "+name)
        else:
            if (old["selection"],record["selection"]) != ("all","cas_temporal"):
                raise ValueError("Unexpected CAS H2 selection change")
            expected = read_table(root/old["output"])
            expected = expected.loc[expected.evaluation_group.eq("temporal_2025")].reset_index(drop=True)
            pd.testing.assert_frame_equal(expected,read_table(root/record["output"]),check_dtype=False,check_exact=True)


def assemble(root, *, migrate_cas_h2=False):
    verification = verify_sources(root)
    records = []
    destination = root/CATALOG
    existing = prep.read_json(destination) if destination.exists() else None
    migration = None
    if existing and existing["version"] != VERSION:
        if not migrate_cas_h2:
            raise ValueError("Catalog version differs; the reviewed V1 CAS H2 correction requires --migrate-cas-h2")
        migration = archive_v1_catalog(root,existing)
        existing = None
    for spec in OUTPUTS:
        frame = selected_frame(root,spec)
        outfile = root/EXPORT_DIR/(spec.name+".csv")
        if existing:
            record = next(r for r in existing["tables"] if r["name"] == spec.name)
            prep.check_hashes(root,{spec.source:record["source_sha256"],record["output"]:record["output_sha256"]})
            current = read_table(outfile)
            pd.testing.assert_frame_equal(frame,current,check_dtype=False,check_exact=True)
        else:
            full.write_csv(outfile,frame)
        records.append({**asdict(spec),"keys":list(spec.keys),"rows":len(frame),"columns":list(frame.columns),
                        "source_sha256":prep.digest(root/spec.source),"output":outfile.relative_to(root).as_posix(),
                        "output_sha256":prep.digest(outfile)})
    payload = {"version":VERSION,"assembled_local":prep.now(),"status":"AUTHOR_VERIFIED_FINAL_RESULT_SELECTION",
               "post_review":True,"fresh_reproduction_complete":False,"software_release_contains_revision":False,
               "tables":records,"verification":verification,
               "omitted_from_revised_manuscript":["Ordered Logit","matched-subset Ordered Logit comparison","tree-count sensitivity"],
               "preserved_history":True,"no_direct_cross_dataset_SHAP_rho_comparison":True,
               "recording_diagnostic":"logs/stats19_feature_revision/recording_system_severity_counts.csv",
               "entrypoint":"run_final_analysis.py",
               "scientific_source_hashes":{full.protocol_path(root).relative_to(root).as_posix():prep.digest(full.protocol_path(root)),
                                          sensitivity.protocol_path(root).relative_to(root).as_posix():prep.digest(sensitivity.protocol_path(root))}}
    if migration:
        payload["catalog_amendment"] = migration
    verify_catalog_amendment(root,existing or payload)
    if existing is None:
        prep.write_json(destination,payload)
    elif existing["tables"] != records:
        raise ValueError("Final catalog changed; create an explicit new version")
    print(f"FINAL_RESULT_CATALOG=PASS TABLES={len(records)}; no models refitted",flush=True)
    return existing or payload


def compare(candidate, reference, dataset="all"):
    contract = prep.read_json(reference/"config/final_analysis/result_catalog.json")
    checks = []
    for spec in OUTPUTS:
        if dataset != "all" and spec.dataset != dataset:
            continue
        record = next(r for r in contract["tables"] if r["name"] == spec.name)
        expected_path = reference/record["output"]
        if prep.digest(expected_path) != record["output_sha256"]:
            raise ValueError("Reference table changed")
        expected = read_table(expected_path)
        actual = selected_frame(candidate,spec)
        if list(expected.columns) != list(actual.columns) or len(expected) != len(actual):
            raise ValueError("Comparison schema/row count differs: " + spec.name)
        for col in expected:
            if pd.api.types.is_float_dtype(expected[col]):
                np.testing.assert_allclose(actual[col],expected[col],atol=1e-10,rtol=1e-8,err_msg=spec.name+":"+col)
            else:
                np.testing.assert_array_equal(actual[col],expected[col],err_msg=spec.name+":"+col)
        checks.append({"table":spec.name,"rows":len(actual),"status":"PASS"})
    return {"status":"PASS","reference_catalog_version":contract["version"],"tables":checks,"float_atol":1e-10,"float_rtol":1e-8,
            "boundary":"No tolerance expansion; any mismatch requires a documented diagnosis, not silent reference replacement"}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assemble",action="store_true")
    parser.add_argument("--migrate-cas-h2",action="store_true",help="explicitly archive V1 and correct its CAS H2 export scope")
    parser.add_argument("--compare",choices=("stats19","cas"))
    args=parser.parse_args()
    if args.assemble or args.migrate_cas_h2:
        assemble(ROOT,migrate_cas_h2=args.migrate_cas_h2)
    if args.compare:
        report=compare(ROOT,ROOT/"config/final_reference",args.compare)
        prep.write_json(ROOT/"logs/final_result_comparison.json",report)
        print("FINAL_RESULT_COMPARISON=PASS")
