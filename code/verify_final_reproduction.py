"""Read-only acceptance of completed clean runs against the corrected catalog."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import final_reproduction as runner
import final_result_catalog as catalog
import stats19_feature_revision as prep


ROOT = runner.ROOT
REVIEWED_NONSCIENTIFIC_CHANGES = {
    "code/final_result_catalog.py": (
        "2280491ba000b1a7802d46999a77ea16fde737ae99d82941edb2b0058787a47a",
        "d0c73f87432b6d09f10d64664398aebbd8836c4d963f5087d860e21146c98e10"),
    "tests/test_final_analysis.py": (
        "3f0f214b47c0ca6055f9587f0ce624607456fef22ba19748c2798f521380e80a",
        "223dd864788bf94521cf81de5581da7919eca08a91803669e59fd669f8b05721"),
}


def verify_source_snapshot(workspace, marker):
    changes = []
    for name, expected in marker["copied_sha256"].items():
        if not name.endswith(".py"):
            continue
        current = runner.digest(ROOT/name)
        if current != expected:
            if REVIEWED_NONSCIENTIFIC_CHANGES.get(name) != (expected,current):
                raise ValueError("Unreviewed executable difference: " + name)
            changes.append({"file":name,"run_sha256":expected,"current_sha256":current,
                            "reason":"Reviewed CAS H2 catalog-only scope amendment and its tests"})
    return changes


def verify_seal_tree(workspace, relative, seen=None):
    seen = set() if seen is None else seen
    if relative in seen:
        return
    seen.add(relative)
    payload = runner.read_json(workspace/relative)
    if not isinstance(payload, dict):
        return
    files = payload.get("files_sha256", {})
    prep.check_hashes(workspace, files)
    for name in files:
        if name.endswith(".json"):
            verify_seal_tree(workspace, name, seen)


def check_completed_workspace(workspace, dataset):
    marker = runner.validate_workspace(workspace, dataset)
    state = runner.read_json(workspace/"logs/final_reproduction_state.log")
    if (state["status"] != "COMPLETE" or state.get("clean_environment") is not True
            or state["completed"] != [name for name, _ in runner.stages(dataset)]):
        raise ValueError("Clean reconstruction has not completed all required stages: " + dataset)
    environment = runner.read_json(workspace/"logs/public_environment_ready.json")
    python = Path(environment["python"]).resolve()
    if (environment["mode"] != "clean_virtual_environment"
            or not python.is_relative_to((workspace/".venv").resolve())
            or environment["requirements_lock_sha256"] != runner.digest(workspace/"requirements-lock.txt")
            or runner.digest(workspace/"requirements-lock.txt") != runner.digest(ROOT/"requirements-lock.txt")):
        raise ValueError("Clean environment identity differs")
    if any(marker[key] is not False for key in (
            "copied_trained_models", "copied_record_predictions", "copied_processed_data")):
        raise ValueError("Reconstruction copied analytical artifacts")
    changes = verify_source_snapshot(workspace, marker)
    if dataset == "stats19":
        for branch in ("full", "exclude2020"):
            relative = f"logs/stats19_feature_revision/{branch}/verification_complete.json"
            if runner.read_json(workspace/relative)["status"] != "COMPLETE":
                raise ValueError("STATS19 verification seal incomplete")
            verify_seal_tree(workspace, relative)
        diagnostic = "logs/stats19_feature_revision/recording_system_severity_counts.csv"
        pd.testing.assert_frame_equal(catalog.read_table(workspace/diagnostic), catalog.read_table(ROOT/diagnostic),
                                      check_dtype=False, check_exact=True)
    else:
        relative = "logs/random_reference_revision/cas/complete.json"
        seal = runner.read_json(workspace/relative)
        if seal["status"] != "POST_REVIEW_RANDOM_REFERENCE_CORRECTION_VERIFIED":
            raise ValueError("CAS correction verification incomplete")
        manifest = "logs/random_reference_revision/cas/artifact_manifest.json"
        prep.check_hashes(workspace, {manifest:seal["manifest_sha256"]})
        prep.check_hashes(workspace, runner.read_json(workspace/manifest))
        ablation = runner.read_json(workspace/"config/cas/cas_feature_ablation_complete.json")
        if ablation["status"] != "POST_HOC_FEATURE_ABLATION_COMPLETE":
            raise ValueError("CAS ablation incomplete")
        prep.check_hashes(workspace, {ablation["manifest"]:ablation["manifest_sha256"]})
        catalog.manifest_check(workspace, ablation["manifest"])
    return {"dataset":dataset,"completed_stages":len(state["completed"]),
            "python_version":environment["python_version"],"clean_environment":True,
            "copied_analytical_artifacts":False,"reviewed_nonscientific_source_changes":changes,
            "numerical_comparison":catalog.compare(workspace, ROOT, dataset)}


def check_figure_files(workspace, dataset):
    folders = ("figures/stats19_feature_revision/full",) if dataset == "stats19" else (
        "figures/random_reference_revision/cas",)
    files = []
    for folder in folders:
        candidates = sorted((workspace/folder).glob("*.png"))
        if not candidates:
            raise ValueError("Expected diagnostic figures are missing: " + folder)
        for file in candidates:
            with Image.open(file) as image:
                image.load()
                width, height = image.size
                pixels = np.asarray(image.convert("RGB"))
            if width < 200 or height < 150 or float(pixels.std()) < 1:
                raise ValueError("Blank or unexpectedly small diagnostic figure: " + file.name)
            files.append({"file":file.relative_to(workspace).as_posix(),"width":width,"height":height,
                          "sha256":runner.digest(file),"status":"NONBLANK_FILE_CHECK_PASS"})
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,required=True)
    args = parser.parse_args()
    base = runner.safe_workspace(args.workspace, ROOT)
    contract = catalog.assemble(ROOT)
    report = {"status":"RUNNING","created_local":prep.now(),"reference_catalog_version":contract["version"],
              "catalog_sha256":runner.digest(ROOT/catalog.CATALOG),"datasets":[],
              "model_training_performed_by_this_check":False,"fresh_download_performed":False,
              "input_boundary":"Previously verified immutable raw snapshots were copied, never models or predictions",
              "manuscript_layout_verified":False,"new_software_release_published":False}
    destination = ROOT/"logs/final_analysis/final_reproduction_acceptance.json"
    try:
        for dataset in ("stats19", "cas"):
            result = check_completed_workspace(base/dataset, dataset)
            result["figures"] = check_figure_files(base/dataset, dataset)
            report["datasets"].append(result)
            print("FINAL_ACCEPTANCE " + dataset + " PASS",flush=True)
        report["status"] = "PASS_CLEAN_RAW_RECONSTRUCTION_AND_V2_RESULTS"
    except Exception as error:
        report.update(status="FAIL",failure_type=type(error).__name__)
        runner.write_json(destination, report)
        raise
    runner.write_json(destination, report)
    print(report["status"],flush=True)


if __name__ == "__main__":
    main()
