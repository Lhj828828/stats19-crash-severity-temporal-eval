"""Plan/check the final scope now; explicitly rebuild in an isolated root later."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import public_reproduction as legacy_stats19
import cas_public_reproduction as legacy_cas

ROOT=Path(__file__).resolve().parents[1]
VERSION="FINAL_REPRODUCTION_ENTRY_V1"
BINDING="code/final_reproduction_binding.py"


def digest(file):
    with Path(file).open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def write_json(file,payload):
    legacy_stats19.write_json_atomic(file,payload)


def read_json(file):
    return json.loads(Path(file).read_text(encoding="utf-8"))


def stages(dataset):
    if dataset=="stats19":
        initial=[(s.name,s.arguments) for s in legacy_stats19.PIPELINE_STAGES[:7]]
        corrected=[("CORRECTED_"+name,(BINDING,"--stage",name)) for name in
                   ("prepare15","smoke15","main_preflight","main_freeze","main_train","main_evaluate",
                    "main_bootstrap","main_shap","main_verify","exclude_preflight","exclude_freeze",
                    "exclude_train","exclude_evaluate","exclude_bootstrap","exclude_verify")]
        return initial+corrected
    if dataset=="cas":
        # Legacy CAS temporal/baseline outputs are needed; old random nonlinear
        # outputs remain diagnostic references, never final results.
        initial=[(s.name,s.arguments) for s in legacy_cas.PIPELINE_STAGES
                 if s.name not in {"CAS_D8_SHAP_STABILITY","CAS_PUBLIC_BIND_REPORT_PROTOCOL",
                                   "CAS_PUBLIC_CLOSEOUT","CAS_PUBLIC_WRITE_WORKSPACE_MANIFEST"}]
        corrected=[("CAS_RANDOM_CORRECTED_"+name,("code/random_reference_revision.py","--dataset","cas","--stage",name))
                   for name in ("freeze","smoke","tune","evaluate","bootstrap","shap","summarize","verify")]
        return initial+corrected
    raise ValueError("Unknown dataset")


def safe_workspace(workspace,source):
    workspace,source=workspace.expanduser().resolve(),source.resolve()
    if workspace==source or workspace.is_relative_to(source) or source.is_relative_to(workspace):
        raise ValueError("Reproduction directory must be separate from the source project and its ancestors")
    return workspace


def supplementary_inputs(source):
    mapping={}
    for folder,pattern in (("code","*.py"),("tests","*.py"),("docs","STATS19*REVISION*.md")):
        for file in (source/folder).glob(pattern):
            mapping[file.relative_to(source).as_posix()]=file
    for name in ("requirements.txt","requirements-lock.txt","run_stats19_feature_revision.py",
                 "run_stats19_feature_revision_full.py","run_stats19_exclude2020_revision.py",
                 "run_random_reference_revision.py","run_final_analysis.py"):
        mapping[name]=source/name
    catalog=read_json(source/"config/final_analysis/result_catalog.json")
    mapping["config/final_reference/config/final_analysis/result_catalog.json"]=source/"config/final_analysis/result_catalog.json"
    for record in catalog["tables"]:
        mapping["config/final_reference/"+record["output"]]=source/record["output"]
    return mapping


def stats19_input_map(source):
    mapping=supplementary_inputs(source)
    # Historical templates preserve the reviewed model settings and SHAP
    # provenance, not old validation outcomes for corrected model selection.
    for name in ("config/public_data_manifest.json","config/d8_baseline_protocol.json",
                 "config/d10_lightgbm_protocol.json","config/d14_shap_protocol.json"):
        mapping[name]=source/name
    for file in (source/"data/external/documentation").glob("*"):
        if file.is_file():
            mapping[file.relative_to(source).as_posix()]=file
    for name in legacy_stats19.verify_source_stats19(source):
        mapping[name]=source/name
    return mapping


def prepare(workspace,dataset,source=ROOT):
    workspace=safe_workspace(workspace,source)
    if workspace.exists():
        raise FileExistsError("Directory exists; use --resume only for a verified matching workspace")
    if dataset=="stats19":
        mapping=stats19_input_map(source)
        workspace.mkdir(parents=True)
    else:
        from public_data import load_manifest,select_specs,verify_files
        _,specs=load_manifest(source/"config/public_data_manifest.json")
        specs=select_specs(specs,["cas"])
        if any(r.status!="PASS" for r in verify_files(source,specs)):
            raise ValueError("CAS public data files not verified")
        snapshot=next(source/s.relative_path for s in specs if s.relative_path.endswith(".jsonl.gz"))
        legacy_cas.prepare_workspace(workspace,source=source,snapshot=snapshot)
        mapping=supplementary_inputs(source)
    for relative,file in mapping.items():
        if not file.is_file():
            raise FileNotFoundError(file)
        destination=(workspace/relative).resolve()
        if not destination.is_relative_to(workspace):
            raise ValueError("Input path escapes workspace")
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists() and digest(destination)!=digest(file):
            raise ValueError("Conflicting immutable workspace input: "+relative)
        shutil.copy2(file,destination)
    for family in ("logs","models","results","figures","data/interim","data/processed"):
        (workspace/family).mkdir(parents=True,exist_ok=True)
    copied={p.relative_to(workspace).as_posix():digest(p) for p in workspace.rglob("*") if p.is_file()}
    if any(name.startswith(("models/","results/","data/processed/","data/interim/")) for name in copied):
        raise ValueError("A prepared workspace contains prior analytical artifacts")
    write_json(workspace/"logs/final_workspace.json",{
        "version":VERSION,"dataset":dataset,"created_local":time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "copied_sha256":copied,"plan":[{"name":n,"arguments":list(a)} for n,a in stages(dataset)],
        "copied_trained_models":False,"copied_record_predictions":False,"copied_processed_data":False,
        "compact_references_only":"config/final_reference", "reproduction_complete":False})
    print("FINAL_WORKSPACE_PREPARED "+dataset+"; no model fitting",flush=True)
    return workspace


def validate_workspace(workspace,dataset):
    marker=read_json(workspace/"logs/final_workspace.json")
    if marker["version"]!=VERSION or marker["dataset"]!=dataset:
        raise ValueError("Workspace identity differs")
    if marker["plan"]!=[{"name":n,"arguments":list(a)} for n,a in stages(dataset)]:
        raise ValueError("Stage plan changed after preparation")
    for name,sha in marker["copied_sha256"].items():
        target=(workspace/name).resolve()
        if not target.is_relative_to(workspace.resolve()) or digest(target)!=sha:
            raise ValueError("Immutable reconstruction input changed: "+name)
    return marker


def reproduce(workspace,dataset,*,resume=False,use_current_environment=False):
    workspace=safe_workspace(workspace,ROOT)
    if not workspace.exists():
        prepare(workspace,dataset)
    elif not resume:
        raise FileExistsError("Use --resume for an existing prepared workspace")
    marker=validate_workspace(workspace,dataset)
    # A .log suffix is intentional: frozen scientific manifests exclude mutable
    # process ledgers. The input marker and completed output seals stay immutable.
    state_file=workspace/"logs/final_reproduction_state.log"
    state=read_json(state_file) if state_file.exists() else {"version":VERSION,"completed":[],"status":"RUNNING"}
    completed=state["completed"]
    plan=stages(dataset)
    if completed!=[name for name,_ in plan[:len(completed)]]:
        raise ValueError("Resume ledger is not an ordered stage prefix")
    environment_record=workspace/"logs/public_environment_ready.json"
    if environment_record.exists():
        recorded=read_json(environment_record)
        expected_mode="existing_environment" if use_current_environment else "clean_virtual_environment"
        if recorded["mode"]!=expected_mode or recorded["requirements_lock_sha256"]!=digest(workspace/"requirements-lock.txt"):
            raise ValueError("Resume environment or locked dependencies changed")
        python=Path(recorded["python"])
        if not python.is_file():
            raise FileNotFoundError("Recorded reproduction interpreter is unavailable")
    else:
        python=legacy_stats19.prepare_python(workspace,None,use_current_environment=use_current_environment)
    env={**os.environ,**legacy_stats19.THREAD_ENVIRONMENT}
    for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","LOKY_MAX_CPU_COUNT"):
        env[name]="8"
    for name,arguments in plan[len(completed):]:
        validate_workspace(workspace,dataset)
        legacy_stats19.run_process([str(python),*arguments],cwd=workspace,environment=env,
            log_file=workspace/"logs/final_stage_logs"/(name+".log"),timeout_seconds=4*60*60,label=name)
        state["completed"].append(name)
        write_json(state_file,state)
    # Run comparison in the reconstructed environment, not the host interpreter.
    legacy_stats19.run_process([str(python),"code/final_result_catalog.py","--compare",dataset],cwd=workspace,
        environment=env,log_file=workspace/"logs/final_stage_logs/FINAL_COMPARISON.log",timeout_seconds=600,label="FINAL_COMPARISON")
    state.update(status="COMPLETE",clean_environment=not use_current_environment)
    write_json(state_file,state)
    print("FINAL_REPRODUCTION_COMPLETE "+dataset,flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("plan","check","package","legacy-check","prepare","reproduce"),default="plan")
    parser.add_argument("--dataset",choices=("stats19","cas","all"),default="all")
    parser.add_argument("--workspace",type=Path)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--use-current-environment",action="store_true",help="engineering check only; not clean-environment evidence")
    args=parser.parse_args(argv)
    datasets=("stats19","cas") if args.dataset=="all" else (args.dataset,)
    if args.stage=="plan":
        for dataset in datasets:
            print(dataset.upper())
            for index,(name,_) in enumerate(stages(dataset),1):
                print(f"{index:02d} {name}")
        print("PLAN_ONLY: no files or models changed. Full reconstruction remains unexecuted for this entry until explicitly requested.")
    elif args.stage in {"check","package"}:
        from final_result_catalog import assemble
        assemble(ROOT)
        if args.stage=="check":
            from historical_d14_validation import verify
            verify(ROOT)
    elif args.stage=="legacy-check":
        from historical_d14_validation import verify
        verify(ROOT)
    else:
        if args.workspace is None:
            parser.error("An explicit separate --workspace is required")
        for dataset in datasets:
            target=args.workspace/dataset if len(datasets)>1 else args.workspace
            if args.stage=="prepare":
                prepare(target,dataset)
            else:
                reproduce(target,dataset,resume=args.resume,use_current_environment=args.use_current_environment)


if __name__=="__main__":
    main()
