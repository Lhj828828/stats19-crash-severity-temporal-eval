"""Bounded entry validation: isolated raw-to-splits only, no model fitting."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import time

import final_reproduction as runner
import stats19_feature_revision as prep


def main():
    root=runner.ROOT
    report={"version":runner.VERSION,"created_local":prep.now(),"full_scale_training_run":False,
            "clean_environment_reproduction":False,"environment":"existing locked project environment",
            "datasets":[],"status":"RUNNING"}
    report_path=root/"logs/final_analysis/preparation_check.json"
    with tempfile.TemporaryDirectory(prefix="final-entry-check-") as directory:
        scratch=Path(directory).resolve()
        if not scratch.is_relative_to(Path(tempfile.gettempdir()).resolve()):
            raise ValueError("Temporary cleanup target outside the intended temporary directory")
        for dataset,limit in (("stats19",7),("cas",5)):
            workspace=scratch/dataset
            runner.prepare(workspace,dataset)
            marker=runner.validate_workspace(workspace,dataset)
            record={"dataset":dataset,"copied_files":len(marker["copied_sha256"]),"copied_models":False,
                    "copied_predictions":False,"copied_processed_data":False,"stages":[]}
            report["datasets"].append(record)
            for name,arguments in runner.stages(dataset)[:limit]:
                print("PREPARATION_CHECK "+dataset+" "+name,flush=True)
                started=time.perf_counter()
                result=subprocess.run([sys.executable,*arguments],cwd=workspace,text=True,encoding="utf-8",errors="replace",
                    capture_output=True,timeout=600,env={**os.environ,"PYTHONIOENCODING":"utf-8","MPLBACKEND":"Agg","OMP_NUM_THREADS":"8"})
                record["stages"].append({"name":name,"seconds":time.perf_counter()-started,"exit_code":result.returncode,
                                         "output":(result.stdout+result.stderr).replace(str(scratch),"TEMPORARY_CHECK_ROOT")})
                if result.returncode:
                    report["status"]="FAIL"
                    runner.write_json(report_path,report)
                    raise RuntimeError("Preparation stage failed: "+name+"; see preparation_check.json")
                runner.validate_workspace(workspace,dataset)
            if dataset=="stats19":
                result=subprocess.run([sys.executable,runner.BINDING,"--stage","prepare15"],cwd=workspace,
                    text=True,encoding="utf-8",errors="replace",capture_output=True,timeout=600)
                record["prepare15_exit_code"]=result.returncode
                record["prepare15_output"]=(result.stdout+result.stderr).replace(str(scratch),"TEMPORARY_CHECK_ROOT")
                if result.returncode:
                    report["status"]="FAIL"
                    runner.write_json(report_path,report)
                    raise RuntimeError("15-feature runtime preparation failed; see preparation_check.json")
                contract=runner.read_json(workspace/"config/stats19_feature_revision/protocol.json")
                if contract["schema"]["feature_columns"]!=list(prep.FEATURES):
                    raise ValueError("Reconstructed feature allowlist differs")
                record["feature_count"]=15
            if any((workspace/"models").rglob("*.joblib")):
                raise ValueError("Bounded preparation unexpectedly trained/copied a model")
            runner.validate_workspace(workspace,dataset)
            record["status"]="PASS"
    report["status"]="PASS_RAW_TO_SPLITS_ONLY_NO_MODEL_RECONSTRUCTION"
    runner.write_json(report_path,report)
    print(report["status"],flush=True)


if __name__=="__main__":
    main()
