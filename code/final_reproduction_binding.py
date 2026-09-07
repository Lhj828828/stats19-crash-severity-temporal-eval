"""Bind unchanged scientific kernels to a newly prepared reconstruction root.

No archived file is edited: hashes identify the newly generated run, while
the marker preserves the source template hashes and comparison references.
"""
from pathlib import Path
import argparse

import stats19_feature_revision as prep
import stats19_feature_revision_full as full
import stats19_exclude2020_revision as sensitivity


def require_workspace(root):
    marker=prep.read_json(root/"logs/final_workspace.json")
    if marker.get("version")!="FINAL_REPRODUCTION_ENTRY_V1" or marker.get("dataset")!="stats19":
        raise ValueError("Runtime binding only allowed in a prepared STATS19 reconstruction workspace")
    prep.check_hashes(root,marker["copied_sha256"])
    return marker


def bind_full(root):
    parent=prep.require_protocol(root)
    if parent["schema"]["feature_columns"]!=list(prep.FEATURES):
        raise ValueError("Rebuilt preparation feature contract differs")
    full.PARENT_HASH=prep.digest(prep.output_path(root,"config","protocol.json"))


def execute(stage,root=prep.ROOT):
    require_workspace(root)
    with prep.execution_lock(root),full.threadpool_limits(limits=8):
        if stage=="prepare15":
            prep.prepare(root)
            return
        if stage=="smoke15":
            prep.smoke(root)
            prep.verify(root)
            _,_,y,metadata,_=prep.load_data(root)
            metadata=metadata.copy()
            metadata[prep.TARGET]=y
            frame=metadata.groupby([prep.YEAR,"meta_injury_based",prep.TARGET]).size().rename("records").reset_index()
            full.write_csv(prep.output_path(root,"logs","recording_system_severity_counts.csv"),frame)
            return
        bind_full(root)
        if stage.startswith("main_"):
            actions={"preflight":full.preflight,"freeze":full.freeze,"train":full.train,"evaluate":full.evaluate,
                     "bootstrap":full.bootstrap,"shap":full.shap_analysis,"verify":full.verify}
            actions[stage.removeprefix("main_")](root)
            return
        if stage.startswith("exclude_"):
            full.require_protocol(root)
            sensitivity.MAIN_HASH=prep.digest(full.protocol_path(root))
            actions={"preflight":sensitivity.preflight,"freeze":sensitivity.freeze,"train":sensitivity.train,
                     "evaluate":sensitivity.evaluate,"bootstrap":sensitivity.bootstrap,"verify":sensitivity.verify}
            actions[stage.removeprefix("exclude_")](root)
            return
        raise ValueError("Unknown bounded reproduction stage")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",required=True)
    execute(parser.parse_args().stage)
