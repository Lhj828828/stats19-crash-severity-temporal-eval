"""Reproduce Appendix A.14.6 separately from a completed 15-feature parent run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

import qwk_prevalence_sensitivity as qwk

VERSION = "QWK_REPRODUCTION_ENTRY_V1"


def input_names():
    names = [qwk.prep.ASSIGNMENTS,
             "config/stats19_feature_revision/full/execution_protocol.json",
             "logs/stats19_feature_revision/full/evaluation_complete.json",
             "logs/stats19_feature_revision/full/bootstrap_complete.json"]
    base = "results/stats19_feature_revision/full/"
    names += [base + n for n in ("test_metrics.csv", "confusion_matrices.csv", "h1_gaps.csv", "h1_summary.csv")]
    names += [f"{base}{part}/random_seed_{seed}_{cohort}.npz"
              for seed in qwk.prep.SEEDS for cohort in ("internal", "2024")
              for part in ("evaluation", "bootstrap")]
    return names


def check_paths(parent, workspace, source=ROOT):
    parent, workspace, source = (Path(p).expanduser().resolve() for p in (parent, workspace, source))
    if not parent.is_dir():
        raise FileNotFoundError("Completed parent directory is missing")
    for protected in (parent, source):
        if workspace == protected or workspace.is_relative_to(protected) or protected.is_relative_to(workspace):
            raise ValueError("Choose a separate workspace outside the parent and source directories")
    return parent, workspace


def prepare(parent, workspace):
    if workspace.exists():
        raise FileExistsError("Workspace exists; use --resume after verifying its provenance")
    mapping = {name: parent / name for name in input_names()}
    mapping.update({name: ROOT / name for name in qwk.SOURCE_FILES})
    for name, path in mapping.items():
        if not path.is_file():
            raise FileNotFoundError("Required source/input missing: " + name)
    # A reconstruction has its own timestamps/hashes, so freeze a new QWK protocol there.
    # Preserve the archived author protocol and the completed parent run unchanged.
    original = {name: qwk.digest(path) for name, path in mapping.items()}
    workspace.mkdir(parents=True)
    for name, path in mapping.items():
        destination = workspace / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    qwk.freeze(workspace)
    if any(qwk.digest(path) != original[name] for name, path in mapping.items()):
        raise ValueError("A source/input changed during preparation")
    qwk.write_json(workspace / "logs/qwk_reproduction/prepare.json", {
        "version": VERSION, "parent_inputs_sha256": {name: original[name] for name in input_names()},
        "executed_source_sha256": {name: original[name] for name in qwk.SOURCE_FILES},
        "wrapper_sha256": qwk.digest(Path(__file__)), "model_fits": 0,
        "prediction_calls": 0, "inputs_reused_from_completed_parent": True,
        "boundary": "Postprocessing reproduction only; no new raw-to-model reconstruction."
    })


def verify_binding(parent, workspace):
    marker = qwk.read_json(workspace / "logs/qwk_reproduction/prepare.json")
    if marker["version"] != VERSION or marker["wrapper_sha256"] != qwk.digest(Path(__file__)):
        raise ValueError("Reproduction entry changed since preparation")
    if set(marker["parent_inputs_sha256"]) != set(input_names()):
        raise ValueError("Incomplete parent input inventory")
    if set(marker["executed_source_sha256"]) != set(qwk.SOURCE_FILES):
        raise ValueError("Incomplete executed-source inventory")
    for name, expected in marker["parent_inputs_sha256"].items():
        if qwk.digest(parent / name) != expected or qwk.digest(workspace / name) != expected:
            raise ValueError("Parent input differs: " + name)
    for name, expected in marker["executed_source_sha256"].items():
        if qwk.digest(ROOT / name) != expected or qwk.digest(workspace / name) != expected:
            raise ValueError("Executed source differs: " + name)
    qwk.verify_protocol(workspace)


def compare_reference(workspace):
    comparisons = []
    base = Path("results/qwk_prevalence_sensitivity")
    for name, keys in (("qwk_gaps.csv", ["seed", "model"]),
                       ("class_weights.csv", ["seed", "class"]),
                       ("seed_summary.csv", ["model", "metric"])):
        expected = qwk.pd.read_csv(ROOT / base / name, float_precision="round_trip").sort_values(keys).reset_index(drop=True)
        actual = qwk.pd.read_csv(workspace / base / name, float_precision="round_trip").sort_values(keys).reset_index(drop=True)
        if expected.shape != actual.shape or list(expected.columns) != list(actual.columns):
            raise ValueError("Reference table shape differs: " + name)
        for column in expected:
            if qwk.pd.api.types.is_float_dtype(expected[column]):
                qwk.np.testing.assert_allclose(actual[column], expected[column], atol=1e-10, rtol=1e-8)
            else:
                qwk.np.testing.assert_array_equal(actual[column], expected[column])
        comparisons.append({"file": name, "rows": len(actual), "columns": len(actual.columns), "pass": True})
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path, help="Completed STATS19 15-feature reconstruction root")
    parser.add_argument("--workspace", required=True, type=Path, help="New separate directory for QWK postprocessing")
    parser.add_argument("--resume", action="store_true", help="Verify and continue the same QWK workspace")
    args = parser.parse_args()
    parent, workspace = check_paths(args.parent, args.workspace)
    if args.resume:
        if not workspace.is_dir():
            raise FileNotFoundError("No prepared QWK workspace to resume")
    else:
        prepare(parent, workspace)
    verify_binding(parent, workspace)
    with qwk.threadpool_limits(limits=8):
        qwk.analyze(workspace)
    comparisons = compare_reference(workspace)
    verify_binding(parent, workspace)
    result = {"status": "PASS_QWK_POSTPROCESSING_REPRODUCTION", "version": VERSION,
              "comparisons": comparisons, "float_atol": 1e-10, "float_rtol": 1e-8,
              "model_fits": 0, "prediction_calls": 0, "parent_inputs_unchanged": True,
              "post_hoc": True, "boundary": "Uses completed parent outputs; not a fresh raw-to-model or third-party validation."}
    qwk.write_json(workspace / "logs/qwk_reproduction/acceptance.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
