# Public STATS19 reproduction validation

This record documents an author-run validation of the public reconstruction workflow. It is release-gate evidence, not a claim that an independent third party has already reproduced the study.

## Validated instance

- Source commit: `baf63c3d9373c64d9845374c280848ae1a45ef25`.
- Execution dates: 2026-09-02 to 2026-09-03 (Asia/Shanghai).
- Platform: Windows 11 x64.
- Environment: a clean virtual environment created by the public runner with Python 3.13.15 and `requirements-lock.txt`.
- Inputs: the seven fixed 2018-2024 STATS19 annual collision files; every byte size and SHA-256 value passed `config/public_data_manifest.json`.
- Isolation boundary: no existing model, prediction, intermediate dataset or retained D16 artifact was copied into the reproduction workspace.

## Outcome

The public runner completed all 32 analysis stages from the annual inputs through the D14 closeout. It then passed 14 independent stage checks and the compact scientific-result verifier:

- run status: `PIPELINE_COMPLETE`;
- passed recorded stages: 47/47;
- compact scientific comparisons: 21/21;
- required nonempty figures: 7/7;
- total recorded stage time: 8235.833 seconds (approximately 2 h 17 min);
- analysis-stage time: 8148.870 seconds;
- independent-check time: 86.533 seconds;
- compact-verification time: 0.430 seconds.

The result verifier used the frozen keys and tolerances in `config/public_result_contract.json`. It did not require byte-identical floating-point files and did not widen a tolerance after inspecting this run.

## Scope and remaining gate

This validation supports the executability of the public STATS19 workflow on the stated Windows/Python environment and the reproduction of its compact scientific outputs. It does not validate Linux or macOS execution, does not establish cross-country generality, and does not turn the descriptive SHAP analysis into causal evidence.

The fixed-input files are intentionally excluded from the software repository. The STATS19 and CAS source-terms reviews are complete; immutable public archive URLs and the data-record DOI remain pending. Until those archive gates are complete, a new user must obtain the exact files separately and pass the recorded hashes; the software release is not yet a self-contained one-command download-and-run package.

The full local run state and record-level outputs are not distributed because they contain local paths and large rebuildable artifacts. This tracked record therefore reports the status, counts and timing needed for the release gate; the public runner recreates its own machine-readable state and verification report in every isolated workspace.
