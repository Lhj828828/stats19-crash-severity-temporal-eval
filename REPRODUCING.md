# Reproducing the analyses

## Current manuscript workflow (v1.3.1)

Start here for the **15-feature STATS19 analysis**, retained exclusion-2020
sensitivity, corrected CAS random references, CAS field-removal sensitivities,
and the QWK analysis in Appendix A.14.6/Table A13. STATS19 and CAS are trained
and evaluated separately. Their labels and features are not interchangeable.
See [FINAL_ANALYSIS.md](FINAL_ANALYSIS.md) for the scientific scope.

### Obtain the fixed source and environment

Download the v1.3.1 software archive from
<https://doi.org/10.5281/zenodo.22793992>, or use its fixed Git tag:

```text
git -c core.autocrlf=false clone --branch v1.3.1 --single-branch https://github.com/Lhj828828/stats19-crash-severity-temporal-eval.git
cd stats19-crash-severity-temporal-eval
git rev-parse HEAD
```

The commit must be `e0ccfeb60ec66a3ea0febe969df58f6868ba1ab6`. A documentation
update on main does not move that tag or replace the published archive.

Use a separate Python **3.13** environment; the author-run validation used
3.13.15 on Windows. In the commands below, `python` means the chosen environment's
interpreter, not an unchecked system default. Install `requirements-lock.txt`,
record the actual Python/package versions, and check each command's exit status
before continuing. The lock does not pin every transitive dependency. Different
machines/platforms require their own validation; a historical pass is not a new
run's completion evidence.

```text
python -m pip install --requirement requirements-lock.txt
python -m pip check
python download_and_verify_data.py download --dataset all
python download_and_verify_data.py verify --dataset all
python run_final_analysis.py --stage plan
```

Require `Verified: 9/9 files` before training. Downloading from the two fixed
data DOIs checks both byte sizes and SHA-256; do not substitute live data.
The plan should list 22 STATS19 stages and 25 CAS stages. `--stage check` is an
author-side audit of existing artifacts, not a fresh clone's installation test.

### Reconstruct both datasets

From the fixed source directory, use a new workspace outside the source tree:

```text
python run_final_analysis.py --stage reproduce --dataset all --workspace ../final-reconstruction
```

This runs the datasets sequentially, creates a clean environment for each, and
regenerates models, processed data, predictions and SHAP arrays. Small archived
reference tables are used only for comparison. The STATS19 branch does not run
the retired Ordered Logit, matched-subset or tree-count-extension stages. CAS
retains required diagnostic dependencies and then applies its random-reference
correction. Its older random LightGBM results are not the final selected results.

To run one dataset at a time, use these commands instead of the combined one:

```text
python run_final_analysis.py --stage reproduce --dataset stats19 --workspace ../final-reconstruction/stats19
python run_final_analysis.py --stage reproduce --dataset cas --workspace ../final-reconstruction/cas
```

With `--dataset all`, the runner appends each dataset name; with a single
dataset, `--workspace` is that dataset's directory itself. Do not launch both
forms concurrently or reuse an existing directory without checking its identity.

### Resume and verify

Resume the same combined workspace with:

```text
python run_final_analysis.py --stage reproduce --dataset all --workspace ../final-reconstruction --resume
```

For a single dataset, keep its original command and workspace and add
`--resume`. Completed stages are skipped after input verification. An interrupted
stage is not guaranteed to resume internally; inspect its log and partial-output
guards rather than deleting seals or editing the stage ledger. The current
runner sets up to eight threads internally and a four-hour timeout per stage.
It does not expose the historical runner's `--stop-after` or `--base-python`
options. `--use-current-environment` is an engineering diagnostic, not a
clean-environment reconstruction.

Each dataset must finish with `FINAL_RESULT_COMPARISON=PASS` and its own
`FINAL_REPRODUCTION_COMPLETE stats19` or `FINAL_REPRODUCTION_COMPLETE cas`.
In each generated dataset directory, inspect:

- `logs/final_reproduction_state.log`: `COMPLETE`, the expected stage count and
  `clean_environment: true`;
- `logs/final_result_comparison.json`: 11 STATS19 or nine CAS table comparisons;
- `logs/public_environment_ready.json` and `logs/final_stage_logs/`: this run's
  interpreter identity and execution logs.

The 20 V2 comparisons check text/integers exactly and floating values with
`atol=1e-10`, `rtol=1e-8`. Investigate failures without widening tolerances or
replacing references. The source archive's old acceptance records do not certify
your new run. See [the author validation record](docs/FINAL_REPRODUCTION_VALIDATION.md)
for the applicable tests and historical exclusions.

### Reproduce the additional QWK analysis

After STATS19 completes, run from the v1.3.1 source directory using an environment
with the locked dependencies, such as the regenerated STATS19 environment:

```text
python reproduce_qwk_sensitivity.py --parent ../final-reconstruction/stats19 --workspace ../qwk-reconstruction
```

This separately regenerates and compares three QWK tables from your completed
STATS19 predictions without model fitting or new prediction calls. Its
`logs/qwk_reproduction/acceptance.json` must report
`PASS_QWK_POSTPROCESSING_REPRODUCTION`. Add `--resume` only for that same prepared
QWK workspace. Details and uncertainty boundaries are in
[QWK_PREVALENCE_SENSITIVITY.md](docs/QWK_PREVALENCE_SENSITIVITY.md).

The workflow verifies computational outputs; it does not recreate every later
manual Origin/Visio or Word layout change. No model rerun is needed solely for
this documentation clarification.

## Historical workflows only

> The commands and completed-run evidence in the remaining sections
> describe the archived 17-feature STATS19 implementation and original CAS
> random reference. They do not reproduce the current manuscript. Use
> [the current workflow above](#current-manuscript-workflow-v131) unless you
> intentionally need a historical release. The historical success records
> below concern the older implementation.

The historical guide below describes starting from an earlier release. The
STATS19 primary analysis and the New Zealand CAS replication are separate
pipelines because their fields and severity labels are not exchangeable.

### Historical availability

The public STATS19 runner has completed an author-run isolated validation from
the seven fixed annual files through D14 on Windows 11 with a clean Python
3.13.15 environment. All 32 analysis stages, 14 independent checks, 21 compact
scientific comparisons and seven required-figure checks passed. This validates
the workflow on the stated environment but is not evidence of an independent
third-party replication. The exact STATS19 and CAS inputs are now available in
separate immutable Zenodo records. A clean retrieval from those records passed
all nine size and SHA-256 checks on 4 September 2026.

The separate public CAS runner has passed a complete isolated run from the
fixed lossless JSONL snapshot, including its eight independent checks and
compact-result verification. CAS is an independent workflow and is not a
replacement for the STATS19 primary analysis.

The existing `v1.0.2` Zenodo record is an immutable software archive. It
documents the author's completed D16 forensic comparison, but it does not
contain the fixed raw snapshots or the large artifacts needed by that D16
comparator.

### Historical requirements

- Python 3.13; the validated author environment used Python 3.13.15.
- A 64-bit operating system and enough memory for approximately 744,000
  STATS19 records.
- At least 5 GB of free disk space is recommended. The author's complete
  isolated Windows workspace, including its environment, was about 3.0 GB.
- Internet access is needed for the initial dependency and data downloads.

The command-line entry point is cross-platform. Windows 11 is the currently
validated operating system; Linux and macOS must not be claimed as validated
until their clean-release checks have actually passed.

### Historical step 0: Obtain the fixed software release

For an exact reproduction, download and extract the immutable `v1.1.0`
software archive from <https://doi.org/10.5281/zenodo.22303858>. Alternatively,
obtain the same tagged source from GitHub:

```text
git clone https://github.com/Lhj828828/stats19-crash-severity-temporal-eval.git
cd stats19-crash-severity-temporal-eval
git checkout v1.1.0
```

Run the remaining commands from that project root. A later `main` revision
must not be described as an exact reproduction of `v1.1.0`.

### Historical step 1: Obtain and verify the fixed inputs

From the software-project root, inspect the exact input contract:

```text
python download_and_verify_data.py list
```

Download and verify all published inputs directly from their version-specific
Zenodo records:

```text
python download_and_verify_data.py download --dataset all
python download_and_verify_data.py verify --dataset all
```

The STATS19 record is <https://doi.org/10.5281/zenodo.22290566> under OGL
v3.0. The separate CAS record is
<https://doi.org/10.5281/zenodo.22296725> under CC BY 4.0. `verify` must report
`PASS` for all nine files before either analysis starts. Interrupted downloads
are resumed with finite retries, but no file is accepted unless its complete
size and SHA-256 match the manifest. Never replace a fixed snapshot with the
mutable DfT `latest-published-year` file or a fresh CAS API response.

### Historical step 2: Inspect the execution plan

```text
python run_public_reproduction.py --self-test
python run_public_reproduction.py --list-stages
```

The self-test checks the data hashes and confirms that no author model,
prediction, intermediate table, CAS input or D16 comparison artifact enters
the STATS19 workspace.

### Historical step 3: Run STATS19 in an isolated workspace

For this historical workflow only, the command is:

```text
python run_public_reproduction.py --all
```

It creates a sibling directory named `<project>_public_reproduction`, copies
the verified 2018-2024 annual files, creates a clean virtual environment from
`requirements-lock.txt`, and runs the 32 frozen analysis stages followed by
14 independent checks and the compact-result verifier. It does not overwrite
the software clone.

To select another workspace or base Python executable:

```text
python run_public_reproduction.py --all --workspace /path/to/reproduction --base-python /path/to/python3.13
```

The path syntax may be changed for Windows. Scientific settings remain fixed:
the main D10 and D10b LightGBM runs use their documented four-thread cap, and
the exclusion-2020 sensitivity retains its frozen eight-thread setting.

A successful full STATS19 run ends with both markers:

```text
PUBLIC_RESULT_VERIFICATION=PASS
PUBLIC_REPRODUCTION_PIPELINE=COMPLETE
```

For a non-formal local check, an already prepared Python 3.13 environment may
be used:

```text
python run_public_reproduction.py --all --use-current-environment
```

This option is convenient but is not a substitute for the clean-environment
release test.

### Historical step 4: Resume or run only the data-preparation prefix

Every stage is logged and recorded in `logs/public_run_state.json`. After an
interruption, resume passed stages without rerunning them:

```text
python run_public_reproduction.py --all --resume
```

For inspection before model fitting:

```text
python run_public_reproduction.py --all --stop-after D7_training_inputs
python run_public_reproduction.py --all --resume
```

Do not edit the state file to skip a failed stage. Read the corresponding file
under `logs/public_stage_logs/`, correct the documented environmental or input
problem, and then resume.

### Historical step 5: Outputs and verification boundary

Generated models, record-level predictions, bootstrap arrays, SHAP arrays and
intermediate datasets remain inside the isolated workspace and are not Git
artifacts. Compact tables and decision summaries are the public scientific
verification surface.

The runner finishes by comparing 21 compact CSV/JSON outputs with the tracked
references under `config/public_result_reference/` and by checking that seven
required figures are nonempty. The comparison aligns CSV rows by declared
keys, ignores only the runtime/provenance fields listed in
`config/public_result_contract.json`, and excludes the documented secondary
Ordered Logit drift from pass/fail. Identity/count fields and discrete
decisions are checked exactly; numerical fields use the frozen absolute and
relative tolerances.

The machine-readable report is written to
`logs/public_result_verification.json` in the isolated workspace. It can also
be rerun from the software-project root:

```text
python verify_public_results.py --candidate-root /path/to/reproduction
```

Byte-identical floating-point files are not promised across unvalidated
platforms. A failed comparison must be investigated; the contract tolerances
must not be widened after seeing a new result merely to obtain a pass.

The retained `run_d16_reproduction.py` is an author-side forensic comparator.
It is historical evidence, not an alternative public entry point, because it
depends on excluded author artifacts.

### Historical step 6: CAS replication

CAS uses its own fixed 2022-2025 snapshot, audit, models, one-time 2025
evaluation and post-hoc feature-ablation sensitivity checks. Do not put CAS
records into the STATS19 workspace or compare the
two datasets as though their labels were identical. The public CAS entry and
verification command have passed an isolated clean run. The exact input is
the lossless JSONL file below, whose SHA-256 is checked before execution:

```text
data/raw/cas/cas_injury_2022_2025_snapshot.jsonl.gz
SHA-256: 7db99dd4ba92716d751dabbc08b03f72373c635025b7d5335cf0b6705a7bd7f3
```

From the software-project root, run:

```text
python run_cas_public_reproduction.py --self-test
python run_cas_public_reproduction.py --all --snapshot /path/to/cas_injury_2022_2025_snapshot.jsonl.gz
python verify_cas_public_results.py --candidate-root /path/to/cas-reproduction
```

The runner creates an external isolated workspace, a clean Python 3.13
environment by default, and all rebuildable models, predictions, Bootstrap
arrays and SHAP arrays inside that workspace. It does not query the live CAS
API or copy author-side artifacts. After the frozen 2025 evaluation, it binds
the post-hoc feature-ablation protocol to the regenerated upstream files and
runs both feature-removal scenarios. See `CAS_PUBLIC_REPRODUCTION.md` for the
nine standalone checks and the required completion markers.

The CAS result supports workflow executability and separately reported
directional agreement or disagreement only. It does not establish
cross-national or universal generalizability.

See `DATA_SOURCES.md` for source attribution and
`docs/REPRODUCIBILITY_BOUNDARY.md` for the exact distinction between public
reconstruction, compact scientific verification and author-side forensics.
The dated author-run validation record is in
`docs/PUBLIC_REPRODUCTION_VALIDATION.md`.
