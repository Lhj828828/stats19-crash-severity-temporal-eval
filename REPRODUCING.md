# Reproducing the analyses

This guide is for a researcher starting from a fresh software release. The
STATS19 primary analysis and the New Zealand CAS replication are separate
pipelines because their fields and severity labels are not exchangeable.

## Current availability

The public STATS19 runner is present on the development branch and has been
tested from the fixed annual files through D7 (feature audit, quality control
and frozen splits). A full clean D8-D14 run and the immutable data-record URLs
remain release gates for `v1.1.0`. Until those gates pass, do not describe the
development branch as a completed third-party reproduction package.

The existing `v1.0.2` Zenodo record is an immutable software archive. It
documents the author's completed D16 forensic comparison, but it does not
contain the fixed raw snapshots or the large artifacts needed by that D16
comparator.

## Requirements

- Python 3.13; the validated author environment used Python 3.13.15.
- A 64-bit operating system and enough memory for approximately 744,000
  STATS19 records.
- At least 5 GB of free disk space is recommended. The author's complete
  isolated Windows workspace, including its environment, was about 3.0 GB.
- Internet access is needed for the initial dependency and data downloads.

The command-line entry point is cross-platform. Windows 11 is the currently
validated operating system; Linux and macOS must not be claimed as validated
until their clean-release checks have actually passed.

## 1. Obtain and verify the fixed inputs

From the software-project root, inspect the exact input contract:

```text
python download_and_verify_data.py list
```

After the separate immutable data record is published and its URLs are added
to `config/public_data_manifest.json`, download STATS19 and verify every file:

```text
python download_and_verify_data.py download --dataset stats19
python download_and_verify_data.py verify --dataset stats19
```

Until those immutable URLs exist, `download` deliberately fails. A researcher
who already possesses the exact annual files may place them under
`data/raw/collisions/`; `verify` must report `PASS` for all seven files before
the analysis starts. Never replace the fixed snapshot with the mutable DfT
`latest-published-year` file merely because its filename is similar.

## 2. Inspect the execution plan

```text
python run_public_reproduction.py --self-test
python run_public_reproduction.py --list-stages
```

The self-test checks the data hashes and confirms that no author model,
prediction, intermediate table, CAS input or D16 comparison artifact enters
the STATS19 workspace.

## 3. Run STATS19 in an isolated workspace

The recommended command is:

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

For a non-formal local check, an already prepared Python 3.13 environment may
be used:

```text
python run_public_reproduction.py --all --use-current-environment
```

This option is convenient but is not a substitute for the clean-environment
release test.

## 4. Resume or run only the data-preparation prefix

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

## 5. Outputs and verification boundary

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

## 6. CAS replication

CAS uses its own fixed 2022-2025 snapshot, audit, models and one-time 2025
evaluation. Do not put CAS records into the STATS19 workspace or compare the
two datasets as though their labels were identical. A separate CAS public
entry and verification command must pass before `v1.1.0`; until then, the
tracked CAS scripts and compact summaries support inspection but not a fresh
snapshot-independent execution claim.

See `DATA_SOURCES.md` for source attribution and
`docs/REPRODUCIBILITY_BOUNDARY.md` for the exact distinction between public
reconstruction, compact scientific verification and author-side forensics.
