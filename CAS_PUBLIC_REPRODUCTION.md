# Public CAS reproduction

This is an independent New Zealand CAS workflow reproduction. CAS records are
never pooled with STATS19 records, and the CAS labels and absolute metrics are
not treated as interchangeable with STATS19 labels and metrics.

## Fixed input

The public run starts from the lossless JSONL snapshot below:

```text
data/raw/cas/cas_injury_2022_2025_snapshot.jsonl.gz
```

Expected SHA-256:

```text
7db99dd4ba92716d751dabbc08b03f72373c635025b7d5335cf0b6705a7bd7f3
```

The snapshot contains 43,121 police-reported personal-injury crashes from
2022-2025. The repository does not substitute a live CAS query for this fixed
input. Place the exact file at the path above, or provide another local path
with `--snapshot`; the hash check remains mandatory.

## Run

Python 3.13 is required. From the repository root:

```text
python run_cas_public_reproduction.py --self-test
python run_cas_public_reproduction.py --list-stages
python run_cas_public_reproduction.py --all
```

The runner creates the sibling directory
`STATS19论文_cas_public_reproduction`, creates a clean virtual environment by
default, and executes the offline audit, deterministic data preparation,
frozen temporal/random splits, baseline models, LightGBM, the one-time 2025
evaluation, Bootstrap uncertainty, SHAP stability, CAS directional closeout,
independent tests and compact-result verification.

To use a snapshot outside the default input path:

```text
python run_cas_public_reproduction.py --all --snapshot /path/to/cas_injury_2022_2025_snapshot.jsonl.gz
```

For a non-formal local check with an already prepared Python 3.13 environment:

```text
python run_cas_public_reproduction.py --all --use-current-environment
```

On Windows, the equivalent convenience entry is:

```powershell
.\run_cas_public_reproduction.ps1
```

An interrupted run can be resumed without rerunning passed stages:

```text
python run_cas_public_reproduction.py --all --resume
```

## Verification

The runner must finish with:

```text
CAS_PUBLIC_COMPACT_RESULT_VERIFICATION ...
PUBLIC_RESULT_VERIFICATION=PASS
CAS_PUBLIC_REPRODUCTION_PIPELINE=COMPLETE
```

The compact contract checks 11 small scientific summaries and 10 required
nonempty files. Numerical timing fields are ignored because they depend on
the machine. Models, record-level predictions, Bootstrap arrays and SHAP
arrays are regenerated for the workflow checks but are not required inputs to
the compact verification contract.

The independent CAS tests can also be run individually from a completed
workspace:

```text
python tests/test_cas_feasibility_audit.py
python tests/test_cas_modeling_inputs.py
python tests/test_cas_baselines.py
python tests/test_cas_lightgbm.py
python tests/test_cas_evaluation.py
python tests/test_cas_bootstrap.py
python tests/test_cas_shap.py
python tests/test_cas_closeout.py
```

The resulting `config/cas/cas_workspace_manifest.json` records that the run
used an isolated copy, did not query the live API, and did not copy author
models or large result artifacts.

## Interpretation boundary

The CAS run supports workflow executability and separately reported directional
agreement or disagreement only. It does not prove cross-national,
cross-domain or universal generalizability. It also does not turn SHAP
attribution into causal evidence.
