# Public CAS reproduction

> Historical CAS entry. This guide preserves the archived pipeline and its
> completed-run evidence. Its random LightGBM and cross-dataset reporting stages
> do not supply the current revised manuscript results. Use
> [FINAL_ANALYSIS.md](FINAL_ANALYSIS.md) for the current entry, which retains
> temporal and baseline results and runs the split-local random correction.
> The corrected combination passed author-run clean reconstruction on
> 7 September 2026; see [the validation record](docs/FINAL_REPRODUCTION_VALIDATION.md).

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
evaluation, Bootstrap uncertainty, SHAP stability, the post-hoc feature-ablation
sensitivity checks, CAS directional closeout,
independent tests and compact-result verification. Before the ablation stages,
the runner binds only the hashes of regenerated upstream files to a copy of
the frozen post-hoc protocol. It does not change the feature-removal
scenarios, model settings or primary CAS results.

To use a snapshot outside the default input path:

```text
python run_cas_public_reproduction.py --all --snapshot /path/to/cas_injury_2022_2025_snapshot.jsonl.gz
```

For a non-formal local check with an already prepared Python 3.13 environment:

```text
python run_cas_public_reproduction.py --all --use-current-environment
```

The full --all run includes the two explicitly post-hoc feature-ablation
scenarios:

- drop_urban: remove feature_urban;
- drop_sparse_speed: remove feature_advisory_speed and
  feature_temporary_speed_limit together.

Both scenarios refit the frozen temporal design with weighted multinomial
Logistic and the frozen C06 LightGBM configuration. They are sensitivity
evidence only: they do not retune, select or replace the 15-feature primary
models. The generated outputs remain in the isolated workspace under
results/cas_feature_ablation/; no raw data, fitted models or record-level
prediction files are copied from the author workspace.

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
python tests/test_cas_feature_ablation.py
```

The resulting config/cas/cas_workspace_manifest.json records that the run
used an isolated copy, did not query the live API, did not copy author models
or large result artifacts, and completed the post-hoc feature-ablation
scenarios without modifying the primary analysis.

## Interpretation boundary

The CAS run supports workflow executability and separately reported directional
agreement or disagreement only. It does not prove cross-national,
cross-domain or universal generalizability. It also does not turn SHAP
attribution into causal evidence.
