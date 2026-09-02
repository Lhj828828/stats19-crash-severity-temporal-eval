# Time-aware crash-severity modelling: STATS19 and CAS

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22231696.svg)](https://doi.org/10.5281/zenodo.22231696)

This repository contains the frozen computational materials for the IEEE
Access manuscript on leakage-aware temporal generalization in collision
severity classification. It combines the UK STATS19 primary analysis with a
separately trained New Zealand Crash Analysis System (CAS) replication.

IEEE Access is a fully open-access journal. An article processing charge and
the current IEEE Access author requirements should be checked before
submission; the journal choice does not change the frozen analyses in this
repository.

## Scope and status

### STATS19 primary analysis

- Statistical unit: one police-reported personal-injury collision.
- Analysis years: 2018-2024.
- Primary deployment-oriented test: the untouched 2024 temporal cohort.
- Reference analysis: five stratified random partitions.
- Features: 17 collision-time observable fields after the D3 leakage audit.
- Models: Dummy, weighted multinomial Logistic and weighted LightGBM.
- Ordered Logit: secondary matched-subset baseline.

### CAS independent replication

- Snapshot: 43,121 injury crashes from 2022-2025.
- Features and severity coding are dataset-specific and were audited
  independently.
- The 2025 cohort was evaluated once after model freezing.
- CAS records were never pooled with STATS19 records.
- The CAS analysis tests workflow executability and directional agreement or
  disagreement only. It does not establish cross-national generalizability.

## Verification boundary

D16 completed an isolated raw-to-results reproduction of the STATS19 workflow
with status `D16_CORE_REPRODUCTION_PASS_WITH_DOCUMENTED_NUMERICAL_DEVIATIONS`.
The primary Logistic/LightGBM chain and scientific decision outputs passed the
frozen checks. Nineteen numerical deviations remain in the secondary Ordered
Logit path, propagated bootstrap outputs and one probability-only four-thread
sensitivity result. They are documented in `logs/d16_checkpoint.md`; no
tolerance was widened to hide them.

The CAS closeout and all eight CAS independent checks passed. The fatal-recall
trade-off direction agreed with STATS19, while the Macro-F1 gain and SHAP-rank
pattern did not reproduce. These results are reported as directional evidence,
not as universal validation.

## Data and provenance

Raw data are intentionally excluded from Git. The planned separate data
archive contains seven annual STATS19 analysis files plus the fixed CAS JSONL
and inspection CSV, approximately 146 MB in total. It does not need the 1.53 GB
all-years STATS19 source. Exact provenance, paths, byte sizes and SHA-256
checksums are recorded in `DATA_SOURCES.md` and
`config/public_data_manifest.json`.

Download the required inputs into the paths documented in `DATA_SOURCES.md`
before running a fresh pipeline. Do not commit access tokens, raw data copied
under different names, or unreviewed local files.

Inspect or verify the fixed public input contract with:

```text
python download_and_verify_data.py list
python download_and_verify_data.py verify --dataset all
```

Immutable download URLs will be enabled only after the separate data archive
passes its final source-terms review. Mutable upstream data are never accepted
as silent replacements for the paper snapshots.

## Environment and reproduction

The validated environment used Python 3.13.15. Direct and serialized-model
dependencies are pinned in `requirements-lock.txt`.

The public STATS19 entry point creates an isolated sibling workspace and, by
default, a clean environment:

```text
python run_public_reproduction.py --self-test
python run_public_reproduction.py --all
```

Resume an interrupted run with:

```text
python run_public_reproduction.py --all --resume
```

The public runner starts from the seven verified annual files, recreates the
D1-compatible input audit, and does not copy existing models, predictions,
intermediate data or D16 artifacts. It finishes with keyed, tolerance-aware
comparison of compact scientific outputs against the tracked public
references. The development version has been exercised through D7;
a clean D8-D14 run and immutable data URLs remain `v1.1.0` release gates. Detailed
commands and current limitations are in `REPRODUCING.md`.

The retained author-side forensic entry point is:

```text
python run_d16_reproduction.py --all
```

On Windows, the convenience wrapper is:

```powershell
.\run_d16_reproduction.ps1
```

It requires the 1.53 GB frozen source and large author artifacts that are not
part of the public archives. Therefore it documents the original forensic
comparison but is not a command for third-party use.

For the retained STATS19 audit and independent checks:

```text
python code/d15_reproducibility.py
python code/test_d6_protocol.py
python code/test_d7_training_inputs.py
python code/test_d8_baselines.py
python code/test_d9_ordered_logit.py
python code/test_d10_lightgbm.py
python code/test_d11_evaluation.py
python code/test_d12_bootstrap.py
python code/test_d13_checkpoint.py
python code/test_d14_shap.py
python code/test_d14_exclude2020.py
python code/test_d14_closeout.py
python code/test_environment.py
```

For the CAS replication, run the standalone checks in `tests/`, for example:

```text
python tests/test_cas_feasibility_audit.py
python tests/test_cas_modeling_inputs.py
python tests/test_cas_baselines.py
python tests/test_cas_lightgbm.py
python tests/test_cas_evaluation.py
python tests/test_cas_bootstrap.py
python tests/test_cas_shap.py
```

The completed one-time evaluation scripts intentionally refuse to overwrite
their frozen outputs. Use the isolated reproduction entry point for a clean
rerun rather than changing a test result in place.

## Project map

- `code/`: STATS19 stages, CAS stages and standalone checks.
- `config/`: frozen schemas, field audits, split protocols and model rules.
- `config/public_result_reference/`: small frozen CSV/JSON summaries used by
  the public verifier; no records, fitted models or large arrays.
- `data/external/`: official documentation and CAS metadata snapshots.
- `figures/`: compact figures retained for the manuscript.
- `logs/`: provenance, checksums, audit trails and checkpoints.
- `results/`: compact summary tables; record-level outputs are excluded by
  default.
- `DATA_SOURCES.md`: source attribution, snapshot hashes and data layout.
- `REPRODUCING.md`: public reconstruction commands, resume behavior and
  verification limits.
- `verify_public_results.py`: cross-platform compact-result verification
  command.
- `RELEASE_NOTES_v1.0.0.md`: the first public version boundary.
- `RELEASE_NOTES_v1.0.1.md`: metadata-only Zenodo archival follow-up to the
  frozen `v1.0.0` materials.

## Interpretation boundaries

- H1 is metric-dependent; no uniform random-split optimism claim is made.
- H2 is not a claim of uniformly safer or better LightGBM performance; the
  Macro-F1 gain coexists with a fatal-recall reduction.
- H3 is descriptive SHAP rank analysis, not causal attribution.
- A police-force regional holdout was not executed because its grouping was
  not frozen before outcomes were known. Spatial generalization is unverified.
- STATS19 and CAS labels, fields, absolute metrics and SHAP ranks are not
  treated as exchangeable.

## Release materials

The repository-level release checklist is in
`docs/IEEE_ACCESS_RELEASE_CHECKLIST.md`. The original `v1.0.0` tag remains
immutable; `v1.0.1` corrects the formal author metadata, and `v1.0.2` points
to the same frozen commit and provides the successful Zenodo archival event.
No analytical code, data-processing rule, model output or study conclusion
changed between these two tags.

The exact archived `v1.0.2` release is available at
<https://doi.org/10.5281/zenodo.22231697>. The concept DOI
<https://doi.org/10.5281/zenodo.22231696> represents all versions and resolves
to the latest Zenodo archive. Cite the version DOI when referring to the
computational materials used for this study.
