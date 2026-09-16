# Time-aware crash-severity modelling: STATS19 and CAS

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22231696.svg)](https://doi.org/10.5281/zenodo.22231696)

This repository contains the frozen computational materials for the Traffic
Injury Prevention manuscript on leakage-aware temporal generalization in collision
severity classification. It combines the UK STATS19 primary analysis with a
separately trained New Zealand Crash Analysis System (CAS) replication.

The current manuscript target is Traffic Injury Prevention. Journal-specific
requirements must be checked again at submission; the journal target does not
change the frozen analyses in this repository.

## Scope and status

### Additive QWK sensitivity (16 September 2026, v1.3.1)

The v1.3.1 source adds the post hoc QWK true-class-prevalence sensitivity for
Appendix A.14.6 and Table A13. It contains the calculation code, recorded protocol,
tests, compact reference outputs and a separate reproduction entry. The final
15-feature primary results, model rules, data archives and v1.3.0 tag remain
unchanged. This was specified after the primary results were known, not
preregistered. The historical v1.3.0 DOI below does not include this addition.

After completing the STATS19 reconstruction, run from this source directory:

```text
python reproduce_qwk_sensitivity.py --parent ../final-reconstruction/stats19 --workspace ../qwk-reconstruction
```

This creates a separate postprocessing workspace, verifies parent inputs,
recalculates QWK without model fitting or prediction calls, and compares its
three summary tables against the recorded references. See
[QWK_PREVALENCE_SENSITIVITY.md](docs/QWK_PREVALENCE_SENSITIVITY.md) for provenance,
resumption, uncertainty and archive boundaries. Publication/DOI verification of
this source version is recorded separately from local preparation.

### Current corrected analysis (7 September 2026)

The current analysis uses the **15-feature STATS19 correction**, including all
six split-local model sets and the fixed-parameter exclusion-2020 sensitivity.
Ordered Logit, its matched-subset comparison and the tree-count extension are
omitted from the revised manuscript. Their historical files are retained.
The 1200-round search boundary remains a limitation and must still be reported.

The current entry point is explicit and defaults to a plan, not model fitting:

```text
python run_final_analysis.py --stage plan
python run_final_analysis.py --stage check
```

`check` is an author-side audit of existing completed artifacts, not a fresh
clone's installation test. First-time reproduction uses `--stage reproduce`
as described under Environment and reproduction below.

`results/final_analysis/v2/` contains 20 curated compact tables. The catalog at
`config/final_analysis/result_catalog.json` fixes their sources, keys, hashes
and the choice of corrected CAS random-reference results. It excludes the old
17-feature STATS19 tables and the superseded CAS random LightGBM estimates.
Catalog V2 restricts CAS H2 to the nine temporal-test contrast rows. V1 had
also exported 90 historical random-reference contrasts; its catalog and tables
are preserved for audit, not recommended as the current reporting source.
An **author-run clean raw-to-results reconstruction passed on 7 September 2026**:
22 STATS19 stages, 25 CAS stages and all 20 selected V2 result-table comparisons.
Separate new virtual environments were used; no fitted models or predictions
were copied. Verified fixed raw files were reused, not freshly downloaded.
This is same-machine Windows validation, not independent third-party or
cross-platform validation. See
[the current validation record](docs/FINAL_REPRODUCTION_VALIDATION.md).
The historical public-validation records below describe older implementations.
See [FINAL_ANALYSIS.md](FINAL_ANALYSIS.md) for the separate reconstruction
command, provenance boundaries and the remaining manuscript/release work.
The corrected source snapshot is published as `v1.3.0` at
<https://doi.org/10.5281/zenodo.22642853>, from commit
`962377ca6cbd0ca12c06bb801aaaf531b3c62ce7`. All 808 files downloaded from this
archive matched that Git tree byte-for-byte. Do not use an earlier version DOI
for this corrected analysis. Main-branch DOI backfill does not move the tag.

### Post-review random-reference correction

A methods audit on 6 September 2026 identified selection overlap in the
archived random-reference LightGBM branch: the temporal validation cohort
used to select shared hyperparameters contains some random internal-test
records. The archived random internal tests are therefore not independent of
the complete upstream selection procedure. This does not involve the 2024
STATS19 or 2025 CAS future cohorts in that upstream selection.

A separately versioned, post-review correction now selects LightGBM settings
within each random split's own training and validation partitions. It retains
the original six candidates, training-only preprocessing and weighting,
split-specific Logistic validation constraints, and documented fallback.
That initial correction did not replace temporal models or H2 results. The
subsequent 15-feature STATS19 correction now supplies the manuscript's STATS19
temporal and random results; historical files are still not overwritten.
The original outcomes were already known; this is not preregistration or a
newly untouched-test experiment. See
`docs/RANDOM_REFERENCE_REVISION.md` for execution, status checks and limitations.
The immutable `v1.2.0` DOI below describes the historical implementation; it
does not yet archive this correction. Do not cite it as containing these new
results.

The author-side correction was completed and verified on 6 September 2026:
ten split-local models, twenty evaluation cohorts, 2,000-draw bootstrap and
SHAP analyses. Both `logs/random_reference_revision/<dataset>/complete.json`
markers passed; 553 protected historical files were unchanged. An additional
audit recomputed 1,300 intervals and all SHAP ranks. STATS19 hard predictions
were unchanged; CAS random-reference predictions changed and the revised
outputs supersede the old random analysis regardless of direction. This is
local verification, not a new third-party or clean-environment replication.
See the correction document for the two historical provenance-check exceptions
in the broader regression run and the separate passing scientific contract.

### STATS19 primary analysis

- Statistical unit: one police-reported personal-injury collision.
- Analysis years: 2018-2024.
- Primary deployment-oriented test: 2024, held out from original temporal
  model development; known outcomes are disclosed for the post-review correction.
- Reference analysis: five stratified random partitions.
- Features: 15 retained fields after the post-review field audit; both
  `special_conditions_at_site` and `carriageway_hazards` are excluded.
- Models: Dummy, unweighted/weighted multinomial Logistic and weighted LightGBM.
- Sensitivity: exclusion of 2020 from temporal training, with fixed main-model
  hyperparameters and freshly fitted preprocessing/weights.
- Labels: police-recorded severity. Recording-system changes remain a
  comparability limitation even though recording metadata are not features.

### CAS independent replication

- Snapshot: 43,121 injury crashes from 2022-2025.
- Features and severity coding are dataset-specific and were audited
  independently.
- The original 2025 evaluation followed model freezing; post-hoc sensitivity
  and corrected random-model evaluations are separately identified.
- CAS records were never pooled with STATS19 records.
- A post-hoc sensitivity analysis refits the frozen temporal design after
  removing urban and the two sparse speed fields separately. It does not
  select or replace the primary model.
- The CAS analysis tests workflow executability and directional agreement or
  disagreement only. It does not establish cross-national generalizability.

## Verification boundary

The historical D16 completed an isolated raw-to-results reproduction of the STATS19 workflow
with status `D16_CORE_REPRODUCTION_PASS_WITH_DOCUMENTED_NUMERICAL_DEVIATIONS`.
The primary Logistic/LightGBM chain and scientific decision outputs passed the
frozen checks. Nineteen numerical deviations remain in the secondary Ordered
Logit path, propagated bootstrap outputs and one probability-only four-thread
sensitivity result. They are documented in `logs/d16_checkpoint.md`; no
tolerance was widened to hide them.

The historical CAS closeout and all nine CAS independent checks passed. Its
cross-dataset wording compared CAS with the then-current STATS19 version and
must not be copied as a comparison with the new 15-feature STATS19 results.
Use the final catalog for updated comparisons; SHAP rho values across the two
different sampling/feature designs are not directly comparable.
The two feature-ablation checks are explicitly
post-hoc sensitivity evidence: removing `urban` leaves the main comparison
nearly unchanged, whereas removing the sparse speed fields changes selected
safety metrics and increases the mean asymmetric cost for both refitted
models. These results are reported as directional evidence, not as universal
validation.

## Data and provenance

Raw data are intentionally excluded from Git. The fixed inputs are published
as two independent Zenodo records because their licences differ:

- STATS19 2018-2024 snapshots, OGL v3.0:
  <https://doi.org/10.5281/zenodo.22290566>;
- CAS 2022-2025 injury-crash snapshot, CC BY 4.0:
  <https://doi.org/10.5281/zenodo.22296725>.

Together, the seven annual STATS19 files and two CAS files are approximately
146 MB. The 1.53 GB all-years STATS19 source is not required by the public
workflow. Exact provenance, paths, byte sizes and SHA-256 checksums are
recorded in `DATA_SOURCES.md` and `config/public_data_manifest.json`.

Inspect or verify the fixed public input contract with:

```text
python download_and_verify_data.py list
python download_and_verify_data.py download --dataset all
python download_and_verify_data.py verify --dataset all
```

The STATS19 and CAS source-terms reviews are complete under OGL v3.0 and CC BY
4.0, respectively. The immutable records were published and their deposited
bytes downloaded by the author and verified 9/9 against the frozen manifest
on 4 September 2026. Mutable upstream data are never accepted as silent
replacements for the paper snapshots.

## Environment and reproduction

The validated environment used Python 3.13.15. Direct and serialized-model
dependencies are pinned in `requirements-lock.txt`.

### Current 15-feature workflow

Download and extract the `v1.3.0` source archive from
<https://doi.org/10.5281/zenodo.22642853>, or clone the repository and check out
`v1.3.0` to reproduce the primary analysis alone. Use the additive `v1.3.1`
source when also reproducing Table A13, following the QWK section above.
From that source directory, use Python 3.13 in a working environment:

```text
python -m pip install --requirement requirements-lock.txt
python download_and_verify_data.py download --dataset all
python download_and_verify_data.py verify --dataset all
python run_final_analysis.py --stage reproduce --dataset all --workspace ../final-reconstruction
```

The last command creates separate clean environments for STATS19 and CAS,
rebuilds from verified raw inputs, and compares compact results without copying
author-fitted models or predictions. Resume in the same workspace with:

```text
python run_final_analysis.py --stage reproduce --dataset all --workspace ../final-reconstruction --resume
```

The corrected source snapshot is `v1.3.0`, DOI
<https://doi.org/10.5281/zenodo.22642853>. The old tags and DOIs below do not
contain this entry.
See [FINAL_ANALYSIS.md](FINAL_ANALYSIS.md) for scope and current verification
status. The author-run clean reconstruction passed against all 20 V2 reference
tables on 7 September 2026. Manuscript editing remains separate from this check.

### Historical v1.2.0 workflow

The following commands reproduce the **historical 17-feature release**, not
the current corrected manuscript analysis. For that historical snapshot, use the immutable
`v1.2.0` software archive at <https://doi.org/10.5281/zenodo.22341639>, or
check out the matching Git tag:

```text
git clone https://github.com/Lhj828828/stats19-crash-severity-temporal-eval.git
cd stats19-crash-severity-temporal-eval
git checkout v1.2.0
```

Do not use a later `main` revision while claiming an exact reproduction of
`v1.2.0`. The earlier `v1.1.0` boundary remains available at
<https://doi.org/10.5281/zenodo.22303858>.

The historical public STATS19 entry point creates an isolated sibling workspace and, by
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
references. An author-run isolated Windows/Python 3.13 validation completed
all D1-D14 stages, all 14 independent checks and all 21 compact-result
comparisons on 2026-09-03. This is workflow validation, not an independent
third-party replication. The published data URLs and hashes passed a separate
clean retrieval check on 2026-09-04. Detailed commands, evidence and
limitations are in `REPRODUCING.md` and
`docs/PUBLIC_REPRODUCTION_VALIDATION.md`.

A successful full STATS19 reconstruction ends with both markers:

```text
PUBLIC_RESULT_VERIFICATION=PASS
PUBLIC_REPRODUCTION_PIPELINE=COMPLETE
```

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

For the retained historical STATS19 audit and independent checks (not the final
revision entry point):

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
python code/historical_d14_validation.py
python code/test_environment.py
```

The two original D14 exact-protocol-equality tests remain unchanged for their
original snapshot. For the later path-normalized historical snapshot, the
separate historical validator checks only explicitly reviewed provenance
migrations and recomputes the scientific outputs. It does not relabel the
original failures or overwrite their hashes. Corrected 15-feature runs use
their own execution-bound validators, not this historical compatibility check.

For the CAS replication, run the standalone checks in `tests/`, for example:

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

The independent CAS public entry point is also available. It uses only the
fixed lossless JSONL snapshot, never queries the live CAS service, and checks
the snapshot SHA-256 before copying it to an isolated workspace:

```text
python run_cas_public_reproduction.py --self-test
python run_cas_public_reproduction.py --all --snapshot /path/to/cas_injury_2022_2025_snapshot.jsonl.gz
python verify_cas_public_results.py --candidate-root /path/to/cas-reproduction
```

The expected snapshot hash and the complete CAS command sequence are in
`CAS_PUBLIC_REPRODUCTION.md`. CAS is a separately trained and evaluated
workflow; its compact results are not pooled with STATS19 results. The
`--all` public entry point also binds the post-hoc feature-ablation protocol to
the regenerated upstream files, runs its smoke and sensitivity stages, and
checks the resulting completion record. The binding changes only
run-specific file hashes; it does not change the frozen feature-removal
scenarios, model settings or interpretation boundary.

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
- `RELEASE_NOTES_v1.1.0.md`: published fixed-input records and verified public
  download support.
- `RELEASE_NOTES_v1.2.0.md`: post-hoc CAS feature-ablation sensitivity checks
  and their public reproduction entry.
- `RELEASE_NOTES_v1.3.0.md`: 15-feature correction, split-local selection and
  the verified final clean reconstruction.

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

The additive QWK release is described in `docs/RELEASE_v1.3.1.md`.
The older `docs/APPLIED_SCIENCES_RELEASE_CHECKLIST.md` is a historical checklist,
not the current journal's author instructions. The original `v1.0.0`-`v1.0.2`
release materials retain the historical IEEE Access target recorded when those
versions were prepared. The corrected parent computational supplement was
published as `v1.3.0`, DOI <https://doi.org/10.5281/zenodo.22642853>, described
in `RELEASE_NOTES_v1.3.0.md`. It replaces the
selected analysis scope, not the historical tags or their archived artifacts.
The `v1.2.0` release added CAS feature-ablation sensitivity to the older analysis.

The exact archived `v1.2.0` release is available at
<https://doi.org/10.5281/zenodo.22341639>. It adds the CAS sensitivity analysis
and its clean-workspace binding. The concept DOI
<https://doi.org/10.5281/zenodo.22231696> represents all versions and resolves
to the latest Zenodo archive. The previous `v1.1.0` archive remains available
at <https://doi.org/10.5281/zenodo.22303858>, and the previous `v1.0.2` archive
remains available at <https://doi.org/10.5281/zenodo.22231697>. Cite the version
DOI matching the software materials used for the study.
