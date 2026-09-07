# v1.3.0: Corrected analysis and verified clean reconstruction

Release date: 2026-09-07

This version supplies the corrected computational scope for the Applied Sciences
study. It supersedes earlier results for the current analysis without moving old
tags, overwriting historical scientific artifacts or changing the input archives.

## Corrected scope

- STATS19 uses 15 features, excluding `special_conditions_at_site` and
  `carriageway_hazards`, with six split-local model sets and eleven evaluation
  cohorts. Models are Dummy, unweighted/weighted multinomial Logistic and
  weighted LightGBM.
- Random-reference LightGBM selection uses each split's own training/validation
  records, correcting overlap in the historical shared-selection procedure.
- The fixed-parameter exclusion-2020 sensitivity is retained. Ordered Logit,
  matched-subset comparisons and tree-count extension are omitted from the
  corrected primary scope; their audit history remains available.
- CAS retains its independently audited temporal workflow and post-hoc urban/
  sparse-speed removals, followed by corrected split-local random references.
- The V2 catalog selects 20 compact result tables. CAS H2 exports only nine
  temporal-test contrasts; 90 superseded random contrasts remain in the preserved
  V1 history, not the current CAS H2 export. No scientific source result was
  changed by this export-only correction.

## Validation

Author-run reconstruction in two new Windows/Python 3.13.15 virtual environments
completed all 22 STATS19 and 25 CAS stages. All 20 selected V2 result comparisons
passed with the frozen tolerances (`atol=1e-10`, `rtol=1e-8`); exact fields were
compared exactly. No fitted models, processed data, predictions or SHAP arrays
were copied. Nine existing raw snapshot files passed size/SHA-256 verification;
they were reused rather than freshly downloaded in this run.

There were 159 passing applicable unit tests and seven passing standalone CAS
audits. The two original historical D14 exact-provenance assertions remain
separate documented exceptions, not relabeled as passing tests. The independent
bounded historical check passed. See `docs/FINAL_REPRODUCTION_VALIDATION.md` and
`logs/final_analysis/final_reproduction_acceptance.json` for evidence and scope.
This is author-run same-machine validation, not third-party or cross-platform
validation. Manuscript layout was not part of this acceptance.

The release preserves the original bytes of newly sealed protocols, result
tables and acceptance records through explicit Git attributes, preventing
automatic line-ending conversion from invalidating their recorded hashes.

## Reproduce this version

Use Python 3.13 from the extracted release source directory:

```text
python -m pip install --requirement requirements-lock.txt
python download_and_verify_data.py download --dataset all
python download_and_verify_data.py verify --dataset all
python run_final_analysis.py --stage reproduce --dataset all --workspace ../final-reconstruction
```

To continue an interrupted run, repeat the last command with `--resume`.
`--stage check` is an author-side audit requiring existing analytical artifacts;
it is not the first command for a fresh clone. See `FINAL_ANALYSIS.md`.

## Interpretation and archive boundaries

Corrections were made after historical outcomes were known; they are not a new
untouched-test experiment or preregistration. H1 remains metric-dependent;
Macro-F1 gains do not establish uniformly safer LightGBM performance. Failed
safety constraints, the 1200-round search limit, label-recording changes,
unverified spatial generalization and dataset-specific SHAP designs remain
explicit limitations. SHAP is descriptive, not causal.

Raw records, models and record-level outputs are excluded from the software
archive. Fixed data archives are unchanged:

- STATS19: <https://doi.org/10.5281/zenodo.22290566> (OGL v3.0).
- CAS: <https://doi.org/10.5281/zenodo.22296725> (CC BY 4.0).

The software concept DOI is <https://doi.org/10.5281/zenodo.22231696>.
The version-specific software DOI is <https://doi.org/10.5281/zenodo.22642853>.
Its published ZIP contains 808 files, all verified byte-for-byte against
commit `962377ca6cbd0ca12c06bb801aaaf531b3c62ce7`. This DOI was backfilled in
the GitHub release description and main-branch citation metadata after archival.
The released tag is unchanged; its initial citation file necessarily predates
the DOI assignment. No new scientific result is introduced by the backfill.
