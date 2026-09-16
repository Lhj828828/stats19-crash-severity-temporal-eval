# Post hoc QWK prevalence sensitivity

This additive analysis supports Appendix A.14.6 of the Traffic Injury Prevention
manuscript. It was specified on 16 September 2026 after the primary results were
known; it is not preregistered. It does not change the final 15-feature models,
primary test results, result catalog, or the archived v1.3.0 software release.

## Estimand and uncertainty

For each of the five existing STATS19 random splits, compare the same fitted
class-weighted multinomial logistic or LightGBM model on its internal test set
and on 2024. The gap is internal QWK minus future QWK.

Set the target proportions to the observed true-class proportions in that
seed's internal test set. Give each 2024 collision in true class k weight
`p_internal(k) / p_2024(k)`. Recompute QWK from the full weighted confusion
matrix, including its updated predicted and true marginals. Within-class
prediction frequencies are preserved. This does not standardize covariates or
remove all temporal change, and is not a causal decomposition.

The original 2,000 true-class-stratified bootstrap replicates are replayed
exactly, using seed 20260828 and the existing group-specific stream derivation.
Internal and future cohorts use independent streams; models and raw versus
standardized metrics share resamples within each cohort. All four original
prediction columns are retained only to reproduce the original joint-pattern
resampling stream; only the two weighted models are analyzed here.

Reference proportions are fixed. Class counts are fixed within resamples,
so the 95% percentile intervals condition on the fitted models and observed
class proportions. They do not include prevalence-estimation, model-refitting,
or split-selection uncertainty. Five-seed means, sample SDs, and ranges are
descriptive; no pooled interval treats overlapping splits as independent studies.

## Reproduction from a completed main analysis

Use the project environment and the completed final 15-feature outputs. First
follow `FINAL_ANALYSIS.md` if those outputs do not yet exist. From the source
release directory, run:

```text
python -m unittest discover -s tests -p "test_qwk*.py" -v
python reproduce_qwk_sensitivity.py --parent ../final-reconstruction/stats19 --workspace ../qwk-reconstruction
```

Replace the parent path with your completed STATS19 reconstruction directory.
The new workspace must be separate from both the source and parent directories.
The wrapper copies 28 required parent files and the exact sensitivity source
files, creates a new protocol for those reconstructed inputs, calculates QWK,
and compares all three summary CSVs with the archived reference tables. It uses
the existing models' saved predictions, with no model fitting or prediction
calls. It does not alter the parent directory or replace the historical author
protocol. No raw data or fitted model binaries are copied into the QWK workspace.
The copied evaluation arrays remain local and are not distributed in this source
release. Integer and text fields must match exactly; floating comparisons use
the main reconstruction tolerances (`atol=1e-10`, `rtol=1e-8`).

To verify or resume that same workspace, repeat the wrapper command with
`--resume`. A mismatched input, source file or completed output is an error, not
a reason to overwrite a protocol or relax a tolerance. The acceptance record is
`logs/qwk_reproduction/acceptance.json` inside the new workspace. This is a
postprocessing reconstruction, not another raw-to-model validation.

The lower-level author entry remains available:

```text
python run_qwk_prevalence_sensitivity.py --stage freeze
python run_qwk_prevalence_sensitivity.py --stage analyze
```

Those commands in the release source root require the exact author-side inputs
listed in `config/qwk_prevalence_sensitivity/protocol.json`; a fresh source
download alone does not contain them. Readers should use the wrapper above so
that their reconstruction timestamps and input hashes are recorded separately.
Existing protocols and completed outputs are verified rather than silently
replaced. The QWK addition is supplied in the v1.3.1 source version; it was not
part of the already published v1.3.0 DOI. Cite the verified version-specific
archive for v1.3.1 when referring to this addition, once that archive is published.

## Artifacts and validation

- `config/qwk_prevalence_sensitivity/protocol.json`: rules and source/input hashes.
- `results/qwk_prevalence_sensitivity/qwk_gaps.csv`: seed/model estimates and CIs.
- `results/qwk_prevalence_sensitivity/class_weights.csv`: counts, proportions, weights.
- `results/qwk_prevalence_sensitivity/seed_summary.csv`: descriptive across-seed summaries.
- `results/qwk_prevalence_sensitivity/bootstrap_draws.npz`: paired raw/standardized draws.
- `results/qwk_prevalence_sensitivity/input_audit.json`: record alignment and replay checks.
- `logs/qwk_prevalence_sensitivity/complete.json`: completion and output hashes.

Four focused tests passed, including agreement with record-level scikit-learn
weighted QWK, identity when proportions already match, a toy example with only
prevalence change, and invalid-input rejection. All ten held-out cohorts matched
their recorded IDs, targets and split positions. All original QWK bootstrap
draws and the ten original weighted-model gap intervals were reproduced.
The standardized point estimates were independently checked with scikit-learn
record weights. Protocol-bound parent inputs remained unchanged.

The reproduction wrapper additionally checks the complete input inventory,
rejects overlapping source/parent/output directories, and refuses to overwrite
an existing output directory. The supplementary submission bundle retains the
author's bootstrap arrays; the source archive includes code, protocol, compact
CSV references and completion records, but excludes record-level arrays.

LightGBM's mean gap increased from +0.01845 to +0.02286; every standardized
seed-specific interval remained above zero. Logistic's mean gap changed from
-0.00245 to +0.00289, but each standardized seed-specific interval included zero.
These are metric-specific descriptive findings, not evidence of a causal effect
of the partition scheme or absence of other distributional changes.
