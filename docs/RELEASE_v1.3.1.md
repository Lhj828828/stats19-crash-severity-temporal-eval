# v1.3.1: Post hoc QWK prevalence sensitivity

This additive source version makes Appendix A.14.6 and Table A13 reproducible
from a completed final 15-feature STATS19 run. It adds the QWK calculation code,
the protocol recorded on 16 September 2026, focused tests, compact reference
outputs, and an isolated postprocessing reproduction entry. No parent model
settings, primary outcomes, raw snapshots or historical tags are changed.

The analysis was specified after the primary results were known. It is not
preregistration, a new untouched-test evaluation or a causal decomposition of
temporal change. It standardizes the future cohort to each internal cohort's
observed true-class proportions, updates both QWK marginals, and replays the
original 2,000 conditional bootstrap resamples. Reference proportions remain
fixed; no model-refitting or prevalence-estimation uncertainty is included.

## Reproduce Table A13

After the STATS19 main reconstruction documented in `FINAL_ANALYSIS.md`, use
the same project environment and run from the extracted source directory:

```text
python -m unittest discover -s tests -p "test_qwk*.py" -v
python reproduce_qwk_sensitivity.py --parent ../final-reconstruction/stats19 --workspace ../qwk-reconstruction
```

The parent may also be an already completed compatible v1.3.0 reconstruction.
Its sealed predictions are reused locally, with no model training or new
prediction calls. A new protocol binds the reconstructed inputs; the author's
original protocol remains unchanged. The wrapper verifies all three summary
CSV tables against fixed references, with exact text/integer comparisons and
`atol=1e-10`, `rtol=1e-8` for floating values. Add `--resume` only to continue or
verify the same workspace. See `docs/QWK_PREVALENCE_SENSITIVITY.md`.

## Archive Boundaries

Validation on 16 September 2026 passed seven QWK-focused tests and a separate
postprocessing reconstruction using the previously completed clean STATS19
parent run. All 10 seed/model QWK rows, 15 class-weight rows and six across-seed
summary rows matched the recorded references at the stated tolerances. All ten
cohorts and original QWK bootstrap draws were checked. No models were retrained
and no prediction calls were made for this validation. The record is
`logs/qwk_prevalence_sensitivity/reproduction_acceptance.json`.

Raw data, fitted models, collision-level predictions and bootstrap arrays are
not included in this source archive. The separate manuscript supplement includes
the saved QWK bootstrap metrics. The source includes the code needed to regenerate
them after the parent reconstruction. Primary-analysis validation remains the
separately recorded 7 September 2026 author-run validation; this source addition
does not claim a fresh full pipeline or independent third-party validation.

The parent v1.3.0 DOI is https://doi.org/10.5281/zenodo.22642853 and does not
contain this addition. Data DOIs are unchanged: STATS19
https://doi.org/10.5281/zenodo.22290566 and CAS
https://doi.org/10.5281/zenodo.22296725. The new version-specific software DOI
must be verified after publication before being used in the manuscript.
