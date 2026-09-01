# D10 LightGBM checkpoint

## Status

**PASS - temporal-only tuning and six validation fits completed without test access.**

## Frozen selection

- Candidate: **C03**.
- Fixed trees for every split: **1200**.
- Eligibility constraints met: **False**.
- Candidate structure and tree count are identical across the temporal and five random-reference models.
- Split-specific class weights come only from each frozen training partition.
- A documented runtime-only amendment capped LightGBM at **4 CPU threads**; all candidates were restarted under that same cap.

## Temporal validation (2023)

- LightGBM: Macro-F1 **0.3573**, QWK **0.1163**, Fatal recall **0.3876**.
- D8 weighted Logistic reference: Macro-F1 **0.3308**, QWK **0.1012**, Fatal recall **0.6018**.
- Validation-only delta Macro-F1: **+0.0264**; this is not the D11 test claim.

## Random-reference validation

- Macro-F1 mean **0.3546** (SD **0.0016**).
- QWK mean **0.1156** (SD **0.0015**).
- Fatal recall mean **0.3721** (SD **0.0133**).

## Leakage controls and handoff

- LightGBM used the 17 frozen predictors only; metadata and target were excluded by allowlist.
- Native category vocabularies were fitted separately on training rows; validation-only levels map to `__UNSEEN__`.
- Speed-limit NaN values were retained for native LightGBM missing-value handling.
- No random test record and no 2024 record was scored.
- D11 is the first permitted one-time test evaluation.
