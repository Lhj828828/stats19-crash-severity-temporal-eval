# D9 ordered-logit checkpoint

## Status

**PASS - unweighted proportional-odds logistic baseline completed on validation data only.**

## Frozen interpretation

- Ordered Logit is a secondary ordinal baseline, not the primary comparator for weighted LightGBM.
- statsmodels OrderedModel has no class-weight interface; the model is explicitly unweighted.
- Reference-level encoding was fitted separately on each training partition.
- No test metric and no 2024 performance was calculated.
- Runtime gate selected **subset** training mode.
- Optimizer convergence: **6/6**; warning records: **2** (one each for random seeds 4409 and 5501).
- D9 V3 did not persist the warning text; no claim of warning-free estimation is made.
- Ordered-model coefficients and standard errors are not used for inferential claims.

## Temporal validation (2023)

- Macro-F1 **0.2886**, QWK **0.0009**, Fatal recall **0.0000**, ordinal MAE **0.2541**.

## Random-reference validation

- Macro-F1 mean **0.2929** (SD **0.0006**).
- QWK mean **0.0019** (SD **0.0014**).

## Handoff

- D10 may tune class-weighted LightGBM using training and validation data only.
- The 2024 temporal test and all random test partitions remain sealed until D11.
