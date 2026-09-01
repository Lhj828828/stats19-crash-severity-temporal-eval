# D15 reproducibility checkpoint

Status: **D15_REPRODUCIBILITY_AUDIT_PASS**
Audit time: 2026-08-31T16:26:25+0800

## Scope

- This checkpoint verifies the frozen D1-D14 artifacts and does not refit models.
- Paths in the D15 runtime are resolved relative to the project root.
- Unified manifest: 411 files, 2152733124 bytes.

## Checks

- [PASS] protocol contract: D6-D14 statuses, selection lock, reporting boundaries and D15 handoff are consistent
- [PASS] data and split contract: 743646 rows, 17 features, 5 metadata columns and all six frozen role columns match the contract
- [PASS] primary prediction contract: Primary 2024 prediction tables and recomputed metrics agree; logistic_weighted: macro_f1=0.348337, fatal_recall=0.217710; lightgbm_weighted: macro_f1=0.362233, fatal_recall=0.117177
- [PASS] frozen model smoke test: Frozen temporal C03-1200 LightGBM loaded and predicted 64 records with a valid 3-class probability matrix
- [PASS] dependency contract: Python 3.13.15; 14 locked package versions match
- [PASS] path portability contract: D15 runtime files resolve paths from the project root; historical absolute provenance records remain archival only
- [PASS] artifact hash manifest: 411 files and 2152733124 bytes checked

## Boundaries

- H1 remains metric-dependent; no uniform random-split optimism claim is permitted.
- H2 includes a fatal-recall trade-off; no uniform LightGBM superiority claim is permitted.
- H3 is descriptive SHAP rank-change evidence, not a causal or binary instability claim.
- The planned regional holdout was not executed and spatial generalization remains unverified.

## Re-run

Use `run_d15_reproducibility.ps1` for the full audit and independent checks.
