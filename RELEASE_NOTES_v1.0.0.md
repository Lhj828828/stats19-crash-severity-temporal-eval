# v1.0.0 release notes

## Scope

This release packages the frozen computational materials for the IEEE Access
manuscript on leakage-aware temporal generalization in collision-severity
classification. It contains the STATS19 primary workflow and the separately
trained New Zealand CAS replication.

The CAS analysis is an independent workflow replication. It is not an
additional training pool, and it does not establish cross-national or
cross-domain generalizability.

## Frozen analysis status

- STATS19: 2018-2024 collision table; 17 collision-time observable features;
  2024 is the untouched temporal test cohort.
- CAS: 2022-2025 injury-crash snapshot; 15 dataset-specific features;
  2025 was evaluated once after model freezing.
- Primary models: Dummy, weighted multinomial Logistic and weighted LightGBM.
- Ordered Logit remains a secondary matched-subset baseline.
- H1-H3, sensitivity analysis, audit logs and independent checks are retained.

## Verification disclosure

D16 completed the raw-to-results reproduction with status
`D16_CORE_REPRODUCTION_PASS_WITH_DOCUMENTED_NUMERICAL_DEVIATIONS`. The core
Logistic/LightGBM chain and scientific decision outputs passed the frozen
checks. Nineteen documented numerical deviations remain for the secondary
Ordered Logit path, propagated bootstrap outputs and one probability-only
four-thread sensitivity result; no tolerance was widened to hide them.

The CAS closeout was completed separately. The direction of the fatal-recall
trade-off agreed with STATS19, whereas the Macro-F1 gain and SHAP-rank pattern
did not reproduce. These results are reported as directional evidence only.

## Distribution boundary

Raw data snapshots, fitted model binaries and record-level predictions are
excluded from the Git repository. Their source URLs, retrieval dates and
checksums are retained in `DATA_SOURCES.md` and the stage manifests. A DOI
archive should state explicitly whether any supplementary binary bundle is
included; no raw-data redistribution is implied by this source release.
