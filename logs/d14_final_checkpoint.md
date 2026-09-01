# D14 final checkpoint

Status: **D14_COMPLETE_WITH_REGIONAL_HOLDOUT_DEVIATION**

## H1-H3

- H1: broad random-split optimism remains **not supported**; the observed gap is metric dependent.
- H2: LightGBM's Macro-F1 rule passed, but the joint incremental-value rule remains **failed** because severe-class trade-offs remain.
- H3: overall SHAP rank rho across five fixed-model temporal comparisons was mean **0.605882**, SD **0.027260**, range **[0.571078, 0.632353]**. Ranking change was quantified; no arbitrary binary materiality threshold was added.

## Excluding 2020

- LightGBM-versus-Logistic Macro-F1 delta: **+0.020623**.
- LightGBM-versus-Logistic fatal-recall delta: **-0.091877**.
- The metric trade-off persists; this is not a causal pandemic analysis.

## Error structure

- Fatal-to-Slight errors: weighted Logistic **952** (63.382% of fatal cases); LightGBM **897** (59.720%).
- Among fatal cases, Logistic alone was correct for **158** records and LightGBM alone for **7** records.
- Individual high-confidence errors are deterministic appendix illustrations, not representative cases.

## Protocol deviation

- The planned regional holdout was not executed because no concrete police-force combination was frozen before outcomes were known.
- No spatial-generalization claim is allowed. Selecting a region now would be post-hoc.

## Handoff

- D14 is complete for SHAP, exclusion-2020 sensitivity, error audit and reporting boundaries.
- D15 may begin with software packaging and end-to-end reproducibility work.
