# CAS bootstrap uncertainty checkpoint

Status: **PASS - H1_H2_BOOTSTRAP_COMPLETE**

## Method

- Iterations: **2000**.
- H1 uses independent class-stratified resampling of internal and 2025 cohorts.
- H2 uses paired class-stratified resampling of shared model-evaluation rows.
- Intervals are percentile 95% intervals; no p-value threshold is used.

## Selected point summaries

- H1 logistic_weighted Macro-F1 internal-minus-2025 mean: **-0.0003** (SD **0.0015**).
- H1 lightgbm_weighted Macro-F1 internal-minus-2025 mean: **0.0028** (SD **0.0043**).
- H2 temporal 2025 macro_f1 LightGBM-minus-Logistic: **-0.0007** (95% interval **[-0.0086, 0.0074]**).
- H2 temporal 2025 fatal_recall LightGBM-minus-Logistic: **-0.1622** (95% interval **[-0.2239, -0.1042]**).

## Interpretation boundary

- A direction is described from the point estimate and interval; absence of interval exclusion of zero is not called equivalence.
- CAS and STATS19 estimates remain separate and are not pooled.
