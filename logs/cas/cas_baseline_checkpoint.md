# CAS baseline validation checkpoint

Status: **PASS - BASELINES_FROZEN_BEFORE_2025_EVALUATION**

## Access control

- Preprocessing was fitted separately within each training partition.
- No random internal-test or 2025 model performance was calculated.
- The 2025 cohort remains locked for one-time evaluation.

## Temporal validation (2024)

- dummy_most_frequent: Macro-F1 **0.2922**, QWK **0.0000**, Fatal recall **0.0000**.
- logistic_weighted: Macro-F1 **0.3258**, QWK **0.0908**, Fatal recall **0.6225**.

## Random-reference validation

- dummy_most_frequent: Macro-F1 mean **0.2921** (SD **0.0000**).
- logistic_weighted: Macro-F1 mean **0.3249** (SD **0.0021**).

## Execution

- Thread limit: **8**.
- Logistic convergence warnings: **0**.
- Weighted Logistic is the frozen linear comparator for LightGBM.
