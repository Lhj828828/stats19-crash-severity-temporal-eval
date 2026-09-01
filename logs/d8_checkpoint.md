# D8 baseline-model checkpoint

## Status

**PASS - Dummy and multinomial-logistic baselines completed on validation data only.**

## Leakage controls

- Preprocessing was fitted separately on each training partition.
- Validation data did not fit category vocabularies, imputation values, scaling or model coefficients.
- No temporal/random test metric was calculated.
- No 2024 record appears in a D8 prediction file.

## Temporal validation (2023)

- dummy_most_frequent: Macro-F1 **0.2880**, QWK **0.0000**, Fatal recall **0.0000**.
- logistic_unweighted: Macro-F1 **0.2881**, QWK **0.0000**, Fatal recall **0.0000**.
- logistic_weighted: Macro-F1 **0.3308**, QWK **0.1012**, Fatal recall **0.6018**.

## Random-reference validation

- dummy_most_frequent: Macro-F1 mean **0.2920** (SD **0.0000**), QWK mean **0.0000**.
- logistic_unweighted: Macro-F1 mean **0.2922** (SD **0.0001**), QWK mean **0.0003**.
- logistic_weighted: Macro-F1 mean **0.3243** (SD **0.0004**), QWK mean **0.0990**.

## Convergence and handoff

- Logistic convergence warnings: **0**.
- Weighted Logistic is the fixed linear comparator for weighted LightGBM.
- These are validation results and are not final generalization estimates.
- D9 may start only when explicitly requested; 2024 remains locked until D11.
