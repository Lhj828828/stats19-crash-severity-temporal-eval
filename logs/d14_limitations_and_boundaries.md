# D14 limitations and result boundaries

## What the study estimates

- The target is police-recorded collision severity conditional on a reported personal-injury collision. The models do not estimate whether a collision will occur.
- Predictors are restricted to 17 collision-table fields observable at the stated prediction time. Vehicle, casualty, consequence, response, exact-location and target-derived fields were excluded.
- Results apply to the frozen STATS19 2018-2024 cohort and the evaluated protocols. One country and one administrative data source do not establish cross-country or cross-domain generality.

## Evaluation boundaries

- The strict temporal test is one future year, 2024. It does not establish stability over longer horizons or under later coding changes.
- Random-internal and 2024 samples are different records and arise from different training/evaluation conditions. Their metric gaps are deployment-oriented descriptions, not causal effects of a split method.
- Bootstrap intervals condition on fitted models and observed test cohorts. They do not include full model-refitting uncertainty. The five random partitions overlap and are summarized descriptively, not treated as independent studies.
- Fatal cases total 1,502 in 2024. Fatal recall is therefore essential but still reflects one year's severe-class sample and the frozen argmax decision rule.
- The asymmetric error-cost matrix is a transparent evaluation convention, not a validated monetary or social cost function.
- Ordered Logit was trained on a fixed 100,000-row subset for computational feasibility; it is a secondary baseline and is not a like-for-like full-data model comparison.

## Model and explanation boundaries

- LightGBM improved temporal-test Macro-F1 relative to weighted Logistic, but the joint incremental-value rule failed because fatal recall deteriorated. No claim of uniform nonlinear-model superiority is permitted.
- H3 compared each frozen random model on its internal test and on 2024, holding the fitted model fixed. Overall feature-rank Spearman rho averaged 0.605882 (SD 0.027260; range 0.571078-0.632353). This quantifies ranking change under the evaluated temporal shift; no preregistered material-instability threshold supports a binary stable/unstable claim.
- TreeSHAP values describe contributions to fitted raw class scores. They are not causal effects. Correlated or substitutable features may exchange rank without a changed data-generating mechanism.
- SHAP used fixed 10,000-row severity-stratified samples and record-level Bootstrap. Its intervals quantify explanation-sample variability for frozen fits, not training or hyperparameter uncertainty.

## Sensitivity and protocol deviations

- Excluding 2020 from temporal training left the main metric trade-off intact: the LightGBM-versus-Logistic Macro-F1 delta was +0.020623, whereas fatal-recall delta was -0.091877. This does not identify a causal pandemic effect and does not test whether 2020 drives H1 random-versus-future gaps.
- A concrete regional holdout was never frozen before modeling. It was therefore not executed at D14, and no empirical spatial-generalization claim is allowed.
- The SHAP implementation protocol was frozen after predictive outcomes were known but before any SHAP ranking was inspected. The exact exclusion-2020 implementation was frozen after main results were known. These are transparent sequential-analysis disclosures, not external preregistration claims.

## Reporting language

- Use association, contribution, predictive importance, ranking agreement and sensitivity.
- Do not use cause, determinant, mechanism change, universal generalization, practical equivalence, or uniform model superiority for these results.
