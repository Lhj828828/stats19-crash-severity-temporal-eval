# CAS independent replication checkpoint

Status: **CAS_REPLICATION_ANALYSIS_COMPLETE**

## Frozen execution facts

- CAS contains 43,121 police-reported injury crashes from 2022-2025.
- Dummy, weighted multinomial Logistic and weighted LightGBM were frozen before the authorized one-time 2025 evaluation.
- The one-time evaluation generated 11 groups, 33 metric rows and 33 prediction files without preprocessing refit or post-test selection.
- H1/H2 use 2,000 class-stratified Bootstrap iterations; H3 explains five frozen LightGBM models with 2,000 record-level Bootstrap iterations.

## Directional comparison

- H1: broad metric-dependent behavior agrees, but the specific QWK and fatal-recall evidence is weaker in CAS.
- H2 Macro-F1: not replicated. STATS19 delta +0.01390 [0.01087, 0.01696]; CAS delta -0.00072 [-0.00864, 0.00738].
- H2 fatal recall: direction replicated. STATS19 delta -0.10053 [-0.11651, -0.08389]; CAS delta -0.16216 [-0.22394, -0.10425].
- H3: different observed pattern. STATS19 mean rho 0.6059; CAS mean rho 0.9964.
- Workflow executability on the second administrative source is demonstrated.

## Reporting boundary

- This reporting contract was written after both datasets' results were known; it is not a preregistration.
- Records, absolute metrics, Bootstrap draws and SHAP ranks were not pooled.
- No new inferential test, model change, retuning or threshold adjustment was performed.
- These results do not prove cross-national or cross-domain generalizability.
