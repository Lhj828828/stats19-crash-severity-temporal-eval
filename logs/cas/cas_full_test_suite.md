# CAS full independent test suite

Status: **PASS**

Date: 2026-09-01

## Checks

- Feasibility audit: 8/8 checks passed.
- Modeling-input audit: 8/8 checks passed.
- Baseline freeze and 2025 embargo checks passed.
- LightGBM freeze and 2025 embargo checks passed.
- One-time evaluation checks passed: 11 groups, 33 metric rows and 263,061 group-level prediction rows.
- H1/H2 checks passed: 135 H1 rows and 99 H2 rows.
- H3 checks passed: five stability rows and raw SHAP shape 5 x 2 x 4,887 x 3 x 15.
- Cross-dataset closeout checks passed: eight deterministic reporting rows.

No test downloaded data, refit preprocessing, retrained a model, repeated the
one-time 2025 evaluation, changed a threshold or selected a model.
