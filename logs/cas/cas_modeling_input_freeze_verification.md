# CAS modeling-input and split freeze verification

Status: **PASS - INPUTS_AND_SPLITS_FROZEN_BEFORE_MODEL_TRAINING**

Date: **2026-09-01 (Asia/Shanghai)**

## Frozen artifacts

- Model-ready table: 43,121 rows and 15 features (12 categorical, 3 numeric).
- Model-table SHA-256: `714ca06b492f209b95cde9042a0b439f38f0d29213f70bc64207d8af9be777cd`.
- Temporal rows: 21,934 training (2022-2023), 10,645 validation (2024), and
  10,542 locked final test rows (2025).
- Random-reference rows per seed: 22,805 training, 4,887 validation, and 4,887
  internal test rows over the 2022-2024 development pool.
- Split-assignment SHA-256:
  `69309305340f25c6bfb6fee7474b47bf015eba28d3c3627f081becc3d0ebdf32`.

## Verification

- 8/8 feasibility and source-contract checks passed.
- 8/8 modeling-table, leakage, split, identity and class-weight checks passed.
- Literal source categories `None` and `Null` remain distinct from genuine
  source missingness.
- Genuine JSON null values are represented by the reserved categorical token
  `__SOURCE_MISSING__`; numeric structural missingness remains `NaN`.
- Class weights were calculated independently from each training role only.

## Test boundary

The 2025 schema, target counts, category coverage, missingness and drift were
inspected during feasibility auditing. No preprocessing component or model has
been fitted on 2025, and no 2025 model prediction or performance metric exists.
This cohort remains prohibited for preprocessing fitting, tuning, threshold
selection and model choice.

## Next gate

Implement and independently test training-only preprocessing and baseline
training. Freeze each selected model before its single 2025 evaluation.
