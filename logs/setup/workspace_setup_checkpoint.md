# CAS workspace setup checkpoint

Status: **READY_FOR_ADAPTER_DEVELOPMENT**

Date: 2026-09-01 (Asia/Shanghai)

## Completed

- Created an independent CAS workspace outside the frozen STATS19 project.
- Copied 26 CAS audit, source, configuration and environment files.
- Verified every copied file against its STATS19 source with SHA-256.
- Passed all 8 frozen CAS feasibility regression checks.
- Confirmed the snapshot contains 43,121 injury crashes from 2022-2025.
- Confirmed the proposed temporal split: 2022-2023 train, 2024 validation,
  2025 final temporal model test.

## Boundary

No CAS model has been trained or evaluated. The 2025 cohort was inspected
during feasibility auditing for schema, target counts, missingness/category
coverage and drift. It remains excluded from preprocessing fitting, tuning,
model selection and performance inspection until the CAS model protocol is
frozen.

## Next gate

Build the CAS adapter and model-ready table, test literal `None`/`Null`
preservation and structural-missingness handling, then freeze the schema and
split assignments before any model training.

