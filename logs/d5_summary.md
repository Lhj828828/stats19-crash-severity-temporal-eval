# D5 data-quality audit and frozen dataset

## Result

- Records audited and retained: **743,646** (100.000%).
- Duplicate collision identifiers: **0**.
- Invalid dates or date/year mismatches: **0**.
- Missing categorical feature values: **0**.
- Unknown speed-limit values retained as NaN: **95 (0.013%)**.
- D5 row deletions: **0**.

## Target distribution

- Slight: 576,901 (77.577%).
- Serious: 155,925 (20.968%).
- Fatal: 10,820 (1.455%).

The target is strongly imbalanced. D5 does not resample the data. Class weights, if used, must be computed inside each training partition; SMOTENC remains an optional training-fold-only sensitivity analysis.

## Category audit

- Categories with fewer than 100 records globally: **6**.
- Categories present in 2024 but absent in 2018-2023: **1**.
- No category is pooled using the complete dataset. Frozen data preserve official labels. Future encoders fit vocabulary on training data only; a validation/test category absent from training maps to `__UNSEEN__`, even when its official label denotes unknown/not applicable. The audit retains the original label.
- The 2018-2023 versus 2024 comparison is a quality diagnostic only; the final temporal split is frozen at D6.

## Provenance

- Parent D4 SHA-256: `2277e47fd899c15116bfa2e62eca9f9c7ea9d45d08d1b5cda2027724c843021e`.
- Frozen D5 SHA-256: `2277e47fd899c15116bfa2e62eca9f9c7ea9d45d08d1b5cda2027724c843021e`.
- The hashes are identical because all hard checks passed and no record required deletion or value rewriting at D5.
