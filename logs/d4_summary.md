# D4 feature-engineering audit

- Input/output rows: **743,646**; no rows dropped.
- Model features before encoding: **17**.
- Metadata columns: **5**; all use the `meta_` prefix.
- Target column: `target_severity` with 0=Slight, 1=Serious, 2=Fatal.
- Target counts: Slight=576,901, Serious=155,925, Fatal=10,820.
- Speed-limit valid values: [20, 30, 40, 50, 60, 70].
- Speed-limit unknown codes listed by the current guide: [-1, 99].
- Speed-limit unknown codes observed in 2018-2024: [-1]; mapped to missing rows=95 (0.013%).
- LightGBM uses native missing-value handling; multinomial logistic regression and ordered logit use imputation fitted within each training fold only.
- Future model code must select exactly the 17-column feature_columns allowlist in d4_output_schema.json; metadata are never admitted through blacklist-style exclusion.
- Categorical codes were mapped to `code:official label`; unknown and not-applicable labels remain explicit categories.
- No one-hot encoding, statistical imputation, resampling or scaling was performed at D4.
- Output SHA-256: `2277e47fd899c15116bfa2e62eca9f9c7ea9d45d08d1b5cda2027724c843021e`.

The D4 table is an interim artifact. D5 must still perform duplicate, missingness and sample-flow quality control before a final processed dataset is frozen.
