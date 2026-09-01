# D7 training-input audit

## Status

**PASS - frozen split assignments verified; no split regenerated.**

## Split storage

- Temporal roles and all five random-seed roles remain in the single collision-level D6 assignment artifact.
- Temporal counts: train **538,461**, validation **104,258**, locked 2024 test **100,927**.
- Each random seed: train **449,903**, validation **96,408**, internal test **96,408**; all 2024 rows remain `locked_temporal_test`.
- Seeds: **1103, 2207, 3301, 4409, 5501**.

## Training-only class weights

Balanced weights use `n_training / (3 * n_training_in_class)` and are computed independently for each training partition.

- Temporal Slight (0): **0.42558181**.
- Temporal Serious (1): **1.64787918**.
- Temporal Fatal (2): **23.02296049**.
- Five random training-specific weight sets saved: **5**.

No validation or test observations contribute to any class weight.

## 2024 embargo

D8-D10 may use 2018-2022 for fitting and 2023 for validation. The 2024 labels must not be inspected for model performance until every model and decision rule has been frozen for the one-time D11 evaluation.

## Decision

D7 is complete. D8 may start only when explicitly requested.
