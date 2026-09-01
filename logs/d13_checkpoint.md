# D13 positioning checkpoint

Status: **D13_PERFORMANCE_CHECKPOINT_COMPLETE_H3_PENDING**

## H1

- Broad random-split optimism: **NOT_SUPPORTED**.
- Branch: **METRIC_DEPENDENT_MIXED_GENERALIZATION_GAP**.
- LightGBM Macro-F1 gap: -0.003181 (SD 0.004529).
- LightGBM QWK gap: +0.053554 (SD 0.006314).
- LightGBM fatal-recall gap: +0.264663 (SD 0.009306).

## H2

- Macro-F1 rule: **PASS**.
- Joint incremental-value rule: **FAIL**.
- Macro-F1 delta: +0.013896, 95% CI [+0.010875, +0.016956].
- Fatal-recall delta: -0.100533, 95% CI [-0.116511, -0.083888].

## H3

- Status: **PENDING_NOT_EVALUATED**.
- No SHAP-stability claim is allowed before the frozen ranking analysis.

## Positioning

Evaluation-protocol optimism is metric dependent, with degradation in ordinal agreement and severe-class recall despite stable Macro-F1.

Nonlinear-model gains depend on the metric and include a fatal-recall trade-off.
