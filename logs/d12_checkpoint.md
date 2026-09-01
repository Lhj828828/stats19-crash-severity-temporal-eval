# D12 Bootstrap checkpoint

Completed: 2026-08-31
Runtime: 12.79 seconds
Iterations: 2000; master seed: 20260828

## H2: LightGBM versus weighted Logistic on the same 2024 test records

- Macro-F1 raw delta: +0.013896 (95% CI +0.010875 to +0.016956).
- QWK raw delta: +0.013727 (95% CI +0.008728 to +0.019243).
- Fatal recall raw delta: -0.100533 (95% CI -0.116511 to -0.083888).
- Asymmetric cost raw delta: -0.040633 (95% CI -0.044339 to -0.037264); negative favors LightGBM.
- Do not call the nonlinear model uniformly superior unless the joint rule is satisfied.

## H1: random LightGBM internal test versus the same fitted model on 2024

- macro_f1: mean optimism gap -0.003181, SD across seeds 0.004529, range [-0.009607, +0.002613].
- qwk: mean optimism gap +0.053554, SD across seeds 0.006314, range [+0.045858, +0.063444].
- fatal_recall: mean optimism gap +0.264663, SD across seeds 0.009306, range [+0.253487, +0.278955].
- mean_asymmetric_cost: mean optimism gap -0.056547, SD across seeds 0.003887, range [-0.059807, -0.050427].

Seed summaries are descriptive because random test partitions overlap. Seed-specific intervals condition on fixed fitted models and do not include retraining uncertainty.
