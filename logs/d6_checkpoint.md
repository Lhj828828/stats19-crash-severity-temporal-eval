# D6 protocol freeze and checkpoint 1

## Status

**PASS - protocol frozen before model training.**

## Locked temporal evaluation

- Training: 2018-2022, **538,461** collisions.
- Validation: 2023, **104,258** collisions.
- Final temporal test: 2024, **100,927** collisions.
- 2024 target counts: Slight **75,858**, Serious **23,567**, Fatal **1,502**.
- Fatal prevalence in 2024: **1.488%**.

The 2024 Fatal count (1,502) exceeds the pre-specified screening threshold (50). This supports retaining fatal recall as a reported metric, but it does not guarantee a narrow confidence interval. Bootstrap uncertainty remains mandatory.

## Random same-distribution reference

- Pool: 2018-2023, **642,719** collisions.
- Per seed: train **449,903**, validation **96,408**, test **96,408**.
- Five frozen seeds: 1103, 2207, 3301, 4409, 5501.
- Splits are stratified by the three-class target. The actual collision-level assignments are saved, not merely regenerated from seeds.
- The random test is an optimistic internal reference. It is not interpreted as future-year performance.

## Leakage controls frozen at D6

- The exact 17-feature allowlist remains unchanged.
- Metadata and target columns cannot enter the feature matrix.
- All preprocessing, class weighting and optional resampling are fitted on training data only.
- Random and temporal test partitions cannot be used for tuning or model selection.
- The optional random-model-on-2024 diagnostic is pre-registered and may only reuse an already-fitted model without retuning.

## Figure 2

- Panel (a): annual Slight, Serious and Fatal distributions for 2018-2024.
- Panel (b): complete sample flow and the frozen temporal/random evaluation structure.

## Checkpoint decision

Proceed to modelling. Bootstrap is fixed at 2,000 iterations with random seed 20260828. Any change to years, features, cleaning rules, split ratios, seeds, bootstrap settings or the 2024 access rule requires a new protocol version and an explicit reason recorded before inspecting model performance.
