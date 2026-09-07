# Corrected exclusion-2020 sensitivity and revised scope

This note supersedes the pending-secondary-work list for the revised manuscript,
not any historical freeze or result. The decision was made after main results
were known. It is not an externally preregistered plan.

Keep the corrected 15-feature main experiment and a temporal-only exclusion-2020
sensitivity. Omit Ordered Logit, its matched-subset comparisons and tree-count
extensions from the revised manuscript. Preserve all historical artifacts for
audit; do not silently delete or rewrite them. The corrected main run's
1200-round search limit must still be disclosed. Larger budgets were not tested.

## Fixed protocol

- Training: 2018, 2019, 2021, 2022; 447,262 records (91,199 fewer than main).
- Validation: 2023; 104,258 unchanged records, descriptive only.
- Test: 2024; 100,927 unchanged records in the same order.
- Features: the same 15-field allowlist, excluding all metadata and target fields.
- Models: weighted Logistic and weighted LightGBM only, fixed to the corrected
  temporal main parameters. No new search, early stopping or threshold tuning.
- Refit category vocabularies, Logistic speed imputation/scaling, and balanced
  class weights on the reduced training set. LightGBM speed missingness stays NaN.
- Seal both sensitivity models and validation artifacts before test evaluation.
- Paired true-class-stratified Bootstrap, 2000 draws, seed 20260828, shared across
  main and exclusion predictions of both models. Compare each refit with its
  main model, compare LightGBM with Logistic, and calculate the change in that
  contrast on the same draws. Do not subtract separate confidence limits.

This analysis assesses the effect of excluding a training year, including the
change in sample size. It does not identify a causal pandemic effect or establish
robustness of H1 random-split gaps or H3 explanation stability. Intervals crossing
zero do not prove noninferiority. CAS is unchanged.

## Execution

Use the locked project environment:

```text
python run_stats19_exclude2020_revision.py --stage all
```

Individual stages are `preflight`, `freeze`, `train`, `evaluate`, `bootstrap`,
and `verify`. Eight CPU threads maximum. Completed checkpoints are hash-checked
and reused. An incomplete stage restarts; frozen code cannot be silently changed.
No raw-data processing or main-model refitting is needed for this sensitivity.

Artifacts go to `config`, `models`, `results` and `logs` under
`stats19_feature_revision/exclude2020/`. Model serialization and record-level
prediction arrays remain covered by the existing repository exclusion rules.

The final verification replays saved-model predictions on every validation/test
record, regenerates paired draws and contrasts, and checks all protected primary
and historical files. This is author-side verification, not a new public
raw-to-figures reproduction. After the manuscript is settled, run the final
paper-consistent clean reproduction and update the public repository/software
DOI. Publication and manuscript edits are separate steps.
