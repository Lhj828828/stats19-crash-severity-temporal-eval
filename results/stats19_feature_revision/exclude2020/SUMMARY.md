# 15-feature exclusion-2020 sensitivity

Status: complete and verified.

Post-review sensitivity specified after main results were known. Not preregistered and not a newly unseen test.
Only 2020 is removed from temporal training. This also reduces training sample size; it does not isolate a causal pandemic effect.

Training: 2018, 2019, 2021, 2022 (447,262 records); validation: 2023 (104,258); identical test: 2024 (100,927).
Fixed corrected-main LightGBM: C03, 1200 rounds. No retuning or early stopping.
Both comparators use reduced-training fitted preprocessing and recomputed balanced class weights.

Model | Macro-F1 | QWK | Fatal recall | Mean cost
--- | ---: | ---: | ---: | ---:
main_logistic_weighted | 0.337163 | 0.105180 | 0.569907 | 0.748293
main_lightgbm_weighted | 0.365790 | 0.104656 | 0.269640 | 0.640017
exclude2020_logistic_weighted | 0.332177 | 0.104295 | 0.601864 | 0.770735
exclude2020_lightgbm_weighted | 0.365987 | 0.105728 | 0.276964 | 0.641137

Contrast | Metric | Difference | Paired 95% interval
--- | --- | ---: | ---
main_LightGBM_minus_Logistic | macro_f1 | +0.028627 | [+0.025425, +0.031922]
exclude2020_LightGBM_minus_Logistic | macro_f1 | +0.033810 | [+0.030696, +0.036976]
exclude2020_minus_main_Logistic | macro_f1 | -0.004986 | [-0.005941, -0.004043]
exclude2020_minus_main_LightGBM | macro_f1 | +0.000197 | [-0.002428, +0.002670]
change_in_model_contrast | macro_f1 | +0.005183 | [+0.002516, +0.007831]
main_LightGBM_minus_Logistic | qwk | -0.000524 | [-0.005697, +0.004883]
exclude2020_LightGBM_minus_Logistic | qwk | +0.001433 | [-0.003728, +0.006578]
exclude2020_minus_main_Logistic | qwk | -0.000885 | [-0.002816, +0.001140]
exclude2020_minus_main_LightGBM | qwk | +0.001072 | [-0.003161, +0.005317]
change_in_model_contrast | qwk | +0.001957 | [-0.002535, +0.006459]
main_LightGBM_minus_Logistic | fatal_recall | -0.300266 | [-0.324917, -0.275632]
exclude2020_LightGBM_minus_Logistic | fatal_recall | -0.324900 | [-0.349551, -0.298935]
exclude2020_minus_main_Logistic | fatal_recall | +0.031957 | [+0.022636, +0.041944]
exclude2020_minus_main_LightGBM | fatal_recall | +0.007324 | [-0.009987, +0.025300]
change_in_model_contrast | fatal_recall | -0.024634 | [-0.044607, -0.004660]
main_LightGBM_minus_Logistic | mean_asymmetric_cost | -0.108276 | [-0.112983, -0.103659]
exclude2020_LightGBM_minus_Logistic | mean_asymmetric_cost | -0.129599 | [-0.134424, -0.124901]
exclude2020_minus_main_Logistic | mean_asymmetric_cost | +0.022442 | [+0.020421, +0.024384]
exclude2020_minus_main_LightGBM | mean_asymmetric_cost | +0.001120 | [-0.002131, +0.004717]
change_in_model_contrast | mean_asymmetric_cost | -0.021322 | [-0.025098, -0.017685]

Main joint descriptive H2 gate (common draws): False.
Exclusion-2020 joint descriptive H2 gate: False.
Intervals containing zero do not establish noninferiority. All contrasts use paired true-class-stratified record Bootstrap (2000 draws).
The change-in-model-contrast interval is calculated on common draws, not by subtracting separate interval endpoints.
Main reference intervals are recomputed only to preserve four-model pairing; previously frozen main intervals remain unchanged.

## Revised scope

Keep the corrected main experiments and this exclusion-2020 sensitivity regardless of outcome.
Omit Ordered Logit/matched-subset and tree-count sensitivity from the revised manuscript; retain their historical artifacts for audit.
Still report the 1200-round search boundary. Larger tree budgets were not tested in the corrected feature run.
This does not test robustness of random-split H1 gaps or SHAP H3 stability. CAS is unchanged.
Final manuscript-consistent clean-room reproduction and repository/software DOI updates remain separate later tasks.
