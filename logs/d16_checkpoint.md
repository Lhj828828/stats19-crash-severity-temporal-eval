# D16 full reproduction checkpoint

Status: **D16_CORE_REPRODUCTION_PASS_WITH_DOCUMENTED_NUMERICAL_DEVIATIONS**
Completed: 2026-09-01T09:21:15+0800

## Execution

- Isolated workspace: `AUTHOR_SIDE_REPRODUCTION_WORKSPACE_NOT_DISTRIBUTED`
- Clean Python environment: `AUTHOR_SIDE_REPRODUCTION_WORKSPACE_NOT_DISTRIBUTED\.venv\Scripts\python.exe`
- Thread cap: **4**
- Completed pipeline/test stages: **52**
- Execution protocol SHA-256: `6c54871b70d967bbf0cfd9fcadeaa2b472a823d85aa7f239e17b77fb4ac5073d`
- Final comparison protocol SHA-256: `ccae94276f7e45e4eb4ca746d0c52ed562ef5dda197a2d1f87eb74df916c0f09`
- The retained D1-D15 source artifacts were read-only references.

## Comparison

- Passed checks: **196/215**
- Failed checks: **0**
- Documented numerical deviations: **19**
- Identifiers, labels, classes and counts were checked exactly.
- Floating outputs used the pre-frozen D16 tolerances.
- Model binary hashes were not the scientific endpoint; regenerated predictions, metrics and tests were checked.

## Scope

- CAS was not run in D16 and remains a separate later validation.
- GitHub Release and Zenodo DOI follow only after D16 passes.

## Documented numerical deviations

- `secondary Ordered Logit:results/d11/d11_test_confusion_matrices.csv`: column count differs after 0 rows
- `secondary Ordered Logit:results/d11/d11_test_metrics.csv`: column macro_f1 differs after 0 rows
- `secondary Ordered Logit:results/d11/predictions/random_seed_2207__ordered_logit_unweighted__2024_diagnostic.csv.gz`: label_differences=1/100927; max_probability_difference=0.02049160073; prob_fatal=0.002227061925, prob_serious=0.0182645388, prob_slight=0.02049160073
- `secondary Ordered Logit:results/d11/predictions/random_seed_2207__ordered_logit_unweighted__test.csv.gz`: label_differences=6/96408; max_probability_difference=0.02140353766; prob_fatal=0.003210391883, prob_serious=0.0194134021, prob_slight=0.02140353766
- `secondary Ordered Logit:results/d11/predictions/random_seed_4409__ordered_logit_unweighted__2024_diagnostic.csv.gz`: label_differences=1/100927; max_probability_difference=0.04394813989; prob_fatal=0.008579151439, prob_serious=0.03680697594, prob_slight=0.04394813989
- `secondary Ordered Logit:results/d11/predictions/random_seed_4409__ordered_logit_unweighted__test.csv.gz`: label_differences=1/96408; max_probability_difference=0.001053173624; prob_fatal=0.0001834261, prob_serious=0.0009252558363, prob_slight=0.001053173624
- `D12 bootstrap propagation:results/d12/d12_model_metric_intervals.csv`: column point_estimate differs after 0 rows
- `D12 bootstrap propagation:results/d12/d12_pairwise_model_differences.csv`: column raw_delta_ci_lower differs after 0 rows
- `D12 bootstrap propagation:results/d12/d12_random_optimism_gaps.csv`: column random_internal_point differs after 0 rows
- `secondary Ordered Logit:results/d12/d12_random_optimism_summary.csv`: column mean_optimism_gap differs after 0 rows
- `secondary Ordered Logit:results/d13/d13_main_performance_table.csv`: column random_internal_mean_5_seeds differs after 0 rows
- `D14 four-thread probabilities:results/d14/exclude2020/predictions/lightgbm_weighted__test.csv.gz`: label_differences=0/100927; max_probability_difference=0.0002085383239; prob_fatal=0.0002085383239, prob_serious=0.0001018890775, prob_slight=0.0001066492464
- `secondary Ordered Logit:results/d9/d9_validation_confusion_matrices.csv`: column count differs after 0 rows
- `secondary Ordered Logit:results/d9/d9_validation_metrics.csv`: column n_iter differs after 0 rows
- `secondary Ordered Logit:results/d9/validation_predictions/random_seed_2207__ordered_logit_unweighted.csv.gz`: label_differences=4/96408; max_probability_difference=0.02156138079; prob_fatal=0.003545397694, prob_serious=0.01931277805, prob_slight=0.02156138079
- `secondary Ordered Logit:results/d9/validation_predictions/random_seed_4409__ordered_logit_unweighted.csv.gz`: label_differences=0/96408; max_probability_difference=0.00101150409; prob_fatal=0.0002031368257, prob_serious=0.0009184554658, prob_slight=0.00101150409
- `secondary Ordered Logit:results/d9s1/d9s1_matched_comparison.csv`: column ordered_logit_macro_f1 differs after 0 rows
- `D12 bootstrap propagation:results/d12/d12_bootstrap_draws.npz`: primary_differing_groups=group_03,group_04,group_07,group_08; primary_max_abs_diff=0.07510729614; ordered_differing_groups=group_03,group_04,group_07,group_08; ordered_max_abs_diff=0.003472206175
- `png:figures/d12_h1_optimism_gaps.png`: pixel max_abs_diff=1 exceeds 0.00784314
