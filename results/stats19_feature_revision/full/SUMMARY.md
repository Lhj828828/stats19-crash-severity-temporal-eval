# Post-review 15-feature analytical run

Legacy outcomes were known before this correction. This is not a newly unseen test or a preregistration.

## Temporal 2024 evaluation

Model | Macro-F1 | QWK | Fatal recall | Mean cost
--- | ---: | ---: | ---: | ---:
dummy_most_frequent | 0.286065 | 0.000000 | 0.000000 | 0.541421
logistic_unweighted | 0.286065 | 0.000000 | 0.000000 | 0.541421
logistic_weighted | 0.337163 | 0.105180 | 0.569907 | 0.748293
lightgbm_weighted | 0.365790 | 0.104656 | 0.269640 | 0.640017

## Interpretation boundaries

H2 joint descriptive gate: False. This is not a noninferiority finding.
H1 intervals represent test-sampling uncertainty within each seed. Across-seed mean, sample SD and range are separate descriptive quantities.
H3 correlates 15 feature mean-absolute raw-margin SHAP importance values, using the same fitted random model on internal and future cohorts.
Fatal-output SHAP uses all explanation records, not only fatal records. Output-specific signs are associative, not causal.
No spatial generalization or cross-country universality is established by this run.

## Not completed by this run

Ordered Logit, matched-subset, exclusion-2020 and tree-count sensitivity analyses have not been rerun on 15 fields.
Those historical 17-feature results cannot be merged into corrected main results. CAS is unchanged.
Repository publication, software DOI update and clean-room public reproduction of this revision are not performed here.
Two previously recorded legacy D14 protocol-hash test failures remain separate; this run does not erase or repair historical freezes.
