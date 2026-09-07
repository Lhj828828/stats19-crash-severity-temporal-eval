# STATS19 random-reference post-review correction

All original outcomes were known before this correction. No preregistration or newly untouched test is claimed.

Only split-local random LightGBM selection and dependent outputs are revised. Historical temporal models and H2 are unchanged.

## Split-local selection
- 1103: C03, 1200 iterations; constraints met: False.
- 2207: C03, 1200 iterations; constraints met: False.
- 3301: C03, 1200 iterations; constraints met: False.
- 4409: C03, 1200 iterations; constraints met: False.
- 5501: C03, 1200 iterations; constraints met: False.

## H1 oriented gaps (mean and sample SD across five seeds)
- macro_f1: -0.003181 +/- 0.004529; range [-0.009607, 0.002613].
- qwk: 0.053554 +/- 0.006314; range [0.045858, 0.063444].
- ordinal_mae: -0.151170 +/- 0.007883; range [-0.164193, -0.144743].
- accuracy: -0.103806 +/- 0.006340; range [-0.114475, -0.098594].
- fatal_recall: 0.264663 +/- 0.009306; range [0.253487, 0.278955].
- serious_or_fatal_recall: 0.296504 +/- 0.023986; range [0.271319, 0.335714].
- mean_asymmetric_cost: -0.056547 +/- 0.003887; range [-0.059807, -0.050427].

## H3
Mean rho 0.605882, sample SD 0.027260, range [0.571078, 0.632353].

Results must be interpreted metric by metric. No causal claim, equivalence claim or outcome-based choice between legacy/revised results is supported.
The two sources are not pooled. Bootstrap conditions on fixed fitted models and observed class counts.
