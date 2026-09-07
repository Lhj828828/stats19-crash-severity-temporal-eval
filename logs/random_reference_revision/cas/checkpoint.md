# CAS random-reference post-review correction

All original outcomes were known before this correction. No preregistration or newly untouched test is claimed.

Only split-local random LightGBM selection and dependent outputs are revised. Historical temporal models and H2 are unchanged.

## Split-local selection
- 1103: C06, 958 iterations; constraints met: False.
- 2207: C03, 851 iterations; constraints met: False.
- 3301: C06, 1101 iterations; constraints met: False.
- 4409: C01, 1191 iterations; constraints met: True.
- 5501: C03, 927 iterations; constraints met: False.

## H1 oriented gaps (mean and sample SD across five seeds)
- macro_f1: 0.000982 +/- 0.005049; range [-0.006083, 0.007791].
- qwk: -0.000008 +/- 0.013307; range [-0.017211, 0.018745].
- ordinal_mae: 0.007851 +/- 0.009281; range [-0.003202, 0.019064].
- accuracy: 0.004151 +/- 0.007705; range [-0.004300, 0.015638].
- fatal_recall: 0.043487 +/- 0.023862; range [0.025198, 0.084739].
- serious_or_fatal_recall: -0.033409 +/- 0.025170; range [-0.058831, 0.000970].
- mean_asymmetric_cost: -0.000241 +/- 0.010065; range [-0.014499, 0.009629].

## H3
Mean rho 0.999286, sample SD 0.001597, range [0.996429, 1.000000].

Results must be interpreted metric by metric. No causal claim, equivalence claim or outcome-based choice between legacy/revised results is supported.
The two sources are not pooled. Bootstrap conditions on fixed fitted models and observed class counts.
