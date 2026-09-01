# CAS H3 SHAP stability checkpoint

Status: **PASS - H3_SHAP_COMPLETE**

## Method

- Five frozen random-reference LightGBM models were explained; no model was refit.
- Each comparison uses all 4,887 internal-test rows and a 4,887-row 2025 sample with exactly matched true-severity counts.
- Importance is mean absolute raw-score SHAP over records and the three output classes.
- Uncertainty uses 2,000 independent class-stratified record Bootstrap iterations on precomputed SHAP values.

## Result

- H3 status: **RANK_CHANGE_QUANTIFIED_NO_BINARY_MATERIALITY_THRESHOLD**.
- Spearman rho across five frozen models: mean **0.9964**, SD **0.0036**, range **[0.9929, 1.0000]**.
- Seed 1103: rho **0.9929**, 95% interval **[0.9857, 0.9964]**.
- Seed 2207: rho **1.0000**, 95% interval **[0.9821, 1.0000]**.
- Seed 3301: rho **1.0000**, 95% interval **[0.9893, 1.0000]**.
- Seed 4409: rho **0.9929**, 95% interval **[0.9821, 1.0000]**.
- Seed 5501: rho **0.9964**, 95% interval **[0.9857, 1.0000]**.

## Interpretation boundary

- SHAP values describe fitted-model attribution, not causal effects.
- No post-test model selection, retuning, threshold adjustment or binary stability cutoff was introduced.
- CAS and STATS19 remain separate; only directional agreement or disagreement may be discussed.
