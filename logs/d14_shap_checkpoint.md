# D14 SHAP stability checkpoint

Status: **D14_SHAP_COMPLETE**

- H3 status: **RANK_CHANGE_QUANTIFIED_NO_BINARY_MATERIALITY_THRESHOLD**.
- Overall ranking rho across five frozen fits: mean 0.605882, SD 0.027260, range [0.571078, 0.632353].
- Seed-specific percentile intervals are reported in results/d14/d14_rank_stability.csv.
- No arbitrary material-instability cutoff was introduced after observing predictive results.
- SHAP values describe fitted-model contributions/associations, not causal effects.
- Regional holdout remains unexecuted because no concrete region was frozen before outcomes were known.
