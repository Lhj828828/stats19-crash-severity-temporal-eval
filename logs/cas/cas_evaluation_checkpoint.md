# CAS one-time test evaluation checkpoint

Status: **PASS - FROZEN_MODELS_EVALUATED_ONCE**

## Primary temporal test (2025)

- dummy_most_frequent: Macro-F1 **0.2921**, QWK **0.0000**, Fatal recall **0.0000**.
- lightgbm_weighted: Macro-F1 **0.3273**, QWK **0.0887**, Fatal recall **0.4517**.
- logistic_weighted: Macro-F1 **0.3281**, QWK **0.1033**, Fatal recall **0.6139**.

## Scope

- 11 evaluation data groups were enumerated before prediction.
- Three frozen models were evaluated on identical records per group.
- No preprocessing, model, threshold or hyperparameter was refitted.
- Random internal tests and random-model 2025 diagnostics are reported separately.
