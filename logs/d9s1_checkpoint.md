# D9-S1 matched-subset sensitivity checkpoint

## Status

**PASS - six matched 100,000-row unweighted Multinomial Logistic models completed.**

## Design controls

- Each training subset was reconstructed with the exact deterministic D9_V3 selection algorithm.
- Row positions, collision identifiers, targets and SHA-256 identities were persisted for all six subsets.
- Preprocessing was fitted only on each matched 100,000-row training subset.
- Each complete validation partition exactly matched the paired D9 Ordered Logit artifact.
- No 2024 record was fitted, predicted or evaluated.

## Validation summary

- Temporal matched Logistic: Macro-F1 **0.2882**, QWK **0.0002**, Fatal recall **0.0000**.
- Temporal Logistic minus Ordered Logit Macro-F1 difference: **-0.0004**.
- Random-reference matched Logistic Macro-F1 mean **0.2923** (SD **0.0002**).

## Interpretation boundary

- This analysis controls training-row count but does not isolate model structure.
- Ordered and multinomial models still differ in constraints, regularization and encoding.
- Results belong in an appendix sensitivity table, not the primary model table.
- The reconstructed D9 subsets are deterministic, but D9 did not persist contemporaneous training identifiers; this provenance limitation must be disclosed.

## Handoff

- D10 may tune full-data class-weighted LightGBM using training and validation data only.
- The 2024 test remains sealed until D11.
