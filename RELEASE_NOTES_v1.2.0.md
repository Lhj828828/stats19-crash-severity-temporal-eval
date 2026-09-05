# v1.2.0: CAS post-hoc sensitivity and public reproduction

Release date: 2026-09-05

This release adds the two feature-removal checks that were identified during
the CAS feasibility audit but were not part of the original primary execution.
It also makes those checks reproducible from the public CAS entry point.

## Scope

- Added `code/cas_feature_ablation_sensitivity.py`.
- Added the frozen post-hoc protocol and completion contract under
  `config/cas/`.
- Added an independent test at
  `tests/test_cas_feature_ablation.py`.
- Added two explicitly labelled scenarios: removal of `feature_urban`, and
  joint removal of `feature_advisory_speed` and
  `feature_temporary_speed_limit`.
- Both scenarios use the fixed 2022-2023 training, 2024 validation and 2025
  test cohorts, weighted multinomial Logistic, and the frozen C06 LightGBM
  configuration with 2,000 record-level bootstrap resamples.

## Interpretation boundary

The primary 2025 CAS results were known before this protocol was frozen. The
new analyses are therefore post-hoc sensitivity evidence. They were not used
to retune, select or replace the original 15-feature models, and the original
primary outputs are unchanged.

Removing `urban` leaves the main comparison nearly unchanged. Removing the
two sparse speed fields changes selected safety metrics and increases the mean
asymmetric cost for both refitted models. The results do not support a claim
that the CAS conclusions are insensitive to every feature choice.

## Public reproduction

The CAS public runner now binds only run-specific hashes of regenerated
upstream files before executing the post-hoc stages. This accommodates benign
runtime metadata differences while preserving the frozen scientific settings.
Raw data, fitted models, record-level predictions, bootstrap arrays and SHAP
arrays remain outside the software release.

The fixed STATS19 and CAS input records are unchanged. The previous `v1.1.0`
release remains an immutable reproduction boundary; users reproducing that
version should continue to use its version-specific archive and DOI.

The software concept DOI is
<https://doi.org/10.5281/zenodo.22231696>. A version-specific DOI for this
release is assigned by Zenodo when the GitHub release is archived.
