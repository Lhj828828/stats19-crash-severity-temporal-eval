# STATS19 post-review feature revision

## Scope and disclosure

The 7 September 2026 revision removes two fields from the original 17-feature
STATS19 model contract. It does not delete records, change severity labels,
redownload data, alter the original split assignments, or modify CAS.

- special_conditions_at_site: the official 2025 data guide maps the legacy
  field into harmonized carriageway_hazards. Its 2024 availability is strongly
  associated with the injury-reporting-system flag. This does not quantify its
  contribution to predictive performance.
- carriageway_hazards: category 19 explicitly specifies a pedestrian who was
  not injured. The entire field is conservatively excluded to retain the
  original outcome-excluded prediction-time definition.

Original test outcomes and explanation rankings are already known. This is a
post-review correction, not external preregistration and not a newly untouched
test set. Report the corrected analysis regardless of direction. Do not choose
between 15- and 17-feature versions after comparing their test performance.

Original D2/D3 audit files and all historical models/results remain intact.
The new field-audit amendment and exact allowlist are under
config/stats19_feature_revision/. Metadata, identifiers, and the target are
not selected as model inputs. Numeric unknown speed values remain missing.

## Preparation milestone

This milestone implements and verifies the 15-feature data contract,
split-local preprocessing, baseline fitting, six-candidate LightGBM selection,
train-only refitting, model serialization, and independent smoke checks.
The command-line entry point intentionally permits only preparation, smoke,
and verification. It has no full-training, test-evaluation or --all option.

Run with the locked project environment:

    python tests/test_stats19_feature_revision.py
    python run_stats19_feature_revision.py --stage prepare
    python run_stats19_feature_revision.py --stage smoke
    python run_stats19_feature_revision.py --stage verify

--project-root /path/to/project identifies a different local workspace.
This author-side preparation requires the frozen processed table, assignments,
parent protocols and reviewed official 2025 dictionary. It is not yet a new
public raw-to-results reproduction entry point.

Preparation hashes its source, upstream files and existing historical artifacts.
It refuses changed dependencies after freezing. Later implementation additions
must be documented and frozen before their execution; do not silently alter a
frozen source hash or delete a protocol to conceal a change.

The smoke run covers the temporal split plus all five random splits. Per split
it uses 3,000 stratified training and 1,200 stratified validation records, four
CPU threads, and six LightGBM candidates capped at 30 rounds. Dummy and both
unweighted/weighted Logistic baselines are also exercised. No internal or
future test performance is computed. Full-table identifiers, years, labels and
split membership may be read for integrity and partition auditing; this is not
a claim that files containing historical test labels are never opened.

Smoke candidates, metrics, saved models and predictions are explicitly marked
non-analytical. Their selected candidate is only a test of the selection code.
It must never determine the subsequent full-data candidate or manuscript result.
Smoke artifacts are stored separately, and production-artifact validation rejects
them. No smoke result should be interpreted as predictive evidence.

Completion is indicated by logs/stats19_feature_revision/preparation_complete.json
with status PREPARATION_VERIFIED_FULL_EXPERIMENT_NOT_RUN. This is not completion
of the scientific revision.

## Fixed method decisions for the later run

- Keep the exact 15-field order and original records, labels and six splits.
- Fit vocabularies, imputation, standardization and balanced weights from each
  training partition only. All models within a split use identical records.
- Fit new 15-feature weighted Logistic validation baselines. Do not reuse the
  old 17-feature baseline metrics as LightGBM eligibility constraints.
- Preserve the six original candidates, maximum 1,200 rounds, 75-round early
  stopping and unweighted validation multi-logloss. Each split uses only its
  own training and validation records. Do not reuse the temporal winner for
  random branches.
- Eligibility is local Logistic QWK minus 0.01 and fatal recall minus 0.05.
  Among eligible candidates, rank by Macro-F1, QWK, fatal recall, lower frozen
  complexity rank, and candidate ID. If none are eligible, use the same ranking
  on all candidates and explicitly report the failure of constraints.
- Refit using training only at the selected iteration and verify its validation
  probabilities match the selection fit. Seal all six model sets before any
  revised test evaluation. Use argmax; no threshold tuning.
- Subsequent inference retains paired within-cohort Bootstrap and independent
  internal/future draws, fixed class counts, 2,000 iterations and descriptive
  five-seed summaries. H3 retains the STATS19 sampling/aggregation design over
  15 features. These full analyses are not executed in preparation.
- Describe recording-system and label drift; feature removal alone does not
  establish cross-year label equivalence, causality or safety noninferiority.

## Not yet executed

Full-data baseline fits, 36 full-data LightGBM candidate fits, final model seals,
11 test-cohort evaluations, Bootstrap, SHAP, final table/figure updates and a
clean-environment public reconstruction remain later work. Their orchestration
must be implemented and separately verified before claiming end-to-end readiness.

Existing Ordered Logit, matched-subset comparisons, exclusion-2020 and tree-count
sensitivity outputs remain historical 17-feature results. They must either be
rerun under the revised feature contract or explicitly excluded/labeled as
historical in the revised report. No preparation output marks them completed.

No Git commit, release, DOI, manuscript replacement or Visio change is performed
by this entry point. Update those only after the final analytical version is
verified.
