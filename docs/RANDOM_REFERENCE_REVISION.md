# Post-review correction of random-reference model selection

## Reason and scope

The 6 September 2026 methods audit identified overlap between the temporal
selection process and the random internal-test records. STATS19 random tests
contain 15,544-15,903 records from the 2023 temporal validation set; CAS random
tests contain 1,541-1,648 records from the 2024 temporal validation set. Other
random internal-test records also enter the training data of the temporal
candidate models. Excluding only the temporal validation year would not
isolate the full upstream selection process.

The correction preserves the frozen records, feature allowlists and five
random partitions. For each split it fits six LightGBM candidates on that
split's training partition, uses only its own validation partition for early
stopping and candidate selection, and computes the validation constraints
relative to the same split's weighted Logistic baseline. QWK and fatal-recall
margins remain 0.01 and 0.05. Ranking and fallback are unchanged. CAS retains
the pre-existing conditional 2,000-round extension; STATS19 retains its
1,200-round cap. Preprocessing and class weights remain training-only.

All five models for a data source must be sealed before the revised tests
are evaluated. The final refit uses only the original training partition,
not training plus validation. Record identities, role hashes, train-only
vocabularies, cached candidate results and selected model hashes are checked.

The historical outcomes are already known. The correction is explicitly
post-review and cannot restore the claim of an entirely unobserved test.
Revised results are reported regardless of whether they strengthen or weaken
the original findings. Legacy outputs are preserved for transparent comparison,
not selected as an alternative based on favorable results.

## Run

Use the locked project environment after reconstructing the original data
processing and baseline outputs. No live data download or model fitting is
performed by the freeze stage. A model run is limited to eight CPU threads;
candidates are processed sequentially.

```text
python tests/test_random_reference_revision.py
python run_random_reference_revision.py --all
```

Individual stages are available:

```text
python run_random_reference_revision.py --stage freeze
python run_random_reference_revision.py --stage smoke
python run_random_reference_revision.py --stage tune
python run_random_reference_revision.py --stage evaluate
python run_random_reference_revision.py --stage bootstrap
python run_random_reference_revision.py --stage shap
python run_random_reference_revision.py --stage summarize
python run_random_reference_revision.py --stage verify
```

Run from a checkout containing the corrected source against separate public
reconstruction workspaces with:

```text
python run_random_reference_revision.py --project-root /path/to/stats19-reproduction --dataset stats19 --all
python run_random_reference_revision.py --project-root /path/to/cas-reproduction --dataset cas --all
```

The original public reconstruction entry points continue to reconstruct the
historical archive. The correction is an explicit additional step, not a
silent change to their frozen references. Its upstream includes the regenerated
processed table, split assignments, Logistic validation predictions, and
historical Dummy/Logistic/LightGBM test predictions. The freeze stage binds
their hashes to the particular reconstruction workspace. Code hashes refer
to the checkout from which this new entry point is invoked.

Reissuing `--all` resumes completed candidate/model/SHAP-chunk checkpoints.
Do not run two correction commands concurrently against the same workspace;
an operating-system file lock prevents concurrent writers. Protocol and source
hash changes are rejected. Do not delete a frozen protocol to conceal a change;
document an implementation amendment or use a separately identified new run.

## Outputs and completion

Outputs are separate from all historical paths:

- `config/random_reference_revision/<dataset>/`: protocol and model seals.
- `models/random_reference_revision/<dataset>/`: five corrected models.
- `results/random_reference_revision/<dataset>/`: candidate/validation results,
  ten evaluation cohorts, metrics, confusion matrices, paired differences,
  seed-specific H1 intervals, SHAP arrays/ranks and cross-seed summaries.
- `figures/random_reference_revision/<dataset>/`: H1 legacy/revised comparison
  and corrected SHAP-rank stability plots.
- `logs/random_reference_revision/<dataset>/`: historical protection manifest,
  stage status, readable checkpoint and final integrity manifest.

A run is complete only when its `complete.json` records
`POST_REVIEW_RANDOM_REFERENCE_CORRECTION_VERIFIED`. Source implementation,
protocol presence or model completion alone does not establish completion.
The final verifier recomputes metrics, checks a sample of serialized-model
predictions, validates local candidate selection and vocabularies, checks
SHAP additivity, and verifies that historical model/config/result/log files
have not changed. The tests additionally include direct overlap rejection,
test-label perturbation and bootstrap/rank-kernel checks.

## Author-side completion on 6 September 2026

Both completion markers passed. All ten random LightGBM models, twenty
evaluation cohorts, bootstrap outputs and SHAP analyses were generated and
verified. Each source's protection manifest confirmed the same 553 historical
files unchanged. The separate independent audit checked 1,300 intervals from
saved draws, all SHAP ranks and sample memberships. No model was selected on
the revised test outcomes.

- STATS19 selected C03/1,200 in all five random splits (fallback in all five).
  All hard test predictions match the legacy branch. Mean H1 Macro-F1 gap is
  -0.00318, QWK +0.05355, and fatal recall +0.26466. Mean overall H3 rho is
  0.60588. Identical predictions do not make the old selection procedure valid.
- CAS selected C06/958, C03/851, C06/1,101, C01/1,191 and C03/927, in seed
  order. Only seed 4409 satisfies both validation constraints. Mean H1
  Macro-F1 gap is +0.00098, QWK -0.00001 and fatal recall +0.04349. Mean
  overall H3 rho is 0.99929. Do not replace these values with legacy CAS
  random outputs or interpret the correction as a uniform improvement.

The broader regression run passed 27 of 29 standalone test files, including
all 16 new correction tests. Two old author-side D14 tests stop at exact
protocol equality because the earlier public path-normalization commits
changed provenance bytes. These are retained historical checks, not fresh-run
public reconstruction checks. See `docs/REPRODUCIBILITY_BOUNDARY.md` and
`logs/random_reference_revision/regression_report.json`. The old hash values
were not rewritten and those failed tests are not relabeled as passes.
The independent historical scientific contract passed all 21 comparisons and
all seven required files; this is a narrower, separately reported check.

A dispatcher error stopped the first run before revised SHAP computation.
`config/random_reference_revision/implementation_amendment_01.json` documents
the exact source hashes and unchanged model seals. It fixes routing of the
SHAP stage, with no scientific rule change or model refit.

The methods draft and outline are synchronized. Final manuscript integration,
the existing Visio workflow figure and a new software release/version DOI
remain separate tasks. No public release was made during this correction.

## Statistical boundaries

H1 is the same fitted model's internal-versus-future gap. Percentile intervals
use internal minus future for higher-is-better metrics and future minus
internal for ordinal MAE and asymmetric cost, so positive always means a more
optimistic internal evaluation. The intervals
use 2,000 independent true-class-stratified draws for the two disjoint cohorts.
Models compared on the same cohort share draws. Five seeds are summarized by
mean, sample SD and range, not treated as independent studies. The exact
joint-prediction-pattern bootstrap is distribution-equivalent to record
resampling and retains model pairing. Models are not retrained per draw.

The corrected SHAP branch reuses the original deterministic sample design:
10,000 proportional records per STATS19 cohort and 4,887 CAS records with
matched true-class counts. TreeSHAP describes raw multiclass scores, with
additivity checks. Global ranks average absolute contributions across records
and equally across output classes; class-specific results are also retained.
Bootstrap acts on cached contributions, not by rerunning TreeSHAP. STATS19
future samples and their bootstrap draws remain shared across random models.

Temporal models, H2, existing Logistic/Dummy baselines, data cleaning, field
audit and original splits are not changed. The correction does not resolve
label-recording differences, establish causation or prove cross-national
generalizability. Its software release/DOI must be published separately after
verification; historical DOIs must not be described as archiving this code.
