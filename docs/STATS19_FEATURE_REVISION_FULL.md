# Corrected 15-feature full execution

This is an isolated post-review correction, not a new untouched-test study.
The two reviewed fields are removed without changing records, targets, split
memberships or the pre-existing model-selection rules. CAS is not rerun here.

The preparation protocol and code remain unchanged. The new execution protocol
binds their hash, the full-run implementation and runtime versions before fits.

```text
python run_stats19_feature_revision_full.py --stage preflight
python run_stats19_feature_revision_full.py --stage freeze
python run_stats19_feature_revision_full.py --stage train
python run_stats19_feature_revision_full.py --stage evaluate
python run_stats19_feature_revision_full.py --stage bootstrap
python run_stats19_feature_revision_full.py --stage shap
python run_stats19_feature_revision_full.py --stage verify
```

Use the project's locked Python environment. Eight CPU threads are the maximum.
Preflight must pass before the first execution freeze; do not rewrite its
frozen evidence file afterward. The other commands resume verified completed stages; an incomplete training
split restarts from its beginning, while SHAP resumes checked 1000-row chunks.
The `all` stage runs the above sequence. An OS lock prevents concurrent writers.

All outputs are under `*/stats19_feature_revision/full/`. No old result or
preparation artifact is overwritten. Smoke models are explicitly rejected.
All six model sets are sealed before revised test evaluation. Each split uses
its own 15-feature weighted Logistic validation reference and candidate search.
Nonconvergent Logistic fits halt before sealing and require a recorded decision.

The eleven evaluation groups are temporal-2024, plus internal and 2024 cohorts
for each of random seeds 1103, 2207, 3301, 4409 and 5501. Four models are evaluated
in each group. Validation data do not appear in these eleven test groups.

Paired within-cohort Bootstrap and independent internal-versus-future Bootstrap
use 2000 true-class-stratified draws. Across-seed sample SD and range are
descriptive and are not the test-sampling confidence interval. A confidence
interval crossing zero does not demonstrate noninferiority or equivalence.

SHAP ranks use 15 mean-absolute raw-margin feature contributions, averaged over
records and equally over three model output classes. Separate output-class
results are also saved. Fatal-output SHAP is not a fatal-record-only analysis.
The same random model is compared across its two held-out cohorts. The temporal
model's 2024 SHAP is descriptive, not an additional independent H3 replicate.
Sample class counts, additivity checks and per-class signed summaries are saved.
No pooled signed risk direction or correlated-feature-group stability is claimed.

The verification stage recomputes metrics, draws, paired differences, gaps,
rank statistics, training-only preprocessing, sampled serialized predictions,
and all protected historical hashes. It is an author-side analytical check,
not a fresh public raw-to-results clean-room reproduction.

Ordered Logit/matched subsets, exclusion-2020 and tree-count extensions are NOT
rerun by this entry point. Do not combine their historical 17-feature values
with corrected primary tables. Public release/DOI updates are separate tasks.
