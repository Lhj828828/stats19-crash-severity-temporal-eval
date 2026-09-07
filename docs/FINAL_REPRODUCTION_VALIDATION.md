# Final corrected reconstruction validation

Date: 7 September 2026. This record concerns the corrected experimental scope,
not manuscript layout, journal acceptance or a newly published software release.
Validation is author-run on the same Windows machine, not independent
third-party or cross-platform validation.

## Status

Both clean reconstructions completed: STATS19 passed all 22 required stages and
eleven V2 table comparisons; CAS passed all 25 stages and nine V2 comparisons.
The combined acceptance status is
`PASS_CLEAN_RAW_RECONSTRUCTION_AND_V2_RESULTS`, recorded in
`logs/final_analysis/final_reproduction_acceptance.json` on 7 September 2026.
No tolerance was increased or scientific reference changed to obtain this pass.

The two isolated dataset directories have independently created virtual
environments. Verified raw files, executable sources, protocol templates and
small comparison references were copied into them. No trained models, processed
tables, record predictions or SHAP arrays were copied. This run reused fixed raw
snapshots; it did not freshly download them. The seven STATS19 and two CAS inputs
passed size and SHA-256 checks (9/9).

## Frozen scope and references

- STATS19: the 15-feature branch, six split-local model sets, eleven evaluation
  cohorts, 2,000-draw Bootstrap, SHAP and fixed-parameter exclusion of 2020.
- CAS: separately audited inputs, temporal models, required diagnostic legacy
  outputs, post-hoc urban/sparse-speed removals and corrected split-local random
  models with their evaluation, Bootstrap and SHAP.
- Ordered Logit, matched-subset comparisons and tree-count extensions are not
  part of the corrected STATS19 reconstruction. Their old artifacts are retained.
- Old CAS random nonlinear outputs are diagnostic dependencies, not the current
  selected results. Old CAS SHAP/closeout stages are not part of this run.

The current result catalog is `FINAL_ANALYSIS_SCOPE_20260907_V2`, with 20 compact
tables in `results/final_analysis/v2/`. Integer/text values are compared exactly;
floating values use the unchanged `atol=1e-10`, `rtol=1e-8` rule. Passing means
agreement under these specified checks, not proof of universal correctness.

An export-only correction occurred during reconstruction: V1 CAS H2 included
90 superseded random-reference rows as well as the nine intended temporal rows.
V2 exports only the nine temporal rows. Scientific source results were unchanged;
the other 19 exported tables are byte-identical. V1 catalog and tables remain
archived. STATS19 was prepared with V1, CAS with V2; final acceptance compares
both against V2. The exact two reviewed STATS19 source-snapshot differences are
enumerated in `code/verify_final_reproduction.py`, not accepted by filename alone.
No scientific kernel or numeric tolerance was changed for this correction.

## Environment and tests

Both clean environments used Python 3.13.15 and the unchanged
`requirements-lock.txt`. Observed package inventories are preserved separately:

- `logs/final_analysis/stats19_environment_packages.txt`
- `logs/final_analysis/cas_environment_packages.txt`

Both `pip check` runs passed. The inventories record the observed installation,
not a claim that every transitive package is pinned by the original lock file.
Full experimental workflows ran serially with at most eight threads per model.

The following tests passed using the new STATS19 interpreter against the current
source tree. Counts exclude repeated preflight runs and standalone CAS audits.

| Suite | Passed tests | Local transcript |
| --- | ---: | --- |
| Final entry/catalog/acceptance | 32 | `logs/final_analysis/clean_entry_tests.log` |
| STATS19 preparation, main and exclusion-2020 | 75 | `logs/final_analysis/clean_scientific_regression_tests.log` |
| Split-local random correction | 16 | `logs/final_analysis/clean_random_revision_tests.log` |
| Public data, input audit, runner, references and documentation | 30 | `logs/final_analysis/clean_public_entry_tests.log` |
| CAS public runner | 6 | `logs/final_analysis/clean_cas_public_entry_tests.log` |
| Total | 159 | Logs are local, Git-ignored execution transcripts. |

Seven standalone CAS audit scripts also passed using the clean CAS interpreter
against its regenerated artifacts: feasibility, modeling inputs, baselines,
LightGBM, evaluation, Bootstrap and field-removal sensitivity. Their transcripts
are `logs/final_analysis/clean_test_cas_*.log`. These check the retained temporal
and baseline branch and required diagnostic artifacts; the old random LightGBM
outputs are not thereby promoted to current results. The corrected branch's
own verifier passed independently, protecting 210 historical input artifacts.
Old `test_cas_shap.py` and `test_cas_closeout.py` require deliberately omitted old
outputs and were not reported as passing tests of the corrected workflow.

Third-party LightGBM and joblib/NumPy deprecation warnings were retained and did
not cause failures. The two historical D14 exact-provenance assertions are not
included in this passing total. Their earlier failures and original frozen
hashes remain unchanged; the separate bounded historical verifier is not a
claim that these original assertions now pass.

## Acceptance evidence

The combined acceptance checked immutable preparation inputs, all stage-ledger
entries, clean interpreter/lock identity, executable snapshot hashes, nested
STATS19 artifact seals, label-recording counts, CAS correction/ablation manifests
and all selected numerical tables. Four diagnostic PNGs were checked for nonblank
pixels and inspected visually. Their dimensions and SHA-256 values are included
in the acceptance report. This is not final manuscript figure-layout approval.

Creation-time flags in the frozen catalog and workspace preparation markers
remain unchanged. Subsequent completion is recorded in the separate acceptance
report; historical records are not rewritten to make their timestamps or status
appear newer.

To repeat this read-only author acceptance (requires the author artifacts and
completed workspaces, not just a fresh source clone):

```text
python code/verify_final_reproduction.py --workspace ../final-reconstruction
```

First-time readers use the commands in `FINAL_ANALYSIS.md`; their reconstruction
already runs its own final numerical comparison against compact references.

## Public-source checks

The pre-release working-tree scan covered tracked and non-ignored untracked
candidate files, not raw/model directories or historical Git objects:

- 812 candidate files, approximately 12.5 MiB; none exceeded 10 MiB.
- All 327 candidate JSON files parsed successfully.
- 772 text files were checked for author-home paths, the author's contact
  identifier and common GitHub/private-key signatures. Only the intentionally
  public author email in `CITATION.cff` matched; no matching local-home path or
  obvious credential was found. This is a bounded signature scan, not a security
  certification or an audit of all Git history and binary metadata.
- No non-placeholder raw/intermediate/processed inputs, model binaries,
  record-level prediction arrays or compressed raw archives were included.
- Python compilation and `git diff --check` passed. Git emitted an informational
  LF-to-CRLF warning for `.gitignore`, not a whitespace failure.
- Seven documentation tests were rerun after the status edits and passed. They
  are already included in the 159-test total, not seven additional unique tests.

## Release boundary

Release packaging is separate from the successful experimental reconstruction
and working-tree checks. No Git commit, new GitHub release or updated software
DOI is claimed by this record. Manuscript writing remains paused.
The current corrected sources must be published as a new version before an old
immutable software DOI can be replaced in citation metadata.
