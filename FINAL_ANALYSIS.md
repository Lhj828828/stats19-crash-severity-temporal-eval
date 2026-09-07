# Final corrected analysis

Target: Applied Sciences. Status: the corrected primary and retained sensitivity
experiments are complete and locally verified. This revision is published as
`v1.3.0`, DOI <https://doi.org/10.5281/zenodo.22642853>. On 7 September 2026
the author requested that experimental
and reproduction validation take priority over manuscript work. The final clean
raw-to-results reconstruction subsequently passed for both datasets that day,
before manuscript table or figure integration. See
[the validation record](docs/FINAL_REPRODUCTION_VALIDATION.md).

## One current entry point

```text
python run_final_analysis.py --stage plan
python run_final_analysis.py --stage check
```

Neither command trains a model. The check validates completed source artifacts,
the selected result catalog and the separately documented historical D14
provenance boundaries. `--stage package` verifies/assembles the compact tables
without running the historical check. Re-running package never overwrites a
changed reference catalog; a changed final scope requires a new version.
`check` requires existing analytical artifacts and is intended for author-side
auditing. A first-time reader should follow the isolated reconstruction commands,
not expect `check` to work before models and results have been generated.

## Authoritative results

Use `config/final_analysis/result_catalog.json` and the 20 tables under
`results/final_analysis/v2/`. Names are descriptive, not final manuscript numbers.

Catalog V2 corrects a V1 export-scope error: `cas_h2` contains only the nine
`temporal_2025` contrast rows. V1 also contained 90 superseded random-reference
contrasts, inconsistent with its stated temporal-only purpose. No scientific
source result was changed. The V1 catalog is archived byte-for-byte under
`config/final_analysis/history/result_catalog_v1.json`; its original 20 exports
are retained outside `v2/`. The other 19 V2 exports are byte-identical to V1.
The assembler checks these preservation and scope conditions on every run.

| Evidence | Authoritative source |
| --- | --- |
| STATS19 primary, H1/H2/H3, errors | `results/stats19_feature_revision/full/` |
| STATS19 exclusion-2020 sensitivity | `results/stats19_feature_revision/exclude2020/` |
| STATS19 label recording diagnostics | `logs/stats19_feature_revision/recording_system_severity_counts.csv` |
| CAS temporal models and H2 | temporal rows in `results/cas_evaluation/` and `results/cas_post_analysis/h2_bootstrap.csv` |
| CAS unchanged random baselines | Dummy/Logistic random rows in `results/cas_evaluation/test_metrics.csv` |
| CAS corrected random LightGBM, H1/H3 | `results/random_reference_revision/cas/` |
| CAS post-hoc field removals | `results/cas_feature_ablation/` |

Do not use the old STATS19 17-feature tables as revised main results. Do not use
the old CAS random LightGBM rows when corrected rows are available. The catalog
filters those rows out; main CAS baseline and temporal results stay unchanged.
Do not copy old cross-dataset conclusion files as conclusions about the final
combination. The data sources have different labels, features and SHAP sampling
designs, so cross-country universality or directly comparable rho is not claimed.

Omit Ordered Logit/matched subsets and the tree-count extension from the revised
manuscript; retain their audit history. Keep the disclosure that the corrected
search stopped at 1200 rounds. Keep the post-review timing disclosure, the
police-recorded label definition, recording-system limitation, fixed-model
Bootstrap boundary and absence of empirical spatial generalization.

## Isolated reconstruction

First download and verify the immutable data snapshots using the existing
`download_and_verify_data.py`. Use Python 3.13 and the locked dependencies.
The new entry point prepares separate STATS19/CAS directories and defaults to
building clean virtual environments. A separate workspace path is mandatory:

```text
python run_final_analysis.py --stage prepare --dataset all --workspace ../final-reconstruction
python run_final_analysis.py --stage reproduce --dataset all --workspace ../final-reconstruction --resume
```

These are two different operations: `prepare` copies verified inputs, source,
fixed-parameter templates and small comparison references only; `reproduce`
actually performs data processing and model fitting. No fitted models,
processed tables, record predictions or SHAP arrays are copied. Expected compact
results are stored separately under `config/final_reference/`, never used for
training or selection. The reproduction command may take hours and is not run
by `check`. Omit `prepare` to let `reproduce` prepare a new directory itself.

The STATS19 plan reuses D1-D7 preprocessing and goes directly to the 15-feature
branch. It does not execute Ordered Logit, matched-subset or tree-count extension
stages. Each random split selects its own settings. The exclusion-2020 branch
refits only the two primary comparators at fixed temporal-main parameters.
CAS retains its separate audited pipeline and post-hoc field removals, then runs
the split-local random correction. Legacy CAS random outputs are generated only
as required diagnostic inputs to that correction, not as final manuscript results.

The runtime binding module records hashes of newly generated preparation and
execution protocols instead of pretending they are the historical author hashes.
The scientific kernels and selection rules are unchanged. This binding is allowed
only inside a prepared reconstruction root; it never edits archived protocols.
Resumption checks immutable inputs and a stage-prefix ledger. Eight threads is
the maximum. `--use-current-environment` is for engineering diagnostics only and
must not be described as a clean-environment reproduction.

The final keyed comparison uses the already selected compact references, with
integer/text fields exact and floating values at `atol=1e-10`, `rtol=1e-8`.
Failures stop completion and require diagnosis. No tolerance may be widened or
reference replaced merely to manufacture a pass. A new numerical-deviation
record may be needed if a supported platform yields different outputs.

The full-scale clean reconstruction completed on 7 September 2026: all 22
STATS19 stages and all 25 CAS stages passed, followed by all 20 V2 numerical
comparisons. STATS19 was prepared using V1 references; its eleven tables are
unchanged in V2. CAS was prepared with V2. Final acceptance checked both data
sources against V2, retaining this catalog-only correction history.

The machine-readable result is
`logs/final_analysis/final_reproduction_acceptance.json`, with status
`PASS_CLEAN_RAW_RECONSTRUCTION_AND_V2_RESULTS`. The frozen catalog and original
preparation markers retain their creation-time reproduction flags; this separate
acceptance record is the evidence of subsequent completion. Do not rewrite
those snapshots to update a historical status field.

This was author-run on the same Windows machine in two newly created virtual
environments, not an independent or cross-platform reproduction. It reused
9/9 size/hash-verified raw files rather than downloading new copies. Models,
processed data, predictions and SHAP arrays were generated afresh.
The earlier bounded preparation check in
`logs/final_analysis/preparation_check.json` remains a separate, non-training
record. All four regenerated diagnostic PNGs were nonblank and visually inspected;
final manuscript layout has not been verified.

## Local checks completed on 7 September 2026

- 159 current-entry, scientific, random-correction and public-interface unit
  tests passed using a clean interpreter, including 32 current-entry tests and
  75 focused main/exclusion-2020 tests. Toy fits are separate from the completed
  full-scale reconstruction.
- Seven applicable standalone CAS audits passed on regenerated outputs.
- The current result catalog and bounded historical D14 verification passed.
- The temporary raw-to-splits checks described above passed for both datasets.
- Python compilation and whitespace checks passed. The public-candidate scan
  found no matching author-home paths or obvious credentials, and no raw/model
  artifacts or files exceeding 10 MiB. The only contact-identifier match was the
  intentionally public author email. All 327 candidate JSON files parsed.
  This does not certify security or audit historical Git objects; see the
  validation record for scan scope and release boundaries.

Current transcript names and test scope are listed in
`docs/FINAL_REPRODUCTION_VALIDATION.md`; `.log` files are local and Git-ignored.
The scientific regression emitted third-party deprecation warnings but no
failures. The two original historical D14 tests are not included in the 159
passing tests or relabeled as passing below.

## Historical D14 checks

```text
python code/historical_d14_validation.py
```

This checks the normalized historical snapshot, not the new analysis. The exact
allowed hash transitions are enumerated in the checker and tied to the earlier
public-path/provenance migration commits. All other changes fail. Scientific
settings must match, and saved predictions, contrasts and error structures are
recomputed. The undistributed working outline is identity-recorded only; its
bytes and original timestamp are not independently verified. The historical
configuration inventory is checked as a preserved subset, not equated to later
added filenames. Original tests, manifests and failures are retained unchanged.

The result is written to `logs/final_analysis/historical_d14_validation.json`.
It explicitly distinguishes the passing bounded check from the two old raw
exact-equality assertions, which do not apply to the normalized snapshot.

## Remaining sequence

The corrected sources are committed and published as `v1.3.0` from
`962377ca6cbd0ca12c06bb801aaaf531b3c62ce7`. Zenodo archive DOI
<https://doi.org/10.5281/zenodo.22642853> contains the fixed tag. All 808 archived
files were downloaded and matched against that Git tree. DOI backfill is a
separate main-branch documentation commit; the released tag is not moved.

Manuscript integration remains paused. No further model training is required
under the current frozen scope. Layout-only edits do not require retraining;
any scientific change needs a separately documented validation.
