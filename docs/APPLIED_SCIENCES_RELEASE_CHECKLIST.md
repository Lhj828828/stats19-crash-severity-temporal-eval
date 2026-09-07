# Applied Sciences release and submission checklist

This checklist covers the computational repository and the current manuscript
target. It is not evidence of editorial approval, acceptance or guaranteed
scope fit. Recheck the live journal instructions immediately before submission.

Official pages checked on 2026-09-02:

- <https://www.mdpi.com/journal/applsci/about>
- <https://www.mdpi.com/journal/applsci/instructions>

## Journal fit and article type

- [ ] Submit as an Article (the journal's original-research manuscript type)
      unless the editorial office advises a different article type.
- [ ] Explain the applied-computing contribution, not merely the STATS19 case
      study: leakage-aware temporal evaluation, ordered-error reporting,
      uncertainty and explanation-stability assessment.
- [ ] Select the most defensible journal section at submission. The journal
      currently lists both Computing and Artificial Intelligence and
      Transportation and Future Mobility within scope; section choice is not
      itself evidence that the paper will pass editorial screening.
- [ ] Do not claim a new prediction algorithm or cross-national
      generalizability. CAS is a separate workflow replication with mixed
      directional findings.

## Manuscript structure

- [ ] Use the current Applied Sciences Word or LaTeX template, or confirm that
      the initial free-format route remains available.
- [ ] Include title, full author names, complete affiliations and one clearly
      identified corresponding author.
- [ ] Keep the abstract to about 200 words maximum, as a single paragraph
      covering background, methods, results and conclusion without headings.
- [ ] Provide 3-10 specific keywords.
- [ ] Include Introduction, Materials and Methods, Results and Discussion;
      Conclusions may be separate when useful.
- [ ] Define every abbreviation independently on first use in the abstract,
      main text, and first figure or table where it appears.
- [ ] Report software names and versions, code availability, split rules,
      frozen test access, uncertainty procedures and all documented numerical
      deviations accurately enough for reproduction.
- [ ] For a Word submission, insert figures in the main text after the
      paragraph containing their first citation.
- [ ] Check that the complete submission package remains below the journal's
      current total-file limit; the instructions checked on 2026-09-02 state
      120 MB.

## Required back matter and ethics

- [ ] Add Supplementary Materials, Author Contributions using CRediT roles,
      Funding, Data Availability Statement, Acknowledgments, Conflicts of
      Interest and References as applicable.
- [ ] Include an explicit Data Availability Statement even though raw data are
      not stored in Git. Distinguish official source data, the separate fixed
      input archive, GitHub code and the version-specific Zenodo archive.
- [ ] Determine and justify the Institutional Review Board and Informed Consent
      statements for analysis of de-identified administrative collision data;
      do not insert "Not applicable" without checking the data terms and
      institutional requirements.
- [ ] Disclose any generative-AI assistance required by the live MDPI policy.
      The disclosure must describe actual use and must not list an AI system as
      an author. Superficial language editing may be treated differently under
      the current instructions, so verify the final wording at submission.
- [ ] Confirm funding information and any role of sponsors.
- [ ] State conflicts of interest explicitly, including a no-conflict statement
      when accurate.
- [ ] Number references in order of first appearance and include complete,
      consistently formatted bibliographic information. Add DOI values where
      available.

## Cover letter and submission declarations

- [ ] Prepare a concise cover letter explaining significance, novelty relative
      to existing work and fit with the selected Applied Sciences section.
- [ ] Include the required statements that the manuscript is not published or
      under consideration elsewhere and that all authors approve submission to
      Applied Sciences.
- [ ] Declare any prior submission of the manuscript to an MDPI journal and
      provide the previous manuscript identifier in the submission system when
      applicable.
- [ ] Enter proposed or excluded reviewers in the submission system rather than
      the cover letter.
- [ ] Confirm every author has reviewed the final manuscript, authorship order,
      affiliations, contribution statement and public display of email details.

## Historical public repository release gate

These checked items describe earlier released implementations, not completion
of the 15-feature revision or its final clean reconstruction. The current gate
below supersedes them for the manuscript and next software release.

- [x] Keep raw STATS19 and CAS records, model binaries, record-level
      predictions, local environments and private working files out of Git.
- [x] Run author-path scanning, JSON parsing and all unit tests.
- [x] Complete a fresh public STATS19 D1-D14 reconstruction from the seven
      verified annual inputs in an isolated workspace.
- [x] Complete the isolated CAS public workflow from the fixed JSONL snapshot.
- [x] Verify all compact results against the tracked scientific contracts.
- [x] Review STATS19 redistribution and attribution requirements under the Open
      Government Licence v3.0; retain the dated evidence record.
- [x] Complete the separate CAS redistribution and attribution review under CC
      BY 4.0; retain the dated evidence record.
- [x] Publish immutable fixed-input archive URLs only after all included source
      terms and attribution requirements have been satisfied.
- [x] Update the data manifest and verify every file hash from a clean download
      directory (9/9 passed on 2026-09-04).
- [x] Create the `v1.1.0` release from commit
      `f49148494f56ad32227dc3d7522d02e4b8d280ce` without moving or rewriting
      the existing `v1.0.0`-`v1.0.2` tags.
- [x] Record the GitHub release
      <https://github.com/Lhj828828/stats19-crash-severity-temporal-eval/releases/tag/v1.1.0>,
      Zenodo version DOI <https://doi.org/10.5281/zenodo.22303858> and Zenodo
      concept DOI <https://doi.org/10.5281/zenodo.22231696>.
- [x] Update `CITATION.cff` and README to cite the `v1.1.0` computational
      artifact.
- [ ] Update the manuscript Data Availability Statement to cite the same
      version-specific artifact before submission.

## Final factual checks

### Post-review random-reference correction gate

The historical reproduction checks above remain records of the historical
implementation. They do not establish independence of random internal tests
from the temporal hyperparameter-selection process identified on 2026-09-06.

- [x] Add a split-local correction entry point and pass 16 invariant tests,
      including direct overlap rejection and test-label perturbation.
- [x] Freeze separate STATS19/CAS correction protocols before corrected fits,
      explicitly recording that legacy outcomes were already known.
- [x] Complete all ten split-local random LightGBM models and dependent test,
      bootstrap and SHAP outputs; require both correction completion markers.
- [x] Check that historical temporal models, H2 and all protected artifacts
      remain unchanged; retain legacy random results as labeled comparisons.
- [ ] Synchronize manuscript methods, results, figures and captions with the
      corrected selection procedure and results, regardless of direction.
      Methods and outline are synchronized; the original Visio workflow and
      final manuscript results/captions still require integration.
- [x] Independently check 1,300 intervals, SHAP samples/ranks, and historical
      compact results (21/21 comparisons and 7/7 required files).
- [x] Record the broader regression outcome honestly: 27/29 standalone files
      passed. Two legacy D14 provenance-equality checks stop on historical
      path-normalization differences; do not rewrite their frozen hashes.
- [ ] Publish the verified correction as a new software release/version DOI.
      Do not claim that the immutable v1.2.0 archive contains the correction.

### Current 15-feature revision gate (7 September 2026)

- [x] Complete and independently verify all six corrected STATS19 model sets,
      eleven evaluation groups, paired Bootstrap contrasts and SHAP summaries.
- [x] Refit and verify the exclusion-2020 sensitivity at fixed corrected-main
      parameters; keep the observed fatal-recall trade-off in the conclusions.
- [x] Select the current results in `config/final_analysis/result_catalog.json`
      and 20 compact tables. These are source tables, not 20 manuscript tables.
- [x] Omit Ordered Logit, matched subsets and the tree-count extension from the
      revised manuscript while preserving their historical artifacts. Continue
      to disclose the 1200-round search boundary and post-review timing.
- [x] Add the explicit current entry in `FINAL_ANALYSIS.md`; check separate
      raw-to-splits preparation for STATS19/CAS without full model fitting.
- [x] Pass the separate bounded historical D14 snapshot check. Keep both original
      exact-equality assertions and frozen hashes unchanged; do not claim the
      original two assertions passed or the unavailable outline was verified.
- [ ] Align the manuscript, numbered figures/tables and Visio with the current
      15-feature catalog. Earlier method/outline edits concern an older scope.
- [x] Run the final full reconstruction in clean isolated environments and
      compare the selected compact results. Completed on 7 September 2026:
      STATS19 22 stages, CAS 25 stages, all 20 V2 tables passed. See
      `docs/FINAL_REPRODUCTION_VALIDATION.md` and
      `logs/final_analysis/final_reproduction_acceptance.json`. This is author-run
      same-machine Windows evidence, using verified existing raw snapshots.
- [x] Pass 159 applicable unit tests and seven standalone CAS audits; verify
      artifact seals and inspect four regenerated diagnostic PNGs. Keep the two
      historical D14 exact-hash failures separate from the passing test count.
- [ ] Visually inspect final manuscript figures later; distinguish layout QA
      from numerical experiment and reproduction verification.
- [ ] Scan and release the corrected public sources with a new software DOI,
      then update the README, citation metadata and manuscript availability
      statement together. Do not repurpose an older immutable version DOI.

### Submission checks

- [ ] Recheck the current Applied Sciences aims, section list, instructions,
      article processing charge, licensing terms and file limits on the day of
      submission.
- [ ] Confirm that the manuscript reports H1 as metric-dependent, H2 as a
      Macro-F1/fatal-recall trade-off and H3 as descriptive SHAP attribution,
      not causal evidence.
- [ ] State that spatial generalization was not evaluated and that two
      administrative datasets do not prove universal or cross-national
      generalizability.
- [ ] Keep the repository MIT license for code distinct from the journal's
      publication license for the article.
