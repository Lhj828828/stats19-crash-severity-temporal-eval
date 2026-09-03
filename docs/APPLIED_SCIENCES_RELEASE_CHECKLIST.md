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

## Public repository release gate

- [x] Keep raw STATS19 and CAS records, model binaries, record-level
      predictions, local environments and private working files out of Git.
- [x] Run author-path scanning, JSON parsing and all unit tests.
- [x] Complete a fresh public STATS19 D1-D14 reconstruction from the seven
      verified annual inputs in an isolated workspace.
- [x] Complete the isolated CAS public workflow from the fixed JSONL snapshot.
- [x] Verify all compact results against the tracked scientific contracts.
- [ ] Publish immutable fixed-input archive URLs only after reviewing source
      terms and attribution requirements.
- [ ] Update the data manifest and verify every file hash from a clean checkout.
- [ ] Create a new versioned release rather than moving or rewriting the
      existing `v1.0.0`-`v1.0.2` tags.
- [ ] Record the exact Git commit, version tag, GitHub release URL, Zenodo
      version DOI and Zenodo concept DOI.
- [ ] Update `CITATION.cff`, README and the manuscript Data Availability
      Statement to cite the same version-specific computational artifact.

## Final factual checks

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
