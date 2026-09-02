# Reproducibility boundary

This document separates the public third-party workflow from the retained
author-side forensic comparison. The distinction prevents a compact software
archive from implying that excluded raw data or multi-gigabyte record-level
artifacts are present.

## Public software archive

The public GitHub and Zenodo software archive is intended to contain:

- analysis code and tests;
- frozen feature, split, model and reporting configurations;
- dependency locks and cross-platform text-format rules;
- data download, provenance and SHA-256 verification utilities;
- compact result tables, figures and expected scientific summaries;
- README, citation, license and reproduction documentation.

It intentionally excludes:

- raw STATS19 and CAS data files;
- fitted model binaries;
- record-level predictions;
- large SHAP and bootstrap arrays;
- rebuildable intermediate tables and local environments.

These exclusions reduce redistribution and repository-size risks. They do not
remove the need to make the exact analysis data snapshots permanently
identifiable and obtainable under their applicable source terms.

## Three verification levels

1. **Public reconstruction** downloads and verifies the fixed input snapshots,
   creates a clean environment and regenerates the analysis outputs.
2. **Public scientific verification** compares regenerated compact tables and
   decision summaries with the tracked references under
   `config/public_result_reference/` and the documented tolerances in
   `config/public_result_contract.json`.
3. **Author-side forensic comparison** additionally compares large
   record-level predictions, arrays and other local frozen artifacts. The
   retained D16 workflow performs this level and is not the public entry point.

The public workflow must not require author-local paths, untracked reference
files or byte hashes that change solely because of Git line-ending conversion.

## Author-local path normalization

Tracked text uses repository-relative paths or explicit placeholders such as
`PUBLIC_SOFTWARE_PROJECT` and
`AUTHOR_SIDE_REPRODUCTION_WORKSPACE_NOT_DISTRIBUTED`. These placeholders record
that an author-side location existed without exposing a workstation username,
drive layout or desktop folder.

SHA-256 values retained inside historical D14-D16 execution reports describe
the author-side artifacts used when those reports were created, before public
path normalization. They are provenance records, not claims that the current
path-normalized text files have the same byte hashes. Fresh public runs bind
newly generated protocols and manifests to their current hashes. The CAS
cross-dataset protocol instead uses the portable STATS19 copies under
`references/stats19_upstream/` and verifies those copies directly.

Path normalization changes location strings and therefore file bytes, but it
does not change samples, predictions, metrics, uncertainty estimates, model
parameters or scientific conclusions.
## Version claims

The `v1.0.2` archive records the frozen code, protocols and compact results and
documents an author-run isolated reproduction. It is not yet a one-command
third-party reconstruction package because the fixed raw snapshots are not in
that software archive and the retained D16 comparator expects excluded local
artifacts.

The development branch now contains an isolated public STATS19 runner. It has
passed a fresh annual-input-to-D7 execution without author artifacts. This is
an implementation milestone, not yet evidence that the entire public workflow
is complete.

The planned `v1.1.0` release may be described as a third-party reproduction
workflow only after all of the following pass from a clean downloaded release:

- fixed inputs can be obtained and their SHA-256 values verified;
- the public reconstruction completes without author-local artifacts;
- compact scientific outputs pass the public verification contract;
- the documented commands work on the supported operating systems;
- the data availability statement matches the actual licensing and archive
  arrangement.

Numerical equivalence must be reported using the frozen scientific tolerances;
byte-identical floating-point outputs are not assumed across unsupported
platforms or library versions.
