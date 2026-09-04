# v1.1.0: immutable public data retrieval

Release date: 2026-09-04

This release closes the fixed-input availability gate for the public
reconstruction workflow. It does not change the frozen samples, feature rules,
model specifications, evaluation protocols, reported results or scientific
conclusions.

## Data availability

- UK STATS19 2018-2024 fixed annual snapshots:
  <https://doi.org/10.5281/zenodo.22290566>, Open Government Licence v3.0.
- New Zealand CAS 2022-2025 fixed injury-crash snapshot:
  <https://doi.org/10.5281/zenodo.22296725>, CC BY 4.0.
- Raw data remain outside Git and outside the MIT software licence.
- The manifest uses version-specific Zenodo file URLs, not mutable upstream
  endpoints.

## Reproduction changes

- Enabled `download_and_verify_data.py download --dataset all`.
- Added finite retry and HTTP Range resume support for interrupted downloads.
- Retained fail-closed byte-size and SHA-256 verification before any file is
  accepted as an analysis input.
- Added published-record metadata, archive README files and machine-checkable
  archive checksum lists.
- Updated README, provenance, reproduction and release-gate documentation.

## Validation

On 2026-09-04, all nine analysis-input files were downloaded from the two
version-specific Zenodo records into a new directory and passed the frozen size
and SHA-256 checks (`9/9 PASS`). The existing clean-environment STATS19 D1-D14
and CAS workflow validations remain the scientific execution evidence. This is
an author-run release validation, not a claim of independent third-party
replication.

## Software citation

The software concept DOI is <https://doi.org/10.5281/zenodo.22231696>. The
version-specific software DOI for `v1.1.0` is
<https://doi.org/10.5281/zenodo.22303858>.
