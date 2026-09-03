# STATS19 public-data redistribution review

- Review date: 2026-09-03
- Status: **REVIEW_COMPLETE_REDISTRIBUTION_PERMITTED_WITH_ATTRIBUTION**
- Scope: the seven 2018-2024 collision CSV snapshots listed in
  `config/public_data_manifest.json`
- Limitation: this documented project-governance assessment is not legal advice

## Material reviewed

1. UK Department for Transport (DfT), *Road safety open data*, last updated
   30 July 2026, accessed 3 September 2026:
   https://www.gov.uk/government/statistical-data-sets/road-safety-open-data
2. DfT, *Road safety statistics: guidance*, last updated 27 November 2025,
   accessed 3 September 2026:
   https://www.gov.uk/guidance/road-accident-and-safety-statistics-guidance#data-protection-and-privacy
3. DfT, *Personal information and data protection*, last updated 24 January
   2023, accessed 3 September 2026:
   https://www.gov.uk/guidance/personal-information-and-data-protection#statistical-uses-of-the-police-recorded-personal-injury-road-accident-data
4. The National Archives, *Open Government Licence for public sector
   information, version 3.0*, accessed 3 September 2026:
   https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/
5. Zenodo licence vocabulary entry `ogl-uk-3.0`, accessed 3 September 2026:
   https://zenodo.org/api/vocabularies/licenses/ogl-uk-3.0

## Factual findings

- The DfT open-data page describes the published collision files as containing
  the non-sensitive fields that can be made public.
- DfT's privacy guidance explicitly states that a limited subset of STATS19 is
  released as open data under an Open Government Licence. Sensitive fields are
  not routinely published and require a separate approved-researcher process.
- The open-data page applies the Open Government Licence v3.0 except where
  otherwise stated; no different licence is stated for the collision CSV.
- OGL v3.0 grants a worldwide, royalty-free, perpetual and non-exclusive right
  to copy, publish, distribute, transmit and adapt covered information, for
  commercial or non-commercial use.
- Reuse requires source acknowledgement and, where possible, a link to the
  licence. If the provider supplies no special attribution statement, the OGL
  specifies: "Contains public sector information licensed under the Open
  Government Licence v3.0."
- OGL v3.0 excludes personal data, unpublished information, specified official
  marks and third-party rights that the provider cannot license. It also
  prohibits implying official status or provider endorsement.
- Zenodo exposes `ogl-uk-3.0` as a data-compatible licence identifier.

## Local snapshot scope

The source was the DfT complete collision CSV retrieved on 27 August 2026:

- source size: 1,534,937,928 bytes;
- source SHA-256:
  `4b60aac426b8fb7771dc9a3fc3e383399e88c9a49041544926aec4bea409e366`;
- official source URL:
  https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-1979-latest-published-year.csv

The seven analysis files total 141,801,866 bytes. They are derived fixed
snapshots produced by selecting records whose `collision_year` is 2018 through
2024. The extraction copied the source header and each selected row
byte-for-byte. The archive must therefore describe them as derived annual
snapshots from the frozen complete file, not as unaltered official standalone
annual releases.

The proposed data archive contains only these already-public collision fields.
It must not include restricted STATS19 variables obtained through an
application, model outputs, private records, or DfT logos. The archive must not
imply DfT endorsement or official status.

## Decision and conditions

The seven specified STATS19 snapshots may be deposited in a public immutable
data archive under OGL v3.0. Publication is conditional on all of the
following:

1. Select Zenodo licence `ogl-uk-3.0` (Open Government Licence v3.0).
2. Include this notice in the archive description and accompanying README:

   > Contains public sector information licensed under the Open Government
   > Licence v3.0. Source: UK Department for Transport, Road Safety Open Data
   > (source snapshot retrieved 27 August 2026).

3. Link both the DfT source page and the OGL v3.0 text.
4. State the filtering operation, retrieval date, source-file SHA-256 and all
   seven output SHA-256 values; do not call the files official annual releases.
5. State that DfT does not endorse this archive or the associated research.
6. Keep the software MIT licence separate from the data's OGL v3.0 licence.
7. Identify UK DfT as the underlying information provider. If Haojie Liu is
   listed as a creator, describe the role as snapshot compiler or depositor,
   not creator of the underlying collision records.

## Residual boundary

This decision covers STATS19 only. The separate review for the New Zealand CAS
snapshot is recorded in `docs/CAS_REDISTRIBUTION_REVIEW.md`. Immutable download
URLs must remain unset until the relevant archive has actually been published
and its uploaded files have been verified against the frozen SHA-256 values.
