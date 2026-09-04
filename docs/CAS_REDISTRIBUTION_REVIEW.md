# CAS public-data redistribution review

- Review date: 2026-09-03
- Status: **REVIEW_COMPLETE_REDISTRIBUTION_PERMITTED_WITH_ATTRIBUTION**
- Scope: the two fixed 2022-2025 CAS snapshot files listed in
  `config/public_data_manifest.json`
- Limitation: this documented project-governance assessment is not legal advice

## Material reviewed

1. Waka Kotahi NZ Transport Agency, *Crash Analysis System (CAS) data*,
   accessed 3 September 2026:
   https://opendata-nzta.opendata.arcgis.com/datasets/NZTA::crash-analysis-system-cas-data-1/about
2. ArcGIS item metadata for item `8d684f1841fa4dbea6afaefc8a1ba0fc`,
   accessed 3 September 2026:
   https://www.arcgis.com/sharing/rest/content/items/8d684f1841fa4dbea6afaefc8a1ba0fc?f=pjson
3. Waka Kotahi, *CAS Data Public* service metadata, accessed 3 September
   2026:
   https://services.arcgis.com/CXBb7LAjgIIdcsPt/arcgis/rest/services/CAS_Data_Public/FeatureServer?f=pjson
4. Waka Kotahi, public CAS layer metadata, accessed 3 September 2026:
   https://services.arcgis.com/CXBb7LAjgIIdcsPt/arcgis/rest/services/CAS_Data_Public/FeatureServer/0?f=pjson
5. Creative Commons, *Attribution 4.0 International (CC BY 4.0)*,
   accessed 3 September 2026:
   https://creativecommons.org/licenses/by/4.0/

## Factual findings

- The ArcGIS item is publicly accessible and identifies Waka Kotahi NZ
  Transport Agency as the owner/provider of the Crash Analysis System data.
- The official item metadata states the licence as Creative Commons
  Attribution 4.0 International (CC BY 4.0).
- CC BY 4.0 permits sharing and adaptation, including commercial reuse,
  provided appropriate credit is given, a link to the licence is supplied,
  and changes are indicated. Additional restrictions may not be imposed.
- The official description states that the public data contain non-personal
  fields and are an appropriately confidentialised open version of CAS.
- CAS is a live, monthly updated service. Its records may be revised after the
  local retrieval date, so a new API response is not an exact replacement for
  the analysis snapshot.
- The service supports attribute queries and data export. The local snapshot
  contains attributes only and no geometry.
- The crash-severity field records the most severe injury outcome associated
  with a crash. It is not assumed to have the same statistical definition as
  the STATS19 severity label.
- The source supplies the data on an "as is, where is" basis and identifies
  data-quality limitations, including incomplete 2020 and 2021 data. Those
  years are outside this project's 2022-2025 CAS analysis window.

## Local snapshot scope and changes

The project queried the live public attribute service on 28 August 2026 and
froze records satisfying:

```text
crashYear BETWEEN 2022 AND 2025 AND crashSeverity <> 'Non-Injury Crash'
```

The resulting snapshot contains 43,121 records, each represented with the same
70-field schema:

- 2022: 10,972 records;
- 2023: 10,962 records;
- 2024: 10,645 records;
- 2025: 10,542 records;
- Minor Crash: 33,635 records;
- Serious Crash: 8,339 records;
- Fatal Crash: 1,147 records.

The proposed archive contains two project-created serialisations of those
queried public attributes:

1. `cas_injury_2022_2025_snapshot.jsonl.gz`, the authoritative lossless input,
   2,771,733 bytes, SHA-256
   `7db99dd4ba92716d751dabbc08b03f72373c635025b7d5335cf0b6705a7bd7f3`;
2. `cas_injury_2022_2025_snapshot.csv.gz`, an inspection copy, 1,867,440
   bytes, SHA-256
   `cc554366351cf4d5ccc7207f583c8ff1437036a615f1bac4dcdcccac76d1745c`.

The filtering, exclusion of geometry, field serialisation and gzip
compression are project modifications and must be stated. The files must not
be described as an unchanged official release or as a current copy of the
live service.

## Decision and conditions

The two specified CAS snapshot files may be deposited in a public immutable
data archive under CC BY 4.0. Publication is conditional on all of the
following:

1. Select Zenodo licence `cc-by-4.0` (Creative Commons Attribution 4.0
   International) for the CAS deposit.
2. Include this attribution in the archive description and accompanying
   README:

   > Source data: Waka Kotahi NZ Transport Agency, Crash Analysis System
   > (CAS), licensed under CC BY 4.0. Fixed attribute snapshot retrieved 28
   > August 2026 from the public CAS service.

3. Link the official CAS landing page and the CC BY 4.0 licence text.
4. Identify the snapshot scope (2022-2025 injury crashes), query filter,
   retrieval date, format conversions and both file SHA-256 values.
5. Preserve the source's live-data, quality and "as is, where is" caveats.
6. State that Waka Kotahi does not endorse this archive, the software or the
   associated research.
7. Keep the repository's MIT software licence separate from the CAS data's
   CC BY 4.0 licence.
8. Identify Waka Kotahi as the underlying data provider. If Haojie Liu is
   listed as a creator, describe the role as snapshot compiler or depositor,
   not creator of the underlying crash records.

## Residual boundary

This decision covers only the two files and hashes listed above. It does not
make a mutable CAS API response reproducibly equivalent to the paper input,
does not transfer rights in third-party marks, and does not imply Waka Kotahi
endorsement. The publication condition was satisfied on 4 September 2026: the
two files were published in Zenodo record 22296725 under CC BY 4.0 and the
deposited bytes were verified against the frozen SHA-256 values.
