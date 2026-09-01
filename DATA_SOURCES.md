# Data sources and provenance

This file records the public sources and frozen snapshots used by the
STATS19 primary analysis and the independent CAS replication. Raw snapshots
are intentionally excluded from the Git repository because the STATS19 source
file is large and the CAS service is live. The checksums below identify the
exact files used in the completed analyses.

## UK STATS19 (primary analysis)

- Maintainer/source: UK Department for Transport (DfT), Road Safety Open Data.
- Landing page:
  https://www.gov.uk/government/statistical-data-sets/road-safety-open-data
- Complete collision file used:
  https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-1979-latest-published-year.csv
- Retrieved: 2026-08-27 (local time recorded in `logs/d1_file_manifest.csv`).
- File used: `data/raw/downloads/dft-road-casualty-statistics-collision-1979-latest-published-year.csv`
- Size: 1,534,937,928 bytes.
- SHA-256:
  `4b60aac426b8fb7771dc9a3fc3e383399e88c9a49041544926aec4bea409e366`
- Analysis years: 2018-2024. The source file contains additional years.

The annual files in `data/raw/collisions/` were copied byte-for-byte from the
complete source file. Their individual checksums and row counts are recorded
in `logs/d1_file_manifest.csv`. The DfT/GOV.UK source terms in force at the
time of reuse apply; check the current landing page before redistributing a
fresh copy.

Required documentation snapshots are listed in the same manifest and are
downloaded by `code/d1_acquire_and_audit.py` when absent.

## New Zealand CAS (independent replication)

- Maintainer: Waka Kotahi NZ Transport Agency, Crash Analysis System (CAS).
- Service metadata:
  https://services.arcgis.com/CXBb7LAjgIIdcsPt/arcgis/rest/services/CAS_Data_Public/FeatureServer?f=pjson
- Layer/query endpoint:
  https://services.arcgis.com/CXBb7LAjgIIdcsPt/arcgis/rest/services/CAS_Data_Public/FeatureServer/0/query
- Field descriptions:
  https://opendata-nzta.opendata.arcgis.com/pages/cas-data-field-descriptions
- Snapshot retrieved: 2026-08-28 (local time recorded in
  `logs/cas/cas_source_manifest.csv`).
- Snapshot file: `data/raw/cas/cas_injury_2022_2025_snapshot.csv.gz`
- Rows: 43,121 injury crashes from 2022-2025.
- SHA-256:
  `cc554366351cf4d5ccc7207f583c8ff1437036a615f1bac4dcdcccac76d1745c`
- License stated by the source metadata: CC BY 4.0 International.
- Required attribution: Waka Kotahi NZ Transport Agency, Crash Analysis
  System (CAS), with the source URL and snapshot date.

CAS is a live service and records can change after publication. The CAS
severity label is the worst injury recorded for a crash and is not assumed to
be identical to the STATS19 severity label. The two datasets were trained and
evaluated separately; records, absolute metrics and SHAP ranks were not
pooled.

## Recreating the local data layout

1. Download the STATS19 complete collision file to
   `data/raw/downloads/` using the exact filename above.
2. Run `python code/d1_acquire_and_audit.py` to extract the annual files and
   refresh the D1 provenance log.
3. Download the CAS snapshot from the documented endpoint, or use the
   archived snapshot supplied with a future DOI release, and place it at the
   exact path above.
4. Verify SHA-256 values against the manifests before running later stages.

Do not combine the two datasets or treat their class labels as exchangeable.
