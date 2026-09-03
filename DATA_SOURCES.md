# Data sources and provenance

This file records the public sources and frozen snapshots used by the
STATS19 primary analysis and the independent CAS replication. Raw snapshots
are intentionally excluded from the Git repository because the STATS19 source
file is large and the CAS service is live. The checksums below identify the
exact files used in the completed analyses.

The machine-readable public input contract is
`config/public_data_manifest.json`. It identifies seven annual STATS19 files
and two CAS snapshot files by path, row count where applicable, byte size and
SHA-256. Together these are the intended compact data archive; the 1.53 GB
full-history STATS19 source file is retained only as provenance and is not
required by the public workflow.

Until each dataset has passed its redistribution review and an immutable data
record has been published, its download URLs remain deliberately unset. The
following commands are already available:

```text
python download_and_verify_data.py list
python download_and_verify_data.py verify --dataset all
```

The `download` command fails closed while immutable URLs are unset. It never
silently substitutes the mutable STATS19 `latest` file or a fresh CAS API
query for the snapshots used in the paper.

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

The annual files in `data/raw/collisions/` are derived fixed snapshots. They
were created by selecting records by `collision_year` from the complete source
file; the header and every selected CSV row were copied byte-for-byte. They are
not represented as the official standalone annual release files. Their
individual checksums, sizes and row counts are frozen in
`config/public_data_manifest.json`; the original extraction record is retained
in `logs/d1_file_manifest.csv`.

DfT states that the public download contains the non-sensitive fields that can
be made public and separately confirms that this limited STATS19 subset is
released as open data under the Open Government Licence v3.0. That licence
permits copying, publication, distribution and adaptation, including
commercial and non-commercial reuse, subject to attribution and its stated
exclusions. The project review therefore permits redistribution of these seven
public-data snapshots with the following notice and links to the source and
licence:

> Contains public sector information licensed under the Open Government
> Licence v3.0. Source: UK Department for Transport, Road Safety Open Data
> (source snapshot retrieved 27 August 2026).

The archive description must identify the files as project-derived snapshots,
must not imply DfT endorsement, and must not include restricted STATS19 fields.
The evidence and decision are recorded in
`docs/STATS19_REDISTRIBUTION_REVIEW.md`. The source-terms review is complete,
but the immutable archive URLs remain unpublished.

Required DfT documentation snapshots are included in the software repository
under `data/external/documentation/`. The public workflow does not run the
original all-years extraction script.

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
- Required authoritative input: `data/raw/cas/cas_injury_2022_2025_snapshot.jsonl.gz`
- Authoritative input SHA-256:
  `7db99dd4ba92716d751dabbc08b03f72373c635025b7d5335cf0b6705a7bd7f3`
- License stated by the source metadata: CC BY 4.0 International.
- Required attribution: Waka Kotahi NZ Transport Agency, Crash Analysis
  System (CAS), with the source URL and snapshot date.

CAS is a live service and records can change after publication. The CAS
severity label is the worst injury recorded for a crash and is not assumed to
be identical to the STATS19 severity label. The two datasets were trained and
evaluated separately; records, absolute metrics and SHAP ranks were not
pooled.

## Recreating the local data layout

1. Obtain the fixed data archive after its DOI and immutable file URLs are
   published. Do not substitute a current live response.
2. Run `python download_and_verify_data.py download --dataset stats19` and
   `python download_and_verify_data.py verify --dataset stats19`.
3. Run `python run_public_reproduction.py --all`; the runner builds an isolated
   workspace and recreates the D1 audit directly from the seven verified
   annual files.
4. For CAS, obtain the exact lossless JSONL snapshot whose SHA-256 is
   `7db99dd4ba92716d751dabbc08b03f72373c635025b7d5335cf0b6705a7bd7f3`, then
   run:

   ```text
   python run_cas_public_reproduction.py --self-test
   python run_cas_public_reproduction.py --all --snapshot /path/to/cas_injury_2022_2025_snapshot.jsonl.gz
   python verify_cas_public_results.py --candidate-root /path/to/cas-reproduction
   ```

   The CAS runner keeps this workflow outside the STATS19 workspace and never
   replaces the fixed snapshot with a live API query. The inspection CSV is
   useful for transparent review; the lossless JSONL is the authoritative CAS
   analysis input.

Before publication of the immutable data record, a researcher who already has
the exact files may place them at the manifest paths and run `verify`. The
mutable DfT complete-file URL and live CAS API remain provenance links, not
exact-snapshot fallbacks.

Do not combine the two datasets or treat their class labels as exchangeable.
