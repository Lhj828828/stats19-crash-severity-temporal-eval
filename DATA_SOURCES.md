# Data sources and provenance

This file records the public sources and frozen snapshots used by the
STATS19 primary analysis and the independent CAS replication. Raw snapshots
are intentionally excluded from the Git repository because the STATS19 source
file is large and the CAS service is live. The checksums below identify the
exact files used in the completed analyses.

The machine-readable public input contract is
`config/public_data_manifest.json`. It identifies seven annual STATS19 files
and two CAS snapshot files by path, row count where applicable, byte size and
SHA-256. Together these comprise the published compact data archives; the 1.53 GB
full-history STATS19 source file is retained only as provenance and is not
required by the public workflow.

Both datasets passed the project redistribution review and were published as
separate immutable records because they use different licences:

- STATS19 version DOI: <https://doi.org/10.5281/zenodo.22290566>;
- CAS version DOI: <https://doi.org/10.5281/zenodo.22296725>.

The deposited inputs can be downloaded and verified with:

```text
python download_and_verify_data.py list
python download_and_verify_data.py download --dataset all
python download_and_verify_data.py verify --dataset all
```

The `download` command uses only version-specific Zenodo file URLs and verifies
each complete size and SHA-256 before accepting a file. It never silently
substitutes the mutable STATS19 `latest` file or a fresh CAS API query for the
snapshots used in the paper.

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
and the seven files are published under OGL v3.0 in Zenodo record 22290566.

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

The local files are project-created exports from the live attribute service:
the query selected 2022-2025 records and excluded `Non-Injury Crash`, geometry
was omitted, and the result was serialised as compressed JSONL and CSV. The
archive must identify those changes rather than describe either file as an
unchanged official release. It must preserve the source's live-data, quality
and "as is, where is" caveats and must not imply Waka Kotahi endorsement.

The CAS source-terms review permits redistribution of the two frozen files
under CC BY 4.0 with attribution and indication of the project modifications.
The full evidence, decision boundary and required archive wording are recorded
in `docs/CAS_REDISTRIBUTION_REVIEW.md`. The CAS data licence is separate from
the repository's MIT software licence. The two files are published under CC BY
4.0 in Zenodo record 22296725.

## Recreating the local data layout

For the current manuscript, use the fixed **v1.3.1** software source from
<https://doi.org/10.5281/zenodo.22793992> or the matching Git tag. Follow the
environment setup in [REPRODUCING.md](REPRODUCING.md#current-manuscript-workflow-v131),
then run from that source directory:

```text
python download_and_verify_data.py download --dataset all
python download_and_verify_data.py verify --dataset all
python run_final_analysis.py --stage plan
python run_final_analysis.py --stage reproduce --dataset all --workspace ../final-reconstruction
```

All nine files must report `PASS` before reconstruction. The final runner
creates separate STATS19 and CAS workspaces and clean environments, regenerates
the analyses, and compares the 20 selected V2 result tables. It does not copy
author-fitted models or predictions. CAS uses the lossless JSONL snapshot;
the inspection CSV is useful for review, not a replacement analysis input.

After the main reconstruction, reproduce the added QWK sensitivity separately:

```text
python reproduce_qwk_sensitivity.py --parent ../final-reconstruction/stats19 --workspace ../qwk-reconstruction
```

For an interrupted run, use the same workspace and add `--resume` to the
corresponding reconstruction command. See the current guide for completion
markers, partial-stage limitations and per-dataset paths.

The top-level `run_public_reproduction.py` and
`run_cas_public_reproduction.py` workflows reproduce historical implementations,
not the current selected results. Their instructions remain in the clearly
marked historical sections of `REPRODUCING.md` and
`CAS_PUBLIC_REPRODUCTION.md`; they are not alternative current entry points.

Researchers who already have the exact files may place them at the manifest
paths and run `verify`. The mutable DfT complete-file URL and live CAS API
remain provenance links, not exact-snapshot fallbacks.

Do not combine the two datasets or treat their class labels as exchangeable.
