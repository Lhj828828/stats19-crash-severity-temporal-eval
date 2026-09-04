# Fixed New Zealand CAS injury-crash snapshot, 2022-2025

## Purpose

These files are the exact fixed inputs used for the independent New Zealand
Crash Analysis System (CAS) workflow replication in the associated
leakage-aware temporal collision-severity evaluation. CAS records were trained
and evaluated separately from STATS19 records; labels, records, absolute
metrics and SHAP ranks were not pooled.

## Source, scope and modifications

- Underlying provider: Waka Kotahi NZ Transport Agency, Crash Analysis System
  (CAS).
- Source page:
  https://opendata-nzta.opendata.arcgis.com/datasets/NZTA::crash-analysis-system-cas-data-1/about
- Public attribute service queried: 28 August 2026.
- Query scope:
  `crashYear BETWEEN 2022 AND 2025 AND crashSeverity <> 'Non-Injury Crash'`.
- Project modifications: year and injury-crash filtering, exclusion of
  geometry, serialisation to JSON Lines and CSV, and gzip compression.

CAS is a live, monthly updated service and records may subsequently change. A
fresh service response is not an exact substitute for this fixed snapshot.
The source describes CAS data as supplied on an "as is, where is" basis; its
quality caveats continue to apply. The source also notes that 2020 and 2021
are incomplete, but those years are outside this archive's 2022-2025 scope.

## Files

| File | Role | Rows | Bytes | SHA-256 |
|---|---|---:|---:|---|
| `cas_injury_2022_2025_snapshot.jsonl.gz` | Authoritative lossless analysis input | 43121 | 2771733 | `7db99dd4ba92716d751dabbc08b03f72373c635025b7d5335cf0b6705a7bd7f3` |
| `cas_injury_2022_2025_snapshot.csv.gz` | Inspection copy | 43121 | 1867440 | `cc554366351cf4d5ccc7207f583c8ff1437036a615f1bac4dcdcccac76d1745c` |

The authoritative JSONL and inspection CSV contain the same 43,121 queried
records represented through different serialisations. `SHA256SUMS.txt`
contains machine-checkable hashes.

## Licence and attribution

Source data: Waka Kotahi NZ Transport Agency, Crash Analysis System (CAS),
licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).
Fixed attribute snapshot retrieved 28 August 2026 from the public CAS service.

Licence text: https://creativecommons.org/licenses/by/4.0/

The filtering and format changes are identified above. Waka Kotahi does not
endorse this archive, its associated software or the associated research.
Haojie Liu compiled and deposited the fixed snapshot but is not the creator of
the underlying crash records.

## Reproduction

Software and instructions:
https://github.com/Lhj828828/stats19-crash-severity-temporal-eval

Run from a clean software checkout:

```text
python run_cas_public_reproduction.py --self-test
python run_cas_public_reproduction.py --all --snapshot /path/to/cas_injury_2022_2025_snapshot.jsonl.gz
python verify_cas_public_results.py --candidate-root /path/to/cas-reproduction
```

The CAS severity label is the worst injury recorded for a crash and is not
assumed to be statistically identical to the STATS19 severity definition. The
replication tests workflow executability and reports directional agreement or
disagreement; it does not establish cross-national generalizability.
