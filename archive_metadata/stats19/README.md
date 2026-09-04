# Fixed UK STATS19 collision snapshots, 2018-2024

## Purpose

These seven CSV files are the exact fixed inputs used for the primary
STATS19 analysis in the associated leakage-aware temporal collision-severity
evaluation. They are distributed separately from the software repository so
that an independent researcher can verify the input bytes before running the
public reconstruction workflow.

## Source and derivation

- Underlying provider: UK Department for Transport (DfT), Road Safety Open
  Data.
- Source page:
  https://www.gov.uk/government/statistical-data-sets/road-safety-open-data
- Complete source file retrieved: 27 August 2026.
- Complete source SHA-256:
  `4b60aac426b8fb7771dc9a3fc3e383399e88c9a49041544926aec4bea409e366`.
- Derivation: records were selected from the frozen complete collision CSV by
  `collision_year`; the source header and selected rows were retained
  byte-for-byte.

The files are project-derived annual snapshots. They are not represented as
official standalone annual releases, and a later version of DfT's mutable
`latest-published-year` file is not an exact substitute.

## Files

| File | Rows | Bytes | SHA-256 |
|---|---:|---:|---|
| `collision_2018.csv` | 122635 | 23546899 | `843061a73214ba8832ed4ca78292c0f0c6686444927b0fc47c9207e4e5985f4f` |
| `collision_2019.csv` | 117536 | 22503365 | `548b97c6afd3d6466779f7a36272f5674f0228e9b820b8d4a4a28c8b4c14b31a` |
| `collision_2020.csv` | 91199 | 17427393 | `49072103f337a9017aef8cab858463c636b33758fec68664592943d1e1d33f93` |
| `collision_2021.csv` | 101087 | 19208797 | `01442848b43bd89d6e9e0f21f17087c974f853ba9220dd8e6cc271b4a9d738de` |
| `collision_2022.csv` | 106004 | 20116923 | `89b2f1460698e471858b3bf1c36a737f632d40c74bfe75aa8964d4260b118dbf` |
| `collision_2023.csv` | 104258 | 19790001 | `ec28a1792ac22ff3382b8d0b3549dc07c9c57d9f8d94b5be0084def09aef47e6` |
| `collision_2024.csv` | 100927 | 19208488 | `ad02c026d0f87074800bd26fbca0e51b7b18eacc3e541fb869c091604b6a8b20` |

`SHA256SUMS.txt` contains the same machine-checkable data-file hashes.

## Licence and attribution

Contains public sector information licensed under the Open Government
Licence v3.0. Source: UK Department for Transport, Road Safety Open Data
(source snapshot retrieved 27 August 2026).

Licence text:
https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/

The UK Department for Transport does not endorse this archive, its associated
software or the associated research. Haojie Liu compiled and deposited the
fixed snapshots but is not the creator of the underlying collision records.

## Reproduction

Software and instructions:
https://github.com/Lhj828828/stats19-crash-severity-temporal-eval

Place the seven CSV files under `data/raw/collisions/` in a clean software
checkout, then run:

```text
python download_and_verify_data.py verify --dataset stats19
python run_public_reproduction.py --all
```

The analysis unit is one police-reported personal-injury collision. Consult
the software repository for the leakage audit, frozen temporal and random
partitions, model protocols and interpretation limits.
