# D2 schema and code audit

- Audit scope: STATS19 collision files, 2018-2024.
- Official dictionary: `dft-road-casualty-statistics-road-safety-open-dataset-data-guide-2025.xlsx`, worksheet `2024_code_list`.
- Annual schemas identical: **True**.
- Columns in each annual file: **44**.
- Fields absent in any year: **0**.
- Raw fields requiring a documented official-name alias: **3**.
- Raw fields with no official dictionary match after aliasing: **0**.
- Observed coded values absent from the official guide: **0**.
- Invalid candidate-field formats: **0**.
- Collision severity codes are exactly 1/2/3: **True**.

## Annual rows

- 2018: 122,635
- 2019: 117,536
- 2020: 91,199
- 2021: 101,087
- 2022: 106,004
- 2023: 104,258
- 2024: 100,927

## Interpretation

All seven annual files use the same current 44-column schema because they were extracted from one current official complete file. This confirms structural consistency, but it does not by itself prove that every field is suitable for prediction.

The complete file provides both historic variables and DfT-harmonized current replacements for junction detail, pedestrian crossing and carriageway hazards. The current harmonized variables should normally be preferred; the historic versions are retained only for traceability and sensitivity checks.

`collision_severity` is consistently coded as 1=Fatal, 2=Serious and 3=Slight. `enhanced_severity_collision`, `collision_injury_based`, `collision_adjusted_severity_serious`, `collision_adjusted_severity_slight`, and `number_of_casualties` are outcome-related fields. Their final exclusion is a D3 leakage-audit decision, not a D2 schema decision.

Category numbers are labels, not continuous magnitudes. Unknown or missing categories must remain explicit during later feature engineering rather than being silently treated as ordinary ordered numbers.
