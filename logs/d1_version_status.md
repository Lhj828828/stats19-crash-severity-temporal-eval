# D1 data version status

- Audit time: 2026-08-27T20:36:44+08:00
- Official landing page: https://www.gov.uk/government/statistical-data-sets/road-safety-open-data
- The official page states that final annual data are released after the final
  annual reported road casualties publication.
- On the audit date, the latest final validated full year shown by DfT was 2025.
  Therefore, 2023 and 2024 are final validated years, not provisional mid-year
  releases.
- DfT also states that previous years can occasionally receive minor revisions.
  In this project, "final" means the current final-validated version frozen by
  the checksums in `d1_file_manifest.csv`, not a promise that DfT will never
  revise a record again.
- The seven annual CSV files were derived from one current official complete
  collision file. Records were selected using `collision_year`; each selected
  CSV row was copied byte-for-byte and the source file was not modified.
- Scope: police-reported personal-injury collisions on public roads in Great
  Britain. The files do not represent all crashes or damage-only crashes.
- Table terminology changed historically from "accident" to "collision". This
  project uses the current DfT term "collision".
