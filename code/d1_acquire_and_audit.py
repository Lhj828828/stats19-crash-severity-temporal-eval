"""Extract and audit STATS19 collision data for 2018-2024.

This script never modifies the downloaded source file. It copies matching raw
CSV records into seven annual files, computes checksums, and writes D1 audit
reports for reproducibility.
"""

from __future__ import annotations

import csv
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = PROJECT_DIR / "data" / "raw" / "downloads"
OUTPUT_DIR = PROJECT_DIR / "data" / "raw" / "collisions"
DOCUMENT_DIR = PROJECT_DIR / "data" / "external" / "documentation"
LOG_DIR = PROJECT_DIR / "logs"

SOURCE_FILE = (
    DOWNLOAD_DIR
    / "dft-road-casualty-statistics-collision-1979-latest-published-year.csv"
)
SOURCE_URL = (
    "https://data.dft.gov.uk/road-accidents-safety-data/"
    "dft-road-casualty-statistics-collision-1979-latest-published-year.csv"
)
LANDING_PAGE = (
    "https://www.gov.uk/government/statistical-data-sets/road-safety-open-data"
)
YEARS = tuple(range(2018, 2025))

DOCUMENT_URLS = {
    "dft-road-casualty-statistics-historical-revisions-data.csv": (
        "https://assets.publishing.service.gov.uk/media/"
        "6a6391812dc18ebe4c3b2bca/"
        "dft-road-casualty-statistics-historical-revisions-data.csv"
    ),
    "dft-road-casualty-statistics-road-safety-open-dataset-data-guide-2024.xlsx": (
        "https://assets.publishing.service.gov.uk/media/"
        "691c6440e39a085bda43eed6/"
        "dft-road-casualty-statistics-road-safety-open-dataset-data-guide-2024.xlsx"
    ),
    "dft-road-casualty-statistics-road-safety-open-dataset-data-guide-2025.xlsx": (
        "https://assets.publishing.service.gov.uk/media/"
        "6a63900b2dc18ebe4c3b2bc8/"
        "dft-road-casualty-statistics-road-safety-open-dataset-data-guide-2025.xlsx"
    ),
    "dft-road-casualty-statistics-severity-adjustment-figure-guidance.docx": (
        "https://assets.publishing.service.gov.uk/media/"
        "691c644021ef5aaa6543eef0/"
        "dft-road-casualty-statistics-severity-adjustment-figure-guidance.docx"
    ),
    "Understanding-historical-road-safety-data.docx": (
        "https://assets.publishing.service.gov.uk/media/"
        "691c6440e39a085bda43eed7/Understanding-historical-road-safety-data.docx"
    ),
    "stats19.pdf": (
        "https://assets.publishing.service.gov.uk/government/uploads/system/"
        "uploads/attachment_data/file/995422/stats19.pdf"
    ),
}


class HashedWriter:
    def __init__(self, path: Path) -> None:
        self.final_path = path
        self.partial_path = path.with_suffix(path.suffix + ".part")
        self.handle: BinaryIO = self.partial_path.open("wb")
        self.md5 = hashlib.md5()
        self.sha256 = hashlib.sha256()
        self.size = 0
        self.rows = 0

    def write(self, value: bytes, *, data_row: bool = False) -> None:
        self.handle.write(value)
        self.md5.update(value)
        self.sha256.update(value)
        self.size += len(value)
        if data_row:
            self.rows += 1

    def finish(self) -> dict[str, object]:
        self.handle.close()
        os.replace(self.partial_path, self.final_path)
        return {
            "size_bytes": self.size,
            "md5": self.md5.hexdigest(),
            "sha256": self.sha256.hexdigest(),
            "extracted_rows": self.rows,
        }

    def abort(self) -> None:
        if not self.handle.closed:
            self.handle.close()
        self.partial_path.unlink(missing_ok=True)


def hash_file(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            md5.update(block)
            sha256.update(block)
    return md5.hexdigest(), sha256.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(PROJECT_DIR).as_posix()


def local_timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(
        timespec="seconds"
    )


def extract_annual_files() -> tuple[list[dict[str, object]], dict[str, object]]:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Downloaded source file not found: {SOURCE_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    writers = {
        year: HashedWriter(OUTPUT_DIR / f"collision_{year}.csv") for year in YEARS
    }
    source_md5 = hashlib.md5()
    source_sha256 = hashlib.sha256()
    source_size = 0
    buffer = b""
    header_seen = False
    next_progress = 250 * 1024 * 1024

    print("Scanning the downloaded complete collision file...")
    try:
        with SOURCE_FILE.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                source_md5.update(chunk)
                source_sha256.update(chunk)
                source_size += len(chunk)
                buffer += chunk
                lines = buffer.split(b"\n")
                buffer = lines.pop()

                for line in lines:
                    complete_line = line + b"\n"
                    if not header_seen:
                        header = line.rstrip(b"\r").decode("utf-8-sig").split(",")
                        if header[:2] != ["collision_index", "collision_year"]:
                            raise ValueError(
                                "Unexpected source columns; extraction stopped to avoid "
                                "misclassifying records."
                            )
                        for writer in writers.values():
                            writer.write(complete_line)
                        header_seen = True
                        continue

                    parts = line.rstrip(b"\r").split(b",", 2)
                    if len(parts) < 2:
                        continue
                    try:
                        year = int(parts[1].strip(b'"'))
                    except ValueError:
                        continue
                    if year in writers:
                        writers[year].write(complete_line, data_row=True)

                if source_size >= next_progress:
                    print(f"  processed {source_size / 1024**3:.2f} GiB")
                    next_progress += 250 * 1024 * 1024

            if buffer:
                parts = buffer.rstrip(b"\r").split(b",", 2)
                if len(parts) >= 2:
                    try:
                        year = int(parts[1].strip(b'"'))
                    except ValueError:
                        year = -1
                    if year in writers:
                        writers[year].write(buffer + b"\n", data_row=True)

        if not header_seen:
            raise ValueError("The source file is empty or has no readable header.")
        if any(writer.rows == 0 for writer in writers.values()):
            empty = [year for year, writer in writers.items() if writer.rows == 0]
            raise ValueError(f"No records found for expected years: {empty}")

        annual_records = []
        for year, writer in writers.items():
            hash_info = writer.finish()
            annual_records.append(
                {
                    "file_role": "derived_annual_collision_data",
                    "year": str(year),
                    "local_path": relative(writer.final_path),
                    "source_url": SOURCE_URL,
                    "downloaded_at_local": local_timestamp(SOURCE_FILE),
                    **hash_info,
                    "note": "Raw rows copied byte-for-byte from the complete DfT file",
                }
            )
    except Exception:
        for writer in writers.values():
            writer.abort()
        raise

    source_record = {
        "file_role": "downloaded_complete_collision_data",
        "year": "1979-2025",
        "local_path": relative(SOURCE_FILE),
        "source_url": SOURCE_URL,
        "downloaded_at_local": local_timestamp(SOURCE_FILE),
        "size_bytes": source_size,
        "md5": source_md5.hexdigest(),
        "sha256": source_sha256.hexdigest(),
        "extracted_rows": "",
        "note": "Official complete final-validated dataset; source file unchanged",
    }
    return annual_records, source_record


def audit_year(year: int, extracted_rows: int) -> dict[str, object]:
    path = OUTPUT_DIR / f"collision_{year}.csv"
    frame = pd.read_csv(path, low_memory=False)
    required = {"collision_index", "collision_year", "collision_severity"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"{path.name} is missing required columns: {sorted(missing)}")

    severity = pd.to_numeric(frame["collision_severity"], errors="coerce")
    counts = severity.value_counts(dropna=False).to_dict()
    rows = len(frame)
    fatal = int(counts.get(1, 0))
    serious = int(counts.get(2, 0))
    slight = int(counts.get(3, 0))
    missing_severity = int(severity.isna().sum())
    other_severity = rows - fatal - serious - slight - missing_severity
    years = pd.to_numeric(frame["collision_year"], errors="coerce")

    return {
        "year": year,
        "rows": rows,
        "extraction_row_match": rows == extracted_rows,
        "columns": len(frame.columns),
        "year_min": int(years.min()),
        "year_max": int(years.max()),
        "year_values_correct": bool((years == year).all()),
        "duplicate_collision_index": int(frame["collision_index"].duplicated().sum()),
        "fatal_n": fatal,
        "fatal_pct": fatal / rows * 100,
        "serious_n": serious,
        "serious_pct": serious / rows * 100,
        "slight_n": slight,
        "slight_pct": slight / rows * 100,
        "severity_missing_n": missing_severity,
        "severity_other_n": other_severity,
    }


def add_document_records(manifest: list[dict[str, object]]) -> None:
    for name, url in DOCUMENT_URLS.items():
        path = DOCUMENT_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Required official document not found: {path}")
        md5, sha256 = hash_file(path)
        manifest.append(
            {
                "file_role": "official_documentation",
                "year": "2018-2024",
                "local_path": relative(path),
                "source_url": url,
                "downloaded_at_local": local_timestamp(path),
                "size_bytes": path.stat().st_size,
                "md5": md5,
                "sha256": sha256,
                "extracted_rows": "",
                "note": "DfT/GOV.UK documentation",
            }
        )


def write_version_note(audit_time: str) -> None:
    note = f"""# D1 data version status

- Audit time: {audit_time}
- Official landing page: {LANDING_PAGE}
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
"""
    (LOG_DIR / "d1_version_status.md").write_text(note, encoding="utf-8")


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    audit_time = datetime.now().astimezone().isoformat(timespec="seconds")

    annual_records, source_record = extract_annual_files()
    manifest = [source_record, *annual_records]
    add_document_records(manifest)

    manifest_columns = [
        "file_role",
        "year",
        "local_path",
        "source_url",
        "downloaded_at_local",
        "size_bytes",
        "md5",
        "sha256",
        "extracted_rows",
        "note",
    ]
    manifest_path = LOG_DIR / "d1_file_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_columns)
        writer.writeheader()
        writer.writerows(manifest)

    extracted = {
        int(record["year"]): int(record["extracted_rows"])
        for record in annual_records
    }
    audit = pd.DataFrame(
        [audit_year(year, extracted[year]) for year in YEARS]
    )
    audit_path = LOG_DIR / "d1_initial_audit.csv"
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    write_version_note(audit_time)

    print("\nD1 initial audit")
    print(audit.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nManifest: {manifest_path}")
    print(f"Audit report: {audit_path}")
    print(f"Version note: {LOG_DIR / 'd1_version_status.md'}")


if __name__ == "__main__":
    main()
