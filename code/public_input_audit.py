"""Rebuild the D1-compatible audit from fixed annual STATS19 inputs.

The public data archive contains the seven annual analysis files, not the
1.53 GB all-years source used during the original author-side extraction. This
module verifies those annual files against the public manifest and recreates
the audit table required by D5 without pretending to repeat that extraction.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from public_data import (
    DataManifestError,
    FileSpec,
    load_manifest,
    local_path,
    select_specs,
    verify_files,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_DIR / "config" / "public_data_manifest.json"
DEFAULT_OUTPUT = PROJECT_DIR / "logs" / "d1_initial_audit.csv"
ANALYSIS_YEARS = tuple(range(2018, 2025))
ANNUAL_NAME_PATTERN = re.compile(r"collision_(\d{4})\.csv$")
AUDIT_COLUMNS = (
    "year",
    "rows",
    "extraction_row_match",
    "columns",
    "year_min",
    "year_max",
    "year_values_correct",
    "duplicate_collision_index",
    "fatal_n",
    "fatal_pct",
    "serious_n",
    "serious_pct",
    "slight_n",
    "slight_pct",
    "severity_missing_n",
    "severity_other_n",
)


class PublicInputAuditError(RuntimeError):
    """Raised when fixed annual inputs cannot reproduce the D1 audit."""


@dataclass(frozen=True)
class AnnualInput:
    year: int
    expected_rows: int
    spec: FileSpec


def load_annual_inputs(manifest_path: Path) -> list[AnnualInput]:
    payload, all_specs = load_manifest(manifest_path)
    specs = select_specs(all_specs, ["stats19"])
    definition = payload["datasets"]["stats19"]
    declared_years = tuple(int(year) for year in definition.get("analysis_years", []))
    if declared_years != ANALYSIS_YEARS:
        raise DataManifestError(
            f"STATS19 analysis_years must be {ANALYSIS_YEARS}, got {declared_years}"
        )

    raw_by_path = {
        str(item.get("relative_path")): item for item in definition.get("files", [])
    }
    annual: list[AnnualInput] = []
    for spec in specs:
        match = ANNUAL_NAME_PATTERN.search(spec.relative_path)
        if match is None:
            raise DataManifestError(
                f"Unexpected STATS19 annual filename: {spec.relative_path}"
            )
        year = int(match.group(1))
        raw = raw_by_path[spec.relative_path]
        expected_rows = raw.get("expected_rows")
        if not isinstance(expected_rows, int) or expected_rows <= 0:
            raise DataManifestError(
                f"Invalid expected_rows for {spec.relative_path}: {expected_rows!r}"
            )
        annual.append(AnnualInput(year, expected_rows, spec))

    annual.sort(key=lambda item: item.year)
    observed_years = tuple(item.year for item in annual)
    if observed_years != ANALYSIS_YEARS:
        raise DataManifestError(
            f"Expected one annual STATS19 file for {ANALYSIS_YEARS}, got {observed_years}"
        )
    return annual


def audit_annual_file(path: Path, year: int, expected_rows: int) -> dict[str, object]:
    frame = pd.read_csv(path, low_memory=False)
    required = {"collision_index", "collision_year", "collision_severity"}
    missing = required.difference(frame.columns)
    if missing:
        raise PublicInputAuditError(
            f"{path.name} is missing required columns: {sorted(missing)}"
        )
    if frame.empty:
        raise PublicInputAuditError(f"{path.name} contains no collision records")

    severity = pd.to_numeric(frame["collision_severity"], errors="coerce")
    counts = severity.value_counts(dropna=False).to_dict()
    rows = len(frame)
    fatal = int(counts.get(1, 0))
    serious = int(counts.get(2, 0))
    slight = int(counts.get(3, 0))
    missing_severity = int(severity.isna().sum())
    other_severity = rows - fatal - serious - slight - missing_severity
    years = pd.to_numeric(frame["collision_year"], errors="coerce")
    if years.isna().any():
        raise PublicInputAuditError(f"{path.name} contains missing or invalid years")

    return {
        "year": year,
        "rows": rows,
        "extraction_row_match": rows == expected_rows,
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


def build_initial_audit(
    root: Path,
    manifest_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    annual = load_annual_inputs(manifest_path)
    verification = verify_files(root, [item.spec for item in annual])
    failures = [result for result in verification if result.status != "PASS"]
    if failures:
        detail = ", ".join(
            f"{result.spec.relative_path}={result.status}" for result in failures
        )
        raise PublicInputAuditError(
            f"Fixed STATS19 inputs failed verification: {detail}"
        )

    rows = [
        audit_annual_file(
            local_path(root, item.spec),
            item.year,
            item.expected_rows,
        )
        for item in annual
    ]
    audit = pd.DataFrame(rows, columns=AUDIT_COLUMNS)
    if not audit["extraction_row_match"].all():
        mismatches = audit.loc[
            ~audit["extraction_row_match"], ["year", "rows"]
        ].to_dict("records")
        raise PublicInputAuditError(
            f"Annual row counts differ from the frozen extraction: {mismatches}"
        )
    if not audit["year_values_correct"].all():
        raise PublicInputAuditError("At least one annual file contains another year")
    if int(audit["duplicate_collision_index"].sum()) != 0:
        raise PublicInputAuditError("Duplicate collision_index values found within a year")
    if int(audit["severity_missing_n"].sum()) != 0:
        raise PublicInputAuditError("Missing collision severity values found")
    if int(audit["severity_other_n"].sum()) != 0:
        raise PublicInputAuditError("Unexpected collision severity codes found")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    return audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_initial_audit(args.root, args.manifest, args.output)
    print(audit.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"PUBLIC_D1_AUDIT=PASS rows={int(audit['rows'].sum())}")
    print(f"Audit report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DataManifestError, PublicInputAuditError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
