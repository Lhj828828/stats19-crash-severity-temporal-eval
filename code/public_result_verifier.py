"""Verify regenerated compact STATS19 outputs against fixed references.

The verifier deliberately excludes fitted models, record-level predictions,
SHAP arrays and bootstrap arrays. CSV rows are aligned by declared keys, and
JSON objects are compared recursively with explicitly ignored runtime fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = PROJECT_DIR / "config" / "public_result_contract.json"
REPORT_VERSION = "PUBLIC_RESULT_VERIFICATION_V1"
MAX_EXAMPLES = 12


class ResultContractError(ValueError):
    """Raised when the compact-result contract is unsafe or malformed."""


class ResultComparisonError(RuntimeError):
    """Raised when one declared result cannot be compared structurally."""


def _relative_parts(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not value:
        raise ResultContractError(f"{field} must be a non-empty string")
    if "\\" in value or "\x00" in value:
        raise ResultContractError(f"Unsafe {field}: {value!r}")
    raw_parts = value.split("/")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in raw_parts)
        or any(":" in part for part in raw_parts)
    ):
        raise ResultContractError(f"Unsafe {field}: {value!r}")
    return tuple(path.parts)


def safe_path(root: Path, value: object, *, field: str = "path") -> Path:
    parts = _relative_parts(value, field=field)
    resolved_root = root.expanduser().resolve()
    candidate = resolved_root.joinpath(*parts).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ResultContractError(f"{field} escapes its root: {value!r}")
    return candidate


def _string_list(value: object, *, field: str, allow_empty: bool) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise ResultContractError(f"{field} must be {qualifier}")
    if any(not isinstance(item, str) or not item for item in value):
        raise ResultContractError(f"{field} must contain non-empty strings")
    if len(value) != len(set(value)):
        raise ResultContractError(f"{field} contains duplicates")
    return list(value)


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultContractError(f"Could not read result contract: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResultContractError("Result contract must be a JSON object")
    if payload.get("schema_version") != "PUBLIC_RESULT_CONTRACT_V1":
        raise ResultContractError("Unexpected public result contract version")

    _relative_parts(payload.get("reference_root"), field="reference_root")
    tolerances = payload.get("tolerances")
    if not isinstance(tolerances, dict):
        raise ResultContractError("tolerances must be an object")
    for name in ("absolute", "relative"):
        value = tolerances.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ResultContractError(f"tolerances.{name} must be numeric")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ResultContractError(f"tolerances.{name} must be finite and non-negative")

    comparisons = payload.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ResultContractError("comparisons must be a non-empty list")
    seen_paths: set[str] = set()
    for index, item in enumerate(comparisons):
        prefix = f"comparisons[{index}]"
        if not isinstance(item, dict):
            raise ResultContractError(f"{prefix} must be an object")
        raw_path = item.get("path")
        _relative_parts(raw_path, field=f"{prefix}.path")
        assert isinstance(raw_path, str)
        if raw_path in seen_paths:
            raise ResultContractError(f"Duplicate comparison path: {raw_path}")
        seen_paths.add(raw_path)
        kind = item.get("kind")
        if kind not in {"csv", "json"}:
            raise ResultContractError(f"{prefix}.kind must be csv or json")
        expected_suffix = f".{kind}"
        if not raw_path.lower().endswith(expected_suffix):
            raise ResultContractError(f"{prefix}.path must end in {expected_suffix}")
        if kind == "csv":
            keys = _string_list(
                item.get("key_columns"), field=f"{prefix}.key_columns", allow_empty=False
            )
            ignored = _string_list(
                item.get("ignore_columns", []),
                field=f"{prefix}.ignore_columns",
                allow_empty=True,
            )
            if set(keys) & set(ignored):
                raise ResultContractError(f"{prefix} ignores one of its key columns")
            filter_spec = item.get("filter")
            if filter_spec is not None:
                if not isinstance(filter_spec, dict):
                    raise ResultContractError(f"{prefix}.filter must be an object")
                column = filter_spec.get("column")
                if not isinstance(column, str) or not column:
                    raise ResultContractError(f"{prefix}.filter.column is invalid")
                _string_list(
                    filter_spec.get("include"),
                    field=f"{prefix}.filter.include",
                    allow_empty=False,
                )
        else:
            _string_list(
                item.get("ignore_keys", []),
                field=f"{prefix}.ignore_keys",
                allow_empty=True,
            )

    required = payload.get("required_nonempty_files")
    _string_list(required, field="required_nonempty_files", allow_empty=False)
    for index, raw_path in enumerate(required):
        _relative_parts(raw_path, field=f"required_nonempty_files[{index}]")
    return payload


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ResultComparisonError(f"CSV has no header: {path}")
            fields = list(reader.fieldnames)
            if any(not field for field in fields) or len(fields) != len(set(fields)):
                raise ResultComparisonError(f"CSV has invalid or duplicate columns: {path}")
            rows = []
            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ResultComparisonError(
                        f"CSV row {row_number} has more values than columns: {path}"
                    )
                rows.append({field: row.get(field, "") or "" for field in fields})
    except OSError as exc:
        raise ResultComparisonError(f"Could not read CSV {path}: {exc}") from exc
    return fields, rows


def _filter_rows(
    fields: list[str], rows: list[dict[str, str]], filter_spec: object, path: Path
) -> list[dict[str, str]]:
    if filter_spec is None:
        return rows
    assert isinstance(filter_spec, dict)
    column = str(filter_spec["column"])
    if column not in fields:
        raise ResultComparisonError(f"Filter column {column!r} is missing from {path}")
    included = set(filter_spec["include"])
    return [row for row in rows if row[column] in included]


def _keyed_rows(
    fields: list[str],
    rows: list[dict[str, str]],
    key_columns: list[str],
    path: Path,
) -> dict[tuple[str, ...], dict[str, str]]:
    missing = [column for column in key_columns if column not in fields]
    if missing:
        raise ResultComparisonError(f"Key columns missing from {path}: {missing}")
    keyed: dict[tuple[str, ...], dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        key = tuple(row[column] for column in key_columns)
        if key in keyed:
            raise ResultComparisonError(
                f"Duplicate key {key!r} at filtered row {row_number} in {path}"
            )
        keyed[key] = row
    return keyed


def _numeric(value: str) -> float | None:
    try:
        return float(value.strip())
    except ValueError:
        return None


def scalar_equal(expected: str, observed: str, *, atol: float, rtol: float) -> bool:
    if expected == observed:
        return True
    expected_number = _numeric(expected)
    observed_number = _numeric(observed)
    if expected_number is None or observed_number is None:
        return False
    if math.isnan(expected_number) or math.isnan(observed_number):
        return math.isnan(expected_number) and math.isnan(observed_number)
    if math.isinf(expected_number) or math.isinf(observed_number):
        return expected_number == observed_number
    return math.isclose(expected_number, observed_number, abs_tol=atol, rel_tol=rtol)


def _display_key(columns: list[str], values: tuple[str, ...]) -> dict[str, str]:
    return dict(zip(columns, values, strict=True))


def compare_csv(
    reference: Path,
    candidate: Path,
    spec: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    reference_fields, reference_rows = _read_csv(reference)
    candidate_fields, candidate_rows = _read_csv(candidate)
    reference_rows = _filter_rows(
        reference_fields, reference_rows, spec.get("filter"), reference
    )
    candidate_rows = _filter_rows(
        candidate_fields, candidate_rows, spec.get("filter"), candidate
    )
    keys = list(spec["key_columns"])
    ignored = set(spec.get("ignore_columns", []))
    reference_columns = set(reference_fields) - ignored
    candidate_columns = set(candidate_fields) - ignored
    examples: list[dict[str, Any]] = []
    mismatch_count = 0
    if reference_columns != candidate_columns:
        mismatch_count += len(reference_columns ^ candidate_columns)
        examples.append(
            {
                "location": "header",
                "missing_columns": sorted(reference_columns - candidate_columns),
                "extra_columns": sorted(candidate_columns - reference_columns),
            }
        )
    for column in keys:
        if column not in reference_columns or column not in candidate_columns:
            raise ResultComparisonError(
                f"Declared key column {column!r} is unavailable after exclusions"
            )

    reference_map = _keyed_rows(reference_fields, reference_rows, keys, reference)
    candidate_map = _keyed_rows(candidate_fields, candidate_rows, keys, candidate)
    reference_keys = set(reference_map)
    candidate_keys = set(candidate_map)
    missing_keys = sorted(reference_keys - candidate_keys)
    extra_keys = sorted(candidate_keys - reference_keys)
    mismatch_count += len(missing_keys) + len(extra_keys)
    for label, values in (("missing_row", missing_keys), ("extra_row", extra_keys)):
        for value in values:
            if len(examples) >= MAX_EXAMPLES:
                break
            examples.append({"location": label, "key": _display_key(keys, value)})

    compared_values = 0
    comparable_columns = sorted((reference_columns & candidate_columns) - set(keys))
    for key in sorted(reference_keys & candidate_keys):
        reference_row = reference_map[key]
        candidate_row = candidate_map[key]
        for column in comparable_columns:
            compared_values += 1
            expected = reference_row[column]
            observed = candidate_row[column]
            if scalar_equal(expected, observed, atol=atol, rtol=rtol):
                continue
            mismatch_count += 1
            if len(examples) < MAX_EXAMPLES:
                examples.append(
                    {
                        "location": "cell",
                        "key": _display_key(keys, key),
                        "column": column,
                        "expected": expected,
                        "observed": observed,
                    }
                )
    return {
        "status": "PASS" if mismatch_count == 0 else "FAIL",
        "reference_rows": len(reference_rows),
        "candidate_rows": len(candidate_rows),
        "compared_values": compared_values,
        "mismatch_count": mismatch_count,
        "examples": examples,
    }


def _json_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compare_json_values(
    expected: object,
    observed: object,
    *,
    ignored_keys: set[str],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    mismatch_count = 0
    compared_values = 0
    examples: list[dict[str, Any]] = []

    def mismatch(location: str, expected_value: object, observed_value: object) -> None:
        nonlocal mismatch_count
        mismatch_count += 1
        if len(examples) < MAX_EXAMPLES:
            examples.append(
                {
                    "location": location,
                    "expected": expected_value,
                    "observed": observed_value,
                }
            )

    def walk(left: object, right: object, location: str) -> None:
        nonlocal compared_values
        if isinstance(left, dict) and isinstance(right, dict):
            left_keys = set(left) - ignored_keys
            right_keys = set(right) - ignored_keys
            for key in sorted(left_keys - right_keys):
                mismatch(f"{location}.{key}", left[key], "<MISSING>")
            for key in sorted(right_keys - left_keys):
                mismatch(f"{location}.{key}", "<MISSING>", right[key])
            for key in sorted(left_keys & right_keys):
                walk(left[key], right[key], f"{location}.{key}")
            return
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                mismatch(f"{location}.length", len(left), len(right))
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                walk(left_item, right_item, f"{location}[{index}]")
            return

        compared_values += 1
        if _json_number(left) and _json_number(right):
            left_number = float(left)
            right_number = float(right)
            if math.isnan(left_number) or math.isnan(right_number):
                equal = math.isnan(left_number) and math.isnan(right_number)
            elif math.isinf(left_number) or math.isinf(right_number):
                equal = left_number == right_number
            else:
                equal = math.isclose(
                    left_number, right_number, abs_tol=atol, rel_tol=rtol
                )
            if not equal:
                mismatch(location, left, right)
            return
        if type(left) is not type(right) or left != right:
            mismatch(location, left, right)

    walk(expected, observed, "$")
    return {
        "status": "PASS" if mismatch_count == 0 else "FAIL",
        "compared_values": compared_values,
        "mismatch_count": mismatch_count,
        "examples": examples,
    }


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultComparisonError(f"Could not read JSON {path}: {exc}") from exc


def compare_json(
    reference: Path,
    candidate: Path,
    spec: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    return compare_json_values(
        _read_json(reference),
        _read_json(candidate),
        ignored_keys=set(spec.get("ignore_keys", [])),
        atol=atol,
        rtol=rtol,
    )


def verify_results(
    *,
    contract_path: Path = DEFAULT_CONTRACT,
    candidate_root: Path = PROJECT_DIR,
    reference_root: Path | None = None,
) -> dict[str, Any]:
    contract_path = contract_path.expanduser().resolve()
    candidate_root = candidate_root.expanduser().resolve()
    contract = load_contract(contract_path)
    if reference_root is None:
        reference_root = safe_path(
            candidate_root, contract["reference_root"], field="reference_root"
        )
    else:
        reference_root = reference_root.expanduser().resolve()
    atol = float(contract["tolerances"]["absolute"])
    rtol = float(contract["tolerances"]["relative"])

    comparisons: list[dict[str, Any]] = []
    for spec in contract["comparisons"]:
        raw_path = spec["path"]
        reference = safe_path(reference_root, raw_path)
        candidate = safe_path(candidate_root, raw_path)
        result: dict[str, Any] = {
            "path": raw_path,
            "kind": spec["kind"],
        }
        try:
            if not reference.is_file():
                raise ResultComparisonError(f"Reference file is missing: {reference}")
            if not candidate.is_file():
                raise ResultComparisonError(f"Candidate file is missing: {candidate}")
            if spec["kind"] == "csv":
                detail = compare_csv(
                    reference, candidate, spec, atol=atol, rtol=rtol
                )
            else:
                detail = compare_json(
                    reference, candidate, spec, atol=atol, rtol=rtol
                )
            result.update(detail)
        except ResultComparisonError as exc:
            result.update(
                {
                    "status": "FAIL",
                    "mismatch_count": 1,
                    "examples": [{"location": "file", "error": str(exc)}],
                }
            )
        comparisons.append(result)

    required_files: list[dict[str, Any]] = []
    for raw_path in contract["required_nonempty_files"]:
        path = safe_path(candidate_root, raw_path)
        size = path.stat().st_size if path.is_file() else None
        required_files.append(
            {
                "path": raw_path,
                "status": "PASS" if size is not None and size > 0 else "FAIL",
                "bytes": size,
            }
        )

    passed_comparisons = sum(item["status"] == "PASS" for item in comparisons)
    passed_files = sum(item["status"] == "PASS" for item in required_files)
    status = (
        "PASS"
        if passed_comparisons == len(comparisons) and passed_files == len(required_files)
        else "FAIL"
    )
    return {
        "version": REPORT_VERSION,
        "status": status,
        "contract": str(contract_path),
        "candidate_root": str(candidate_root),
        "reference_root": str(reference_root),
        "tolerances": {"absolute": atol, "relative": rtol},
        "comparison_summary": {
            "passed": passed_comparisons,
            "total": len(comparisons),
        },
        "required_file_summary": {
            "passed": passed_files,
            "total": len(required_files),
        },
        "comparisons": comparisons,
        "required_nonempty_files": required_files,
    }


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def print_report(report: dict[str, Any]) -> None:
    summary = report["comparison_summary"]
    files = report["required_file_summary"]
    print(
        f"COMPACT_COMPARISONS={summary['passed']}/{summary['total']} "
        f"REQUIRED_FILES={files['passed']}/{files['total']}"
    )
    for item in report["comparisons"]:
        if item["status"] != "PASS":
            print(f"[FAIL] {item['path']}: {item.get('examples', [])}")
    for item in report["required_nonempty_files"]:
        if item["status"] != "PASS":
            print(f"[FAIL] missing or empty: {item['path']}")
    print(f"PUBLIC_RESULT_VERIFICATION={report['status']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--candidate-root", type=Path, default=PROJECT_DIR)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = verify_results(
        contract_path=args.contract,
        candidate_root=args.candidate_root,
        reference_root=args.reference_root,
    )
    report["completed_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if args.report is not None:
        write_json_atomic(args.report.expanduser().resolve(), report)
    print_report(report)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ResultContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
