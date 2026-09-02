"""Write the fixed-snapshot CAS feasibility report for a public run."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_JSONL = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "cas"
    / "cas_injury_2022_2025_snapshot.jsonl.gz"
)
RAW_CSV = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "cas"
    / "cas_injury_2022_2025_snapshot.csv.gz"
)
REPORT = PROJECT_DIR / "logs" / "cas" / "CAS_feasibility_audit_report.md"
EXPECTED_JSONL_SHA256 = (
    "7db99dd4ba92716d751dabbc08b03f72373c635025b7d5335cf0b6705a7bd7f3"
)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_shape() -> tuple[int, int]:
    rows = 0
    fields: list[str] | None = None
    with gzip.open(RAW_JSONL, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"Snapshot line {line_number} is not an object")
            if fields is None:
                fields = list(row)
            elif list(row) != fields:
                raise ValueError(f"Snapshot field order changed at line {line_number}")
            rows += 1
    if fields is None:
        raise ValueError("The fixed CAS JSONL snapshot is empty")
    return rows, len(fields)


def main() -> None:
    if not RAW_JSONL.is_file() or not RAW_CSV.is_file():
        raise FileNotFoundError("Both fixed CAS snapshot representations are required")
    jsonl_hash = hash_file(RAW_JSONL)
    if jsonl_hash != EXPECTED_JSONL_SHA256:
        raise ValueError(
            "Fixed CAS JSONL hash differs from the public input contract: "
            f"{jsonl_hash}"
        )
    rows, fields = snapshot_shape()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CAS第二数据集可行性审计报告",
        "",
        "审计版本：CAS_FEASIBILITY_V1",
        "审计模式：offline_fixed_snapshot（不访问实时ArcGIS服务）",
        "审计对象：NZ Transport Agency Waka Kotahi Crash Analysis System (CAS) open data",
        "研究用途：作为STATS19时间泛化研究的第二数据源独立复核，不与STATS19合并训练",
        "",
        "## 审计结论",
        "",
        "固定快照审计完成；CAS仅作为独立工作流复核，不作为STATS19的直接外部测试集，也不用于证明跨国家普适性。",
        "",
        "## 固定输入",
        "",
        f"- 快照行数：{rows:,}",
        f"- 字段数：{fields}",
        f"- 快照SHA-256：`{hash_file(RAW_CSV)}`",
        f"- 无损JSONL SHA-256：`{jsonl_hash}`",
        "- 输入来源：固定JSONL快照；本次审计未查询实时API。",
        "",
        "## 目标与特征",
        "",
        "- 纳入2022—2025年Minor、Serious、Fatal伤害事故。",
        "- 15个候选字段用于后续建模；OBJECTID、年份、地区和目标字段不进入特征矩阵。",
        "- 训练、验证和测试分别为2022—2023、2024和2025。",
        "",
        "## 解释边界",
        "",
        "- CAS与STATS19分别训练、分别评估；不合并记录、不比较绝对指标水平。",
        "- 只比较流程可执行性和结果方向的一致或不一致，不作跨国家或跨领域普适性结论。",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("CAS_PUBLIC_FEASIBILITY_REPORT=WRITTEN")
    print(f"CAS_PUBLIC_FEASIBILITY_REPORT_SHA256={hash_file(REPORT)}")


if __name__ == "__main__":
    main()
