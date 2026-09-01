"""Freeze the D6 analysis protocol, split assignments and Figure 2.

This script must run before model training. It reads only the D5 frozen table,
creates deterministic split assignments, records all design decisions and
produces descriptive target-only outputs. It does not fit or evaluate a model.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "data" / "processed" / "stats19_modeling_dataset.csv.gz"
D5_SCHEMA_FILE = PROJECT_DIR / "config" / "d5_dataset_schema.json"
D5_RULES_FILE = PROJECT_DIR / "config" / "d5_cleaning_rules.json"

PROTOCOL_FILE = PROJECT_DIR / "config" / "d6_analysis_protocol.json"
ASSIGNMENTS_FILE = PROJECT_DIR / "data" / "processed" / "d6_split_assignments.csv.gz"
LOG_DIR = PROJECT_DIR / "logs"
FIGURE_DIR = PROJECT_DIR / "figures"
ANNUAL_AUDIT_FILE = LOG_DIR / "d6_annual_severity.csv"
TEMPORAL_AUDIT_FILE = LOG_DIR / "d6_temporal_split_audit.csv"
RANDOM_AUDIT_FILE = LOG_DIR / "d6_random_split_audit.csv"
CHECKPOINT_FILE = LOG_DIR / "d6_checkpoint.md"
FIGURE_PNG = FIGURE_DIR / "figure2_dataset_overview.png"
FIGURE_PDF = FIGURE_DIR / "figure2_dataset_overview.pdf"

ALL_YEARS = tuple(range(2018, 2025))
DEVELOPMENT_YEARS = tuple(range(2018, 2024))
TEMPORAL_TRAIN_YEARS = tuple(range(2018, 2023))
TEMPORAL_VALIDATION_YEAR = 2023
TEMPORAL_TEST_YEAR = 2024

RANDOM_TRAIN_RATIO = 0.70
RANDOM_VALIDATION_RATIO = 0.15
RANDOM_TEST_RATIO = 0.15
RANDOM_SEEDS = (1103, 2207, 3301, 4409, 5501)
RANDOM_SECOND_STAGE_OFFSET = 1_000_003
FATAL_PREFLIGHT_THRESHOLD = 50
BOOTSTRAP_ITERATIONS = 2_000
BOOTSTRAP_SEED = 20_260_828

TARGET_LABELS = {0: "Slight", 1: "Serious", 2: "Fatal"}
TARGET_COLORS = {
    "Slight": "#4C78A8",
    "Serious": "#E6A43A",
    "Fatal": "#C44E52",
}


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows generated for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_and_validate_d5() -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    schema = json.loads(D5_SCHEMA_FILE.read_text(encoding="utf-8"))
    rules = json.loads(D5_RULES_FILE.read_text(encoding="utf-8"))
    if schema.get("status") != "QC_PASSED_AND_FROZEN":
        raise ValueError("D6 requires a QC-passed and frozen D5 schema")
    if rules.get("status") != "FROZEN_BEFORE_MODELING":
        raise ValueError("D6 requires frozen D5 cleaning rules")

    actual_hash = hash_file(DATA_FILE)
    if actual_hash != schema["sha256"]:
        raise ValueError("D5 dataset hash differs from the frozen schema")

    required = [
        "meta_collision_index",
        "meta_collision_year",
        "target_severity",
    ]
    table = pd.read_csv(
        DATA_FILE,
        usecols=required,
        dtype={"meta_collision_index": "string"},
        low_memory=False,
    )
    if len(table) != int(schema["row_count"]):
        raise ValueError("D5 row count changed before D6")
    if table["meta_collision_index"].isna().any():
        raise ValueError("Missing collision identifiers")
    if table["meta_collision_index"].duplicated().any():
        raise ValueError("Duplicate collision identifiers")

    observed_years = set(table["meta_collision_year"].astype(int).unique())
    if observed_years != set(ALL_YEARS):
        raise ValueError(f"Unexpected years: {sorted(observed_years)}")
    observed_targets = set(table["target_severity"].astype(int).unique())
    if observed_targets != set(TARGET_LABELS):
        raise ValueError(f"Unexpected target codes: {sorted(observed_targets)}")
    return table, schema, rules


def role_summary(
    table: pd.DataFrame,
    role_column: str,
    roles: tuple[str, ...],
    *,
    seed: int | str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for role in roles:
        subset = table.loc[table[role_column].eq(role)]
        counts = subset["target_severity"].value_counts().to_dict()
        total = len(subset)
        years = sorted(subset["meta_collision_year"].astype(int).unique())
        row: dict[str, object] = {
            "seed": seed,
            "split": role,
            "years": ";".join(str(year) for year in years),
            "rows": total,
            "share_of_relevant_pool": "",
        }
        for target, label in TARGET_LABELS.items():
            count = int(counts.get(target, 0))
            row[f"{label.lower()}_n"] = count
            row[f"{label.lower()}_pct"] = count / total if total else np.nan
        rows.append(row)
    relevant_total = sum(int(row["rows"]) for row in rows)
    for row in rows:
        row["share_of_relevant_pool"] = int(row["rows"]) / relevant_total
    return rows


def make_annual_audit(table: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for year in ALL_YEARS:
        subset = table.loc[table["meta_collision_year"].eq(year)]
        counts = subset["target_severity"].value_counts().to_dict()
        for target, label in TARGET_LABELS.items():
            count = int(counts.get(target, 0))
            rows.append(
                {
                    "year": year,
                    "target_code": target,
                    "target_label": label,
                    "count": count,
                    "pct": count / len(subset),
                    "year_total": len(subset),
                }
            )
    return rows


def assign_splits(table: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    assignments = table.copy()
    year = assignments["meta_collision_year"].astype(int)
    temporal_role = np.full(len(assignments), "", dtype=object)
    temporal_role[year.isin(TEMPORAL_TRAIN_YEARS).to_numpy()] = "train"
    temporal_role[year.eq(TEMPORAL_VALIDATION_YEAR).to_numpy()] = "validation"
    temporal_role[year.eq(TEMPORAL_TEST_YEAR).to_numpy()] = "test"
    if np.any(temporal_role == ""):
        raise AssertionError("A row was not assigned to the temporal protocol")
    assignments["temporal_role"] = temporal_role

    temporal_rows = role_summary(
        assignments,
        "temporal_role",
        ("train", "validation", "test"),
        seed="year_based",
    )

    development_mask = year.isin(DEVELOPMENT_YEARS).to_numpy()
    development_positions = np.flatnonzero(development_mask)
    development_targets = assignments.loc[
        development_mask, "target_severity"
    ].astype(int).to_numpy()
    local_positions = np.arange(len(development_positions))
    random_rows: list[dict[str, object]] = []

    for seed in RANDOM_SEEDS:
        local_train, local_remainder = train_test_split(
            local_positions,
            train_size=RANDOM_TRAIN_RATIO,
            random_state=seed,
            stratify=development_targets,
        )
        local_validation, local_test = train_test_split(
            local_remainder,
            test_size=0.5,
            random_state=seed + RANDOM_SECOND_STAGE_OFFSET,
            stratify=development_targets[local_remainder],
        )

        role = np.full(len(assignments), "locked_temporal_test", dtype=object)
        role[development_positions[local_train]] = "train"
        role[development_positions[local_validation]] = "validation"
        role[development_positions[local_test]] = "test"
        column = f"random_role_seed_{seed}"
        assignments[column] = role

        dev_role = role[development_mask]
        if set(dev_role) != {"train", "validation", "test"}:
            raise AssertionError(f"Incomplete random assignment for seed {seed}")
        if np.any(role[~development_mask] != "locked_temporal_test"):
            raise AssertionError(f"2024 was not locked for seed {seed}")

        summary_rows = role_summary(
            assignments.loc[development_mask],
            column,
            ("train", "validation", "test"),
            seed=seed,
        )
        random_rows.extend(summary_rows)

    return assignments, temporal_rows, random_rows


def save_assignments(assignments: pd.DataFrame) -> str:
    ASSIGNMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(
        ASSIGNMENTS_FILE,
        index=False,
        lineterminator="\n",
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    return hash_file(ASSIGNMENTS_FILE)


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 7.7,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.0,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#222222",
        transform=ax.transAxes,
        linespacing=1.25,
    )


def draw_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops={"arrowstyle": "-|>", "color": "#666666", "lw": 1.0},
    )


def make_figure(annual_rows: list[dict[str, object]], temporal_rows: list[dict[str, object]]) -> None:
    annual = pd.DataFrame(annual_rows)
    temporal = {str(row["split"]): int(row["rows"]) for row in temporal_rows}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.8,
        }
    )
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(7.25, 4.05),
        gridspec_kw={"width_ratios": [0.98, 1.02]},
    )

    years = np.array(ALL_YEARS)
    bottoms = np.zeros(len(years), dtype=float)
    for label in ("Slight", "Serious", "Fatal"):
        values = (
            annual.loc[annual["target_label"].eq(label)]
            .set_index("year")
            .loc[list(ALL_YEARS), "pct"]
            .to_numpy(dtype=float)
            * 100
        )
        ax_a.bar(
            years,
            values,
            bottom=bottoms,
            width=0.68,
            color=TARGET_COLORS[label],
            edgecolor="white",
            linewidth=0.55,
            label=label,
        )
        if label == "Serious":
            for x, value, bottom in zip(years, values, bottoms, strict=True):
                ax_a.text(
                    x,
                    bottom + value / 2,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=6.7,
                    color="#222222",
                )
        if label == "Fatal":
            for x, value in zip(years, values, strict=True):
                ax_a.text(
                    x,
                    101.0,
                    f"{value:.2f}%",
                    ha="center",
                    va="bottom",
                    fontsize=6.4,
                    color=TARGET_COLORS[label],
                )
        bottoms += values

    ax_a.set_title("(a) Annual collision-severity distribution", loc="left", pad=6)
    ax_a.set_ylabel("Share of collisions (%)")
    ax_a.set_ylim(0, 106)
    ax_a.set_xticks(years)
    ax_a.set_xticklabels([str(year) for year in years], rotation=0)
    ax_a.set_yticks(np.arange(0, 101, 20))
    ax_a.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    ax_a.set_axisbelow(True)
    ax_a.spines[["top", "right"]].set_visible(False)
    ax_a.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        ncol=3,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.2,
    )

    ax_b.set_title("(b) Frozen cohort and evaluation partitions", loc="left", pad=6)
    ax_b.set_axis_off()
    draw_box(
        ax_b,
        0.10,
        0.82,
        0.80,
        0.105,
        "Official STATS19 collisions, 2018-2024\nn=743,646",
        facecolor="#EDF3F8",
        edgecolor="#4C78A8",
    )
    draw_arrow(ax_b, (0.50, 0.82), (0.50, 0.745))
    draw_box(
        ax_b,
        0.08,
        0.605,
        0.84,
        0.14,
        "D5 frozen cohort after schema, leakage and QC audits\nn=743,646 retained (100.0%; 0 excluded)",
        facecolor="#F4F4F4",
        edgecolor="#777777",
        fontsize=7.2,
    )
    draw_arrow(ax_b, (0.50, 0.605), (0.50, 0.545))
    draw_box(
        ax_b,
        0.02,
        0.385,
        0.59,
        0.16,
        "Development pool\n2018-2023, n=642,719\nRandom IID: 70/15/15 (5 seeds)",
        facecolor="#EEF6EE",
        edgecolor="#5B8E5A",
        fontsize=6.9,
    )
    draw_box(
        ax_b,
        0.66,
        0.385,
        0.32,
        0.16,
        "Locked temporal\ntest: 2024\nn=100,927\nFatal, n=1,502",
        facecolor="#FBEDEE",
        edgecolor="#C44E52",
        fontsize=6.7,
    )
    draw_arrow(ax_b, (0.25, 0.385), (0.22, 0.315))
    draw_arrow(ax_b, (0.48, 0.385), (0.53, 0.315))
    draw_box(
        ax_b,
        0.02,
        0.14,
        0.39,
        0.175,
        f"Temporal training\n2018-2022\nn={temporal['train']:,}",
        facecolor="#F7F7F7",
        edgecolor="#777777",
        fontsize=7.1,
    )
    draw_box(
        ax_b,
        0.45,
        0.14,
        0.34,
        0.175,
        f"Temporal validation\n2023\nn={temporal['validation']:,}",
        facecolor="#FFF6E6",
        edgecolor="#E6A43A",
        fontsize=7.1,
    )
    ax_b.text(
        0.50,
        0.035,
        "2024 is excluded from preprocessing, tuning and model selection.",
        ha="center",
        va="center",
        fontsize=7.0,
        color="#555555",
        transform=ax_b.transAxes,
    )

    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.12, top=0.87, wspace=0.25)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_protocol(
    schema: dict[str, object],
    rules: dict[str, object],
    assignments_hash: str,
    temporal_rows: list[dict[str, object]],
    random_rows: list[dict[str, object]],
) -> dict[str, object]:
    temporal_counts = {str(row["split"]): int(row["rows"]) for row in temporal_rows}
    first_seed_counts = {
        str(row["split"]): int(row["rows"])
        for row in random_rows
        if int(row["seed"]) == RANDOM_SEEDS[0]
    }
    test_row = next(row for row in temporal_rows if row["split"] == "test")
    fatal_count = int(test_row["fatal_n"])
    return {
        "version": "D6_V2",
        "status": "FROZEN_BEFORE_MODEL_TRAINING",
        "freeze_date_local": "2026-08-28",
        "decision_basis": (
            "Protocol frozen before any model fitting or performance inspection. "
            "Only target counts were inspected for descriptive reporting and the "
            "pre-specified fatal-class feasibility check."
        ),
        "parent_artifacts": {
            "D5_dataset": DATA_FILE.relative_to(PROJECT_DIR).as_posix(),
            "D5_dataset_sha256": schema["sha256"],
            "D5_schema": D5_SCHEMA_FILE.relative_to(PROJECT_DIR).as_posix(),
            "D5_schema_sha256": hash_file(D5_SCHEMA_FILE),
            "D5_cleaning_rules": D5_RULES_FILE.relative_to(PROJECT_DIR).as_posix(),
            "D5_cleaning_rules_sha256": hash_file(D5_RULES_FILE),
        },
        "frozen_population": {
            "years": list(ALL_YEARS),
            "row_count": int(schema["row_count"]),
            "statistical_unit": "police-reported personal-injury collision",
            "prediction_scope": rules["prediction_scope"],
        },
        "frozen_schema": {
            "target": schema["target_column"],
            "target_order": ["Slight", "Serious", "Fatal"],
            "feature_count": len(schema["feature_columns"]),
            "feature_columns": schema["feature_columns"],
            "metadata_columns": schema["metadata_columns"],
            "feature_selection_rule": schema["feature_selection_policy"],
        },
        "frozen_cleaning": {
            "row_deletions_at_D5": schema["rows_removed_at_D5"],
            "speed_limit_missing": "retain NaN",
            "categorical_unknowns": "retain explicit official categories",
            "unseen_categories": "map to __UNSEEN__ using training vocabulary only",
            "rare_category_pooling": (
                "off by default; if used, thresholds are learned on training data only"
            ),
        },
        "temporal_protocol": {
            "purpose": "primary deployment-oriented evaluation",
            "train_years": list(TEMPORAL_TRAIN_YEARS),
            "validation_years": [TEMPORAL_VALIDATION_YEAR],
            "test_years": [TEMPORAL_TEST_YEAR],
            "row_counts": temporal_counts,
            "model_selection": "training and validation only",
            "final_test_rule": (
                "2024 remains untouched by preprocessing fit, hyperparameter tuning, "
                "threshold choice and model selection; evaluate once after freezing."
            ),
        },
        "random_reference_protocol": {
            "purpose": (
                "optimistic same-distribution reference, not a deployment estimate"
            ),
            "pool_years": list(DEVELOPMENT_YEARS),
            "pool_row_count": sum(first_seed_counts.values()),
            "ratios": {
                "train": RANDOM_TRAIN_RATIO,
                "validation": RANDOM_VALIDATION_RATIO,
                "test": RANDOM_TEST_RATIO,
            },
            "row_counts_per_seed": first_seed_counts,
            "stratify_by": "target_severity",
            "seeds": list(RANDOM_SEEDS),
            "second_stage_seed_rule": f"seed + {RANDOM_SECOND_STAGE_OFFSET}",
            "random_test_rule": (
                "Random test partitions are not used for preprocessing, tuning or "
                "model selection. Report mean and standard deviation across five seeds."
            ),
            "secondary_2024_diagnostic": (
                "Pre-registered optional diagnostic: apply each already-fitted random "
                "split model to 2024 without retuning; report separately from the "
                "strict temporal protocol."
            ),
        },
        "class_feasibility_preflight": {
            "year": TEMPORAL_TEST_YEAR,
            "fatal_count": fatal_count,
            "fatal_pct": float(test_row["fatal_pct"]),
            "screening_threshold": FATAL_PREFLIGHT_THRESHOLD,
            "decision": (
                "proceed" if fatal_count >= FATAL_PREFLIGHT_THRESHOLD else "proceed_with_sparse_class_warning"
            ),
            "interpretation": (
                "The threshold checks only that the class is not extremely sparse; it "
                "does not guarantee narrow confidence intervals. Fatal recall and all "
                "primary test metrics still require bootstrap confidence intervals."
            ),
        },
        "training_only_operations": [
            "category vocabulary fitting",
            "unseen and optional rare-category mapping",
            "numeric imputation for non-LightGBM models",
            "class-weight calculation",
            "optional resampling sensitivity analysis",
            "hyperparameter tuning",
        ],
        "frozen_metrics": {
            "model_selection": "Macro-F1 on validation data",
            "required_test_reporting": [
                "Macro-F1",
                "quadratic weighted kappa",
                "fatal recall",
                "serious-or-fatal recall",
                "pre-specified asymmetric error cost",
            ],
            "uncertainty": (
                "paired, class-stratified nonparametric bootstrap over collision records "
                f"within each test set; {BOOTSTRAP_ITERATIONS:,} iterations with "
                f"random seed {BOOTSTRAP_SEED}"
            ),
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "split_assignment_artifact": {
            "file": ASSIGNMENTS_FILE.relative_to(PROJECT_DIR).as_posix(),
            "sha256": assignments_hash,
            "role_columns": [
                "temporal_role",
                *[f"random_role_seed_{seed}" for seed in RANDOM_SEEDS],
            ],
        },
    }


def write_checkpoint(
    protocol: dict[str, object],
    temporal_rows: list[dict[str, object]],
    random_rows: list[dict[str, object]],
) -> None:
    temporal = {str(row["split"]): row for row in temporal_rows}
    first_seed = {
        str(row["split"]): row
        for row in random_rows
        if int(row["seed"]) == RANDOM_SEEDS[0]
    }
    preflight = protocol["class_feasibility_preflight"]
    text = f"""# D6 protocol freeze and checkpoint 1

## Status

**PASS - protocol frozen before model training.**

## Locked temporal evaluation

- Training: 2018-2022, **{int(temporal['train']['rows']):,}** collisions.
- Validation: 2023, **{int(temporal['validation']['rows']):,}** collisions.
- Final temporal test: 2024, **{int(temporal['test']['rows']):,}** collisions.
- 2024 target counts: Slight **{int(temporal['test']['slight_n']):,}**, Serious **{int(temporal['test']['serious_n']):,}**, Fatal **{int(temporal['test']['fatal_n']):,}**.
- Fatal prevalence in 2024: **{float(temporal['test']['fatal_pct']):.3%}**.

The 2024 Fatal count ({int(preflight['fatal_count']):,}) exceeds the pre-specified screening threshold ({int(preflight['screening_threshold'])}). This supports retaining fatal recall as a reported metric, but it does not guarantee a narrow confidence interval. Bootstrap uncertainty remains mandatory.

## Random same-distribution reference

- Pool: 2018-2023, **{sum(int(row['rows']) for row in first_seed.values()):,}** collisions.
- Per seed: train **{int(first_seed['train']['rows']):,}**, validation **{int(first_seed['validation']['rows']):,}**, test **{int(first_seed['test']['rows']):,}**.
- Five frozen seeds: {', '.join(str(seed) for seed in RANDOM_SEEDS)}.
- Splits are stratified by the three-class target. The actual collision-level assignments are saved, not merely regenerated from seeds.
- The random test is an optimistic internal reference. It is not interpreted as future-year performance.

## Leakage controls frozen at D6

- The exact 17-feature allowlist remains unchanged.
- Metadata and target columns cannot enter the feature matrix.
- All preprocessing, class weighting and optional resampling are fitted on training data only.
- Random and temporal test partitions cannot be used for tuning or model selection.
- The optional random-model-on-2024 diagnostic is pre-registered and may only reuse an already-fitted model without retuning.

## Figure 2

- Panel (a): annual Slight, Serious and Fatal distributions for 2018-2024.
- Panel (b): complete sample flow and the frozen temporal/random evaluation structure.

## Checkpoint decision

Proceed to modelling. Bootstrap is fixed at 2,000 iterations with random seed 20260828. Any change to years, features, cleaning rules, split ratios, seeds, bootstrap settings or the 2024 access rule requires a new protocol version and an explicit reason recorded before inspecting model performance.
"""
    CHECKPOINT_FILE.write_text(text, encoding="utf-8")


def run_assertions(
    assignments: pd.DataFrame,
    protocol: dict[str, object],
    temporal_rows: list[dict[str, object]],
    random_rows: list[dict[str, object]],
) -> None:
    if sum(int(row["rows"]) for row in temporal_rows) != len(assignments):
        raise AssertionError("Temporal split does not cover the full cohort")
    if int(next(row for row in temporal_rows if row["split"] == "test")["fatal_n"]) != 1502:
        raise AssertionError("Unexpected 2024 Fatal count")
    for seed in RANDOM_SEEDS:
        rows = [row for row in random_rows if int(row["seed"]) == seed]
        if sum(int(row["rows"]) for row in rows) != 642_719:
            raise AssertionError(f"Random pool row mismatch for seed {seed}")
        if {str(row["split"]) for row in rows} != {"train", "validation", "test"}:
            raise AssertionError(f"Random roles incomplete for seed {seed}")
        if any(int(row["fatal_n"]) <= FATAL_PREFLIGHT_THRESHOLD for row in rows):
            raise AssertionError(f"Fatal class unexpectedly sparse for seed {seed}")
    if protocol["status"] != "FROZEN_BEFORE_MODEL_TRAINING":
        raise AssertionError("Protocol was not frozen")
    if not FIGURE_PNG.exists() or FIGURE_PNG.stat().st_size == 0:
        raise AssertionError("Figure 2 PNG was not created")
    if not FIGURE_PDF.exists() or FIGURE_PDF.stat().st_size == 0:
        raise AssertionError("Figure 2 PDF was not created")


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    table, schema, rules = load_and_validate_d5()
    annual_rows = make_annual_audit(table)
    assignments, temporal_rows, random_rows = assign_splits(table)

    write_csv(ANNUAL_AUDIT_FILE, annual_rows)
    write_csv(TEMPORAL_AUDIT_FILE, temporal_rows)
    write_csv(RANDOM_AUDIT_FILE, random_rows)
    assignments_hash = save_assignments(assignments)
    make_figure(annual_rows, temporal_rows)

    protocol = make_protocol(
        schema,
        rules,
        assignments_hash,
        temporal_rows,
        random_rows,
    )
    write_json(PROTOCOL_FILE, protocol)
    write_checkpoint(protocol, temporal_rows, random_rows)
    run_assertions(assignments, protocol, temporal_rows, random_rows)

    print("D6 protocol status:", protocol["status"])
    print("2024 rows:", next(row["rows"] for row in temporal_rows if row["split"] == "test"))
    print("2024 Fatal collisions:", protocol["class_feasibility_preflight"]["fatal_count"])
    print("Random seeds:", list(RANDOM_SEEDS))
    print("Assignments SHA-256:", assignments_hash)
    print("FINAL_D6_ASSERTIONS=PASS")


if __name__ == "__main__":
    main()

