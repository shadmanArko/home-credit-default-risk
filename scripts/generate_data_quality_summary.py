"""Consolidate every table's data-quality checks into one scorecard.

Reads reports/data_profile/*.json (no new DuckDB queries — everything
needed was already computed by scripts/profile_data.py) and writes
reports/data_quality_summary.md: one row per table covering full-row
duplicates, primary-key duplicates, every foreign-key orphan check, the
worst missing-value column, and the constant-column count.

This is the direct answer to "how much duplicate/bad data is in all my
tables" — one file, no digging through notebook cells or diffing JSON.
Run after scripts/profile_data.py.
"""

import json
from pathlib import Path

PROFILE_DIR = Path("reports/data_profile")
OUTPUT_PATH = Path("reports/data_quality_summary.md")


def worst_missing(profile: dict) -> tuple[str | None, float]:
    worst_col, worst_pct = None, 0.0
    for col, entry in profile["columns"].items():
        pct = entry["null_pct"] or 0.0
        if pct > worst_pct:
            worst_col, worst_pct = col, pct
    return worst_col, worst_pct


def constant_column_count(profile: dict) -> int:
    return sum(1 for e in profile["columns"].values() if e["distinct_count"] <= 1)


def render_row(profile: dict) -> list[str]:
    table = profile["table"]
    row_count = profile["row_count"]
    dup_rows = profile["duplicate_row_check"]["duplicate_row_count"]

    pk = profile.get("primary_key_check")
    pk_str = f"{pk['column']}: {pk['duplicate_key_count']} dup keys" if pk else "—"

    fk_checks = profile.get("foreign_key_checks") or []
    if fk_checks:
        fk_str = "; ".join(
            f"{fk['column']}→{fk['references']}: {fk['orphan_key_count']} orphans"
            for fk in fk_checks
        )
    else:
        fk_str = "—"

    worst_col, worst_pct = worst_missing(profile)
    missing_str = f"{worst_col} ({worst_pct}%)" if worst_col else "—"
    const_count = constant_column_count(profile)

    return [
        f"`{table}`",
        f"{row_count:,}",
        str(dup_rows),
        pk_str,
        fk_str,
        missing_str,
        str(const_count),
    ]


def main() -> None:
    profile_paths = sorted(PROFILE_DIR.glob("*.json"))
    if not profile_paths:
        raise FileNotFoundError(
            f"No profile JSON files found in {PROFILE_DIR}. "
            "Run scripts/profile_data.py first."
        )

    header = [
        "Table",
        "Rows",
        "Duplicate rows",
        "PK check",
        "FK orphan checks",
        "Worst missing column",
        "Constant columns",
    ]
    rows = [render_row(json.loads(p.read_text())) for p in profile_paths]

    lines = [
        "# Data Quality Scorecard — Home Credit Default Risk",
        "",
        "Generated from `reports/data_profile/*.json` by "
        "`scripts/generate_data_quality_summary.py`. Regenerate after "
        "re-running `scripts/profile_data.py`. One row per table, every "
        "number a full-table DuckDB scan — not a sample.",
        "",
        f"| {' | '.join(header)} |",
        f"| {' | '.join(['---'] * len(header))} |",
    ]
    for row in rows:
        lines.append(f"| {' | '.join(row)} |")

    OUTPUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
