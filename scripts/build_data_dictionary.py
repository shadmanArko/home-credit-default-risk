"""Merge per-table JSON profiles + the official column descriptions into a
single, human- and LLM-readable reports/data_dictionary.md.

This is the artifact meant to be read instead of the raw CSVs: everything
needed to reason about the dataset's shape, quality, and semantics in one
place, cheap to diff in git, and small enough to re-read every session
instead of re-scanning gigabytes of source data.

Run after scripts/profile_data.py.
"""

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/raw")
PROFILE_DIR = Path("reports/data_profile")
OUTPUT_PATH = Path("reports/data_dictionary.md")
DESCRIPTION_FILE = DATA_DIR / "HomeCredit_columns_description.csv"

# Maps a profiled table name to the description file's "Table" label,
# since application_train/test share one row in the description file.
DESCRIPTION_TABLE_ALIASES = {
    "application_train": "application_{train|test}.csv",
    "application_test": "application_{train|test}.csv",
}


def load_descriptions() -> dict[str, dict[str, str]]:
    df = pd.read_csv(DESCRIPTION_FILE, encoding="latin-1", index_col=0)
    lookup: dict[str, dict[str, str]] = {}
    for table, group in df.groupby("Table"):
        lookup[table] = dict(zip(group["Row"], group["Description"], strict=True))
    return lookup


def describe_column(name: str, entry: dict, descriptions: dict[str, str]) -> str:
    desc = descriptions.get(name, "")
    parts = [
        f"`{entry['dtype']}`",
        f"null={entry['null_pct']}%",
        f"distinct={entry['distinct_count']}",
    ]
    if "min" in entry:
        parts.append(f"range=[{entry['min']}, {entry['max']}]")
        parts.append(
            f"mean={round(entry['avg'], 4) if entry['avg'] is not None else None}"
        )
        parts.append(f"median={entry['median']}")
    elif entry.get("top_values"):
        top = ", ".join(f"{v['value']}={v['count']}" for v in entry["top_values"][:5])
        parts.append(f"top={top}")
    stats = " | ".join(parts)
    return f"| `{name}` | {desc} | {stats} |"


def render_table(
    profile: dict, descriptions_by_table: dict[str, dict[str, str]]
) -> str:
    name = profile["table"].removesuffix(".csv")
    descriptions = descriptions_by_table.get(
        DESCRIPTION_TABLE_ALIASES.get(name, f"{name}.csv"), {}
    )

    lines = [f"## `{profile['table']}`", ""]
    lines.append(
        f"- Rows: **{profile['row_count']:,}**, Columns: **{profile['column_count']}**"
    )

    pk = profile.get("primary_key_check")
    if pk:
        status = (
            "unique"
            if pk["is_unique"]
            else f"**{pk['duplicate_key_count']} duplicate keys**"
        )
        lines.append(f"- Primary key `{pk['column']}`: {status}")

    fk = profile.get("foreign_key_check")
    if fk:
        rate = fk["orphan_key_count"]
        status = "no orphans" if rate == 0 else f"**{rate} orphan keys**"
        lines.append(f"- Foreign key `{fk['column']}` -> {fk['references']}: {status}")

    lines.append("")
    lines.append("| Column | Description | Stats |")
    lines.append("|---|---|---|")
    for col_name, entry in profile["columns"].items():
        lines.append(describe_column(col_name, entry, descriptions))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    descriptions_by_table = load_descriptions()
    profile_paths = sorted(PROFILE_DIR.glob("*.json"))
    if not profile_paths:
        raise FileNotFoundError(
            f"No profile JSON files found in {PROFILE_DIR}. "
            "Run scripts/profile_data.py first."
        )

    sections = [
        "# Data Dictionary — Home Credit Default Risk",
        "",
        "Generated from `reports/data_profile/*.json` "
        "(produced by `scripts/profile_data.py`) merged with "
        "`HomeCredit_columns_description.csv`. Regenerate with "
        "`uv run python scripts/build_data_dictionary.py` after re-running "
        "the profiler. This file is the intended reference for understanding "
        "the dataset — read this instead of the raw CSVs.",
        "",
    ]
    for path in profile_paths:
        profile = json.loads(path.read_text())
        sections.append(render_table(profile, descriptions_by_table))

    OUTPUT_PATH.write_text("\n".join(sections))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
