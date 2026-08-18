"""Profile every raw CSV with DuckDB and write one JSON summary per table.

Why DuckDB instead of pandas: DuckDB executes SQL against the data with an
out-of-core, vectorized engine, so it never materializes a full file in
Python memory. That property scales unchanged to files far larger than
RAM (and to Parquet with a one-line change).

Landing before profiling: reading directly off `read_csv_auto` for every
query (schema probe, aggregates, then one top-values query per categorical
column) re-parses the raw CSV text from scratch each time, which is the
main cost for a text format. Instead we land each CSV once into a DuckDB
native columnar table on disk (a throwaway "bronze -> query-ready" step,
the same pattern real pipelines use before repeated analytical queries),
then run every profiling query against that table. One parse per file,
many cheap columnar scans after.

Output: one JSON file per table in reports/data_profile/, containing row
count, per-column null rate / cardinality / numeric stats, a primary-key
uniqueness check, and (for child tables) a foreign-key orphan check against
the SK_ID_CURR spine. This JSON is the artifact downstream notebooks and
future sessions should read instead of re-scanning the raw CSVs.
"""

import json
from pathlib import Path

import duckdb

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("reports/data_profile")
CACHE_DB = OUTPUT_DIR / ".landing_cache.duckdb"
DESCRIPTION_FILE = "HomeCredit_columns_description.csv"

# Candidate primary key per table, and the FK column (if any) that should
# resolve into the SK_ID_CURR spine (application_train + application_test).
TABLE_KEYS = {
    "application_train.csv": {"pk": "SK_ID_CURR", "fk": None},
    "application_test.csv": {"pk": "SK_ID_CURR", "fk": None},
    "bureau.csv": {"pk": "SK_ID_BUREAU", "fk": "SK_ID_CURR"},
    "bureau_balance.csv": {"pk": None, "fk": None},
    "previous_application.csv": {"pk": "SK_ID_PREV", "fk": "SK_ID_CURR"},
    "POS_CASH_balance.csv": {"pk": None, "fk": "SK_ID_CURR"},
    "credit_card_balance.csv": {"pk": None, "fk": "SK_ID_CURR"},
    "installments_payments.csv": {"pk": None, "fk": "SK_ID_CURR"},
}

NUMERIC_TYPES = {
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "FLOAT",
    "DOUBLE",
    "DECIMAL",
}

# Categorical columns above this cardinality are not worth a top-N query
# (either a near-unique key or free text) — skip them, don't blow up the
# GROUP BY.
MAX_CATEGORICAL_CARDINALITY = 100
TOP_N_VALUES = 10

# DuckDB's read_csv_auto type-sniffing misdetects this column as BOOLEAN
# because its values are the literal strings "Yes"/"No", which match its
# boolean-literal heuristic. It's actually a categorical flag like every
# other *_MODE column. Left unfixed, this doesn't just mislabel the profile
# (top_values would show true/false instead of Yes/No) — a nullable
# BOOLEAN column mixed with nullable string columns in one pandas
# DataFrame trips a real pandas/scikit-learn interop bug downstream
# (`TypeError: boolean value of NA is ambiguous` inside SimpleImputer),
# discovered while building the HC-M1-07 modeling pipeline.
COLUMN_TYPE_OVERRIDES = {
    "application_train.csv": {"EMERGENCYSTATE_MODE": "VARCHAR"},
    "application_test.csv": {"EMERGENCYSTATE_MODE": "VARCHAR"},
}


def quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def column_schema(con: duckdb.DuckDBPyConnection, table: str) -> list[tuple[str, str]]:
    rows = con.sql(f"DESCRIBE {table}").fetchall()
    return [(row[0], row[1]) for row in rows]


def profile_table(
    con: duckdb.DuckDBPyConnection,
    csv_name: str,
    table: str,
    spine_relation: str | None,
) -> dict:
    schema = column_schema(con, table)
    source = table

    exprs = ["COUNT(*) AS row_count"]
    for col, dtype in schema:
        q = quote(col)
        exprs.append(
            f"SUM(CASE WHEN {q} IS NULL THEN 1 ELSE 0 END) AS {quote(col + '__nulls')}"
        )
        exprs.append(f"COUNT(DISTINCT {q}) AS {quote(col + '__distinct')}")
        if dtype.split("(")[0] in NUMERIC_TYPES:
            exprs.append(f"MIN({q}) AS {quote(col + '__min')}")
            exprs.append(f"MAX({q}) AS {quote(col + '__max')}")
            exprs.append(f"AVG({q}) AS {quote(col + '__avg')}")
            exprs.append(f"median({q}) AS {quote(col + '__median')}")

    agg_sql = f"SELECT {', '.join(exprs)} FROM {source}"
    agg_row = con.sql(agg_sql).fetchone()
    agg_cols = [d[0] for d in con.sql(agg_sql).description]
    agg = dict(zip(agg_cols, agg_row, strict=True))

    row_count = agg["row_count"]
    columns = {}
    for col, dtype in schema:
        null_count = agg[col + "__nulls"]
        distinct_count = agg[col + "__distinct"]
        entry = {
            "dtype": dtype,
            "null_count": null_count,
            "null_pct": round(100 * null_count / row_count, 4) if row_count else None,
            "distinct_count": distinct_count,
        }
        if dtype.split("(")[0] in NUMERIC_TYPES:
            entry["min"] = agg[col + "__min"]
            entry["max"] = agg[col + "__max"]
            entry["avg"] = agg[col + "__avg"]
            entry["median"] = agg[col + "__median"]
        elif 0 < distinct_count <= MAX_CATEGORICAL_CARDINALITY:
            q = quote(col)
            top = con.sql(
                f"SELECT {q} AS value, COUNT(*) AS n FROM {source} "
                f"GROUP BY {q} ORDER BY n DESC LIMIT {TOP_N_VALUES}"
            ).fetchall()
            entry["top_values"] = [{"value": v, "count": n} for v, n in top]
        columns[col] = entry

    profile = {
        "table": csv_name,
        "row_count": row_count,
        "column_count": len(schema),
        "columns": columns,
    }

    keys = TABLE_KEYS.get(csv_name, {"pk": None, "fk": None})
    pk = keys["pk"]
    if pk:
        dup = con.sql(
            f"SELECT COUNT(*) FROM (SELECT {quote(pk)} FROM {source} "
            f"GROUP BY {quote(pk)} HAVING COUNT(*) > 1)"
        ).fetchone()[0]
        profile["primary_key_check"] = {
            "column": pk,
            "duplicate_key_count": dup,
            "is_unique": dup == 0,
        }

    fk = keys["fk"]
    if fk and spine_relation:
        orphan = con.sql(
            f"SELECT COUNT(*) FROM (SELECT DISTINCT {quote(fk)} FROM {source}) t "
            f"LEFT JOIN {spine_relation} s ON t.{quote(fk)} = s.SK_ID_CURR "
            f"WHERE s.SK_ID_CURR IS NULL"
        ).fetchone()[0]
        profile["foreign_key_check"] = {
            "column": fk,
            "references": "SK_ID_CURR spine (application_train + application_test)",
            "orphan_key_count": orphan,
        }

    return profile


def land_table(con: duckdb.DuckDBPyConnection, table: str, csv_path: Path) -> None:
    exists = con.sql(
        "SELECT 1 FROM information_schema.tables WHERE table_name = ?", params=[table]
    ).fetchone()
    if exists:
        return
    overrides = COLUMN_TYPE_OVERRIDES.get(csv_path.name)
    types_arg = f", types={overrides!r}" if overrides else ""
    con.sql(
        f"CREATE TABLE {table} AS "
        f"SELECT * FROM read_csv_auto({str(csv_path)!r}, sample_size=-1{types_arg})"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Persistent, disk-backed connection: landing a table parses its CSV
    # once; re-running the script reuses already-landed tables instead of
    # re-parsing, which matters once files are large.
    con = duckdb.connect(str(CACHE_DB))

    train_path = DATA_DIR / "application_train.csv"
    test_path = DATA_DIR / "application_test.csv"
    land_table(con, "application_train", train_path)
    land_table(con, "application_test", test_path)
    con.sql(
        "CREATE OR REPLACE VIEW spine AS "
        "SELECT SK_ID_CURR FROM application_train "
        "UNION SELECT SK_ID_CURR FROM application_test"
    )

    csv_files = sorted(p for p in DATA_DIR.glob("*.csv") if p.name != DESCRIPTION_FILE)
    for path in csv_files:
        table = path.stem
        print(f"Landing {path.name} -> table {table} ...")
        land_table(con, table, path)
        print(f"Profiling {table} ...")
        profile = profile_table(con, path.name, table, spine_relation="spine")
        out_path = OUTPUT_DIR / f"{path.stem}.json"
        out_path.write_text(json.dumps(profile, indent=2, default=str))
        print(f"  -> {out_path} (row_count={profile['row_count']})")

    con.close()


if __name__ == "__main__":
    main()
