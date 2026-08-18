"""Generate a Sweetviz train-vs-test comparison report for application_*.

Why this tool, and why scoped this way:

- `ydata-profiling` was evaluated first and rejected: its dependency pins
  (pandas <=1.1 / ==1.4.0) are incompatible with this project's pandas 3.x,
  and forcing an install around that is exactly the kind of thing that
  breaks silently later. Documented here instead of quietly worked around.
- Sweetviz's `compare()` mode is used specifically for its train-vs-test
  distribution comparison, which is structural/covariate information
  (does this feature's distribution differ between train and test?) —
  not a feature-vs-target relationship. Per the project's EDA discipline
  (see notebooks/01_data_understanding.ipynb), feature-vs-target analysis
  is deferred until after the stratified split and computed train-fold
  only, to avoid drawing conclusions from data that will become the
  validation fold. This report intentionally excludes TARGET and skips
  the expensive pairwise feature-association matrix (not needed for a
  train/test drift check, and another place "relational" analysis could
  creep in before it's appropriate).
- `application_train`/`application_test` are the only tables small enough
  (307k / 49k rows) to load into pandas directly; this is a deliberate
  exception to the "never load the full file into pandas" rule used
  elsewhere in this project; the child tables are two orders of magnitude
  larger and are not profiled this way.
- Reads from the DuckDB landing cache built by scripts/profile_data.py
  rather than re-parsing the raw CSVs, for consistency with the rest of
  the profiling pipeline.
"""

from pathlib import Path

import duckdb
import sweetviz as sv

CACHE_DB = Path("reports/data_profile/.landing_cache.duckdb")
OUTPUT_PATH = Path("reports/eda_train_vs_test_compare.html")


def main() -> None:
    if not CACHE_DB.exists():
        raise FileNotFoundError(
            f"{CACHE_DB} not found. Run scripts/profile_data.py first."
        )

    con = duckdb.connect(str(CACHE_DB), read_only=True)
    train_df = con.sql("SELECT * FROM application_train").df().drop(columns=["TARGET"])
    test_df = con.sql("SELECT * FROM application_test").df()
    con.close()

    report = sv.compare(
        [train_df, "Train"],
        [test_df, "Test"],
        pairwise_analysis="off",
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.show_html(str(OUTPUT_PATH), open_browser=False)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
