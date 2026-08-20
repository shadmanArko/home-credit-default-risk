"""`HC-M4-07` — benchmark: full DuckDB aggregation vs. `LocalFeatureStore`.

Answers, with real numbers, the exact problem `HC-M4-04`'s `FeatureStore`
port exists to solve: how much faster is retrieving one applicant's
historical features from the materialized store
(`scripts/materialize_features.py`) than recomputing them from the raw
tables the way `HC-M3-06`'s `build_historical_features()` always has?

Run `scripts/materialize_features.py` first if
`data/processed/historical_features.parquet` doesn't exist yet.
"""

import time

import duckdb
import pandas as pd

from home_credit_default_risk import config
from home_credit_default_risk.adapters.local_store import LocalFeatureStore
from home_credit_default_risk.aggregations import build_historical_features

SAMPLE_SK_ID_CURR = 100001


def main() -> None:
    con = duckdb.connect(str(config.CACHE_DB), read_only=True)
    start = time.perf_counter()
    build_historical_features(con, pd.Series([SAMPLE_SK_ID_CURR]))
    before_seconds = time.perf_counter() - start
    con.close()

    store = LocalFeatureStore(config.FEATURE_STORE_PATH)
    start = time.perf_counter()
    store.get_online_features(SAMPLE_SK_ID_CURR)
    after_seconds = time.perf_counter() - start

    before_ms = before_seconds * 1000
    after_ms = after_seconds * 1000
    print(f"Before (full DuckDB aggregation, one applicant): {before_ms:.1f}ms")
    print(f"After  (LocalFeatureStore lookup, one applicant): {after_ms:.3f}ms")
    print(f"Speedup: {before_seconds / after_seconds:,.0f}x")


if __name__ == "__main__":
    main()
