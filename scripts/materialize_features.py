"""`HC-M4-06` — materialize historical features for every known applicant.

Runs `build_historical_features()` (`aggregations.py`, unchanged from
`HC-M3-06`) exactly once over the full applicant spine -- every
`SK_ID_CURR` in *both* `application_train` and `application_test`, not
just the development pool -- and writes the result to a Parquet file.
That file is this project's offline feature store; `LocalFeatureStore`
(`adapters/local_store.py`) loads it for fast online lookups.

Re-run this whenever the raw tables or `aggregations.py` change. There is
no incremental/streaming update here -- a deliberate scope limit for a
local, single-node demo, not an oversight (see `HC-M4-07`'s benchmark
write-up for what this buys over the alternative).
"""

import time

import duckdb

from home_credit_default_risk import config
from home_credit_default_risk.aggregations import build_historical_features


def main() -> None:
    con = duckdb.connect(str(config.CACHE_DB), read_only=True)
    spine_ids = con.sql(
        """
        SELECT SK_ID_CURR FROM application_train
        UNION
        SELECT SK_ID_CURR FROM application_test
        """
    ).df()["SK_ID_CURR"]

    start = time.time()
    features = build_historical_features(con, spine_ids)
    con.close()
    elapsed = time.time() - start

    assert len(features) == len(spine_ids), "Row count changed during materialization"
    assert features["SK_ID_CURR"].is_unique, "SK_ID_CURR must be unique per row"

    config.FEATURE_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(config.FEATURE_STORE_PATH, index=False)

    print(f"Materialized {len(features):,} applicants' features in {elapsed:.1f}s")
    print(f"Wrote {config.FEATURE_STORE_PATH}")


if __name__ == "__main__":
    main()
