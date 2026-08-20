"""`HC-M4-06` — materialize per-applicant features for every known applicant.

Runs `build_historical_features()` (`aggregations.py`, unchanged from
`HC-M3-06`) exactly once over the full applicant spine -- every
`SK_ID_CURR` in *both* `application_train` and `application_test`, not
just the development pool -- and merges the result with each applicant's
raw application-time fields (income, requested credit, demographics,
etc.). That combined table is this project's offline feature store;
`LocalFeatureStore` (`adapters/local_store.py`) loads it for fast online
lookups.

**Extended in `HC-M4-08` beyond Chunk 2's original historical-only
scope**, once building the scoring use case made the gap concrete: a
model input row needs the applicant's raw application-time fields too,
not just their historical aggregates -- `HC-M3-05`'s ratio/age features
(cheap, pure, computed from those raw fields) are still deliberately
*not* materialized here, since they were never the bottleneck `HC-M4-04`
identified; the scoring use case computes them on the fly via
`add_basic_features()`.

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

    application_fields = con.sql(
        """
        SELECT * FROM application_train
        UNION ALL BY NAME
        SELECT * FROM application_test
        """
    ).df()
    application_fields = application_fields.drop(columns=["TARGET"])
    spine_ids = application_fields["SK_ID_CURR"]

    start = time.time()
    historical_features = build_historical_features(con, spine_ids)
    con.close()
    elapsed = time.time() - start

    features = application_fields.merge(
        historical_features, on="SK_ID_CURR", how="left"
    )

    assert len(features) == len(spine_ids), "Row count changed during materialization"
    assert features["SK_ID_CURR"].is_unique, "SK_ID_CURR must be unique per row"

    config.FEATURE_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(config.FEATURE_STORE_PATH, index=False)

    print(f"Materialized {len(features):,} applicants' features in {elapsed:.1f}s")
    print(f"Wrote {config.FEATURE_STORE_PATH}")


if __name__ == "__main__":
    main()
