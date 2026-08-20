"""`HC-M4-14` — build a tiny, fully synthetic model + feature store for CI.

The real training/materialization scripts (`train_with_mlflow.py`,
`materialize_features.py`) need the ~2.9 GB real Kaggle dataset, which
is gitignored and unavailable to GitHub Actions without Kaggle
credentials most forks/PRs wouldn't have. This script reuses the exact
same synthetic-table pattern already used in `tests/test_pipeline.py`
(a handful of hand-built rows across the same six historical tables) to
produce a *real*, structurally valid model and feature store -- just a
tiny one -- so CI can genuinely start the serving container and hit
`/health`, `/score`, and `/apply` for real 200 responses, not skip the
check because production data isn't available.

Not meant to run anywhere except CI (or local debugging of the CI job)
-- `scripts/train_with_mlflow.py` and `scripts/materialize_features.py`
remain the real, production-scale versions of this same recipe.
"""

import duckdb
import pandas as pd

from home_credit_default_risk import config
from home_credit_default_risk.adapters.mlflow_registry import MlflowModelRegistry
from home_credit_default_risk.aggregations import build_historical_features
from home_credit_default_risk.pipeline import build_feature_matrix

SYNTHETIC_APPLICANT_IDS = list(range(100, 120))


def _build_synthetic_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.sql("""
        CREATE TABLE bureau (
            SK_ID_BUREAU BIGINT, SK_ID_CURR BIGINT, CREDIT_ACTIVE VARCHAR,
            AMT_CREDIT_SUM DOUBLE, AMT_CREDIT_SUM_DEBT DOUBLE,
            CREDIT_DAY_OVERDUE BIGINT, CNT_CREDIT_PROLONG BIGINT
        );
        INSERT INTO bureau VALUES (1, 100, 'Active', 10000.0, 5000.0, 0, 0);
    """)
    con.sql("""
        CREATE TABLE bureau_balance (SK_ID_BUREAU BIGINT, STATUS VARCHAR);
        INSERT INTO bureau_balance VALUES (1, '0');
    """)
    con.sql("""
        CREATE TABLE previous_application (
            SK_ID_PREV BIGINT, SK_ID_CURR BIGINT, NAME_CONTRACT_STATUS VARCHAR,
            AMT_APPLICATION DOUBLE, AMT_CREDIT DOUBLE
        );
        INSERT INTO previous_application VALUES (1, 100, 'Approved', 9000.0, 10000.0);
    """)
    con.sql("""
        CREATE TABLE installments_payments (
            SK_ID_PREV BIGINT, SK_ID_CURR BIGINT,
            DAYS_INSTALMENT DOUBLE, DAYS_ENTRY_PAYMENT DOUBLE,
            AMT_INSTALMENT DOUBLE, AMT_PAYMENT DOUBLE
        );
        INSERT INTO installments_payments VALUES (1, 100, -100, -105, 1000.0, 1000.0);
    """)
    con.sql("""
        CREATE TABLE POS_CASH_balance (
            SK_ID_PREV BIGINT, SK_ID_CURR BIGINT, SK_DPD BIGINT
        );
        INSERT INTO POS_CASH_balance VALUES (1, 100, 0);
    """)
    con.sql("""
        CREATE TABLE credit_card_balance (
            SK_ID_PREV BIGINT, SK_ID_CURR BIGINT,
            AMT_BALANCE DOUBLE, AMT_CREDIT_LIMIT_ACTUAL DOUBLE, SK_DPD BIGINT
        );
        INSERT INTO credit_card_balance VALUES (1, 100, 500.0, 1000.0, 0);
    """)
    return con


def _build_synthetic_application_df() -> pd.DataFrame:
    n = len(SYNTHETIC_APPLICANT_IDS)
    return pd.DataFrame(
        {
            "SK_ID_CURR": SYNTHETIC_APPLICANT_IDS,
            "TARGET": [i % 2 for i in range(n)],  # both classes present
            "AMT_CREDIT": [200_000.0 + 10_000 * i for i in range(n)],
            "AMT_INCOME_TOTAL": [50_000.0 + 5_000 * i for i in range(n)],
            "AMT_ANNUITY": [15_000.0 + 500 * i for i in range(n)],
            "AMT_GOODS_PRICE": [180_000.0 + 9_000 * i for i in range(n)],
            "DAYS_BIRTH": [-12_000 - 100 * i for i in range(n)],
            "DAYS_EMPLOYED": [-2_000 - 50 * i for i in range(n)],
            "CODE_GENDER": ["F" if i % 2 == 0 else "M" for i in range(n)],
        }
    )


def main() -> None:
    from train_final_model import build_final_pipeline

    con = _build_synthetic_connection()
    application_df = _build_synthetic_application_df()

    features = build_feature_matrix(con, application_df.drop(columns=["TARGET"]))
    X = features.drop(columns=["SK_ID_CURR"])
    y = application_df["TARGET"]

    pipeline = build_final_pipeline(X)
    pipeline.fit(X, y)

    registry = MlflowModelRegistry(
        tracking_uri=config.MLFLOW_TRACKING_URI,
        model_name=config.MLFLOW_MODEL_NAME,
        alias=config.MLFLOW_PRODUCTION_ALIAS,
    )
    registry.register_and_promote(
        pipeline,
        metrics={"note": 1.0},
        params={"fixture": "ci_smoke_test", "n_rows": len(X)},
    )

    historical = build_historical_features(
        con, application_df["SK_ID_CURR"]
    )
    con.close()
    feature_store = application_df.drop(columns=["TARGET"]).merge(
        historical, on="SK_ID_CURR", how="left"
    )
    config.FEATURE_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_store.to_parquet(config.FEATURE_STORE_PATH, index=False)

    print(f"Synthetic model trained + registered on {len(X)} rows")
    print(f"Wrote synthetic feature store to {config.FEATURE_STORE_PATH}")
    print(f"Sample SK_ID_CURR for smoke testing: {SYNTHETIC_APPLICANT_IDS[0]}")


if __name__ == "__main__":
    main()
