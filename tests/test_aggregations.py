import duckdb
import pandas as pd
import pytest

from home_credit_default_risk.aggregations import (
    aggregate_bureau_features,
    aggregate_payment_behavior_features,
    aggregate_previous_application_features,
    build_historical_features,
)


@pytest.fixture
def con():
    """Small synthetic in-memory DuckDB connection — fast, deterministic,
    no dependency on the real ~3GB dataset. Values below are chosen so
    every aggregate can be hand-verified in the assertions."""
    connection = duckdb.connect(":memory:")

    connection.sql("""
        CREATE TABLE bureau (
            SK_ID_BUREAU BIGINT, SK_ID_CURR BIGINT, CREDIT_ACTIVE VARCHAR,
            AMT_CREDIT_SUM DOUBLE, AMT_CREDIT_SUM_DEBT DOUBLE,
            CREDIT_DAY_OVERDUE BIGINT, CNT_CREDIT_PROLONG BIGINT
        );
        INSERT INTO bureau VALUES
            (1, 100, 'Active', 10000.0, 5000.0, 0, 0),
            (2, 100, 'Closed', 20000.0, 0.0, 10, 1),
            (3, 200, 'Active', 5000.0, 5000.0, 0, 0);
    """)

    connection.sql("""
        CREATE TABLE bureau_balance (SK_ID_BUREAU BIGINT, STATUS VARCHAR);
        INSERT INTO bureau_balance VALUES
            (1, '0'), (1, '1'), (2, 'C'), (3, '2');
    """)

    connection.sql("""
        CREATE TABLE previous_application (
            SK_ID_PREV BIGINT, SK_ID_CURR BIGINT, NAME_CONTRACT_STATUS VARCHAR,
            AMT_APPLICATION DOUBLE, AMT_CREDIT DOUBLE
        );
        INSERT INTO previous_application VALUES
            (1, 100, 'Approved', 9000.0, 10000.0),
            (2, 100, 'Refused', 5000.0, 5000.0),
            (3, 300, 'Approved', 2000.0, 2000.0),
            (4, 400, 'Approved', 100.0, 0.0);
    """)

    connection.sql("""
        CREATE TABLE installments_payments (
            SK_ID_PREV BIGINT, SK_ID_CURR BIGINT,
            DAYS_INSTALMENT DOUBLE, DAYS_ENTRY_PAYMENT DOUBLE,
            AMT_INSTALMENT DOUBLE, AMT_PAYMENT DOUBLE
        );
        INSERT INTO installments_payments VALUES
            (1, 100, -100, -105, 1000.0, 1000.0),
            (1, 100, -50, -40, 1000.0, 800.0);
    """)

    connection.sql("""
        CREATE TABLE POS_CASH_balance (
            SK_ID_PREV BIGINT, SK_ID_CURR BIGINT, SK_DPD BIGINT
        );
        INSERT INTO POS_CASH_balance VALUES
            (1, 100, 0), (1, 100, 15);
    """)

    connection.sql("""
        CREATE TABLE credit_card_balance (
            SK_ID_PREV BIGINT, SK_ID_CURR BIGINT,
            AMT_BALANCE DOUBLE, AMT_CREDIT_LIMIT_ACTUAL DOUBLE, SK_DPD BIGINT
        );
        INSERT INTO credit_card_balance VALUES
            (1, 100, 500.0, 1000.0, 5);
    """)

    yield connection
    connection.close()


def test_bureau_aggregates(con):
    result = aggregate_bureau_features(con).set_index("SK_ID_CURR")

    row_100 = result.loc[100]
    assert row_100["bureau_credit_count"] == 2
    assert row_100["bureau_active_credit_count"] == 1
    assert row_100["bureau_total_credit_sum"] == 30000.0
    assert row_100["bureau_total_debt_sum"] == 5000.0
    assert row_100["bureau_debt_to_credit_ratio"] == pytest.approx(5000 / 30000)
    assert row_100["bureau_max_days_overdue"] == 10
    assert row_100["bureau_credit_prolong_count"] == 1
    assert (
        row_100["bureau_balance_bad_month_count"] == 1
    )  # status '1' on SK_ID_BUREAU=1

    row_200 = result.loc[200]
    assert row_200["bureau_credit_count"] == 1
    assert row_200["bureau_debt_to_credit_ratio"] == pytest.approx(1.0)
    assert (
        row_200["bureau_balance_bad_month_count"] == 1
    )  # status '2' on SK_ID_BUREAU=3


def test_previous_application_aggregates(con):
    result = aggregate_previous_application_features(con).set_index("SK_ID_CURR")

    row_100 = result.loc[100]
    assert row_100["prev_application_count"] == 2
    assert row_100["prev_refused_count"] == 1
    assert row_100["prev_refused_rate"] == pytest.approx(0.5)
    assert row_100["prev_mean_credit"] == pytest.approx(7500.0)
    assert row_100["prev_max_credit"] == 10000.0
    assert row_100["prev_application_to_credit_ratio"] == pytest.approx(0.95)

    row_300 = result.loc[300]
    assert row_300["prev_refused_rate"] == 0.0


def test_previous_application_zero_credit_does_not_produce_inf(con):
    # SK_ID_CURR 400 has AMT_CREDIT == 0; NULLIF should skip it, not divide by zero
    result = aggregate_previous_application_features(con).set_index("SK_ID_CURR")
    row_400 = result.loc[400]
    assert pd.isna(row_400["prev_application_to_credit_ratio"])


def test_payment_behavior_aggregates(con):
    result = aggregate_payment_behavior_features(con).set_index("SK_ID_CURR")

    row_100 = result.loc[100]
    assert row_100["installment_count"] == 2
    assert row_100["late_payment_count"] == 1  # second installment: -40 > -50
    assert row_100["late_payment_rate"] == pytest.approx(0.5)
    assert row_100["mean_payment_ratio"] == pytest.approx((1.0 + 0.8) / 2)
    assert row_100["pos_max_dpd"] == 15
    assert row_100["cc_max_dpd"] == 5
    assert row_100["cc_mean_utilization"] == pytest.approx(0.5)


def test_build_historical_features_row_count_matches_spine(con):
    spine_ids = pd.Series([100, 200, 300, 400, 999])  # 999 has no history anywhere
    result = build_historical_features(con, spine_ids)
    assert len(result) == len(spine_ids)
    assert set(result["SK_ID_CURR"]) == set(spine_ids)


def test_build_historical_features_fills_counts_not_ratios_for_missing_history(con):
    spine_ids = pd.Series([100, 999])
    result = build_historical_features(con, spine_ids).set_index("SK_ID_CURR")

    no_history = result.loc[999]
    assert no_history["bureau_credit_count"] == 0
    assert no_history["prev_application_count"] == 0
    assert pd.isna(no_history["bureau_debt_to_credit_ratio"])
    assert pd.isna(no_history["mean_payment_ratio"])


def test_build_historical_features_no_fanout_when_source_has_multiple_rows(con):
    # SK_ID_CURR 100 has 2 bureau rows, 2 previous_application rows, 2
    # installment rows -- the join must not multiply these together.
    spine_ids = pd.Series([100])
    result = build_historical_features(con, spine_ids)
    assert len(result) == 1
